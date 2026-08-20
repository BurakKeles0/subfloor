"""M1 -- the two gates.

Spec v6 section 5.2, re-anchored to the E8P survivor band (plan section H2):
budgets 1.75 / 1.60 / 1.50, all of them below the PTQ floor.

    Gate A (feasibility)  does the best sparse config beat dense low-bit?
    Gate B (the thesis)   is the optimal T interior, or at an edge?

The two are independent on purpose.  Gate A can fail while Gate B holds, and
that outcome narrows the framing rather than stopping the project (Spec v6's
decision table, corrected in plan section B/11).

SCOPE.  This is a layer-level driver.  It measures ||X W^T - X W_hat^T||_F,
which is the objective every method here actually optimizes, and it is a proxy
for perplexity, not a substitute.  Model loading, sequential calibration and
perplexity evaluation are separate deliverables; `LayerProblem` is the seam they
plug into.  `--synthetic` runs the whole grid on generated data as a smoke test.

Gate B is deliberately NOT a bare argmin over T.  With a handful of calibration
draws, the argmin of a noisy curve lands in the interior by chance often enough
to manufacture a positive result (plan section B5).  It is reported as a paired
bootstrap on the differences instead, and stays "undetermined" unless the
interior really separates from both edges.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import accounting as A            # noqa: E402
import compact as C               # noqa: E402
import prune as P                 # noqa: E402
import quantize as Qz             # noqa: E402
import rotation as R              # noqa: E402
import tiling as Tl               # noqa: E402
from calibrate import LayerProblem, synthetic_problem   # noqa: E402,F401

E8P_BITS = Qz.E8P_BITS_PER_WEIGHT          # 2.0
DEFAULT_BUDGETS = (1.75, 1.60, 1.50)
DEFAULT_TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #

def tile_hessians(
    problem: LayerProblem, cw: C.CompactWeights, Q: Tensor | None = None
) -> Tensor:
    """Each tile's input sub-Hessian H[S_t, S_t], optionally rotated to match.

    If the block was rotated as `B Q^T`, the error rotates the same way, so the
    Hessian that keeps the objective invariant is `Q H Q^T`:

        tr((E Q^T)(Q H Q^T)(E Q^T)^T) = tr(E H E^T)
    """
    H = problem.H
    S = cw.idx_index                                     # [n_tiles, k]
    Ht = H[S.unsqueeze(-1), S.unsqueeze(1)]              # [n_tiles, k, k]
    return Ht if Q is None else Q @ Ht @ Q.transpose(-1, -2)


def run_config(
    problem: LayerProblem,
    *,
    budget_bits: float,
    tile_size: int | str,
    axis: str = "B",
    metric: str = "wanda",
    compensate: bool = True,
    rotate_axis: str | None = "index",
    quantize: bool = True,
    ldlq: bool = True,
    seed: int = 0,
    vq_bits: float = E8P_BITS,
) -> dict:
    """Prune -> compact -> rotate -> quantize, in that order, and measure.

    The order is the invariant (plan H1) and `prune` enforces it.

    `ldlq=True` rounds against each tile's sub-Hessian, rotated into the same
    basis as its block.  Without it the rotation costs inference time and buys
    nothing on the activation-weighted objective (plan section I3).  It needs
    the survivor count aligned to 8, so the mask is built with `align=8`.
    """
    if ldlq and quantize and axis != "B":
        raise NotImplementedError(
            "LDLQ is wired for Axis B, where the compacted block's index axis is "
            "input channels and the Hessian applies directly. Axis A needs the "
            "sweep along its tile's columns instead; pass ldlq=False for now."
        )
    scheme = {1: "unstructured", Tl.MAX_TILE: "structured"}.get(tile_size, "tile")
    requested = A.density_for_budget(
        scheme, budget_bits, None, problem.n_in if axis == "B" else problem.n_out,
        tile_size=tile_size, vq_bits=vq_bits,
    )
    if requested is None or not 0.0 < requested <= 1.0:
        return {"skipped": "budget unreachable at this tile size",
                "budget_bits": budget_bits, "tile_size": tile_size}

    pruned = P.prune(
        problem.W, axis=axis, tile_size=tile_size, density=requested,
        metric=metric, act_norm=problem.act_norm,
        H=problem.H if (compensate or metric == "obs_diag") else None,
        compensate=compensate,
        align=Qz.E8P_DIM if (quantize and ldlq) else 1,
    )

    W_hat = pruned.W
    if quantize:
        cw = C.compact(pruned.W, pruned.mask)
        rotated, Qm = (R.rotate(cw, axis=rotate_axis, seed=seed)
                       if rotate_axis else (cw, None))
        if ldlq:
            qb = Qz.ldlq_quantize_blocks(
                rotated.blocks,
                tile_hessians(problem, cw, Qm if rotate_axis == "index" else None),
            )
        else:
            qb = Qz.quantize_blocks(rotated.blocks)
        restored = rotated.with_blocks(qb.values)
        if rotate_axis:
            restored = R.unrotate(restored, Qm, axis=rotate_axis)
        W_hat = C.scatter(restored)

    # Realized density differs from the requested one by the per-tile rounding,
    # so the bits are recomputed from what actually happened -- never assumed.
    realized = pruned.mask.density()
    bits = A.bits_per_position(
        scheme, realized, None, pruned.mask.n_idx,
        tile_size=tile_size, vq_bits=vq_bits,
    )
    return {
        "budget_bits": budget_bits,
        "bits_realized": bits,
        "offset": bits - budget_bits,
        "offset_pct": (bits - budget_bits) / budget_bits,
        "flagged": abs(bits - budget_bits) / budget_bits > A.OFFSET_FLAG_THRESHOLD,
        "scheme": scheme,
        "tile_size": tile_size,
        "axis": axis,
        "n_idx": pruned.mask.n_idx,
        "density_requested": requested,
        "density_realized": realized,
        "vq_bits": vq_bits,
        "q_over_scales_with_density": A.Q_OVERHEAD_SCALES_WITH_DENSITY[scheme],
        "metric": metric,
        "compensate": compensate,
        "rotate_axis": rotate_axis,
        "quantize": quantize,
        "ldlq": ldlq,
        "seed": seed,
        "rel_output_error": problem.output_error(W_hat),
        "snr_db": Qz.quantization_snr(problem.W, W_hat),
        "in_bitmap_regime": A.in_bitmap_regime(
            budget_bits, None, pruned.mask.n_idx,
            tile_size=tile_size if scheme == "tile" else 1, vq_bits=vq_bits,
        ),
    }


def dense_wall(problem: LayerProblem, seed: int = 0) -> dict:
    """The PTQ floor we claim to go under: dense E8P at its natural 2.0 bits.

    NOT budget-matched with the sparse configs, and that is the point -- the
    comparison is "less than 2 bits against exactly 2 bits".  Reported with the
    offset spelled out so no table can quietly imply otherwise.
    """
    blocks = problem.W.unsqueeze(0)
    qb = Qz.quantize_blocks(blocks)
    W_hat = qb.values[0]
    return {
        "label": "dense E8P (PTQ floor reference)",
        "budget_bits": None,
        "bits_realized": E8P_BITS,
        "density_realized": 1.0,
        "tile_size": None,
        "seed": seed,
        "rel_output_error": problem.output_error(W_hat),
        "snr_db": Qz.quantization_snr(problem.W, W_hat),
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the mean."""
    if not values:
        raise ValueError("no values to bootstrap")
    v = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(v), (n_boot, len(v)), generator=g)
    means = v[idx].mean(dim=1)
    lo = torch.quantile(means, alpha / 2)
    hi = torch.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def paired_bootstrap_ci(
    a: list[float], b: list[float], n_boot: int = 10000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for mean(a - b), resampling the PAIRS.

    Pairing matters: the same calibration draw feeds every tile size, so the
    draw-to-draw noise is shared and largely cancels in the difference.  This is
    the same reason the pre-registration requires tau to be a paired difference
    (plan section B4).
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)}, {len(b)}")
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(d), (n_boot, len(d)), generator=g)
    means = d[idx].mean(dim=1)
    return float(torch.quantile(means, alpha / 2)), float(
        torch.quantile(means, 1 - alpha / 2)
    )


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def gate_a(records: list[dict], wall: dict, alpha: float = 0.05) -> dict:
    """Does the best sparse config beat the dense low-bit reference?

    A fortiori applies here (Spec v6 section 5.2): winning with the weaker
    saliency settles it; losing only means the stronger one should be tried.
    """
    by_tile: dict[object, list[float]] = {}
    for r in records:
        if "skipped" not in r:
            by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])
    if not by_tile:
        return {"verdict": "no runs"}

    best_tile = min(by_tile, key=lambda t: sum(by_tile[t]) / len(by_tile[t]))
    best = by_tile[best_tile]
    lo, hi = bootstrap_ci(best, alpha=alpha)
    mean = sum(best) / len(best)
    passes = hi < wall["rel_output_error"]
    return {
        "verdict": "pass" if passes else "fail",
        "best_tile": best_tile,
        "best_mean_error": mean,
        "best_ci": (lo, hi),
        "wall_error": wall["rel_output_error"],
        "wall_bits": wall["bits_realized"],
        "note": (
            "sparse budget is below the wall's 2.0 bits; this is a "
            "cheaper-and-better claim, not a budget-matched one"
        ),
    }


def gate_b(records: list[dict], alpha: float = 0.05, min_seeds: int = 5) -> dict:
    """Is the optimum T interior, or at an edge?

    Two corrections stand between this and a false positive, and BOTH are load
    bearing -- without them the gate reports 'interior' on data with no effect
    in it at all (see tests/test_m1_gates.py):

    1. Selection.  T* is chosen as the argmin over the interior tile sizes and
       then tested on the same draws, so the test is Bonferroni-corrected by the
       number of candidates.  Skipping this is double dipping.

    2. Too few draws.  A percentile bootstrap over three calibration draws
       resamples three numbers; its 95% interval does not have 95% coverage.
       Below `min_seeds` the honest verdict is 'undetermined', not a p-value.

    Note this puts Gate B in tension with Spec v6 section 6, which asks only for
    seeds >= 3.  Three is enough to report a mean; it is not enough to decide
    this gate.
    """
    by_tile: dict[object, list[float]] = {}
    for r in records:
        if "skipped" not in r:
            by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])

    tiles = [t for t in by_tile if t not in (1, Tl.MAX_TILE)]
    if not tiles or 1 not in by_tile or Tl.MAX_TILE not in by_tile:
        return {"verdict": "undetermined", "reason": "both edges must be present"}

    means = {t: sum(v) / len(v) for t, v in by_tile.items()}
    t_star = min(tiles, key=lambda t: means[t])

    n_draws = min(len(v) for v in by_tile.values())
    if n_draws < min_seeds:
        return {
            "verdict": "undetermined",
            "reason": (
                f"{n_draws} calibration draws is too few for a bootstrap CI; "
                f"need >= {min_seeds}"
            ),
            "t_star": t_star,
            "means": {str(k): v for k, v in means.items()},
        }

    # Bonferroni over the interior candidates we selected T* from.
    alpha_eff = alpha / max(1, len(tiles))
    lo_f, hi_f = paired_bootstrap_ci(by_tile[t_star], by_tile[1], alpha=alpha_eff)
    lo_c, hi_c = paired_bootstrap_ci(
        by_tile[t_star], by_tile[Tl.MAX_TILE], alpha=alpha_eff
    )
    beats_fine, beats_coarse = hi_f < 0.0, hi_c < 0.0

    if beats_fine and beats_coarse:
        verdict = "interior"
    elif means[t_star] >= min(means[1], means[Tl.MAX_TILE]):
        verdict = "edge"
    else:
        verdict = "undetermined"
    return {
        "verdict": verdict,
        "t_star": t_star,
        "means": {str(k): v for k, v in means.items()},
        "vs_T1_ci": (lo_f, hi_f),
        "vs_Tmax_ci": (lo_c, hi_c),
        "beats_fine": beats_fine,
        "beats_coarse": beats_coarse,
        "n_draws": n_draws,
        "alpha_effective": alpha_eff,
        "note": "argmin alone is not evidence; both edges must be separated",
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class GateRun:
    budgets: tuple = DEFAULT_BUDGETS
    tiles: tuple = DEFAULT_TILES
    seeds: tuple = (0, 1, 2)
    axis: str = "B"
    metric: str = "wanda"
    compensate: bool = True
    rotate_axis: str | None = "index"
    records: list = field(default_factory=list)

    def run(self, problem: LayerProblem) -> dict:
        wall = dense_wall(problem)
        out = {
            "meta": {
                "git": _git_hash(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "layer": problem.name,
                "n_out": problem.n_out,
                "n_in": problem.n_in,
                "axis": self.axis,
                "metric": self.metric,
                "compensate": self.compensate,
                "rotate_axis": self.rotate_axis,
                "seeds": list(self.seeds),
                "survivor_quantizer": "E8P",
                "vq_bits": E8P_BITS,
            },
            "wall": wall,
            "budgets": {},
        }
        for b in self.budgets:
            recs = []
            for t in self.tiles:
                for s in self.seeds:
                    r = run_config(
                        problem, budget_bits=b, tile_size=t, axis=self.axis,
                        metric=self.metric, compensate=self.compensate,
                        rotate_axis=self.rotate_axis, seed=s,
                    )
                    recs.append(r)
            self.records.extend(recs)
            out["budgets"][str(b)] = {
                "records": recs,
                "gate_a": gate_a(recs, wall),
                "gate_b": gate_b(recs),
                "live": A.is_live(
                    A.Config(scheme="tile", vq_bits=E8P_BITS, n_idx=problem.n_in,
                             tile_size=16, budget_bits=b)
                ),
            }
        return out


def _report(out: dict) -> None:
    m = out["meta"]
    print(f"layer {m['layer']}  axis={m['axis']}  metric={m['metric']}  "
          f"quantizer=E8P({m['vq_bits']} bit)  seeds={m['seeds']}")
    print(f"PTQ floor reference: dense E8P @ {out['wall']['bits_realized']} bit  "
          f"-> rel.err {out['wall']['rel_output_error']:.4f}\n")
    for b, blk in out["budgets"].items():
        print(f"=== B = {b} bit {'(live)' if blk['live'] else '(NOT live)'} ===")
        seen = set()
        for r in blk["records"]:
            if "skipped" in r or r["tile_size"] in seen:
                continue
            seen.add(r["tile_size"])
            same = [x["rel_output_error"] for x in blk["records"]
                    if x.get("tile_size") == r["tile_size"] and "skipped" not in x]
            print(f"  T={str(r['tile_size']):<4} d={r['density_realized']:.4f}  "
                  f"bits={r['bits_realized']:.4f} ({r['offset_pct']*100:+.2f}%)  "
                  f"rel.err={sum(same)/len(same):.4f}")
        ga, gb = blk["gate_a"], blk["gate_b"]
        print(f"  Gate A: {ga['verdict']}  (best T={ga.get('best_tile')}, "
              f"{ga.get('best_mean_error', float('nan')):.4f} vs wall "
              f"{ga.get('wall_error', float('nan')):.4f})")
        print(f"  Gate B: {gb['verdict']}  (T*={gb.get('t_star')})\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data as a smoke test")
    ap.add_argument("--n-out", type=int, default=128)
    ap.add_argument("--n-in", type=int, default=256)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budgets", type=float, nargs="*", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--axis", default="B", choices=["A", "B"])
    ap.add_argument("--no-compensate", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.synthetic:
        print("only --synthetic is wired up: model loading and sequential "
              "calibration are separate deliverables. LayerProblem is the seam.",
              file=sys.stderr)
        return 2

    problem = synthetic_problem(args.n_out, args.n_in)
    run = GateRun(
        budgets=tuple(args.budgets), seeds=tuple(range(args.seeds)), axis=args.axis,
        compensate=not args.no_compensate,
        rotate_axis=None if args.no_rotate else "index",
    )
    out = run.run(problem)
    _report(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
