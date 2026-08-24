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
from collections.abc import Sequence
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

#: Width the LDLQ feedback is confined to.  Not a speed compromise -- the block
#: sweep measured 512 as the BEST arm at every tile size, better than the
#: unconstrained feedback by 11-23% (`docs/STATUS.md` section 5.9).  It is also
#: what makes a useful `chunk` affordable, since a confined factor is k*512 per
#: tile instead of k^2.
HESSIAN_BLOCK = 512
DEFAULT_BUDGETS = (1.75, 1.60, 1.50)
DEFAULT_TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)

# --------------------------------------------------------------------------- #
# What the pipeline runs with (user decision, 2026-08-25)
#
# All three were measured, tested and left OFF on 2026-08-24 because every
# quality number to that point had been taken without them.  They are on now,
# and together they take M1 from 15.0 days to 7.5.  The comparability that costs
# is not being waved at: the shakedown run measures the same configuration both
# ways once, so every later number can be tied back to the M0 baseline through a
# MEASURED offset instead of an assumption.
#
# Named here rather than written into the parameter defaults for a reason.
# `run_config` is two things at once -- the pipeline, and the harness the M0
# experiments study it with -- and those want different defaults.  A constant
# the pipeline reads, with the resolved value written into every record, lets
# both be true without either being silent.
#
# TF32 is NOT here and is not a fourth lever: it breaks the pipeline outright
# (rotated sub-Hessian fails Cholesky, 85% of the damping margin gone, section
# 6.9).
# --------------------------------------------------------------------------- #

#: Contract the sub-Hessian rotation against its Kronecker factors instead of
#: forming it densely.  5.52x on the rotation term; on a real layer the quality
#: moves -0.03..-0.31%, i.e. in our favour (section 6.8).
PIPELINE_ROTATE_KRON = True

#: Search the codebook in fp16.  1.24-1.52x on the codebook term, at most 0.90%
#: quality (section 6.9).  Verified not to disturb the search routing: the
#: decoder's miss fraction is the same to within 0.1 pp in fp16 and fp32, so the
#: inequality section 6.13 rests on still holds.
PIPELINE_SEARCH_DTYPE = torch.float16

#: Defer each block of the compensation sweep's errors into one matmul.  6.63x
#: on the term; not bit-identical, but the difference is 2.7e-06..4.8e-06, which
#: is float32's own epsilon at these sizes (section 6.11c).
PIPELINE_COMPENSATE_BLOCK = 512

#: "Whatever the pipeline runs with."  A sentinel rather than `None` because
#: `None` is already a meaningful value for two of the three -- it means "no
#: cast" for `search_dtype` and "the exact column-by-column sweep" for
#: `compensate_block` -- and a default that cannot be distinguished from an
#: explicit request is how a caller loses the ability to ask for the old
#: behaviour.
_PIPELINE = object()


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #

def tile_hessians(
    problem: LayerProblem, cw: C.CompactWeights, Q: Tensor | None = None,
    factors: tuple | None = None,
) -> Tensor:
    """Each tile's input sub-Hessian H[S_t, S_t], optionally rotated to match.

    If the block was rotated as `B Q^T`, the error rotates the same way, so the
    Hessian that keeps the objective invariant is `Q H Q^T`:

        tr((E Q^T)(Q H Q^T)(E Q^T)^T) = tr(E H E^T)
    """
    H = problem.H
    S = cw.idx_index                                     # [n_tiles, k]
    Ht = H[S.unsqueeze(-1), S.unsqueeze(1)]              # [n_tiles, k, k]
    if Q is None:
        return Ht
    if factors is None:
        return Q @ Ht @ Q.transpose(-1, -2)
    return torch.stack([R.rotate_hessian(Ht[t], factors=factors)
                        for t in range(Ht.shape[0])])


def tile_hessian_stream(problem: LayerProblem, cw: C.CompactWeights,
                        Q: Tensor | None = None, factors: tuple | None = None):
    """The same thing, one tile at a time.

    `tile_hessians` builds every tile's sub-Hessian in a single tensor, which is
    fine at test widths and impossible at real ones: a Llama-2-7B `down_proj` at
    T=16 is 256 tiles of 7912 x 7912, or 119 GiB.  LDLQ consumes them strictly in
    order, so nothing is gained by holding them all.

    Returns a callable, because that is what `ldlq_quantize_blocks` takes.
    """
    H = problem.H
    S = cw.idx_index                                     # [n_tiles, k]

    def one(t: int) -> Tensor:
        idx = S[t]
        Ht = H[idx.unsqueeze(-1), idx.unsqueeze(0)]      # [k, k]
        if Q is None:
            return Ht
        # Q is [n_tiles, k, k] when the rotation differs per tile, [k, k] when
        # every tile shares it; both are accepted so the caller need not care.
        q = Q[t] if Q.ndim == 3 else Q
        return R.rotate_hessian(Ht, q, factors=factors)

    return one


def run_config(
    problem: LayerProblem,
    *,
    budget_bits: float,
    tile_size: int | str,
    axis: str = "B",
    metric: str = "wanda",
    compensate: bool = True,
    rotate_axis: str | None = "index",
    rotate_block: int | None = None,
    rotate_kron: bool = _PIPELINE,
    compensate_block: int | None = _PIPELINE,
    hessian_block: int | None = HESSIAN_BLOCK,
    chunk: int | str = "auto",
    scale_sample: int | None = None,
    scale_steps: int = Qz.FIT_STEPS,
    scale_seed: int = 0,
    search_dtype: torch.dtype | None = _PIPELINE,
    quantize: bool = True,
    ldlq: bool = True,
    align: int | None = None,
    scale: str | float = "per_tile",
    seed: int = 0,
    vq_bits: float = E8P_BITS,
    return_weight: bool = False,
) -> dict:
    """Prune -> compact -> rotate -> quantize, in that order, and measure.

    The order is the invariant (plan H1) and `prune` enforces it.

    `return_weight=True` puts the compressed W under `"W_hat"`.  Off by default
    because the grid runs thousands of configs and keeping a weight per record
    would hold the whole sweep in memory; on, it is what lets this function be
    used as `calibrate.sequential_calibrate`'s `compress_fn`, which has to
    RETURN a weight.  Without it the two halves of the seam do not connect and
    a full-model driver cannot be a thin adapter over them -- which is half of
    why `experiments/m1_run.py` (`docs/STATUS.md` section 8.1) does not exist.

    `ldlq=True` rounds against each tile's sub-Hessian, rotated into the same
    basis as its block.  Without it the rotation costs inference time and buys
    nothing on the activation-weighted objective (plan section I3).  It needs
    the survivor count aligned to 8, so the mask is built with `align=8`.

    `align=None` follows that rule.  Pass a number to force it, which the
    transfer pilot needs: it compares a quantized run against an unquantized one
    at EQUAL DENSITY, and letting the alignment differ between them would move
    the realized density and quietly compare two different sparsity levels.

    `hessian_block` defaults to 512 because the block-width sweep measured that
    as the best arm at every tile size -- better than the unconstrained feedback,
    not merely cheaper (`docs/STATUS.md` section 5.9).  `rotate_block` defaults
    to None for the same reason, from the same measurement: confining the
    ROTATION costs quality at every width tried.  `chunk="auto"` sweeps as many
    tiles together as memory and saturation allow, which is bit-identical to one
    at a time and 5-12x faster.

    `rotate_block` and `hessian_block` are the two halves of the block-diagonal
    proposal (`docs/STATUS.md` section 6.3), and they are separate arguments on
    purpose.  `rotate_block` confines the rotation; `hessian_block` drops the
    sub-Hessian couplings that reach past the same width.  Only the second one
    saves the factorization -- the first is what makes dropping them defensible
    -- so an experiment that moved them together could not say which of the two
    cost the quality.

    `scale_sample` and `scale_steps` cap the per-tile scale fit, which after the
    sweep was chunked was most of what a tile costs -- 83% then, 28% now that
    `fit_scale` batches its candidates.  They default to the full
    fit because that is what every quality number so far was measured under;
    `experiments/m0_scale_fit.py` is what prices moving them.

    `scale="per_layer"` fits the quantizer's scale once from a sample instead of
    once inside every tile.  That sweep was 83% of the pipeline's runtime and is
    28%, so the switch is no longer worth several-fold -- about 1.4 days off M1.
    It is not the default, and now has neither a cost case nor a quality one
    (measured 11% worse, 2026-08-23).
    """
    # The three pipeline levers, resolved before anything reads them, and
    # written into the record below so a row always says what produced it.
    kron_explicit = rotate_kron is not _PIPELINE
    if not kron_explicit:
        rotate_kron = PIPELINE_ROTATE_KRON
    if search_dtype is _PIPELINE:
        search_dtype = PIPELINE_SEARCH_DTYPE
    if compensate_block is _PIPELINE:
        compensate_block = PIPELINE_COMPENSATE_BLOCK

    # The Kronecker contraction is of the FULL index-axis rotation; a
    # block-diagonal or line-axis one is a different matrix.  Asking for it
    # explicitly there is an error, but inheriting it from the pipeline default
    # must not turn every block-width arm into a crash -- so it resolves off,
    # and `rotate_kron_auto_disabled` says that it did.  Silence is what would
    # be wrong here, not the downgrade.
    kron_incompatible = rotate_axis != "index" or rotate_block is not None
    kron_auto_disabled = False
    if rotate_kron and kron_incompatible:
        if kron_explicit:
            raise ValueError(
                "rotate_kron applies to the full index-axis rotation; "
                "a block-diagonal one is a different matrix"
            )
        rotate_kron, kron_auto_disabled = False, True

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
        compensate_block=compensate_block,
        align=(Qz.E8P_DIM if (quantize and ldlq) else 1) if align is None else align,
    )

    W_hat = pruned.W
    if quantize:
        cw = C.compact(pruned.W, pruned.mask)
        rotated, Qm = (R.rotate(cw, axis=rotate_axis, seed=seed,
                                block=rotate_block)
                       if rotate_axis else (cw, None))
        if ldlq:
            # Streamed: at real widths the stacked form is hundreds of GiB.
            n_chunk = (Qz.auto_chunk(cw.n_tiles, cw.lines_per_tile, cw.k,
                                     rotated.blocks.element_size(),
                                     hessian_block)
                       if chunk == "auto" else int(chunk))
            factors = None
            if rotate_kron:
                factors = R.kronecker_factors(
                    cw.k, seed, rotated.blocks.dtype, rotated.blocks.device)
            qb = Qz.ldlq_quantize_blocks(
                rotated.blocks,
                tile_hessian_stream(
                    problem, cw, Qm if rotate_axis == "index" else None,
                    factors=factors),
                scale=scale,
                scale_sample=scale_sample,
                scale_steps=scale_steps,
                scale_seed=scale_seed,
                search_dtype=search_dtype,
                hessian_block=hessian_block,
                chunk=n_chunk,
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
    out = {
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
        "rotate_block": rotate_block,
        "rotate_kron": rotate_kron,
        "rotate_kron_auto_disabled": kron_auto_disabled,
        "compensate_block": compensate_block,
        "hessian_block": hessian_block,
        "chunk": chunk,
        "quantize": quantize,
        "ldlq": ldlq,
        "scale_policy": scale,
        "scale_sample": scale_sample,
        "scale_steps": scale_steps,
        "scale_seed": scale_seed,
        "search_dtype": None if search_dtype is None else str(search_dtype),
        "align": (Qz.E8P_DIM if (quantize and ldlq) else 1) if align is None else align,
        "survivors_per_tile": int(pruned.mask.survivors_per_tile().max()),
        "seed": seed,
        "rel_output_error": problem.output_error(W_hat),
        "snr_db": Qz.quantization_snr(problem.W, W_hat),
        "in_bitmap_regime": A.in_bitmap_regime(
            budget_bits, None, pruned.mask.n_idx,
            tile_size=tile_size if scheme == "tile" else 1, vq_bits=vq_bits,
        ),
    }
    if return_weight:
        out["W_hat"] = W_hat
    return out


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


def t_star_set(records: list[dict], alpha: float = 0.05) -> dict:
    """Which granularities are NOT distinguishable from the best one.

    Gate B's verdict and the headline T* are different claims with different
    evidence behind them, and the second is the weaker one.  Separating the
    optimum from the EDGES is a large difference; separating it from its
    NEIGHBOUR is a small one, and the power analysis
    (`experiments/m0_gate_b_power.py`) puts numbers on the gap: with a flat
    interior at one sigma and twenty draws, the verdict is right 76% of the time
    while the argmin is right 48% -- barely better than picking between the two
    tiles nearest the bottom.

    So T* is reported as a SET: the argmin, plus every interior tile whose
    paired difference from it cannot be shown to be positive.  A one-element set
    is a real claim about granularity; a four-element set says the curve is flat
    and the honest headline is "interior", not "T = 8".

    The test is one-sided by construction: every other tile has a mean at or
    above the argmin's, so only the lower end of the interval can settle
    anything.  Reading one end of a two-sided interval at `alpha_eff` makes the
    effective level `alpha_eff / 2`, i.e. the set errs toward being too large.
    That is the right direction to err -- an over-wide set understates the
    claim, a too-narrow one manufactures a granularity result.
    """
    by_tile: dict[object, list[float]] = {}
    for r in records:
        if "skipped" not in r:
            by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])

    tiles = [t for t in by_tile if t not in (1, Tl.MAX_TILE)]
    if not tiles:
        return {"t_star": None, "set": [], "reason": "no interior tiles"}

    means = {t: sum(by_tile[t]) / len(by_tile[t]) for t in tiles}
    t_star = min(tiles, key=lambda t: means[t])
    alpha_eff = alpha / max(1, len(tiles) - 1)

    keep, detail = [t_star], {}
    for t in tiles:
        if t == t_star:
            continue
        lo, hi = paired_bootstrap_ci(by_tile[t], by_tile[t_star], alpha=alpha_eff)
        separated = lo > 0.0
        detail[str(t)] = {"ci": (lo, hi), "separated_from_t_star": separated}
        if not separated:
            keep.append(t)

    return {
        "t_star": t_star,
        "set": sorted(keep, key=lambda t: (t == Tl.MAX_TILE, t)),
        "n_candidates": len(tiles),
        "alpha_effective": alpha_eff,
        "detail": detail,
        "note": ("a set larger than one means the granularity axis is flat "
                 "near the optimum; report the set, not the argmin"),
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
    """The M1 grid: budgets x tile sizes x draws.

    A DRAW is a calibration draw -- a different sample of text the layer sees --
    and that is what `run` wants: pass a sequence of `LayerProblem`s, one per
    draw, sharing a fixed rotation seed.

    Passing a single problem instead falls back to varying the ROTATION seed,
    which is a different and much smaller noise source: measured on a synthetic
    layer at 0.72% of the error level against 1.41% for calibration draws
    (`experiments/m0_gate_b_power.py`).  Gate B run on rotation seeds would
    therefore be roughly twice as confident as the evidence supports, so the
    fallback records what it did and `gate_b`'s output is marked.
    """
    budgets: tuple = DEFAULT_BUDGETS
    tiles: tuple = DEFAULT_TILES
    seeds: tuple = (0, 1, 2)
    axis: str = "B"
    metric: str = "wanda"
    compensate: bool = True
    rotate_axis: str | None = "index"
    records: list = field(default_factory=list)

    def run(self, problem: LayerProblem | Sequence[LayerProblem]) -> dict:
        problems = [problem] if isinstance(problem, LayerProblem) else list(problem)
        if not problems:
            raise ValueError("need at least one LayerProblem")
        draw_axis = "calibration" if len(problems) > 1 else "rotation_seed"
        # One draw per problem when problems vary; otherwise one per seed.
        draws = ([(p, self.seeds[0]) for p in problems] if draw_axis == "calibration"
                 else [(problems[0], s) for s in self.seeds])
        wall = dense_wall(problems[0])
        out = {
            "meta": {
                "git": _git_hash(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "layer": problems[0].name,
                "n_out": problems[0].n_out,
                "n_in": problems[0].n_in,
                "axis": self.axis,
                "metric": self.metric,
                "compensate": self.compensate,
                "rotate_axis": self.rotate_axis,
                "seeds": list(self.seeds),
                "draw_axis": draw_axis,
                "n_draws": len(draws),
                "survivor_quantizer": "E8P",
                "vq_bits": E8P_BITS,
            },
            "wall": wall,
            "budgets": {},
        }
        for b in self.budgets:
            recs = []
            for t in self.tiles:
                for prob, s in draws:
                    r = run_config(
                        prob, budget_bits=b, tile_size=t, axis=self.axis,
                        metric=self.metric, compensate=self.compensate,
                        rotate_axis=self.rotate_axis, seed=s,
                    )
                    r["draw_axis"] = draw_axis
                    recs.append(r)
            self.records.extend(recs)
            out["budgets"][str(b)] = {
                "records": recs,
                "gate_a": gate_a(recs, wall),
                "gate_b": dict(gate_b(recs), draw_axis=draw_axis),
                "t_star_set": t_star_set(recs),
                "live": A.is_live(
                    A.Config(scheme="tile", vq_bits=E8P_BITS,
                             n_idx=problems[0].n_in, tile_size=16, budget_bits=b)
                ),
            }
        return out


def _report(out: dict) -> None:
    m = out["meta"]
    print(f"layer {m['layer']}  axis={m['axis']}  metric={m['metric']}  "
          f"quantizer=E8P({m['vq_bits']} bit)  "
          f"{m.get('n_draws', len(m['seeds']))} draws over "
          f"{m.get('draw_axis', 'rotation_seed')}")
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
        ts = blk["t_star_set"]
        members = ", ".join(str(t) for t in ts["set"])
        print(f"  Gate B: {gb['verdict']}  (T*={gb.get('t_star')}; "
              f"not separable from it: {{{members}}})\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data as a smoke test")
    ap.add_argument("--n-out", type=int, default=128)
    ap.add_argument("--n-in", type=int, default=256)
    ap.add_argument("--draws", type=int, default=3,
                    help="calibration draws -- the axis Gate B's CIs are over")
    ap.add_argument("--rotation-seeds-as-draws", action="store_true",
                    help="replicate over the rotation seed instead; measured at "
                         "about half the noise, so Gate B comes out overconfident")
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

    run = GateRun(
        budgets=tuple(args.budgets), seeds=tuple(range(args.draws)), axis=args.axis,
        compensate=not args.no_compensate,
        rotate_axis=None if args.no_rotate else "index",
    )
    if args.rotation_seeds_as_draws:
        out = run.run(synthetic_problem(args.n_out, args.n_in))
    else:
        out = run.run([synthetic_problem(args.n_out, args.n_in, seed=d)
                       for d in range(args.draws)])
    _report(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
