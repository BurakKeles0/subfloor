"""Is 0.25 a floor or a fault?  Four arms over block 0's cached problems.

`docs/STATUS.md` section 5.12 left three questions and a rule: do not spend
another GPU point until block 0's layers go below ~0.15.  This script answers
the three without loading the model, reading `m0_lever_audit --build`'s cache.

    pipeline    B=1.5,  T=16                  what the driver ran (47.64 ppl)
    quantizer   B=2.0625, T=16 -> d=0.9974    ladder step 4: no pruning
    prune_only  B=1.5,  quantize=False        ladder step 2: the other half
    full_H      B=1.5,  hessian_block=None    ladder step 3: the contradiction

The reading:

  * `quantizer` lands near 0.25 as well  ->  2 bits IS the floor here.  Neither
    `hessian_block` nor the mask can move it, and section 3.2's assumption is
    the thing that fell, not the pipeline.
  * `quantizer` lands at 0.05-0.10  ->  the quantizer is sound and the cost is
    in the pruning; `prune_only` says how much of it.
  * `full_H` beats 512 at the driver's working point  ->  section 5.12 is right
    and section 5.7 was a 512-row artefact.  `--rows 512` runs 5.7's own
    control so the two are compared at the point where they disagree.

BOTH errors are recorded per arm, and the ratio is the point.  The pipeline's
metric is H-weighted (`tr(E H E^T)`); the rate-distortion floor 2^(-2R) that
makes 0.25 suspicious is derived UNWEIGHTED.  Section 5.12 flags the comparison
as not-quite-closed for exactly this reason and calls the ratio cheap.  It is:
one Frobenius norm next to a quadratic form already being computed.

`quantizer` is also the reference `m1_gates.dense_wall` should have been.  That
one takes the whole layer as a single tile with no rotation and no LDLQ, so
`ratio_to_dense < 1` only says the pipeline beats a bare quantizer -- it never
touched the question section 3.2 actually asks, which is whether E8P holds its
quality on the COMPACTED survivor submatrix.  Same pipeline, d~=1, does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accounting as A                                          # noqa: E402
import calibrate as Cal                                         # noqa: E402
import m1_gates as M                                            # noqa: E402
import prune as P                                               # noqa: E402
import quantize as Qz                                           # noqa: E402
from m0_lever_audit import ALL_LAYERS, load_problem             # noqa: E402

#: B=2.0625 at T=16 solves to d=0.997405 -- the closest the grid comes to "no
#: pruning at all" while still running the identical code path.  Hard-coded
#: rather than solved so the arm cannot drift silently; asserted below.
QUANTIZER_BUDGET = 2.0625

ARMS = {
    "pipeline":   dict(budget_bits=1.5),
    "quantizer":  dict(budget_bits=QUANTIZER_BUDGET),
    "prune_only": dict(budget_bits=1.5, quantize=False, align=8),
    "full_H":     dict(budget_bits=1.5, hessian_block=None),
}


def mask_stats(problem: Cal.LayerProblem, *, tile_size: int,
               budget_bits: float = 1.5) -> dict:
    """Is the mask semi-structured, or structured channel pruning wearing a hat?

    Section 5.12 raises this and does not answer it: the saliency score varies
    10-100x across input channels, so every tile may be choosing very nearly the
    same columns.  If it is, `T` is not doing what the `(T, d)` plane assumes --
    T=16 and T=max would be the same mask -- and the thesis's own axis is
    degenerate at this budget.  No quantizer runs; this is set arithmetic on the
    frozen support.

    `union_ratio` is the number of DISTINCT columns any tile kept, over the
    number one tile keeps.  1.0 means every tile picked the identical set
    (structured).  Toward `n_tiles` means the tiles disagree (semi-structured).
    """
    scheme = {1: "unstructured", 4096: "structured"}.get(tile_size, "tile")
    d = A.density_for_budget(scheme, budget_bits, None, problem.n_in,
                             tile_size=tile_size, vq_bits=A.E8P_STORED_BITS)
    pruned = P.prune(problem.W, axis="B", tile_size=tile_size, density=d,
                     metric="wanda", act_norm=problem.act_norm,
                     compensate=False, align=Qz.E8P_DIM)
    sup = pruned.mask.support                       # bool [n_tiles, n_idx]
    n_tiles, n_idx = sup.shape
    per_tile = float(sup.sum(dim=1).float().mean())
    union = int(sup.any(dim=0).sum())

    # Pairwise Jaccard over a bounded sample; 256 tiles is 32k pairs, and the
    # widest layer has 688.  Sampling keeps this seconds rather than minutes.
    jac = None
    if n_tiles > 1:
        g = torch.Generator(device="cpu").manual_seed(0)
        m = min(64, n_tiles)
        pick = torch.randperm(n_tiles, generator=g)[:m]
        sub = sup[pick].float()
        inter = sub @ sub.T
        size = sub.sum(dim=1)
        uni = size[:, None] + size[None, :] - inter
        iu = torch.triu_indices(m, m, offset=1)
        jac = float((inter[iu[0], iu[1]] / uni[iu[0], iu[1]]).mean())

    return {
        "n_tiles": n_tiles, "n_idx": n_idx,
        "survivors_per_tile": per_tile,
        "union_columns": union,
        "union_ratio": union / per_tile if per_tile else None,
        "mean_jaccard": jac,
    }


def unweighted_error(problem: Cal.LayerProblem, W_hat: torch.Tensor) -> float:
    """||W - W_hat||_F / ||W||_F -- the metric the 2^(-2R) floor is derived in."""
    return float((problem.W - W_hat).norm() / problem.W.norm())


def run_arm(problem: Cal.LayerProblem, name: str, *, tile_size: int,
            seed: int = 0) -> dict:
    kw = dict(ARMS[name])
    t0 = time.time()
    out = M.run_config(problem, tile_size=tile_size, seed=seed,
                       return_weight=True, **kw)
    return {
        "arm": name,
        "budget_bits": kw["budget_bits"],
        "bits_realized": out["bits_realized"],
        "density_realized": out["density_realized"],
        "survivors_per_tile": out["survivors_per_tile"],
        "weighted": out["rel_output_error"],
        "unweighted": unweighted_error(problem, out["W_hat"]),
        "seconds": time.time() - t0,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=Path("results/block0_problems"))
    ap.add_argument("--layers", nargs="*", default=list(ALL_LAYERS))
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--mask-only", action="store_true",
                    help="run only the support-overlap diagnostic, which needs "
                         "no quantizer and takes seconds a layer")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--rows", type=int, default=None,
                    help="keep only the first N output rows; 512 reproduces "
                         "section 5.7's measurement point, where H512 won")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path,
                    default=Path("results/floor_probe.json"))
    args = ap.parse_args(argv)

    d = A.density_for_budget("tile", QUANTIZER_BUDGET, n_idx=4096,
                             tile_size=args.tile, vq_bits=A.E8P_STORED_BITS)
    assert d is not None and d > 0.99, f"quantizer arm is not dense: d={d}"
    print(f"quantizer arm: B={QUANTIZER_BUDGET} T={args.tile} -> d={d:.6f}")
    if args.rows:
        print(f"rows truncated to {args.rows} (section 5.7's point, not the "
              f"driver's)")

    rows = []
    for layer in args.layers:
        problem = load_problem(args.cache, layer, device=args.device)
        if args.rows:
            problem = Cal.LayerProblem.from_statistics(
                problem.W[:args.rows].contiguous(), problem.H,
                name=f"{problem.name}[:{args.rows}]",
                n_tokens=problem.n_tokens)
        print(f"\n{layer}  {tuple(problem.W.shape)}  "
              f"{problem.n_tokens} tokens")

        ms = mask_stats(problem, tile_size=args.tile)
        print(f"  mask        {ms['n_tiles']} tiles x "
              f"{ms['survivors_per_tile']:.0f} survivors of {ms['n_idx']}   "
              f"union {ms['union_columns']} (ratio {ms['union_ratio']:.2f})   "
              f"jaccard {ms['mean_jaccard']:.3f}"
              if ms["mean_jaccard"] is not None else
              f"  mask        single tile, no overlap to measure")
        rows.append({"arm": "mask", "layer": layer, **ms})
        if args.mask_only:
            del problem
            continue

        for arm in args.arms:
            r = run_arm(problem, arm, tile_size=args.tile)
            r["layer"] = layer
            r["n_out"], r["n_in"] = problem.W.shape
            rows.append(r)
            print(f"  {arm:<11} weighted {r['weighted']:.4f}   "
                  f"unweighted {r['unweighted']:.4f}   "
                  f"ratio {r['weighted'] / r['unweighted']:.3f}   "
                  f"({r['seconds']:.1f}s)")
        del problem
        torch.cuda.empty_cache() if args.device == "cuda" else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "when": datetime.now(timezone.utc).isoformat(),
        "tile_size": args.tile,
        "rows": args.rows,
        "records": rows,
    }, indent=1))
    print(f"\nwrote {args.out}")

    print("\nby arm, mean over layers:")
    for arm in ([] if args.mask_only else args.arms):
        sel = [r for r in rows if r["arm"] == arm]
        if sel:
            w = sum(r["weighted"] for r in sel) / len(sel)
            u = sum(r["unweighted"] for r in sel) / len(sel)
            print(f"  {arm:<11} weighted {w:.4f}   unweighted {u:.4f}")
    print("\nRule (section 5.12, step 6): if these do not go below ~0.15, do "
          "not spend a GPU point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
