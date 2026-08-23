"""M0 -- what does a cheaper scale fit cost in quality?

After the sweep was chunked across tiles, `fit_scale` is nearly all of what a
tile costs: it finds ONE scalar per tile by scanning that tile's vectors 24
times, which is 24x the work of actually quantizing them.  Priced on this
machine, removing it entirely would take M1 from 48 days to 14.  It is the last
large lever, and unlike the block width it can only COST quality -- so the bar
here is "indistinguishable", not "acceptable".

Two knobs, and they multiply:

    sample   how many of the tile's vectors each pass looks at
    steps    how many candidate scales are tried

They are not the same lever.  Sampling keeps the search resolution and
estimates alpha from fewer vectors; fewer steps keeps every vector and coarsens
the search.  Which one degrades first is an empirical question and the reason
this sweeps both rather than picking one.

A correction this experiment exists to respect: `docs/STATUS.md` once read the
scale lever as "sampled scale, 68 days".  It is not.  `fit_scale(sample=N)`
cannot look at more vectors than a tile HAS, and at B=1.5 a tile holds 128 at
T=1, 1,280 at T=4 and 5,888 at T=16.  The default cap of 8,192 is inert at
every tile size where the cost lives.  The caps that bite are small, which is
exactly why their quality has to be measured rather than assumed.

`per_layer` is included as an anchor, not as a candidate: it was measured 11%
worse and rejected on 2026-08-23.  It is here so the sampled arms can be read
against a known-bad option as well as against the full fit.

Scope, as in `m0_rotation_value.py`: one real layer, layer output error rather
than perplexity.  Enough for the mechanism, not enough to quote.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m1_gates as M                        # noqa: E402
import quantize as Qz                       # noqa: E402
import tiling as Tl                         # noqa: E402
from m0_rotation_value import build_problem, DEFAULT_LAYER, DEFAULT_MODEL  # noqa: E402

DEFAULT_TILES = (4, 16, Tl.MAX_TILE)
DEFAULT_SAMPLES = (2048, 512, 256, 128)
DEFAULT_STEPS = (24, 12, 6)


@dataclass(frozen=True)
class Arm:
    """One scale-fitting configuration.  `sample=None` means every vector."""
    name: str
    sample: int | None = None
    steps: int = Qz.FIT_STEPS
    seed: int = 0
    policy: str | float = "per_tile"


def arms_for(samples=DEFAULT_SAMPLES, steps=DEFAULT_STEPS,
             seeds=(0,), anchor: bool = True) -> list[Arm]:
    """The full fit, then every (sample, steps) pair, then the rejected anchor.

    Steps are swept at `sample=None` too, so the two knobs can be told apart:
    if `s512/n6` degrades and `full/n6` does not, the sample is what hurt.
    """
    out = [Arm("full")]
    for n in steps:
        if n != Qz.FIT_STEPS:
            out.append(Arm(f"n{n}", steps=n))
    for s in samples:
        for n in steps:
            out.append(Arm(f"s{s}/n{n}", sample=s, steps=n))
    for seed in seeds:
        if seed:
            out.append(Arm(f"s512/n24#{seed}", sample=512, seed=seed))
    if anchor:
        out.append(Arm("per_layer", policy="per_layer"))
    return out


def compare(problem, *, budget: float = 1.5, tiles=DEFAULT_TILES,
            arms=None, progress=print) -> list:
    """Every arm at each tile size, on the same layer and the same Hessian.

    Wall time is recorded beside the error because the whole point is a trade,
    and a table with only one of the two cannot express it.  The time is this
    script's, on a 512-row slice, so it is a RATIO to the full fit and not a
    figure to carry anywhere else.
    """
    arms = list(arms or arms_for())
    rows = []
    for t in tiles:
        measured = {}
        for arm in arms:
            t0 = time.time()
            r = M.run_config(problem, budget_bits=budget, tile_size=t,
                             scale=arm.policy, scale_sample=arm.sample,
                             scale_steps=arm.steps, scale_seed=arm.seed)
            if "skipped" in r:
                measured = {}
                break
            r["seconds"] = time.time() - t0
            measured[arm.name] = r
            progress(f"  T={t} {arm.name:>12}: rel.err {r['rel_output_error']:.6f}"
                     f"  ({r['seconds']:.1f}s)")
        if len(measured) != len(arms):
            continue
        full = measured["full"]
        for arm in arms:
            r = measured[arm.name]
            rows.append({
                "tile_size": t,
                "arm": arm.name,
                "sample": arm.sample,
                "steps": arm.steps,
                "scale_seed": arm.seed,
                "policy": arm.policy,
                "k": full["survivors_per_tile"],
                "density": r["density_realized"],
                "rel_output_error": r["rel_output_error"],
                "snr_db": r["snr_db"],
                "vs_full": ((r["rel_output_error"] - full["rel_output_error"])
                            / full["rel_output_error"]),
                "seconds": r["seconds"],
                "speedup": full["seconds"] / r["seconds"],
            })
    return rows


def run(model_name: str = DEFAULT_MODEL, *, layer: str = DEFAULT_LAYER,
        budget: float = 1.5, tiles=DEFAULT_TILES, arms=None, n_seqs: int = 16,
        seqlen: int = 2048, dataset: str = "wikitext2", rows: int | None = None,
        solve_device: str = "cuda", solve_dtype: torch.dtype = torch.float32,
        progress=print) -> dict:
    arms = list(arms or arms_for())
    problem = build_problem(model_name, layer=layer, n_seqs=n_seqs,
                            seqlen=seqlen, dataset=dataset, rows=rows,
                            solve_device=solve_device, solve_dtype=solve_dtype,
                            progress=progress)
    measured = compare(problem, budget=budget, tiles=tiles, arms=arms,
                       progress=progress)
    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": ("what does a cheaper per-tile scale fit cost in "
                         "quality -- the last large runtime lever, and the "
                         "only one that can only cost"),
            "model": model_name, "layer": f"layers.0.{layer}",
            "calibration": dataset,
            "n_out": problem.n_out, "n_in": problem.n_in,
            "output_rows_used": rows,
            "solve_device": solve_device, "solve_dtype": str(solve_dtype),
            "n_tokens": problem.n_tokens, "budget": budget,
            "arms": [a.name for a in arms],
            "scope": ("layer output error, not perplexity; wall times are this "
                      "script's on a row slice and are ratios, not figures"),
        },
        "rows": measured,
    }


def _verdict(out: dict, tolerance: float = 0.02) -> None:
    m, rows = out["meta"], out["rows"]
    print("\n" + "=" * 78)
    sliced = ("" if m.get("output_rows_used") is None
              else f" (first {m['output_rows_used']} output rows)")
    print(f"  {m['layer']}  {m['n_out']}x{m['n_in']}{sliced}  "
          f"{m['n_tokens']:,} calibration tokens  B={m['budget']}")

    for t in dict.fromkeys(r["tile_size"] for r in rows):
        here = [r for r in rows if r["tile_size"] == t]
        print(f"\n  T={t}   k={here[0]['k']}   d={here[0]['density']:.4f}")
        print(f"    {'arm':>12} {'rel.err':>10} {'vs full':>9} {'hiz':>7}")
        for r in here:
            print(f"    {r['arm']:>12} {r['rel_output_error']:10.6f}"
                  f" {r['vs_full'] * 100:+8.2f}% {r['speedup']:6.2f}x")

    print("\n" + "-" * 78)
    # Sampling can only cost quality, so the bar is "indistinguishable" rather
    # than "acceptable" -- unlike the block width, which improved it.
    tiles = {r["tile_size"] for r in rows}
    cand = [r for r in rows if r["policy"] == "per_tile" and r["arm"] != "full"]
    names = dict.fromkeys(r["arm"] for r in cand)
    ok = []
    for name in names:
        at = [r for r in cand if r["arm"] == name]
        if len(at) == len(tiles) and all(r["vs_full"] <= tolerance for r in at):
            ok.append((name, max(r["vs_full"] for r in at),
                       min(r["speedup"] for r in at)))
    if not ok:
        best = min(names, key=lambda n: max(
            r["vs_full"] for r in cand if r["arm"] == n))
        worst = max(r["vs_full"] for r in cand if r["arm"] == best)
        print(f"  Nothing stays within {tolerance:.0%} of the full fit at every "
              f"tile size.")
        print(f"  Closest is {best} at {worst * 100:+.2f}%.  The scale fit is "
              f"not free after all.")
    else:
        fastest = max(ok, key=lambda o: o[2])
        print(f"  Within {tolerance:.0%} at every tile size: "
              + ", ".join(n for n, _, _ in ok))
        print(f"  Fastest of those: {fastest[0]} -- {fastest[2]:.2f}x, "
              f"worst {fastest[1] * 100:+.2f}% against the full fit")

    anchor = [r for r in rows if r["policy"] == "per_layer"]
    if anchor:
        print(f"  anchor: per_layer runs "
              f"{sum(r['vs_full'] for r in anchor) / len(anchor) * 100:+.1f}% "
              f"on average (rejected 2026-08-23 at +11%)")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", default=DEFAULT_LAYER)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tiles", nargs="*", default=[str(t) for t in DEFAULT_TILES])
    ap.add_argument("--samples", nargs="*", type=int, default=list(DEFAULT_SAMPLES))
    ap.add_argument("--steps", nargs="*", type=int, default=list(DEFAULT_STEPS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="extra sampling seeds at s512/n24, to see the spread")
    ap.add_argument("--tolerance", type=float, default=0.02)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--solve-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--solve-dtype", default="float32",
                    choices=["float64", "float32"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_scale_fit.json"))
    args = ap.parse_args(argv)

    tiles = [Tl.MAX_TILE if t == Tl.MAX_TILE else int(t) for t in args.tiles]
    out = run(args.model, layer=args.layer, budget=args.budget, tiles=tiles,
              arms=arms_for(tuple(args.samples), tuple(args.steps),
                            tuple(args.seeds)),
              n_seqs=args.seqs, seqlen=args.seqlen, dataset=args.dataset,
              rows=args.rows, solve_device=args.solve_device,
              solve_dtype=getattr(torch, args.solve_dtype),
              progress=lambda s: print(s, flush=True))
    _verdict(out, tolerance=args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
