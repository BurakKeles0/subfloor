"""What does batching the scale fit ACROSS TILES cost in quality?

`docs/STATUS.md` section 7.2 carries this as measured-and-not-taken: 2.16x,
"not bit-identical (reduction order)".  Section 8.6 asks for it to be re-priced,
and section 6.18 is why -- once the Kronecker congruence was priced correctly
the codebook term became 52% of the grid, and `fit_scale` is 1.70x of that at
T=16 and more at the fine end.  It is the last lever standing on the largest
term.

THE REJECTION MAY REST ON AN IMPLEMENTATION RATHER THAN ON THE IDEA.  "It
reduces every tile's error together" describes a batched fit that shares the
reduction; one that keeps each tile's error on its own [n, 8] does not have that
property, and `quantize.fit_scales` is written that way.  Whether the output
then moves at all is a question with a measurable answer, and this file is that
measurement.

The same shape has already appeared twice in this project and gone both ways:
`compensate_block` was rejected in section 7.2 on the identical "not
bit-identical" grounds and turned out to cost nothing at all (section 6.18),
while `search_dtype=float16` really did move the answer by up to 0.90%.  So the
premise is not decidable from the outside.

WHAT IS MEASURED, in the order that can end the question early:

  1. THE WEIGHT, BIT FOR BIT.  If `W_hat` is identical the quality cost is
     exactly zero and there is nothing left to quantify.  Asked first because it
     is the cheapest question and the strongest answer.
  2. The scales themselves, tile by tile, so a difference has a place.
  3. The layer objective, if the weights do differ.
  4. The time, because a lever that costs nothing still has to buy something.

ACROSS THE TILE AXIS, not at one cell.  The whole reason this lever exists is
that `_nearest` is launch-bound at the fine end and saturated at the coarse end,
so a measurement at T=16 alone would report ~1.00x and read as a rejection --
which is exactly how the fp16 lever survived a day in the wrong direction, and
how blocking the compensation sweep was written off for eight days (section
6.11c).  T=1 is also the grid's costliest cell and the thesis's unstructured
baseline, so it is the opposite of a corner.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate as Cal              # noqa: E402
import m1_gates as M                 # noqa: E402
import quantize as Qz                # noqa: E402
import tiling as Tl                  # noqa: E402
from bench_guard import alternating, foreign_compute_pids, require_quiet_gpu  # noqa: E402
from m0_lever_audit import (DEFAULT_LAYERS, host_memory_gib,   # noqa: E402
                            load_problem, no_newcomers)

DEFAULT_TILES = ("1", "2", "4", "16", "max")


def run_counted(problem: Cal.LayerProblem, **kw) -> tuple[dict, dict]:
    """`run_config`, counting which fit ran and how many rows it saw.

    Watching the record is not enough -- it can say `batch_fit=True` while the
    argument goes nowhere, which is exactly how `compensate_block` stayed
    unreachable from the driver for a day (section 6.12).  So both entry points
    are counted, and an arm that did not move is refused below.
    """
    seen = {"per_tile": 0, "batched": 0, "tiles_batched": 0}
    real_one, real_many = Qz.fit_scale, Qz.fit_scales

    def spy_one(*a, **k):
        seen["per_tile"] += 1
        return real_one(*a, **k)

    def spy_many(x, *a, **k):
        seen["batched"] += 1
        seen["tiles_batched"] += int(x.shape[0])
        return real_many(x, *a, **k)

    Qz.fit_scale, Qz.fit_scales = spy_one, spy_many
    try:
        record = M.run_config(problem, **kw)
    finally:
        Qz.fit_scale, Qz.fit_scales = real_one, real_many
    return record, seen


def compare(problem: Cal.LayerProblem, *, budget_bits: float, tile_size,
            reps: int = 3, warmup: int = 1, strict: bool = True,
            baseline: dict | None = None, progress=print) -> dict:
    base = {"budget_bits": budget_bits, "tile_size": tile_size,
            "return_weight": True}
    no_newcomers(baseline or {}, strict=strict)

    off, seen_off = run_counted(problem, **base, batch_fit=False)
    on, seen_on = run_counted(problem, **base, batch_fit=True)
    if "skipped" in off:
        progress(f"    T={tile_size}: {off['skipped']}")
        return {"tile_size": str(tile_size), "skipped": off["skipped"]}

    # Both arms must have taken their own path, and only their own.
    problems = []
    if seen_off["per_tile"] == 0 or seen_off["batched"]:
        problems.append(f"batch_fit=False did not fit per tile: {seen_off}")
    if seen_on["batched"] == 0 or seen_on["per_tile"]:
        problems.append(f"batch_fit=True did not batch: {seen_on}")
    if seen_on["tiles_batched"] != seen_off["per_tile"]:
        problems.append(
            f"the arms fitted different tile counts: {seen_on['tiles_batched']} "
            f"batched against {seen_off['per_tile']} one at a time")
    if problems:
        raise RuntimeError("the arms do not isolate the lever:\n  "
                           + "\n  ".join(problems))

    identical = bool(torch.equal(off["W_hat"], on["W_hat"]))
    delta = float((off["W_hat"] - on["W_hat"]).abs().max())
    tiles_per_call = seen_on["tiles_batched"] / max(seen_on["batched"], 1)
    del off["W_hat"], on["W_hat"]
    gc.collect()

    def make(flag):
        def fn():
            M.run_config(problem, budget_bits=budget_bits, tile_size=tile_size,
                         batch_fit=flag)
        return fn

    timed = alternating({"per_tile": make(False), "batched": make(True)},
                        reps=reps, warmup=warmup, strict=strict)
    speed = timed["per_tile"]["median"] / timed["batched"]["median"]

    out = {
        "tile_size": str(tile_size),
        "n_tiles": seen_off["per_tile"],
        "fit_calls": {"per_tile": seen_off["per_tile"],
                      "batched": seen_on["batched"]},
        "tiles_per_batched_call": tiles_per_call,
        "weights_bit_identical": identical,
        "max_abs_weight_delta": delta,
        "rel_output_error": {"per_tile": off["rel_output_error"],
                             "batched": on["rel_output_error"]},
        "quality_pct": (on["rel_output_error"] / off["rel_output_error"] - 1.0)
        * 100.0 if off["rel_output_error"] else float("nan"),
        "seconds": {k: v["median"] for k, v in timed.items()},
        "spread": {k: v["spread"] for k, v in timed.items()},
        "speedup": speed,
    }
    progress(f"    T={str(tile_size):<4} {out['n_tiles']:>5} tile, "
             f"{seen_on['batched']:>4} toplu cagri "
             f"({tiles_per_call:.0f} tile/cagri)  "
             f"bit-birebir {str(identical):<5} "
             f"kalite {out['quality_pct']:+.4f}%  "
             f"{timed['per_tile']['median']:6.2f}s -> "
             f"{timed['batched']['median']:6.2f}s  {speed:.2f}x")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=Path("results/block0_problems"))
    ap.add_argument("--layers", nargs="*", default=list(DEFAULT_LAYERS))
    ap.add_argument("--tiles", nargs="*", default=list(DEFAULT_TILES))
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-strict", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("results/m0_fit_batch.json"))
    args = ap.parse_args(argv)
    strict = not args.no_strict

    baseline: dict[int, str] = {}
    if torch.cuda.is_available():
        print(f"pre-flight: {require_quiet_gpu(strict=strict)}")
        baseline = dict(foreign_compute_pids())
    avail, total = host_memory_gib()
    print(f"pre-flight: host RAM {avail:.1f}/{total:.1f} GiB free")

    out = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "budget_bits": args.budget,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "reps": args.reps, "warmup": args.warmup,
        "pipeline_batch_fit": M.PIPELINE_BATCH_FIT,
        "layers": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for name in args.layers:
        print(f"\n--- {name} ---")
        problem = load_problem(args.cache, name, device=args.device)
        rows = []
        for t in args.tiles:
            tile = Tl.MAX_TILE if t == "max" else int(t)
            try:
                rows.append(compare(problem, budget_bits=args.budget,
                                    tile_size=tile, reps=args.reps,
                                    warmup=args.warmup, strict=strict,
                                    baseline=baseline))
            except Exception as exc:                    # noqa: BLE001
                rows.append({"tile_size": t,
                             "error": f"{type(exc).__name__}: {exc}"})
                print(f"    T={t}: FAILED {type(exc).__name__}: {exc}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        out["layers"].append({"layer": name, "n_out": problem.n_out,
                              "n_in": problem.n_in, "tiles": rows})
        del problem
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    every = [r for L in out["layers"] for r in L["tiles"] if "speedup" in r]
    if every:
        allsame = all(r["weights_bit_identical"] for r in every)
        print(f"BIT-BIREBIR, {len(every)} hucrenin hepsinde: {allsame}")
        if not allsame:
            worst = max(every, key=lambda r: abs(r["quality_pct"]))
            print(f"  en buyuk kalite farki {worst['quality_pct']:+.4f}% "
                  f"(T={worst['tile_size']})")
        best = max(every, key=lambda r: r["speedup"])
        print(f"EN BUYUK HIZ  {best['speedup']:.2f}x  (T={best['tile_size']})")
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
