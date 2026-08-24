"""M0 -- the three precision levers, alone and together.

Three changes are available that trade exactness for speed, and each has been
priced in isolation and never against the others:

    fp16    run the codeword SEARCH in float16 and gather the codeword in the
            caller's dtype (`quantize._nearest(search_dtype=...)`)
    kron    contract `q @ H @ q.T` against the rotation's Kronecker factors
            instead of forming it densely (`rotation.rotate_hessian`)
    tf32    let float32 matmuls use TF32 tensor cores
            (`torch.backends.cuda.matmul.allow_tf32`)

Why they cannot be priced separately.  They do not touch disjoint work.  `fp16`
lands on the codeword search, `kron` on the sub-Hessian rotation -- those two are
independent -- but `tf32` lands on the rotation as well, so its gain and
`kron`'s are drawn from one pot.  Assuming a gain that another change has
already collected is exactly the mistake `docs/STATUS.md` section 6.3 records
against the Triton estimate.

And the obvious null is the wrong one.  Two levers on disjoint work do NOT
multiply their speedups, they beat the product: `S = 1/(1-a)` for a lever that
removes fraction `a`, so disjoint savings add their FRACTIONS and give
`1/(1-a-b)`, which exceeds `1/((1-a)(1-b))`.  Reading a pair that beats the
product as evidence of synergy would be reading Amdahl's law as a discovery.
`_disjoint` is the comparison this file actually uses.

What is new here is TF32's QUALITY.  Section 3.3 lists it as never measured, and
section 7.1 turned it down on that basis rather than on a number: it drops the
mantissa to 10 bits and the Hessian is LDLQ's input, so "1.66x on the rotation"
was never enough to decide with.  This measures it on a real layer.

Two scopes worth stating.

  * The window is `run_config`: prune, compact, rotate, stream the sub-Hessians,
    LDLQ.  It does NOT include accumulating the Hessian, which happens once per
    block during calibration and which TF32 also speeds up (1.74x, section 7.1).
    The cost model does not charge that term either, so leaving it out keeps the
    two consistent -- but it means TF32's real end-to-end value is understated
    here, and by an amount nobody has measured.
  * Quality is layer output error, not perplexity.  Enough to decide whether a
    lever is safe to switch on, not enough to quote.

Read against Gate B: neighbouring tile sizes differ by 0.31 sigma, which is 3.2%
of the error level (sections 5.6 and 5.8).  A lever that moves quality far under
that is invisible to M1; one that approaches it is competing with the signal.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m0_rotation_value as RV        # noqa: E402
import m1_gates as M                  # noqa: E402
import tiling as Tl                   # noqa: E402

DEFAULT_TILES = (4, 16, Tl.MAX_TILE)
LEVERS = ("fp16", "kron", "tf32")


@dataclass(frozen=True)
class Arm:
    """One combination of the three levers.  `name` is "-" for none of them."""
    fp16: bool = False
    kron: bool = False
    tf32: bool = False

    @property
    def name(self) -> str:
        on = [n for n, v in zip(LEVERS, (self.fp16, self.kron, self.tf32)) if v]
        return "+".join(on) if on else "-"

    @property
    def kwargs(self) -> dict:
        return {"search_dtype": torch.float16 if self.fp16 else None,
                "rotate_kron": self.kron}


def all_arms() -> list[Arm]:
    """All eight, ordered so the singles come before the pairs.

    Ordered rather than enumerated in binary because the reading order is the
    argument: each single is read against `-`, each pair against its two
    singles, and the triple against the three pairs.  That is how a gain that
    two levers are sharing shows up as a gap rather than as a surprise.
    """
    combos = sorted(itertools.product((False, True), repeat=3), key=sum)
    return [Arm(*c) for c in combos]


class tf32:
    """Toggle TF32 matmuls, restoring whatever was there before.

    A context manager rather than a flag set once at startup, because the whole
    point is to measure with it both ways in one process -- and because leaving
    a global precision setting flipped is how an unrelated later measurement
    silently becomes about something else.
    """

    def __init__(self, on: bool):
        self.on = on

    def __enter__(self):
        self.prev = (torch.backends.cuda.matmul.allow_tf32,
                     torch.get_float32_matmul_precision())
        torch.backends.cuda.matmul.allow_tf32 = self.on
        torch.set_float32_matmul_precision("high" if self.on else "highest")
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.prev[0]
        torch.set_float32_matmul_precision(self.prev[1])
        return False


def _run(problem, arm: Arm, tile, budget: float, seed: int) -> dict:
    """One arm at one tile size.  A numerical failure is a RESULT, not a crash.

    TF32 makes the damped sub-Hessian fail its Cholesky on this layer, and that
    is the most useful thing the lever has to say -- far more decisive than the
    quality percentage `docs/STATUS.md` section 3.3 was waiting for.  Letting it
    propagate would lose the other seven arms to it.
    """
    with tf32(arm.tf32):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            r = M.run_config(problem, budget_bits=budget, tile_size=tile,
                             seed=seed, **arm.kwargs)
        except Exception as e:                      # noqa: BLE001 -- see above
            return {"failed": f"{type(e).__name__}: {e}".split(chr(10))[0],
                    "seconds": time.perf_counter() - t0}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        r["seconds"] = time.perf_counter() - t0
    return r


def speed(problem, *, tiles=DEFAULT_TILES, budget: float = 1.5, passes: int = 3,
          arms=None, progress=print) -> list[dict]:
    """Wall time per arm, INTERLEAVED across arms within each pass.

    Interleaved because absolute timings on this machine are not comparable
    between runs -- an unrelated job, or nothing identifiable at all, moves them
    by a third (`docs/STATUS.md` section 10).  Taking the median of interleaved
    passes measures the arms against each other, which is the only comparison
    being asked for.
    """
    arms = list(arms or all_arms())
    out = []
    for t in tiles:
        times: dict[str, list[float]] = {a.name: [] for a in arms}
        failed: dict[str, str] = {}
        for p in range(passes):
            for a in arms:
                r = _run(problem, a, t, budget, seed=0)
                if "skipped" in r:
                    times = {}
                    break
                if "failed" in r:
                    failed[a.name] = r["failed"]
                    continue
                times[a.name].append(r["seconds"])
            if not times:
                break
            progress(f"  T={t} pass {p + 1}/{passes} done")
        if not times or not times[arms[0].name]:
            continue
        base = statistics.median(times[arms[0].name])
        for a in arms:
            if not times[a.name]:
                out.append({"tile_size": t, "arm": a.name, "seconds": None,
                            "speedup": None, "failed": failed.get(a.name)})
                continue
            med = statistics.median(times[a.name])
            out.append({"tile_size": t, "arm": a.name, "seconds": med,
                        "speedup": base / med})
    return out


def quality(problem, *, tiles=DEFAULT_TILES, budget: float = 1.5, seed: int = 0,
            arms=None, progress=print) -> list[dict]:
    """Layer output error per arm.  Deterministic, so one run per cell."""
    arms = list(arms or all_arms())
    out = []
    for t in tiles:
        measured = {}
        for a in arms:
            r = _run(problem, a, t, budget, seed)
            if "skipped" in r:
                measured = {}
                break
            measured[a.name] = r
            if "failed" in r:
                progress(f"  T={t} {a.name:>16}: FAILED -- {r['failed']}")
            else:
                progress(f"  T={t} {a.name:>16}: rel.err "
                         f"{r['rel_output_error']:.6f}  ({r['seconds']:.0f}s)")
        if not measured or "failed" in measured[arms[0].name]:
            continue
        base = measured[arms[0].name]["rel_output_error"]
        for a in arms:
            r = measured[a.name]
            failed = r.get("failed")
            err = None if failed else r["rel_output_error"]
            out.append({
                "tile_size": t, "arm": a.name,
                "fp16": a.fp16, "kron": a.kron, "tf32": a.tf32,
                "rel_output_error": err,
                "snr_db": None if failed else r["snr_db"],
                "vs_none": None if failed else err / base - 1.0,
                "seconds": r["seconds"],
                "failed": failed,
            })
    return out


#: Gate B resolves a difference of this size in the error level (sections 5.6,
#: 5.8).  Not a tolerance anyone chose -- it is what the experiment downstream
#: can physically see, so it is the only threshold that means anything here.
GATE_B_RESOLUTION = 0.032


def _disjoint(*speedups: float) -> float:
    """Speedup if every lever removed a DISJOINT slice of the time.

    `S = 1/(1-a)` for a lever that removes fraction `a`, so the fractions add
    and the combined speedup is `1/(1 - sum(a))`.  Returns infinity if the
    fractions would consume the whole runtime, which only happens when the
    inputs are not really disjoint.
    """
    left = 1.0 - sum(1.0 - 1.0 / s for s in speedups)
    return float("inf") if left <= 0 else 1.0 / left


def _verdict(out: dict) -> None:
    q, sp = out["quality"], out["speed"]
    m = out["meta"]
    print("\n" + "=" * 78)
    print(f"  {m['layer']}  {m['n_out']}x{m['n_in']} "
          f"(first {m['output_rows_used']} output rows)  "
          f"{m['n_tokens']:,} calibration tokens  B={m['budget']}")

    # From whichever phase ran: `--phase speed` leaves `quality` empty, and
    # deriving the axes from it would silently print an empty table over a full
    # result file.
    rowsrc = q or sp
    tiles = list(dict.fromkeys(r["tile_size"] for r in rowsrc))
    names = list(dict.fromkeys(r["arm"] for r in rowsrc))

    print("\n  QUALITY -- layer output error relative to no lever")
    print(f"    {'arm':>16}" + "".join(f"{'T=' + str(t):>12}" for t in tiles)
          + f"{'worst':>10}")
    worst_of, broke = {}, {}
    for n in names:
        cells, w, bad = [], 0.0, 0
        for t in tiles:
            r = next((x for x in q if x["arm"] == n and x["tile_size"] == t), None)
            if r is None:
                cells.append("")
            elif r.get("failed"):
                cells.append(f"{'FAILS':>11}")
                bad += 1
            else:
                cells.append(f"{r['vs_none'] * 100:+11.3f}%")
                w = max(w, abs(r["vs_none"]))
        worst_of[n], broke[n] = w, bad
        tail = f"{'--':>9}" if bad else f"{w * 100:9.3f}%"
        print(f"    {n:>16}" + "".join(f"{c:>12}" for c in cells) + tail)

    print("\n  SPEED -- relative to no lever, median of interleaved passes")
    print(f"    {'arm':>16}" + "".join(f"{'T=' + str(t):>12}" for t in tiles))
    speed_of = {}
    for n in names:
        cells, best = [], []
        for t in tiles:
            r = next((x for x in sp if x["arm"] == n and x["tile_size"] == t), None)
            if r is None or r.get("speedup") is None:
                cells.append(f"{'FAILS':>11}" if r is not None else "")
                continue
            cells.append(f"{r['speedup']:11.2f}x")
            best.append(r["speedup"])
        speed_of[n] = statistics.median(best) if best else float("nan")
        print(f"    {n:>16}" + "".join(f"{c:>12}" for c in cells))

    print("\n" + "-" * 78)
    print(f"  {'arm':>16} {'speed':>8} {'worst quality':>15} {'vs Gate B 3.2%':>16}")
    for n in names:
        w = worst_of[n]
        if broke.get(n):
            print(f"  {n:>16} {'--':>7}  {'--':>14} {'DOES NOT RUN':>16}")
            continue
        where = ("invisible" if w < GATE_B_RESOLUTION / 4 else
                 "under it" if w < GATE_B_RESOLUTION else "COMPETES")
        print(f"  {n:>16} {speed_of[n]:7.2f}x {w * 100:14.3f}% {where:>16}")

    # Whether the levers compose is the question a single-lever table cannot
    # answer, and it is why this file exists.
    #
    # The null is NOT the product of the two speedups.  If a lever removes
    # fraction `a` of the time then `S = 1/(1-a)`, so two levers on DISJOINT
    # work give `1/(1-a-b)`, while the product gives `1/(1-a-b+ab)` -- strictly
    # smaller.  Speedups on disjoint work compound better than multiplication,
    # and reading a pair that beats the product as "synergy" would be reading
    # Amdahl's law as a discovery.  Inverting to fractions and adding is the
    # comparison that means something: at or above 100% the levers are
    # independent, below it they are drawing from the same pot.
    print("\n  do they compose?  measured pair against disjoint-saving "
          "1/(1/A + 1/B - 1)")
    import math
    for a, b in (("fp16", "kron"), ("fp16", "tf32"), ("kron", "tf32")):
        pair = f"{a}+{b}"
        vals = [speed_of.get(x) for x in (a, b, pair)]
        if any(v is None or math.isnan(v) for v in vals):
            print(f"    {pair:>16}  not comparable: one of its arms does not run")
            continue
        pred = _disjoint(speed_of[a], speed_of[b])
        print(f"    {pair:>16}  disjoint would be {pred:5.2f}x   measured "
              f"{speed_of[pair]:5.2f}x   {speed_of[pair] / pred * 100:5.0f}% of it")
    triple = "+".join(LEVERS)
    if triple in speed_of and not any(
            speed_of.get(x) is None or math.isnan(speed_of.get(x, float("nan")))
            for x in ("fp16", "kron", "tf32", triple)):
        pred = _disjoint(speed_of["fp16"], speed_of["kron"], speed_of["tf32"])
        print(f"    {triple:>16}  disjoint would be {pred:5.2f}x   measured "
              f"{speed_of[triple]:5.2f}x   {speed_of[triple] / pred * 100:5.0f}% of it")


def run(model_name: str = RV.DEFAULT_MODEL, *, layer: str = RV.DEFAULT_LAYER,
        budget: float = 1.5, tiles=DEFAULT_TILES, n_seqs: int = 16,
        seqlen: int = 2048, dataset: str = "wikitext2", rows: int | None = 512,
        speed_rows: int = 256, passes: int = 3, solve_device: str = "cuda",
        solve_dtype: torch.dtype = torch.float32, phase: str = "both",
        progress=print) -> dict:
    arms = all_arms()

    progress("building the real layer ...")
    problem = RV.build_problem(model_name, layer=layer, n_seqs=n_seqs,
                               seqlen=seqlen, dataset=dataset, rows=rows,
                               solve_device=solve_device,
                               solve_dtype=solve_dtype, progress=progress)

    q = []
    if phase in ("both", "quality"):
        progress("\nquality (real layer) ...")
        q = quality(problem, tiles=tiles, budget=budget, arms=arms,
                    progress=progress)

    # Timed on a narrower slice of the SAME layer: per-tile economics depend on
    # the line count and the width, not on how many tiles there are, so fewer
    # output rows measures the same thing for less wall clock.
    #
    # Separable from the quality phase because they have different hygiene.
    # Quality is deterministic and needs one pass whatever else the machine is
    # doing; speed needs an IDLE GPU, and contention here does not merely add
    # noise -- it moves the bottleneck onto the card and hides exactly the
    # launch overhead these levers remove, so a contended run reads 1.00x and
    # looks like a result (`docs/STATUS.md` section 10).
    sp = []
    if phase in ("both", "speed"):
        progress(f"\nspeed ({speed_rows} rows, {passes} interleaved passes) ...")
        narrow = M.LayerProblem.from_statistics(
            problem.W[:speed_rows].contiguous(), problem.H,
            name=problem.name + f"[:{speed_rows}]", n_tokens=problem.n_tokens)
        sp = speed(narrow, tiles=tiles, budget=budget, passes=passes, arms=arms,
                   progress=progress)

    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": ("what do fp16 search, the Kronecker congruence and "
                         "TF32 cost in quality and buy in speed, alone and in "
                         "every combination"),
            "model": model_name, "layer": f"layers.0.{layer}",
            "n_out": problem.n_out, "n_in": problem.n_in,
            "output_rows_used": rows, "speed_rows": speed_rows,
            "n_tokens": problem.n_tokens, "budget": budget,
            "passes": passes, "phase": phase,
            "solve_device": solve_device, "solve_dtype": str(solve_dtype),
            "gate_b_resolution": GATE_B_RESOLUTION,
            "scope": ("run_config only -- excludes Hessian accumulation, which "
                      "TF32 also speeds up (1.74x) and which the cost model "
                      "does not charge either"),
        },
        "quality": q,
        "speed": sp,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=RV.DEFAULT_MODEL)
    ap.add_argument("--layer", default=RV.DEFAULT_LAYER)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tiles", nargs="*", default=[str(t) for t in DEFAULT_TILES])
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--speed-rows", type=int, default=256)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--phase", default="both",
                    choices=["both", "quality", "speed"],
                    help="quality is deterministic and tolerates a busy "
                         "machine; speed does not, so they are separable")
    ap.add_argument("--solve-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--solve-dtype", default="float32",
                    choices=["float64", "float32"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_precision_levers.json"))
    args = ap.parse_args(argv)

    tiles = [Tl.MAX_TILE if t == Tl.MAX_TILE else int(t) for t in args.tiles]
    out = run(args.model, layer=args.layer, budget=args.budget, tiles=tiles,
              n_seqs=args.seqs, rows=args.rows, speed_rows=args.speed_rows,
              passes=args.passes, phase=args.phase,
              solve_device=args.solve_device,
              solve_dtype=getattr(torch, args.solve_dtype),
              progress=lambda s: print(s, flush=True))
    _verdict(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
