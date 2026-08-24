"""M0 -- where does one compression pass actually go, phase by phase?

The cost model charges four terms: calibration, codebook, rotation, Cholesky.
`m1_gates.run_config` does more than those four, and the extras are charged
NOWHERE:

    prune            saliency, the top-k mask, and `forward_compensate` --
                     a Python loop the length of `n_in`
    compact          survivors gathered into dense per-tile blocks
    block rotation   `blocks @ Q.T`, once per layer rather than per tile
    sub-Hessian      `H[idx, idx]`, k^2 elements gathered PER TILE, which at
    gather           k=7912 is 250 MiB a time

`ROT_TIMINGS` times `q @ H @ q.T` and not the gather in front of it;
`TILE_TIMINGS` times `ldlq_quantize_blocks` and not the prune before it.  This
file measures what is left over.

It is worth being blunt about why.  The model has been wrong six times, and FIVE
of those were terms it did not know about rather than rates it got wrong -- most
recently calibration, which was worth 28 days of M1 and went unnoticed because
nothing had ever run the whole driver.  A seventh omission is likelier than a
seventh mis-estimate, so the useful question is not "is the rate right" but
"what is not on the list".

HOW IT MEASURES.  By wrapping the real functions, not by re-implementing the
sequence: a second copy of `run_config`'s phase order would drift from the first
and then this file would be describing a pipeline nobody runs.  The wrappers
synchronize so the attribution is real, which perturbs the total, so an
UNWRAPPED call is timed alongside and both are reported.  If the two disagree by
much the attribution is not to be trusted and the run says so.

It also counts ROWS through each nearest-codeword path, because the same harness
answers a second question: `_nearest` opens its fast path with
`_LATTICE_MIN_ROWS` (1024 on cuda) and only consults `_ANALYTIC_MIN_ROWS` (384)
INSIDE that gate, so 384 <= rows < 1024 never reaches the analytic form and
scans 65536 codewords instead.  The LDLQ sweep hands it `chunk * lines_per_tile`
rows, which lands in that window for the whole T=1, T=2 and T=4 columns -- where
the tile counts are largest.  Counting the rows says how much of the pass is in
there rather than arguing about it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compact as C                   # noqa: E402
import m0_rotation_value as RV        # noqa: E402
import m1_gates as M                  # noqa: E402
import prune as P                     # noqa: E402
import quantize as Qz                 # noqa: E402
import rotation as R                  # noqa: E402
import scoring as S                   # noqa: E402
import tiling as Tl                   # noqa: E402

DEFAULT_TILES = (1, 4, 16, Tl.MAX_TILE)

#: (module, attribute) pairs timed by wrapping.  Ordered outermost first so the
#: report reads as a nesting: `prune` contains `forward_compensate`, and
#: `ldlq_quantize_blocks` contains the rest.
PHASES = [
    ("prune", P, "prune"),
    ("  damped_hessian_inverse", S, "damped_hessian_inverse"),
    ("  tile_scores", S, "tile_scores"),
    ("  forward_compensate", P, "forward_compensate"),
    ("compact", C, "compact"),
    ("rotate (blocks)", R, "rotate"),
    ("rotate_hessian", R, "rotate_hessian"),
    ("ldlq_quantize_blocks", Qz, "ldlq_quantize_blocks"),
    ("  fit_scale", Qz, "fit_scale"),
    ("  _tile_factors", Qz, "_tile_factors"),
    ("  _ldlq_sweep", Qz, "_ldlq_sweep"),
]

#: Nearest-codeword entry points, counted by rows as well as timed.  These are
#: what say whether the routing gate is costing anything.
SEARCHES = [
    ("_brute_force", Qz, "_brute_force"),
    ("nearest_e8p", Qz, "nearest_e8p"),
    ("nearest_e8p_analytic", Qz, "nearest_e8p_analytic"),
]


class Ledger:
    """Wall time and row counts per wrapped function, self-time not inclusive.

    Nested wrappers double count by construction -- `fit_scale` is inside
    `ldlq_quantize_blocks` -- so the report indents rather than subtracting.
    Subtracting would invent a "self time" the measurement does not support,
    since the same GPU work can be attributed to either frame.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.rows: dict[str, int] = defaultdict(int)
        self._saved: list[tuple] = []

    def _sync(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def wrap(self, label: str, module, name: str, count_rows: bool = False):
        original = getattr(module, name)
        self._saved.append((module, name, original))

        def timed(*args, **kwargs):
            self._sync()
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            self._sync()
            self.seconds[label] += time.perf_counter() - t0
            self.calls[label] += 1
            if count_rows and args and torch.is_tensor(args[0]):
                self.rows[label] += args[0].shape[0]
            return out

        setattr(module, name, timed)

    def restore(self) -> None:
        for module, name, original in reversed(self._saved):
            setattr(module, name, original)
        self._saved.clear()

    def __enter__(self):
        for label, module, name in PHASES:
            self.wrap(label, module, name)
        for label, module, name in SEARCHES:
            self.wrap(label, module, name, count_rows=True)
        return self

    def __exit__(self, *exc):
        self.restore()
        return False


def one_tile_size(problem, tile_size, *, budget: float = 1.5, seed: int = 0,
                  progress=print) -> dict | None:
    """One `run_config`, timed clean, then again with the wrappers on."""
    def call():
        return M.run_config(problem, budget_bits=budget, tile_size=tile_size,
                            seed=seed)

    probe = call()
    if "skipped" in probe:
        progress(f"  T={tile_size}: {probe['skipped']}")
        return None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    clean = call()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    clean_seconds = time.perf_counter() - t0

    with Ledger() as led:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        wrapped = call()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wrapped_seconds = time.perf_counter() - t0

    # The wrappers must not have changed the answer, only the clock.  Reported
    # with its magnitude rather than as a bare failure: a difference at
    # float32's epsilon says the pipeline has a non-deterministic reduction and
    # the guard is too strict, while a large one says the instrumentation is
    # measuring something else -- and those want opposite responses.
    if wrapped["rel_output_error"] != clean["rel_output_error"]:
        c, w = clean["rel_output_error"], wrapped["rel_output_error"]
        raise AssertionError(
            f"wrapping changed the result -- the breakdown is of something "
            f"else.  clean={c!r} wrapped={w!r} relative={abs(w - c) / c:.3e}"
        )

    progress(f"  T={tile_size}: {clean_seconds:.1f}s clean, "
             f"{wrapped_seconds:.1f}s wrapped "
             f"({wrapped_seconds / clean_seconds:.2f}x perturbation)")
    return {
        "tile_size": tile_size,
        "k": clean["survivors_per_tile"],
        "density": clean["density_realized"],
        "rel_output_error": clean["rel_output_error"],
        "clean_seconds": clean_seconds,
        "wrapped_seconds": wrapped_seconds,
        "seconds": dict(led.seconds),
        "calls": dict(led.calls),
        "rows": dict(led.rows),
    }


def run(model_name: str = RV.DEFAULT_MODEL, *, layer: str = RV.DEFAULT_LAYER,
        budget: float = 1.5, tiles=DEFAULT_TILES, n_seqs: int = 16,
        rows: int | None = 256, solve_device: str = "cuda",
        solve_dtype: torch.dtype = torch.float32, progress=print) -> dict:
    progress("building the real layer ...")
    problem = RV.build_problem(model_name, layer=layer, n_seqs=n_seqs,
                               rows=rows, solve_device=solve_device,
                               solve_dtype=solve_dtype, progress=progress)
    progress("")
    measured = [r for t in tiles
                if (r := one_tile_size(problem, t, budget=budget,
                                       progress=progress)) is not None]
    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": ("which phases of a compression pass the cost model "
                         "does not charge, and how many rows reach each "
                         "nearest-codeword path"),
            "model": model_name, "layer": f"layers.0.{layer}",
            "n_out": problem.n_out, "n_in": problem.n_in,
            "output_rows_used": rows, "n_tokens": problem.n_tokens,
            "budget": budget,
            "lattice_min_rows": Qz._LATTICE_MIN_ROWS,
            "analytic_min_rows": Qz._ANALYTIC_MIN_ROWS,
            "charged_by_the_model": ["codebook", "rotation", "cholesky",
                                     "calibration", "eval"],
        },
        "rows": measured,
    }


def _verdict(out: dict) -> None:
    m = out["meta"]
    print("\n" + "=" * 78)
    print(f"  {m['layer']}  {m['n_out']}x{m['n_in']} "
          f"(first {m['output_rows_used']} output rows)  B={m['budget']}")
    print(f"  the model charges: {', '.join(m['charged_by_the_model'])}")

    for r in out["rows"]:
        print(f"\n  T={r['tile_size']}   k={r['k']}   "
              f"{r['clean_seconds']:.2f}s clean "
              f"({r['wrapped_seconds'] / r['clean_seconds']:.2f}x wrapped)")
        print(f"    {'phase':>24} {'seconds':>9} {'% of clean':>11} {'calls':>7}")
        for label, _mod, _name in PHASES:
            s = r["seconds"].get(label)
            if not s:
                continue
            print(f"    {label:>24} {s:8.3f}s {100 * s / r['clean_seconds']:10.1f}%"
                  f" {r['calls'].get(label, 0):7d}")

        print(f"\n    {'search path':>24} {'seconds':>9} {'calls':>7} {'rows':>12}"
              f" {'rows/call':>10}")
        for label, _mod, _name in SEARCHES:
            n = r["calls"].get(label, 0)
            if not n:
                continue
            print(f"    {label:>24} {r['seconds'].get(label, 0):8.3f}s {n:7d}"
                  f" {r['rows'].get(label, 0):12,d}"
                  f" {r['rows'].get(label, 0) / n:10.0f}")

    print("\n" + "-" * 78)
    print("  UNCHARGED: everything the model does not price, as a share of the pass")
    print(f"    {'T':>5} {'prune':>9} {'compact':>9} {'blok rot':>9} "
          f"{'gather+rot':>11} {'toplam':>9}")
    for r in out["rows"]:
        got = r["seconds"]
        clean = r["clean_seconds"]
        prune = got.get("prune", 0.0)
        comp = got.get("compact", 0.0)
        brot = got.get("rotate (blocks)", 0.0)
        # `rotate_hessian` IS charged (ROT_TIMINGS); the gather in front of it
        # is not, and the two are only separable by subtraction here.
        unpriced = prune + comp + brot
        print(f"    {str(r['tile_size']):>5} {100 * prune / clean:8.1f}%"
              f" {100 * comp / clean:8.1f}% {100 * brot / clean:8.1f}%"
              f" {100 * got.get('rotate_hessian', 0.0) / clean:10.1f}%"
              f" {100 * unpriced / clean:8.1f}%")

    print("\n  ROUTING: rows that reached the 65536-codeword scan")
    print(f"    {'T':>5} {'tarama satiri':>14} {'analitik satir':>15} "
          f"{'kafes satir':>12} {'tarama pay':>11}")
    for r in out["rows"]:
        scan = r["rows"].get("_brute_force", 0)
        an = r["rows"].get("nearest_e8p_analytic", 0)
        lat = r["rows"].get("nearest_e8p", 0)
        total = scan + an
        share = f"{100 * scan / total:10.1f}%" if total else f"{'--':>11}"
        print(f"    {str(r['tile_size']):>5} {scan:14,d} {an:15,d} {lat:12,d} {share}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=RV.DEFAULT_MODEL)
    ap.add_argument("--layer", default=RV.DEFAULT_LAYER)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tiles", nargs="*", default=[str(t) for t in DEFAULT_TILES])
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--solve-device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--solve-dtype", default="float32",
                    choices=["float64", "float32"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_pass_breakdown.json"))
    args = ap.parse_args(argv)

    tiles = [Tl.MAX_TILE if t == Tl.MAX_TILE else int(t) for t in args.tiles]
    out = run(args.model, layer=args.layer, budget=args.budget, tiles=tiles,
              n_seqs=args.seqs, rows=args.rows,
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
