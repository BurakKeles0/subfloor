"""M0 -- what does a real run cost, before we commit to one?

`docs/STATUS.md` puts a warning on the tau sweep: the spec estimates it at 25
GPU-hours, and that estimate should be checked against one measured point before
anything is committed.  This checks it -- and the answer is not the one the
question expected, because the sweep is not the first thing that stops working.

Everything here rests on constants measured ON THIS MACHINE, not on peak
figures.  Two kernels dominate and both are benchmarked at the sizes they will
actually be called at:

  cholesky   the per-tile sub-Hessian factorization LDLQ needs, O(k^3) per tile
  codebook   the nearest-codeword search, a [rows, 8] x [8, 65536] product per
             group of eight coordinates

The profile at a small synthetic size says LDLQ is 99.7% of the pipeline and the
codebook search is nearly all of that -- but `k` is small there, and the
factorization grows as `k^3` while the search grows as `k`.  At real widths they
end up the same order, which is why both are modelled rather than one.

The third quantity is memory, and it is the one that bites first: `tile_hessians`
materializes [n_tiles, k, k] in one tensor.

The model is deliberately made of separable pieces so the levers are visible:
device and dtype move the rates, streaming the sub-Hessians moves the memory,
and `T` moves the work itself.
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

import accounting as A                # noqa: E402
import quantize as Qz                 # noqa: E402
import tiling as Tl                   # noqa: E402

#: Llama-2-7B: (n_out, n_in, count per block).  Axis B tiles the output rows and
#: indexes the input channels, so n_in is what sets `k`.
LLAMA2_7B = (
    (4096, 4096, 4),        # q, k, v, o
    (11008, 4096, 2),       # gate, up
    (4096, 11008, 1),       # down
)
N_BLOCKS = 32

#: Cholesky, cholesky_inverse, cholesky again -- roughly k^3/3 + k^3 + k^3/3.
CHOL_FLOPS_PER_K3 = 5.0 / 3.0

#: One nearest-codeword search: a [rows, 8] x [8, 2^16] product.
CODEBOOK_SIZE = 1 << 16
E8P_DIM = Qz.E8P_DIM

#: Measured, `experiments/m0_dense_ppl.py`: one streamed WikiText-2 pass at
#: seqlen 4096 on this 8 GB card.
EVAL_SECONDS = 238.0

TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)


# --------------------------------------------------------------------------- #
# Rates, measured here
# --------------------------------------------------------------------------- #

def _spd(k: int, dtype, device):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(k, k, generator=g, dtype=torch.float64)
    return (a @ a.T / k + torch.eye(k, dtype=torch.float64)).to(dtype).to(device)


def cholesky_rate(k: int, dtype, device, reps: int = 3) -> float:
    """Effective flop/s for the triple LDLQ performs."""
    h = _spd(k, dtype, device)
    torch.linalg.cholesky(h)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        low = torch.linalg.cholesky(h)
        torch.linalg.cholesky(torch.cholesky_inverse(low), upper=True)
    if device == "cuda":
        torch.cuda.synchronize()
    return CHOL_FLOPS_PER_K3 * k ** 3 * reps / (time.perf_counter() - t0)


def codebook_rate(rows: int, dtype, device, min_seconds: float = 0.25) -> float:
    """Effective flop/s for one nearest-codeword search.

    `rows` matters: LDLQ as written calls this once per tile per group, so rows
    is the tile's line count -- sixteen, say, which is far too small to fill a
    GPU.  Batching the tiles together would make it `n_out`, which is why the
    rate is measured at both.
    """
    cb = Qz.e8p_codebook(dtype).to(device)
    x = torch.randn(rows, E8P_DIM, dtype=dtype, device=device)
    Qz._nearest(x, cb)
    if device == "cuda":
        torch.cuda.synchronize()
    # Repeat until the timer has something to measure: at sixteen rows a single
    # call lands under the clock's resolution, and dividing by zero elapsed
    # would report an infinite rate.
    reps, elapsed = 0, 0.0
    t0 = time.perf_counter()
    while elapsed < min_seconds:
        for _ in range(16):
            Qz._nearest(x, cb)
        reps += 16
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    return 2 * rows * E8P_DIM * CODEBOOK_SIZE * reps / elapsed


def measure_rates(*, k: int = 2048, cache: Path | None = None) -> dict:
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    setups = [("cpu_f64", torch.float64, "cpu"), ("cpu_f32", torch.float32, "cpu")]
    if torch.cuda.is_available():
        setups.append(("cuda_f32", torch.float32, "cuda"))

    out = {"k_benchmarked": k, "setups": {}}
    for name, dtype, device in setups:
        out["setups"][name] = {
            "cholesky_flops_per_s": cholesky_rate(k, dtype, device),
            # small = one tile at a time, as the code stands; large = batched
            "codebook_flops_per_s_small": codebook_rate(16, dtype, device),
            "codebook_flops_per_s_large": codebook_rate(4096, dtype, device),
        }
    if torch.cuda.is_available():
        out["device"] = torch.cuda.get_device_name(0)
    out["threads"] = torch.get_num_threads()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

def _scheme(t):
    return {1: "unstructured", Tl.MAX_TILE: "structured"}.get(t, "tile")


def layer_cost(n_out: int, n_in: int, tile_size, budget: float, *,
               vq_bits: float = 2.0, dtype_bytes: int = 8) -> dict | None:
    """Flops and bytes for one linear at one tile size.

    `k` is the aligned survivor count, so it is what the code will really
    factorize -- not the requested density times n_in.
    """
    d = A.density_for_budget(_scheme(tile_size), budget, None, n_in,
                             tile_size=tile_size, vq_bits=vq_bits)
    if d is None or not 0.0 < d <= 1.0:
        return None
    k = Tl.uniform_survivor_count(n_in, d, align=E8P_DIM)
    n_tiles = n_out if tile_size == 1 else (
        1 if tile_size == Tl.MAX_TILE else n_out // tile_size)
    lines_per_tile = n_out // n_tiles

    return {
        "n_out": n_out, "n_in": n_in, "tile_size": tile_size,
        "density": d, "k": k, "n_tiles": n_tiles,
        "lines_per_tile": lines_per_tile,
        # one factorization per tile
        "cholesky_flops": n_tiles * CHOL_FLOPS_PER_K3 * k ** 3,
        # one search per tile per group of eight, over that tile's lines
        "codebook_flops": n_out * (k / E8P_DIM) * 2 * E8P_DIM * CODEBOOK_SIZE,
        # what `tile_hessians` allocates in one go
        "hessian_bytes": n_tiles * k * k * dtype_bytes,
        # what it would allocate if the tiles were streamed instead
        "hessian_bytes_streamed": k * k * dtype_bytes,
    }


def model_cost(tile_size, budget: float, rates: dict, setup: str, *,
               batched: bool = False, inventory=LLAMA2_7B,
               n_blocks: int = N_BLOCKS) -> dict:
    """One full compression pass over Llama-2-7B at one tile size."""
    r = rates["setups"][setup]
    cb_rate = r["codebook_flops_per_s_large" if batched else
               "codebook_flops_per_s_small"]

    chol = cb = peak = peak_streamed = 0.0
    for n_out, n_in, count in inventory:
        c = layer_cost(n_out, n_in, tile_size, budget)
        if c is None:
            return {"tile_size": tile_size, "skipped": "budget unreachable"}
        chol += count * c["cholesky_flops"]
        cb += count * c["codebook_flops"]
        peak = max(peak, c["hessian_bytes"])
        peak_streamed = max(peak_streamed, c["hessian_bytes_streamed"])

    chol *= n_blocks
    cb *= n_blocks
    seconds = chol / r["cholesky_flops_per_s"] + cb / cb_rate
    return {
        "tile_size": tile_size, "setup": setup, "batched": batched,
        "cholesky_flops": chol, "codebook_flops": cb,
        "cholesky_seconds": chol / r["cholesky_flops_per_s"],
        "codebook_seconds": cb / cb_rate,
        "compress_seconds": seconds,
        "eval_seconds": EVAL_SECONDS,
        "point_seconds": seconds + EVAL_SECONDS,
        "peak_hessian_bytes": peak,
        "peak_hessian_bytes_streamed": peak_streamed,
    }


def affordable(rates: dict, setup: str, hours: float, *,
               budgets=(1.75, 1.60, 1.50), tiles=TILES, batched: bool = False,
               n_draws: int = 5) -> dict:
    """The inverse question: given a wall-clock budget, what grid fits?

    More useful than the forward number, because the forward number is not a
    quantity anyone can act on.  Tile sizes are dropped cheapest-first from the
    expensive end, since the cost is dominated by `(n_out/T) * k^3` and the fine
    tiles are where it concentrates -- dropping T=2 buys more than dropping
    everything else combined.

    Note what that costs scientifically: `T=1` is one of Gate B's two edges, so
    it can never be dropped, and dropping the tiles next to it thins exactly the
    region where an interior optimum would have to show itself.
    """
    per_tile = {}
    for t in tiles:
        row = 0.0
        for b in budgets:
            c = model_cost(t, b, rates, setup, batched=batched)
            if "skipped" not in c:
                row += n_draws * c["point_seconds"]
        per_tile[t] = row

    # Never drop the edges: Gate B is defined against them.
    droppable = sorted((t for t in per_tile if t not in (1, Tl.MAX_TILE)),
                       key=lambda t: -per_tile[t])
    keep = list(tiles)
    dropped = []
    while sum(per_tile[t] for t in keep) > hours * 3600 and droppable:
        t = droppable.pop(0)
        keep.remove(t)
        dropped.append(t)

    total = sum(per_tile[t] for t in keep)
    return {
        "setup": setup, "hours_allowed": hours, "n_draws": n_draws,
        "tiles_kept": [str(t) for t in keep],
        "tiles_dropped": [str(t) for t in dropped],
        "seconds": total, "hours": total / 3600,
        "fits": total <= hours * 3600,
        "per_tile_hours": {str(t): v / 3600 for t, v in per_tile.items()},
    }


def sweep_cost(rates: dict, setup: str, *, budget: float = 1.5,
               n_tau_points: int = 25, n_q_points: int = 5, n_q_seeds: int = 3,
               tiles=TILES, batched: bool = False) -> dict:
    """The pre-registered tau sweep, priced.

    Section 5 asks for `Q(d)` at a few densities with three seeds and `tau(T,d)`
    over a grid with one seed.  `Q` is a T=1 curve, so it is priced at T=1; the
    `tau` points spread over the tile grid, so they are priced at the grid's
    mean cost -- costs vary by orders of magnitude across `T`, and quoting the
    cheapest would flatter the estimate.
    """
    per_tile = {}
    for t in tiles:
        c = model_cost(t, budget, rates, setup, batched=batched)
        if "skipped" not in c:
            per_tile[str(t)] = c

    q_point = per_tile["1"]["point_seconds"]
    tau_points = [v["point_seconds"] for k, v in per_tile.items() if k != "1"]
    tau_mean = sum(tau_points) / len(tau_points)

    q_total = n_q_points * n_q_seeds * q_point
    tau_total = n_tau_points * tau_mean
    return {
        "setup": setup, "batched": batched, "budget": budget,
        "per_tile": per_tile,
        "q_seconds": q_total, "tau_seconds": tau_total,
        "total_seconds": q_total + tau_total,
        "total_hours": (q_total + tau_total) / 3600,
        "spec_estimate_hours": 25.0,
        "over_spec": (q_total + tau_total) / 3600 / 25.0,
    }


def m1_cost(rates: dict, setup: str, *, budgets=(1.75, 1.60, 1.50),
            n_draws: int = 5, tiles=TILES, batched: bool = False) -> dict:
    """M1's own grid, for scale: budgets x tiles x draws."""
    total = 0.0
    for b in budgets:
        for t in tiles:
            c = model_cost(t, b, rates, setup, batched=batched)
            if "skipped" not in c:
                total += n_draws * c["point_seconds"]
    return {"setup": setup, "batched": batched, "n_draws": n_draws,
            "seconds": total, "hours": total / 3600, "days": total / 86400}


# --------------------------------------------------------------------------- #

def run(*, budget: float = 1.5, cache: Path | None = None) -> dict:
    rates = measure_rates(cache=cache)
    setups = list(rates["setups"])
    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": "what does one compression pass over Llama-2-7B cost",
            "budget": budget,
            "eval_seconds_measured": EVAL_SECONDS,
        },
        "rates": rates,
        "per_tile": {s: {str(t): model_cost(t, budget, rates, s)
                         for t in TILES} for s in setups},
        "sweep": {f"{s}{'_batched' if b else ''}": sweep_cost(
            rates, s, budget=budget, batched=b)
            for s in setups for b in (False, True)},
        "m1": {f"{s}{'_batched' if b else ''}": m1_cost(rates, s, batched=b)
               for s in setups for b in (False, True)},
        "affordable": {f"{h}h": affordable(rates, setups[-1], h, batched=True)
                       for h in (24, 72, 168)},
    }


def _hms(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400 * 2:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _verdict(out: dict) -> None:
    rates = out["rates"]
    print("\n" + "=" * 78)
    print(f"  measured on this machine "
          f"({rates.get('device', 'cpu only')}, {rates['threads']} threads), "
          f"k={rates['k_benchmarked']}")
    print(f"    {'setup':10s} {'cholesky':>12} {'codebook 16':>13}"
          f" {'codebook 4096':>15}")
    for name, r in rates["setups"].items():
        print(f"    {name:10s} {r['cholesky_flops_per_s'] / 1e9:9.0f} Gf/s"
              f" {r['codebook_flops_per_s_small'] / 1e9:10.1f} Gf/s"
              f" {r['codebook_flops_per_s_large'] / 1e9:12.1f} Gf/s")

    slowest = list(rates["setups"])[0]
    print(f"\n  one compression pass over Llama-2-7B at B={out['meta']['budget']},"
          f" {slowest} as the code stands:")
    print(f"    {'T':>5} {'d':>7} {'k':>6} {'tiles':>7} {'chol':>8}"
          f" {'codebook':>9} {'total':>8} {'H memory':>11}")
    for t, c in out["per_tile"][slowest].items():
        if "skipped" in c:
            print(f"    {t:>5}  (budget unreachable)")
            continue
        lc = layer_cost(4096, 11008, c["tile_size"], out["meta"]["budget"])
        print(f"    {t:>5} {lc['density']:7.4f} {lc['k']:6d} {lc['n_tiles']:7d}"
              f" {_hms(c['cholesky_seconds']):>8}"
              f" {_hms(c['codebook_seconds']):>9}"
              f" {_hms(c['point_seconds']):>8}"
              f" {c['peak_hessian_bytes'] / 2**30:9.1f} GiB")

    worst = max((c for c in out["per_tile"][slowest].values()
                 if "skipped" not in c), key=lambda c: c["peak_hessian_bytes"])
    print(f"\n  MEMORY IS THE FIRST WALL: `tile_hessians` builds [n_tiles, k, k]"
          f" in one tensor.")
    print(f"    worst case {worst['peak_hessian_bytes'] / 2**30:,.0f} GiB at "
          f"T={worst['tile_size']}; streamed per tile it would be "
          f"{worst['peak_hessian_bytes_streamed'] / 2**20:,.0f} MiB.")

    print(f"\n  the tau sweep (25 tau points + 5 Q densities x 3 seeds):")
    print(f"    {'configuration':22s} {'total':>10} {'vs spec 25h':>12}")
    for name, s in out["sweep"].items():
        print(f"    {name:22s} {_hms(s['total_seconds']):>10}"
              f" {s['over_spec']:11.1f}x")

    print(f"\n  M1 itself (3 budgets x 7 tiles x 5 draws):")
    for name, m in out["m1"].items():
        print(f"    {name:22s} {_hms(m['seconds']):>10}")

    aff = out.get("affordable")
    if aff:
        setup = next(iter(aff.values()))["setup"]
        print(f"\n  what fits in a wall-clock budget ({setup}, batched),")
        print("  dropping the most expensive interior tiles first:")
        for label, v in aff.items():
            kept = ", ".join(v["tiles_kept"])
            print(f"    {label:>5}: {'fits' if v['fits'] else 'DOES NOT FIT'}"
                  f" at {v['hours']:.0f}h   keep {{{kept}}}")
            if v["tiles_dropped"]:
                print(f"           dropped {', '.join(v['tiles_dropped'])}")
        print("    per-tile cost of the 3 x 5 grid, hours:")
        row = next(iter(aff.values()))["per_tile_hours"]
        print("      " + "  ".join(f"T={k}:{v:.0f}" for k, v in row.items()))
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--cache", type=Path, default=Path("results/m0_rates.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_cost_model.json"))
    args = ap.parse_args(argv)

    out = run(budget=args.budget, cache=args.cache)
    _verdict(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
