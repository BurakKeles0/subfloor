"""Re-measure `m0_cost_model.TILE_TIMINGS`, and record how.

WHY THIS IS A SCRIPT NOW.  The three rows it replaces were measured by hand, and
what they never recorded was the TILE COUNT.  That is not a footnote: `auto_chunk`
turns `n_tiles` into a chunk and the chunk into a row count, and the row count
decides which search path `_nearest` takes.  So a tile time is only meaningful
next to the tile count it was taken at, and for months it was not.

The gap surfaced on 2026-08-25, when two constants moved and the honest answer to
"how much faster is a tile now?" turned out to be unavailable: the old rows could
not be scaled, because nobody knew what regime they were in.  Re-measuring by
hand would have reproduced exactly that problem one version later.

WHAT IS MEASURED.  The grid's own cells, derived rather than chosen:
`accounting.density_for_budget` gives the survivor count `k` at each tile size,
the tile size gives the line count, and `n_out / T` gives the tile count.  One
layer shape per line count, which is what the model's per-weight residual wants
-- it divides by `lines * k`, so `k` is normalized out and `lines` is the axis.

TWO SAMPLES CHANGED SHAPE, DELIBERATELY:

  * `(3072, 128)` is retired.  No cell of the grid has 128 lines; it was a probe
    standing in for the coarse end, and it stood in badly.  At T=max the real
    line count is `n_out` -- 4096 -- with ONE tile, and 128 lines with a
    synthetic tile count puts the sweep in a different regime than the cell it
    was meant to represent.
  * T=1, 2, 8 and 32 are added.  The model picks the nearest measured line count
    in log space, and with samples only at 4, 16 and 128 the whole fine end
    snapped to a single point.

`cpu_f64` is NOT re-measured.  The pipeline runs cuda/float32 (decision
2026-08-23) and the CPU row exists for comparison; its recorded arrangement is
explicitly one tile at a time, so its tile count is 1 by construction and that
much of the provenance was never actually missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import accounting as A                                        # noqa: E402
import bench_guard as BG                                      # noqa: E402
import quantize as Qz                                         # noqa: E402
import tiling as Tl                                           # noqa: E402

#: The layer the samples are taken from.  `q/k/v/o` at 4096x4096 -- the most
#: numerous shape in the model (four of every seven linears) and the one whose
#: `k` the recorded rows already used.
LAYER = (4096, 4096)

#: Budget the shapes are derived at.  B=1.5 is the grid's densest point and the
#: one every other measurement in the project quotes.
BUDGET = 1.5

TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)
HESSIAN_BLOCK = 512


def grid_shapes(n_out: int = LAYER[0], n_in: int = LAYER[1],
                budget: float = BUDGET, tiles=TILES) -> list[tuple]:
    """(tile_size, k, lines, n_tiles) for each cell, derived from accounting."""
    out = []
    for t in tiles:
        scheme = {1: "unstructured", Tl.MAX_TILE: "structured"}.get(t, "tile")
        d = A.density_for_budget(scheme, budget, None, n_in, tile_size=t,
                                 vq_bits=Qz.E8P_BITS_PER_WEIGHT)
        if d is None or not 0.0 < d <= 1.0:
            continue
        lines = n_out if t == Tl.MAX_TILE else t
        n_tiles = 1 if t == Tl.MAX_TILE else n_out // t
        # The mask is aligned to the quantizer group, so this is what the code
        # will really factorize -- not the requested density times n_in.
        k = int(round(d * n_in / Qz.E8P_DIM)) * Qz.E8P_DIM
        out.append((t, k, lines, n_tiles))
    return out


def measure(k: int, lines: int, n_tiles: int, *, device: str, dtype,
            reps: int, seed: int = 0) -> dict:
    """Seconds per tile, with the per-tile scale fit and without it.

    Both arms in ONE alternating run rather than two, because the RATIO between
    them is what `SCALE_FIT_MULTIPLIER` carries, and absolute times on this
    machine move 14-37% between runs -- a ratio taken across two runs would be
    measuring the machine.

    The no-fit arm gets the scale `fit_scale` would have found, not an arbitrary
    one.  That matters more than it looks: a badly scaled input misses far more
    often in the lattice decoder, so a fixed `scale=1.0` would quietly change how
    much search work the sweep does, and the ratio would be pricing two
    different amounts of work rather than the fit.
    """
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randn((n_tiles, lines, k), generator=g).to(device=device,
                                                              dtype=dtype)
    a = torch.randn((k, k), generator=g)
    H = ((a @ a.T) / k + torch.eye(k)).to(device=device, dtype=dtype)

    # Assert the fast path is open.  A benchmark that quietly measures the scan
    # is the failure `is_canonical_codebook` was exported for, and it has cost
    # this project four measurements.
    cb = Qz._on_device(blocks.dtype, str(blocks.device))
    assert Qz.is_canonical_codebook(cb)

    chunk = Qz.auto_chunk(n_tiles, lines, k, blocks.element_size(),
                          HESSIAN_BLOCK)
    alpha = Qz.fit_scale(blocks[0].reshape(-1, Qz.E8P_DIM), cb)

    def arm(scale):
        def f():
            Qz.ldlq_quantize_blocks(blocks, lambda t: H, scale=scale,
                                    hessian_block=HESSIAN_BLOCK, chunk=chunk)
        return f

    timed = BG.alternating({"fit": arm("per_tile"), "no_fit": arm(alpha)},
                           reps=reps, warmup=2)
    del blocks, H
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "k": k, "lines": lines, "n_tiles": n_tiles, "chunk": chunk,
        "rows_into_nearest": chunk * lines,
        "seconds_per_tile": timed["fit"]["median"] / n_tiles,
        "seconds_per_tile_no_fit": timed["no_fit"]["median"] / n_tiles,
        "tile_ratio": timed["fit"]["median"] / timed["no_fit"]["median"],
        "seconds_total": timed["fit"]["median"],
        "spread": timed["fit"]["spread"],
        "spread_no_fit": timed["no_fit"]["spread"],
    }


def run(device: str = "cuda", dtype=torch.float32, reps: int = 3,
        strict: bool = True) -> list[dict]:
    if device == "cuda":
        print(f"  GPU: {BG.require_quiet_gpu(strict=strict)}\n")

    rows = []
    print("%-6s %6s %6s %8s %7s %7s %12s %12s %7s %10s"
          % ("T", "k", "lines", "n_tiles", "chunk", "satir", "s/tile",
             "fitsiz", "oran", "yayilim"))
    for t, k, lines, n_tiles in grid_shapes():
        r = measure(k, lines, n_tiles, device=device, dtype=dtype, reps=reps)
        r["tile_size"] = t
        rows.append(r)
        print("%-6s %6d %6d %8d %7d %7d %12.6f %12.6f %7.3f %6.0f%%/%.0f%%"
              % (t, k, lines, n_tiles, r["chunk"], r["rows_into_nearest"],
                 r["seconds_per_tile"], r["seconds_per_tile_no_fit"],
                 r["tile_ratio"], r["spread"] * 100, r["spread_no_fit"] * 100))
    return rows


def as_literal(rows: list[dict], setup: str = "cuda_f32") -> str:
    """The `TILE_TIMINGS` entry, ready to paste, provenance included."""
    body = ",\n".join(
        f"        ({r['k']}, {r['lines']}, {r['n_tiles']}, "
        f"{r['seconds_per_tile']:.6g}, {r['seconds_per_tile_no_fit']:.6g})"
        for r in rows)
    return f'    "{setup}": (\n{body},\n    ),'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    rows = run(device=args.device, reps=args.reps, strict=not args.allow_busy)
    print("\n" + as_literal(rows))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
