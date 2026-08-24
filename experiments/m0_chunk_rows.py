"""What `auto_chunk`'s row target costs, now that the search below it changed.

THE FINDING THIS EXISTS TO PRICE.  Three constants were tuned independently,
years of measurement apart, and nothing ever wrote down that they have to
satisfy one inequality:

    CHUNK_TARGET_ROWS * DECODER_MISS_FRACTION  >  _ANALYTIC_MIN_ROWS

`auto_chunk` aims the sweep at the first, the decoder hands on the middle
fraction, and the leftover takes the analytic path only if it clears the last.
At 1024, 0.349 and 384 that reads 357 > 384 -- false -- so EVERY group of the
sweep fell through to a 65,536-codeword scan.

It is not a corner of the grid, it is the grid's middle.  `auto_chunk`'s
saturation ceiling is `ceil(target / lines)`, so wherever `lines` divides 1024
the chunk lands on exactly 1024 rows, and eight of the twenty-one layer-by-tile
cells at B=1.5 do:

    T=1, 2, 4 and down_proj at T=8    199-816 rows   analytic direct, clean
    T=8, 16, 32  (eight cells)             1024      decoder + FULL SCAN
    T=max                            4096-11008      decoder + analytic, clean

Both constants had to move, and neither alone is enough.  Raising the row target
fixes the cells it can reach; it cannot reach `down_proj`, where `k=7912` caps
the chunk at 67 tiles on MEMORY and the rows stop at 1072.  Lowering the
threshold fixes that one and is worth little anywhere else.

The file reports two independent quantities so a timing is never believed alone:

  * rows that reached `_brute_force` -- a COUNT, which contention cannot move
  * wall time, alternating A/B, with the spread printed

and the timing is gated on `bench_guard.require_quiet_gpu`, because the rule
that used to protect it was a habit and the habit read a counter that reads 42%
on an idle card here (`docs/STATUS.md` section 14.2).

A NOTE ON WHAT THE MICROBENCHMARK PROMISED.  Analytic against scan on the
353-row leftover measures 2.03x in isolation.  Removing those scans from the
sweep is worth 1.04x.  The scan overlaps with the triangular solve and the
feedback matmul that follow it, so most of what it costs in isolation is hidden
in place -- section 6.3's rule, once more: composing a cost out of kernel
microbenchmarks does not work here, and it does not become safe just because the
kernel in question is one you are deleting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bench_guard as BG                                      # noqa: E402
import quantize as Qz                                         # noqa: E402

#: (label, k, lines_per_tile, n_tiles) -- the REAL grid cells at B=1.5, with
#: `k` and `n_tiles` derived from `accounting.density_for_budget` rather than
#: chosen.  The tile counts are the real ones: an artificially small `n_tiles`
#: makes `min(n_tiles, ...)` the binding ceiling in `auto_chunk` and quietly
#: measures a chunk the grid would never use, which cost one run of this file.
SHAPES = (
    ("T=8   k=2816 (q/k/v/o)", 2816, 8, 512),
    ("T=16  k=2944 (q/k/v/o)", 2944, 16, 256),
    ("T=32  k=3008 (q/k/v/o)", 3008, 32, 128),
    ("T=16  k=7912 (down)", 7912, 16, 256),
)

#: The constant PAIRS to compare, as (label, _ANALYTIC_MIN_ROWS,
#: CHUNK_TARGET_ROWS).  They have to move together and be measured together:
#: the row target rescues the cells it can reach, the threshold rescues the one
#: it cannot (`down_proj`, held at 67 tiles by memory), and each alone leaves
#: half the grid scanning.
ARMS = (
    ("eski 384/1024", 384, 1024),
    ("yeni 320/2048", 320, 2048),
)

HESSIAN_BLOCK = 512


def _problem(k: int, lines: int, n_tiles: int, device: str, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randn(n_tiles, lines, k, generator=g).to(device)
    a = torch.randn(k, k, generator=g)
    H = ((a @ a.T) / k + torch.eye(k)).to(device)
    return blocks, H


def _chunk_for(target_rows: int, lines: int, k: int, itemsize: int,
               n_tiles: int) -> tuple[int, str]:
    """The chunk `auto_chunk` would pick for this row target, and what bound it.

    Reported because the memory ceiling can bind first, in which case the row
    target does nothing at all -- which is itself an answer and one this file
    would otherwise hide.
    """
    parts = Qz._partition(k, HESSIAN_BLOCK, Qz.E8P_DIM)
    per_tile = sum(w * w for _, w in parts) * itemsize
    by_memory = max(1, Qz.CHUNK_BUDGET_BYTES // max(per_tile, 1))
    by_saturation = max(1, -(-target_rows // max(lines, 1)))
    chunk = int(min(n_tiles, by_memory, by_saturation))
    which = ("memory" if by_memory <= min(by_saturation, n_tiles) else
             "tiles" if n_tiles <= by_saturation else "rows")
    return chunk, which


class _constants:
    """Hold `_ANALYTIC_MIN_ROWS` and `CHUNK_TARGET_ROWS` at a given pair.

    A context manager rather than two assignments because every arm has to put
    them back: `alternating` interleaves the arms in one process, so an arm that
    leaked its constants would poison the next one's repetition and the drift
    would look like noise rather than a bug.
    """

    def __init__(self, analytic_min_rows: int, chunk_target_rows: int) -> None:
        self.new = (analytic_min_rows, chunk_target_rows)

    def __enter__(self):
        self.old = (Qz._ANALYTIC_MIN_ROWS, Qz.CHUNK_TARGET_ROWS)
        Qz._ANALYTIC_MIN_ROWS, Qz.CHUNK_TARGET_ROWS = self.new
        return self

    def __exit__(self, *exc):
        Qz._ANALYTIC_MIN_ROWS, Qz.CHUNK_TARGET_ROWS = self.old
        return False


def _arm(blocks, H, chunk: int, analytic_min_rows: int, chunk_target_rows: int):
    def run():
        with _constants(analytic_min_rows, chunk_target_rows):
            Qz.ldlq_quantize_blocks(blocks, lambda t: H,
                                    hessian_block=HESSIAN_BLOCK, chunk=chunk)
    return run


def count_paths(blocks, H, chunk: int) -> dict:
    """Rows reaching each search path.  A COUNT -- contention cannot move it."""
    seen = {"brute_calls": 0, "brute_rows": 0, "analytic_rows": 0}
    real_bf, real_an = Qz._brute_force, Qz.nearest_e8p_analytic

    def bf(x, *a, **k):
        seen["brute_calls"] += 1
        seen["brute_rows"] += x.shape[0]
        return real_bf(x, *a, **k)

    def an(x, *a, **k):
        seen["analytic_rows"] += x.shape[0]
        return real_an(x, *a, **k)

    Qz._brute_force, Qz.nearest_e8p_analytic = bf, an
    try:
        Qz.ldlq_quantize_blocks(blocks, lambda t: H,
                                hessian_block=HESSIAN_BLOCK, chunk=chunk)
    finally:
        Qz._brute_force, Qz.nearest_e8p_analytic = real_bf, real_an
    return seen


def run(device: str = "cuda", reps: int = 5, strict: bool = True,
        shapes=SHAPES) -> list[dict]:
    if device == "cuda":
        state = BG.require_quiet_gpu(strict=strict)
        print(f"  GPU: {state}\n")

    out = []
    for label, k, lines, n_tiles in shapes:
        blocks, H = _problem(k, lines, n_tiles, device)
        # The fast path has to be OPEN or this times the scan and reports no
        # difference -- the failure `is_canonical_codebook` was exported for.
        assert Qz.is_canonical_codebook(
            Qz._on_device(blocks.dtype, str(blocks.device)))

        row = {"shape": label, "k": k, "lines": lines, "arms": {}}
        arms = {}
        for name, amr, ctr in ARMS:
            with _constants(amr, ctr):
                chunk, bound = _chunk_for(ctr, lines, k,
                                          blocks.element_size(), n_tiles)
                paths = count_paths(blocks, H, chunk)
            row["arms"][name] = {
                "chunk": chunk, "rows": chunk * lines, "bound_by": bound,
                **paths,
            }
            arms[name] = _arm(blocks, H, chunk, amr, ctr)

        if device == "cuda":
            timed = BG.alternating(arms, reps=reps)
            for name, _, _ in ARMS:
                row["arms"][name]["seconds"] = timed[name]["median"]
                row["arms"][name]["spread"] = timed[name]["spread"]

        base = row["arms"][ARMS[0][0]]
        print(f"{label}")
        for name, _, _ in ARMS:
            a = row["arms"][name]
            line = (f"  {name}  chunk {a['chunk']:4d} "
                    f"({a['rows']:5d} satir, {a['bound_by']}-bound)  "
                    f"tarama {a['brute_calls']:4d} cagri / {a['brute_rows']:7d} satir")
            if "seconds" in a:
                speed = base["seconds"] / a["seconds"]
                line += (f"  {a['seconds'] * 1000:8.1f} ms  {speed:4.2f}x"
                         f"  (yayilim %{a['spread'] * 100:.0f})")
            print(line)
        print()
        out.append(row)
        del blocks, H
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--allow-busy", action="store_true",
                    help="record the numbers with the contention caveat "
                         "attached rather than refusing to measure")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    rows = run(device=args.device, reps=args.reps, strict=not args.allow_busy)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
