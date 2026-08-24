"""M0 -- what does a real run cost, before we commit to one?

`docs/STATUS.md` puts a warning on the tau sweep: the spec estimates it at 25
GPU-hours, and that estimate should be checked against one measured point before
anything is committed.  This checks it -- and the answer is not the one the
question expected, because the sweep is not the first thing that stops working.

Everything here rests on constants measured ON THIS MACHINE, not on peak
figures.  Five terms are charged, each from its own measured curve:

  calibration  two passes over every block, per point: gather the Hessians,
               then re-run so the next block sees the compressed output
  compensate   `prune.forward_compensate`, a sweep the length of `n_in`
  codebook     the nearest-codeword search and the scale sweep in front of it
  rotation     `q @ H_t @ q.T`, once per tile
  cholesky     the per-tile sub-Hessian factorization, O(k^3) per tile

The last three were the whole model until 2026-08-24.  The first two were added
that day and were worth 28 days and 1.6 days of M1 -- both found by asking what
was not on the list, neither by anything failing.

That order is historical, not current.  The factorization LOOKS like it should
dominate -- it is the only cubic term -- and for a while the model said it did.
It never did.  But as of 2026-08-24 the codebook does not either: at B=1.5, T=4
the pass is 1.92 hours of ROTATION against 1.34 of codebook and 0.21 of
Cholesky.  The lead has changed hands twice; read the numbers, not the order.

The model describes the pipeline as it now runs: feedback confined to width-512
blocks, the sweep chunked across tiles, the scale fit's 24 candidates evaluated
in one search call -- all three bit-identical to what they replace -- and the
calibration Hessians accumulated on the block's own device rather than copied to
the CPU, which is 25x on a term that had never been charged.  M1 is 14.9 days;
without the calibration fix the same grid would be 41.

What is left is no longer `fit_scale`.  It was 83% of a tile and is 28%, so the
saving from dropping the per-tile fit fell from several-fold to 1.4 days, and
with it the cost case for `per_layer` and for sampling -- both already rejected
on quality.  The largest term is now `q @ H_t @ q.T`, which is computed as a
dense GEMM against a matrix that is a Kronecker product of a Hadamard and a
small orthogonal factor (`rotation.structured_orthogonal`).  That is a
structural cost, not a launch-bound one, and it has been measured
(`rotation.rotate_hessian`, 5.52x grid-weighted) but is not on by default.

The corrections, in order, because each one is a way this file was wrong:

  1. it omitted `fit_scale` entirely -- 6x too low.  `ldlq_quantize` sweeps 24
     candidate scales over the whole tile before it quantizes anything.
  2. it priced the codebook search at a rate measured for SIXTEEN rows in a
     tight loop, where the codebook stays in cache.  Real calls interleave with
     Hessian updates that evict it.  Fixed by measuring END-TO-END tile times.
  3. it charged one per-weight constant for every tile size, when the constant
     falls threefold with the line count -- overstating the coarse end, which is
     the end the granularity question is about.
  4. it charged every width the Cholesky rate measured at k=2048, from a
     benchmark that warmed `cholesky` but not `cholesky_inverse`.  9.4x too HIGH
     at real widths, and this one mattered most: it is the number that says
     whether M1 can be run, and it read 120 days when the answer was 94.
  5. it subtracted a FULL-WIDTH Cholesky out of tile timings that were measured
     with `hessian_block=512`, taking out time the tile never spent and so
     undercharging the codebook -- 34% at (2560,4), 24% at (2944,16), 9% at
     (3072,128).  Optimistic, and worst at the fine granularities where the
     grid's cost lives.  Fixed by recording each row's measurement width
     (`TILE_TIMING_BLOCK`), because the cpu and cuda rows were taken under
     different arrangements and no single assumption is right for both.

  6. it charged NO CALIBRATION AT ALL.  `calibrate.sequential_calibrate` walks
     every block twice per point -- once with hooks to accumulate the Hessians,
     once more so the next block sees the compressed output -- and neither pass
     appeared here.  At the configuration the code shipped with, accumulators
     pinned to the CPU, that was 5.59 hours per point, more than the entire
     compression pass at every tile size.  M1 read 12 days when the answer was
     40.  See `CALIBRATION_TIMINGS`, and note what let it hide for six versions:
     nothing had ever run the full driver, because `experiments/m1_run.py` does
     not exist.

  7. it charged NO COMPENSATION either.  `run_config` calls `prune` before
     anything on the list, and `TILE_TIMINGS` starts at
     `ldlq_quantize_blocks`, so `forward_compensate` -- a Python loop the
     length of `n_in` whose every iteration touches the whole remaining width
     -- was priced nowhere.  40.7 s per block, 0.362 h per point, 1.58 days of
     M1.  See `COMPENSATE_TIMINGS`, and note that this one was found by looking
     for what was missing rather than by anything failing.

  8. it had NO SAMPLE BELOW FOUR LINES, so T=1 and T=2 borrowed the four-line
     per-weight rate.  They are nothing like it: the constant falls 41x from one
     line to 4096, and 8x over the first step alone, because `fit_scale`'s fixed
     cost is amortized over 128 vectors in a one-line tile against 1280 at T=4.
     The correction does not move M1 much -- the fine end goes up and the coarse
     end comes down, 15.0 to 14.91 days -- but it moves the SHAPE: this file
     used to report the grid's cost peaking in the middle, at T=4, and it peaks
     at the fine end.  Found by fixing provenance rather than by looking for it,
     which is the third time that has happened here.

And one thing that was not a modelling error but a measurement that did not
hold: two of the three `cuda_f32` tile timings recorded on 2026-08-24 do not
reproduce, both optimistically (1.28x and 1.65x).  They have since been replaced
outright, along with the tile counts they never recorded (`TILE_TIMINGS`).

The through-line is that composing a cost from kernel microbenchmarks does not
work here.  Where a curve is measured, it is measured at the sizes the code will
call it at, and the residual is fitted against a pipeline timing rather than
assumed.

The remaining quantity is memory, and it is the one that bites first:
`tile_hessians` materializes [n_tiles, k, k] in one tensor.

The model is deliberately made of separable pieces so the levers are visible:
device and dtype move the rates, streaming the sub-Hessians moves the memory,
and `T` moves the work itself.
"""

from __future__ import annotations

import argparse
import json
import math
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

#: The width the pipeline confines its LDLQ feedback to
#: (`m1_gates.HESSIAN_BLOCK`).  This model's default has to describe what the
#: code DOES, not what it once did -- an optimistic default is how a cost model
#: stops being one, and so is a pessimistic one.
DEFAULT_HESSIAN_BLOCK = 512

#: Cholesky, cholesky_inverse, cholesky again -- roughly k^3/3 + k^3 + k^3/3.
CHOL_FLOPS_PER_K3 = 5.0 / 3.0

#: One nearest-codeword search: a [rows, 8] x [8, 2^16] product.
CODEBOOK_SIZE = 1 << 16
E8P_DIM = Qz.E8P_DIM

#: `ldlq_quantize` does not search the codebook once per group -- it first calls
#: `fit_scale`, which sweeps 24 candidate scales over the whole tile.
#:
#: THIS CONSTANT WAS 6.0 AND IS NOW 1.39, and the change is the point.  The
#: sweep used to be 83% of an o_proj-shaped tile (4 lines x 2560 survivors)
#: because it paid a launch-bound search's fixed cost once per candidate.  Since
#: the candidates are evaluated together (`quantize.fit_scale`, 2026-08-24) it
#: is 28% at four lines, 17% at sixteen and 12.5% at 128 -- measured against the
#: same code with a fixed `scale`, which is exactly what `scale_fit=False`
#: prices.  1.39 is the four-line figure, the most favourable of the three, kept
#: on the old convention of overstating the lever being argued about so that
#: rejecting it stays robust.
#:
#: What that costs the argument: dropping the per-tile fit used to be the
#: largest single saving available, worth several-fold.  It is now worth at most
#: 28% of a tile, so `scale="per_layer"` -- already rejected on quality
#: (11% worse, 2026-08-23) -- has lost its cost case as well.  So has sampling
#: the fit, which `docs/STATUS.md` section 5.8 rejected on variance.
#:
#: READ `scale_fit=False` AS A CEILING, NOT AS A PLAN.  It prices removing the
#: per-tile fit altogether -- a fixed scale, or `scale="per_layer"`.  It does
#: NOT price `fit_scale(sample=N)`, which is the option `docs/STATUS.md` names,
#: and the two are not close: the sweep only looks at the vectors a tile HAS, so
#: capping it at N does nothing to a tile with fewer than N.  At B=1.5 a tile
#: holds 128 vectors at T=1, 1,280 at T=4 and 5,888 at T=16, against a default
#: cap of 8,192 -- so per-tile sampling at that cap is a no-op at exactly the
#: tile sizes where the cost lives, and only bites at T=max, already the
#: cheapest column.  `scale_sample_bites` is the check; the caps that would
#: bite are small enough that their quality cost has to be measured rather than
#: assumed.
SCALE_FIT_MULTIPLIER = 1.39

#: END-TO-END per-tile wall times measured on this machine: (k, lines, seconds).
#: One `ldlq_quantize` call each, at o_proj-shaped widths.
#:
#: A microbenchmark measures a kernel; this measures the pipeline.  Corrections
#: 1-3 in the module docstring are all here.  Note what these DO NOT include:
#: `ldlq_quantize` alone, so the sub-Hessian rotation is charged separately
#: (`ROT_TIMINGS`) and is not double counted in the fitted constant.
#: Re-measured 2026-08-23 after `quantize.nearest_e8p` replaced the brute-force
#: scan where it pays.  The scan numbers it supersedes, for the record:
#:   cpu_f64  (2560,4,4.49) (2944,16,29.26) (3072,128,266.04)
#:   cuda_f32 (2560,4,0.28) (2944,16, 0.83) (3072,128,  5.93)
#: `cuda_f32` re-measured 2026-08-24 with `torch.compile` active on both
#: elementwise kernels (`quantize._analytic_shift`, `quantize._lattice_shift`),
#: which `triton-windows` makes possible on this machine: 1.64x / 1.72x / 1.87x,
#: output bit-identical.  THESE NUMBERS ASSUME A WORKING TRITON.  Without one
#: the code still runs and still gives the same answer, but roughly 1.7x slower,
#: and the model would then be optimistic by that factor.
#:   superseded eager: (2560,4,0.0887) (2944,16,0.1201) (3072,128,0.3883)
#:
#: Earlier note:
#: `cuda_f32` re-measured again 2026-08-23 after `quantize.nearest_e8p_analytic`
#: replaced the SCAN that unsettled rows used to fall back to: 1.35x / 2.65x /
#: 5.62x per tile on top of what the chunked sweep already gave.  The gain grows
#: with the line count, the opposite of the chunking's, because the two fix
#: different things -- chunking made small tiles fill the card, and this makes
#: `fit_scale` stop scanning 65536 codewords per vector per candidate scale.
#:   superseded chunked+scan: (2560,4,0.1194) (2944,16,0.3177) (3072,128,2.1827)
#:
#: Earlier note, still true of `cpu_f64`:
#: `cuda_f32` re-measured 2026-08-23 with the chunked sweep and
#: `hessian_block=512`, which is what the pipeline now does.  Per tile it is
#: 2.07x / 1.43x / 1.06x faster than the one-tile-at-a-time arrangement it
#: replaces -- far less than the 5-12x the SWEEP gained, because the sweep is no
#: longer what a tile spends its time on.  `fit_scale` is, and it is not
#: chunked: it fits one scalar per tile by scanning that tile's vectors 24
#: times.  The gain also shrinks with the line count (2.07x at four lines,
#: 1.06x at 128) for the same reason the chunking helped in the first place --
#: a 128-line tile already filled the card.
#:
#: `cpu_f64` still describes the ONE-TILE arrangement; it has not been
#: re-measured because the pipeline runs on cuda/float32 (decision 2026-08-23)
#: and the CPU row exists only for comparison.  Do not read the two rows as a
#: like-for-like device comparison any more.
#:   superseded cuda_f32, chunk=1: (2560,4,0.247) (2944,16,0.454) (3072,128,2.309)
#: `cuda_f32` re-measured 2026-08-24 after `fit_scale` began evaluating its 24
#: candidate scales in one search call instead of one apiece.  Measured against
#: the same code with `FIT_ROW_BUDGET = 1`, which reproduces the old
#: one-candidate-per-pass arrangement exactly: 3.78x / 2.01x / 1.09x, output
#: bit-identical.  The gain is largest at the fine granularities, where a tile
#: holds too few vectors to pay for 24 separate launches.
#:   superseded batched-fit-off, this machine, today: (2560,4,0.0535)
#:   (2944,16,0.0810) (3072,128,0.3058)
#:
#: TWO OF THE THREE PREVIOUS `cuda_f32` VALUES DID NOT REPRODUCE, both
#: optimistically: the table said 0.0631 at (2944,16) where the same
#: configuration measures 0.0810 today (1.28x), and 0.1851 at (3072,128)
#: against 0.3058 (1.65x).  This is not a difference in setup -- (2560,4)
#: reproduces to 1.00x, and the superseded EAGER row reproduces to 0.2%
#: (0.3883 against 0.3874 measured with `TILESPARSE_NO_COMPILE=1`).  What did
#: not hold up is the Triton gain claimed for the two coarse widths: 1.72x and
#: 1.87x were recorded, 1.18x and 1.09x measure today.  Treat the pre-08-24
#: coarse rows as withdrawn rather than superseded.
#:
#: The mechanism is worth carrying, because it says these levers do not
#: multiply: Triton's gain WAS the launch overhead, and batching the candidates
#: removes the same overhead a level up.  Two fixes for one waste share it;
#: they do not compound.  The model must never be handed both factors.
#: (k, lines, n_tiles, seconds per tile) for `ldlq_quantize_blocks`.
#:
#: THE TILE COUNT IS PART OF THE MEASUREMENT, and leaving it out was a real gap
#: rather than a missing detail.  `auto_chunk` turns `n_tiles` into a chunk and
#: the chunk into the row count `_nearest` sees, and the row count decides which
#: search path runs.  A tile time is therefore only interpretable next to the
#: tile count it was taken at -- and until 2026-08-25 these rows carried none.
#: The cost of that surfaced the same day: two constants moved, and the old rows
#: could not be scaled to the new behaviour because nobody could say what regime
#: they had been measured in.
#:
#: Re-measured by `experiments/m0_tile_timings.py`, which DERIVES the shapes from
#: `accounting.density_for_budget` rather than choosing them: every cell of the
#: tile axis at B=1.5 on the 4096x4096 layer, four of every seven linears.
#:
#: TWO SAMPLE SHAPES CHANGED, DELIBERATELY:
#:   * (3072, 128) is retired.  No cell of the grid has 128 lines -- it was a
#:     probe standing in for the coarse end, and at T=max the real shape is 4096
#:     lines in ONE tile, a different regime entirely.
#:   * T=1, 2, 8 and 32 are added.  `codebook_seconds_per_vector` picks the
#:     nearest measured line count in log space, and with samples only at 4, 16
#:     and 128 the entire fine end of the grid snapped to a single point.
#:
#: THE NEW NUMBERS ARE NOT COMPARABLE TO THE OLD ONES, and saying so is the
#: lesson.  At the two shapes both tables share the new times are 1.42x and
#: 2.15x lower; the constant change measured 1.38x at (2944, 16), so the rest is
#: the old measurement's unknown tile count -- fewer tiles give a smaller chunk
#: and a higher per-tile cost.  There is no way to separate the two after the
#: fact, which is precisely why the column now exists.
#:
#: `cpu_f64` is NOT re-measured: the pipeline runs cuda/float32 (decision
#: 2026-08-23) and that row exists only for comparison.  Its tile count is 1 by
#: construction -- it describes the one-tile-at-a-time arrangement -- so it is
#: the one place the provenance was never actually missing.
TILE_TIMINGS = {
    "cpu_f64": ((2560, 4, 1, 1.741), (2944, 16, 1, 8.721),
                (3072, 128, 1, 95.83)),
    "cuda_f32": (
        (1024, 1, 4096, 0.00729346),
        (2048, 2, 2048, 0.00880856),
        (2560, 4, 1024, 0.00996955),
        (2816, 8, 512, 0.0140972),
        (2944, 16, 256, 0.018819),
        (3008, 32, 128, 0.0300036),
        (3072, 4096, 1, 1.9503),
    ),
}

#: The `hessian_block` each row of `TILE_TIMINGS` was measured under, or `None`
#: for a full-width factorization.
#:
#: `codebook_seconds_per_vector` subtracts the Cholesky out of a tile time
#: before fitting the codebook constant, and it has to subtract the one that was
#: actually IN the measurement.  The two rows were taken under different
#: arrangements -- `cuda_f32` was re-measured with `hessian_block=512` on
#: 2026-08-23, `cpu_f64` still describes the full-width one-tile form -- so a
#: single assumption cannot be right for both.
#:
#: Getting this wrong was the model's FIFTH error and its second optimistic one.
#: Subtracting a full-width Cholesky from a blocked measurement removed time the
#: tile never spent, undercharging the codebook by 34% at (2560, 4), 24% at
#: (2944, 16) and 9% at (3072, 128) -- worst at the fine granularities, which is
#: where the grid's cost lives.  A setup may override this on its rates entry as
#: `tile_timing_block`; anything unlisted keeps the full-width assumption.
TILE_TIMING_BLOCK = {"cpu_f64": None, "cuda_f32": 512}

#: Seconds for ONE `quantize._upper_inverse_factor` call, measured on this
#: machine with both kernels warmed: (k, seconds).
#:
#: A single flop/s number does not describe this kernel.  A Cholesky at k=1024
#: cannot fill the card; at k=8192 it nearly does, and the effective rate runs
#: 5.7e11 to 3.8e12 across that range -- a factor of 6.8.  Charging every width
#: the rate measured at k=2048 overstated the widest layers by 2.6x, on top of
#: the 1.6x the missing warmup cost, and `down_proj` at T=16 has k=7912.
#:
#: This is the fourth time this model has been wrong and the first time it was
#: wrong PESSIMISTICALLY -- which is worse than it sounds, because this is the
#: number that decides whether M1 gets run at all.
CHOL_TIMINGS = {
    "cuda_f32": ((1024, 0.003143), (2048, 0.010045),
                 (4096, 0.045681), (8192, 0.238243)),
    "cpu_f64": ((1024, 0.008980), (2048, 0.069634),
                (4096, 0.418345), (8192, 3.196408)),
}

#: Seconds for ONE `q @ H_t @ q.T` -- rotating a tile's sub-Hessian into the
#: block's basis, which `m1_gates.tile_hessian_stream` does once per tile.
#:
#: It was never modelled because the Cholesky dwarfed it.  It no longer does:
#: at k=7912 the rotation is 0.25s against the Cholesky's 0.22s, so once the
#: factorization is confined to blocks THIS becomes the larger of the two.
#: Leaving out a term bigger than one we itemize is exactly how this model went
#: wrong the first three times.
ROT_TIMINGS = {
    "cuda_f32": ((1024, 0.000478), (2048, 0.003390),
                 (4096, 0.025949), (8192, 0.250471)),
    "cpu_f64": ((1024, 0.016567), (2048, 0.102614),
                (4096, 0.693413), (8192, 5.558341)),
}

#: Measured, `experiments/m0_dense_ppl.py`: one streamed WikiText-2 pass at
#: seqlen 4096 on this 8 GB card.
EVAL_SECONDS = 238.0

#: Calibration tokens one M1 point sees: the preregistration's 128 samples at
#: its primary seqlen of 4096.
CALIBRATION_TOKENS = 128 * 4096

#: (tokens, seqlen, statistics seconds, block-forward seconds) for ONE block,
#: measured on this machine over Llama-2-7B block 0 with its seven linears.
#:
#: THE MODEL'S SIXTH ERROR, AND ITS LARGEST.  This term was never charged at
#: all.  `calibrate.sequential_calibrate` runs every block TWICE per point --
#: once with hooks to accumulate each linear's Hessian, once more so the next
#: block sees what the compressed model actually produces (Spec v6 trap 20) --
#: and neither pass appeared anywhere in this file.  Nothing caught it because
#: nothing had ever run the full driver: `experiments/m1_run.py` does not exist,
#: which is the same gap `docs/STATUS.md` section 8.1 is about.
#:
#: At the configuration the code shipped with -- accumulators pinned to the CPU
#: -- it was 5.59 hours per point, MORE than the entire compression pass at
#: every tile size, and M1 would have taken 39.8 days rather than the 12 this
#: model reported.  Accumulating on the block's own device in float32 is 25x
#: (`calibrate.collect_block_statistics`) and brings it to 0.26.
#:
#: The variants, measured over 16,384 tokens on one block, for the record:
#:   cpu float64 (the old default)   19.65 s
#:   cuda float64                    29.86 s   -- fp64 is ~1/64 rate here
#:   cuda float32                     0.91 s   -- 5.06e-06 from the float64 answer
#:   cuda float64, float32 products   0.99 s   -- 5.08e-06, so it buys nothing
#:
#: The statistics term is linear in tokens and independent of seqlen.  The
#: forward term is linear in tokens too but its attention is quadratic in
#: SEQLEN, and this was measured at 2048 against M1's 4096 -- so that half is
#: understated roughly twofold.  It is 0.25 s against 0.91, so the total is not
#: sensitive to it; say so rather than quietly scale it.
CALIBRATION_TIMINGS = {
    "cuda_f32": (16384, 2048, 0.91, 0.25),
}

#: (n_out, n_in, exact seconds, seconds at block=512) for ONE
#: `prune.forward_compensate`, measured on this machine at Llama-2-7B's shapes.
#:
#: THE MODEL'S SEVENTH ERROR, AND THE THIRD IN A ROW THAT IS AN OMISSION.
#: `m1_gates.run_config` calls `prune` before anything this file charges, and
#: `TILE_TIMINGS` starts at `ldlq_quantize_blocks` -- so the compensation sweep,
#: a Python loop the length of `n_in` whose every iteration touches the whole
#: remaining width, was priced nowhere.  It is 40.7 s per block, 0.362 h per
#: point, 1.58 days of M1.  Like calibration it does not depend on the tile
#: size, so it is another flat per-point term and it shifts the design
#: economics the same way.
#:
#: The second column is what `compensate_block=512` costs instead.  It is 6.63x
#: on the term and NOT bit-identical (float32 epsilon, 2.7e-06 to 4.8e-06), so
#: it is priced here and left off by default.
#:
#: Note what nearly happened.  Blocking was measured once at (512, 2048) and
#: (512, 4096), read 0.87-1.06x, and went into `docs/STATUS.md` section 7.2 as
#: "no gain, do not try again".  Those widths are launch-bound; the real ones
#: are bandwidth-bound and blocking is up to 9.9x there.  A rejection measured
#: in the wrong regime is worse than no measurement, because it stops the next
#: person looking.
COMPENSATE_TIMINGS = {
    "cuda_f32": ((4096, 4096, 2.431, 0.665),
                 (11008, 4096, 6.345, 0.820),
                 (4096, 11008, 18.260, 1.837)),
}

TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)


# --------------------------------------------------------------------------- #
# Rates, measured here
# --------------------------------------------------------------------------- #

def _spd(k: int, dtype, device):
    g = torch.Generator().manual_seed(0)
    a = torch.randn(k, k, generator=g, dtype=torch.float64)
    return (a @ a.T / k + torch.eye(k, dtype=torch.float64)).to(dtype).to(device)


def cholesky_rate(k: int, dtype, device, reps: int = 5) -> float:
    """Effective flop/s for the triple LDLQ performs.

    The warmup runs the WHOLE triple, not just the first `cholesky`.  It used to
    run only the first, so the opening timed rep paid cuSOLVER's handle setup
    for `cholesky_inverse`; over three reps that was most of the measurement and
    it understated the rate 1.6x.  Combined with the rate's k-dependence
    (`CHOL_TIMINGS`), the model was charging real widths 9.4x too much.
    """
    h = _spd(k, dtype, device)
    for _ in range(3):
        torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)),
                              upper=True)
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
               vq_bits: float = 2.0, dtype_bytes: int = 8,
               scale_fit: bool = True,
               hessian_block: int | None = DEFAULT_HESSIAN_BLOCK) -> dict | None:
    """Flops and bytes for one linear at one tile size.

    `k` is the aligned survivor count, so it is what the code will really
    factorize -- not the requested density times n_in.

    `hessian_block=b` prices `quantize.ldlq_quantize(hessian_block=b)`: the
    factorization becomes (k/b) blocks of b^3 rather than one of k^3, so the
    term falls by (k/b)^2 -- three orders of magnitude at the widths this grid
    actually uses.  It is the only lever here that changes an exponent, and it
    is charged against a measured quality cost, not assumed free.
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
        "hessian_block": hessian_block,
        # One factorization per tile -- per TILE, because each tile owns a
        # different column set.  The rotation is already shared across tiles
        # (`rotation.rotate(share_across_tiles=True)`), so it is not what makes
        # this per-tile and no rotation width can make it shared.
        "cholesky_flops": n_tiles * CHOL_FLOPS_PER_K3 * (
            k ** 3 if hessian_block is None or hessian_block >= k
            else _blockdiag_chol_k3(k, hessian_block)),
        # one search per tile per group of eight, over that tile's lines --
        # times the scale-fitting sweep that precedes every tile
        "codebook_flops": (n_out * (k / E8P_DIM) * 2 * E8P_DIM * CODEBOOK_SIZE
                           * (SCALE_FIT_MULTIPLIER if scale_fit else 1.0)),
        "scale_fit": scale_fit,
        # what `tile_hessians` allocates in one go
        "hessian_bytes": n_tiles * k * k * dtype_bytes,
        # what it would allocate if the tiles were streamed instead
        "hessian_bytes_streamed": k * k * dtype_bytes,
    }


def cholesky_seconds(k: int, rates: dict, setup: str,
                     block: int | None = None) -> float:
    """Seconds for one tile's factorization at width `k`.

    Uses the measured `CHOL_TIMINGS` curve when the setup has one, taking the
    rate from the nearest measured k in log space -- the samples are octaves
    apart, so interpolating between them would invent precision the data lacks.
    Falls back to a flat `cholesky_flops_per_s` for setups that carry no curve.

    `block` prices the block-diagonal form as (k/block) separate factorizations
    at the block's own rate.  That OVERSTATES it, because the real code issues
    them as one batched call, and overstating is the correct direction for the
    lever we are arguing in favour of.
    """
    if block is not None and block < k:
        return sum(cholesky_seconds(min(block, k - o), rates, setup)
                   for o in range(0, k, block))
    entry = rates["setups"][setup]
    samples = entry.get("cholesky_timings") or CHOL_TIMINGS.get(setup)
    if not samples:
        return CHOL_FLOPS_PER_K3 * k ** 3 / entry["cholesky_flops_per_s"]
    ref_k, ref_s = min(samples, key=lambda s: abs(math.log(s[0] / max(k, 1))))
    rate = CHOL_FLOPS_PER_K3 * ref_k ** 3 / ref_s
    return CHOL_FLOPS_PER_K3 * k ** 3 / rate


def rotation_seconds(k: int, rates: dict, setup: str) -> float:
    """Seconds to rotate one tile's sub-Hessian, `2*k^3` at matmul rates.

    Charged whenever the pipeline rotates -- which, after the block-width sweep,
    is always: narrowing the rotation costs quality at every width measured
    (`experiments/m0_rotation_value.py`), so the rotation stays full even when
    the feedback does not.  Returns 0 for setups with no measured curve, so the
    fixtures keep testing the shapes they were written for.
    """
    samples = (rates["setups"][setup].get("rotation_timings")
               or ROT_TIMINGS.get(setup))
    if not samples:
        return 0.0
    ref_k, ref_s = min(samples, key=lambda s: abs(math.log(s[0] / max(k, 1))))
    return ref_s * (k / ref_k) ** 3


def scale_sample_bites(lines_per_tile: int, k: int, sample: int) -> bool:
    """Does capping `fit_scale` at `sample` vectors change anything for this tile?

    A tile holds `lines_per_tile * k / E8P_DIM` vectors and the sweep visits each
    one.  If that is already below the cap, the cap is inert -- and reporting a
    saving for it would be the fourth time this model was optimistic.
    """
    return lines_per_tile * k // E8P_DIM > sample


def _blockdiag_chol_k3(k: int, block: int) -> float:
    """Cubed work for a block-diagonal factorization, ragged tail included."""
    return float(sum(min(block, k - o) ** 3 for o in range(0, k, block)))


def codebook_seconds_per_vector(rates: dict, setup: str,
                                lines: int | None = None) -> float:
    """Seconds per quantized weight, from the measured tile times.

    The Cholesky is subtracted first at its microbenchmarked rate -- that one
    IS a clean LAPACK measurement -- and what is left is charged to the codebook
    work, the scale-fitting sweep included.  It is subtracted at the width the
    timings were MEASURED under (`TILE_TIMING_BLOCK`), not at full width:
    subtracting a full-width factorization from a blocked measurement takes out
    time the tile never spent and undercharges the codebook by up to 34%.

    The constant is NOT one number.  A tile's line count is its tile size, and
    per-weight cost falls sharply with it: bigger batches amortize the codebook
    load, the lattice decoder's fixed cost, and above all `fit_scale`'s, since
    that is fitted once per tile over however many vectors the tile holds.
    Measured across the whole tile axis it runs 6.36e-06 per weight at one line
    down to 1.55e-07 at 4096 -- a factor of 41, and 8x over the first step
    alone.

    Which is why the sample set now covers every line count the grid uses.  With
    samples only at 4, 16 and 128 the lookup below snapped T=1 and T=2 onto the
    four-line rate and understated them roughly twofold, and that is the end of
    the grid where the unstructured baseline lives.

    `lines=None` keeps the old conservative behaviour and returns the worst.
    """
    samples = (rates["setups"][setup].get("tile_timings")
               or TILE_TIMINGS.get(setup))
    if not samples:
        raise ValueError(
            f"no measured tile timings for setup {setup!r}; add them to "
            "TILE_TIMINGS or carry them on the rates entry"
        )
    measured_block = rates["setups"][setup].get(
        "tile_timing_block", TILE_TIMING_BLOCK.get(setup))
    fitted = []
    for k, sample_lines, _n_tiles, seconds in samples:
        residual = seconds - cholesky_seconds(k, rates, setup,
                                              block=measured_block)
        if residual > 0:
            fitted.append((sample_lines, residual / (sample_lines * k)))
    if not fitted:
        raise ValueError(f"tile timings for {setup!r} leave nothing to fit")
    if lines is None:
        return max(v for _, v in fitted)
    # Nearest measured line count in log space -- the samples are octaves apart,
    # so interpolating between them would invent precision the data lacks.
    return min(fitted, key=lambda s: abs(math.log(s[0] / max(lines, 1))))[1]


def calibration_seconds(rates: dict, setup: str, *,
                        tokens: int = CALIBRATION_TOKENS,
                        n_blocks: int = N_BLOCKS) -> float:
    """One point's calibration: statistics plus the re-forward, over all blocks.

    Independent of the tile size and of the budget -- it is the same walk
    whatever is being compressed -- so it is a flat addition per point, and that
    changes which DESIGNS are cheap.  When compression dominated, cost followed
    which tiles were run; when calibration does, it follows how many POINTS are
    run, and design G (two tile sizes, five draws, ten points) stops being
    cheaper than design F (seven tile sizes, one draw, seven points).

    Returns 0.0 for a setup with no measurement rather than guessing: this term
    was invisible for six versions of this model and a fabricated number would
    put it back.
    """
    entry = rates["setups"][setup].get("calibration_timings") \
        or CALIBRATION_TIMINGS.get(setup)
    if not entry:
        return 0.0
    ref_tokens, _seqlen, stats_s, fwd_s = entry
    return (stats_s + fwd_s) * (tokens / ref_tokens) * n_blocks


def compensate_seconds(rates: dict, setup: str, *, inventory=LLAMA2_7B,
                       n_blocks: int = N_BLOCKS,
                       compensate_block: int | None = None) -> float:
    """One point's forward compensation, over every linear in every block.

    Like calibration this is flat in the tile size -- the sweep is over `n_in`
    and the mask does not change its length -- so it is another per-POINT term,
    and per-point terms are what decide which designs are cheap.

    Requires an exact (n_out, n_in) match and returns 0.0 for anything
    unmeasured, rather than interpolating.  Five of this model's seven errors
    were terms it did not know about; a plausible-looking guess is how the
    sixth would have been missed as well.
    """
    entry = rates["setups"][setup].get("compensate_timings") \
        or COMPENSATE_TIMINGS.get(setup)
    if not entry:
        return 0.0
    table = {(n_out, n_in): (exact, blocked)
             for n_out, n_in, exact, blocked in entry}
    total = 0.0
    for n_out, n_in, count in inventory:
        measured = table.get((n_out, n_in))
        if measured is None:
            return 0.0
        total += count * (measured[0] if compensate_block is None
                          else measured[1])
    return total * n_blocks


def model_cost(tile_size, budget: float, rates: dict, setup: str, *,
               batched: bool = False, scale_fit: bool = True,
               hessian_block: int | None = DEFAULT_HESSIAN_BLOCK,
               inventory=LLAMA2_7B, n_blocks: int = N_BLOCKS,
               calibration_tokens: int = CALIBRATION_TOKENS,
               compensate_block: int | None = None) -> dict:
    """One full compression pass over Llama-2-7B at one tile size.

    Timing comes from the measured per-tile fit, not from flops over kernel
    rates.  `batched` is kept for the flop accounting only -- the measured times
    already contain whatever batching the code does.
    """
    r = rates["setups"][setup]

    chol = cb = peak = peak_streamed = 0.0
    chol_seconds = cb_seconds = rot_seconds = 0.0
    for n_out, n_in, count in inventory:
        c = layer_cost(n_out, n_in, tile_size, budget, scale_fit=scale_fit,
                       hessian_block=hessian_block)
        if c is None:
            return {"tile_size": tile_size, "skipped": "budget unreachable"}
        chol += count * c["cholesky_flops"]
        cb += count * c["codebook_flops"]
        chol_seconds += count * c["n_tiles"] * cholesky_seconds(
            c["k"], rates, setup, block=hessian_block)
        rot_seconds += count * c["n_tiles"] * rotation_seconds(
            c["k"], rates, setup)
        # n_tiles * lines * k == n_out * k, and the tile fit is per vector
        per_vector = codebook_seconds_per_vector(rates, setup,
                                                 c["lines_per_tile"])
        vectors = n_out * c["k"] / E8P_DIM
        cb_seconds += count * vectors * E8P_DIM * per_vector * (
            1.0 if scale_fit else 1.0 / SCALE_FIT_MULTIPLIER)
        peak = max(peak, c["hessian_bytes"])
        peak_streamed = max(peak_streamed, c["hessian_bytes_streamed"])

    chol *= n_blocks
    cb *= n_blocks
    chol_seconds *= n_blocks
    rot_seconds *= n_blocks
    cb_seconds *= n_blocks
    # `TILE_TIMINGS` measures `ldlq_quantize` alone, so the rotation is a
    # genuinely separate term rather than a double count of one already inside
    # the fitted codebook constant.
    seconds = chol_seconds + rot_seconds + cb_seconds
    cal_seconds = calibration_seconds(rates, setup, tokens=calibration_tokens,
                                      n_blocks=n_blocks)
    comp_seconds = compensate_seconds(rates, setup, inventory=inventory,
                                      n_blocks=n_blocks,
                                      compensate_block=compensate_block)
    return {
        "tile_size": tile_size, "setup": setup, "batched": batched,
        "scale_fit": scale_fit,
        "cholesky_flops": chol, "codebook_flops": cb,
        "cholesky_seconds": chol_seconds,
        "rotation_seconds": rot_seconds,
        "codebook_seconds": cb_seconds,
        "codebook_seconds_per_vector": per_vector,   # last layer's
        "compress_seconds": seconds,
        "calibration_seconds": cal_seconds,
        "compensate_seconds": comp_seconds,
        "eval_seconds": EVAL_SECONDS,
        # A point is what `m1_run.py` will actually do: gather the statistics,
        # prune with compensation, compress, evaluate.  `compress_seconds` was
        # standing in for all four until 2026-08-24, and the two terms added
        # that day were worth 28 days and 1.6 days of M1 respectively.
        "point_seconds": seconds + cal_seconds + comp_seconds + EVAL_SECONDS,
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
            n_draws: int = 5, tiles=TILES, batched: bool = False,
            scale_fit: bool = True,
            hessian_block: int | None = DEFAULT_HESSIAN_BLOCK) -> dict:
    """M1's own grid, for scale: budgets x tiles x draws."""
    total = 0.0
    for b in budgets:
        for t in tiles:
            c = model_cost(t, b, rates, setup, batched=batched,
                           scale_fit=scale_fit, hessian_block=hessian_block)
            if "skipped" not in c:
                total += n_draws * c["point_seconds"]
    return {"setup": setup, "batched": batched, "scale_fit": scale_fit,
            "hessian_block": hessian_block,
            "n_draws": n_draws, "seconds": total, "hours": total / 3600,
            "days": total / 86400}


# --------------------------------------------------------------------------- #

def modellable_setups(rates: dict) -> list[str]:
    """Setups we hold END-TO-END tile timings for.

    `measure_rates` benchmarks every dtype/device combination the machine has,
    but the model's clock comes from measured tile times and those exist only
    for the two configurations the pipeline is actually run in.  Reporting a
    cost for `cpu_f32` would mean inventing its constant, so it is dropped
    rather than guessed.
    """
    return [s for s in rates["setups"]
            if rates["setups"][s].get("tile_timings") or s in TILE_TIMINGS]


def run(*, budget: float = 1.5, cache: Path | None = None) -> dict:
    rates = measure_rates(cache=cache)
    setups = modellable_setups(rates)
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
