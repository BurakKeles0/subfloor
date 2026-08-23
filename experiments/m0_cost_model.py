"""M0 -- what does a real run cost, before we commit to one?

`docs/STATUS.md` puts a warning on the tau sweep: the spec estimates it at 25
GPU-hours, and that estimate should be checked against one measured point before
anything is committed.  This checks it -- and the answer is not the one the
question expected, because the sweep is not the first thing that stops working.

Everything here rests on constants measured ON THIS MACHINE, not on peak
figures.  Three terms are charged, each from its own measured curve:

  codebook   the nearest-codeword search and the scale sweep in front of it
  rotation   `q @ H_t @ q.T`, once per tile
  cholesky   the per-tile sub-Hessian factorization LDLQ needs, O(k^3) per tile

They are listed in that order because that is their size, and it took four
corrections to find out.  The factorization LOOKS like it should dominate --
it is the only cubic term -- and for a while the model said it did.  It does
not: at B=1.5, T=4 the pass is 25.0 hours of codebook against 2.96 of Cholesky
and 1.92 of rotation.  Confining the factorization to blocks
(`quantize.ldlq_quantize(hessian_block=...)`) is worth 9% of M1; dropping the
per-tile scale fit is worth 70%.

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
     whether M1 can be run, and it read 120 days when the answer is 94.

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

#: Cholesky, cholesky_inverse, cholesky again -- roughly k^3/3 + k^3 + k^3/3.
CHOL_FLOPS_PER_K3 = 5.0 / 3.0

#: One nearest-codeword search: a [rows, 8] x [8, 2^16] product.
CODEBOOK_SIZE = 1 << 16
E8P_DIM = Qz.E8P_DIM

#: `ldlq_quantize` does not search the codebook once per group -- it first calls
#: `fit_scale`, which sweeps 24 candidate scales and searches the WHOLE tile at
#: each one.  Measured on an o_proj-shaped tile (4 lines x 2560 survivors), that
#: sweep is 83% of the tile's total time, a 6x multiplier on everything below.
#:
#: It is also the most avoidable cost in the pipeline: QuIP# fits one scale per
#: LAYER, and a per-tile scale buys little while costing this.  Left in the
#: model as a measured multiplier rather than quietly assumed away, with
#: `--no-scale-fit` to price the alternative.
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
SCALE_FIT_MULTIPLIER = 6.0

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
TILE_TIMINGS = {
    "cpu_f64": ((2560, 4, 1.741), (2944, 16, 8.721), (3072, 128, 95.83)),
    "cuda_f32": ((2560, 4, 0.247), (2944, 16, 0.454), (3072, 128, 2.309)),
}

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
               hessian_block: int | None = None) -> dict | None:
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
    work, the scale-fitting sweep included.

    The constant is NOT one number.  A tile's line count is its tile size, and
    per-weight cost falls sharply with it: bigger batches amortize both the
    codebook load and the lattice decoder's fixed cost, and below a threshold
    the decoder is not used at all.  On this machine the residual runs 1.72e-5
    at four lines down to 5.6e-6 at 128.  Charging every tile size the
    four-line rate would overstate the coarse end threefold, which is precisely
    the end the granularity question cares about.

    `lines=None` keeps the old conservative behaviour and returns the worst.
    """
    samples = (rates["setups"][setup].get("tile_timings")
               or TILE_TIMINGS.get(setup))
    if not samples:
        raise ValueError(
            f"no measured tile timings for setup {setup!r}; add them to "
            "TILE_TIMINGS or carry them on the rates entry"
        )
    fitted = []
    for k, sample_lines, seconds in samples:
        residual = seconds - cholesky_seconds(k, rates, setup)
        if residual > 0:
            fitted.append((sample_lines, residual / (sample_lines * k)))
    if not fitted:
        raise ValueError(f"tile timings for {setup!r} leave nothing to fit")
    if lines is None:
        return max(v for _, v in fitted)
    # Nearest measured line count in log space -- the samples are octaves apart,
    # so interpolating between them would invent precision the data lacks.
    return min(fitted, key=lambda s: abs(math.log(s[0] / max(lines, 1))))[1]


def model_cost(tile_size, budget: float, rates: dict, setup: str, *,
               batched: bool = False, scale_fit: bool = True,
               hessian_block: int | None = None,
               inventory=LLAMA2_7B, n_blocks: int = N_BLOCKS) -> dict:
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
    return {
        "tile_size": tile_size, "setup": setup, "batched": batched,
        "scale_fit": scale_fit,
        "cholesky_flops": chol, "codebook_flops": cb,
        "cholesky_seconds": chol_seconds,
        "rotation_seconds": rot_seconds,
        "codebook_seconds": cb_seconds,
        "codebook_seconds_per_vector": per_vector,   # last layer's
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
            n_draws: int = 5, tiles=TILES, batched: bool = False,
            scale_fit: bool = True, hessian_block: int | None = None) -> dict:
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
