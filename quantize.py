"""E8P lattice vector quantization for compacted survivors.

QuIP#'s E8P codebook, reconstructed from the paper and verified by enumeration:

    source codebook S : 227 non-negative half-integer patterns with norm^2 <= 10,
                        plus 29 padding patterns with norm^2 == 12  ->  256 total
    codeword (16 bit) : 8 bits index into S
                        7 bits sign the first seven coordinates
                        1 bit  shift the whole vector by +1/4 or -1/4
                        (the eighth sign is not stored: it is whichever makes
                         the coordinate sum even, i.e. lands in the lattice)

    256 * 2^7 * 2 = 2^16 codewords over 8 dimensions  ->  EXACTLY 2 bits/weight

Two facts matter for the accounting and both are structural, not empirical:

  * 2 bits per weight, so `vq_bits = 2.0`.
  * the codebook is a 256-entry table (~1 KiB) fixed for all models, so Spec v6
    section 3.2's `codebook_amortization` really is 0 -- unlike AQLM, whose
    codebook is trained per model and amortizes at +0.186 bits.

Why half-integers: they are never zero, so every sign flip yields a distinct
vector.  That is what makes the 7-bit sign field lossless and the count close
exactly.

SCOPE.  This is a faithful reconstruction of the codebook GEOMETRY and rate, and
it is what the experiments quantize with.  It is not bit-compatible with QuIP#'s
released kernels -- the choice of which 29 padding patterns to use is not
specified in the paper text, and we take the lexicographically smallest.  That
only matters if kernels ever come into scope (Spec v6 section 8 says they do
not).

WARNING (plan H5).  That E8P holds its quality on a COMPACTED SURVIVOR
submatrix is an explicit, untested assumption.  Survivors are the fat tail of
the weight distribution by construction, while a lattice quantizer wants
something Gaussian.  `quantization_snr` exists so the caller can watch for it:
if a layer's SNR falls far short of the dense reference, the assumption is
breaking and the fallback is rotation + GPTQ-3bit.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

__all__ = [
    "E8P_DIM",
    "E8P_INDEX_BITS",
    "E8P_BITS_PER_WEIGHT",
    "source_codebook",
    "e8p_codebook",
    "in_e8_plus_quarter",
    "is_canonical_codebook",
    "fit_scale",
    "fit_scales",
    "FIT_STEPS",
    "quantize_vectors",
    "quantize_blocks",
    "LDLQResult",
    "ldlq_quantize",
    "auto_chunk",
    "ldlq_quantize_blocks",
    "quantization_snr",
]

E8P_DIM = 8
E8P_INDEX_BITS = 16
E8P_BITS_PER_WEIGHT = E8P_INDEX_BITS / E8P_DIM        # 2.0, exactly

_SOURCE_SIZE = 256
_INNER_NORM2 = 10          # ||s||^2 <= 10  -> 227 patterns
_PAD_NORM2 = 12            # ||s||^2 == 12  -> 29 taken as padding


# --------------------------------------------------------------------------- #
# Codebook
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=4)
def source_codebook(dtype: torch.dtype = torch.float64) -> Tensor:
    """The 256-entry source table S, [256, 8].

    Non-negative half-integer patterns: 227 with norm^2 <= 10, then 29 of the
    224 patterns with norm^2 == 12, taken in lexicographic order.
    """
    grid = [i + 0.5 for i in range(5)]                 # 0.5 .. 4.5 covers norm^2<=12
    inner, pad = [], []
    for v in itertools.product(grid, repeat=E8P_DIM):
        n2 = sum(x * x for x in v)
        if n2 <= _INNER_NORM2:
            inner.append(v)
        elif abs(n2 - _PAD_NORM2) < 1e-9:
            pad.append(v)

    if len(inner) != 227:
        raise AssertionError(
            f"expected 227 patterns with norm^2 <= {_INNER_NORM2}, got {len(inner)} "
            "-- the E8P reconstruction is wrong, do not use these numbers"
        )
    need = _SOURCE_SIZE - len(inner)
    S = inner + sorted(pad)[:need]
    return torch.tensor(S, dtype=dtype)


@lru_cache(maxsize=4)
def e8p_codebook(dtype: torch.dtype = torch.float64) -> Tensor:
    """All 2^16 codewords, [65536, 8].

    For each source pattern and each of 2^7 sign choices on the first seven
    coordinates, the eighth sign is set so the coordinate sum is even (lattice
    membership).  Then the vector is shifted by +1/4 or -1/4.
    """
    S = source_codebook(dtype)                                    # [256, 8]

    bits = torch.arange(2 ** (E8P_DIM - 1))
    head = torch.stack(
        [1 - 2 * ((bits >> i) & 1) for i in range(E8P_DIM - 1)], dim=1
    ).to(dtype)                                                   # [128, 7], +-1

    signed = S.unsqueeze(1).clone().expand(-1, head.shape[0], -1).clone()
    signed[:, :, : E8P_DIM - 1] *= head.unsqueeze(0)              # [256, 128, 8]

    # Eighth sign: pick it so that the sum is even.  Sums are integers here
    # because eight half-integers always add to an integer.
    partial = signed[:, :, : E8P_DIM - 1].sum(dim=2)
    last = signed[:, :, E8P_DIM - 1]
    flip = torch.remainder(partial + last, 2.0) != 0
    signed[:, :, E8P_DIM - 1] = torch.where(flip, -last, last)

    flat = signed.reshape(-1, E8P_DIM)                            # [32768, 8]
    return torch.cat([flat + 0.25, flat - 0.25], dim=0).contiguous()


def in_e8_plus_quarter(x: Tensor, atol: float = 1e-9) -> Tensor:
    """Membership test for E8 +- 1/4, elementwise over rows of `x` [..., 8].

    Undo the shift, then require all-half-integer coordinates with an even sum.
    """
    if x.shape[-1] != E8P_DIM:
        raise ValueError(f"last dim must be {E8P_DIM}, got {x.shape[-1]}")
    ok = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
    for shift in (0.25, -0.25):
        y = x - shift
        half = ((y - 0.5).remainder(1.0).abs() < atol) | (
            (y - 0.5).remainder(1.0).abs() > 1.0 - atol
        )
        even = (y.sum(dim=-1).remainder(2.0).abs() < atol) | (
            y.sum(dim=-1).remainder(2.0).abs() > 2.0 - atol
        )
        ok |= half.all(dim=-1) & even
    return ok


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #

#: |h_i| for a codeword's unshifted part is 0.5, 1.5 or 2.5 -- 3.5 is impossible
#: because 3.5^2 alone exceeds the largest norm^2 the codebook keeps.  Three
#: levels per coordinate makes the whole pattern space 3^8 = 6561 entries.
_LEVELS = 3

#: Rows below which the decoder is not worth it.  Its cost is fixed -- some
#: forty small elementwise kernels and two gathers, regardless of how few rows
#: it is given -- while the scan it replaces is proportional to them.  Measured
#: crossovers on this machine: around 64 rows on CPU, around 1000 on GPU, where
#: launch overhead is far heavier.
#:
#: This is why the win lands where it does.  `fit_scale` sweeps whole tiles at
#: once (thousands of rows) and takes the fast path; the LDLQ group sweep asks
#: for one group of lines at a time (sixteen, say) and keeps the scan.  Since
#: the scale sweep was 83% of a tile's cost, that was the useful half.  It is
#: 28% since the candidates were batched (`fit_scale`), so this floor now
#: matters less than it did -- but it still decides which path a fit takes.
_LATTICE_MIN_ROWS = {"cpu": 64, "cuda": 1024}

#: Fraction of its rows `nearest_e8p` cannot settle, so they need a second pass.
#:
#: Measured 34.9% at every shape tried -- k=2560/2944/3072, tile counts from 8
#: to 256 -- which makes it a property of how much of R^8 the codebook's norm
#: ball covers rather than of any particular tile.  Not exactly constant, and
#: the exception is on the record because it bit a test: at k=512, a width the
#: grid never runs, it is 30.6%.  So treat 0.349 as the typical value with the
#: observed range 31-35%, and leave the inequality below margin rather than
#: equality -- at 2048 rows even 31% clears the threshold twice over.
#:
#: It describes the SWEEP, which is what reads it: by then `fit_scale` has
#: matched the scale to the codebook.  The small-alpha steps INSIDE the fit are
#: a different regime entirely and miss on up to 99% of rows
#: (`docs/STATUS.md` section 6.4).
#:
#: Written down because three constants have to satisfy one inequality and
#: nothing said so until they had already violated it for months:
#:
#:      CHUNK_TARGET_ROWS * DECODER_MISS_FRACTION  >  _ANALYTIC_MIN_ROWS
#:
#: Left to right: how many rows `auto_chunk` aims the sweep at, how many of them
#: the decoder hands on, and whether that leftover is big enough to take the
#: analytic path instead of a 65536-codeword scan.  With 1024, 0.349 and 384 it
#: read 357 > 384, which is false, so every group of the sweep scanned.
#: `tests/test_quantize.py` now asserts the inequality directly.
DECODER_MISS_FRACTION = 0.349

#: Unsettled rows below which a scan beats `nearest_e8p_analytic`.
#:
#: The analytic form does real work proportional to its input but has a fixed
#: cost of roughly a millisecond -- a dozen kernel launches against the 256
#: source patterns -- so on a handful of rows the scan, which is launch-bound
#: at that size too but launches less, gets there first.  It matters in both
#: directions: a heavy-tailed tile at T=4 misses on very few rows and would
#: otherwise pay the fixed cost 24 times inside `fit_scale` for nothing.
#:
#: 384 WAS TOO HIGH, AND THE COST OF THAT WAS STRUCTURAL RATHER THAN MARGINAL.
#: The decoder leaves about 34.9% of its rows unsettled (`DECODER_MISS_FRACTION`,
#: and see there for the range) and `auto_chunk` aimed the sweep at 1024 rows.  That
#: puts the leftover set at 357, just under this threshold, so EVERY group of
#: the sweep fell through to a 65536-codeword scan.  Eight of the twenty-one
#: layer-by-tile cells at B=1.5 land there, because the saturation ceiling is
#: `ceil(1024 / lines)` and `lines` divides 1024 at T=8, 16 and 32.  Counted on
#: a REAL layer -- Llama-2-7B block 0 `o_proj`, 2048 rows, B=1.5 -- one call per
#: group of the sweep, every time:
#:
#:      T=8    581 calls   184,915 rows scanned
#:      T=16   623 calls   193,184 rows
#:      T=32   645 calls   196,712 rows
#:
#: and zero under the constants below.
#:
#: Re-measured on the leftover set, where the decode is already paid either way,
#: at three input scales -- and the answer is NOT the first row count where the
#: analytic form wins.  It is the first where it wins at every scale:
#:
#:      rows    a=0.05   a=0.6   a=6.0
#:       192     0.74x   0.90x   0.66x
#:       224     0.77x   1.31x   0.76x
#:       256     0.93x   1.42x   1.42x
#:       320     1.65x   1.78x   1.61x   <- first row that wins everywhere
#:       384     2.11x   1.31x   1.85x
#:
#: Measured at a=0.6 alone, 192 looked like the crossover and would have been
#: 1.13x there while losing at both other scales -- the same trap
#: `_ANALYTIC_DIRECT_MIN_ROWS` records ("256 rather than 192 because 192 still
#: loses on one of the three").  A threshold has to be somewhere it never costs
#: anything to cross.
#:
#: A shape whose leftover falls under 320 drops to the scan exactly as it does
#: today, so a thin margin is a lost gain and never a regression.
#:
#: WHICH CELLS THIS STILL DECIDES, now that `CHUNK_TARGET_ROWS` is 2048.  Where
#: the row target binds, the leftover is 715 and clears either threshold; this
#: constant only matters where MEMORY binds first and the chunk cannot reach the
#: target.  That is `down_proj` -- k=7912, capped at 67 tiles, 1072 rows, 374
#: left over -- which is the single most expensive cell in the grid.  Measured
#: there with the row target already raised, moving this threshold alone took it
#: from 1.07x to 1.13x.  That is the whole reason both constants had to move:
#: one fixes the cells the row target reaches, the other the cell it cannot.
#:
#: Not to be unified with `_ANALYTIC_DIRECT_MIN_ROWS` (256) even though both
#: price the same comparison -- analytic against a scan on N rows, with the
#: decode paid on neither side or both.  The gap is measurement margin, and
#: closing it upward would push the T=4 `down_proj` cell, which hands the sweep
#: 308 rows, off the analytic path it currently takes.
_ANALYTIC_MIN_ROWS = 320

#: Rows below which a scan beats going STRAIGHT to `nearest_e8p_analytic`,
#: without the lattice decoder in front.
#:
#: A different question from `_ANALYTIC_MIN_ROWS`, which prices the analytic
#: form against a scan for rows the decoder has ALREADY failed on -- there its
#: fixed cost is marginal, because the decode is paid either way.  Reached
#: directly it has to cover that cost itself, so the crossover could only be
#: found by measuring, and it comes out LOWER rather than higher: the decoder's
#: own fixed cost was in the comparison.
#:
#: Measured on this machine, analytic against scan, three input scales:
#:      n=128   0.41x  0.63x  0.41x    scan wins
#:      n=192   0.68x  1.23x  1.20x    mixed
#:      n=256   0.99x  1.21x  1.68x    break-even to 1.7x
#:      n=384   1.41x  2.13x  2.46x
#:      n=512   2.04x  1.61x  3.33x
#:      n=816   2.68x  5.19x  6.04x
#: 256 rather than 192 because 192 still loses on one of the three and the
#: whole point of this constant is that it never costs anything to cross.
_ANALYTIC_DIRECT_MIN_ROWS = 256


def _device_key(device: torch.device | str) -> str:
    """Canonical cache key for `device`.

    `"cuda"` and `"cuda:0"` name one card and hash to two entries.  Cached on
    the spelling, that hands two callers two DIFFERENT codebook tensors -- and
    since the fast path is selected by an `is` against the cached one, the
    caller who spelled it short silently drops to the brute-force scan.

    `docs/STATUS.md` section 10 carried this as a benchmarking hazard for three
    sessions without the code being fixed.  On 2026-08-24 it invalidated four
    measurements, two of which were first misdiagnosed as GPU contention and as
    clock throttling, because the symptom is an optimisation that reads 1.00x.

    Tensors always report a fully qualified device, so `str(t.device)` -- what
    the pipeline itself passes -- takes the cheap branch.  Only a hand-written
    spelling reaches the resolver, which asks torch where a tensor would
    actually land rather than assuming device zero, so it stays right under
    `torch.cuda.set_device`.
    """
    d = device if isinstance(device, torch.device) else torch.device(device)
    if d.type == "cpu":
        return "cpu"                       # tensors report plain "cpu"
    if d.index is not None:
        return f"{d.type}:{d.index}"
    return str(torch.empty(0, device=d).device)


@lru_cache(maxsize=16)
def _codebook_cached(dtype: torch.dtype, device_key: str) -> Tensor:
    return e8p_codebook(dtype).to(device_key)


def _on_device(dtype: torch.dtype, device: torch.device | str) -> Tensor:
    """The codebook, cached PER DEVICE, keyed on the device rather than on how
    it was spelled (`_device_key`).

    `e8p_codebook(dtype).to(device)` copies two megabytes on every call, which
    is more work than the search it was meant to serve and makes an `is` check
    against it always false.  Caching the moved tensor is what lets the fast
    path be selected at all.
    """
    return _codebook_cached(dtype, _device_key(device))


def is_canonical_codebook(codebook: Tensor) -> bool:
    """Is this THE cached E8P table, so `_nearest` will take its fast path?

    Exported because the failure it guards against is silent and expensive: a
    benchmark that builds its own codebook, or spells the device short, measures
    the brute-force scan and reports no speedup for a change that has one.
    Assert this next to any timing of `_nearest`, `fit_scale` or a tile.
    """
    return codebook is _on_device(codebook.dtype, codebook.device)


@lru_cache(maxsize=16)
def _table_cached(device_key: str) -> Tensor:
    return _source_index_table().to(device_key)


def _table_on_device(device: torch.device | str) -> Tensor:
    return _table_cached(_device_key(device))


@lru_cache(maxsize=2)
def _source_index_table() -> Tensor:
    """[3^8] -> position in the source codebook, or -1 for "not a codeword".

    The key is the per-coordinate level of |h|, base 3.  This is what turns
    membership from a search into a gather: the codebook is a lattice
    INTERSECTED with a norm ball plus 29 arbitrarily chosen padding patterns, so
    landing on a lattice point proves nothing by itself.
    """
    S = source_codebook(torch.float64)                       # [256, 8]
    levels = (S - 0.5).round().to(torch.int64)
    powers = _LEVELS ** torch.arange(E8P_DIM, dtype=torch.int64)
    table = torch.full((_LEVELS ** E8P_DIM,), -1, dtype=torch.int64)
    table[(levels * powers).sum(dim=1)] = torch.arange(S.shape[0])
    return table


def _nearest_halfinteger_even(y: Tensor) -> Tensor:
    """Nearest point of D8 + 1/2 -- half-integers with an even coordinate sum.

    Conway and Sloane's D_n decoder.  Round every coordinate to its nearest
    half-integer; if the sum comes out odd, move the single worst-rounded
    coordinate to its second choice, which flips the parity at the smallest
    possible cost.
    """
    floor = torch.floor(y)
    base = floor + 0.5
    resid = y - floor                                        # [0, 1)
    # Distance to the chosen half-integer, and which way the runner-up lies.
    d0 = (resid - 0.5).abs()
    # Built from `resid` so the dtype follows the input: a Python-float `where`
    # would silently produce float32 and break the scatter under float64.
    ones = torch.ones_like(resid)
    step = torch.where(resid > 0.5, ones, -ones)

    odd = (floor.sum(dim=-1) % 2) != 0                       # sum(h) = sum(floor) + 4
    worst = d0.argmax(dim=-1, keepdim=True)
    adjust = torch.zeros_like(base)
    adjust.scatter_(-1, worst, step.gather(-1, worst))
    return torch.where(odd.unsqueeze(-1), base + adjust, base)


def _lattice_shift(x: Tensor, table: Tensor, powers: Tensor, pow2: Tensor,
                   shift: float) -> tuple[Tensor, Tensor, Tensor]:
    """One shift of the lattice decode: (index, distance, is-a-codeword).

    Split out for the same reason as `_analytic_shift`: it is a long chain of
    small elementwise steps writing [n, 8] intermediates, which is what a fused
    backend removes.  Measured 2.3x compiled, output bit-identical.
    """
    h = _nearest_halfinteger_even(x - shift)

    # Levels outside 0..2 cannot be codewords; clamp so the gather is safe and
    # let the membership test reject them.
    level = (h.abs() - 0.5).round().to(torch.int64)
    in_range = (level >= 0).all(dim=-1) & (level < _LEVELS).all(dim=-1)
    key = (level.clamp(0, _LEVELS - 1) * powers).sum(dim=-1)
    src = torch.where(in_range, table[key], torch.full_like(key, -1))

    sign_idx = ((h[:, : E8P_DIM - 1] < 0).to(torch.int64) * pow2).sum(dim=-1)
    idx = src.clamp_min(0) * 128 + sign_idx
    d = (x - (h + shift)).square().sum(dim=-1)
    return idx, d, src >= 0


def _lattice_kernel(device: torch.device, dtype: torch.dtype):
    """`_lattice_shift`, compiled where the backend allows.  See
    `_shift_kernel` -- same probe, same fallback, same guarantee."""
    key = ("lattice", device.type, dtype)
    if key not in _SHIFT_KERNEL:
        fn = _lattice_shift
        if not os.environ.get(_NO_COMPILE_ENV):
            try:
                candidate = torch.compile(_lattice_shift, dynamic=True)
                candidate(
                    torch.zeros(E8P_DIM, E8P_DIM, dtype=dtype, device=device),
                    _table_on_device(str(device)),
                    (_LEVELS ** torch.arange(E8P_DIM, device=device)).to(torch.int64),
                    (2 ** torch.arange(E8P_DIM - 1, device=device)).to(torch.int64),
                    0.25,
                )
                fn = candidate
            except Exception:
                fn = _lattice_shift
        _SHIFT_KERNEL[key] = fn
    return _SHIFT_KERNEL[key]


def nearest_e8p(x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Nearest E8P codeword by lattice decoding.  (index, codeword, exact).

    Every codeword is `h + s` with `h` in D8 + 1/2 and `s` either +1/4 or -1/4
    on every coordinate, so the nearest codeword can be found by decoding twice
    instead of comparing against 65536 rows.

    `exact` marks the rows where that is PROVEN, and the proof is exactly this:
    the codebook is contained in the union of the two shifted lattices, so the
    nearest point of that union is a lower bound on the distance to any
    codeword.  If that point happens to BE a codeword, it is the nearest one.
    If it is not -- and the codebook is a lattice truncated to a norm ball plus
    29 arbitrary padding patterns, so misses are common -- the true answer may
    be a point this never visited, and the caller has to fall back.

    Requiring both shifts to land on members would also be sound but is far too
    strict: a codeword decodes to itself under its own shift at distance zero,
    which settles the row no matter what the other shift does.

    For rows where `exact` is False the returned index and codeword are
    meaningless placeholders, not a best effort.
    """
    if x.shape[-1] != E8P_DIM:
        raise ValueError(f"last dim must be {E8P_DIM}, got {x.shape[-1]}")

    table = _table_on_device(str(x.device))
    powers = (_LEVELS ** torch.arange(E8P_DIM, device=x.device)).to(torch.int64)
    pow2 = (2 ** torch.arange(E8P_DIM - 1, device=x.device)).to(torch.int64)
    kernel = _lattice_kernel(x.device, x.dtype)

    best_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    best_d = torch.full((x.shape[0],), float("inf"), dtype=x.dtype, device=x.device)
    exact = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    for shift_bit, shift in enumerate((0.25, -0.25)):
        idx, d, member = kernel(x, table, powers, pow2, shift)
        idx = idx + shift_bit * 32768
        # Track the nearest point of the UNION, member or not, and carry its
        # membership along: that is what decides whether the row is settled.
        take = d < best_d
        best_idx = torch.where(take, idx, best_idx)
        best_d = torch.where(take, d, best_d)
        exact = torch.where(take, member, exact)

    return best_idx, _on_device(x.dtype, str(x.device))[best_idx], exact


@lru_cache(maxsize=16)
def _source_cached(dtype: torch.dtype, device_key: str) -> Tensor:
    return source_codebook(dtype).to(device_key)


def _source_on_device(dtype: torch.dtype, device: torch.device | str) -> Tensor:
    return _source_cached(dtype, _device_key(device))


def _analytic_shift(z: Tensor, St: Tensor, s_norm2: Tensor,
                    pow2: Tensor) -> tuple[Tensor, Tensor]:
    """One shift of the analytic search: (distance, index within the shift).

    Kept as a free function taking everything it needs, with no cached lookups
    and no data-dependent control flow, because it is the piece `torch.compile`
    fuses.  Eager, it is about forty small kernels writing [m, 256] and
    [m, 8, 256] intermediates to global memory; fused, those stay in registers
    and the launches collapse into one.  Measured 5.9-6.6x with Triton, output
    bit-identical.
    """
    az = z.abs()
    neg = z < 0
    sgn = torch.where(neg, -torch.ones_like(z), torch.ones_like(z))

    gain = az @ St                                       # [m, 256]
    # `sum_i sign(z_i) p_i` is an integer; odd means not a codeword.
    odd = torch.remainder((sgn @ St).round(), 2.0) != 0

    # Cost of the cheapest repair flip, and which coordinate it is.
    per_coord = az.unsqueeze(2) * St.unsqueeze(0)        # [m, 8, 256]
    head_cost, head_arg = per_coord[:, :E8P_DIM - 1, :].min(dim=1)
    last_cost = per_coord[:, E8P_DIM - 1, :]
    # `<=` prefers coordinate eight: flipping it sets no stored bit, so it is
    # the lower index, which is what a scan's argmin would pick.
    use_last = last_cost <= head_cost
    penalty = 2.0 * torch.where(use_last, last_cost, head_cost)

    adjusted = gain - torch.where(odd, penalty, torch.zeros_like(penalty))
    d = (z.square().sum(dim=1, keepdim=True) - 2.0 * adjusted
         + s_norm2.unsqueeze(0))                         # [m, 256]
    d_min, src = d.min(dim=1)                            # lowest src on ties

    base = (neg[:, :E8P_DIM - 1].to(torch.int64) * pow2).sum(dim=1)
    flipped = odd.gather(1, src.unsqueeze(1)).squeeze(1) & ~(
        use_last.gather(1, src.unsqueeze(1)).squeeze(1))
    j = head_arg.gather(1, src.unsqueeze(1)).squeeze(1)
    return d_min, src * 128 + torch.where(flipped, base ^ pow2[j], base)


#: Set to anything to keep `_analytic_shift` in eager mode.  There to make a
#: compiled/uncompiled comparison a one-liner, and to have an escape hatch if a
#: toolchain ever miscompiles it.
_NO_COMPILE_ENV = "TILESPARSE_NO_COMPILE"

_SHIFT_KERNEL: dict = {}


def _shift_kernel(device: torch.device, dtype: torch.dtype):
    """`_analytic_shift`, compiled if this machine can and eager if not.

    `dynamic=True` matters: the row count is the number of rows the lattice
    decoder could not settle, which changes call to call.  Compiled for static
    shapes it would recompile on every new one at several seconds each; dynamic,
    it compiles once and handles every size -- measured with zero recompiles
    across five row counts spanning 64x.

    The compile is FORCED here, on a token input, rather than left to happen
    inside a real call.  Inductor is lazy, so a missing backend surfaces the
    first time the function actually runs, and on this machine that is exactly
    what happens: CUDA compiles through Triton, CPU asks for `cl` and does not
    find it.  Probing per (device, dtype) keeps that failure a startup detail
    instead of a crash halfway through a layer.

    Falling back is not a degraded mode.  Eager and compiled are bit-identical
    -- the tests require it -- so this only ever changes how long a run takes.
    """
    key = (device.type, dtype)
    if key not in _SHIFT_KERNEL:
        fn = _analytic_shift
        if not os.environ.get(_NO_COMPILE_ENV):
            try:
                candidate = torch.compile(_analytic_shift, dynamic=True)
                S = _source_on_device(dtype, str(device))
                candidate(
                    torch.zeros(E8P_DIM, E8P_DIM, dtype=dtype, device=device),
                    S.T.contiguous(), S.square().sum(dim=1),
                    (2 ** torch.arange(E8P_DIM - 1, device=device)).to(torch.int64),
                )
                fn = candidate
            except Exception:
                fn = _analytic_shift
        _SHIFT_KERNEL[key] = fn
    return _SHIFT_KERNEL[key]


#: Rows per pass in `nearest_e8p_analytic`.  Larger than the scan's 4096 on
#: purpose: the analytic form is launch-bound, not memory-bound, so halving the
#: number of passes is worth more than the working set it costs.  Measured
#: 1.10-1.24x end to end going from 4096 to this; past it the curve is flat.
#: The working set is `chunk * 8 * 256 * itemsize`, 128 MiB here.
ANALYTIC_CHUNK = 16384


def nearest_e8p_analytic(x: Tensor,
                         chunk: int = ANALYTIC_CHUNK) -> tuple[Tensor, Tensor]:
    """Nearest E8P codeword, EXACTLY, without scanning 65536 rows.

    The scan was the pipeline's dominant cost and it was never necessary.  A
    codeword is `sigma * p + s`: `p` one of 256 source patterns, `s` either
    +1/4 or -1/4, `sigma` free on the first seven coordinates with the eighth
    set so the coordinate sum is even (`e8p_codebook`).  The 128 sign choices
    are therefore not a search space at all:

    `p` is NON-NEGATIVE, so for a fixed pattern the inner product
    `<z, sigma*p> = sum_i sigma_i z_i p_i` is maximized coordinate by
    coordinate at `sigma_i = sign(z_i)`.  If that assignment has an odd
    coordinate sum it is not in the codebook -- and since every coordinate is a
    half-integer, flipping ANY single sign changes the sum by an odd number and
    so flips the parity.  The repair is therefore one flip, the cheapest one,
    costing `2 |z_i| p_i`.

    So the optimum over 2^16 codewords is: one matmul against 256 patterns, one
    parity test, one min over eight coordinates.  Measured 8-19x faster than
    the scan, with distances identical to it on every row tried.

    This supersedes `nearest_e8p`, which decoded the lattice and could only
    PROVE its answer for the rows that landed on a codebook member -- 0.7% of
    them at the small end of `fit_scale`'s sweep, where the rest fell back to
    the full scan and cost 88% of the fit.  There is no fallback here: every
    row is settled.

    Ties are broken to match a scan's `argmin`, which takes the lowest index:
    the lowest source pattern, and among equal-cost flips the one that leaves
    the sign field smallest -- coordinate eight first, since flipping it sets no
    stored bit, then the lowest coordinate.
    """
    if x.ndim != 2 or x.shape[1] != E8P_DIM:
        raise ValueError(f"x must be [n, {E8P_DIM}], got {tuple(x.shape)}")

    device = str(x.device)
    S = _source_on_device(x.dtype, device)                   # [256, 8] >= 0
    St = S.T.contiguous()                                    # [8, 256]
    s_norm2 = S.square().sum(dim=1)                          # [256]
    pow2 = (2 ** torch.arange(E8P_DIM - 1, device=x.device)).to(torch.int64)

    kernel = _shift_kernel(x.device, x.dtype)

    n = x.shape[0]
    out_idx = torch.empty(n, dtype=torch.long, device=x.device)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        xc = x[lo:hi]
        best_d = torch.full((hi - lo,), float("inf"), dtype=x.dtype,
                            device=x.device)
        best_i = torch.zeros(hi - lo, dtype=torch.long, device=x.device)

        for shift_bit, shift in enumerate((0.25, -0.25)):
            d_min, idx = kernel(xc - shift, St, s_norm2, pow2)
            idx = idx + shift_bit * 32768
            take = d_min < best_d
            best_d = torch.where(take, d_min, best_d)
            best_i = torch.where(take, idx, best_i)
        out_idx[lo:hi] = best_i

    return out_idx, _on_device(x.dtype, device)[out_idx]


def _nearest(x: Tensor, codebook: Tensor, chunk: int = 4096,
             search_dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
    """Nearest codeword for each row of `x` [n, 8].  Returns (index, codeword).

    `search_dtype` runs the SEARCH in a narrower type and still gathers the
    codeword from the caller's, so the returned values keep full precision and
    only the choice is made in low precision.  Measured on 262,144 vectors,
    float16 picks a different codeword for 0.393% of rows and costs 0.0012% of
    total squared error -- those rows are genuine near-ties, and the choice is
    never BETTER than the full-precision one, which is what a near-tie looks
    like.  It buys 1.8x, and it buys it on `fit_scale` too, which is where a
    tile now spends its time.

    Unlike sampling the scale fit, this adds no NOISE: it is deterministic, so
    it shifts a number rather than widening it.  That distinction is what makes
    it acceptable and sampling not (`experiments/m0_scale_fit.py`).

    Brute force over 65536 codewords, chunked over `x`.  ||x-c||^2 expands to
    ||c||^2 - 2 x.c (the ||x||^2 term does not affect the argmin).

    For the canonical E8P table this defers to `nearest_e8p`, which decodes the
    lattice instead of scanning it, and only scans the rows the decoder could
    not settle.  That search WAS the pipeline's dominant cost -- 79% of a GPU
    pass -- and it is no longer: the scale fit is 28% of a tile since its
    candidates were batched, and the largest term is now the sub-Hessian
    rotation (`experiments/m0_cost_model.py`).
    """
    if search_dtype is not None and search_dtype != x.dtype:
        idx, _ = _nearest(x.to(search_dtype),
                          _on_device(search_dtype, str(x.device)), chunk)
        return idx, codebook[idx]

    floor_rows = _LATTICE_MIN_ROWS.get(x.device.type, 64)
    if is_canonical_codebook(codebook) and floor_rows > x.shape[0] >= \
            _ANALYTIC_DIRECT_MIN_ROWS:
        # Too few rows for the lattice decoder to be worth its fixed cost, but
        # enough for the analytic form to beat a scan.  That window went to the
        # scan until 2026-08-24 because `_ANALYTIC_MIN_ROWS` was only ever read
        # INSIDE the decoder's gate, so a row count between the two thresholds
        # could not reach the analytic search at all.
        #
        # It is not a corner: the LDLQ sweep hands `_nearest`
        # `chunk * lines_per_tile` rows, which is 512 at T=1 and T=2 and 816 at
        # T=4 -- the whole fine end of the grid, where the tile counts are
        # largest.  Ten of the twenty-one layer-by-tile cells at B=1.5 landed in
        # it.  Same shape as the `_on_device` bug: a gate calibrated for one
        # algorithm silently excluding the better one that arrived later, and
        # the symptom is not a wrong answer but a slow one.
        return nearest_e8p_analytic(x)

    if x.shape[0] >= floor_rows and is_canonical_codebook(codebook):
        idx, code, exact = nearest_e8p(x)
        if bool(exact.all()):
            return idx, code
        miss = (~exact).nonzero(as_tuple=True)[0]
        # The rows the decoder could not settle go to `nearest_e8p_analytic`,
        # not to a scan.  Both are exact; the analytic one is 8-19x cheaper,
        # and this is where nearly all of the pipeline's time used to go --
        # `fit_scale`'s small-scale steps miss on 99% of rows.
        #
        # The decoder stays in front of it because when it DOES settle a row it
        # is cheaper still: measured, it takes the same 0.2 ms for 8K rows as
        # for 80K, being launch-bound rather than compute-bound, while the
        # analytic form does real work proportional to the rows it is given.
        m_idx, m_code = (
            nearest_e8p_analytic(x[miss])
            if miss.numel() >= _ANALYTIC_MIN_ROWS
            else _brute_force(x[miss], codebook, chunk))
        idx = idx.clone()
        idx[miss] = m_idx
        code = code.clone()
        code[miss] = m_code
        return idx, code
    return _brute_force(x, codebook, chunk)


def _brute_force(x: Tensor, codebook: Tensor, chunk: int = 4096
                 ) -> tuple[Tensor, Tensor]:
    c_sq = codebook.square().sum(dim=1)
    idx = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for lo in range(0, x.shape[0], chunk):
        hi = min(lo + chunk, x.shape[0])
        d = c_sq.unsqueeze(0) - 2.0 * (x[lo:hi] @ codebook.T)
        idx[lo:hi] = d.argmin(dim=1)
    return idx, codebook[idx]


#: Candidate scales `fit_scale` tries.  Never questioned: the objective is not
#: convex in alpha so a search beats a closed form, but 24 uniform steps is a
#: guess, and each one is a full pass over the tile.  Measured, six steps land
#: within 1.4% of what 24 finds -- whether that 1.4% costs anything is what
#: `experiments/m0_scale_fit.py` is for.
FIT_STEPS = 24

#: Rows one batched pass of `fit_scale` may hand `_nearest`.  The candidates are
#: split into groups of `FIT_ROW_BUDGET // len(x)` so a tile with many vectors
#: does not materialize `n_steps` copies of itself at once; the peak is two
#: tensors of this many rows, 64 MiB together at float32.
#:
#: Measured, the gain is already saturated well below it -- a 5,888-vector tile
#: sweeps all 24 candidates in 141,312 rows -- so the budget only ever bites at
#: `T=max`, where the fit was the cheapest column to begin with.
FIT_ROW_BUDGET = 1 << 20


def fit_scale(
    x: Tensor, codebook: Tensor, n_steps: int = FIT_STEPS,
    lo: float = 0.4, hi: float = 2.0,
    sample: int | None = None, seed_rng: int = 0,
    search_dtype: torch.dtype | None = None,
) -> float:
    """Scale alpha minimizing ||x - alpha * Q(x/alpha)||^2.

    Seeded by matching RMS to the codebook's, then refined by a coarse sweep --
    the objective is not convex in alpha, so a search beats a closed form.

    The candidates are evaluated TOGETHER, in one nearest-codeword call per
    group rather than one per candidate.  They are independent -- each asks what
    a different scaling of the same vectors rounds to -- so stacking them is
    only a rearrangement, and the search is launch-bound rather than
    compute-bound: measured, 1,280 vectors cost 41.3 ms and 5,888 cost 43.4 ms,
    4.6x the work for 1.05x the time.  Twenty-four separate passes therefore
    paid the fixed cost twenty-four times.  Measured end to end on
    `ldlq_quantize_blocks`, against the same code with `FIT_ROW_BUDGET = 1`:
    3.78x at four lines, 2.01x at sixteen, 1.09x at 128, output bit-identical.

    This is the same lever as chunking the sweep across tiles, one level up, and
    it is NOT the rejected one.  Batching the fit ACROSS TILES was measured at
    2.16x and turned down because it reduces every tile's error together and so
    changes the arithmetic (`docs/STATUS.md` section 7.2).  Batching across
    CANDIDATES leaves each candidate's error on its own [n, 8] tensor, summed in
    the same order as before, and `tests/test_quantize.py` requires the alpha to
    come out identical.

    `sample` caps how many vectors the sweep looks at.  Alpha is one scalar;
    estimating it from thousands of 8-dimensional vectors is already far past
    the point of diminishing returns, and the vectors not sampled are still
    quantized with the result.
    """
    if sample is not None and sample < x.shape[0]:
        g = torch.Generator(device="cpu").manual_seed(seed_rng)
        idx = torch.randperm(x.shape[0], generator=g)[:sample].to(x.device)
        x = x[idx]

    rms_x = float(x.square().mean().sqrt())
    rms_c = float(codebook.square().mean().sqrt())
    if rms_x == 0.0:
        return 1.0
    seed = rms_x / rms_c

    alphas = [seed * f for f in torch.linspace(lo, hi, n_steps).tolist()]
    n, width = x.shape[0], x.shape[1]
    per_pass = max(1, FIT_ROW_BUDGET // max(n, 1))

    best, best_err = seed, float("inf")
    for start in range(0, len(alphas), per_pass):
        group = alphas[start:start + per_pass]
        # Divided one candidate at a time, by the same Python float the
        # unbatched form used, so the search sees exactly the same numbers.
        scaled = torch.empty((len(group), n, width), dtype=x.dtype,
                             device=x.device)
        for i, a in enumerate(group):
            torch.div(x, a, out=scaled[i])
        _, q = _nearest(scaled.reshape(-1, width), codebook,
                        search_dtype=search_dtype)
        q = q.reshape(len(group), n, width)
        # One reduction per candidate over its own [n, width], never a single
        # reduction across the stack.  This is insurance rather than a measured
        # need: a joint reduction sums the same terms in a different order, and
        # over 40 float32 draws at n = 1,280 / 5,888 / 49,152 it never moved the
        # argmin, the two error vectors agreeing to 1.1e-07 relative.  It is
        # kept because it costs 24 tiny reductions and makes "summed in the same
        # order as the unbatched form" exactly true rather than nearly true.
        errs = torch.stack([(x - group[i] * q[i]).square().sum()
                            for i in range(len(group))]).tolist()
        for a, err in zip(group, errs):
            if err < best_err:
                best, best_err = a, err
    return best


def fit_scales(
    x: Tensor, codebook: Tensor, n_steps: int = FIT_STEPS,
    lo: float = 0.4, hi: float = 2.0,
    sample: int | None = None, seed_rng: int = 0,
    search_dtype: torch.dtype | None = None,
) -> list[float]:
    """`fit_scale` for a stack of tiles, batched ACROSS them.

    `x` is [n_tiles, n, 8] and the result is one alpha per tile -- the same
    quantities `fit_scale` returns one call at a time, computed with the tiles'
    candidate passes packed into shared `_nearest` calls.

    WHY THIS IS A DIFFERENT LEVER FROM THE ONE ALREADY TAKEN.  `fit_scale`
    batches across CANDIDATES, which fills a pass at the coarse end -- a
    5,888-vector tile already hands `_nearest` 141,312 rows and is nowhere near
    launch-bound.  At the FINE end it does not: a T=1 tile at k=1024 holds 128
    vectors, so all 24 candidates together are 3,072 rows, and there are 4,096
    such tiles in a layer.  That is 4,096 sequential calls of a size the card
    finishes before it has filled.  Packing tiles into the same pass is the only
    thing left that changes it, and the fine end is where the grid is expensive
    (`docs/STATUS.md` section 6.14: T=1 is the costliest cell, not the cheapest).

    WHAT IS PRESERVED, DELIBERATELY.  Every tile keeps its OWN alpha, its own
    seed from its own RMS, and its own error reduction over its own [n, 8] --
    the same terms in the same order as the unbatched form.  `docs/STATUS.md`
    section 7.2 turned this lever down as "not bit-identical, it reduces every
    tile's error together", and that describes an implementation which shares
    the reduction.  This one does not, so whether the output moves at all is a
    question to measure rather than to assume -- which is the whole reason the
    rejection was worth re-opening (`experiments/m0_fit_batch.py`).

    Not to be confused with `scale="per_layer"`, which shares one alpha across
    tiles and was measured 11% worse (2026-08-23).  Sharing the WORK is not
    sharing the ANSWER.

    `FIT_ROW_BUDGET` still caps a pass, now over (tile, candidate) slots rather
    than candidates alone.  That is what keeps the peak bounded at the coarse
    end -- and it is also why the gain lands at the fine end and nowhere else:
    a coarse tile fills the budget by itself, so there is no room to pack.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [n_tiles, n, width], got {tuple(x.shape)}")
    n_tiles, n, width = x.shape

    tiles: list[Tensor] = []
    for t in range(n_tiles):
        xt = x[t]
        if sample is not None and sample < n:
            # Per tile, exactly as `ldlq_quantize_blocks` seeds it: a shared
            # subset would correlate the tiles' scales in a way a full fit
            # never does.
            g = torch.Generator(device="cpu").manual_seed(seed_rng + t)
            idx = torch.randperm(n, generator=g)[:sample].to(xt.device)
            xt = xt[idx]
        tiles.append(xt)
    rows = tiles[0].shape[0] if tiles else 0

    rms_c = float(codebook.square().mean().sqrt())
    steps = torch.linspace(lo, hi, n_steps).tolist()
    best = [1.0] * n_tiles
    best_err = [float("inf")] * n_tiles
    slots: list[tuple[int, float]] = []
    for t, xt in enumerate(tiles):
        rms_x = float(xt.square().mean().sqrt())
        if rms_x == 0.0:
            continue                      # `fit_scale` returns 1.0 here
        seed = rms_x / rms_c
        best[t] = seed
        slots += [(t, seed * f) for f in steps]

    per_pass = max(1, FIT_ROW_BUDGET // max(rows, 1))
    for start in range(0, len(slots), per_pass):
        group = slots[start:start + per_pass]
        scaled = torch.empty((len(group), rows, width), dtype=x.dtype,
                             device=x.device)
        for i, (t, a) in enumerate(group):
            # Divided by the same Python float the unbatched form used, so the
            # search sees exactly the same numbers.
            torch.div(tiles[t], a, out=scaled[i])
        _, q = _nearest(scaled.reshape(-1, width), codebook,
                        search_dtype=search_dtype)
        q = q.reshape(len(group), rows, width)
        # One reduction per SLOT over its own [rows, width] -- never one across
        # the group, and never one across the tiles in it.  That is the whole
        # difference between this and the arrangement section 7.2 rejected.
        errs = torch.stack([(tiles[group[i][0]] - group[i][1] * q[i])
                            .square().sum()
                            for i in range(len(group))]).tolist()
        for (t, a), err in zip(group, errs):
            if err < best_err[t]:
                best[t], best_err[t] = a, err
    return best


def quantize_vectors(
    x: Tensor, scale: float | None = None, dtype: torch.dtype | None = None
) -> tuple[Tensor, Tensor, float]:
    """Quantize rows of `x` [n, 8].  Returns (dequantized, indices, scale)."""
    if x.ndim != 2 or x.shape[1] != E8P_DIM:
        raise ValueError(f"x must be [n, {E8P_DIM}], got {tuple(x.shape)}")
    cb = _on_device(dtype or x.dtype, str(x.device))
    a = fit_scale(x, cb) if scale is None else float(scale)
    idx, q = _nearest(x / a, cb)
    return a * q, idx, a


@dataclass(frozen=True)
class QuantizedBlocks:
    values: Tensor            # dequantized, same shape as the input blocks
    indices: Tensor           # long, one index per 8-wide group
    scales: Tensor            # one scale per tile
    padding: int              # zeros appended to reach a multiple of 8

    @property
    def bits_per_weight(self) -> float:
        return E8P_BITS_PER_WEIGHT


def quantize_blocks(blocks: Tensor, per_tile_scale: bool = True) -> QuantizedBlocks:
    """Quantize compacted survivor blocks [n_tiles, lines_per_tile, k].

    Vectors are formed along the INDEX axis -- eight consecutive survivors of one
    line -- because that is the axis a rotation mixes and the axis whose
    covariance the Hessian describes.  `k` is zero-padded up to a multiple of 8;
    the padding is dropped on the way out.
    """
    if blocks.ndim != 3:
        raise ValueError(f"blocks must be 3-D, got {tuple(blocks.shape)}")
    n_tiles, lpt, k = blocks.shape

    pad = (-k) % E8P_DIM
    x = blocks
    if pad:
        x = torch.cat(
            [x, torch.zeros((n_tiles, lpt, pad), dtype=x.dtype, device=x.device)],
            dim=2,
        )

    cb = _on_device(x.dtype, str(x.device))
    out = torch.empty_like(x)
    idx_all, scales = [], []
    for t in range(n_tiles):
        v = x[t].reshape(-1, E8P_DIM)
        a = fit_scale(v, cb) if per_tile_scale else 1.0
        i, q = _nearest(v / a, cb)
        out[t] = (a * q).reshape(lpt, -1)
        idx_all.append(i)
        scales.append(a)

    values = out[:, :, :k] if pad else out
    return QuantizedBlocks(
        values=values.contiguous(),
        indices=torch.stack(idx_all),
        scales=torch.tensor(scales, dtype=blocks.dtype),
        padding=pad,
    )


@dataclass(frozen=True)
class LDLQResult:
    values: Tensor            # [n_lines, k], dequantized
    indices: Tensor           # long, one index per 8-wide group
    scale: float


def _upper_inverse_factor(Hd: Tensor) -> Tensor:
    """chol(inv(Hd), upper) -- the feedback matrix LDLQ sweeps against.

    Batched: `Hd` may be [k, k] or [m, k, k], and the [m, ...] form is what
    makes a block-diagonal Hessian cheap, since m small factorizations issue as
    one kernel instead of m.
    """
    return torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(Hd)), upper=True)


def _partition(k: int, block: int | None, group: int) -> list[tuple[int, int]]:
    """Consecutive chunks of at most `block` coordinates; `None` means one chunk.

    Every boundary must fall between E8P groups: eight coordinates quantized as
    one codeword cannot draw their feedback from two different factorizations.
    """
    if block is None or block >= k:
        return [(0, k)]
    if block % group:
        raise ValueError(
            f"hessian_block must be a multiple of the quantizer group {group} "
            f"so no block boundary falls inside a codeword, got {block}"
        )
    return [(o, min(block, k - o)) for o in range(0, k, block)]


def _tile_factors(H: Tensor, percdamp: float, parts) -> list[Tensor]:
    """chol(inv(H_part + lambda I), upper) for one tile, one entry per part.

    Dropping the couplings that reach past a part is exactly equivalent to
    running LDLQ independently on each part: the sweep's feedback never reaches
    outside the part it is in.  Cost goes from k^3 to sum(width^3).

    The damping comes from the WHOLE diagonal whatever the partition, so a
    blocked run and a full-width run regularize identically and any difference
    between them is attributable to the dropped couplings alone.
    """
    damp = percdamp * torch.diagonal(H).mean()
    by_width: dict[int, list[tuple[int, int]]] = {}
    for i, (off, width) in enumerate(parts):
        by_width.setdefault(width, []).append((i, off))

    out: list[Tensor | None] = [None] * len(parts)
    for width, items in by_width.items():
        eye = torch.eye(width, dtype=H.dtype, device=H.device)
        stacked = torch.stack([H[o:o + width, o:o + width] for _, o in items])
        factors = _upper_inverse_factor(stacked + damp * eye)
        for n, (i, _) in enumerate(items):
            out[i] = factors[n]
    return out                                                    # type: ignore[return-value]


def _ldlq_sweep(W: Tensor, factors: list[Tensor], parts, alpha: Tensor,
                codebook: Tensor, group: int,
                search_dtype: torch.dtype | None = None
                ) -> tuple[Tensor, Tensor]:
    """The sweep itself, over C tiles at once.  `W` [C, lines, k] is consumed.

    Why C tiles and not one.  Tiles are independent given their own Hessians, so
    the group loop can be hoisted out of the tile loop: at each group every tile
    in the chunk is quantized together.  That matters because the sweep is not
    compute-bound -- measured on this machine, a group costs 0.248 ms of wall
    time against 0.0034 ms of arithmetic, so 99.6% of it is kernel launch.
    Batching C tiles hands `_nearest` C*lines rows instead of `lines`, which
    crosses the threshold where the lattice decoder takes over and the card
    fills.

    The arithmetic per tile is untouched -- same feedback matrix, same alpha,
    same sequential group order -- so the output must be identical to running
    the tiles one at a time, and `tests/test_quantize.py` requires exactly that.
    """
    C, lines, _ = W.shape
    out = torch.empty_like(W)
    a = alpha.reshape(C, 1, 1)
    per_group = []
    for part, (off, width) in enumerate(parts):
        U = factors[part]                                    # [C, width, width]
        for jj in range(0, width, group):
            j = off + jj
            g = slice(j, j + group)
            Wg = W[:, :, g]
            idx, q = _nearest((Wg / a).reshape(-1, group), codebook,
                              search_dtype=search_dtype)
            Qg = q.reshape(C, lines, group) * a
            out[:, :, g] = Qg
            per_group.append(idx.reshape(C, lines))

            # err = (Wg - Qg) inv(U[g, g]), via a triangular solve.
            Ugg = U[:, jj:jj + group, jj:jj + group]
            err = torch.linalg.solve_triangular(
                Ugg.transpose(-1, -2), (Wg - Qg).transpose(-1, -2),
                upper=False).transpose(-1, -2)               # [C, lines, group]
            if jj + group < width:
                W[:, :, j + group:off + width] -= (
                    err @ U[:, jj:jj + group, jj + group:width])
    return out, torch.stack(per_group, dim=1).reshape(C, -1)


#: Memory the chunked sweep may spend on feedback matrices, in bytes.  One GiB
#: is a judgement: the card has 8 and the compressed layer, its sub-Hessian and
#: the activations all want room too.
CHUNK_BUDGET_BYTES = 1 << 30

#: Rows past which `_nearest` stops getting faster.
#:
#: 1024 WAS STALE, AND STALE IN THE WORST PLACE.  It was measured before the
#: analytic search, before `fit_scale` batched its candidates and before Triton,
#: and it priced SATURATION only -- it never priced which search path the row
#: count selects.  Re-measured at the grid's real shapes and tile counts, with
#: the 1024 arm as the base:
#:
#:      shape (n_tiles)       1024    2048    3072    4096    8192   binds
#:      T=8  k=2816 (512)    1.00x   1.16x   1.17x   1.17x   1.19x   memory
#:      T=16 k=2944 (256)    1.00x   1.37x   1.33x   1.35x   1.35x   memory
#:      T=32 k=3008 (128)    1.00x   1.49x   1.53x   1.96x   1.91x   tiles
#:      T=16 k=7912 (256)    1.00x   1.08x   1.04x   1.05x   1.07x   memory
#:      total                1.00x   1.20x   1.19x   1.23x   1.24x
#:
#: 2048 rather than more because the curve is flat past it: 1.20 against 1.24 at
#: four times the target, inside the 2-5% these timings spread, and what little
#: is left comes from one cell where the TILE COUNT binds -- T=32 fitting a
#: whole layer in one chunk, which is not a row-target effect at all.  Past 2048
#: three of the four shapes are held by `CHUNK_BUDGET_BYTES` anyway, so raising
#: it further mostly moves the control to the memory ceiling without moving the
#: clock.  Peak allocation over the sweep reached 1.7-3.7 GiB, on a card with 8
#: that also has to hold the layer and its activations.
#:
#: Note what the old value cost beyond saturation.  `auto_chunk`'s ceiling is
#: `ceil(target / lines)`, so at T=8, 16 and 32 -- where `lines` divides 1024 --
#: it landed on EXACTLY 1024 rows, whose 34.9% leftover (357) sat just under the
#: old `_ANALYTIC_MIN_ROWS`, and every group of the sweep fell through to a
#: 65536-codeword scan.  Eight of twenty-one cells.  The two constants have to
#: be read together, and `tests/test_quantize.py` now asserts the inequality.
CHUNK_TARGET_ROWS = 2048


def auto_chunk(n_tiles: int, lines_per_tile: int, k: int, itemsize: int,
               hessian_block: int | None = None,
               budget_bytes: int = CHUNK_BUDGET_BYTES) -> int:
    """How many tiles to sweep together, from memory and from saturation.

    Two ceilings, and the binding one is usually memory.  A chunk holds every
    member's feedback matrix: k*block per tile when the feedback is confined,
    k^2 when it is not.  At k=7912 that is 16 MiB against 250 MiB, which is why
    `hessian_block` is what makes a useful chunk affordable at all.

    The other ceiling is that `_nearest` stops improving somewhere above a
    thousand rows, so a larger chunk past that spends memory for nothing.
    """
    parts = _partition(k, hessian_block, E8P_DIM)
    per_tile = sum(width * width for _, width in parts) * itemsize
    by_memory = max(1, budget_bytes // max(per_tile, 1))
    by_saturation = max(1, -(-CHUNK_TARGET_ROWS // max(lines_per_tile, 1)))
    return int(min(n_tiles, by_memory, by_saturation))


def ldlq_quantize(
    block: Tensor,
    H: Tensor,
    *,
    percdamp: float = 0.01,
    scale: float | None = None,
    group: int = E8P_DIM,
    hessian_block: int | None = None,
    scale_sample: int | None = None,
    scale_steps: int = FIT_STEPS,
    scale_seed: int = 0,
    search_dtype: torch.dtype | None = None,
) -> LDLQResult:
    """Hessian-aware rounding: LDLQ / block-GPTQ with a vector quantizer.

    Plain nearest-neighbour minimizes ||W - W_hat||^2.  The objective that
    matters is tr(E H E^T) -- error is cheap in directions the activations
    rarely visit and expensive in the ones they do.  LDLQ sweeps the index axis
    once, quantizing eight coordinates at a time and pushing each group's error
    onto the coordinates not yet visited, weighted by the Hessian.

    This is what makes ROTATION pay.  An RHT deliberately makes quantization
    error isotropic, which is the wrong shape unless the Hessian is isotropic
    too -- so rotating without Hessian-aware rounding costs inference time and
    buys nothing (see plan section I3).  Rotate the block by V, rotate the
    sub-Hessian to V H V^T, and the objective is preserved exactly:

        E_rot = E V^T,  H_rot = V H V^T  =>  tr(E_rot H_rot E_rot^T) = tr(E H E^T)

    `H` must already be in the same basis as `block`.

    The update generalizes GPTQ's scalar rule to a group of `group` columns:

        err = (W_g - Q_g) inv(U_gg),   W_[after] -= err U_[g, after]

    with U the upper Cholesky factor of (H + lambda I)^-1.  At group=1 it
    reduces exactly to the per-column rule in `prune.forward_compensate`.

    `hessian_block=b` keeps only the width-b diagonal blocks of that factor,
    turning the k^3 factorization into sum(b^3).  It is also what makes the
    batched path affordable: a block-diagonal factor is k*b per tile instead of
    k^2, so a chunk of tiles fits in memory (`ldlq_quantize_blocks(chunk=...)`).

    `scale_sample` and `scale_steps` are the two ways to make the scale fit
    cheaper, and they multiply.  The fit scans the tile `scale_steps` times to
    find ONE scalar, which after the sweep was chunked is most of what a tile
    costs; `scale_sample` caps how many of the tile's vectors each pass looks
    at.  Both default to the full-cost behaviour, because that is the
    arrangement every quality number so far was measured under.

    One tile is the C=1 case of `_ldlq_sweep`, deliberately -- a second
    implementation of this arithmetic would be free to drift from the first.
    """
    if block.ndim != 2:
        raise ValueError(f"block must be 2-D, got {tuple(block.shape)}")
    n_lines, k = block.shape
    if k % group:
        raise ValueError(
            f"LDLQ needs the index axis to be a multiple of {group}, got k={k}. "
            "Align the survivor count (tiling.uniform_survivor_count(align=8)); "
            "tensor cores want that alignment anyway."
        )
    if H.shape != (k, k):
        raise ValueError(f"H must be ({k}, {k}) to match the block, got {tuple(H.shape)}")

    parts = _partition(k, hessian_block, group)
    cb = _on_device(block.dtype, str(block.device))
    a = (float(scale) if scale is not None else
         fit_scale(block.reshape(-1, group), cb, n_steps=scale_steps,
                   sample=scale_sample, seed_rng=scale_seed,
                   search_dtype=search_dtype))
    factors = [f.unsqueeze(0) for f in _tile_factors(H, percdamp, parts)]
    alpha = torch.tensor([a], dtype=block.dtype, device=block.device)
    values, idx = _ldlq_sweep(block.unsqueeze(0).clone(), factors, parts,
                              alpha, cb, group, search_dtype)
    return LDLQResult(values=values[0], indices=idx[0], scale=a)


def ldlq_quantize_blocks(
    blocks: Tensor,
    hessians: Tensor | Callable[[int], Tensor],
    *,
    percdamp: float = 0.01,
    scale: str | float = "per_tile",
    layer_scale_sample: int = 8192,
    scale_sample: int | None = None,
    scale_steps: int = FIT_STEPS,
    scale_seed: int = 0,
    search_dtype: torch.dtype | None = None,
    hessian_block: int | None = None,
    chunk: int = 1,
    batch_fit: bool = False,
) -> QuantizedBlocks:
    """`ldlq_quantize` over every tile.

    `hessians` is either a [n_tiles, k, k] tensor -- each entry the tile's
    sub-Hessian in the SAME basis as its block -- or a callable returning one
    tile's [k, k] on demand.

    Prefer the callable at real widths.  The tiles are consumed strictly one at
    a time, so materializing all of them costs `n_tiles` times more memory for
    no benefit, and `n_tiles` is in the hundreds: a Llama-2-7B `down_proj` at
    T=16 wants 119 GiB as a single tensor and 239 MiB one tile at a time.  See
    `experiments/m0_cost_model.py`.

    `scale` decides where alpha comes from:

      "per_tile"   fit it inside every tile -- what the pipeline has always
                   done.  It was 83% of a tile and is 28% since `fit_scale`
                   batched its candidates, which is why the alternatives below
                   no longer have a cost case
      "per_layer"  fit it once from a sample of `layer_scale_sample` vectors
                   drawn across all tiles, then use it everywhere.  This is what
                   QuIP# does, and it is the cheapest large saving available:
                   the sweep stops scaling with the layer.
      a float      use exactly this, fit nothing

    Fitting once over EVERY vector would save nothing at all -- same total work,
    differently arranged -- so the saving is in the sampling, not in the sharing.

    `scale_seed` offsets the sampling RNG PER TILE (`scale_seed + t`), so two
    tiles never draw the same subset -- a shared subset would correlate their
    scales in a way a full fit never would.

    `scale_sample` and `scale_steps` cap the PER-TILE fit and are the levers
    that matter now.  Note they are not interchangeable with `per_layer`: that
    one was measured 11% worse and rejected (2026-08-23), while sampling keeps a
    scale per tile and only estimates it from fewer vectors.  Both default to
    the full-cost behaviour.

    `hessian_block` is passed straight through; it is the other half of the same
    runtime question, and the two levers are independent -- one shrinks the
    factorization, the other the codebook sweep.

    `chunk` sweeps that many tiles TOGETHER.  The sweep is not compute-bound:
    measured here, a group costs 0.248 ms of wall time against 0.0034 ms of
    arithmetic, because a [lines, 8] search against 65536 codewords cannot fill
    a GPU and there are k/8 of them in a row.  Chunking hands `_nearest`
    `chunk * lines` rows instead of `lines`.

    It pairs with `hessian_block` rather than standing alone: the chunk has to
    hold every member's feedback matrix at once, which is k*block per tile when
    the feedback is confined and k^2 when it is not -- 16 MiB against 250 MiB at
    k=7912.  Sub-Hessians are still built ONE at a time whatever the chunk, so
    the streaming callable keeps doing its job.

    `chunk=1` is the default because it is the arrangement every measurement so
    far was taken under.  Larger values must produce bit-identical output.

    `batch_fit=True` fits the chunk's tiles in ONE pass instead of one apiece
    (`fit_scales`).  Off by default: `docs/STATUS.md` section 7.2 recorded this
    as measured-and-not-taken on the grounds that it is not bit-identical, and
    whether that is true of an implementation which keeps each tile's reduction
    to itself is what `experiments/m0_fit_batch.py` measures.  It is the one
    lever left on the codebook term, which after the rotation was priced
    correctly is 52% of the grid (section 6.18).
    """
    if blocks.ndim != 3:
        raise ValueError(f"blocks must be 3-D, got {tuple(blocks.shape)}")
    n_tiles, lpt, k = blocks.shape
    streaming = callable(hessians)
    if not streaming and hessians.shape != (n_tiles, k, k):
        raise ValueError(
            f"hessians must be ({n_tiles}, {k}, {k}), got {tuple(hessians.shape)}"
        )
    if scale == "per_layer":
        cb = _on_device(blocks.dtype, str(blocks.device))
        tile_scale = fit_scale(blocks.reshape(-1, E8P_DIM), cb,
                               n_steps=scale_steps, sample=layer_scale_sample,
                               search_dtype=search_dtype)
    elif scale == "per_tile":
        tile_scale = None
    elif isinstance(scale, (int, float)):
        tile_scale = float(scale)
    else:
        raise ValueError(f"scale must be 'per_tile', 'per_layer' or a number, "
                         f"got {scale!r}")

    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    if k % E8P_DIM:
        raise ValueError(
            f"LDLQ needs the index axis to be a multiple of {E8P_DIM}, got k={k}"
        )

    cb = _on_device(blocks.dtype, str(blocks.device))
    parts = _partition(k, hessian_block, E8P_DIM)
    out = torch.empty_like(blocks)
    idxs, scales = [], []

    for start in range(0, n_tiles, chunk):
        members = range(start, min(start + chunk, n_tiles))
        # One sub-Hessian resident at a time; only its factors are kept, and
        # those are k*block rather than k^2 once the feedback is confined.
        per_tile = []
        for t in members:
            h = hessians(t) if streaming else hessians[t]
            if h.shape != (k, k):
                raise ValueError(
                    f"tile {t}: hessian must be ({k}, {k}), got {tuple(h.shape)}"
                )
            per_tile.append(_tile_factors(h, percdamp, parts))
            del h
        factors = [torch.stack([f[i] for f in per_tile])
                   for i in range(len(parts))]

        if tile_scale is not None:
            alphas = [tile_scale] * len(members)
        elif batch_fit:
            # One fit for the whole chunk.  Each tile still gets its own alpha
            # and its own reduction; what is shared is the `_nearest` call.
            alphas = fit_scales(
                blocks[start:start + len(members)].reshape(
                    len(members), -1, E8P_DIM),
                cb, n_steps=scale_steps, sample=scale_sample,
                seed_rng=scale_seed + start, search_dtype=search_dtype)
        else:
            alphas = [fit_scale(blocks[t].reshape(-1, E8P_DIM), cb,
                                n_steps=scale_steps, sample=scale_sample,
                                seed_rng=scale_seed + t,
                                search_dtype=search_dtype)
                      for t in members]
        alpha = torch.tensor(alphas, dtype=blocks.dtype, device=blocks.device)
        sl = slice(start, start + len(alphas))
        values, index = _ldlq_sweep(blocks[sl].clone(), factors, parts,
                                    alpha, cb, E8P_DIM, search_dtype)
        out[sl] = values
        idxs.append(index)
        scales.extend(alphas)

    return QuantizedBlocks(
        values=out, indices=torch.cat(idxs),
        scales=torch.tensor(scales, dtype=blocks.dtype), padding=0,
    )


def quantization_snr(original: Tensor, reconstructed: Tensor) -> float:
    """Signal-to-noise ratio in dB.  The early-warning signal for plan H5."""
    err = (original - reconstructed).square().sum()
    sig = original.square().sum()
    if float(err) == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(sig / err))
