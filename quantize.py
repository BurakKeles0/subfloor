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
    "fit_scale",
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
#: the scale sweep is 83% of a tile's cost, that is the useful half.
_LATTICE_MIN_ROWS = {"cpu": 64, "cuda": 1024}


@lru_cache(maxsize=16)
def _on_device(dtype: torch.dtype, device: str) -> Tensor:
    """The codebook, cached PER DEVICE.

    `e8p_codebook(dtype).to(device)` copies two megabytes on every call, which
    is more work than the search it was meant to serve and makes an `is` check
    against it always false.  Caching the moved tensor is what lets the fast
    path be selected at all.
    """
    return e8p_codebook(dtype).to(device)


@lru_cache(maxsize=16)
def _table_on_device(device: str) -> Tensor:
    return _source_index_table().to(device)


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

    best_idx = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
    best_d = torch.full((x.shape[0],), float("inf"), dtype=x.dtype, device=x.device)
    exact = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)

    for shift_bit, shift in enumerate((0.25, -0.25)):
        h = _nearest_halfinteger_even(x - shift)
        mag = h.abs()

        # Levels outside 0..2 cannot be codewords; clamp so the gather is safe
        # and let the membership test reject them.
        level = (mag - 0.5).round().to(torch.int64)
        in_range = (level >= 0).all(dim=-1) & (level < _LEVELS).all(dim=-1)
        key = (level.clamp(0, _LEVELS - 1) * powers).sum(dim=-1)
        src = torch.where(in_range, table[key], torch.full_like(key, -1))
        member = src >= 0

        sign_bits = (h[:, : E8P_DIM - 1] < 0).to(torch.int64)
        sign_idx = (sign_bits * (2 ** torch.arange(
            E8P_DIM - 1, device=x.device))).sum(dim=-1)
        idx = shift_bit * 32768 + src.clamp_min(0) * 128 + sign_idx

        # Track the nearest point of the UNION, member or not, and carry its
        # membership along: that is what decides whether the row is settled.
        d = (x - (h + shift)).square().sum(dim=-1)
        take = d < best_d
        best_idx = torch.where(take, idx, best_idx)
        best_d = torch.where(take, d, best_d)
        exact = torch.where(take, member, exact)

    return best_idx, _on_device(x.dtype, str(x.device))[best_idx], exact


def _nearest(x: Tensor, codebook: Tensor, chunk: int = 4096) -> tuple[Tensor, Tensor]:
    """Nearest codeword for each row of `x` [n, 8].  Returns (index, codeword).

    Brute force over 65536 codewords, chunked over `x`.  ||x-c||^2 expands to
    ||c||^2 - 2 x.c (the ||x||^2 term does not affect the argmin).

    For the canonical E8P table this defers to `nearest_e8p`, which decodes the
    lattice instead of scanning it, and only scans the rows the decoder could
    not settle.  That search is the pipeline's dominant cost -- 79% of a GPU
    pass, `experiments/m0_cost_model.py` -- and it is the one part of the
    dominant cost that is engineering rather than structure.
    """
    floor_rows = _LATTICE_MIN_ROWS.get(x.device.type, 64)
    if (x.shape[0] >= floor_rows
            and codebook is _on_device(x.dtype, str(codebook.device))):
        idx, code, exact = nearest_e8p(x)
        if bool(exact.all()):
            return idx, code
        miss = (~exact).nonzero(as_tuple=True)[0]
        m_idx, m_code = _brute_force(x[miss], codebook, chunk)
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


def fit_scale(
    x: Tensor, codebook: Tensor, n_steps: int = 24, lo: float = 0.4, hi: float = 2.0,
    sample: int | None = None, seed_rng: int = 0,
) -> float:
    """Scale alpha minimizing ||x - alpha * Q(x/alpha)||^2.

    Seeded by matching RMS to the codebook's, then refined by a coarse sweep --
    the objective is not convex in alpha, so a search beats a closed form.

    That sweep is the single most expensive thing in the pipeline: `n_steps`
    passes of nearest-codeword search over every vector, measured at 83% of
    `ldlq_quantize`'s time on a real layer.  `sample` caps how many vectors the
    sweep looks at.  Alpha is one scalar; estimating it from thousands of
    8-dimensional vectors is already far past the point of diminishing returns,
    and the vectors not sampled are still quantized with the result.
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

    best, best_err = seed, float("inf")
    for f in torch.linspace(lo, hi, n_steps).tolist():
        a = seed * f
        _, q = _nearest(x / a, codebook)
        err = float((x - a * q).square().sum())
        if err < best_err:
            best, best_err = a, err
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
                codebook: Tensor, group: int) -> tuple[Tensor, Tensor]:
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
            idx, q = _nearest((Wg / a).reshape(-1, group), codebook)
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

#: Rows past which `_nearest` stops getting faster.  Measured on this machine
#: the sweep gains 12x going from 4 rows to 256 and 3% more from 256 to 1024, so
#: there is nothing to buy above roughly this and the memory is better left free.
CHUNK_TARGET_ROWS = 1024


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
    a = fit_scale(block.reshape(-1, group), cb) if scale is None else float(scale)
    factors = [f.unsqueeze(0) for f in _tile_factors(H, percdamp, parts)]
    alpha = torch.tensor([a], dtype=block.dtype, device=block.device)
    values, idx = _ldlq_sweep(block.unsqueeze(0).clone(), factors, parts,
                              alpha, cb, group)
    return LDLQResult(values=values[0], indices=idx[0], scale=a)


def ldlq_quantize_blocks(
    blocks: Tensor,
    hessians: Tensor | Callable[[int], Tensor],
    *,
    percdamp: float = 0.01,
    scale: str | float = "per_tile",
    scale_sample: int = 8192,
    hessian_block: int | None = None,
    chunk: int = 1,
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
                   done, and 83% of its runtime
      "per_layer"  fit it once from a sample of `scale_sample` vectors drawn
                   across all tiles, then use it everywhere.  This is what
                   QuIP# does, and it is the cheapest large saving available:
                   the sweep stops scaling with the layer.
      a float      use exactly this, fit nothing

    Fitting once over EVERY vector would save nothing at all -- same total work,
    differently arranged -- so the saving is in the sampling, not in the sharing.

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
                               sample=scale_sample)
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

        alphas = [fit_scale(blocks[t].reshape(-1, E8P_DIM), cb)
                  if tile_scale is None else tile_scale for t in members]
        alpha = torch.tensor(alphas, dtype=blocks.dtype, device=blocks.device)
        sl = slice(start, start + len(alphas))
        values, index = _ldlq_sweep(blocks[sl].clone(), factors, parts,
                                    alpha, cb, E8P_DIM)
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
