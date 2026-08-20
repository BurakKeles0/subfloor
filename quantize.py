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

def _nearest(x: Tensor, codebook: Tensor, chunk: int = 4096) -> tuple[Tensor, Tensor]:
    """Nearest codeword for each row of `x` [n, 8].  Returns (index, codeword).

    Brute force over 65536 codewords, chunked over `x`.  ||x-c||^2 expands to
    ||c||^2 - 2 x.c (the ||x||^2 term does not affect the argmin).
    """
    c_sq = codebook.square().sum(dim=1)
    idx = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    for lo in range(0, x.shape[0], chunk):
        hi = min(lo + chunk, x.shape[0])
        d = c_sq.unsqueeze(0) - 2.0 * (x[lo:hi] @ codebook.T)
        idx[lo:hi] = d.argmin(dim=1)
    return idx, codebook[idx]


def fit_scale(
    x: Tensor, codebook: Tensor, n_steps: int = 24, lo: float = 0.4, hi: float = 2.0
) -> float:
    """Scale alpha minimizing ||x - alpha * Q(x/alpha)||^2.

    Seeded by matching RMS to the codebook's, then refined by a coarse sweep --
    the objective is not convex in alpha, so a search beats a closed form.
    """
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
    cb = e8p_codebook(dtype or x.dtype).to(x.device)
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

    cb = e8p_codebook(x.dtype).to(x.device)
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


def ldlq_quantize(
    block: Tensor,
    H: Tensor,
    *,
    percdamp: float = 0.01,
    scale: float | None = None,
    group: int = E8P_DIM,
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

    damp = percdamp * torch.diagonal(H).mean()
    Hd = H + damp * torch.eye(k, dtype=H.dtype, device=H.device)
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Hd))
    U = torch.linalg.cholesky(Hinv, upper=True)

    cb = e8p_codebook(block.dtype).to(block.device)
    a = fit_scale(block.reshape(-1, group), cb) if scale is None else float(scale)

    W = block.clone()
    out = torch.empty_like(W)
    idxs = []
    for j in range(0, k, group):
        g = slice(j, j + group)
        Wg = W[:, g]
        i, q = _nearest(Wg.reshape(-1, group) / a, cb)
        Qg = (a * q).reshape(n_lines, group)
        out[:, g] = Qg
        idxs.append(i)

        # err = (Wg - Qg) @ inv(U[g, g]), via a triangular solve.
        Rt = (Wg - Qg).T
        err = torch.linalg.solve_triangular(U[g, g].T, Rt, upper=False).T
        if j + group < k:
            W[:, j + group:] -= err @ U[g, j + group:]

    return LDLQResult(values=out, indices=torch.cat(idxs), scale=a)


def ldlq_quantize_blocks(
    blocks: Tensor,
    hessians: Tensor,
    *,
    percdamp: float = 0.01,
) -> QuantizedBlocks:
    """`ldlq_quantize` over every tile.  `hessians` is [n_tiles, k, k], each one
    the tile's sub-Hessian in the SAME basis as its block."""
    if blocks.ndim != 3:
        raise ValueError(f"blocks must be 3-D, got {tuple(blocks.shape)}")
    n_tiles, lpt, k = blocks.shape
    if hessians.shape != (n_tiles, k, k):
        raise ValueError(
            f"hessians must be ({n_tiles}, {k}, {k}), got {tuple(hessians.shape)}"
        )
    out = torch.empty_like(blocks)
    idxs, scales = [], []
    for t in range(n_tiles):
        r = ldlq_quantize(blocks[t], hessians[t], percdamp=percdamp)
        out[t] = r.values
        idxs.append(r.indices)
        scales.append(r.scale)
    return QuantizedBlocks(
        values=out, indices=torch.stack(idxs),
        scales=torch.tensor(scales, dtype=blocks.dtype), padding=0,
    )


def quantization_snr(original: Tensor, reconstructed: Tensor) -> float:
    """Signal-to-noise ratio in dB.  The early-warning signal for plan H5."""
    err = (original - reconstructed).square().sum()
    sig = original.square().sum()
    if float(err) == 0.0:
        return float("inf")
    return float(10.0 * torch.log10(sig / err))
