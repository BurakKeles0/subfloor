"""Mask-preserving rotations on compacted survivors.

Spec v6 section 0.5 excludes incoherence processing wholesale: "a sparse matrix
densifies after a global rotation."  That is true of a rotation applied BEFORE
the mask -- and the literature is brutal about it: QuaRot+Wanda at 50% sparsity
gives 5868 ppl on Llama-2-7B (OBR, arXiv:2509.11177), because pruning decisions
taken in the rotated basis are wrong.  Rotation flattens the magnitude
distribution; pruning feeds on concentrated magnitude.

But once the mask is FROZEN there is nothing left to destroy.  A rotation
applied to an already-compacted block cannot move a survivor outside its tile's
index set, because the block spans exactly that set.  Both axes are then
mask-preserving:

    line axis  (Q @ block)    mixes the tile's lines;  block size = lines_per_tile
    index axis (block @ V.T)  mixes the tile's survivors; block size = k

They differ entirely in what they cost at inference (see the overhead ratios
below) and in how much Gaussianization they buy -- the index axis mixes
thousands of coordinates, the line axis only T.

Plan section H1 is the invariant this module exists to respect:
    the mask is always chosen in the unrotated basis; rotation happens only
    after the mask is frozen, on the compacted survivors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from compact import CompactWeights

__all__ = [
    "is_power_of_two",
    "randomized_hadamard",
    "random_orthogonal",
    "structured_orthogonal",
    "block_diagonal_orthogonal",
    "block_partition",
    "rotate",
    "unrotate",
    "line_axis_overhead_ratio",
    "index_axis_overhead_ratio",
]


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _generator(seed: int, device: torch.device) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return g


def _hadamard(n: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Unnormalized Sylvester Hadamard, n a power of two."""
    if not is_power_of_two(n):
        raise ValueError(f"Hadamard needs a power of two, got {n}")
    H = torch.ones((1, 1), dtype=dtype, device=device)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    return H


def randomized_hadamard(
    n: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """RHT: diag(+-1) @ H / sqrt(n).  Orthogonal; n must be a power of two."""
    device = torch.device(device)
    H = _hadamard(n, dtype, device) / math.sqrt(n)
    signs = (
        torch.randint(0, 2, (n,), generator=_generator(seed, device)) * 2 - 1
    ).to(dtype=dtype, device=device)
    return signs.unsqueeze(1) * H


def random_orthogonal(
    n: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Haar-ish orthogonal via QR of a Gaussian.  O(n^2) to apply, so it is only
    used for the small odd factor of a Kronecker construction."""
    device = torch.device(device)
    A = torch.randn((n, n), generator=_generator(seed, device), dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)     # fix the QR sign gauge
    return Q.to(dtype=dtype, device=device)


def structured_orthogonal(
    n: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Orthogonal matrix for arbitrary n, QuIP#-style.

    Factor n = 2^a * m and take kron(RHT(2^a), random_orthogonal(m)).  The
    Hadamard part carries the O(n log n) mixing; the odd factor m is small
    enough that its dense O(m^2) apply does not matter.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    a = (n & -n).bit_length() - 1          # largest power of two dividing n
    p, m = 1 << a, n >> a
    if m == 1:
        return randomized_hadamard(p, seed, dtype, device)
    if p == 1:
        return random_orthogonal(m, seed, dtype, device)
    # .contiguous(): torch.kron reshapes internally and rejects strided inputs.
    return torch.kron(
        randomized_hadamard(p, seed, dtype, device).contiguous(),
        random_orthogonal(m, seed + 1, dtype, device).contiguous(),
    )


def block_partition(n: int, block: int) -> list[tuple[int, int]]:
    """Split `n` coordinates into consecutive chunks of at most `block`.

    The tail is short rather than padded.  Survivor counts are aligned to eight
    (`tiling.uniform_survivor_count(align=8)`) and every block width we use is a
    multiple of eight, so the tail is a multiple of eight too -- which is what
    keeps a block boundary from ever falling inside an E8P group.
    """
    if block < 1:
        raise ValueError(f"block must be positive, got {block}")
    return [(o, min(block, n - o)) for o in range(0, n, block)]


def block_diagonal_orthogonal(
    n: int,
    block: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Orthogonal, but confined to consecutive groups of `block` coordinates.

    Why confine it at all.  A full rotation costs `log2(k)/T` at inference and,
    more expensively, it is the reason LDLQ factorizes in the rotated basis --
    though not the reason it factorizes per tile, which is the column set.  What
    a width-`block` rotation buys is that the rotated sub-Hessian's off-block
    couplings can be dropped consistently: rotation and error feedback then
    share one block structure, and `quantize.ldlq_quantize(hessian_block=...)`
    turns `k^3` into `k * block^2`.

    What it costs is mixing range.  A rotation cannot change the norm of the
    coordinates it spans, only their direction, so a width-8 rotation leaves the
    variation in norm BETWEEN groups of eight exactly as it found it -- and that
    between-group spread is what a single E8P scale has to cover.  The width at
    which this stops mattering is an empirical question, which is what
    `experiments/m0_rotation_value.py` measures.

    `block >= n` degenerates to the full rotation, deliberately: it lets the
    sweep include the unconstrained arm without a special case.
    """
    device = torch.device(device)
    if block >= n:
        return structured_orthogonal(n, seed, dtype, device)
    out = torch.zeros((n, n), dtype=dtype, device=device)
    for j, (off, width) in enumerate(block_partition(n, block)):
        out[off:off + width, off:off + width] = structured_orthogonal(
            width, seed + 7919 * j, dtype, device)
    return out


def _rotations(
    cw: CompactWeights, axis: str, seed: int, share_across_tiles: bool,
    block: int | None = None,
) -> Tensor:
    n = cw.lines_per_tile if axis == "line" else cw.k
    dtype, device = cw.blocks.dtype, cw.blocks.device
    width = n if block is None else block

    def one(s: int) -> Tensor:
        return block_diagonal_orthogonal(n, width, s, dtype, device)

    if share_across_tiles:
        return one(seed).unsqueeze(0).expand(cw.n_tiles, n, n)
    return torch.stack([one(seed + 104729 * t) for t in range(cw.n_tiles)])


def rotate(
    cw: CompactWeights,
    axis: str = "index",
    seed: int = 0,
    share_across_tiles: bool = True,
    block: int | None = None,
) -> tuple[CompactWeights, Tensor]:
    """Rotate every tile's block.  Returns (rotated weights, the rotations).

    axis="index" -> block @ Q.T   (mixes survivors; strong, costs at inference)
    axis="line"  -> Q @ block     (mixes the tile's lines; weak, nearly free)

    Sharing one rotation across tiles is statistically fine -- each tile applies
    it to a different index set -- and it does not change the inference cost,
    which is per-tile either way.  Note what that sharing means for the compute
    wall: the rotation is ALREADY one matrix for the whole layer, so it is not
    what makes LDLQ factorize once per tile.  The per-tile column set is.

    `block=b` confines the rotation to consecutive groups of `b` coordinates
    (`block_diagonal_orthogonal`).  `None` is the full-width rotation.
    """
    if axis not in ("line", "index"):
        raise ValueError(f"axis must be 'line' or 'index', got {axis!r}")
    Q = _rotations(cw, axis, seed, share_across_tiles, block)
    if axis == "line":
        blocks = Q @ cw.blocks
    else:
        blocks = cw.blocks @ Q.transpose(-1, -2)
    return cw.with_blocks(blocks), Q


def unrotate(cw: CompactWeights, Q: Tensor, axis: str = "index") -> CompactWeights:
    """Undo `rotate` exactly."""
    if axis == "line":
        return cw.with_blocks(Q.transpose(-1, -2) @ cw.blocks)
    return cw.with_blocks(cw.blocks @ Q)


# --------------------------------------------------------------------------- #
# Inference cost model  (plan sections C, G4.4)
# --------------------------------------------------------------------------- #
# QuaRot's rotations are folded into neighbouring layers and are therefore free.
# Ours cannot be: every tile owns a different index set, so the transform must
# run after the gather.  The cost is real, and it is what pushes T up.

def index_axis_overhead_ratio(tile_size: int | str, density: float, n_idx: int,
                              block: int | None = None) -> float:
    """Per-tile index-axis rotation, relative to the GEMV it sits on.

        (n_lines/T) * k * log2(k)  /  (n_lines * k)  =  log2(k) / T,  k = d*n_idx

    T=1 is hopeless (~11x), T=16 is expensive but possible, T=max is free.

    `block=b` confines the rotation to width-`b` groups, so each coordinate is
    mixed through log2(b) butterfly stages instead of log2(k) and the ratio
    falls to log2(b)/T.  The saving here is real but modest -- log2 of anything
    is a small number.  The block width earns its keep on the OFFLINE side, in
    `quantize.ldlq_quantize(hessian_block=...)`, where it turns a k^3
    factorization into k*b^2.
    """
    if tile_size == "max":
        return 0.0
    k = density * n_idx
    if k <= 1:
        return 0.0
    width = k if block is None else min(block, k)
    if width <= 1:
        return 0.0
    return math.log2(width) / tile_size


def line_axis_overhead_ratio(tile_size: int | str, density: float, n_idx: int) -> float:
    """Line-axis rotation, relative to the GEMV:  log2(T) / k.

    Essentially free at any T -- but it only mixes T coordinates, so it buys
    correspondingly little incoherence.  Cheap and weak, against the index
    axis's expensive and strong.
    """
    if tile_size == "max" or tile_size == 1:
        return 0.0
    k = density * n_idx
    if k <= 1:
        return 0.0
    return math.log2(tile_size) / k
