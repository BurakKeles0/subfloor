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


def _rotations(
    cw: CompactWeights, axis: str, seed: int, share_across_tiles: bool
) -> Tensor:
    n = cw.lines_per_tile if axis == "line" else cw.k
    dtype, device = cw.blocks.dtype, cw.blocks.device
    if share_across_tiles:
        Q = structured_orthogonal(n, seed, dtype, device)
        return Q.unsqueeze(0).expand(cw.n_tiles, n, n)
    return torch.stack([
        structured_orthogonal(n, seed + 7919 * t, dtype, device)
        for t in range(cw.n_tiles)
    ])


def rotate(
    cw: CompactWeights,
    axis: str = "index",
    seed: int = 0,
    share_across_tiles: bool = True,
) -> tuple[CompactWeights, Tensor]:
    """Rotate every tile's block.  Returns (rotated weights, the rotations).

    axis="index" -> block @ Q.T   (mixes survivors; strong, costs at inference)
    axis="line"  -> Q @ block     (mixes the tile's lines; weak, nearly free)

    Sharing one rotation across tiles is statistically fine -- each tile applies
    it to a different index set -- and it does not change the inference cost,
    which is per-tile either way.
    """
    if axis not in ("line", "index"):
        raise ValueError(f"axis must be 'line' or 'index', got {axis!r}")
    Q = _rotations(cw, axis, seed, share_across_tiles)
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

def index_axis_overhead_ratio(tile_size: int | str, density: float, n_idx: int) -> float:
    """Per-tile index-axis rotation, relative to the GEMV it sits on.

        (n_lines/T) * k * log2(k)  /  (n_lines * k)  =  log2(k) / T,  k = d*n_idx

    T=1 is hopeless (~11x), T=16 is expensive but possible, T=max is free.
    """
    if tile_size == "max":
        return 0.0
    k = density * n_idx
    if k <= 1:
        return 0.0
    return math.log2(k) / tile_size


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
