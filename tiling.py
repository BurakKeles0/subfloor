"""Tile partitioning and frozen masks.

Spec v6 section 4.1.  Two axes:

  Axis B (row-tile)    rows are split into n_out/T tiles; each tile picks ONE set
                       of columns, shared by its T rows.
                       T=1   -> unstructured (every row picks its own columns)
                       T=max -> input-channel pruning (one column set for all)

  Axis A (column-tile) the transpose: columns are split into tiles, each tile
                       picks one set of rows.

A tile family is CONTINUOUS in T, which is what separates it from the N:M
lattice.  Everything downstream (compaction, rotation, quantization) consumes a
frozen `TileMask` and may never re-derive it -- see plan section H1.

Internally every operation is canonicalized to the row-tile orientation: for
Axis A we simply work on W.T, where a column-tile problem IS a row-tile problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["MAX_TILE", "TileMask", "n_tiles_for", "contiguous_assignment",
           "uniform_survivor_count", "make_topk_mask"]

MAX_TILE = "max"


def n_tiles_for(n_lines: int, tile_size: int | str) -> int:
    """How many tiles partition `n_lines` lines.

    A "line" is a row for Axis B and a column for Axis A.
    """
    if tile_size == MAX_TILE:
        return 1
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError(f"tile_size must be a positive int or {MAX_TILE!r}")
    if n_lines % tile_size != 0:
        raise ValueError(
            f"tile_size {tile_size} does not divide n_lines {n_lines}; "
            "ragged tiles are not supported (they break per_tile_uniform)"
        )
    return n_lines // tile_size


def contiguous_assignment(n_lines: int, tile_size: int | str) -> Tensor:
    """Default line -> tile map: contiguous blocks.

    Balanced clustering (Sinkhorn / min-cost flow, Spec v6 section 4.4 -- NOT
    Hungarian, NOT plain k-means) produces a permuted assignment instead; the
    rest of the pipeline only cares that every tile has equal size.
    """
    n_t = n_tiles_for(n_lines, tile_size)
    per = n_lines // n_t
    return torch.arange(n_lines) // per


def uniform_survivor_count(n_idx: int, density: float, align: int = 1) -> int:
    """Survivors per tile under `mode='per_tile_uniform'` (Spec v6 section 4.4).

    Rounds to nearest.  The realized density is k/n_idx, which is what the
    accounting must be told -- never the requested density.

    `align` rounds k to a multiple of that many survivors.  Two reasons to use
    it: LDLQ quantizes eight coordinates at a time and needs k % 8 == 0, and
    tensor cores want the compacted block aligned anyway (a cost the spec's
    accounting does not yet model).
    """
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    if align < 1:
        raise ValueError(f"align must be positive, got {align}")
    k = max(1, int(round(density * n_idx)))
    if align > 1:
        if n_idx < align:
            raise ValueError(f"cannot align {k} to {align} when n_idx is {n_idx}")
        k = max(align, int(round(k / align)) * align)
        k = min(k, (n_idx // align) * align)
    return k


@dataclass(frozen=True)
class TileMask:
    """A frozen mask.  `support[t]` is the index set chosen by tile t.

    For Axis B: lines are rows (n_lines = n_out), the index axis is columns
    (n_idx = n_in).  For Axis A the two swap.
    """

    axis: str                 # 'A' or 'B'
    tile_size: int | str
    n_out: int
    n_in: int
    support: Tensor           # bool [n_tiles, n_idx]
    assignment: Tensor        # long [n_lines] -> tile id

    def __post_init__(self) -> None:
        if self.axis not in ("A", "B"):
            raise ValueError(f"axis must be 'A' or 'B', got {self.axis!r}")
        if self.support.dtype != torch.bool:
            raise TypeError(f"support must be bool, got {self.support.dtype}")
        if self.support.ndim != 2:
            raise ValueError(f"support must be 2-D, got shape {tuple(self.support.shape)}")
        if self.assignment.ndim != 1:
            raise ValueError("assignment must be 1-D")
        if self.support.shape != (self.n_tiles, self.n_idx):
            raise ValueError(
                f"support shape {tuple(self.support.shape)} != "
                f"({self.n_tiles}, {self.n_idx})"
            )
        if self.assignment.numel() != self.n_lines:
            raise ValueError(
                f"assignment has {self.assignment.numel()} entries, "
                f"expected {self.n_lines}"
            )
        if int(self.assignment.max()) >= self.n_tiles or int(self.assignment.min()) < 0:
            raise ValueError("assignment contains an out-of-range tile id")
        counts = torch.bincount(self.assignment, minlength=self.n_tiles)
        if int(counts.min()) != int(counts.max()):
            raise ValueError(
                "tiles must be equal-sized; got line counts "
                f"{int(counts.min())}..{int(counts.max())}"
            )

    # -- geometry ---------------------------------------------------------- #

    @property
    def n_lines(self) -> int:
        return self.n_out if self.axis == "B" else self.n_in

    @property
    def n_idx(self) -> int:
        """Spec v6 section 3.2: n_idx = d_in on Axis B, n_out on Axis A."""
        return self.n_in if self.axis == "B" else self.n_out

    @property
    def n_tiles(self) -> int:
        return n_tiles_for(self.n_lines, self.tile_size)

    @property
    def lines_per_tile(self) -> int:
        return self.n_lines // self.n_tiles

    def survivors_per_tile(self) -> Tensor:
        return self.support.sum(dim=1)

    def is_uniform(self) -> bool:
        k = self.survivors_per_tile()
        return bool((k == k[0]).all())

    def density(self) -> float:
        """Realized density -- what the accounting must be given."""
        return float(self.support.to(torch.float64).mean())

    # -- views ------------------------------------------------------------- #

    def expand(self) -> Tensor:
        """Dense bool mask over the weight matrix, [n_out, n_in]."""
        dense = self.support[self.assignment]
        return dense if self.axis == "B" else dense.T.contiguous()

    def apply(self, W: Tensor) -> Tensor:
        """W with pruned positions zeroed."""
        if W.shape != (self.n_out, self.n_in):
            raise ValueError(
                f"W has shape {tuple(W.shape)}, mask expects "
                f"({self.n_out}, {self.n_in})"
            )
        return W * self.expand().to(W.dtype)


def make_topk_mask(
    score: Tensor,
    axis: str,
    tile_size: int | str,
    density: float,
    n_out: int,
    n_in: int,
    assignment: Tensor | None = None,
    align: int = 1,
) -> TileMask:
    """Select the top-k index positions per tile by `score`.

    `score` is [n_tiles, n_idx] -- already aggregated over the lines of each
    tile by whichever saliency the caller chose (Spec v6 section 4.3).  Keeping
    aggregation out of here is deliberate: the axis comparison is only valid if
    both axes use the SAME saliency (plan section D2).
    """
    n_lines = n_out if axis == "B" else n_in
    n_idx = n_in if axis == "B" else n_out
    if assignment is None:
        assignment = contiguous_assignment(n_lines, tile_size)
    n_t = n_tiles_for(n_lines, tile_size)
    if score.shape != (n_t, n_idx):
        raise ValueError(
            f"score shape {tuple(score.shape)} != ({n_t}, {n_idx})"
        )

    k = uniform_survivor_count(n_idx, density, align)
    support = torch.zeros((n_t, n_idx), dtype=torch.bool, device=score.device)
    keep = score.topk(k, dim=1).indices
    support.scatter_(1, keep, True)
    return TileMask(axis=axis, tile_size=tile_size, n_out=n_out, n_in=n_in,
                    support=support, assignment=assignment)
