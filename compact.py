"""Gathering survivors into dense per-tile blocks.

Compaction is what makes the rest of the pipeline possible: once a tile's
survivors sit in a dense [lines_per_tile, k] block, a rotation and a vector
quantizer can be applied to it exactly as if it were a small dense layer.

Plan section H1 -- compaction happens AFTER the mask is frozen and never before.
The mask is chosen in the untouched basis; only the compacted block is rotated.

Both axes are handled by canonicalizing to the row-tile orientation: for Axis A
we work on W.T, where a column-tile problem is literally a row-tile problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from tiling import TileMask

__all__ = ["CompactWeights", "compact", "scatter"]


@dataclass(frozen=True)
class CompactWeights:
    """Survivors, tile by tile.

    `blocks[t]` is [lines_per_tile, k] and holds the surviving weights of tile
    t, in the canonical (row-tile) orientation.  `line_index[t]` and
    `idx_index[t]` say where they came from.
    """

    blocks: Tensor            # [n_tiles, lines_per_tile, k]
    line_index: Tensor        # long [n_tiles, lines_per_tile]
    idx_index: Tensor         # long [n_tiles, k]
    mask: TileMask

    @property
    def n_tiles(self) -> int:
        return self.blocks.shape[0]

    @property
    def lines_per_tile(self) -> int:
        return self.blocks.shape[1]

    @property
    def k(self) -> int:
        return self.blocks.shape[2]

    def with_blocks(self, blocks: Tensor) -> "CompactWeights":
        """Same geometry, new values -- for rotation and quantization results."""
        if blocks.shape != self.blocks.shape:
            raise ValueError(
                f"blocks shape {tuple(blocks.shape)} != {tuple(self.blocks.shape)}"
            )
        return CompactWeights(blocks, self.line_index, self.idx_index, self.mask)


def _canonical(W: Tensor, mask: TileMask) -> Tensor:
    """View W in the row-tile orientation: [n_lines, n_idx]."""
    if W.shape != (mask.n_out, mask.n_in):
        raise ValueError(
            f"W has shape {tuple(W.shape)}, mask expects "
            f"({mask.n_out}, {mask.n_in})"
        )
    return W if mask.axis == "B" else W.T


def compact(W: Tensor, mask: TileMask) -> CompactWeights:
    """Gather each tile's survivors into a dense block."""
    if not mask.is_uniform():
        raise ValueError(
            "compaction needs a uniform survivor count per tile; got "
            f"{mask.survivors_per_tile().tolist()[:8]}... "
            "(use mode='per_tile_uniform')"
        )
    W2 = _canonical(W, mask)

    order = torch.argsort(mask.assignment, stable=True)
    line_index = order.view(mask.n_tiles, mask.lines_per_tile)
    k = int(mask.survivors_per_tile()[0])
    idx_index = mask.support.nonzero()[:, 1].view(mask.n_tiles, k)

    blocks = W2[line_index.unsqueeze(-1), idx_index.unsqueeze(1)]
    return CompactWeights(blocks, line_index, idx_index, mask)


def scatter(cw: CompactWeights) -> Tensor:
    """Inverse of `compact`: place the blocks back, zeros elsewhere.

    `scatter(compact(W, m))` equals `m.apply(W)` exactly.
    """
    m = cw.mask
    out = torch.zeros(
        (m.n_lines, m.n_idx), dtype=cw.blocks.dtype, device=cw.blocks.device
    )
    out[cw.line_index.unsqueeze(-1), cw.idx_index.unsqueeze(1)] = cw.blocks
    return out if m.axis == "B" else out.T.contiguous()
