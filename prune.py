"""Mask selection and forward-only error compensation.

This is where the pipeline's invariant lives (plan section H1):

    the mask is selected in the UNROTATED basis, and only then frozen.

Rotating first and pruning second is the documented failure mode -- QuaRot+Wanda
at 50% sparsity gives 5868 ppl on Llama-2-7B (OBR, arXiv:2509.11177) -- because
rotation flattens the magnitude distribution that saliency reads.  `prune`
refuses to run on a matrix the caller marks as rotated rather than trusting
anyone to remember.

Compensation follows SparseGPT: walk the index axis once, and push each removed
weight's error onto columns that have NOT been visited yet.  Forward only
(Spec v6 section 4.6); pushing backwards would corrupt decisions already made.

Selection scope:
  'upfront'   -- score once, take the top-k per tile over the whole index axis.
                 This is the Wanda-style comparison group and what M1 needs.
  'blockwise' -- re-score inside each block of `block_size` columns using the
                 running, partially compensated weights.  That is full
                 SparseGPT and belongs to M3; not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

import scoring
from tiling import TileMask, make_topk_mask

__all__ = ["PruneResult", "prune", "forward_compensate"]


@dataclass(frozen=True)
class PruneResult:
    mask: TileMask
    W: Tensor                 # masked, and compensated when requested
    metric: str
    compensated: bool

    def density(self) -> float:
        return self.mask.density()


def forward_compensate(
    W: Tensor,
    keep: Tensor,
    Hinv: Tensor,
    block: int | None = None,
) -> Tensor:
    """SparseGPT's single forward sweep over the index axis.

    `keep` is a bool [n_out, n_in] mask in the SAME orientation as W.  Every
    dropped weight's error is pushed onto the columns to its right, which have
    not been decided yet; columns to the left are already final.

    Uses the upper Cholesky factor of H^-1, so `Hinv[j, j]` is the standard
    per-weight denominator and `Hinv[j, j+1:]` carries the propagation.

    `block=b` holds each group of `b` columns' errors back and applies them to
    everything past the group in one matmul, the arrangement GPTQ and SparseGPT
    both use.  The sequential dependency INSIDE a group is untouched; only the
    columns beyond it, which no longer influence the sweep, are deferred.

    IT IS NOT AN OPTIMISATION EVERYWHERE, AND MEASURING IT ONCE IS HOW I GOT
    THAT WRONG.  Priced first at (512, 2048) and (512, 4096) it read 0.87-1.06x
    and was written off; at those widths the inner loop is launch-bound and
    blocking removes nothing.  Real layers are wider, the rank-1 update touches
    the whole remaining width every iteration, and the thing becomes bandwidth
    bound.  Re-measured at the sizes the model actually has:

        4096 x 4096     2435 ms -> 645 ms    3.77x
        4096 x 11008   18237 ms -> 1839 ms   9.92x
        1024 x 11008    4941 ms -> 1801 ms   2.74x

    Not bit-identical: the deferred tail is one matmul where the sweep did `b`
    rank-1 subtractions, so it sums in a different order.  Measured relative
    difference 2.7e-06 to 4.8e-06, float32's own epsilon at these sizes.  Which
    is why `None` -- the exact arrangement -- remains the default: every quality
    number in this project was taken under it.
    """
    if W.shape != keep.shape:
        raise ValueError(
            f"keep {tuple(keep.shape)} does not match W {tuple(W.shape)}"
        )
    n_in = W.shape[1]
    if Hinv.shape != (n_in, n_in):
        raise ValueError(
            f"Hinv must be ({n_in}, {n_in}), got {tuple(Hinv.shape)}"
        )
    if block is not None and block < 1:
        raise ValueError(f"block must be positive, got {block}")

    U = torch.linalg.cholesky(Hinv, upper=True)
    out = W.clone()
    width = n_in if block is None else block
    for start in range(0, n_in, width):
        stop = min(start + width, n_in)
        deferred = stop < n_in
        # Only allocated when something is actually deferred, so the default
        # path does not carry an [n_out, n_in] scratch it never reads.
        errs = (torch.empty((W.shape[0], stop - start), dtype=W.dtype,
                            device=W.device) if deferred else None)
        for j in range(start, stop):
            w = out[:, j].clone()
            q = torch.where(keep[:, j], w, torch.zeros_like(w))
            err = (w - q) / U[j, j]
            out[:, j] = q
            if deferred:
                errs[:, j - start] = err
            if j + 1 < stop:
                out[:, j + 1:stop] -= err.unsqueeze(1) * U[j, j + 1:stop].unsqueeze(0)
        if deferred:
            out[:, stop:] -= errs @ U[start:stop, stop:]
    return out


def prune(
    W: Tensor,
    *,
    axis: str,
    tile_size: int | str,
    density: float,
    metric: str = "wanda",
    act_norm: Tensor | None = None,
    H: Tensor | None = None,
    compensate: bool = False,
    percdamp: float = 0.01,
    assignment: Tensor | None = None,
    select: str = "upfront",
    block_size: int = 128,
    compensate_block: int | None = None,
    align: int = 1,
    already_rotated: bool = False,
) -> PruneResult:
    """Select a tile mask and apply it, optionally with OBS compensation.

    `already_rotated=True` is refused: see the module docstring.  Callers that
    rotate must do so on the compacted survivors, after this returns.

    `compensate_block` is passed to `forward_compensate`; `None` keeps the exact
    column-by-column arrangement every quality number so far was measured
    under, and a width trades a float32-epsilon difference for up to 9.9x on
    the widest layers.
    """
    if already_rotated:
        raise ValueError(
            "refusing to select a mask on a rotated matrix. Rotation flattens "
            "the magnitude distribution that saliency reads, and pruning in the "
            "rotated basis collapses the model (QuaRot+Wanda 50%: 5868 ppl). "
            "Prune first, freeze the mask, then rotate the compacted survivors."
        )
    if select == "blockwise":
        raise NotImplementedError(
            "blockwise (full SparseGPT) selection is an M3 deliverable; "
            "use select='upfront' for M1"
        )
    if select != "upfront":
        raise ValueError(f"unknown select {select!r}")
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got {tuple(W.shape)}")

    n_out, n_in = W.shape
    Hinv = None
    hinv_diag = None
    if metric == "obs_diag" or compensate:
        if H is None:
            raise ValueError(
                f"metric={metric!r} / compensate={compensate} requires the Hessian H"
            )
        Hinv = scoring.damped_hessian_inverse(H, percdamp)
        hinv_diag = torch.diagonal(Hinv)

    score = scoring.tile_scores(
        W, axis, tile_size, metric,
        act_norm=act_norm, hinv_diag=hinv_diag, assignment=assignment,
    )
    mask = make_topk_mask(
        score, axis, tile_size, density, n_out, n_in,
        assignment=assignment, align=align,
    )

    if not compensate:
        return PruneResult(mask, mask.apply(W), metric, False)

    # Compensation always sweeps the INPUT channels, whichever axis the tiling
    # uses: H is the input covariance, and a removed weight can only be absorbed
    # by other weights reading the same activations.  The tiling axis decides
    # WHICH weights are kept, not the direction of the sweep.
    W_out = forward_compensate(W, mask.expand(), Hinv, block=compensate_block)
    return PruneResult(mask, W_out.contiguous(), metric, True)
