"""Saliency: how much each weight is worth keeping.

Spec v6 section 4.3 lists four formulas.  They are really TWO metrics:

    Grup-Wanda, Axis A:  eps(i,t) = sum_{j in C_t} (|w_ij| * ||X_j||)^2
    Grup-Wanda, Axis B:  eps(j,t) = ||X_j||^2 * sum_{i in R_t} w_ij^2

Both aggregate the SAME per-weight quantity  w_ij^2 * ||X_j||^2 ; they differ
only in the direction of the sum.  The same holds for the OBS pair, which both
aggregate  w_ij^2 / [H^-1]_jj .

Building it that way is not a refactor for its own sake -- it is the fix for a
real defect (plan section D2).  Spec v6 requires saliency to be held constant
across the axis comparison, but pairs Axis A with EXACT group-OBS and Axis B
with the diagonal approximation.  Those are different-fidelity metrics, so the
comparison would measure saliency accuracy rather than granularity.  Here the
axes cannot diverge: they share the per-weight metric by construction, and only
`aggregate_to_tiles` knows about direction.

The exact group-OBS form is still provided -- as `group_obs_error` -- but it is
an ABLATION, not the axis-comparison scorer.

Scale warning (Spec v6 section 4.3): Wanda and OBS are not on the same scale.
Never mix them under one lambda.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tiling import contiguous_assignment, n_tiles_for

__all__ = [
    "PER_WEIGHT_METRICS",
    "per_weight_saliency",
    "aggregate_to_tiles",
    "tile_scores",
    "damped_hessian_inverse",
    "group_obs_error",
    "group_obs_compensation",
    "normalizer",
    "coordinate_ffn_saliency",
]

PER_WEIGHT_METRICS = ("wanda", "wanda_l1", "obs_diag", "magnitude")


# --------------------------------------------------------------------------- #
# Per-weight metrics
# --------------------------------------------------------------------------- #

def per_weight_saliency(
    W: Tensor,
    metric: str = "wanda",
    *,
    act_norm: Tensor | None = None,
    hinv_diag: Tensor | None = None,
    eps: float = 1e-12,
) -> Tensor:
    """Cost of removing each weight on its own, as a [n_out, n_in] matrix.

    metric="wanda"     : (|w| * ||X_j||)^2   -- squared, so that a group sum is
                         an L2 aggregation and lines up with the OBS quadratic.
                         This IS a choice: original Wanda ranks by |w| * ||X_j||
                         and squaring changes the ranking of a GROUP (though not
                         of individual weights, since squaring is monotone).
                         Ablate it against "wanda_l1" (plan section E4).
    metric="wanda_l1"  : |w| * ||X_j||       -- original Wanda, L1 aggregation.
    metric="obs_diag"  : w^2 / [H^-1]_jj     -- SparseGPT's per-weight error.
    metric="magnitude" : w^2                 -- baseline only.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {tuple(W.shape)}")

    if metric in ("wanda", "wanda_l1"):
        if act_norm is None:
            raise ValueError(f"metric={metric!r} requires act_norm (||X_j||_2)")
        if act_norm.shape != (W.shape[1],):
            raise ValueError(
                f"act_norm must have shape ({W.shape[1]},), "
                f"got {tuple(act_norm.shape)}"
            )
        if bool((act_norm < 0).any()):
            raise ValueError("act_norm is a norm; it cannot be negative")
        s = W.abs() * act_norm.unsqueeze(0)
        return s.square() if metric == "wanda" else s

    if metric == "obs_diag":
        if hinv_diag is None:
            raise ValueError("metric='obs_diag' requires hinv_diag")
        if hinv_diag.shape != (W.shape[1],):
            raise ValueError(
                f"hinv_diag must have shape ({W.shape[1]},), "
                f"got {tuple(hinv_diag.shape)}"
            )
        if bool((hinv_diag <= 0).any()):
            raise ValueError(
                "hinv_diag has a non-positive entry; H^-1 must be SPD -- "
                "increase percdamp in damped_hessian_inverse"
            )
        return W.square() / hinv_diag.clamp_min(eps).unsqueeze(0)

    if metric == "magnitude":
        return W.square()

    raise ValueError(f"unknown metric {metric!r}; expected one of {PER_WEIGHT_METRICS}")


# --------------------------------------------------------------------------- #
# Aggregation -- the only place that knows about the axis
# --------------------------------------------------------------------------- #

def aggregate_to_tiles(
    saliency: Tensor,
    axis: str,
    tile_size: int | str,
    assignment: Tensor | None = None,
) -> Tensor:
    """Sum a per-weight saliency over each tile's lines.

    Axis B: tiles group ROWS,    result is [n_tiles, n_in]  (score per column)
    Axis A: tiles group COLUMNS, result is [n_tiles, n_out] (score per row)

    The axes share `saliency`; only the direction of this sum differs.
    """
    if axis not in ("A", "B"):
        raise ValueError(f"axis must be 'A' or 'B', got {axis!r}")
    n_out, n_in = saliency.shape

    # Canonicalize: lines on dim 0, index axis on dim 1.
    lines = saliency if axis == "B" else saliency.T
    n_lines = lines.shape[0]

    if assignment is None:
        assignment = contiguous_assignment(n_lines, tile_size)
    elif assignment.numel() != n_lines:
        raise ValueError(
            f"assignment has {assignment.numel()} entries, expected {n_lines}"
        )

    n_t = n_tiles_for(n_lines, tile_size)
    out = torch.zeros(
        (n_t, lines.shape[1]), dtype=saliency.dtype, device=saliency.device
    )
    return out.index_add_(0, assignment.to(saliency.device), lines)


def tile_scores(
    W: Tensor,
    axis: str,
    tile_size: int | str,
    metric: str = "wanda",
    *,
    act_norm: Tensor | None = None,
    hinv_diag: Tensor | None = None,
    assignment: Tensor | None = None,
) -> Tensor:
    """`per_weight_saliency` then `aggregate_to_tiles`; feeds `make_topk_mask`."""
    s = per_weight_saliency(
        W, metric, act_norm=act_norm, hinv_diag=hinv_diag
    )
    return aggregate_to_tiles(s, axis, tile_size, assignment)


# --------------------------------------------------------------------------- #
# Hessian
# --------------------------------------------------------------------------- #

def damped_hessian_inverse(H: Tensor, percdamp: float = 0.01) -> Tensor:
    """(H + lambda I)^-1 with lambda = percdamp * mean(diag(H)).

    Returns the INVERSE.  Everything downstream wants [H^-1]_jj and (H^-1)_SS,
    never H_SS -- Spec v6 section 7, trap 16: (H^-1)_SS is NOT (H_SS)^-1.
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"H must be square, got shape {tuple(H.shape)}")
    if percdamp < 0:
        raise ValueError(f"percdamp must be non-negative, got {percdamp}")
    n = H.shape[0]
    damp = percdamp * torch.diagonal(H).mean()
    Hd = H + damp * torch.eye(n, dtype=H.dtype, device=H.device)
    return torch.cholesky_inverse(torch.linalg.cholesky(Hd))


# --------------------------------------------------------------------------- #
# Exact group-OBS  (ABLATION ONLY -- see the module docstring)
# --------------------------------------------------------------------------- #

def group_obs_error(W_S: Tensor, Hinv_SS: Tensor) -> Tensor:
    """Exact cost of removing the index set S from every line of a tile:

        eps_S = 1/2 * tr(W_S [(H^-1)_SS]^-1 W_S^T)

    Being a trace, this is invariant under an orthogonal mixing of the tile's
    lines -- which is what makes a line-axis rotation legal on a frozen mask
    (Spec v6 section 7.19, and `test_line_rotation_...` in tests/).

    `Hinv_SS` is a submatrix of H^-1, then inverted.  Do not pass (H_SS)^-1.
    """
    if W_S.shape[1] != Hinv_SS.shape[0] or Hinv_SS.shape[0] != Hinv_SS.shape[1]:
        raise ValueError(
            f"shape mismatch: W_S {tuple(W_S.shape)}, Hinv_SS {tuple(Hinv_SS.shape)}"
        )
    M = torch.linalg.inv(Hinv_SS)
    return 0.5 * torch.einsum("ij,jk,ik->", W_S, M, W_S)


def group_obs_compensation(W_S: Tensor, Hinv: Tensor, S: Tensor) -> Tensor:
    """Update applied to the SURVIVING weights when S is removed:

        dW = - W_S [(H^-1)_SS]^-1 (H^-1[:, S])^T        -> [n_lines, n_in]

    Spec v6 section 4.6: compensation is applied FORWARD only.  Masking dW to
    the not-yet-visited columns is the caller's job -- this returns the full
    update so the caller can decide.
    """
    Hinv_SS = Hinv[S][:, S]
    M = torch.linalg.inv(Hinv_SS)
    return -(W_S @ M) @ Hinv[:, S].T


# --------------------------------------------------------------------------- #
# Cross-layer coordination for T=max  (Spec v6 section 4.2)
# --------------------------------------------------------------------------- #

def normalizer(eps: Tensor, mode: str = "mean") -> Tensor:
    """c_l, the per-layer scale that makes cross-layer sums meaningful.

    Spec v6 section 4.2: the three FFN terms come from different Hessians and
    are not on a common scale.  Summing them raw is trap 17.
    """
    if mode == "mean":
        c = eps.mean()
    elif mode == "median":
        c = eps.median()
    elif mode == "max":
        c = eps.max()
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if not torch.isfinite(c) or c <= 0:
        raise ValueError(f"normalizer is degenerate ({c}); check the saliency input")
    return c


def coordinate_ffn_saliency(
    eps_gate: Tensor,
    eps_up: Tensor,
    eps_down: Tensor,
    mode: str = "mean",
) -> Tensor:
    """Saliency of FFN intermediate channel k, pooled across the three matrices:

        eps_FFN(k) = eps_gate(k)/c_gate + eps_up(k)/c_up + eps_down(k)/c_down

    Only needed at T=max, where an intermediate channel must be dropped from
    gate_proj, up_proj AND down_proj together.

    APPROXIMATION -- say so in the paper.  A SwiGLU channel contributes
    SiLU(gate_k) * up_k, which is multiplicative; a sum over the three terms
    does not capture that coupling (Spec v6 section 4.2).
    """
    if not (eps_gate.shape == eps_up.shape == eps_down.shape):
        raise ValueError(
            "the three saliencies must be per-channel vectors of equal length, got "
            f"{tuple(eps_gate.shape)}, {tuple(eps_up.shape)}, {tuple(eps_down.shape)}"
        )
    return (
        eps_gate / normalizer(eps_gate, mode)
        + eps_up / normalizer(eps_up, mode)
        + eps_down / normalizer(eps_down, mode)
    )
