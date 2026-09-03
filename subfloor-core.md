This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

- Pay special attention to the Repository Instruction. These contain important context and guidelines specific to this project.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: *.py, eval/**, experiments/m1_gates.py, experiments/m1_run.py, README.md, docs/STATUS.md
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
````
docs/
  STATUS.md
eval/
  perplexity.py
  streamed.py
experiments/
  m1_gates.py
  m1_run.py
accounting.py
calibrate.py
compact.py
conftest.py
hf_llama.py
prune.py
quantize.py
README.md
rotation.py
scoring.py
tiling.py
````

# Files

## File: compact.py
````python
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
````

## File: conftest.py
````python
"""Put the repo root and tests/ on sys.path so the flat layout imports cleanly."""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for _p in (_ROOT, _ROOT / "tests", _ROOT / "experiments", _ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
````

## File: scoring.py
````python
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
````

## File: tiling.py
````python
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
````

## File: eval/perplexity.py
````python
"""Perplexity, and the protocol discipline around it.

Computing perplexity is easy.  The trap is comparing it.  While building the
Gate A dry run (docs/gate_a_dry_run.md) two incompatible protocols turned up in
the literature for the SAME model:

    dense Llama-2-7B = 5.12    Wanda, QTIP
    dense Llama-2-7B = 5.47    QuIP#, SliceGPT

QuIP# 2-bit is 6.66 in its own paper and 6.19 in QTIP's table.  That 0.47 gap is
larger than the effect Gate B is trying to resolve, so a number quoted from the
wrong family does not just add noise -- it can invert a conclusion.

CONFIRMED (2026-08-21, results/m0_dense_ppl.json).  The cause is the evaluation
window.  Measured on Llama-2-7B fp16, WikiText-2 test:

    seqlen 2048 -> 5.4675   (published 5.47, off by 0.0025)
    seqlen 4096 -> 5.1143   (published 5.12, off by 0.0057)

So both families are reproducible here and neither is wrong -- they are the same
model measured through different windows.  The consequence is not "pick one"
but "pin the window": a published number may be quoted only next to one of ours
taken at the SAME seqlen.  `identify_protocol` maps a measured dense baseline to
its family, and `compare` refuses to subtract across them.

Windowing follows the GPTQ/SparseGPT/Wanda convention so the numbers are
comparable with the literature -- including its small quirk of dividing the
summed loss by `seqlen` when only `seqlen - 1` tokens were predicted.  Pass
`convention="exact"` for the textbook definition.  The two differ by exactly
`ppl_gptq = ppl_exact ** ((seqlen - 1) / seqlen)` -- under 0.1% at seqlen 2048
for a realistic perplexity, but growing as the window shrinks, so the
convention is part of the protocol key rather than a rounding detail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

__all__ = [
    "PerplexityResult",
    "PUBLISHED",
    "perplexity",
    "identify_protocol",
    "compare",
    "load_eval_tokens",
]

#: WikiText moved under a namespace; a bare "wikitext" is rejected by current
#: huggingface_hub ("Repository id must be 'namespace/name'").
WIKITEXT_REPO = "Salesforce/wikitext"


# --------------------------------------------------------------------------- #
# Published reference numbers, grouped by the protocol they belong to
# --------------------------------------------------------------------------- #
# Keyed by the DENSE baseline, which every one of these papers reports and which
# we can reproduce.  Never mix families.  See docs/gate_a_dry_run.md.

#: Our own reproduction, so later runs can be checked against a number we
#: produced rather than one we read.  Llama-2-7B fp16, WikiText-2 test,
#: convention="gptq" (results/m0_dense_ppl.json).
MEASURED_DENSE = {
    ("llama-2-7b", 2048): 5.4675,
    ("llama-2-7b", 4096): 5.1143,
}

PUBLISHED: dict[str, dict] = {
    "llama-2-7b/dense-5.12": {
        "dense": 5.12,
        "sources": ["Wanda arXiv:2306.11695 Table 3", "QTIP arXiv:2406.11235 Table 5"],
        "results": {
            "wanda-50%-unstructured": 6.42,
            "sparsegpt-50%-unstructured": 6.51,
            "magnitude-50%": 14.89,
            "wanda-4:8": 7.97,
            "sparsegpt-4:8": 8.12,
            "sparsegpt-2:4": 10.17,
            "wanda-2:4": 11.02,
            "qtip-4bit": 5.17,
            "quip#-4bit": 5.19,
            "qtip-3bit": 5.28,
            "quip#-3bit": 5.41,
            "qtip-2bit": 5.86,
            "quip#-2bit": 6.19,
        },
    },
    "llama-2-7b/dense-5.47": {
        "dense": 5.47,
        "sources": ["QuIP# arXiv:2402.04396 Table 2", "SliceGPT arXiv:2401.15024 Table 1"],
        "results": {
            "quip#-4bit": 5.56,
            "quip#-3bit": 5.79,
            "quip#-2bit": 6.66,
            "quarot-gptq-4bit": 5.60,
            "quarot-gptq-3bit": 6.09,
            "quarot-gptq-2bit": 22.07,
            "slicegpt-10%": 5.89,
            "slicegpt-20%": 6.64,
            "slicegpt-25%": 7.24,
            "slicegpt-30%": 8.12,
            "sparsegpt-2:4": 8.69,
        },
    },
    "llama-2-13b/dense-4.57": {
        "dense": 4.57,
        "sources": ["Wanda arXiv:2306.11695 Table 3"],
        "results": {
            "wanda-50%-unstructured": 5.56,
            "sparsegpt-50%-unstructured": 5.63,
            "wanda-4:8": 6.55,
            "wanda-2:4": 8.27,
        },
    },
}


@dataclass
class PerplexityResult:
    """A perplexity, plus everything needed to know what it may be compared to."""

    perplexity: float
    nll: float
    n_tokens: int
    n_windows: int
    seqlen: int
    dataset: str
    convention: str = "gptq"
    model: str = "unknown"
    protocol: str | None = None
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """What must match before two results may be subtracted."""
        return (self.model, self.dataset, self.seqlen, self.convention)


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #

def _logits(model: Callable, batch: Tensor) -> Tensor:
    out = model(batch)
    return getattr(out, "logits", out)


def perplexity(
    model: Callable,
    tokens: Tensor,
    *,
    seqlen: int,
    dataset: str = "unknown",
    model_name: str = "unknown",
    device: torch.device | str | None = None,
    batch_size: int = 1,
    convention: str = "gptq",
    max_windows: int | None = None,
) -> PerplexityResult:
    """Sliding-window perplexity over a flat token stream.

    `tokens` is 1-D; it is cut into non-overlapping windows of `seqlen` and any
    tail shorter than a full window is dropped, which is what every reference
    implementation does.

    convention="gptq"  divide the summed loss by n_windows * seqlen
    convention="exact" divide by the number of tokens actually predicted,
                       n_windows * (seqlen - 1)
    """
    if convention not in ("gptq", "exact"):
        raise ValueError(f"unknown convention {convention!r}")
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be a 1-D stream, got {tuple(tokens.shape)}")
    if seqlen < 2:
        raise ValueError(f"seqlen must be at least 2, got {seqlen}")

    n_windows = tokens.numel() // seqlen
    if n_windows == 0:
        raise ValueError(
            f"stream of {tokens.numel()} tokens holds no window of {seqlen}"
        )
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    windows = tokens[: n_windows * seqlen].view(n_windows, seqlen)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    total = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for lo in range(0, n_windows, batch_size):
            batch = windows[lo: lo + batch_size]
            if device is not None:
                batch = batch.to(device)
            logits = _logits(model, batch).float()
            # Predict token t+1 from everything up to t.
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = batch[:, 1:].reshape(-1)
            total += loss_fn(shift_logits, shift_labels).double().cpu()

    denom = n_windows * (seqlen if convention == "gptq" else seqlen - 1)
    nll = float(total) / denom
    return PerplexityResult(
        perplexity=math.exp(nll),
        nll=nll,
        n_tokens=n_windows * seqlen,
        n_windows=n_windows,
        seqlen=seqlen,
        dataset=dataset,
        convention=convention,
        model=model_name,
    )


# --------------------------------------------------------------------------- #
# Protocol discipline
# --------------------------------------------------------------------------- #

def identify_protocol(
    dense_result: PerplexityResult, tol: float = 0.05
) -> str | None:
    """Which published family does our dense measurement belong to?

    Measure the dense model first; whichever baseline it reproduces is the only
    family whose numbers may be quoted alongside ours.  Landing between them
    means neither -- report our own dense number and stop borrowing.
    """
    hits = [
        name for name, block in PUBLISHED.items()
        if abs(dense_result.perplexity - block["dense"]) <= tol
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def compare(a: PerplexityResult, b: PerplexityResult) -> float:
    """b.perplexity - a.perplexity, but only when the protocols match.

    Refusing here is the whole point: a difference between families is not a
    difference between methods.
    """
    if a.key() != b.key():
        raise ValueError(
            "refusing to compare across protocols:\n"
            f"  a: model={a.model} dataset={a.dataset} seqlen={a.seqlen} "
            f"convention={a.convention}\n"
            f"  b: model={b.model} dataset={b.dataset} seqlen={b.seqlen} "
            f"convention={b.convention}\n"
            "See docs/gate_a_dry_run.md -- the same method differs by 0.47 ppl "
            "between the two published families for Llama-2-7B."
        )
    return b.perplexity - a.perplexity


def published(protocol: str, name: str) -> float:
    """One published number, from an explicitly named protocol."""
    if protocol not in PUBLISHED:
        raise KeyError(f"unknown protocol {protocol!r}; have {sorted(PUBLISHED)}")
    block = PUBLISHED[protocol]
    if name == "dense":
        return block["dense"]
    if name not in block["results"]:
        raise KeyError(
            f"{name!r} is not recorded for {protocol!r}; "
            f"have {sorted(block['results'])}"
        )
    return block["results"][name]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_eval_tokens(tokenizer, dataset: str = "wikitext2", split: str = "test") -> Tensor:
    """The whole evaluation set as one flat token stream.

    WikiText-2 is joined with newlines and encoded in one pass, matching the
    reference implementations; C4 uses its validation shard.
    """
    from datasets import load_dataset

    if dataset == "wikitext2":
        raw = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(raw["text"])
    elif dataset == "c4":
        raw = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = " ".join(raw[:1100]["text"])
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return tokenizer(text, return_tensors="pt").input_ids.reshape(-1)
````

## File: eval/streamed.py
````python
"""Perplexity for a model that does not fit on the GPU.

Llama-2-7B is 13.5 GiB in fp16 and this card has 8.  But the blocks do not all
need to be resident at once: run every window through block 0, then block 1, and
so on, with one block on the device at a time.

    hidden states   n_windows x seqlen x hidden, fp16   ~2.35 GiB for WikiText-2
    one block       202M params, fp16                    ~0.40 GiB
                                                        -------
                                                        ~2.75 GiB

That is the same shape as `calibrate.sequential_calibrate`, which is the point:
the streaming machinery is needed for calibration anyway, and evaluation reuses
it rather than duplicating it.

The correctness condition is exact agreement with a full-model forward -- see
`test_streamed_matches_the_full_model`.  A streamed evaluator that quietly
drifts would corrupt the one number this project has to get right first: which
published protocol family our dense measurement belongs to (spec v7 section 6).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

import hf_llama as HF
from perplexity import PerplexityResult

__all__ = ["streamed_perplexity", "stream_blocks"]


def _block_out(out: Any) -> Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


def stream_blocks(
    blocks: Sequence[nn.Module],
    hidden: list[Tensor],
    *,
    block_kwargs: dict | None = None,
    device: torch.device | str = "cpu",
    keep_on_device: bool = False,
    progress: Callable[[int], None] | None = None,
) -> list[Tensor]:
    """Push every chunk through every block, one block resident at a time.

    `hidden` is updated in place and returned.  With `keep_on_device` the
    activations stay on the GPU between blocks (faster, needs them to fit);
    otherwise they live on the host and each chunk is copied across per block.
    """
    kwargs_dev = HF.to_device(block_kwargs or {}, device)
    for i, block in enumerate(blocks):
        block.to(device)
        try:
            with torch.no_grad():
                for j, chunk in enumerate(hidden):
                    out = _block_out(block(chunk.to(device), **kwargs_dev))
                    hidden[j] = out if keep_on_device else out.to("cpu")
        finally:
            block.to("cpu")
        if progress:
            progress(i)
    return hidden


def streamed_perplexity(
    model: nn.Module,
    tokens: Tensor,
    *,
    seqlen: int,
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    dataset: str = "unknown",
    model_name: str = "unknown",
    convention: str = "gptq",
    max_windows: int | None = None,
    keep_on_device: bool = False,
    progress: Callable[[int], None] | None = None,
) -> PerplexityResult:
    """Perplexity with one transformer block on the device at a time.

    Numerically identical to `perplexity.perplexity`; the difference is only
    where the weights live while the arithmetic happens.
    """
    if convention not in ("gptq", "exact"):
        raise ValueError(f"unknown convention {convention!r}")
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be a 1-D stream, got {tuple(tokens.shape)}")
    if seqlen < 2:
        raise ValueError(f"seqlen must be at least 2, got {seqlen}")

    n_windows = tokens.numel() // seqlen
    if n_windows == 0:
        raise ValueError(
            f"stream of {tokens.numel()} tokens holds no window of {seqlen}"
        )
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    windows = tokens[: n_windows * seqlen].view(n_windows, seqlen)

    batches = [
        windows[lo: lo + batch_size] for lo in range(0, n_windows, batch_size)
    ]

    # The embedding, causal mask and rotary come from the model itself.
    hidden, block_kwargs = HF.capture_block_inputs(model, batches)
    blocks = HF.get_blocks(model)
    stream_blocks(
        blocks, hidden, block_kwargs=block_kwargs, device=device,
        keep_on_device=keep_on_device, progress=progress,
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total = torch.zeros((), dtype=torch.float64)
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None) or getattr(inner, "ln_f", None)
    head = model.lm_head

    if norm is not None:
        norm.to(device)
    head.to(device)
    try:
        with torch.no_grad():
            for chunk, ids in zip(hidden, batches):
                logits = HF.forward_head(model, chunk.to(device)).float()
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = ids[:, 1:].reshape(-1).to(device)
                total += loss_fn(shift_logits, shift_labels).double().cpu()
    finally:
        if norm is not None:
            norm.to("cpu")
        head.to("cpu")

    denom = n_windows * (seqlen if convention == "gptq" else seqlen - 1)
    nll = float(total) / denom
    return PerplexityResult(
        perplexity=math.exp(nll),
        nll=nll,
        n_tokens=n_windows * seqlen,
        n_windows=n_windows,
        seqlen=seqlen,
        dataset=dataset,
        convention=convention,
        model=model_name,
        extra={"streamed": True, "device": str(device)},
    )
````

## File: experiments/m1_run.py
````python
"""M1's driver: compress a whole model at one configuration, then measure it.

WHAT THIS CLOSES.  `docs/STATUS.md` section 3.3 has said "the compressed model's
perplexity has never been measured" since the project started, and section 8.1
has named this file as the reason.  Not a scientific obstacle -- the script did
not exist.  Everything it needs has: `calibrate.sequential_calibrate` walks the
blocks, `m1_gates.run_config` is the pipeline, `eval.streamed.streamed_perplexity`
scores a model too large for the card, and `hf_llama` joins them to a real
checkpoint.  On 2026-08-25 that chain was run end to end for the first time and
five defects fell out of it; this file is what makes running it routine.

ONE POINT is one (budget, tile size, calibration draw): calibrate and compress
all 32 blocks in order, then evaluate.  The order is Spec v6 trap 20 -- each
block's Hessians come from the COMPRESSED model above it, so a block is
compressed and only then re-run to produce the next one's inputs.

CHECKPOINTING IS NOT AN ADD-ON.  A point is hours and a laptop closes.  The unit
is the block, because that is the only moment the state is consistent: the block
is final and the activations are exactly what the next one will see.  Anything
finer would have to checkpoint mid-Hessian; anything coarser loses the point.

    resume/                       one directory per point
      state.json                  which block is next, and the records so far
      inputs.pt                   the NEXT block's inputs -- overwritten
      block-<i>.pt                that block's compressed weights -- appended

The activations are what makes this correct rather than merely fast.  They are
the compressed model's output, and they cannot be recomputed from a fresh
checkpoint without redoing every block above.  A resume that restored only the
weights and re-ran the dense model would calibrate block 17 against activations
no version of the model ever produced -- and would not fail, it would just be
quietly wrong.  `test_a_resumed_run_lands_where_an_uninterrupted_one_does` is
the check that this is not happening.

TWO ARGUMENTS THAT ARE NOT DEFAULTS, and both are load-bearing:
`dtype=torch.float32` on the calibration (float64 on a GPU is 1/64 rate and
worth 36 days of M1) and `return_weight=True` on `run_config` (without it the
pipeline computes the compressed weight and drops it).  Both are passed here.

WHAT IT REPORTS BESIDES PERPLEXITY.  Per layer: the relative output error and
the SNR, which are free -- the pipeline already computes them.  And for block 0,
the dense E8P reference, because section 3.2's early-warning rule is defined
against it: if a compressed layer's error exceeds twice the dense E8P
reference's, the assumption that E8P holds its quality on a compacted survivor
submatrix has failed, and that assumption is this project's single largest risk.
Checking it costs one extra quantization of seven layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import calibrate as Cal              # noqa: E402
import hf_llama as HF                # noqa: E402
import m1_gates as M                 # noqa: E402
import perplexity as PPL             # noqa: E402
import streamed as ST                # noqa: E402
import tiling as Tl                  # noqa: E402

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"

#: Where a finished point's JSON lands when `--out` is not given.  It has a
#: default because the checkpoint is cleared on success: without one, the only
#: record of a multi-hour run was the terminal it was launched from.
DEFAULT_OUT_DIR = Path("results/m1_points")

#: The preregistration's calibration set: 128 windows at the primary seqlen.
CALIB_SAMPLES = 128
CALIB_SEQLEN = 4096

#: Both are required for the gates (preregistration section 4).  The five
#: zero-shot tasks are reported separately and do NOT enter the gates, so they
#: are not run per point -- once, at the end, on the chosen configuration.
EVAL_DATASETS = ("wikitext2", "c4")

#: Section 3.2: a compressed layer whose error exceeds this multiple of the
#: dense E8P reference means the survivor-submatrix assumption has failed.  The
#: fallback is rotation + GPTQ-3bit and the band moves to 1.83-2.83.
EARLY_WARNING_RATIO = 2.0


@dataclass(frozen=True)
class PointSpec:
    """One grid point.  Also the checkpoint key (`docs/STATUS.md` section 8.1)."""
    model: str = DEFAULT_MODEL
    budget_bits: float = 1.5
    tile_size: int | str = 16
    draw: int = 0

    def slug(self) -> str:
        name = self.model.rsplit("/", 1)[-1]
        return f"{name}_b{self.budget_bits}_t{self.tile_size}_d{self.draw}"


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #

def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a sibling temporary file, then rename onto `path`.

    `os.replace` is atomic on POSIX and on Windows, and the sibling keeps it on
    one volume, which is what the guarantee requires.  A failed write leaves the
    temporary behind and `path` as it was; it is removed on the next attempt.
    """
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class Checkpoint:
    """Block-granular resume for one point.

    Deliberately three files rather than one blob.  The activations are large
    and rewritten every block; the weights are large and written once each; the
    state is tiny and must survive a crash mid-write of either.  Bundling them
    would mean rewriting gigabytes to advance a counter.
    """

    def __init__(self, root: Path, spec: PointSpec) -> None:
        self.dir = root / spec.slug()
        self.spec = spec

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def inputs_path(self) -> Path:
        return self.dir / "inputs.pt"

    def block_path(self, i: int) -> Path:
        return self.dir / f"block-{i:03d}.pt"

    def load(self) -> dict | None:
        """The point's state, or None if it has not started."""
        if not self.state_path.exists():
            return None
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("spec") != asdict(self.spec):
            raise ValueError(
                f"{self.dir} holds a checkpoint for {state.get('spec')}, not "
                f"{asdict(self.spec)} -- refusing to resume across configurations"
            )
        return state

    def save_block(self, index: int, block: torch.nn.Module,
                   inputs: list[torch.Tensor], records: list[dict],
                   diagnostics: list[dict] | None = None,
                   seconds: float = 0.0) -> None:
        """Persist one finished block.  Weights first, state last.

        The order is the crash contract: `state.json` is what says a block is
        done, so it is written only once the weights and activations it refers
        to are on disk.  A crash between them leaves a block file nothing points
        at, which the next run overwrites -- the harmless direction.

        The two files that are REWRITTEN every block go through a temporary
        path and `os.replace` (`_atomic`).  In place they are truncated the
        moment the write opens, and `inputs.pt` is gigabytes: a crash inside
        that window leaves `state.json` still saying block `i` is next while
        the activations it names are half this block's and half the last one's.
        That does not fail on resume -- it calibrates the next block against
        activations no version of the model ever produced, which is exactly the
        silent wrongness this checkpoint exists to prevent (see the module
        docstring).  Renaming makes the failure land on the harmless side: the
        previous complete pair survives untouched.

        `diagnostics` and `seconds` are carried for the same reason `records`
        is: the point's answer is assembled across sessions, and a cloud point
        is several by construction.  Left out, both belonged to whichever
        session happened to finish -- the E8P early warning could be `[]` and
        the wall clock counted only the last leg.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in block.state_dict().items()},
                   self.block_path(index))
        _atomic(self.inputs_path,
                lambda p: torch.save([t.detach().cpu() for t in inputs], p))
        _atomic(self.state_path, lambda p: p.write_text(json.dumps({
            "spec": asdict(self.spec),
            "next_block": index + 1,
            "records": records,
            "diagnostics": list(diagnostics or []),
            "seconds": seconds,
        }, indent=2), encoding="utf-8"))

    def restore_inputs(self) -> list[torch.Tensor]:
        return list(torch.load(self.inputs_path, weights_only=True))

    def apply_saved_blocks(self, blocks, upto: int) -> None:
        """Put the compressed weights back before evaluating.

        Resuming the COMPRESSION needs only the activations -- they already
        carry what the blocks above did.  Resuming the MEASUREMENT needs the
        weights, and they are the thing a fresh `load_llama` does not have.
        """
        for i in range(upto):
            path = self.block_path(i)
            if not path.exists():
                raise FileNotFoundError(
                    f"block {i} is marked done but {path} is missing; the "
                    "checkpoint cannot be completed"
                )
            blocks[i].load_state_dict(torch.load(path, weights_only=True))

    def clear(self) -> None:
        if self.dir.exists():
            for f in self.dir.iterdir():
                f.unlink()
            self.dir.rmdir()


# --------------------------------------------------------------------------- #
# One point
# --------------------------------------------------------------------------- #

def _early_warning(problem: Cal.LayerProblem, record: dict) -> dict:
    """Section 3.2's rule, on one layer.

    The reference is dense E8P at its natural 2.0 bits -- what a PTQ method
    would pay -- and the claim under test is that the same quantizer keeps its
    quality on a compacted submatrix of survivors, which are the fat tail of the
    distribution by construction.
    """
    wall = M.dense_wall(problem)
    ratio = record["rel_output_error"] / max(wall["rel_output_error"], 1e-30)
    return {
        "dense_e8p_error": wall["rel_output_error"],
        "dense_e8p_snr_db": wall["snr_db"],
        "ratio_to_dense": ratio,
        "assumption_broken": ratio > EARLY_WARNING_RATIO,
    }


def run_point(
    spec: PointSpec = PointSpec(),
    *,
    device: str = "cuda",
    resume_root: Path | None = None,
    calib_samples: int = CALIB_SAMPLES,
    calib_seqlen: int = CALIB_SEQLEN,
    calib_batch: int = 1,
    eval_datasets=EVAL_DATASETS,
    eval_seqlen: int = 4096,
    max_eval_windows: int | None = None,
    stop_after_block: int | None = None,
    progress=print,
) -> dict:
    """Compress the whole model at `spec`, then measure it.

    `stop_after_block` exists for the resume test and for nothing else: it
    aborts mid-run the way a closed laptop would, leaving a checkpoint behind.
    """
    t0 = time.time()
    harness = HF.load_llama(spec.model, dtype=torch.float16)
    blocks = harness.blocks
    ckpt = Checkpoint(resume_root, spec) if resume_root else None

    state = ckpt.load() if ckpt else None
    start_block = state["next_block"] if state else 0
    records: list[dict] = list(state["records"]) if state else []
    # Restored, not restarted.  `.get` rather than `[]` so a checkpoint written
    # before these two were carried still resumes -- it loses them, which is
    # what it had anyway.
    diagnostics: list[dict] = list(state.get("diagnostics", [])) if state else []
    seconds_before: float = float(state.get("seconds", 0.0)) if state else 0.0

    if start_block >= len(blocks):
        progress(f"  all {len(blocks)} blocks already compressed; evaluating")
        inputs, block_kwargs = [], {}
    elif state:
        progress(f"  resuming at block {start_block} of {len(blocks)}")
        inputs = ckpt.restore_inputs()
        # `block_kwargs` is the causal mask and the rotary embeddings for this
        # window shape.  Rebuilt rather than stored: it is a pure function of
        # the token shape, and a stored copy is one more thing that can
        # disagree with the activations it is supposed to accompany.
        _, block_kwargs = HF.capture_block_inputs(
            harness.model, _dummy_ids(inputs, calib_seqlen))
    else:
        progress(f"  tokenizing {calib_samples} x {calib_seqlen} calibration tokens")
        tokens = Cal.load_calibration_tokens(
            harness.tokenizer, n_samples=calib_samples, seqlen=calib_seqlen,
            seed=spec.draw, dataset="c4")
        batches = [tokens[i:i + calib_batch]
                   for i in range(0, tokens.shape[0], calib_batch)]
        inputs, block_kwargs = HF.capture_block_inputs(harness.model, batches)

    if ckpt and start_block > 0:
        ckpt.apply_saved_blocks(blocks, start_block)

    def compress(i: int, name: str, problem: Cal.LayerProblem) -> torch.Tensor:
        r = M.run_config(problem, budget_bits=spec.budget_bits,
                         tile_size=spec.tile_size, seed=spec.draw,
                         return_weight=True)
        if "W_hat" not in r:
            raise RuntimeError(
                f"block {i} {name}: {r.get('skipped')} -- the budget is "
                f"unreachable at tile size {spec.tile_size}"
            )
        if i == 0:
            diagnostics.append({"block": i, "name": name,
                                **_early_warning(problem, r)})
        return r["W_hat"]

    def block_done(i: int, ins, recs) -> None:
        done = seconds_before + (time.time() - t0)
        if ckpt:
            ckpt.save_block(i, blocks[i], ins, recs,
                            diagnostics=diagnostics, seconds=done)
        progress(f"  block {i + 1}/{len(blocks)} done ({done / 60:.1f} min)")
        if stop_after_block is not None and i >= stop_after_block:
            raise _StopRun()

    if start_block < len(blocks):
        try:
            recs = Cal.sequential_calibrate(
                blocks[start_block:], inputs, compress,
                block_kwargs=block_kwargs,
                dtype=torch.float32,          # NOT the default; see the docstring
                device=device,
                progress=None,
                on_block_done=block_done,
                block_offset=start_block,
            )
        except _StopRun:
            progress(f"  stopped after block {stop_after_block} (checkpoint kept)")
            return {"spec": asdict(spec), "stopped_after_block": stop_after_block,
                    "records": records, "diagnostics": diagnostics}
        records = records + [r for r in recs if r["block"] >= start_block]

    ppl = {}
    for dataset in eval_datasets:
        progress(f"  evaluating {dataset}")
        stream = PPL.load_eval_tokens(harness.tokenizer, dataset=dataset)
        r = ST.streamed_perplexity(
            harness.model, stream, seqlen=eval_seqlen, device=device,
            dataset=dataset, model_name=spec.model,
            max_windows=max_eval_windows)
        ppl[dataset] = r.perplexity
        progress(f"    {dataset}: {r.perplexity:.4f} over {r.n_windows} windows")

    # `seconds` is the POINT's cost, summed over every session that built it --
    # a cloud point is several by construction, and this run is also the
    # measurement of what a 16 GiB card does with one (`cloud/README.md`).
    # Reported from `t0` alone it counted only the last leg, which for a
    # resumed point could be the evaluation and nothing else.
    out = {
        "spec": asdict(spec),
        "seconds": seconds_before + (time.time() - t0),
        "seconds_this_session": time.time() - t0,
        "perplexity": ppl,
        "records": records,
        "diagnostics": diagnostics,
        "levers": {
            "rotate_kron": M.PIPELINE_ROTATE_KRON,
            "search_dtype": str(M.PIPELINE_SEARCH_DTYPE),
            "compensate_block": M.PIPELINE_COMPENSATE_BLOCK,
        },
    }
    if ckpt:
        ckpt.clear()
    return out


class _StopRun(Exception):
    """Interrupt a run the way a closed laptop would."""


def _dummy_ids(inputs, seqlen):
    """Token batches shaped like the run's, only to rebuild `block_kwargs`.

    The kwargs are the causal mask and the rotary embeddings, which depend on
    the window SHAPE and nothing else -- so any ids of the right shape produce
    the ones the run was using.  The hidden states they come with are discarded;
    the real ones come off the checkpoint.
    """
    batch = inputs[0].shape[0]
    return [torch.zeros((batch, seqlen), dtype=torch.long)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tile", default="16",
                    help="tile size, or 'max' for the structured end")
    ap.add_argument("--draw", type=int, default=0,
                    help="calibration draw -- the axis Gate B's CIs are over")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume-root", type=Path, default=Path("results/m1_resume"),
                    help="checkpoint directory; --no-resume disables it")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES)
    ap.add_argument("--calib-seqlen", type=int, default=CALIB_SEQLEN)
    ap.add_argument("--eval-seqlen", type=int, default=4096)
    ap.add_argument("--datasets", nargs="*", default=list(EVAL_DATASETS))
    ap.add_argument("--max-eval-windows", type=int, default=None)
    ap.add_argument("--stop-after-block", type=int, default=None,
                    help="abort mid-run, for the resume test")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the finished point's JSON goes; defaults to "
                         f"{DEFAULT_OUT_DIR}/<slug>.json")
    args = ap.parse_args(argv)

    tile = Tl.MAX_TILE if args.tile == "max" else int(args.tile)
    spec = PointSpec(model=args.model, budget_bits=args.budget,
                     tile_size=tile, draw=args.draw)
    print(f"point: {spec.slug()}")

    out = run_point(
        spec, device=args.device,
        resume_root=None if args.no_resume else args.resume_root,
        calib_samples=args.calib_samples, calib_seqlen=args.calib_seqlen,
        eval_datasets=tuple(args.datasets), eval_seqlen=args.eval_seqlen,
        max_eval_windows=args.max_eval_windows,
        stop_after_block=args.stop_after_block,
    )
    # Written unconditionally, and this is not a convenience.  A point is hours
    # and `run_point` clears the checkpoint once it has evaluated, so with no
    # `--out` the perplexity, the per-layer records and block 0's E8P
    # early-warning diagnostic -- the check on this project's largest single
    # risk -- survived only as stdout.  An interrupted run writes nothing: it
    # has no result yet, and clobbering a finished point's JSON with a partial
    # one is the failure this avoids.
    if "stopped_after_block" not in out:
        dest = args.out or (DEFAULT_OUT_DIR / f"{spec.slug()}.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str),
                        encoding="utf-8")
        print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

## File: hf_llama.py
````python
"""HuggingFace adapter: a real Llama into `calibrate.sequential_calibrate`.

`sequential_calibrate` takes any `list[nn.Module]` and a list of input batches.
Everything here exists to produce those two things from a HF causal LM, and to
put the result back together for evaluation.

The inputs to block 0 are CAPTURED, not reconstructed.  Rebuilding the causal
mask and rotary embeddings by hand means duplicating whatever the installed
transformers does, and drifting silently when it changes.  Instead block 0 is
temporarily wrapped, the model's own forward runs, and the wrapper records the
exact `(hidden_states, kwargs)` the model would have passed -- then aborts
before any real work happens.  This is what the reference pruning codebases do
and it is version-agnostic by construction.

Testable without a download: `tiny_llama()` builds a randomly initialized
LlamaForCausalLM from a small config.  Same class, same forward path, same
rotary and GQA code -- just small.  A 7B checkpoint proves nothing here that a
2-layer model does not.

Memory: the model stays on CPU and one block at a time moves to the GPU.  For
Llama-2-7B a 128 x 2048 calibration set is ~2.1 GiB of fp16 activations, so an
8 GiB card can hold the activations plus one block, but not the whole model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "LlamaHarness",
    "load_llama",
    "tiny_llama",
    "get_blocks",
    "capture_block_inputs",
    "forward_head",
    "HeadOnlyLM",
]


class _StopForward(Exception):
    """Raised by the catcher once block 0's inputs are in hand."""


class _Catcher(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.hidden: list[Tensor] = []
        self.kwargs: dict[str, Any] | None = None

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        self.hidden.append(hidden_states.detach())
        if self.kwargs is None:
            # Keep everything except the cache: calibration is a pure forward,
            # and a live Cache would accumulate across batches.
            self.kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ("past_key_values", "past_key_value", "use_cache")
            }
        raise _StopForward


def get_blocks(model: nn.Module) -> list[nn.Module]:
    """The transformer blocks, in order.

    Covers the usual layouts: `model.model.layers` (Llama, Mistral, Qwen),
    `model.transformer.h` (GPT-2/NeoX style), and a bare `model.layers`.
    """
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        else:
            return list(obj)
    raise AttributeError(
        f"cannot find the block list on {type(model).__name__}; "
        "pass the blocks explicitly"
    )


def capture_block_inputs(
    model: nn.Module,
    token_batches: Sequence[Tensor],
    *,
    device: torch.device | str = "cpu",
) -> tuple[list[Tensor], dict[str, Any]]:
    """Run the model far enough to see what block 0 receives.

    Returns `(hidden_states_per_batch, block_kwargs)` -- exactly the two things
    `sequential_calibrate` needs.  `block_kwargs` holds the causal mask, the
    rotary embeddings and the position ids, produced by the model itself.
    """
    if not token_batches:
        raise ValueError("no calibration batches")

    blocks = get_blocks(model)
    if not blocks:
        raise ValueError("model has no transformer blocks")

    catcher = _Catcher(blocks[0])
    parent, index = _locate_blocks(model)
    parent[index] = catcher
    try:
        with torch.no_grad():
            for batch in token_batches:
                try:
                    model(batch.to(device))
                except _StopForward:
                    pass
    finally:
        parent[index] = catcher.inner

    if len(catcher.hidden) != len(token_batches):
        raise RuntimeError(
            f"captured {len(catcher.hidden)} of {len(token_batches)} batches; "
            "the model did not reach block 0 for every batch"
        )
    return catcher.hidden, catcher.kwargs or {}


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Move every tensor inside a nested structure; leave everything else alone.

    `capture_block_inputs` returns `block_kwargs` holding the causal mask and the
    rotary embeddings, and the rotary entry is a TUPLE of tensors.  A flat
    `{k: v.to(device) ...}` therefore leaves half of it behind and the block
    forward dies on a device mismatch several frames deep in transformers -- so
    this lives next to the function that produces the structure rather than
    being rewritten by each caller.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def _locate_blocks(model: nn.Module) -> tuple[Any, int]:
    """The container holding the blocks, so block 0 can be swapped in place."""
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        else:
            return obj, 0
    raise AttributeError("cannot locate the block container")


def forward_head(model: nn.Module, hidden: Tensor) -> Tensor:
    """Final norm + lm_head, for turning calibrated hidden states into logits."""
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None) or getattr(inner, "ln_f", None)
    if norm is not None:
        hidden = norm(hidden)
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError("model has no lm_head")
    return head(hidden)


class HeadOnlyLM(nn.Module):
    """Adapts a compressed model to `eval.perplexity`, which wants ids -> logits."""

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device_ = device

    def forward(self, ids: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(ids.to(self.device_)).logits


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass
class LlamaHarness:
    model: nn.Module
    tokenizer: Any = None
    blocks: list[nn.Module] = field(default_factory=list)

    @property
    def config(self) -> Any:
        return self.model.config

    def prepare(
        self, token_batches: Sequence[Tensor], device: torch.device | str = "cpu"
    ) -> tuple[list[Tensor], dict[str, Any]]:
        return capture_block_inputs(self.model, token_batches, device=device)


def load_llama(
    name: str = "meta-llama/Llama-2-7b-hf",
    *,
    dtype: torch.dtype = torch.float16,
    device_map: str | None = None,
) -> LlamaHarness:
    """Load a checkpoint, on CPU by default.

    `device_map=None` keeps the whole model on CPU on purpose: the calibration
    loop moves one block at a time, which is what makes a 7B model tractable on
    a small card.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return LlamaHarness(model=model, tokenizer=tokenizer, blocks=get_blocks(model))


def tiny_llama(
    *,
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    max_position_embeddings: int = 128,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> LlamaHarness:
    """A randomly initialized Llama, small enough to run anywhere.

    `num_key_value_heads < num_attention_heads` on purpose: that exercises GQA,
    which is what makes T=max coordination non-trivial (spec v7 section 4.2).
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg).to(dtype)
    model.eval()
    return LlamaHarness(model=model, tokenizer=None, blocks=get_blocks(model))
````

## File: prune.py
````python
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
````

## File: accounting.py
````python
"""Bit-budget accounting for the tile-sparsity study.

Implements Spec v6 section 3 (Muhasebe).  Every budget and every density the
experiments are anchored to must come from this module -- nothing downstream may
hard-code a bit budget or a density.

Audit corrections baked in (plan file sections A1-A3, B1, D1):

  * Golden constants are DERIVED, never typed.  Spec v6 section 5.2's anchor-1
    table had two wrong cells (tile-4, T=max) and section 3.4 carried a stale
    value of log2(11008).  Both error classes are structurally impossible here.

  * `nm_index_bits(..., packing="combinatorial")` is exposed as a PRACTICAL
    encoding, not an information-theoretic bound.  A fixed-count block code is
    decodable in O(1) per block, so it keeps random access -- which is the whole
    justification for the `practical` column.  The default `index_model`
    ("practical") is left exactly as Spec v6 froze it, so the pre-registered
    accounting is unchanged; the correction is available and testable next to it
    rather than silently replacing it.

  * `is_live` answers "is this cell a usable GRANULARITY probe".  That is not the
    same question as "is this row reportable" -- see `live_diagnostics`.  Spec
    v6's own strongest Gate A candidate (AQLM-survivor) fails `is_live` at the
    T=max edge while remaining a perfectly valid Gate A row.

Conventions frozen by the spec:
  W(wb)   = wb + q_over,  q_over = (scale_bits + wb) / group_size
            4-bit -> 532/128, 3-bit -> 403/128, 2-bit -> 274/128
  index   = min(1, d * log2(n_idx)) / T     (bitmap / fixed-width-list cascade)
  B*(T)   = 1/T + W / log2(n_idx)           (the two index branches meet here)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

__all__ = [
    "SCHEMES",
    "FIXED_DENSITY_SCHEMES",
    "Q_OVERHEAD_SCALES_WITH_DENSITY",
    "MAX_TILE",
    "Config",
    "q_overhead",
    "weight_cost",
    "vq_bits_from_spec",
    "E8P_PAYLOAD_BITS",
    "E8P_SIDE_INFO_BITS",
    "E8P_QUANTIZED_WEIGHTS",
    "E8P_STORED_BITS",
    "rotation_side_bits",
    "entropy_bits",
    "nm_index_bits",
    "vnm_index_bits",
    "index_bits",
    "bits_per_position",
    "anchor_budget_to",
    "density_for_budget",
    "scheme_floor",
    "b_star",
    "d_star",
    "in_bitmap_regime",
    "tile_density_advantage",
    "live_band",
    "live_diagnostics",
    "is_live",
    "budget_matched_grid",
    "roofline_bytes",
]

# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

SCHEMES = ("dense", "unstructured", "tile", "structured", "nm", "vnm")

#: Schemes whose density is fixed by the scheme itself, so `density_for_budget`
#: is meaningless for them (Spec v6 section 3.3, Kural 2).  "vq_dense" is not a
#: sparsity scheme but is accepted as a name so callers get the same error.
#: "dense" is added to the spec's three because it is equally fixed (d == 1).
FIXED_DENSITY_SCHEMES = frozenset({"nm", "vnm", "vq_dense", "dense"})

#: Spec v6 section 3.1.  Whether the quantization overhead (group scale +
#: zero-point) is charged per SURVIVING weight (True -> d * (wb + q_over)) or on
#: top of the surviving weights (False -> d * wb + q_over).
#:
#: This convention flips the sign of the M1 2:4 comparison and MUST be printed in
#: every table:  True  -> 2:4 @ 4-bit = 3.078125 (below dense 3-bit)
#:               False -> 2:4 @ 4-bit = 3.156250 (above dense 3-bit)
#:
#: There is deliberately no "vq" key: the VQ branch does not use q_over at all,
#: it uses vq_bits (section 3.2).
Q_OVERHEAD_SCALES_WITH_DENSITY = {
    "dense": True,
    "unstructured": True,
    "tile": True,
    "structured": True,
    "nm": False,
    "vnm": False,
}

#: Sentinel for the coarse edge of the tile family, T = n (one column set for the
#: whole matrix).  Numerically identical to scheme="structured".
MAX_TILE = "max"

DEFAULT_GROUP_SIZE = 128
DEFAULT_SCALE_BITS = 16

#: Spec v6 section 3.5.  A budget cell is a usable granularity probe only if the
#: fine end of the family is actually sparse and the coarse end is not already
#: dense.  Outside this band the cell measures quantization, not granularity.
LIVE_DENSITY_MIN = 0.2   # required: d(T=1)   >  LIVE_DENSITY_MIN
LIVE_DENSITY_MAX = 0.9   # required: d(T=max) <  LIVE_DENSITY_MAX

#: Spec v6 section 3.3, Kural 2: a fixed-density scheme reported at its own
#: natural cost gets a signed offset column, flagged past this threshold.
OFFSET_FLAG_THRESHOLD = 0.01

# --------------------------------------------------------------------------- #
# E8P, measured
# --------------------------------------------------------------------------- #
# Spec v7 section 3.2 requires the VQ cost to come off a real checkpoint before
# it anchors anything, because the whole live band hangs off it.  Measured by
# `experiments/m0_vq_bits.py` against relaxml/Llama-2-7b-E8P-2Bit, 2026-08-21;
# the manifest arithmetic and the total file size agree exactly.

#: Codeword payload alone: 2^16 codewords over 8 dims.  Verified against the
#: released index shapes, not assumed -- e.g. down_proj stores 344 int64 per row
#: for 11008 weights, and 344*64 == 2*11008 on the nose.
E8P_PAYLOAD_BITS = 2.0

#: Everything the QuIP# release keeps per linear besides the codewords: SU, SV,
#: Wscale, codebook_id, fuse_scales, summed over all 32 blocks.  Two integers
#: read off the manifest; the division is left to the machine, per the rule at
#: the top of this file.  (Typing the quotient instead is not hypothetical --
#: the first draft of this line was wrong in the seventh decimal, and
#: `tests/golden.py` caught it.)
E8P_SIDE_INFO_BITS = 33_698_304
E8P_QUANTIZED_WEIGHTS = 6_476_005_376
E8P_STORED_BITS = E8P_PAYLOAD_BITS + E8P_SIDE_INFO_BITS / E8P_QUANTIZED_WEIGHTS

#: What our pipeline should pay instead.  QuIP#'s SU and SV turn out to be
#: different objects: SU (input side) is a sign vector fine-tuning barely moved
#: off +-1, SV (output side) carries real per-channel scale.  So the transform
#: separates into two GLOBAL diagonals and a per-tile rotation, and a rotation
#: drawn from a seed carries no payload.  See `rotation_side_bits`.
ROTATION_SEED_BITS = 32


# --------------------------------------------------------------------------- #
# Weight cost
# --------------------------------------------------------------------------- #

def q_overhead(
    weight_bits: int,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """Quantization overhead per weight: one FP16 scale + one wb-bit zero-point
    per group.

    >>> q_overhead(4)
    0.15625
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return (scale_bits + weight_bits) / group_size


def weight_cost(
    weight_bits: int,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """W = wb + q_over -- the cost of ONE SURVIVING weight.

    This is the denominator of the `1 - 1/T` identity, so it is the single most
    load-bearing quantity in the spec.

    >>> weight_cost(4) * 128
    532.0
    """
    return weight_bits + q_overhead(
        weight_bits, group_size=group_size, scale_bits=scale_bits
    )


def vq_bits_from_spec(
    idx_bits: int,
    dim: int,
    *,
    entry_bits: int = 16,
    weights_per_codebook: float | None = None,
) -> float:
    """Spec v6 section 3.2, VQ branch.

        vq_bits = idx_bits / dim + codebook_amortization

    `weights_per_codebook=None` means a structured codebook with no per-model
    storage (QuIP# E8P), i.e. zero amortization.

    AQLM 1x16, dim=8, over a Llama-2-7B FFN block (45.1M weights):

    >>> round(vq_bits_from_spec(16, 8, weights_per_codebook=45.1e6), 6)
    2.186

    This is the paper-arithmetic value.  Spec v6 required the real cost to be
    measured from a checkpoint before it anchors anything; that is now done for
    E8P and the payload term is exact -- see `E8P_PAYLOAD_BITS`.  What the
    formula omits is the per-linear side info, `E8P_STORED_BITS - 2.0` in the
    QuIP# release, so keep using the measured constant to anchor budgets.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    base = idx_bits / dim
    if weights_per_codebook is None:
        return base
    if weights_per_codebook <= 0:
        raise ValueError("weights_per_codebook must be positive or None")
    amortization = (2 ** idx_bits) * dim * entry_bits / weights_per_codebook
    return base + amortization


def _weight_terms(
    weight_bits: int | None,
    vq_bits: float | None,
    group_size: int,
    scale_bits: int,
) -> tuple[float, float]:
    """Return (payload_bits, q_over) per surviving weight.

    The VQ branch carries no q_over, which is why it has no entry in
    Q_OVERHEAD_SCALES_WITH_DENSITY.
    """
    if vq_bits is not None:
        if weight_bits is not None:
            raise ValueError("pass either weight_bits or vq_bits, not both")
        if vq_bits <= 0:
            raise ValueError(f"vq_bits must be positive, got {vq_bits}")
        return float(vq_bits), 0.0
    if weight_bits is None:
        raise ValueError("one of weight_bits / vq_bits is required")
    if weight_bits <= 0:
        raise ValueError(f"weight_bits must be positive, got {weight_bits}")
    return float(weight_bits), q_overhead(
        weight_bits, group_size=group_size, scale_bits=scale_bits
    )


def total_weight_cost(
    weight_bits: int | None = None,
    *,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """W, resolved for either the scalar-quantization or the VQ branch."""
    payload, q_over = _weight_terms(weight_bits, vq_bits, group_size, scale_bits)
    return payload + q_over


# --------------------------------------------------------------------------- #
# Index cost
# --------------------------------------------------------------------------- #

def entropy_bits(density: float) -> float:
    """Binary entropy H(d) in bits.  H(0) = H(1) = 0."""
    if density <= 0.0 or density >= 1.0:
        return 0.0
    return -(
        density * math.log2(density) + (1.0 - density) * math.log2(1.0 - density)
    )


def nm_index_bits(n: int, m: int, *, packing: str = "fixed_width") -> float:
    """Index cost of an N:M mask, in bits per position.

    packing="fixed_width"   : ceil(log2(M)) * N / M      (Spec v6, `practical`)
    packing="combinatorial" : log2(C(M, N)) / M          (Spec v6, `info_theoretic`)

    AUDIT (plan section B1): the combinatorial packing is *also practical*.  A
    block of M positions with exactly N survivors is decodable in O(1) without
    touching any other block, so random access survives.  Spec v6 files it under
    `info_theoretic`, which understates how cheap a random-accessible index can
    be and therefore overstates the tile advantage.

    >>> nm_index_bits(2, 4)
    1.0
    >>> round(nm_index_bits(2, 8, packing="combinatorial"), 6)
    0.600919
    """
    if not (0 <= n <= m) or m <= 0:
        raise ValueError(f"need 0 <= N <= M and M > 0, got N={n}, M={m}")
    if packing == "fixed_width":
        if n == 0:
            return 0.0
        return math.ceil(math.log2(m)) * n / m
    if packing == "combinatorial":
        return math.log2(math.comb(m, n)) / m
    raise ValueError(f"unknown packing {packing!r}")


def vnm_index_bits(
    v: int, n: int, m: int, *, native_m: int = 4, packing: str = "fixed_width"
) -> float:
    """Index cost of VENOM's V:N:M format, in bits per position.

    Reconstructed from Castro et al., SC'23 (arXiv:2310.02065).  An R x K matrix
    is cut into V x M blocks and pruned in two stages:

      1. vector-wise -- each block selects `native_m` (=4) of its M columns, and
         all V rows of the block share that selection;
      2. within those 4 columns the hardware's native 2:4 applies, so each row
         keeps N of them.

    Two metadata structures follow, and the paper gives their shapes:

      m-indices   R x K/M x N,     2 bits each   -> 2N/M per position
      column-loc  R/V x K/M x 4,   one column id -> 4*ceil(log2 M)/(V*M)

    THE STRUCTURAL POINT: `V` is a row-tile.  A group of V rows sharing one
    column selection is exactly this project's Axis B at T=V, with the extra
    constraint that the selection is block-local (4 out of each M) rather than
    free across the row.  VENOM is therefore much closer prior work than Spec v6
    credited, and it is also a concrete instance of the block-local index that
    section 3.2 says a bitmap is not the floor of.

    The 2-bit width of m-indices is stated in the paper.  The width of a
    column-loc entry is INFERRED as ceil(log2 M) from the array's shape and
    meaning; it is not quoted.  At M == native_m the vector stage is degenerate
    (choosing 4 of 4) and the honest cost is zero -- `packing="combinatorial"`
    reports that, `fixed_width` reports what VENOM's array actually stores.

    >>> round(vnm_index_bits(64, 2, 8), 6)
    0.523438
    >>> vnm_index_bits(64, 2, 4, packing="combinatorial")   # plain 2:4
    1.0
    """
    if not (0 < n <= native_m):
        raise ValueError(f"need 0 < N <= {native_m}, got N={n}")
    if m < native_m or m % native_m:
        raise ValueError(f"M must be a multiple of {native_m}, got M={m}")
    if v < 1:
        raise ValueError(f"V must be positive, got {v}")

    per_nonzero = math.log2(native_m)                 # 2 bits for the native 2:4
    m_indices = per_nonzero * n / m

    if packing == "fixed_width":
        column_loc = native_m * math.ceil(math.log2(m)) / (v * m)
    elif packing == "combinatorial":
        column_loc = math.log2(math.comb(m, native_m)) / (v * m)
    else:
        raise ValueError(f"unknown packing {packing!r}")
    return m_indices + column_loc


def _elementwise_index(density: float, n_idx: int, model: str) -> float:
    """Index cost of a free (unstructured) mask over `n_idx` positions, before
    any tile amortization.

    `practical` is the min(bitmap, fixed-width list) cascade: 1.0 bit per
    position, or d*log2(n_idx) bits when the survivors are sparse enough that
    listing them is cheaper.  Both keep O(1) random access.
    """
    if model == "practical":
        if n_idx is None or n_idx <= 1:
            raise ValueError(f"n_idx must be > 1, got {n_idx}")
        return min(1.0, density * math.log2(n_idx))
    if model == "info_theoretic":
        return entropy_bits(density)
    raise ValueError(f"unknown index_model {model!r}")


def rotation_side_bits(
    tile_size: int | str,
    survivors_per_tile: int,
    n_out: int,
    *,
    n_in: int | None = None,
    scheme: str = "separated",
    entry_bits: int = DEFAULT_SCALE_BITS,
    seed_bits: int = ROTATION_SEED_BITS,
) -> float:
    """Bits per SURVIVING weight for the rotation's side information.

    Rotating compacted survivors is not free of storage the way QuIP#'s is: each
    tile owns a different column set, so anything held per column is held once
    per tile.  Three ways to pay it, and which one applies is a design decision
    the measurement settled rather than a fact about the scheme:

      "per_tile_fp16"   a learned column vector per tile     -> entry_bits / T
      "per_tile_sign"   a stored sign vector per tile        -> 1 / T
      "separated"       two global diagonals plus a seeded rotation per tile

    The first two carry a `1/T` term, which would sit right on top of the index
    and eat a fixed share of the tiling gain -- a quarter of it at T=16 even in
    the packed-sign form.  The third does not, and the third is available: a
    diagonal commutes with the gather, so the input-side diagonal can be applied
    before compaction and shared by every tile, while only the rotation proper
    stays per-tile and a seeded orthogonal stores nothing but its seed.

    >>> round(rotation_side_bits(16, 7926, 4096, n_in=11008), 6)
    0.007696
    >>> rotation_side_bits(16, 7926, 4096, scheme="per_tile_sign")
    0.0625
    """
    if survivors_per_tile <= 0:
        raise ValueError(f"survivors_per_tile must be positive, "
                         f"got {survivors_per_tile}")
    if n_out <= 0:
        raise ValueError(f"n_out must be positive, got {n_out}")
    t = n_out if tile_size == MAX_TILE else tile_size
    if not isinstance(t, int) or t < 1:
        raise ValueError(
            f"tile_size must be a positive int or {MAX_TILE!r}, got {tile_size!r}"
        )
    if t > n_out:
        raise ValueError(f"tile_size {t} exceeds n_out {n_out}")

    if scheme == "per_tile_fp16":
        return entry_bits / t
    if scheme == "per_tile_sign":
        return 1.0 / t
    if scheme != "separated":
        raise ValueError(f"unknown rotation side-info scheme {scheme!r}")
    if n_in is None:
        raise ValueError("scheme='separated' requires n_in for the input diagonal")
    if n_in <= 0:
        raise ValueError(f"n_in must be positive, got {n_in}")

    n_tiles = n_out / t
    n_survivors = n_out * survivors_per_tile
    return (seed_bits * n_tiles + entry_bits * (n_in + n_out)) / n_survivors


def _inv_tile(scheme: str, tile_size: int | str | None) -> float:
    """1/T, the factor the index is amortized by.

    0.0 means "no index at all" (T = n, the structured edge of the family).
    """
    if scheme == "structured":
        return 0.0
    if scheme == "unstructured":
        return 1.0
    if scheme == "tile":
        if tile_size is None:
            raise ValueError("scheme='tile' requires tile_size")
        if tile_size == MAX_TILE:
            return 0.0
        if not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError(
                f"tile_size must be a positive int or {MAX_TILE!r}, got {tile_size!r}"
            )
        return 1.0 / tile_size
    raise ValueError(f"scheme {scheme!r} has no tile amortization")


def index_bits(
    scheme: str,
    density: float,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    nm: tuple[int, int] | None = None,
    vnm: tuple[int, int, int] | None = None,
    index_model: str = "practical",
    nm_packing: str = "fixed_width",
) -> float:
    """Index cost in bits per position (Spec v6 section 3.2).

    Note that `n_idx` is layer-dependent (Axis B: n_idx = d_in, Axis A:
    n_idx = n_out).  In the bitmap regime the index is 1/T and therefore layer
    INDEPENDENT; the layer dependence only switches on below B*.
    """
    if scheme in ("dense", "structured"):
        return 0.0
    if scheme == "nm":
        if nm is None:
            raise ValueError("scheme='nm' requires nm=(N, M)")
        return nm_index_bits(nm[0], nm[1], packing=nm_packing)
    if scheme == "vnm":
        if vnm is None:
            raise ValueError("scheme='vnm' requires vnm=(V, N, M)")
        return vnm_index_bits(*vnm, packing=nm_packing)
    if scheme in ("unstructured", "tile"):
        return _elementwise_index(density, n_idx, index_model) * _inv_tile(
            scheme, tile_size
        )
    raise ValueError(f"unknown scheme {scheme!r}")


# --------------------------------------------------------------------------- #
# Bits per position
# --------------------------------------------------------------------------- #

def bits_per_position(
    scheme: str,
    density: float | None = None,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    nm: tuple[int, int] | None = None,
    vnm: tuple[int, int, int] | None = None,
    vq_bits: float | None = None,
    index_model: str = "practical",
    nm_packing: str = "fixed_width",
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """Total bits per weight POSITION (not per surviving weight).

    >>> bits_per_position("dense", 1.0, 4)
    4.15625
    >>> round(bits_per_position("tile", 0.5, 4, 11008, tile_size=16), 6)
    2.140625
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    if scheme == "dense":
        if density is not None and abs(density - 1.0) > 1e-12:
            raise ValueError(f"scheme='dense' implies density=1.0, got {density}")
        density = 1.0
    elif scheme in ("nm", "vnm"):
        if scheme == "vnm":
            if vnm is None:
                raise ValueError("scheme='vnm' requires vnm=(V, N, M)")
            implied = vnm[1] / vnm[2]
        else:
            if nm is None:
                raise ValueError("scheme='nm' requires nm=(N, M)")
            implied = nm[0] / nm[1]
        if density is not None and abs(density - implied) > 1e-12:
            raise ValueError(
                f"scheme={scheme!r} with nm={nm} implies density={implied}, "
                f"got {density}"
            )
        density = implied
    if density is None:
        raise ValueError(f"scheme={scheme!r} requires an explicit density")
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0, 1], got {density}")

    payload, q_over = _weight_terms(weight_bits, vq_bits, group_size, scale_bits)

    if Q_OVERHEAD_SCALES_WITH_DENSITY[scheme]:
        weight_term = density * (payload + q_over)
    else:
        weight_term = density * payload + q_over

    idx = index_bits(
        scheme,
        density,
        n_idx,
        tile_size=tile_size,
        nm=nm,
        vnm=vnm,
        index_model=index_model,
        nm_packing=nm_packing,
    )
    return weight_term + idx


def anchor_budget_to(
    scheme: str,
    weight_bits: int | None = None,
    **kw,
) -> float:
    """Kural 1: every budget is anchored to the FULL cost of a dense baseline.

    Never anchor to a round number -- Spec v6 section 7, trap 6.

    >>> anchor_budget_to("dense", 3)
    3.1484375
    """
    kw.pop("density", None)
    return bits_per_position(scheme, 1.0, weight_bits, **kw)


# --------------------------------------------------------------------------- #
# Inverting the budget
# --------------------------------------------------------------------------- #

def _solve_density_numeric(
    budget_bits: float,
    scheme: str,
    weight_bits: int | None,
    n_idx: int | None,
    kw: dict,
) -> float | None:
    """Bisection fallback for index models with no closed-form inverse
    (e.g. info_theoretic, where the index is H(d))."""
    def cost(d: float) -> float:
        return bits_per_position(
            scheme, d, weight_bits, n_idx, **kw
        )

    lo, hi = 0.0, 1.0
    if cost(hi) < budget_bits:
        return None                      # budget exceeds even the dense cost
    if cost(1e-15) > budget_bits:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cost(mid) <= budget_bits:
            lo = mid
        else:
            hi = mid
    return lo


def density_for_budget(
    scheme: str,
    budget_bits: float,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    vq_bits: float | None = None,
    index_model: str = "practical",
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float | None:
    """Largest density reachable at `budget_bits`, or None if unreachable.

    Returns None when the budget cannot be met at any density in (0, 1] -- either
    it is below the scheme's floor or it exceeds the dense cost.

    Raises ValueError for fixed-density schemes (Spec v6 section 3.3, Kural 2):
    those are reported at their own natural cost with a signed offset column, not
    solved for.

    Closed form for `practical`; the min(bitmap, list) cascade makes the two
    branches mutually exclusive across B*(T), so the dispatch below is exact:

        bitmap : d = (B - 1/T) / W          valid iff d * log2(n_idx) >= 1
        list   : d = B / (W + log2(n_idx)/T) valid iff d * log2(n_idx) <  1
    """
    if scheme in FIXED_DENSITY_SCHEMES:
        raise ValueError(
            f"scheme={scheme!r} has a fixed density; report it at its own cost "
            "with a signed offset column (Spec v6 section 3.3, Kural 2) instead "
            "of solving for density."
        )
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    kw = dict(
        tile_size=tile_size,
        vq_bits=vq_bits,
        index_model=index_model,
        group_size=group_size,
        scale_bits=scale_bits,
    )

    if index_model != "practical":
        d = _solve_density_numeric(budget_bits, scheme, weight_bits, n_idx, kw)
        return d if (d is not None and 0.0 < d <= 1.0) else None

    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    inv_t = _inv_tile(scheme, tile_size)

    if inv_t == 0.0:
        d = budget_bits / W                      # structured / T=max: no index
    else:
        L = math.log2(n_idx)
        d_bitmap = (budget_bits - inv_t) / W
        if d_bitmap * L >= 1.0:
            d = d_bitmap
        else:
            d = budget_bits / (W + L * inv_t)

    return d if 0.0 < d <= 1.0 else None


def scheme_floor(
    scheme: str,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    nm: tuple[int, int] | None = None,
    vnm: tuple[int, int, int] | None = None,
    vq_bits: float | None = None,
    index_model: str = "practical",
    nm_packing: str = "fixed_width",
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """Infimum of bits_per_position over achievable densities.

    Under the min(bitmap, list) cascade BOTH the weight term and the index term
    vanish as d -> 0, so density-scaling schemes have NO hard bit floor.  This is
    the section 3.4 correction: `scheme_floor("unstructured", 4, 11008) == 0.0`,
    not 1.0.  "Unstructured cannot go below 1.0 bit" is a statement about the
    BITMAP, not about unstructured sparsity (Spec v6 section 7, trap 5).
    """
    if scheme == "dense":
        return total_weight_cost(
            weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
        )
    if scheme in ("nm", "vnm"):
        return bits_per_position(
            scheme,
            None,
            weight_bits,
            n_idx,
            nm=nm,
            vnm=vnm,
            vq_bits=vq_bits,
            index_model=index_model,
            nm_packing=nm_packing,
            group_size=group_size,
            scale_bits=scale_bits,
        )
    if scheme in ("unstructured", "tile", "structured"):
        _inv_tile(scheme, tile_size)     # validate tile_size even though unused
        return 0.0
    raise ValueError(f"unknown scheme {scheme!r}")


# --------------------------------------------------------------------------- #
# The B* wall and the 1 - 1/T identity
# --------------------------------------------------------------------------- #

def b_star(
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = 1,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float | None:
    """Budget at which the bitmap and list index branches meet:

        B*(T) = 1/T + W / log2(n_idx)

    At and above B*(T) the `1 - 1/T` identity holds exactly.  Below it the
    unstructured index gets cheaper than a bitmap and the tile advantage ERODES
    -- B* is the lowest budget at which the advantage is fully preserved, and it
    is a closed-form limit on how far the exploratory band can be pushed.

    Returns None for T = max (no index, hence no crossover).

    >>> round(b_star(4, 4096), 7)
    1.3463542
    """
    scheme = "tile" if tile_size not in (1, None) else "unstructured"
    inv_t = _inv_tile(scheme, tile_size if scheme == "tile" else None)
    if inv_t == 0.0:
        return None
    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    return inv_t + W / math.log2(n_idx)


def d_star(n_idx: int) -> float:
    """Density at which a fixed-width index costs exactly one bit: 1 / log2(n_idx)."""
    return 1.0 / math.log2(n_idx)


def in_bitmap_regime(
    budget_bits: float,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = 1,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> bool:
    """Validity domain of the `1 - 1/T` identity (Spec v6 section 7, trap 1).

    Never use the identity without asserting this.
    """
    bs = b_star(
        weight_bits,
        n_idx,
        tile_size=tile_size,
        vq_bits=vq_bits,
        group_size=group_size,
        scale_bits=scale_bits,
    )
    return True if bs is None else budget_bits >= bs


def tile_density_advantage(
    tile_size: int | str,
    weight_bits: int | None = None,
    *,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """The headline identity:  d(T) - d(1) = (1 - 1/T) / W.

    Independent of the budget B.  The ABSOLUTE advantage is constant; the ratio
    d(T)/d(1) grows only because the denominator shrinks (Spec v6 section 7,
    trap 2 -- never write the headline as "the ratio grows").

    Valid only for B >= B*(1); check with `in_bitmap_regime`.
    """
    inv_t = _inv_tile("tile", tile_size)
    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    return (1.0 - inv_t) / W


# --------------------------------------------------------------------------- #
# Live-band filter
# --------------------------------------------------------------------------- #

def live_band(
    weight_bits: int | None = None,
    *,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> tuple[float, float]:
    """(B_min, B_max) over which a budget cell is a usable granularity probe.

        d(T=1)   > 0.2  =>  B > 1 + 0.2 * W
        d(T=max) < 0.9  =>  B < 0.9 * W

    >>> lo, hi = live_band(2)
    >>> lo * 320, hi * 640
    (457.0, 1233.0)
    """
    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    return 1.0 + LIVE_DENSITY_MIN * W, LIVE_DENSITY_MAX * W


def live_diagnostics(config: "Config") -> dict:
    """Why a cell is or is not a granularity probe.

    AUDIT (plan section D1): `is_live` is a question about the GRANULARITY AXIS,
    not about whether a row may be reported.  A config that fails at the T=max
    edge -- AQLM-survivor at anchor 2 has d(T=max) = 0.979 -- is still a
    perfectly valid Gate A row.  Split the two uses; do not silently drop rows.
    """
    if config.budget_bits is None:
        raise ValueError("live_diagnostics needs config.budget_bits")
    common = dict(
        n_idx=config.n_idx,
        vq_bits=config.vq_bits,
        index_model=config.index_model,
        group_size=config.group_size,
        scale_bits=config.scale_bits,
    )
    d_fine = density_for_budget(
        "unstructured", config.budget_bits, config.weight_bits, tile_size=1, **common
    )
    d_coarse = density_for_budget(
        "structured", config.budget_bits, config.weight_bits, **common
    )
    lo, hi = live_band(
        config.weight_bits,
        vq_bits=config.vq_bits,
        group_size=config.group_size,
        scale_bits=config.scale_bits,
    )
    fine_ok = d_fine is not None and d_fine > LIVE_DENSITY_MIN
    coarse_ok = d_coarse is not None and d_coarse < LIVE_DENSITY_MAX
    reasons = []
    if not fine_ok:
        reasons.append(
            f"fine end not sparse enough: d(T=1)="
            f"{'None' if d_fine is None else f'{d_fine:.6f}'} <= {LIVE_DENSITY_MIN}"
        )
    if not coarse_ok:
        reasons.append(
            f"coarse end already dense: d(T=max)="
            f"{'None' if d_coarse is None else f'{d_coarse:.6f}'} >= {LIVE_DENSITY_MAX}"
        )
    return {
        "live": fine_ok and coarse_ok,
        "d_fine": d_fine,
        "d_coarse": d_coarse,
        "band": (lo, hi),
        "reasons": reasons,
    }


def is_live(config: "Config") -> bool:
    """Spec v6 section 3.5.  True iff the cell measures granularity rather than
    quantization.

    wb=2 is DEAD across the whole primary band (2.0-2.3): at anchor 2 its
    d(T=max) is exactly 1.0, so "T=max wins" there would mean "2-bit quantization
    wins", not "structured pruning wins".  Reporting such a cell invites exactly
    that misreading (Spec v6 section 7, trap 14).
    """
    return live_diagnostics(config)["live"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    """One accounting cell.  Everything a results/*.json row must carry."""

    scheme: str
    weight_bits: int | None = None
    density: float | None = None
    n_idx: int | None = None
    tile_size: int | str | None = None
    nm: tuple[int, int] | None = None
    vq_bits: float | None = None
    budget_bits: float | None = None
    index_model: str = "practical"
    nm_packing: str = "fixed_width"
    group_size: int = DEFAULT_GROUP_SIZE
    scale_bits: int = DEFAULT_SCALE_BITS
    label: str = ""

    def resolved_density(self) -> float | None:
        """Density, solved from budget_bits when the scheme allows it."""
        if self.density is not None:
            return self.density
        if self.scheme == "dense":
            return 1.0
        if self.nm is not None:
            return self.nm[0] / self.nm[1]
        if self.budget_bits is None:
            return None
        return density_for_budget(
            self.scheme,
            self.budget_bits,
            self.weight_bits,
            self.n_idx,
            tile_size=self.tile_size,
            vq_bits=self.vq_bits,
            index_model=self.index_model,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
        )

    def bits_per_position(self) -> float:
        d = self.resolved_density()
        if d is None:
            raise ValueError(f"{self.label or self.scheme}: density is unresolvable")
        return bits_per_position(
            self.scheme,
            d,
            self.weight_bits,
            self.n_idx,
            tile_size=self.tile_size,
            nm=self.nm,
            vq_bits=self.vq_bits,
            index_model=self.index_model,
            nm_packing=self.nm_packing,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
        )

    def weight_cost(self) -> float:
        return total_weight_cost(
            self.weight_bits,
            vq_bits=self.vq_bits,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
        )


# --------------------------------------------------------------------------- #
# Budget-matched grid
# --------------------------------------------------------------------------- #

def default_candidates(
    weight_bits: int,
    n_idx: int,
    *,
    tile_grid: Sequence[int] = (2, 4, 8, 16, 32),
    nm_variants: Sequence[tuple[int, int]] = ((2, 4), (4, 8)),
    vq_bits: float | None = None,
) -> list[Config]:
    """The standard family at one weight_bits: unstructured, the tile ladder,
    the structured edge, and the N:M lattice."""
    kw = dict(weight_bits=weight_bits, n_idx=n_idx, vq_bits=vq_bits)
    if vq_bits is not None:
        kw["weight_bits"] = None
    tag = f"vq{vq_bits:g}" if vq_bits is not None else f"{weight_bits}-bit"
    out = [Config(scheme="unstructured", tile_size=1, label=f"{tag} + unstructured", **kw)]
    out += [
        Config(scheme="tile", tile_size=t, label=f"{tag} + tile-{t}", **kw)
        for t in tile_grid
    ]
    out.append(Config(scheme="structured", label=f"{tag} + T=max (structured)", **kw))
    out += [
        Config(scheme="nm", nm=v, label=f"{tag} + {v[0]}:{v[1]}", **kw)
        for v in nm_variants
    ]
    return out


def budget_matched_grid(
    budget_bits: float,
    tol: float = 0.02,
    *,
    n_idx: int = 11008,
    weight_bits_grid: Sequence[int] = (2, 3, 4),
    tile_grid: Sequence[int] = (2, 4, 8, 16, 32),
    nm_variants: Sequence[tuple[int, int]] = ((2, 4), (4, 8)),
    vq_bits_grid: Sequence[float] = (),
    candidates: Iterable[Config] | None = None,
    apply_live_filter: bool = True,
) -> list[dict]:
    """Every config that sits at `budget_bits`, with a signed offset column.

    `tol` is the inclusion tolerance for FIXED-density schemes, which cannot be
    solved onto the budget and are reported at their own natural cost.  Rows past
    OFFSET_FLAG_THRESHOLD (1%) are flagged.

    `apply_live_filter` drops cells that are not granularity probes (section 3.5).
    Do NOT sweep the full product without it -- degenerate cells get misread.
    Set it False when building a Gate A table, where a coarse-end-dense row is
    still meaningful (see `live_diagnostics`).
    """
    if candidates is None:
        cands: list[Config] = []
        for wb in weight_bits_grid:
            cands += default_candidates(
                wb, n_idx, tile_grid=tile_grid, nm_variants=nm_variants
            )
        for vb in vq_bits_grid:
            cands += default_candidates(
                None, n_idx, tile_grid=tile_grid, nm_variants=(), vq_bits=vb
            )
    else:
        cands = list(candidates)

    rows: list[dict] = []
    for c in cands:
        c = replace(c, budget_bits=budget_bits)
        fixed = c.scheme in FIXED_DENSITY_SCHEMES
        try:
            density = c.resolved_density()
        except (ValueError, NotImplementedError):
            continue
        if density is None or not (0.0 < density <= 1.0):
            continue
        try:
            bits = bits_per_position(
                c.scheme, density, c.weight_bits, c.n_idx,
                tile_size=c.tile_size, nm=c.nm, vq_bits=c.vq_bits,
                index_model=c.index_model, nm_packing=c.nm_packing,
                group_size=c.group_size, scale_bits=c.scale_bits,
            )
        except (ValueError, NotImplementedError):
            continue

        offset = bits - budget_bits
        offset_pct = offset / budget_bits
        if fixed and abs(offset_pct) > tol:
            continue

        diag = live_diagnostics(c)
        if apply_live_filter and not diag["live"]:
            continue

        rows.append(
            {
                "label": c.label or c.scheme,
                "scheme": c.scheme,
                "weight_bits": c.weight_bits,
                "vq_bits": c.vq_bits,
                "tile_size": c.tile_size,
                "nm": c.nm,
                "density": density,
                "bits_per_position": bits,
                "offset": offset,
                "offset_pct": offset_pct,
                "flagged": abs(offset_pct) > OFFSET_FLAG_THRESHOLD,
                "n_idx": c.n_idx,
                "q_over_scales_with_density": Q_OVERHEAD_SCALES_WITH_DENSITY[c.scheme],
                "anchor": budget_bits,
                "live": diag["live"],
                "live_reasons": diag["reasons"],
                "in_bitmap_regime": in_bitmap_regime(
                    budget_bits, c.weight_bits, c.n_idx,
                    tile_size=c.tile_size if c.scheme == "tile" else 1,
                    vq_bits=c.vq_bits,
                ),
            }
        )
    rows.sort(key=lambda r: r["density"])
    return rows


# --------------------------------------------------------------------------- #
# Roofline
# --------------------------------------------------------------------------- #

def roofline_bytes(config: Config, n_params: int) -> int:
    """Weight bytes moved for one batch=1 decode step through `n_params`
    positions.

    Spec v6 section 0.6: at batch=1 decode, time ~ bytes moved / bandwidth, so
    this is the roofline LOWER BOUND for the scheme.  It deliberately excludes
    activations, the KV cache, and any gather overhead -- a scheme with a lower
    bound here has not been shown to be faster, only to move fewer weight bytes.
    """
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    return math.ceil(n_params * config.bits_per_position() / 8)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    failures, _ = doctest.testmod()
    raise SystemExit(1 if failures else 0)
````

## File: rotation.py
````python
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


def kronecker_factors(
    n: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[Tensor | None, Tensor | None]:
    """(A, B) with `kron(A, B)` EXACTLY `structured_orthogonal(n, seed, ...)`.

    Either may be `None` when its factor is trivial: `B` for a power of two,
    `A` for an odd n.  Mirrors `structured_orthogonal`'s seeding, including the
    `seed + 1` it gives the odd factor only when both factors are present --
    getting that wrong would produce a valid rotation that is not THE rotation,
    and every downstream number would be quietly about a different matrix.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    device = torch.device(device)
    a = (n & -n).bit_length() - 1
    p, m = 1 << a, n >> a
    if m == 1:
        return randomized_hadamard(p, seed, dtype, device), None
    if p == 1:
        return None, random_orthogonal(m, seed, dtype, device)
    return (randomized_hadamard(p, seed, dtype, device),
            random_orthogonal(m, seed + 1, dtype, device))


def rotate_hessian(H: Tensor, Q: Tensor | None = None, *,
                   factors: tuple[Tensor | None, Tensor | None] | None = None
                   ) -> Tensor:
    """`Q H Q^T` -- a tile's sub-Hessian moved into the rotated basis.

    With `factors` this contracts against the Kronecker factors instead of
    forming the product densely.  Same quantity, different arithmetic: the dense
    form costs `2 k^3`, this costs `2 k^2 (p + m)`, so at k=2944 = 128*23 it is
    2944/151 = 19x fewer operations.

    Which matters twice over.  It is the largest term in a pass since the scale
    fit stopped being one, AND it is the more ACCURATE of the two in float32:
    the dense form accumulates k terms per output element and this one
    accumulates p then m, so against a float64 reference it is 2.3x closer at
    k=2944 and 2.9x at k=7912.  It gains nothing at a pure power of two, where
    `B` is None and there is no odd factor to peel off -- a fast Hadamard
    transform is what that case would need, and this is not one.

    Not bit-identical to the dense form, which is why it is a caller's choice
    rather than the default (`experiments/m0_rotation_value.py`).
    """
    if factors is None:
        if Q is None:
            raise ValueError("pass either Q or factors")
        return Q @ H @ Q.transpose(-1, -2)

    A, B = factors
    n = H.shape[-1]
    p = A.shape[0] if A is not None else 1
    m = B.shape[0] if B is not None else 1
    if p * m != n:
        raise ValueError(
            f"factors are {p}x{p} and {m}x{m}, which do not span {n} coordinates"
        )
    # Axes of X are (p-row, m-row, p-col, m-col); each contraction rewrites one.
    X = H.reshape(p, m, p, m)
    if A is not None:
        X = torch.einsum("xa,abcd->xbcd", A, X)
    if B is not None:
        X = torch.einsum("yb,abcd->aycd", B, X)
    if A is not None:
        X = torch.einsum("zc,abcd->abzd", A, X)
    if B is not None:
        X = torch.einsum("wd,abcd->abcw", B, X)
    return X.reshape(n, n)


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
````

## File: calibrate.py
````python
"""Sequential layer-wise calibration.

Spec v6 section 4.6 makes sequential calibration mandatory, and section 7 trap 20
says why: statistics must come from the COMPRESSED model, not the dense one.
Once layer L is compressed, layer L+1 sees different activations, so a Hessian
gathered from the dense forward pass describes a model that no longer exists.
`sequential_calibrate` therefore compresses a block and only then runs it to
produce the inputs of the next one.

The other constraint is memory.  For Llama-2-7B a calibration set of 128x2048
tokens through an 11008-wide layer is ~5.8 GiB of activations in fp16, so X is
never materialized.  Only the sufficient statistics are kept:

    H        = sum_t x_t x_t^T      [n_in, n_in]
    act_norm = sqrt(diag(H))        [n_in]

which is all the pipeline needs -- Wanda reads `act_norm`, OBS and LDLQ read `H`,
and the objective itself is a function of H alone:

    ||X (W - W_hat)^T||_F^2 = tr(E H E^T)

`LayerProblem` lives here rather than in the experiment driver because it is the
seam: `from_statistics` for real layers, `synthetic_problem` for smoke tests, and
the same object feeds both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "LayerProblem",
    "synthetic_problem",
    "HessianAccumulator",
    "find_linears",
    "collect_block_statistics",
    "sequential_calibrate",
    "load_calibration_tokens",
]

#: WikiText moved under a namespace; a bare "wikitext" is rejected by current
#: huggingface_hub ("Repository id must be 'namespace/name'").
WIKITEXT_REPO = "Salesforce/wikitext"


# --------------------------------------------------------------------------- #
# The unit of work
# --------------------------------------------------------------------------- #

@dataclass
class LayerProblem:
    """One linear layer plus the calibration statistics it is scored against.

    Build it from raw activations (`LayerProblem(W, X)`) for tests, or from
    accumulated statistics (`LayerProblem.from_statistics`) for real layers,
    where X is far too large to keep.
    """

    W: Tensor                             # [n_out, n_in]
    X: Tensor | None = None               # [n_samples, n_in], optional
    name: str = "layer"
    _H: Tensor | None = None
    _act_norm: Tensor | None = None
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if self.W.ndim != 2:
            raise ValueError(f"W must be 2-D, got {tuple(self.W.shape)}")
        if self.X is not None:
            if self.X.ndim != 2:
                raise ValueError(f"X must be 2-D, got {tuple(self.X.shape)}")
            if self.W.shape[1] != self.X.shape[1]:
                raise ValueError(
                    f"W has {self.W.shape[1]} input channels but X has "
                    f"{self.X.shape[1]}"
                )
        elif self._H is None:
            raise ValueError("give either X or accumulated statistics")

    @classmethod
    def from_statistics(
        cls, W: Tensor, H: Tensor, act_norm: Tensor | None = None,
        name: str = "layer", n_tokens: int = 0,
    ) -> "LayerProblem":
        """For real layers: H and the activation norms, no raw activations."""
        n_in = W.shape[1]
        if H.shape != (n_in, n_in):
            raise ValueError(
                f"H must be ({n_in}, {n_in}) to match W, got {tuple(H.shape)}"
            )
        if act_norm is None:
            act_norm = torch.diagonal(H).clamp_min(0).sqrt()
        return cls(W=W, X=None, name=name, _H=H, _act_norm=act_norm,
                   n_tokens=n_tokens)

    @property
    def n_out(self) -> int:
        return self.W.shape[0]

    @property
    def n_in(self) -> int:
        return self.W.shape[1]

    @property
    def act_norm(self) -> Tensor:
        if self._act_norm is None:
            self._act_norm = self.X.norm(dim=0)
        return self._act_norm

    @property
    def H(self) -> Tensor:
        if self._H is None:
            self._H = self.X.T @ self.X
        return self._H

    def output_error(self, W_hat: Tensor) -> float:
        """Relative layer-output error.

        Computed through H so it works without X:
        ||X E^T||_F^2 = tr(E H E^T), with E = W - W_hat.
        """
        E = self.W - W_hat
        H = self.H
        num = torch.einsum("ij,jk,ik->", E, H, E)
        den = torch.einsum("ij,jk,ik->", self.W, H, self.W)
        return float((num / den).clamp_min(0).sqrt())


def synthetic_problem(
    n_out: int = 128, n_in: int = 256, n_samples: int = 512, seed: int = 0
) -> LayerProblem:
    """Weights and activations shaped like a real layer: correlated input
    channels, a few fat ones, and a heavy-tailed weight distribution."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((n_out, n_in), generator=g, dtype=torch.float64)
    W *= torch.exp(torch.randn((n_out, 1), generator=g, dtype=torch.float64) * 0.5)
    mixing = torch.randn((n_in, n_in), generator=g, dtype=torch.float64) / n_in ** 0.5
    X = torch.randn((n_samples, n_in), generator=g, dtype=torch.float64) @ mixing
    X[:, ::37] *= 8.0
    return LayerProblem(W, X, name=f"synthetic-{n_out}x{n_in}")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

class HessianAccumulator:
    """Accumulates H = sum_t x_t x_t^T over a forward hook.

    Kept in float64 (or float32 on device) regardless of the model's dtype: H is
    a sum over hundreds of thousands of tokens and fp16 loses it.

    `compute_dtype` splits the two jobs the dtype was doing: the PRODUCT is
    `O(tokens * n_in^2)` and wants to be fast, the SUM is `O(n_in^2)` per batch
    and could be accurate.

    MEASURED, AND IT BUYS NOTHING HERE.  Adding float32 products into a float64
    H costs 9% more time and lands on 5.08e-06 against float64's answer, where
    a plain float32 accumulator lands on 5.06e-06 -- the same number.  The error
    is in the PRODUCT, not in the sum: each batch's `x^T x` already rounds
    `sqrt(tokens) * eps` worth of it away, and a wider accumulator cannot undo
    what the multiply discarded.  Splitting the batches finer does not help
    either, since the error over the whole pass goes as `sqrt(total tokens)`
    however it is grouped.

    Kept anyway, and documented, for two reasons: a card with real float64
    throughput would make the trade differently, and it is the first thing
    anyone will reach for on seeing 5e-06 -- better to find the measurement here
    than to repeat it.
    """

    def __init__(self, n_in: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float64,
                 compute_dtype: torch.dtype | None = None) -> None:
        self.n_in = n_in
        self.H = torch.zeros((n_in, n_in), dtype=dtype, device=device)
        self.compute_dtype = compute_dtype or dtype
        self.n_tokens = 0

    def update(self, x: Tensor) -> None:
        """`x` is [..., n_in]; every leading dimension counts as tokens.

        `x` is moved to H's device rather than H to x's: the accumulator is the
        thing that must not be reallocated per batch.
        """
        flat = x.reshape(-1, self.n_in).to(device=self.H.device,
                                           dtype=self.compute_dtype)
        if flat.shape[1] != self.n_in:
            raise ValueError(
                f"expected last dim {self.n_in}, got {flat.shape[1]}"
            )
        self.H += (flat.T @ flat).to(self.H.dtype)
        self.n_tokens += flat.shape[0]

    @property
    def act_norm(self) -> Tensor:
        """||X_j||_2 -- the quantity Wanda scores with."""
        return torch.diagonal(self.H).clamp_min(0).sqrt()


def find_linears(block: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    """Every nn.Linear inside a block, by dotted name."""
    return {
        f"{prefix}{name}" if prefix else name: mod
        for name, mod in block.named_modules()
        if isinstance(mod, nn.Linear)
    }


def collect_block_statistics(
    block: nn.Module,
    inputs: Sequence[Tensor],
    *,
    block_kwargs: dict | None = None,
    dtype: torch.dtype = torch.float64,
    names: Iterable[str] | None = None,
    device: torch.device | str | None = None,
    compute_dtype: torch.dtype | None = None,
) -> dict[str, HessianAccumulator]:
    """Run `block` over `inputs` and accumulate each linear's input Hessian.

    `inputs` is a sequence of batches so the caller controls peak memory; each
    entry is fed through the block once.

    `device` is where the Hessians live and where `x^T x` therefore runs.  It
    used to be hard-wired to the CPU, which meant copying every activation off
    the card it was already on and doing the largest matmul in the calibration
    on the slower processor.  Measured on one Llama-2-7B block over 16,384
    tokens: 22.4 s that way, 0.89 s accumulating on the GPU in float32 -- 25x,
    and this is the single biggest term in a full-model run.  `None` means the
    block's own device.

    Two costs to weigh when overriding it.  The Hessians are resident for the
    whole pass: seven linears of a 7B block are 0.87 GiB at float32 and 1.73 GiB
    at float64, against activations that are already several GiB. And float64
    matmuls are roughly 1/64 rate on a consumer card, so `dtype=float64` alone
    would trade one slow path for another -- `compute_dtype=torch.float32` is
    what makes the accurate accumulator affordable.
    """
    linears = find_linears(block)
    if names is not None:
        keep = set(names)
        linears = {k: v for k, v in linears.items() if k in keep}
    if not linears:
        raise ValueError("block contains no nn.Linear modules")

    if device is None:
        device = next(block.parameters()).device
    accs = {
        name: HessianAccumulator(mod.in_features, device=device, dtype=dtype,
                                 compute_dtype=compute_dtype)
        for name, mod in linears.items()
    }

    handles = []
    for name, mod in linears.items():
        def hook(_mod, args, _out, _name=name):
            # No `.to(...)` here: the accumulator moves what it needs, so a
            # same-device activation is never copied.
            accs[_name].update(args[0].detach())
        handles.append(mod.register_forward_hook(hook, with_kwargs=False))

    block_device = next(block.parameters()).device
    try:
        with torch.no_grad():
            for batch in inputs:
                # ONE batch on the device at a time.  The whole calibration set
                # is 4.0 GiB at the preregistration's 128 x 4096 on a 4096-wide
                # model, against 31 MiB for a single window -- and it would sit
                # there beside the Hessians, the block and the compression's own
                # chunk, on a card with 6.8 GiB usable.  The transfer it costs
                # was measured at about 220 s per point, which is nothing
                # against hours (`docs/STATUS.md` section 7.2).
                block(batch.to(block_device), **(block_kwargs or {}))
    finally:
        for h in handles:
            h.remove()
    return accs


# --------------------------------------------------------------------------- #
# The sequential loop
# --------------------------------------------------------------------------- #

def _block_forward(block: nn.Module, batch: Tensor, block_kwargs: dict | None) -> Tensor:
    out = block(batch, **(block_kwargs or {}))
    return out[0] if isinstance(out, (tuple, list)) else out


def sequential_calibrate(
    blocks: Sequence[nn.Module],
    inputs: list[Tensor],
    compress_fn: Callable[[int, str, LayerProblem], Tensor],
    *,
    block_kwargs: dict | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    compute_dtype: torch.dtype | None = None,
    progress: Callable[[int, str], None] | None = None,
    on_block_done: Callable[[int, list[Tensor], list[dict]], None] | None = None,
    block_offset: int = 0,
) -> list[dict]:
    """Walk the blocks in order, compressing each before moving on.

    For every block: gather statistics from the CURRENT inputs, compress each
    linear, then re-run the block so the next one sees what the compressed model
    actually produces.  Getting that order wrong is Spec v6 trap 20 -- the
    Hessians would describe a model that no longer exists.

    `compress_fn(block_index, name, problem) -> new_weight` is where the pipeline
    plugs in.  `inputs` is a list of batches and is updated IN PLACE, so peak
    memory is one block plus the activations -- and so a run that checkpoints
    can snapshot that list to resume from (`docs/STATUS.md` section 8.1).

    `device` moves the whole walk: the block, the batches, `block_kwargs`, and
    the `LayerProblem` handed to `compress_fn`.  All four have to agree, and
    until 2026-08-25 two of them did not.  `block_kwargs` was never moved at
    all, so the rotary embeddings stayed behind and the block died inside
    `apply_rotary_pos_emb`; and the problem's W was pinned to the CPU while its
    Hessian followed the block, which no argument could reconcile -- so this
    seam could not hand the pipeline a GPU problem at all, though `run_config`
    runs on one perfectly well.  Nothing caught either: every test ran on the
    CPU, where `.cpu()` and a CPU `block_kwargs` are both no-ops.  That is the
    blind spot `docs/STATUS.md` section 14.1 records, hit again by the commit
    that moved the accumulator onto the card.

    `dtype` is the problem's dtype AND the accumulator's, and on a GPU that
    choice is expensive: float64 runs at roughly 1/64 rate here, 29.9 s per
    block against 0.9 s in float32 -- slower even than the CPU float64 it
    replaced (19.7 s).  `experiments/m0_cost_model.py` prices the `cuda_f32`
    arm, so a full-model run wants `dtype=torch.float32`; leaving the default
    in place is 36 days of M1 (15.0 -> 51.0).  The default stays float64
    anyway, because it is the reference the float32 arm's 5.06e-06 is measured
    against and nothing has yet been measured THROUGH this driver -- picking
    the cheaper arm is a decision for whoever runs it, not a side effect.

    `on_block_done(index, inputs, records)` fires after a block is compressed
    AND re-run, which is the only moment a run can be resumed from: by then the
    block is final and `inputs` holds what the NEXT one will see.  That is the
    checkpoint unit `docs/STATUS.md` section 8.1 asks for, and the reason it is a
    callback rather than something the caller does around this function is that
    the caller has no way to observe that moment from outside.

    `block_offset` is added to the index in every record and callback, so a run
    resumed at block 17 reports block 17 rather than block 0.  Without it a
    resumed run's records silently renumber and nothing downstream can tell two
    halves of one point apart.

    Returns one record per compressed layer.
    """
    if not blocks:
        raise ValueError("no blocks to calibrate")
    if not inputs:
        raise ValueError("no calibration batches")

    if device is not None:
        # Imported here rather than at module scope, the way
        # `load_calibration_tokens` treats `datasets`: the walk is generic but
        # the structure is the adapter's.  `to_device` recurses because the
        # rotary entry is a TUPLE of tensors, so a flat comprehension over
        # `block_kwargs` moves half of it and the failure surfaces several
        # frames into transformers.
        from hf_llama import to_device
        block_kwargs = to_device(block_kwargs or {}, device)

    records: list[dict] = []
    for i, block in enumerate(blocks):
        # The block's index IN THE MODEL, which is not `i` when a resumed run
        # hands us `blocks[start:]`.  Everything a caller can see is keyed on
        # this: the progress line, the record, `on_block_done`, the layer name,
        # and `compress_fn`.  The last two used to be keyed on `i`, so a run
        # resuming at block 17 compressed `blocks.0.q_proj` -- one record, two
        # different block numbers in it -- and any `compress_fn` branching on
        # "is this block 0" (m1_run's E8P early warning) fired on the wrong
        # block.  Named once so the two cannot drift apart again.
        block_index = i + block_offset
        if device is not None:
            block.to(device)

        accs = collect_block_statistics(
            block, inputs, block_kwargs=block_kwargs, dtype=dtype,
            compute_dtype=compute_dtype,
        )
        linears = find_linears(block)

        # `list(...)`: each accumulator is dropped from `accs` as its layer
        # finishes, so the dict is mutated while it is walked.
        for name in list(accs):
            acc = accs[name]
            mod = linears[name]
            W = mod.weight.data
            # W follows H, wherever the accumulator put it.  `run_config` reads
            # the two inside single expressions -- `prune` scores W against H,
            # `output_error` contracts E with H -- so a problem split across
            # devices is not slow, it is unusable.
            W_ref = W.to(device=acc.H.device, dtype=dtype)
            problem = LayerProblem.from_statistics(
                W_ref, acc.H, acc.act_norm,
                name=f"blocks.{block_index}.{name}", n_tokens=acc.n_tokens,
            )
            if progress:
                progress(block_index, name)
            new_W = compress_fn(block_index, name, problem)
            if new_W.shape != W.shape:
                raise ValueError(
                    f"{problem.name}: compress_fn returned {tuple(new_W.shape)}, "
                    f"expected {tuple(W.shape)}"
                )
            mod.weight.data = new_W.to(W.dtype).to(W.device)
            records.append({
                "block": block_index,
                "name": name,
                "layer": problem.name,
                "n_in": problem.n_in,
                "n_out": problem.n_out,
                "n_tokens": acc.n_tokens,
                "rel_output_error": problem.output_error(
                    new_W.to(device=acc.H.device, dtype=dtype)),
            })

            # RELEASE THIS LAYER'S HESSIAN NOW, and the measurement is why.
            #
            # A block's seven accumulators are 846 MB at Llama-2-7B's widths,
            # and compressing one layer peaks at 5.4 GiB against 6.8 usable on
            # this card.  Holding all seven for the whole block leaves the
            # allocator evicting and re-requesting rather than reusing, and it
            # costs far more than the bytes suggest: measured on a real block,
            # 122.7 s holding against 84.2 s releasing -- 1.46x -- while the
            # peak moved only 5.40 -> 5.02 GiB.  The win is not the peak, it is
            # the room to reuse.
            #
            # Nothing downstream needs it: `problem` is finished with, the
            # record carries the error, and the next layer builds its own.
            del problem, acc, W_ref, new_W
            accs.pop(name)

        # Only now: the next block must see the COMPRESSED output.  Written
        # back through `inputs[j]`, not into a new list: the caller's list is
        # the checkpoint unit, and rebinding would leave it frozen at block 0.
        target = next(block.parameters()).device
        with torch.no_grad():
            for j, batch in enumerate(inputs):
                out = _block_forward(block, batch.to(target), block_kwargs)
                inputs[j] = out.to(batch.device)

        if device is not None:
            block.to("cpu")

        # Only here: the block is final and `inputs` is what block i+1 sees.
        if on_block_done:
            on_block_done(block_index, inputs, records)

    return records


# --------------------------------------------------------------------------- #
# Calibration data
# --------------------------------------------------------------------------- #

def load_calibration_tokens(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    dataset: str = "c4",
) -> Tensor:
    """`n_samples` random windows of `seqlen` tokens -> [n_samples, seqlen].

    Spec v6 section 6: the seed IS the calibration draw, and results are
    reported over at least three of them.  Gate B needs more than that
    (plan section I1).
    """
    from datasets import load_dataset

    g = torch.Generator().manual_seed(seed)

    if dataset == "wikitext2":
        # WikiText rows are single LINES, and a line almost never reaches 2048
        # tokens -- sampling per row finds nothing.  The reference
        # implementations join the split and cut windows out of the stream, the
        # same way `eval.perplexity.load_eval_tokens` does for the test split.
        raw = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split="train")
        stream = tokenizer("\n\n".join(raw["text"]),
                           return_tensors="pt").input_ids.reshape(-1)
        if stream.numel() <= seqlen:
            raise RuntimeError(
                f"wikitext-2 train has {stream.numel()} tokens, need > {seqlen}"
            )
        starts = torch.randint(stream.numel() - seqlen, (n_samples,), generator=g)
        return torch.stack([stream[s:s + seqlen] for s in starts.tolist()])

    if dataset != "c4":
        raise ValueError(f"unknown dataset {dataset!r}")

    # C4 rows are whole documents, so per-document sampling is right here and is
    # what the reference implementations do.
    raw = load_dataset(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    out = []
    tries = 0
    while len(out) < n_samples:
        tries += 1
        if tries > 100 * n_samples:
            raise RuntimeError(
                f"only found {len(out)}/{n_samples} windows of {seqlen} tokens"
            )
        i = int(torch.randint(len(raw), (1,), generator=g))
        enc = tokenizer(raw[i]["text"], return_tensors="pt").input_ids
        if enc.shape[1] <= seqlen:
            continue
        start = int(torch.randint(enc.shape[1] - seqlen, (1,), generator=g))
        out.append(enc[:, start:start + seqlen])
    return torch.cat(out, dim=0)
````

## File: experiments/m1_gates.py
````python
"""M1 -- the two gates.

Spec v6 section 5.2, re-anchored to the E8P survivor band (plan section H2):
budgets 1.75 / 1.60 / 1.50, all of them below the PTQ floor.

    Gate A (feasibility)  does the best sparse config beat dense low-bit?
    Gate B (the thesis)   is the optimal T interior, or at an edge?

The two are independent on purpose.  Gate A can fail while Gate B holds, and
that outcome narrows the framing rather than stopping the project (Spec v6's
decision table, corrected in plan section B/11).

SCOPE.  This is a layer-level driver.  It measures ||X W^T - X W_hat^T||_F,
which is the objective every method here actually optimizes, and it is a proxy
for perplexity, not a substitute.  Model loading, sequential calibration and
perplexity evaluation are separate deliverables; `LayerProblem` is the seam they
plug into.  `--synthetic` runs the whole grid on generated data as a smoke test.

Gate B is deliberately NOT a bare argmin over T.  With a handful of calibration
draws, the argmin of a noisy curve lands in the interior by chance often enough
to manufacture a positive result (plan section B5).  It is reported as a paired
bootstrap on the differences instead, and stays "undetermined" unless the
interior really separates from both edges.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import accounting as A            # noqa: E402
import compact as C               # noqa: E402
import prune as P                 # noqa: E402
import quantize as Qz             # noqa: E402
import rotation as R              # noqa: E402
import tiling as Tl               # noqa: E402
from calibrate import LayerProblem, synthetic_problem   # noqa: E402,F401

E8P_BITS = Qz.E8P_BITS_PER_WEIGHT          # 2.0

#: Width the LDLQ feedback is confined to.  Not a speed compromise -- the block
#: sweep measured 512 as the BEST arm at every tile size, better than the
#: unconstrained feedback by 11-23% (`docs/STATUS.md` section 5.9).  It is also
#: what makes a useful `chunk` affordable, since a confined factor is k*512 per
#: tile instead of k^2.
HESSIAN_BLOCK = 512
DEFAULT_BUDGETS = (1.75, 1.60, 1.50)
DEFAULT_TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)

# --------------------------------------------------------------------------- #
# What the pipeline runs with (user decision, 2026-08-25)
#
# All three were measured, tested and left OFF on 2026-08-24 because every
# quality number to that point had been taken without them.  Two are on now.
#
# THE THIRD WAS TURNED BACK OFF THE SAME DAY.  fp16 search was on for eight
# hours and then re-measured at the shapes a real block has: 1.00x.  What had
# justified it was a single 512-row layer, and the factor fed to the cost model
# was derived from that rather than measured.  Its docstring below carries the
# numbers, because a rejection is worth more written down than quietly reverted.
#
# The two that remain still owe the same audit: `rotate_kron`'s 5.52x is a
# tile-weighted average from section 6.8 and the compensation's 6.63x is a term
# ratio, and NEITHER has been measured through a real block.  The cost model is
# 5.2x optimistic on one (section 6.16) and derived lever factors are the named
# suspect.
#
# Named here rather than written into the parameter defaults for a reason.
# `run_config` is two things at once -- the pipeline, and the harness the M0
# experiments study it with -- and those want different defaults.  A constant
# the pipeline reads, with the resolved value written into every record, lets
# both be true without either being silent.
#
# TF32 is NOT here and is not a fourth lever: it breaks the pipeline outright
# (rotated sub-Hessian fails Cholesky, 85% of the damping margin gone, section
# 6.9).
# --------------------------------------------------------------------------- #

#: Contract the sub-Hessian rotation against its Kronecker factors instead of
#: forming it densely.  5.52x on the rotation term; on a real layer the quality
#: moves -0.03..-0.31%, i.e. in our favour (section 6.8).
PIPELINE_ROTATE_KRON = True

#: Search the codebook in fp16.  OFF, and this is a rejection rather than a
#: default that happens to be false -- it was on for eight hours on 2026-08-25
#: and every reason for turning it on failed measurement.
#:
#: What justified it: `m0_precision_levers` on 2026-08-24 measured 1.09-1.22x
#: end to end.  That was ONE layer -- `o_proj` at 512 of its 4096 output rows --
#: and the 1.38x term ratio handed to the cost model was derived from the
#: median rather than measured at all.
#:
#: Re-measured on a verified-quiet card at the three shapes a Llama-2-7B block
#: actually has, alternating in one process:
#:
#:      q_proj    4096x4096   fp32  6.22 s   fp16  6.21 s   1.002x
#:      gate_proj 11008x4096  fp32 15.99 s   fp16 15.98 s   1.000x
#:
#: Nothing.  At a 512-row layer the search is a large fraction of the pass and
#: launch-bound, so halving the bytes per launch helps; at a real block's widths
#: the sweep is bandwidth bound and there is nothing left for fp16 to remove.
#: Same shape as every other constant corrected this week -- measured in one
#: regime, applied in all of them.
#:
#: And it is not free elsewhere.  On the CPU it is 4.3x SLOWER (0.55 s a call
#: against 2.34) because the arithmetic is emulated, and it costs up to 0.90%
#: quality (section 6.9).  A lever that buys nothing where it was aimed, loses
#: badly where it was not, and is not free either way has no defence.
#:
#: `search_dtype=torch.float16` still works when asked for explicitly; what is
#: gone is the pipeline choosing it.
PIPELINE_SEARCH_DTYPE = None

#: Defer each block of the compensation sweep's errors into one matmul.  6.63x
#: on the term; not bit-identical, but the difference is 2.7e-06..4.8e-06, which
#: is float32's own epsilon at these sizes (section 6.11c).
PIPELINE_COMPENSATE_BLOCK = 512

#: Fit the chunk's tiles in one pass instead of one apiece
#: (`quantize.fit_scales`).  OFF, and not yet a rejection -- `docs/STATUS.md`
#: section 7.2 recorded it as measured-and-not-taken because it "is not
#: bit-identical", and section 8.6 says that rejection deserves re-pricing now
#: that the codebook term is 52% of the grid (section 6.18).
#:
#: It is the one lever left on that term.  `fit_scale` already batches across
#: CANDIDATES, which fills a pass at the coarse end and not at the fine end: a
#: T=1 tile at k=1024 holds 128 vectors, so its 24 candidates are 3,072 rows and
#: there are 4,096 such tiles in a layer.  T=1 is also the grid's costliest cell
#: (section 6.14) and the thesis's unstructured baseline.
PIPELINE_BATCH_FIT = False

#: "Whatever the pipeline runs with."  A sentinel rather than `None` because
#: `None` is already a meaningful value for two of the three -- it means "no
#: cast" for `search_dtype` and "the exact column-by-column sweep" for
#: `compensate_block` -- and a default that cannot be distinguished from an
#: explicit request is how a caller loses the ability to ask for the old
#: behaviour.
_PIPELINE = object()


# --------------------------------------------------------------------------- #
# One configuration
# --------------------------------------------------------------------------- #

def tile_hessians(
    problem: LayerProblem, cw: C.CompactWeights, Q: Tensor | None = None,
    factors: tuple | None = None,
) -> Tensor:
    """Each tile's input sub-Hessian H[S_t, S_t], optionally rotated to match.

    If the block was rotated as `B Q^T`, the error rotates the same way, so the
    Hessian that keeps the objective invariant is `Q H Q^T`:

        tr((E Q^T)(Q H Q^T)(E Q^T)^T) = tr(E H E^T)
    """
    H = problem.H
    S = cw.idx_index                                     # [n_tiles, k]
    Ht = H[S.unsqueeze(-1), S.unsqueeze(1)]              # [n_tiles, k, k]
    if Q is None:
        return Ht
    if factors is None:
        return Q @ Ht @ Q.transpose(-1, -2)
    return torch.stack([R.rotate_hessian(Ht[t], factors=factors)
                        for t in range(Ht.shape[0])])


def tile_hessian_stream(problem: LayerProblem, cw: C.CompactWeights,
                        Q: Tensor | None = None, factors: tuple | None = None):
    """The same thing, one tile at a time.

    `tile_hessians` builds every tile's sub-Hessian in a single tensor, which is
    fine at test widths and impossible at real ones: a Llama-2-7B `down_proj` at
    T=16 is 256 tiles of 7912 x 7912, or 119 GiB.  LDLQ consumes them strictly in
    order, so nothing is gained by holding them all.

    Returns a callable, because that is what `ldlq_quantize_blocks` takes.
    """
    H = problem.H
    S = cw.idx_index                                     # [n_tiles, k]

    def one(t: int) -> Tensor:
        idx = S[t]
        Ht = H[idx.unsqueeze(-1), idx.unsqueeze(0)]      # [k, k]
        if Q is None:
            return Ht
        # Q is [n_tiles, k, k] when the rotation differs per tile, [k, k] when
        # every tile shares it; both are accepted so the caller need not care.
        q = Q[t] if Q.ndim == 3 else Q
        return R.rotate_hessian(Ht, q, factors=factors)

    return one


def run_config(
    problem: LayerProblem,
    *,
    budget_bits: float,
    tile_size: int | str,
    axis: str = "B",
    metric: str = "wanda",
    compensate: bool = True,
    rotate_axis: str | None = "index",
    rotate_block: int | None = None,
    rotate_kron: bool = _PIPELINE,
    compensate_block: int | None = _PIPELINE,
    hessian_block: int | None = HESSIAN_BLOCK,
    chunk: int | str = "auto",
    scale_sample: int | None = None,
    scale_steps: int = Qz.FIT_STEPS,
    scale_seed: int = 0,
    search_dtype: torch.dtype | None = _PIPELINE,
    batch_fit: bool = _PIPELINE,
    quantize: bool = True,
    ldlq: bool = True,
    align: int | None = None,
    scale: str | float = "per_tile",
    seed: int = 0,
    vq_bits: float = E8P_BITS,
    return_weight: bool = False,
) -> dict:
    """Prune -> compact -> rotate -> quantize, in that order, and measure.

    The order is the invariant (plan H1) and `prune` enforces it.

    `return_weight=True` puts the compressed W under `"W_hat"`.  Off by default
    because the grid runs thousands of configs and keeping a weight per record
    would hold the whole sweep in memory; on, it is what lets this function be
    used as `calibrate.sequential_calibrate`'s `compress_fn`, which has to
    RETURN a weight.  Without it the two halves of the seam do not connect and
    a full-model driver cannot be a thin adapter over them -- which is half of
    why `experiments/m1_run.py` (`docs/STATUS.md` section 8.1) does not exist.

    `ldlq=True` rounds against each tile's sub-Hessian, rotated into the same
    basis as its block.  Without it the rotation costs inference time and buys
    nothing on the activation-weighted objective (plan section I3).  It needs
    the survivor count aligned to 8, so the mask is built with `align=8`.

    `align=None` follows that rule.  Pass a number to force it, which the
    transfer pilot needs: it compares a quantized run against an unquantized one
    at EQUAL DENSITY, and letting the alignment differ between them would move
    the realized density and quietly compare two different sparsity levels.

    `hessian_block` defaults to 512 because the block-width sweep measured that
    as the best arm at every tile size -- better than the unconstrained feedback,
    not merely cheaper (`docs/STATUS.md` section 5.9).  `rotate_block` defaults
    to None for the same reason, from the same measurement: confining the
    ROTATION costs quality at every width tried.  `chunk="auto"` sweeps as many
    tiles together as memory and saturation allow, which is bit-identical to one
    at a time and 5-12x faster.

    `rotate_block` and `hessian_block` are the two halves of the block-diagonal
    proposal (`docs/STATUS.md` section 6.3), and they are separate arguments on
    purpose.  `rotate_block` confines the rotation; `hessian_block` drops the
    sub-Hessian couplings that reach past the same width.  Only the second one
    saves the factorization -- the first is what makes dropping them defensible
    -- so an experiment that moved them together could not say which of the two
    cost the quality.

    `scale_sample` and `scale_steps` cap the per-tile scale fit, which after the
    sweep was chunked was most of what a tile costs -- 83% then, 28% now that
    `fit_scale` batches its candidates.  They default to the full
    fit because that is what every quality number so far was measured under;
    `experiments/m0_scale_fit.py` is what prices moving them.

    `scale="per_layer"` fits the quantizer's scale once from a sample instead of
    once inside every tile.  That sweep was 83% of the pipeline's runtime and is
    28%, so the switch is no longer worth several-fold -- about 1.4 days off M1.
    It is not the default, and now has neither a cost case nor a quality one
    (measured 11% worse, 2026-08-23).
    """
    # The three pipeline levers, resolved before anything reads them, and
    # written into the record below so a row always says what produced it.
    kron_explicit = rotate_kron is not _PIPELINE
    if not kron_explicit:
        rotate_kron = PIPELINE_ROTATE_KRON
    if search_dtype is _PIPELINE:
        search_dtype = PIPELINE_SEARCH_DTYPE
    if compensate_block is _PIPELINE:
        compensate_block = PIPELINE_COMPENSATE_BLOCK
    if batch_fit is _PIPELINE:
        batch_fit = PIPELINE_BATCH_FIT

    # The Kronecker contraction is of the FULL index-axis rotation; a
    # block-diagonal or line-axis one is a different matrix.  Asking for it
    # explicitly there is an error, but inheriting it from the pipeline default
    # must not turn every block-width arm into a crash -- so it resolves off,
    # and `rotate_kron_auto_disabled` says that it did.  Silence is what would
    # be wrong here, not the downgrade.
    kron_incompatible = rotate_axis != "index" or rotate_block is not None
    kron_auto_disabled = False
    if rotate_kron and kron_incompatible:
        if kron_explicit:
            raise ValueError(
                "rotate_kron applies to the full index-axis rotation; "
                "a block-diagonal one is a different matrix"
            )
        rotate_kron, kron_auto_disabled = False, True

    if ldlq and quantize and axis != "B":
        raise NotImplementedError(
            "LDLQ is wired for Axis B, where the compacted block's index axis is "
            "input channels and the Hessian applies directly. Axis A needs the "
            "sweep along its tile's columns instead; pass ldlq=False for now."
        )
    scheme = {1: "unstructured", Tl.MAX_TILE: "structured"}.get(tile_size, "tile")
    requested = A.density_for_budget(
        scheme, budget_bits, None, problem.n_in if axis == "B" else problem.n_out,
        tile_size=tile_size, vq_bits=vq_bits,
    )
    if requested is None or not 0.0 < requested <= 1.0:
        return {"skipped": "budget unreachable at this tile size",
                "budget_bits": budget_bits, "tile_size": tile_size}

    pruned = P.prune(
        problem.W, axis=axis, tile_size=tile_size, density=requested,
        metric=metric, act_norm=problem.act_norm,
        H=problem.H if (compensate or metric == "obs_diag") else None,
        compensate=compensate,
        compensate_block=compensate_block,
        align=(Qz.E8P_DIM if (quantize and ldlq) else 1) if align is None else align,
    )

    W_hat = pruned.W
    if quantize:
        cw = C.compact(pruned.W, pruned.mask)
        rotated, Qm = (R.rotate(cw, axis=rotate_axis, seed=seed,
                                block=rotate_block)
                       if rotate_axis else (cw, None))
        if ldlq:
            # Streamed: at real widths the stacked form is hundreds of GiB.
            n_chunk = (Qz.auto_chunk(cw.n_tiles, cw.lines_per_tile, cw.k,
                                     rotated.blocks.element_size(),
                                     hessian_block)
                       if chunk == "auto" else int(chunk))
            factors = None
            if rotate_kron:
                factors = R.kronecker_factors(
                    cw.k, seed, rotated.blocks.dtype, rotated.blocks.device)
            qb = Qz.ldlq_quantize_blocks(
                rotated.blocks,
                tile_hessian_stream(
                    problem, cw, Qm if rotate_axis == "index" else None,
                    factors=factors),
                scale=scale,
                scale_sample=scale_sample,
                scale_steps=scale_steps,
                scale_seed=scale_seed,
                search_dtype=search_dtype,
                hessian_block=hessian_block,
                chunk=n_chunk,
                batch_fit=batch_fit,
            )
        else:
            qb = Qz.quantize_blocks(rotated.blocks)
        restored = rotated.with_blocks(qb.values)
        if rotate_axis:
            restored = R.unrotate(restored, Qm, axis=rotate_axis)
        W_hat = C.scatter(restored)

    # Realized density differs from the requested one by the per-tile rounding,
    # so the bits are recomputed from what actually happened -- never assumed.
    realized = pruned.mask.density()
    bits = A.bits_per_position(
        scheme, realized, None, pruned.mask.n_idx,
        tile_size=tile_size, vq_bits=vq_bits,
    )
    out = {
        "budget_bits": budget_bits,
        "bits_realized": bits,
        "offset": bits - budget_bits,
        "offset_pct": (bits - budget_bits) / budget_bits,
        "flagged": abs(bits - budget_bits) / budget_bits > A.OFFSET_FLAG_THRESHOLD,
        "scheme": scheme,
        "tile_size": tile_size,
        "axis": axis,
        "n_idx": pruned.mask.n_idx,
        "density_requested": requested,
        "density_realized": realized,
        "vq_bits": vq_bits,
        "q_over_scales_with_density": A.Q_OVERHEAD_SCALES_WITH_DENSITY[scheme],
        "metric": metric,
        "compensate": compensate,
        "rotate_axis": rotate_axis,
        "rotate_block": rotate_block,
        "rotate_kron": rotate_kron,
        "rotate_kron_auto_disabled": kron_auto_disabled,
        "compensate_block": compensate_block,
        "hessian_block": hessian_block,
        "chunk": chunk,
        "quantize": quantize,
        "ldlq": ldlq,
        "scale_policy": scale,
        "scale_sample": scale_sample,
        "scale_steps": scale_steps,
        "scale_seed": scale_seed,
        "search_dtype": None if search_dtype is None else str(search_dtype),
        "batch_fit": batch_fit,
        "align": (Qz.E8P_DIM if (quantize and ldlq) else 1) if align is None else align,
        "survivors_per_tile": int(pruned.mask.survivors_per_tile().max()),
        "seed": seed,
        "rel_output_error": problem.output_error(W_hat),
        "snr_db": Qz.quantization_snr(problem.W, W_hat),
        "in_bitmap_regime": A.in_bitmap_regime(
            budget_bits, None, pruned.mask.n_idx,
            tile_size=tile_size if scheme == "tile" else 1, vq_bits=vq_bits,
        ),
    }
    if return_weight:
        out["W_hat"] = W_hat
    return out


def dense_wall(problem: LayerProblem, seed: int = 0) -> dict:
    """The PTQ floor we claim to go under: dense E8P at its natural 2.0 bits.

    NOT budget-matched with the sparse configs, and that is the point -- the
    comparison is "less than 2 bits against exactly 2 bits".  Reported with the
    offset spelled out so no table can quietly imply otherwise.
    """
    blocks = problem.W.unsqueeze(0)
    qb = Qz.quantize_blocks(blocks)
    W_hat = qb.values[0]
    return {
        "label": "dense E8P (PTQ floor reference)",
        "budget_bits": None,
        "bits_realized": E8P_BITS,
        "density_realized": 1.0,
        "tile_size": None,
        "seed": seed,
        "rel_output_error": problem.output_error(W_hat),
        "snr_db": Qz.quantization_snr(problem.W, W_hat),
    }


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    values: list[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile CI for the mean."""
    if not values:
        raise ValueError("no values to bootstrap")
    v = torch.tensor(values, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(v), (n_boot, len(v)), generator=g)
    means = v[idx].mean(dim=1)
    lo = torch.quantile(means, alpha / 2)
    hi = torch.quantile(means, 1 - alpha / 2)
    return float(lo), float(hi)


def paired_bootstrap_ci(
    a: list[float], b: list[float], n_boot: int = 10000, alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for mean(a - b), resampling the PAIRS.

    Pairing matters: the same calibration draw feeds every tile size, so the
    draw-to-draw noise is shared and largely cancels in the difference.  This is
    the same reason the pre-registration requires tau to be a paired difference
    (plan section B4).
    """
    if len(a) != len(b):
        raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)}, {len(b)}")
    d = torch.tensor(a, dtype=torch.float64) - torch.tensor(b, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(d), (n_boot, len(d)), generator=g)
    means = d[idx].mean(dim=1)
    return float(torch.quantile(means, alpha / 2)), float(
        torch.quantile(means, 1 - alpha / 2)
    )


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def gate_a(records: list[dict], wall: dict, alpha: float = 0.05) -> dict:
    """Does the best sparse config beat the dense low-bit reference?

    A fortiori applies here (Spec v6 section 5.2): winning with the weaker
    saliency settles it; losing only means the stronger one should be tried.
    """
    by_tile: dict[object, list[float]] = {}
    for r in records:
        if "skipped" not in r:
            by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])
    if not by_tile:
        return {"verdict": "no runs"}

    best_tile = min(by_tile, key=lambda t: sum(by_tile[t]) / len(by_tile[t]))
    best = by_tile[best_tile]
    lo, hi = bootstrap_ci(best, alpha=alpha)
    mean = sum(best) / len(best)
    passes = hi < wall["rel_output_error"]
    return {
        "verdict": "pass" if passes else "fail",
        "best_tile": best_tile,
        "best_mean_error": mean,
        "best_ci": (lo, hi),
        "wall_error": wall["rel_output_error"],
        "wall_bits": wall["bits_realized"],
        "note": (
            "sparse budget is below the wall's 2.0 bits; this is a "
            "cheaper-and-better claim, not a budget-matched one"
        ),
    }


def t_star_set(records: list[dict], alpha: float = 0.05) -> dict:
    """Which granularities are NOT distinguishable from the best one.

    Gate B's verdict and the headline T* are different claims with different
    evidence behind them, and the second is the weaker one.  Separating the
    optimum from the EDGES is a large difference; separating it from its
    NEIGHBOUR is a small one, and the power analysis
    (`experiments/m0_gate_b_power.py`) puts numbers on the gap: with a flat
    interior at one sigma and twenty draws, the verdict is right 76% of the time
    while the argmin is right 48% -- barely better than picking between the two
    tiles nearest the bottom.

    So T* is reported as a SET: the argmin, plus every interior tile whose
    paired difference from it cannot be shown to be positive.  A one-element set
    is a real claim about granularity; a four-element set says the curve is flat
    and the honest headline is "interior", not "T = 8".

    The test is one-sided by construction: every other tile has a mean at or
    above the argmin's, so only the lower end of the interval can settle
    anything.  Reading one end of a two-sided interval at `alpha_eff` makes the
    effective level `alpha_eff / 2`, i.e. the set errs toward being too large.
    That is the right direction to err -- an over-wide set understates the
    claim, a too-narrow one manufactures a granularity result.
    """
    by_tile: dict[object, list[float]] = {}
    for r in records:
        if "skipped" not in r:
            by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])

    tiles = [t for t in by_tile if t not in (1, Tl.MAX_TILE)]
    if not tiles:
        return {"t_star": None, "set": [], "reason": "no interior tiles"}

    means = {t: sum(by_tile[t]) / len(by_tile[t]) for t in tiles}
    t_star = min(tiles, key=lambda t: means[t])
    alpha_eff = alpha / max(1, len(tiles) - 1)

    keep, detail = [t_star], {}
    for t in tiles:
        if t == t_star:
            continue
        lo, hi = paired_bootstrap_ci(by_tile[t], by_tile[t_star], alpha=alpha_eff)
        separated = lo > 0.0
        detail[str(t)] = {"ci": (lo, hi), "separated_from_t_star": separated}
        if not separated:
            keep.append(t)

    return {
        "t_star": t_star,
        "set": sorted(keep, key=lambda t: (t == Tl.MAX_TILE, t)),
        "n_candidates": len(tiles),
        "alpha_effective": alpha_eff,
        "detail": detail,
        "note": ("a set larger than one means the granularity axis is flat "
                 "near the optimum; report the set, not the argmin"),
    }


def gate_b(records: list[dict], alpha: float = 0.05, min_seeds: int = 5) -> dict:
    """Is the optimum T interior, or at an edge?

    Two corrections stand between this and a false positive, and BOTH are load
    bearing -- without them the gate reports 'interior' on data with no effect
    in it at all (see tests/test_m1_gates.py):

    1. Selection.  T* is chosen as the argmin over the interior tile sizes and
       then tested on the same draws, so the test is Bonferroni-corrected by the
       number of candidates.  Skipping this is double dipping.

    2. Too few draws.  A percentile bootstrap over three calibration draws
       resamples three numbers; its 95% interval does not have 95% coverage.
       Below `min_seeds` the honest verdict is 'undetermined', not a p-value.

    Note this puts Gate B in tension with Spec v6 section 6, which asks only for
    seeds >= 3.  Three is enough to report a mean; it is not enough to decide
    this gate.
    """
    by_tile: dict[object, list[float]] = {}
    for r in records:
        if "skipped" not in r:
            by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])

    tiles = [t for t in by_tile if t not in (1, Tl.MAX_TILE)]
    if not tiles or 1 not in by_tile or Tl.MAX_TILE not in by_tile:
        return {"verdict": "undetermined", "reason": "both edges must be present"}

    means = {t: sum(v) / len(v) for t, v in by_tile.items()}
    t_star = min(tiles, key=lambda t: means[t])

    n_draws = min(len(v) for v in by_tile.values())
    if n_draws < min_seeds:
        return {
            "verdict": "undetermined",
            "reason": (
                f"{n_draws} calibration draws is too few for a bootstrap CI; "
                f"need >= {min_seeds}"
            ),
            "t_star": t_star,
            "means": {str(k): v for k, v in means.items()},
        }

    # Bonferroni over the interior candidates we selected T* from.
    alpha_eff = alpha / max(1, len(tiles))
    lo_f, hi_f = paired_bootstrap_ci(by_tile[t_star], by_tile[1], alpha=alpha_eff)
    lo_c, hi_c = paired_bootstrap_ci(
        by_tile[t_star], by_tile[Tl.MAX_TILE], alpha=alpha_eff
    )
    beats_fine, beats_coarse = hi_f < 0.0, hi_c < 0.0

    if beats_fine and beats_coarse:
        verdict = "interior"
    elif means[t_star] >= min(means[1], means[Tl.MAX_TILE]):
        verdict = "edge"
    else:
        verdict = "undetermined"
    return {
        "verdict": verdict,
        "t_star": t_star,
        "means": {str(k): v for k, v in means.items()},
        "vs_T1_ci": (lo_f, hi_f),
        "vs_Tmax_ci": (lo_c, hi_c),
        "beats_fine": beats_fine,
        "beats_coarse": beats_coarse,
        "n_draws": n_draws,
        "alpha_effective": alpha_eff,
        "note": "argmin alone is not evidence; both edges must be separated",
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class GateRun:
    """The M1 grid: budgets x tile sizes x draws.

    A DRAW is a calibration draw -- a different sample of text the layer sees --
    and that is what `run` wants: pass a sequence of `LayerProblem`s, one per
    draw, sharing a fixed rotation seed.

    Passing a single problem instead falls back to varying the ROTATION seed,
    which is a different and much smaller noise source: measured on a synthetic
    layer at 0.72% of the error level against 1.41% for calibration draws
    (`experiments/m0_gate_b_power.py`).  Gate B run on rotation seeds would
    therefore be roughly twice as confident as the evidence supports, so the
    fallback records what it did and `gate_b`'s output is marked.
    """
    budgets: tuple = DEFAULT_BUDGETS
    tiles: tuple = DEFAULT_TILES
    seeds: tuple = (0, 1, 2)
    axis: str = "B"
    metric: str = "wanda"
    compensate: bool = True
    rotate_axis: str | None = "index"
    records: list = field(default_factory=list)

    def run(self, problem: LayerProblem | Sequence[LayerProblem]) -> dict:
        problems = [problem] if isinstance(problem, LayerProblem) else list(problem)
        if not problems:
            raise ValueError("need at least one LayerProblem")
        draw_axis = "calibration" if len(problems) > 1 else "rotation_seed"
        # One draw per problem when problems vary; otherwise one per seed.
        draws = ([(p, self.seeds[0]) for p in problems] if draw_axis == "calibration"
                 else [(problems[0], s) for s in self.seeds])
        wall = dense_wall(problems[0])
        out = {
            "meta": {
                "git": _git_hash(),
                "utc": datetime.now(timezone.utc).isoformat(),
                "layer": problems[0].name,
                "n_out": problems[0].n_out,
                "n_in": problems[0].n_in,
                "axis": self.axis,
                "metric": self.metric,
                "compensate": self.compensate,
                "rotate_axis": self.rotate_axis,
                "seeds": list(self.seeds),
                "draw_axis": draw_axis,
                "n_draws": len(draws),
                "survivor_quantizer": "E8P",
                "vq_bits": E8P_BITS,
            },
            "wall": wall,
            "budgets": {},
        }
        for b in self.budgets:
            recs = []
            for t in self.tiles:
                for prob, s in draws:
                    r = run_config(
                        prob, budget_bits=b, tile_size=t, axis=self.axis,
                        metric=self.metric, compensate=self.compensate,
                        rotate_axis=self.rotate_axis, seed=s,
                    )
                    r["draw_axis"] = draw_axis
                    recs.append(r)
            self.records.extend(recs)
            out["budgets"][str(b)] = {
                "records": recs,
                "gate_a": gate_a(recs, wall),
                "gate_b": dict(gate_b(recs), draw_axis=draw_axis),
                "t_star_set": t_star_set(recs),
                "live": A.is_live(
                    A.Config(scheme="tile", vq_bits=E8P_BITS,
                             n_idx=problems[0].n_in, tile_size=16, budget_bits=b)
                ),
            }
        return out


def _report(out: dict) -> None:
    m = out["meta"]
    print(f"layer {m['layer']}  axis={m['axis']}  metric={m['metric']}  "
          f"quantizer=E8P({m['vq_bits']} bit)  "
          f"{m.get('n_draws', len(m['seeds']))} draws over "
          f"{m.get('draw_axis', 'rotation_seed')}")
    print(f"PTQ floor reference: dense E8P @ {out['wall']['bits_realized']} bit  "
          f"-> rel.err {out['wall']['rel_output_error']:.4f}\n")
    for b, blk in out["budgets"].items():
        print(f"=== B = {b} bit {'(live)' if blk['live'] else '(NOT live)'} ===")
        seen = set()
        for r in blk["records"]:
            if "skipped" in r or r["tile_size"] in seen:
                continue
            seen.add(r["tile_size"])
            same = [x["rel_output_error"] for x in blk["records"]
                    if x.get("tile_size") == r["tile_size"] and "skipped" not in x]
            print(f"  T={str(r['tile_size']):<4} d={r['density_realized']:.4f}  "
                  f"bits={r['bits_realized']:.4f} ({r['offset_pct']*100:+.2f}%)  "
                  f"rel.err={sum(same)/len(same):.4f}")
        ga, gb = blk["gate_a"], blk["gate_b"]
        print(f"  Gate A: {ga['verdict']}  (best T={ga.get('best_tile')}, "
              f"{ga.get('best_mean_error', float('nan')):.4f} vs wall "
              f"{ga.get('wall_error', float('nan')):.4f})")
        ts = blk["t_star_set"]
        members = ", ".join(str(t) for t in ts["set"])
        print(f"  Gate B: {gb['verdict']}  (T*={gb.get('t_star')}; "
              f"not separable from it: {{{members}}})\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="run on generated data as a smoke test")
    ap.add_argument("--n-out", type=int, default=128)
    ap.add_argument("--n-in", type=int, default=256)
    ap.add_argument("--draws", type=int, default=3,
                    help="calibration draws -- the axis Gate B's CIs are over")
    ap.add_argument("--rotation-seeds-as-draws", action="store_true",
                    help="replicate over the rotation seed instead; measured at "
                         "about half the noise, so Gate B comes out overconfident")
    ap.add_argument("--budgets", type=float, nargs="*", default=list(DEFAULT_BUDGETS))
    ap.add_argument("--axis", default="B", choices=["A", "B"])
    ap.add_argument("--no-compensate", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.synthetic:
        print("only --synthetic is wired up: model loading and sequential "
              "calibration are separate deliverables. LayerProblem is the seam.",
              file=sys.stderr)
        return 2

    run = GateRun(
        budgets=tuple(args.budgets), seeds=tuple(range(args.draws)), axis=args.axis,
        compensate=not args.no_compensate,
        rotate_axis=None if args.no_rotate else "index",
    )
    if args.rotation_seeds_as_draws:
        out = run.run(synthetic_problem(args.n_out, args.n_in))
    else:
        out = run.run([synthetic_problem(args.n_out, args.n_in, seed=d)
                       for d in range(args.draws)])
    _report(out)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
````

## File: quantize.py
````python
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
````

## File: README.md
````markdown
# subfloor

*English · [Türkçe](README.tr.md)*

**Sparsity Below the Quantization Floor** — jointly optimising
`(survivor quantizer, granularity, density)` once the bit budget drops below the
PTQ floor.

> **Status.** The pipeline runs end to end against a real Llama-2-7B: dense
> perplexity reproduces the published figure to within 0.006, the full-model
> driver compresses real weights, and an interrupted run lands where an
> uninterrupted one does. **The compressed model's perplexity has never been
> measured** — every quality number below is either a layer-level proxy or
> synthetic data. Details in [`docs/STATUS.md`](docs/STATUS.md).

---

## The question

Dense post-training quantization has a practical floor around 2 bits. Below it,
things fall apart: on Llama-2-7B, QuaRot-GPTQ at 2 bits gives **22.07** ppl and
QuIP# at 2 bits gives **6.66**, against a dense 5.47. Sparsity has a floor too,
but it is set by the **index format** — a bitmap cannot go below 1 bit per
position, while an index shared across a tile falls to `1/T`.

Why that matters: the bit budget *is* context length. Llama-2-70B on a 24 GiB
card, 22.5 GiB usable:

| bits/position | context that fits |
|---|---|
| 2.0 (the PTQ floor) | ~15.6k |
| 1.5 | ~28.4k |

Going below 2 bits means doubling the context.

## The core identity

With `W` bits per survivor, `d` density and `T` the tile size, in the bitmap
regime:

```
d(T) − d(1) = (1 − 1/T) / W               ← independent of the budget, CONSTANT
[d(T) − d(1)] / [d(∞) − d(1)] = 1 − 1/T   ← independent of W as well
```

The absolute advantage is constant. The ratio grows only because the denominator
shrinks — reading that as "the advantage grows with the budget" is a common and
wrong interpretation.

The lever grows as `W` falls: `0.2256` at GPTQ-4bit, **`0.4688`** at lattice VQ
(`W = 2.0`). Which is why the choice of survivor quantizer decides the budget
regime rather than merely improving it.

## The design invariant

```
score → select mask (in the UNROTATED basis) → freeze
      → compact → rotate → LDLQ → compensate
```

**The mask is always selected in the unrotated basis.** Rotation flattens the
magnitude distribution, and pruning feeds on concentrated energy; pruning in the
rotated basis destroys the model (QuaRot+Wanda at 50% sparsity → **5868 ppl**,
OBR Table 1).

But this is an *ordering* problem, not a prohibition. Once the mask is frozen,
rotation cannot spoil it — the selection has already happened. `prune()`
**raises** if called in the wrong order; it is not left to convention.

---

## Install and run

`transformers >= 5` is required: `hf_llama.load_llama` calls
`from_pretrained(dtype=...)`, which is a v5 keyword. On v4 it fails several
frames deep with an unrelated-looking message.

```bash
pip install "torch" "transformers>=5" datasets numpy pytest
python -m pytest -q                    # 674 tests
```

Synthetic smoke test — the whole pipeline including both gates, no GPU needed:

```bash
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --draws 3 --budgets 1.5
```

Real model, one grid point (calibrate → compress → evaluate, checkpointed at
block granularity):

```bash
python -u experiments/m1_run.py --budget 1.5 --tile 16 --calib-seqlen 2048
```

Re-measure this machine's constants — **mandatory on a different card**, because
every timing constant here says "measured on this machine" and means it:

```bash
python experiments/m0_cost_model.py
python -u experiments/m0_lever_audit.py --build --rot-sweep
```

To run on a free cloud GPU: [`cloud/README.md`](cloud/README.md).

---

## Layout

| Module | Job |
|---|---|
| `accounting.py` | bit budgets, the `1−1/T` identity, the `B*` wall, the live-band filter |
| `scoring.py` | saliency — two per-weight metrics, two aggregation directions |
| `tiling.py` | tile partition, frozen mask (`T=1` unstructured, `T=max` structured) |
| `prune.py` | mask selection + forward compensation; asserts the design invariant |
| `compact.py` | gather survivors into dense per-tile blocks |
| `rotation.py` | mask-preserving rotation, `kron(RHT(2^a), orthogonal(m))` |
| `quantize.py` | QuIP# E8P codebook + LDLQ (Hessian-aware rounding) |
| `calibrate.py` | sequential calibration; statistics come from the **compressed** model |
| `eval/perplexity.py` | ppl + protocol guard |
| `eval/streamed.py` | layer-streamed ppl for a model that does not fit the GPU |
| `hf_llama.py` | HuggingFace adapter — captures what block 0 receives |
| `experiments/m1_gates.py` | M1's two gates; the pipeline itself (`run_config`) |
| `experiments/m1_run.py` | **full-model driver** — block-granular checkpoint and resume |
| `experiments/m0_cost_model.py` | what a real run costs, from measured curves |
| `experiments/bench_guard.py` | asserts, by **raising**, that the card is idle enough to time on |
| `cloud/` | run one point on a free cloud session; the pipeline gets no additions |

The remaining `experiments/m0_*.py` files are individual measurements: the value
of rotation, the scale fit, the precision levers, the tile timings, the lever
audit. Each docstring carries what it measured, why, and where it was wrong.

**Documents:**

- [`docs/STATUS.md`](docs/STATUS.md) — **start here.** What is verified, what is
  assumed, why each decision was taken, what is next, and which environment traps
  cost hours
- [`docs/spec_v7.md`](docs/spec_v7.md) — the specification. The maths, the
  accounting and the protocol are binding
- [`preregistration.md`](preregistration.md) — M1's preregistration. **Not frozen
  yet**; §9 lists what is missing
- [`docs/audit.md`](docs/audit.md) — the pre-M0 audit of v6. Kept as written: the
  record of what was known when each decision was made
- [`docs/gate_a_dry_run.md`](docs/gate_a_dry_run.md) — Gate A rehearsed against
  the literature before spending any GPU time

---

## What is verified, what is assumed

This distinction has to be made explicitly.

**Verified (tested):**

- All of the accounting. The golden constants are derived independently in exact
  rational arithmetic; `accounting.py` computes them through a general dispatch.
  Two routes, one answer
- The E8P codebook construction — 227+29 source patterns by enumeration, 2¹⁶
  distinct codewords, lattice membership, **exactly 2 bits per weight**
- **The cost side of `vq_bits = 2.0`, from real checkpoints** — in the QuIP# E8P
  and QTIP releases the codeword payload is exactly 2.000000, and 2.005204 /
  2.006740 with per-layer side information. The manifest arithmetic and the total
  file size agree exactly (`experiments/m0_vq_bits.py`)
- That rotation preserves the mask (on both axes, at every `T`)
- That compensation is forward-only, and that its gain comes from channel
  correlation
- That calibration reads the compressed model — on synthetic blocks **and on a
  real Llama**
- That the adapter reproduces the model's own computation exactly (hand-driven
  blocks → the model's logits, 1e-5)
- That Gate B does **not** say "interior" on noise
- **The `Δ = Q + τ` transfer bias** — the predictor was built and run alongside
  the real pipeline. The `T=1` identity check is exactly zero; the bias exceeds
  draw noise by 12.3×, so the preregistration's tolerance cannot be derived from
  seed variance
- **Gate B's statistical power** — 5 draws detect 2.29 σ and the measured effect
  is 6.7 σ. But neighbouring tiles (0.31 σ) do not separate, which is why `T*` is
  reported as a **set** rather than a point (`experiments/m0_gate_b_power.py`)
- **The value of rotation on a real layer** — synthetic data read 3%, the real
  `o_proj` reads **−70%**. The synthetic measurement was two orders of magnitude
  out, because rotation's job is to spread a heavy tail and synthetic data has no
  tail (`experiments/m0_rotation_value.py`)
- **The full-model driver and its resume** — real Llama-2-7B blocks were
  compressed, a checkpoint was written, and the test is not "resume works" but
  **"resume is invisible in the answer"**: an interrupted run lands on the
  uninterrupted one's perplexity. The alternative would be calibrating block 17
  against activations no version of the model ever produced — and that does not
  crash, it is quietly wrong
- **Two precision levers audited through a real block** — each measured twice
  (the term alone, and the term inside the pipeline), and in six comparisons out
  of six the in-situ saving is 95–107% of the isolated one. A third lever (fp16)
  **failed** the same audit and was withdrawn
  (`experiments/m0_lever_audit.py`)

**Assumed — not verified:**

> That E8P holds its 2-bit quality on the **compacted survivor submatrix**.
> Survivors are by definition the heavy tail of the distribution, and a lattice
> quantizer wants near-Gaussian input. Rotation should fix that, but it has not
> been shown.
>
> Measurement closed the **cost** side, not the quality side: it is certain that
> 2 bits are paid, and open whether 2 bits of *quality* are received for them.

The cheap experiment that would test this assumption was deliberately skipped.
The early-warning rule and the fallback are defined in `preregistration.md` §9.1.

**The first real measurement (2026-08-21).** Llama-2-7B dense perplexity,
WikiText-2, layer-streamed on an 8 GB card:

| seqlen | measured | published | difference |
|---|---|---|---|
| 2048 | **5.4675** | 5.47 | −0.0025 |
| 4096 | **5.1143** | 5.12 | −0.0057 |

This gives two things at once: the pipeline reproduces published numbers, **and**
the 5.12/5.47 split in the literature is confirmed to be a sequence-length
artefact — a hypothesis that was recorded before the measurement.

**But compression quality is still unmeasured.** Outside the dense baseline,
every number is either a layer-level `tr(E H Eᵀ)` proxy or synthetic data. The
synthetic smoke test does produce a U-shaped error curve and Gate A does pass —
**but we generated the data, so that is not evidence for the thesis.**

---

## Cost — and the cost model is itself a result

The M1 grid (3 budgets × 7 tile sizes × 5 draws) first priced out at **120 days**
on this machine. It is now **11.7 days**, and most of the difference is not
faster code but a more accurate model:

| | what happened |
|---|---|
| 120 → 12 days | the Cholesky rate was measured at k=2048 and applied at every width; at real widths it overcharged **9.4×** |
| 12 → ~40 days | two terms the model **did not know existed** were found: calibration and forward compensation. The number went **up** once |
| ~40 → 15 days | those two terms were fixed (Hessians on the block's own device: **25×**) |
| 15 → 11.7 days | the model began pricing the arithmetic the pipeline **actually runs** |

The last one was of this kind: `rotation_seconds` had no Kronecker path at all
and was billing the dense form the pipeline had stopped running. It was
**pessimistic**, which is why nothing ever complained.

The model has been wrong nine times, and **seven of those were missing terms
rather than wrong ratios.** So the question to ask this model is not "is the
ratio right" but **"what is not on the list"**.

**Measurement hygiene is this project's real output**, and it lives in
`docs/STATUS.md` §14. The traps that kept landing: a constant measured in one
regime and applied in all of them, a test that watches the answer instead of the
path, a composition nobody ever ran, and a measurement that does not traverse the
path it changed. As rules: **a test must watch the path, not the answer**, and
**a new test is not accepted until it has been shown red against the old code**.

---

## Gaps

- **The compressed model's perplexity.** This is the critical path; the driver
  exists, it has not been run
- **The E8P assumption** (above) — the project's single largest risk
- **Freezing the preregistration.** The `Δ(T)` prediction curve and `T*_predicted`
  are open; both depend on the `τ` sweep and the sweep script is not written
- **The driver's context cost.** The arithmetic is priced to within 1.03× on a
  real layer, but the driver runs the same seven layers far slower, and the
  difference is hitting the memory ceiling — compression peaks at 5.4 GiB on an
  8 GiB card
- **LDLQ for Axis A** — currently `NotImplementedError`

---

## Licence

[Apache License 2.0](LICENSE). Chosen over MIT for the explicit patent grant:
this repository's contribution is a *method*, and MIT is silent on patents, which
leaves anyone using the code unprotected if a third party later claims one
nearby.

Nothing here is vendored. QuIP#, QTIP, QuaRot, SparseGPT, Wanda, GPTQ, VENOM and
OBR are all reimplemented from their papers — the E8P codebook by enumeration,
the accounting from the identity — so no upstream licence obligation is
inherited.

---

## Notes

Test code is **1.4×** the size of production code (5824 against 4115 lines, 674
tests) and that is deliberate: most of the code carries a mathematical claim that
has to be verified, and the most expensive class of error here is silently
producing a wrong number. Most of the tests check a **claim** rather than a
behaviour — where the identity is valid, how the codebook is constructed, that
rotation preserves the mask, that the gate does not pass on noise.

`tests/golden.py` is a special file — it does **not** import `accounting.py`. A
test that calls the thing it is checking proves nothing, so the same numbers are
reached along two independent routes.

Parts of this work were developed with AI assistance.
````

## File: docs/STATUS.md
````markdown
# Durum ve Devir Belgesi

> **Bağlam kaybolduğunda projeye kaldığı yerden devam edebilmek için var.**
> Kod ne yaptığını söyler; bu belge **neden öyle olduğunu** söyler.
> Son güncelleme: 2026-08-25 · HEAD `515eb35` · Testler: **632 geçiyor, 6 atlanıyor**
> Bu oturumun ölçüm dersleri **§14**'te — hız kazançlarından daha taşınabilir.

---

## 1. Nerede duruyoruz — beş cümle

Hat uçtan uca çalışıyor, gerçek Llama-2-7B'ye bağlı, ve gerçek ağırlıklar
üzerinde üç ölçüm var: dense perplexity (yayımlanmıştan 0.006 içinde),
rotasyonun katman değeri (**−70%**), ve blok genişliğinin etkisi. M0'ın
uçuş-öncesi kalemleri kapandı. **Maliyet artık bağlayıcı kısıt değil:** M1 bu
makinede 120 günden **11.7 güne**, `τ` süpürmesi 29 günden **4.2 güne** indi — yani
ön-kaydı bloke eden şey ortadan kalktı. Son adım 08-25'te modelin **hattın
koştuğu aritmetiği** fiyatlamasıyla geldi (15.0 → 11.7), ve model o gün gerçek
bir blokta 1.03× doğrulukla sınandı (§6.18). Sayının bir kez **yukarı** gittiğine
dikkat: 12'ye inmişti, sonra modelin hiç yazmadığı iki terim bulununca gerçek
maliyetin ~40 olduğu anlaşıldı, ve 15'e o terimler düzeltilerek inildi (§6.2). Ama **sıkıştırılmış modelin
perplexity'si hâlâ hiç ölçülmedi**; Kapı A'nın ve Kapı B'nin tek bir gerçek
verisi yok. Ve bunun sebebi bilimsel bir karar değil: **tam modeli sıkıştıran
deney betiği hiç yazılmadı.**

---

## 2. Proje 60 saniyede

**Soru.** Yoğun PTQ'nun pratik tabanı ~2 bit; altında çöküyor (QuaRot-GPTQ
2-bit → 22.07 ppl). Seyrekliğin tabanı ise **indeks formatına** bağlı: bitmap
1 bit/pozisyonun altına inemez, ama `T` satırın paylaştığı bir indeks `1/T`'ye
iner. 2 bitin altındaki bütçelerde `(survivor quantizer, granularity, density)`
üçlüsü nasıl seçilmeli?

**Neden önemli.** Bit bütçesi doğrudan bağlam uzunluğudur. Llama-2-70B, 24 GiB
kart: 2.0 bit → ~15.6k bağlam, 1.5 bit → ~28.4k.

**Çekirdek özdeşlik.** Bitmap rejiminde `d(T) − d(1) = (1 − 1/T)/W` —
**bütçeden bağımsız sabit**. Oranın büyümesi paydanın küçülmesinden; "oran
büyüyor" yaygın ve yanlış bir okuma. Kaldıraç `W` küçüldükçe büyür: GPTQ-4bit
`0.2256`, E8P (`W=2.0`) **`0.4688`**.

**Tasarım değişmezi (H1).**
```
skorla → maske seç (döndürülmemiş bazda) → dondur
       → kompaktla → döndür → LDLQ → telafi
```
Maske **her zaman** döndürülmemiş bazda seçilir. Rotasyonlu bazda budama modeli
yok ediyor (QuaRot+Wanda %50 → **5868 ppl**, OBR Tablo 1). `prune()` yanlış
sırayla çağrılırsa hata fırlatır.

**İki kapı.** Kapı A: en iyi seyrek konfigürasyon PTQ tabanını (QTIP 2-bit)
geçiyor mu? Kapı B: optimum `T` içeride mi, uçta mı? **Bağımsızdırlar** — A
düşerken B ayakta kalabilir, ve o durumda çerçeve daralır, proje durmaz.

---

## 3. Ne doğrulandı, ne varsayıldı, ne hiç ölçülmedi

### 3.1 Doğrulanmış

| Ne | Nasıl |
|---|---|
| Muhasebenin tamamı | `tests/golden.py` tam kesirli aritmetikle bağımsız türetiyor, `accounting.py` genel dispatch ile hesaplıyor. İki yol, tek cevap |
| E8P codebook | 227+29 kaynak örüntüsü enumerasyonla, 2¹⁶ ayrık kodsözcüğü, kafes üyeliği, **tam 2 bit/ağırlık** |
| Rotasyonun maskeyi koruduğu | Her iki eksende, her `T`'de, destek birebir |
| Telafinin ileriye-only olduğu | Son sütunu budayınca öndekiler değişmiyor |
| Telafinin kanal korelasyonundan beslendiği | Bağımsız kanallarda oran 0.7+, korelasyonlu 0.13 |
| Kalibrasyonun **sıkıştırılmış** modeli okuduğu | Sentetik bloklarda ve gerçek Llama'da |
| Adaptörün modelin hesabını yeniden ürettiği | Elle sürülen bloklar → modelin logit'leri, 1e-5 |
| Akışlı eval == tam model eval | 1e-6 |
| Kapı B'nin gürültüde geçmediği | 6 ayrı gürültü çekilişi |
| **Dense ppl (gerçek model)** | 2048 → 5.4675, 4096 → 5.1143; yayımlanmıştan <0.006 |
| **`vq_bits = 2.0`'ın maliyet tarafı** | QuIP# E8P ve QTIP release'lerinin manifest'i. Yük **tam 2.000000**, yan bilgiyle 2.005204 / 2.006740. Manifest ve dosya boyutu iki bağımsız yol, tam aynı sayı |
| **Kapı B'nin istatistiksel gücü** | 600 denemelik simülasyon, **gerçek `gate_b` çağrılarak**. 5 çekiliş 2.29 σ saptıyor; ölçülen etki 6.7 σ |
| **Transfer sapması** | `Δ = Q + τ` tahmin edicisi gerçek hattın yanında koşuldu. `T=1` kimlik kontrolü **tam sıfır**; sapma çekiliş gürültüsünü **12.3×** aşıyor |
| **Rotasyonun değeri (gerçek katman)** | `o_proj`, gerçek ağırlıklar + 32,768 gerçek token: **−70.1% ortalama** (§5.1) |
| **Kronecker kongrüansının kalite bedeli (gerçek katman)** | Aynı kurulum, `full` kolu birinci turu 7.0e-07 ile üretiyor. Hattın kolunda −0.03…−0.31%, tam genişlik geri beslemede +0.5…+1.9% (§6.8) |
| **Blok genişliğinin etkisi** | 51 kol, 5 genişlik × 3 aile × 3 tile (§5.7) |
| **Kafes çözücünün taramaya denkliği** | `nearest_e8p` ile kaba kuvvet, dört ölçekte birebir aynı indeks |
| **Analitik aramanın taramaya denkliği** | float64'te bir milyon vektörde **sıfır** uyuşmazlık; 65,536 kodsözcüğünün hepsi kendine çözülüyor (§6.4) |
| **Derlenmiş ile eager'ın denkliği** | Üç ölçek × üç satır sayısı (düzensiz dâhil), `torch.equal` |
| **Toplu süpürmenin tile-tile'a denkliği** | İki cihaz, iki dtype, iki ölçek politikası, chunk 2'den tile sayısının 4 katına |
| **Akıtılan alt-Hessian'ın yığılmışa denkliği** | Bit-birebir aynı çıktı |
| **Maliyet modelinin kendini doğrulaması** | Hiç uydurulmadığı bir genişlikte (k=7912) gerçek tile süresini 16 satırda %11.9 içinde tahmin ediyor. 4 satırda %39.8 sapıyor — **fazla** yazarak, yani 12 gün bir üst sınır (§6.3) |
| **§8.1 dikişinin GPU'da uçtan uca koştuğu** | Gerçek Llama blokları → `sequential_calibrate` → `run_config` → sıkıştırılmış ağırlıklar → `streamed_perplexity`, hepsi cuda'da. 08-25'e kadar **koşmuyordu** ve bunu hiçbir test görmüyordu (§6.12) |

### 3.2 Varsayım — doğrulanmadı

> **E8P'nin kompaktlanmış survivor alt-matrisinde 2 bit KALİTESİNİ koruduğu.**
> Survivor'lar tanım gereği dağılımın kalın kuyruğu; kafes quantizer Gauss'a
> yakın girdi ister.

Maliyet tarafı 08-21'de kapandı (tam 2 bit ödendiği kesin). Açık olan
**karşılığında 2 bitlik kalite alınıp alınmadığı.** Bu varsayımı sınayacak ucuz
deney bilinçli olarak atlandı (kullanıcı kararı, 08-20).

Rotasyonun gerçek katmanda −70% çıkması varsayımı **dolaylı olarak
güçlendiriyor** — gerekçesi "rotasyon kalın kuyruğu düzeltir"di ve rotasyonun
çalıştığı artık ölçüldü. Ama doğrudan kanıt değil.

**Erken uyarı kuralı:** ilk katman E8P'den geçtiğinde katman-çıkışı MSE'si dense
E8P referansının 2 katını aşarsa varsayım düşmüş sayılır; geri dönüş yolu
rotasyon + GPTQ-3bit (`W=3.148`), bant 1.83–2.83'e kayar.

### 3.3 Henüz hiç ölçülmemiş

**Sıkıştırılmış modelin perplexity'si.** Kapı A ve Kapı B'nin **hiçbir gerçek
verisi yok**. Sentetik smoke testte hata eğrisi U şeklinde çıkıyor ve Kapı A
geçiyor — **ama veriyi biz ürettik, bu tez lehine kanıt değil.**

Sebebi artık maliyet değil (bir U eğrisi **13.7 saat**): **tam modeli sıkıştıran
betik yok.** `calibrate.sequential_calibrate` kütüphane olarak var ama yalnızca
testlerden çağrılıyor. Aynı şey `τ` süpürmesi için de geçerli — maliyeti
modellenmiş, kodu yazılmamış (§8.3).

Ayrıca hiç ölçülmemiş: **eval'in gerçek maliyeti** (238 s yalnız WikiText-2;
ön-kayıt §4 C4'ü de şart koşuyor ve 5 zero-shot görev istiyor) ve
**`fit_scale`'in doğru hedefe uydurulması**. *(TF32 bu listedeydi; 08-24'te
ölçüldü ve reddedildi — §6.9.)*

---

## 4. Alınan kararlar ve gerekçeleri

| Karar | Tarih | Gerekçe |
|---|---|---|
| Survivor quantizer **GPTQ-4bit → QuIP# E8P** | 08-20 | Kapı A provası: GPTQ-4bit survivor'larla literatürün konuşabildiği her yerde kaybediliyor. `W` 4.156 → 2.000, `B=1.5`'te `T=16` seyrekliği %65 → %28 |
| Ucuz E8P doğrulama deneyi **atlandı** | 08-20 | Kullanıcı kararı; risk §3.2'de açık varsayım olarak taşınıyor |
| Bant **1.75 / 1.60 / 1.50** | 08-20 | E8P'nin canlı bandı 1.40–1.80; çalışma kendiliğinden 2 bitin altına kaydı — motivasyonun tuttuğu yere |
| Çapa **QTIP/QuIP#**, GPTQ değil | 08-20 | GPTQ 3-bit sınıfın en zayıfı; ona çapalanırsa Kapı A kolay geçer ama savunulamaz |
| **LDLQ eklendi** | 08-20 | Rotasyon, Hessian-farkında yuvarlama olmadan maliyeti ödeyip faydasını toplamıyordu |
| Kapı B için **≥5 çekiliş** | 08-20 | 3 seed ile `gate_b` saf gürültüde "interior" verdi. Spec §6'nın "seed ≥ 3"ü bu kapı için yetersiz |
| Checkpoint: **NousResearch aynası** | 08-21 | Resmi repo kapılı; dense ppl ölçümü ağırlıkların doğruluğunu zaten teyit etti |
| **seqlen 4096 birincil** | 08-21 | `dense-5.12` ailesi hem budama baseline'larını hem QTIP/QuIP#'i taşıyor; Kapı A'nın rakibi orada |
| Izgara **`vq_bits = 2.0`'da donduruldu** | 08-21 | Düzeltme her hücrede aynı göreli miktarda (%0.26) — bütçe-eşleşmesini bozmuyor. Tam 2 ise `B=1.5` ızgarasını tam dyadic yapıyor ve `golden.py`'nin bağımsız türetmesi buna dayanıyor |
| Tolerans kuralı **`1.5 × max_T \|sapma(T)\|`** | 08-21 | Seed varyansından türetilseydi **12.3 kat** küçük olurdu ve ön-kayıt tanım gereği "tutmadı" dalına kilitlenirdi |
| `T*` **nokta değil küme** olarak raporlanır | 08-21 | Verdikt ile `T*` aynı güvenilirlikte değil: düz iç bölgede 20 çekilişle verdikt %77, argmin %41 |
| Çekiliş ekseni **kalibrasyon**, rotasyon seed'i değil | 08-21 | Ölçüldü: kalibrasyon gürültüsü rotasyon gürültüsünün **1.95 katı** |
| Hat **cuda/float32**'ye taşındı | 08-23 | Uçtan uca **16–45×**, ağırlık farkı 5e-08 — float32'nin kendi epsilon'u düzeyinde |
| **Katman-başı ölçek reddedildi** | 08-23 | Ölçüldü: %11 kalite kaybı, hız kazancı yok. Yeniden ölçümde T=4'te **+87.9%** |
| **E8 kafes çözücü** kaba kuvvetin yerine | 08-23 | Baskın terim buydu. CPU 3.51×, GPU 1.87×, çıktı birebir aynı |
| **`hessian_block=512`**, rotasyon tam genişlikte | 08-23 | Geri beslemeyi daraltmak kaliteyi **iyileştiriyor** (−11/−23/−16%), rotasyonu daraltmak bozuyor (+43/+38/+44%). Sonradan çıktı ki toplu süpürmenin de önkoşulu |
| **Süpürme tile'lar arasında toplu** (`chunk="auto"`) | 08-23 | Süpürme %99.6 boşta duruyordu. Süpürmede 5–12×, çıktı bit-birebir aynı |
| **Ölçek örneklemesi reddedildi** | 08-23 | Ortalama bedeli küçük ama tohumdan tohuma **15.8 puan** oynuyor — Kapı B'nin ayırmaya çalıştığı 0.31 σ'yı boğar (§5.8) |
| fp16 arama **eklendi, varsayılan kapalı** | 08-23 | 1.3–1.7×, bedel ≤%1 ve **belirlenimci**. Kaliteyi ölçülebilir biçimde değiştirdiği için varsayılan olması bir karar gerektirir |
| **Kronecker kongrüansı eklendi, varsayılan kapalı** | 08-24 | Gerçek katmanda `H512` kolunda −0.03…−0.31% (lehte), rotasyon terimi **5.52×**. Bit-birebir olmadığı için açmak ayrı bir karar (§6.8, §8.5) |
| ~~**fp16 arama ve telafi bloklaması da kapalı kaldı**~~ | 08-24 | Kullanıcı kararı: şimdiye kadarki her kalite sayısı üçü de kapalıyken alındı. **08-25'te geçersiz kılındı** — silinmedi, çünkü M0'ın sayıları hâlâ o rejimde |
| **fp16 arama geri KAPANDI** | 08-25 | Sekiz saat açık kaldıktan sonra gerçek blok şekillerinde yeniden ölçüldü: **1.00×**. Onu haklı çıkaran 1.09–1.22× tek katmanın 512 satırıyla alınmıştı. CPU'da 4.3× yavaş, kalite ≤%0.90. Açıkça istenirse çalışıyor (§6.17) |
| **Kalan iki kaldıraç DENETLENDİ ve kaldı** | 08-25 | `rotate_kron` ve `compensate_block` gerçek blokta iki kez ölçüldü (izole + yerinde): altı karşılaştırmanın altısında yerinde tasarruf izolenin %95–107'si. Blokta 2.35×. fp16'yı düşüren denetim bu ikisini **doğruladı** (§6.18) |
| **Maliyet modeli hattın koştuğu aritmetiği fiyatlıyor** | 08-25 | İki kusur, ikisi de kötümser: `rotation_seconds`'ın Kronecker yolu **yoktu** ve `compensate_block` varsayılanı `None`'dı. Modelin 5.2× iyimserliği bunlarla birlikte yok oldu — gerçek katmanda **1.03×**, ve boşluğun tamamı sürücünün bağlamı (§6.18) |
| **Üç kaldıraç AÇILDI** | 08-25 | Kullanıcı kararı. `run_config`'in varsayılanı artık kron + fp16 + telafi bloklaması; O günkü tahmin M1 15.0 → 7.5 g'dü; **tahmin yanlıştı** — elle uygulanan
5.52× ve fp16'nın 1.38×'i fazla kredi verdi. Ölçülüp modele yazılınca **11.7 g**
(§6.18). Kıyaslanabilirlik bedeli **not düşülerek değil ölçülerek** kapatılıyor: sarsma koşusu aynı konfigürasyonu bir kez iki kolda da koşuyor (§8.5) |
| **TF32 kapandı — kalite yüzdesiyle değil** | 08-24 | Hattı kırıyor: döndürülmüş alt-Hessian Cholesky'den geçmiyor, sönümleme payının %85'i gidiyor. Çalıştığı yerde de %3.2'yi aşan tek kol (§6.9) |
| **Analitik en-yakın-kodsözcüğü** taramanın yerine | 08-23 | Kodsözcüğü uzayının yapısı aramayı çözüyor. Uçtan uca 1.3–4.0×, float64'te kesin (§6.4) |
| **Triton kuruldu, iki zincir füzyonlandı** | 08-24 | GPU %28.4 meşguldü; boşta geçenin %80'i fırlatma. Uçtan uca 1.64–1.87×, çıktı birebir aynı (§6.5) |
| Maliyet modelinin varsayılanı **hattın koştuğu konfigürasyon** | 08-24 | Kimsenin koşmadığı bir konfigürasyonu fiyatlamak, iyimser bir sabitle aynı hata — yalnız ters yöne bakıyor |
| **Ölçek adayları tek aramada toplandı** | 08-24 | Arama fırlatma bağımlı: 1,280 vektör 41.3 ms, 5,888 vektör 43.4 ms. 24 ayrı geçiş sabit bedeli 24 kez ödüyordu. Tile başına 3.78×/2.01×/1.09×, çıktı **bit-birebir** (§6.7) |

---

## 5. Bilimsel bulgular — planı değiştirenler

Önem sırasına göre, kronolojik değil.

### 5.1 ⭐ Rotasyon gerçek katmanda sandığımızdan çok daha değerli

`layers.0.self_attn.o_proj` (512 çıktı satırı), gerçek Llama-2-7B ağırlıkları,
32,768 gerçek kalibrasyon token'ı, `B=1.5`, cuda/float32:

| T | d | düz | rotasyonlu | **değişim** | sentetik hattın dediği |
|---|---|---|---|---|---|
| 4 | 0.6250 | 0.47422 | 0.09649 | **−79.7%** | −29.5% |
| 16 | 0.7188 | 0.54423 | 0.19530 | **−64.1%** | −31.0% |
| max | 0.7500 | 0.55738 | 0.18655 | **−66.5%** | — |

Ortalama gerçek **−70.1%**, sentetik −30.2%.

> **Bir çerçeveyi çürüttü.** "Sentetik bir kazanç için yapısal bir bedel
> ödüyoruz" diyordum. Kazanç sentetik değil — **sentetik olan, kazancı iki-üç
> kat eksik ölçmüş.**

**Sonuç: "rotasyonu bırak" seçeneği kapandı.** **Kapsam:** tek katman, tek
çekiliş, katman-çıkışı hatası — perplexity değil.

### 5.2 §0.5 tersine döndü

v6 incoherence processing'i en büyük risk sayıp QuIP#/QTIP'i toptan eliyordu.
Eleme fazla genişti: maske dondurulduktan sonra rotasyon onu bozamaz. Belgelenen
çöküş bir **sıra** problemi. Bu, E8P'ye geçişin kapısını açan adımdı.

### 5.3 Rotasyonun değeri LDLQ'dan değil, dağılımdan geliyor

İzole ölçüm (16×64 blok, korelasyonlu Hessian):

| blok | rotasyon, düz | rotasyon, LDLQ |
|---|---|---|
| Gaussian | +17.5% (zarar) | +4.8% (zarar) |
| kalın kuyruklu | **−61.7%** | **−39.0%** |

LDLQ yine de zorunlu: onsuz rotasyon maliyeti ödeyip faydasını toplamıyor
(hat ölçümü: T=4'te +2.6% → −29.5%).

### 5.4 SU ve SV aynı şey değil

QuIP#'in yan bilgisini ölçerken çıktı: `SU` (girdi ekseni) ince ayarın ±1'den
zar zor kıpırdattığı bir işaret vektörü; `SV` (çıktı ekseni) gerçek kanal-başı
ölçek. Önemi: tile başına **öğrenilmiş** bir sütun vektörü `16/T` bit demekti
(T=16'da 1.0 — bant kaldırmaz). Ölçüm bunu ödemek zorunda olmadığımızı
gösterdi: ayrıştırılmış tasarımda **0.0077 bit/survivor**, `T` ile neredeyse
sabit (`accounting.rotation_side_bits`).

### 5.5 Ayrılabilirlik varsayımı büyük `T`'ye karşı önyargılı

Transfer pilotu `τ`'nun quantization'sız ölçüldüğünde sistematik olarak
**büyük** çıktığını gösterdi — `T=2` dışında her yerde, fark `T` ile büyüyor.
Mekanizma: 2 bitte quantization hatası maske kalitesi farkının bir kısmını
zaten örtüyor. Sonuç: **model tile'ların maliyetini olduğundan pahalı
gösteriyor.** M1'de büyük `T` tahminden iyi çıkarsa bu beklenen bir şeydir,
tez lehine kanıt değil — ön-kayıt §5.1'e yazıldı.

İyi haber: sapma işaret değiştirdiği halde `T*` kaymadı (tahmin de ölçüm de
`T*=4`). Garanti değildi; her koşuda `argmin_agreement` olarak raporlanıyor.

### 5.6 Kapı B'nin verdikti güvenli, `T*` değil

Verdikt 5 çekilişle rahat kararlanıyor — bağlayıcı uç `T=max`, eşiğin üç katı
uzakta. Ama **komşu tile'lar ayrılmıyor**: `T=4` ↔ `T=8` arası 0.31 σ, %90
güvenilir bir argmin için ~53 çekiliş gerekir. Sonuç `m1_gates.t_star_set`:
argmin değil, argmin'den ayrılamayanların kümesi. Dürüst manşet "T=16 optimal"
değil, "optimum içeride, yeri 2–16 arasında".

### 5.7 Daraltılabilen şey geri besleme, rotasyon değil

Eski §6.3 "rotasyonu 8'lik gruplara blok-köşegen kısıtla" diyordu. İki yönden
yanlış çıktı.

**Öncül.** `rotate` zaten `share_across_tiles=True` ile **tek bir rotasyonu
bütün katmana** uyguluyor. Yani rotasyon, LDLQ'nun tile başına faktorize
etmesinin sebebi değil — sebep tile başına farklı sütun kümesi.

**Genişlik.** Maliyet eğrisi 512'de düzleşiyor; 512'den 8'e inmek toplam
tasarrufun %1.9'unu ekliyor ama atılan Hessian bağlantısını 64 katına çıkarıyor.

Ölçüm (`o_proj`, 512 satır, B=1.5), `full`'e karşı, T=4 / T=16 / T=max:

| genişlik | R (rotasyon daraltıldı) | H (geri besleme daraltıldı) |
|---|---|---|
| 2048 | +12.8 / +12.8 / +22.3% | −8.6 / −9.2 / −2.0% |
| 1024 | +25.3 / +13.4 / +23.5% | −7.1 / −16.7 / −6.8% |
| **512** | +43.0 / +38.1 / +44.3% | **−11.1 / −23.4 / −15.8%** |
| 128 | +117 / +69 / +95% | +13.4 / −20.6 / −11.7% |
| 8 | +375 / +169 / +192% | +147 / +49 / +57% |

**`H512` her tile boyutunda bütün ızgaranın en iyi kolu.** `R8` neredeyse
rotasyonsuza eşit (−3.3%): rotasyonu daraltmak mekanizmayı yok ediyor.

Mekanizma iki tarafta da tutarlı. Rotasyonun işi §5.3'te kurulduğu gibi kalın
kuyruğu **olabildiğince geniş** yaymak; 8 koordinat içinde döndürmek o
koordinatların normunu değiştiremez, yalnızca yönünü. Geri besleme tarafında ise
2560×2560'lık alt-Hessian 32,768 token'dan kestiriliyor; uzun menzilli
bağlantılar gürültülü, atmak düzenlileştirme gibi davranıyor.

### 5.8 Ölçek uydurmayı örneklemek ucuz değil — gürültülü

Tavanı M1'i 17 → 8.6 güne indiriyordu. Ölçüldü (`m0_scale_fit.py`, 54 kol);
**alınamaz.** Sebep ortalama bedel değil, **varyans**. Aynı ayar, yalnız hangi
vektörlerin örneklendiği farklı:

| T | tohum 0 | tohum 1 | tohum 2 | aralık |
|---|---|---|---|---|
| 4 | **+17.08%** | +1.49% | +1.25% | **15.8 pp** |
| 16 | −3.70% | −3.14% | −3.38% | 0.6 pp |
| max | −9.97% | +4.58% | −7.48% | **14.6 pp** |

Kapı B'nin ayırmaya çalıştığı komşu tile farkı **0.31 σ**, saptanabilir fark
hata seviyesinin **%3.2'si**. 15 puanlık, tile'dan tile'a bağımsız bir gürültü
kaynağı tam olarak ölçmeye çalıştığımız şeyi boğar.

**Adım sayısını düşürmek de çalışmıyor:** `n6` +45.6 / +17.8 / +13.6%,
`n12` +8.6 / −7.5 / −7.4% — işareti bile tutarsız.

> **Ayrıca kaydetmeye değer bir ders.** Daha önce "6 adım α'yı %1.4 içinde
> buluyor" demiştim; doğruydu ve yanıltıcıydı. **α'daki %1.4, çıktı hatasında
> %45.** Vekil bir ölçü, ölçtüğünü sandığın şey değildir.

**Yan bulgu — `fit_scale` yanlış hedefi optimize ediyor.** T=16 ve T=max'te
örnekleme sistematik olarak **iyileştiriyor** (s256/n12 T=max'te −20.8%). Sebep:
`fit_scale` `‖x − αQ(x/α)‖²`'yi **ağırlık uzayında** minimize ediyor, oysa
hattın hedefi `tr(E H Eᵀ)`. Daha kesin bir α, yanlış ölçüye göre daha kesin.
**Ölçülmedi, ve artık en büyük açık fikir bu.**

### 5.9 VENOM'un `V`'si bizim `T`'miz

V:N:M formülü VENOM'dan dolduruldu ve yapısal bir şey çıktı: VENOM `V` satırın
paylaştığı bir sütun seçimi kullanıyor, yani indeksi `1/V` ile amortize ediyor.
**İndeks amortizasyonu yeni değil.** Katkı "amortize etmek" değil, `(T, d)`
düzlemini bir bütçe altında taramak. Özgünlük iddiası buna göre daraltıldı.

### 5.10 Protokol ayrımı dizi uzunluğuymuş

Ölçümden önce hipotez olarak kaydedildi, ölçümde tuttu. Kural "birini seç"
değil **"pencereyi sabitle"**.

---

## 6. Maliyet: 120 gün → 15 gün — ve bir kez yukarı gitti

Bu bölüm mühendislik, §5 bilim. Ayrı tutuluyor çünkü buradaki hiçbir şey tezi
değiştirmiyor — yalnızca sınanabilir hâle getiriyor.

### 6.1 Bugünkü tablo (B=1.5, cuda/float32, Triton açık)

**Kaldıraçlar açık, ve 08-25'te ilk kez modelin fiyatladığı aritmetik hattın
koştuğu aritmetik** (§6.18 — o güne kadar rotasyon yoğun, telafi bloklamasız
fiyatlanıyordu):

| T | d | **nokta** | codebook | rotasyon | telafi | kalib | chol | eval |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2500 | **4.01 h** | **2.89** | 0.34 | 0.05 | 0.33 | 0.34 | 0.07 |
| 2 | 0.5000 | 3.34 h | 1.65 | 0.89 | 0.05 | 0.33 | 0.34 | 0.07 |
| 4 | 0.6250 | 2.07 h | 0.91 | 0.50 | 0.05 | 0.33 | 0.21 | 0.07 |
| 8 | 0.6875 | 1.54 h | 0.68 | 0.30 | 0.05 | 0.33 | 0.11 | 0.07 |
| 16 | 0.7188 | 1.14 h | 0.46 | 0.17 | 0.05 | 0.33 | 0.06 | 0.07 |
| 32 | 0.7344 | 0.92 h | 0.37 | 0.07 | 0.05 | 0.33 | 0.03 | 0.07 |
| max | 0.7500 | 0.66 h | 0.21 | 0.00 | 0.05 | 0.33 | 0.00 | 0.07 |

Kalibrasyon ve eval **tile boyutundan bağımsız** — aynı 32 bloğu aynı şekilde
dolaşıyorlar; telafi de öyle, ama artık 0.05 h olduğu için tasarımı sürükleyen
kalem değil. Bu, tasarım ekonomisini yeniden sıralıyor: sıkıştırma baskınken
maliyeti *hangi* tile'ları koştuğun belirliyordu, düz nokta-başı terimler
varken **kaç nokta** koştuğun belirliyor (§6.10).

Toplam paylar (7 tile): **codebook %52.4**, kalibrasyon %16.9, rotasyon %16.5,
cholesky %8.0, eval %3.4, telafi %2.8.

> **Ve baskın terim üçüncü kez el değiştirdi.** Codebook → rotasyon (08-24) →
> codebook (08-25). Sonuncusu kodun hızlanmasıyla değil, **modelin nihayet
> kodun yaptığını fiyatlamasıyla** oldu: rotasyon yalnız yoğun GEMM olarak
> yazıldığı sürece öndeydi, oysa hat 08-25 sabahından beri Kronecker
> çarpanlarına kasıyordu. T=4'te terim 1.92h'ten 0.50h'e iniyor. Bir terim,
> model kimsenin koşmadığı aritmetiği tarif ettiği için bir gün boyunca
> ızgaraya liderlik edebiliyor (§6.18).

Eğri artık T=1'den itibaren **kesintisiz azalıyor** (4.01 → 0.66). Önceki tablo
T=1 ve T=2'yi neredeyse eşit gösteriyordu (4.45'e karşı 4.48); o eşitlik yoğun
rotasyonun T=2'de yaptığı tepeydi ve o tepe gerçek değildi.

> **Tepe 08-25'te yer değiştirdi ve bu tablodaki en önemli değişiklik.** Bu
> bölüm "maliyet ızgaranın **ortasında** tepe yapıyor" diyordu; yapmıyor.
> `TILE_TIMINGS` 4 satırın altında hiç örnek taşımıyordu, yani T=1 ve T=2
> 4-satır oranını **ödünç alıyordu** — ve ağırlık başına maliyet 1 satırdan
> 4096'ya **41 kat**, ilk adımda tek başına 8 kat düşüyor. Gerçek eğri T=1'den
> itibaren **monoton azalıyor**, ve pahalı uç ince uç (§6.14).

**M1 (3 bütçe × 7 tile × 5 çekiliş): 11.7 gün** — ve bu **yukarı yanlı**:
B=1.60 ile B=1.75'in genişlikleri kron için ölçülmediği için orada yoğun
rotasyonla fiyatlanıyorlar ve `m1_cost` `rotate_kron_priced=False` diyor.
**`τ` süpürmesi: 4.2 gün** (spec 25 *saat* diyordu).
**Tasarım F (ilk gerçek U eğrisi): 13.7 saat.**

> **Duvar artık her yerde codebook, ve T=1'de ezici.** 2.89h, diğer bütün
> terimlerin toplamının 2.6 katı. Orada duvar `fit_scale`'in tile başına sabit
> bedeli: tek satırlık bir tile ona amortize edecek 128 vektör veriyor, T=4
> 1280. Ve T=1 tezin kıyas grubu olan yapısız taban — yani bu, ızgaranın
> kenarında bir köşe değil.

**Ölçek kaldıracı çökmüştü, kısmen geri geldi.** Per-tile fit'i tamamen atmak
bir zamanlar 8.8 gün, sonra **1.4 gün** yazılmıştı; 08-25'te fitin payı satır
eksenine yayılınca **2.76 gün** çıktı (T=1'de nokta başına 1.66×, §6.15). Yine de
`per_layer` ve örnekleme geri gelmiyor — ikisi de **kalite** gerekçesiyle
reddedildi (§5.8, §7.1), ve örnekleme ince uçta zaten no-op (tile 128 vektör
tutuyor, sınır 8,192). Erişilebilir tek biçim fiti **tile'lar arasında**
toplamak, ve o bit-birebir değil (§7.2).

### 6.2 Kapatılan duvarlar, sırayla

| # | Duvar | Neydi | Nasıl kapandı |
|---|---|---|---|
| 1 | **Bellek** | `T=2`'de 462 GiB tek tensör | `tile_hessian_stream` → 239 MiB. Yığılmış yolla bit-birebir aynı |
| 2 | **Yanlış cihaz** | Hat CPU/float64'te koşuyordu | cuda/float32, uçtan uca **16–45×**, ağırlık farkı 5e-08 |
| 3 | **Kaba kuvvet arama** | Baskın terim | `nearest_e8p` kafesi çözüyor: CPU 3.51×, GPU 1.87× |
| 4 | **Modelin kendi hatası** | Cholesky'yi 9.4× fazla yazıyordu | §6.3. 120 → **94 gün** |
| 5 | **Süpürme %99.6 boşta** | Grup başına 0.248 ms, hesap 0.0034 ms | Tile'lar arası toplu. 94 → **48 gün** |
| 6 | **Tarama hiç gerekmiyordu** | Kesinlik küçük α'da %0.7'ye çöküyor | Analitik arama. 48 → **29 gün** |
| 7 | **GPU %28 meşgul** | Boştanın %80'i çekirdek fırlatma | Triton füzyonu. 29 → **17 gün** |
| 8 | **Fit sabit bedeli 24 kez ödüyordu** | 1,280 vektör 41.3 ms, 5,888 vektör 43.4 ms | Adaylar tek aramada. 17 → **12 gün** (§6.7), iki iyimser ölçüm geri çekilerek |

**Ve burada sayı yukarı gitti.** Dokuzuncu duvar bir hızlanma değil, modelin
ölçmediği bir şeydi — ve bulunduğunda M1'in gerçekte 12 değil ~40 gün olduğu
ortaya çıktı:

| # | Duvar | Neydi | Nasıl kapandı |
|---|---|---|---|
| 9 | **Kalibrasyon hiç yazılmamış** | Nokta başına 5.6 saat, sıkıştırmanın tamamından pahalı | Modele yazıldı: 12 → **~40 gün** (§6.10). Sonra Hessian GPU'da biriktirildi (25×): → **13.4 gün** |
| 10 | **İleri telafi de yazılmamış** | Nokta başına 0.36 saat, tile boyutundan bağımsız | Modele yazıldı: → **15.0 gün** (§6.11b). Bloklanabilir ama bit-birebir değil, varsayılan kapalı |
| 11 | **`_nearest`'in ikinci kapısı** | 21 hücrenin 10'u 65,536 kodsözcüğü tarıyordu | 384–1024 aralığı analitik yola açıldı. Süpürmede **2.0–3.5×** (§6.11a) |
| 12 | **Üç sabitin yazılmamış eşitsizliği** | Süpürmenin her grubu taranıyordu; 21 hücrenin 8'i | İki sabit ölçülerek oynadı. Tile'da **1.25×**, kalite bit-birebir (§6.13) |
| 13 | **`TILE_TIMINGS`'te 4 satırın altı yok** | T=1 ve T=2 4-satır oranını ödünç alıyordu | `n_tiles` kaydedilerek yeniden ölçüldü: toplam **15.0'da kaldı** ama **tepe ortadan ince uca kaydı** (§6.14) |

> **9, 10 ve 13 hızlanma değil, düzeltme.** İlk ikisi M1'i *pahalılaştırdı* çünkü
> zaten öyleydi; 13 toplamı neredeyse hiç oynatmadı ama **eğrinin şeklini**
> değiştirdi — tepe ortadan T=1'e kaydı, ve T=1 tezin kıyas grubu. Bu tablodaki
> en değerli satırlar onlar: 1–8 ve 11–12 kodu hızlandırdı, 9, 10 ve 13
> **sayıyı doğru yaptı**.
>
> Ve üçü de aranarak değil, **başka bir şey düzeltilirken** çıktı: ikisi eksik
> terim arayışında, biri provenance düzeltilirken.

### 6.3 Maliyet modelinin on hatası

İlk üçü iyimser, dördüncüsü **kötümser** — ve o en çok zarar veren oldu, çünkü
bu sayı M1'in koşulup koşulmayacağını söyleyen sayı. Beşincisi yine iyimser.
Altıncı, yedinci, sekizinci ve dokuzuncu (§6.10, §6.11b, §6.14, §6.15)
**modelin bilmediği şeyler** — ve dokuzun yedisi bu sınıftan. Yani bu modelde asıl soru "oran doğru mu" değil, **"listede
ne yok"**.

1. **`fit_scale` hiç yoktu** — 6× az. `ldlq_quantize` quantize etmeden önce 24
   aday ölçeği tile'ın tamamı üzerinde tarıyor.
2. **Arama, 16 satırlık sıkı bir döngüde ölçülmüş hızla fiyatlandı** — orada
   codebook önbellekte kalıyor, gerçek çağrılar arasında Hessian güncellemeleri
   onu atıyor. Uçtan uca **tile** süreleri ölçülerek düzeldi.
3. **Her tile boyutuna tek bir ağırlık-başı sabit** — sabit satır sayısıyla üç
   kat düşüyor, yani ızgaranın kaba ucu fazla yazılıyordu. Ve kaba uç, tam da
   granülerlik sorusunun ilgilendiği uç.
4. **Cholesky hızı k=2048'de, ısınmamış benchmark'la ölçülmüş** —
   `cholesky_inverse` ısıtılmıyordu (1.6×), ve tek bir flop/s bu çekirdeği
   tanımlamıyor (k=1024→8192 arası **6.8×** değişiyor, 2.6× daha). Gerçek
   genişliklerde **9.4× fazla**.

5. **Tile süresinden yanlış genişlikte Cholesky çıkarılıyordu** — `TILE_TIMINGS`
   `hessian_block=512` ile ölçülmüş, ama `codebook_seconds_per_vector` **tam
   genişlikte** bir Cholesky çıkarıyordu, yani tile'ın hiç harcamadığı zamanı.
   Codebook %34 / %24 / %9 eksik yazılıyordu — ve en çok ince granülerlikte,
   ızgaranın maliyetinin yaşadığı yerde. `TILE_TIMING_BLOCK` her satırın hangi
   genişlikte ölçüldüğünü kaydediyor; cpu ve cuda satırları farklı düzenlerde
   alındığı için tek bir varsayım ikisi için birden doğru olamıyor.

**Ayrıca: iki ölçüm geri çekildi.** 08-24'te kaydedilen üç `cuda_f32` tile
süresinden ikisi tekrar üretilmiyor, ikisi de iyimser yönde: (2944,16) için
0.0631 yazılmış, aynı konfigürasyon bugün 0.0810 ölçüyor (**1.28×**);
(3072,128) için 0.1851 yazılmış, bugün 0.3058 (**1.65×**). Bu bir kurulum farkı
**değil**: (2560,4) satırı 1.00× üretiliyor, ve superseded *eager* satırı
%0.2 içinde üretiliyor (0.3883'e karşı `TILESPARSE_NO_COMPILE=1` ile ölçülen
0.3874). Tutmayan şey, o iki geniş satır için iddia edilen **Triton kazancı**:
1.72× ve 1.87× yazılmış, bugün 1.18× ve 1.09× ölçülüyor.

> **Mekanizma taşınmaya değer: bu iki kaldıraç çarpılmıyor.** Triton'un
> kazandırdığı şey fırlatma yüküydü; adayları toplamak *aynı* yükü bir kat
> yukarıdan siliyor. Tek bir israfa iki çare onu paylaşır, katlamaz. Modele
> ikisinin çarpımı asla verilmemeli.

Ders: **kernel mikro-benchmark'larından maliyet kurmak burada işlemiyor.** Her
eğri, kodun onu çağıracağı boyutlarda ölçülüyor, ve artık ileri telafi de
modelde (ölçülen %8.6). Modelin kendi dışında sınanması da sürüyor: k=7912'de
16 satırda %11.9, 4 satırda %39.8 sapıyor — ikisi de **fazla** yazarak, yani
12 gün bir üst sınır.

### 6.4 Aramayı taramaktan çıkarmak

**65536 kodsözcüğünü taramaya hiç gerek yokmuş.**

Bir kodsözcüğü `σ⊙p + s`: `p` 256 **negatif olmayan** kaynak örüntüsünden biri,
`σ` ilk yedi koordinatta serbest, sekizinci koordinat toplamı çift yapacak
şekilde belirli. `p` negatif olmadığı için sabit `p` altında en iyi işaretler
koordinat koordinat okunuyor (`σ_i = sign(z_i)`); bu atama tek parite ise
geçersiz, ve her koordinat yarım-tamsayı olduğundan **herhangi bir tek işaret
çevirisi pariteyi değiştiriyor** — yani onarım tek ve en ucuz çeviri, bedeli
`2|z_i|p_i`. 128 işaret seçimi bir arama uzayı değil, aritmetik.

Bu, taramanın yerine değil **geri düşme yolunun** yerine kondu. Kafes çözücü bir
satırı çözebildiğinde hâlâ daha ucuz (8K ve 80K satırda aynı 0.2 ms — fırlatma
bağımlı). Düzelttiği şey `fit_scale`'in küçük-α adımları:

| f | kesinlik | geri düşen | adım |
|---|---|---|---|
| 0.40 | **%0.7** | 5,845 / 5,888 | 30.0 ms |
| 1.03 | %63.7 | 2,136 | 12.5 ms |
| 2.00 | %99.9 | 8 | 2.0 ms |

**Fit'in %88'i oradaydı.** Kazanç: `fit_scale` 3.25× (5,888 vektör) → **10.8×**
(196,608 vektör); tile başına 1.35 / 2.65 / 5.62×; gerçek katmanda 1.29 / 2.17 /
3.96×.

**Kesinlik.** float64'te bir milyon vektörde **sıfır** uyuşmazlık; her
kodsözcüğü kendine çözülüyor. float32'de milyonda bir satır farklı seçiliyor ve
o satırlar gerçek berabere — mesafe farkı 3e-6. İddia "kesin", "float32'de
bit-birebir" değil, ve test hangisi olduğunu söylüyor.

### 6.5 Triton: Windows'ta var

GPU **%28.4** meşguldü. Bir chunk'ta 414,841 çekirdek çağrısı, ölçülen fırlatma
maliyeti **10.1 µs**, yani boşta geçen 5,258 ms'nin **4,190 ms'si** doğrudan
fırlatma.

Upstream `triton` Windows tekerleği yayımlamıyor — `has_triton()` bu yüzden
False'tu ve ilk deneme 15 dakika asıldı. Ama **`triton-windows` PyPI'da** ve
sürümler tutuyor: torch 2.12 → triton 3.7.0 → `triton-windows==3.7.0.post26`.

İki elementwise zincir ayrı saf fonksiyonlara çıkarıldı ve `dynamic=True` ile
derlendi (`_analytic_shift`, `_lattice_shift`):

| ölçek | kazanç |
|---|---|
| `_nearest_halfinteger_even` | 2.30× |
| analitik aramanın gövdesi | 5.96–6.62× |
| **tile başına uçtan uca** | ~~1.64× / 1.72× / 1.87×~~ |

> **Bu satırın son sütunu geri çekildi (§6.3).** İnce satır tutuyor, iki geniş
> satır tutmuyor: bugün 1.18× ve 1.09×. Ve §6.7 geldikten sonra Triton'un
> marjinal katkısı zaten küçüldü — ikisi aynı israfı paylaşıyor.

**Tahminim 3.5–5× idi, gerçekleşen 1.7×.** Fazla iyimserdim: hesabım bütün
fırlatma yükünün gideceğini varsayıyordu, oysa yalnız iki blok derlendi ve LDLQ
süpürmesinin küçük işlemleri hâlâ eager.

İki ayrıntı taşıyıcı:
- **`dynamic=True` şart.** Satır sayısı, kafes çözücünün çözemediği satır
  sayısı — her çağrıda değişiyor. Statik derlense her yeni şekilde birkaç saniye
  harcardı; dinamik, bir kez derleyip 64 kat aralıkta **sıfır yeniden derleme**.
- **Derleme bir sondayla zorlanıyor.** Inductor tembel; bırakılsa hata katmanın
  ortasında patlardı. Burada CUDA Triton'la derleniyor ama **CPU `cl` (MSVC)
  istiyor ve bulamıyor**, o yüzden cihaz/dtype başına sondalanıyor.

**Çıktı birebir aynı, ve bu teste bağlandı.** Tören değil: çekirdek,
toolchain'i olmayan yerde eager'a düşüyor; iki yol farklı sonuç verseydi işi
hangi makinenin koştuğu modeli değiştirirdi ve başka hiçbir test bunu
yakalamazdı.

> **Uyarı:** `TILE_TIMINGS` artık **Triton'lu bir makineyi** tanımlıyor.
> Triton'suz makinede cevaplar aynı, saat ~1.7× yavaş — model o kadar iyimser
> olur. `TILESPARSE_NO_COMPILE=1` derlemeyi kapatır.

### 6.6 Izgara seçenekleri

Bağlayıcı kısıtlar: **`min_seeds=5`** (Kapı B'nin verdikti için 08-21'de
ölçülerek donduruldu) ve **`T ∈ {1,2,4,8,16,32,max}`** (ön-kayıt `{1,16,max}`
üçlüsünü *açıkça* reddediyor — "yanlış-durdurma taşırdı"; tile eksenini budamak
tezin kendi eksenini budamak). Bağlı **olmayan**: 5 çekilişin kaç bütçede
koşacağı.

| tasarım | nokta | süre |
|---|---|---|
| A. Tam M1 (3 bütçe × 7 tile × 5 çekiliş) | 105 | **11.7 g** |
| C. B=1.5'te 5 çekiliş, diğer bütçeler 1 | 49 | 4.6 g |
| D. Tek bütçe, 5 çekiliş, 7 tile | 35 | 2.9 g |
| **F. Tek bütçe, 1 çekiliş, 7 tile — ilk gerçek U eğrisi** | **7** | **13.7 saat** |
| G. Yalnız iki uç (T=1, T=max), 5 çekiliş | 10 | **23.4 saat** |

**Ve G/F kıyası 08-25'te bir kez daha döndü — üçüncü kez.** Önce G, F'in otuzda
biriydi; kalibrasyon modele yazılınca nokta sayısı baskın oldu ve ikisi eşitlendi
(19.1'e karşı 20.4); şimdi `TILE_TIMINGS` ince ucu doğru fiyatlayınca G **F'ten
pahalı** (23.4'e karşı 13.7 — 08-25'te kaldıraçlar fiyatlanınca ikisi de indi ve
sıralama **korundu**, çünkü kaldıraçlar T=1'e en az dokunan yer). Sebep tek:
G'nin iki ucundan biri **T=1**, ve T=1 ızgaranın en pahalı hücresi çıktı
(§6.14).

Ders G hakkında değil: **ucuz kaçış kapısı diye bir tasarıma bakmak, maliyetin
nerede olduğunu bildiğini varsayıyor.** Üç kez yanlış bilindi — ve F zaten daha
ucuz, üstelik **yedi tile'ın tamamını** veriyor.

fp16 sütunu kaldırıldı: üç kaldıraç da §8.5'te tek yerde toplandı.

---

### 6.7 Aday ölçekleri tek aramada toplamak

`fit_scale` 24 adayı **tek tek** geçiyordu. Arama fırlatma bağımlı olduğu için
bu, sabit bedeli 24 kez ödemek demekti. İmza net:

| vektör | süre |
|---|---|
| 1,280 | 41.3 ms |
| 5,888 | 43.4 ms |

**4.6 kat veri, 1.05 kat süre.** Profil: GPU **%21.6** meşgul, tek `fit_scale`
çağrısında **3,380** çekirdek — aday başına 141.

Adaylar bağımsız (her biri aynı vektörlerin farklı bir ölçeklemesinin neye
yuvarlandığını soruyor), yani yığmak salt bir yeniden düzenleme. Ölçülen
(`FIT_ROW_BUDGET=1`, yani eski düzeni birebir üreten kola karşı):

| tile | `fit_scale` | tile başına uçtan uca | çıktı |
|---|---|---|---|
| 4 × 2560 | 9.0× | **3.78×** | bit-birebir |
| 16 × 2944 | 10.2× | **2.01×** | bit-birebir |
| 128 × 3072 | 2.0× | 1.09× | bit-birebir |

Kazanç ince granülerlikte en büyük — ızgaranın pahalı ucunda.

**§7.2'deki reddedilen kalemle karıştırılmamalı.** O, fit'i *tile'lar arasında*
toplamaktı ve her tile'ın hatasını birlikte indirgediği için aritmetiği
değiştiriyordu. Aday ekseninde her adayın hatası kendi `[n,8]`'i üzerinde,
eskisiyle aynı sırada toplanıyor.

**Testlerin gerçekten ısırdığı mutasyonla doğrulandı** — geçen bir test hiçbir
şey kanıtlamaz. Öldürdükleri: adayı komşunun kodsözcükleriyle eşleştirmek (8),
satırları harmanlamak (18), her adayı seed ölçeğiyle puanlamak (15), α'yı başka
adayın hatasıyla eşleştirmek (18), her geçişten bir aday düşürmek (1).
**Öldüremedikleri**, çünkü hiçbiri cevabı oynatmıyor: hatayı adaylar arasında
ortak indirgemek (40 float32 çekilişinde argmin aynı), berabereyi `<=` ile
bozmak, Python float yerine tensör ile bölmek, her adayı %0.1 oynatmak. İlk üçü
kimse "taşıyıcı" diye savunmasın diye kaydedildi.

### 6.8 Rotasyonun Kronecker yapısı — gerçek katmanda ölçüldü

`tile_hessian_stream` `q @ H @ q.T`'yi yoğun GEMM olarak yapıyordu, oysa `q`
`kron(RHT(p), O(m))` (`rotation.structured_orthogonal`). Çarpanlara kasılınca
maliyet `2k³`'ten `2k²(p+m)`'e iniyor.

**Kalite — gerçek Llama-2-7B blok 0 `o_proj`, 512 satır, 32,768 gerçek token,
B=1.5** (`m0_rotation_value.py --families K`). `full` kolu birinci turu
**7.0e-07** sapmayla yeniden üretiyor, yani koşu geçerli:

| çift | T=4 | T=16 | T=max |
|---|---|---|---|
| `full` → `fullK` (tam genişlik geri besleme) | **+1.85%** | +0.94% | +0.53% |
| **`H512` → `H512K` (hattın koştuğu kol)** | **−0.31%** | **−0.03%** | **−0.15%** |

> **Sentetik ölçüm iki mertebe yanıldı — §5.1'in aynı deseni.** Sentetik
> Hessian'larda etki %0.003 ve işareti rastgeleydi. Gerçek katmanda tam
> genişlik geri beslemeyle **%0.5–1.9**, ve **sistematik** (üç tile'da da aynı
> işaret). Yani "sentetikte önemsiz" bu projede bir kanıt değil.

**Ama hattın koştuğu kolda tersine dönüyor.** `hessian_block=512` ile fark
−0.03% … −0.31%, yani ölçülemeyecek kadar küçük ve **lehte**. Mekanizma §5.7 ile
tutarlı: geri besleme 512'lik bloklara kapatılınca faktorizasyon k×k değil
512×512, ve rotasyonun yuvarlama farkını büyüten şey o koşullanma. Uzun menzilli
bağlantıları atmak düzenlileştirdiği gibi hata yayılımını da bastırıyor.

Kapı B'nin ayırabildiği fark hata seviyesinin %3.2'si (§5.6). `H512K` bunun
**10 katı altında**; ama `fullK`'nın %1.85'i yalnız 1.7 kat altında — tam
genişlik geri besleme koşulacaksa yeniden ölçülmeli.

**Maliyet — ızgaranın gerçek genişliklerinde ölçüldü**, tile sayısıyla ağırlıklı:

| k | çarpanlar | yoğun | kron | |
|---|---|---|---|---|
| 2048 | 2048×1 | 3.51 ms | 3.54 ms | **0.99×** |
| 2560 | 512×5 | 7.64 ms | 2.87 ms | 2.66× |
| 2944 | 128×23 | 11.12 ms | 1.82 ms | 6.11× |
| 5504 | 128×43 | 78.3 ms | 6.06 ms | 12.91× |
| 7912 | 8×989 | 238.8 ms | 36.3 ms | 6.58× |
| 8256 | 64×129 | 271.6 ms | 15.7 ms | **17.31×** |

Ağırlıklı: **5.52×**. Tam ikinin kuvvetinde hiç kazandırmıyor (`m=1`, çarpanlara
ayrılacak tek sayı yok) ve k=2048 ızgaranın en kalabalık genişliği — ortalamayı
aşağı çeken şey o. Orada kazanmak için gerçek bir hızlı Hadamard gerekir.

> **08-25: bu tablo YERİNDE DURUYOR ama artık modelin okuduğu tablo değil.**
> Buradaki süreler kasmanın *aritmetiğini* ölçüyor; hattın ödediği hâlinde tile
> başına bir de `H[idx, idx]` gather'ı var ve o, ucuz kolu pahalı koldan çok
> daha fazla seyreltiyor (k=2944'te kron 1.82 → 3.02 ms). Gather dahil aynı
> ağırlıkla oran **3.53×**, ve ızgaranın on dört genişliği tek tek ölçülüp
> `m0_cost_model.ROT_TILE_TIMINGS`'e yazıldı (§6.18).

> **Aşağıdaki mutlak günler o günün tabanına ait (11.98).** Taban sonra üç kez
> değişti — kalibrasyon (§6.10), telafi (§6.11b) ve `TILE_TIMINGS`'in yeniden
> ölçümü (§6.14) ile 15.0'da kaldı.
> **Ölçülen şey oran**, ve oran duruyor: rotasyon terimi 5.52×. Güncel toplam
> için §8.5.

| | o günkü taban | +kron |
|---|---|---|
| **M1** | 11.98 g | **8.17 g** (1.47×) |
| Tasarım F | 15.56 saat | **10.61 saat** |
| `τ` süpürmesi | 3.34 g | **2.28 g** |

Tahminim 7.8 gündü, ölçülen 8.17 — bu kez %5 iyimserdim.

**Varsayılan kapalı** (`run_config(rotate_kron=False)`): bit-birebir değil, ve
şimdiye kadarki her kalite sayısı yoğun kongrüansla ölçüldü. Açmak bir karar
gerektiriyor, fp16 gibi.

### 6.9 Üç hassasiyet kaldıracı, tek tek ve kombine

`m0_precision_levers.py` — 8 kol (2³), gerçek katman, kalite ve hız ayrı
fazlarda (hız faz'ı boş GPU istiyor, kalite istemiyor).

**Kalite** (gerçek `o_proj`, `-` kolun'a göre; eksi = **daha iyi**):

| kol | T=4 | T=16 | T=max |
|---|---|---|---|
| `tf32` | **ÇALIŞMIYOR** | +2.36% | −0.78% |
| `kron` | −0.31% | −0.03% | −0.15% |
| `fp16` | −0.34% | +0.90% | −0.07% |
| `fp16+kron` | −1.54% | +1.26% | +0.11% |
| `kron+tf32` | **ÇALIŞMIYOR** | **ÇALIŞMIYOR** | +4.80% |
| `fp16+kron+tf32` | **ÇALIŞMIYOR** | **ÇALIŞMIYOR** | +4.65% |

**Hız** (boş GPU, 4 dönüşümlü geçiş, medyan):

| kol | T=4 | T=16 | T=max | medyan |
|---|---|---|---|---|
| `tf32` | ÇALIŞMIYOR | 1.06× | 1.03× | 1.04× |
| `kron` | 1.18× | 1.11× | 1.02× | 1.11× |
| `fp16` | 1.09× | 1.16× | 1.22× | 1.16× |
| **`fp16+kron`** | **1.29×** | **1.30×** | 1.20× | **1.29×** |

> **TF32 ölçülecek bir kalite bedeli değil — hattı kırıyor.** §3.3 onu
> "hiç ölçülmemiş" diye taşıyordu; ölçüldü ve T=4'te döndürülmüş alt-Hessian
> Cholesky'den geçmiyor. Kongrüansı tek başına TF32'ye almak hiçbir bloku
> çökertmiyor ama **sönümleme payının %85'ini yiyor** (kalan 0.154×); maske ve
> telafi de TF32 altında değişince bir tile payı aşıyor. Çalıştığı yerlerde de
> en kötü kalite onda: kombine hâlde **+%4.8**, Kapı B'nin ayırabildiği %3.2'yi
> **aşan tek kol**. Kapandı.

**Bileşim — ve sıfır hipotezi çarpım değil.** Zamanın `a` kesrini kaldıran bir
kaldıraç `1/(1−a)` verir, ayrık işe binen iki tanesi `1/(1−a−b)` — ki bu
`1/((1−a)(1−b))` çarpımından **büyük**. Çarpımı beklenti saymak her bağımsız
çifti sinerjik gösterir ve asıl örtüşmeyi gizler. Doğru null'a göre:

| çift | ayrık olsaydı | ölçülen | |
|---|---|---|---|
| `fp16+kron` | 1.30× | 1.29× | **%99 — bağımsız** |
| `fp16+tf32` | 1.21× | 1.25× | %103 — bağımsız |
| `kron+tf32` | 1.16× | 1.07× | **%92 — rotasyonu paylaşıyorlar** |

Yani `kron` ve `tf32` aynı terime biniyor, `fp16` ayrı terime. Öngörülmüştü ve
ölçüm tuttu.

**M1'e etkisi** (model, ölçülen terim oranlarıyla; `o_proj`'un %24'ü rotasyon,
M1 ortalaması `down_proj` ağırlıklı olduğu için daha büyük):

> Yine o günün tabanı (11.98). Oranlar geçerli, mutlak günler §8.5'te güncel.

| kol | M1 | hızlanma | Tasarım F | `τ` |
|---|---|---|---|---|
| — | 11.98 g | 1.00× | 15.6 saat | 3.34 g |
| `fp16` | 10.44 g | 1.15× | 13.6 saat | 2.91 g |
| `kron` | **8.17 g** | 1.47× | 10.6 saat | 2.28 g |
| **`fp16+kron`** | **6.63 g** | **1.81×** | **8.6 saat** | 1.85 g |

M1 düzeyinde de ayrık: 1.15 ve 1.47'den ayrık-null 1.82×, ölçülen 1.81×.

---

### 6.10 Maliyet modelinin altıncı hatası: kalibrasyon hiç yazılmamış

**Bu, altı sürümde bulunanların en büyüğü, ve `m1_run.py`'nin süresini soran
soru ortaya çıkardı.**

`sequential_calibrate` nokta başına her bloğu **iki kez** dolaşıyor: bir kez
hook'larla Hessian'ları toplamak, bir kez de sonraki blok sıkıştırılmış çıktıyı
görsün diye (Spec v6 tuzak 20). Model ikisini de yazmıyordu.

Ölçüldü (Llama-2-7B blok 0, 7 linear, 16,384 token):

| biriktirici | süre | nokta başına | float64'e bağıl fark |
|---|---|---|---|
| **CPU float64** (kodun geldiği hâl) | 19.65 s | **5.59 h** | 0 |
| CUDA float64 | 29.86 s | 8.49 h | 3.1e-17 |
| **CUDA float32** | **0.91 s** | **0.26 h** | 5.06e-06 |
| CUDA float64 + float32 çarpım | 0.99 s | 0.28 h | 5.08e-06 |

Nokta başına 5.59 saat — **her tile boyutunda sıkıştırmanın tamamından
pahalı**. M1'in 105 noktasında 28 gün.

**Sebebi mekanik.** `collect_block_statistics` biriktiriciyi `device="cpu"` ile
kuruyordu ve hook her aktivasyonu `.to("cpu")` ile kopyalıyordu — yani `Xᵀ X`,
aktivasyonlar ve blok zaten GPU'dayken CPU'da yapılıyordu. Bloğun kendi
cihazında biriktirmek **25×**.

> **Ve kendi önerimi çürüttüm.** "float32 çarpımı float64 bir toplayıcıya
> eklemek hassasiyeti neredeyse bedava geri alır" demiştim. Ölçüldü: %9 daha
> pahalı ve **hiçbir şey kazandırmıyor** (5.08e-6'ya karşı 5.06e-6). Hata
> toplamada değil **çarpımda**; daha geniş bir toplayıcı çarpımın attığını geri
> getiremiyor. Parçaları küçültmek de işe yaramıyor — toplam hata her hâlükârda
> `√(toplam token)` gidiyor. API'de kaldı ve *neden işe yaramadığı* yazıldı.

**Tasarım ekonomisi tersine döndü.** Sıkıştırma baskınken maliyeti *hangi*
tile'ları koştuğun belirliyordu; kalibrasyon baskınken **kaç nokta** koştuğun
belirliyor. Tasarım G (2 tile × 5 çekiliş = 10 nokta) Tasarım F'ten (7 nokta)
**pahalı** hâle geliyordu — ucuz kaçış kapısı olmaktan çıkıyordu. Hessian
GPU'ya alınınca G yeniden ucuza döndü ve ikisi eşitlendi (19.1'e karşı 20.4).
**08-25'te üçüncü kez döndü:** `TILE_TIMINGS` ince ucu doğru fiyatlayınca G,
F'ten belirgin biçimde pahalı çıktı (27.1'e karşı 20.1), çünkü uçlarından biri
T=1 (§6.14). Gözlem duruyor; yönü üç kez değişti.

**`m1_run.py` bugün başlatılsaydı:**

| senaryo | M1 | Tasarım F |
|---|---|---|
| modelin söylediği (iki terim de yok) | 11.98 g | 15.6 saat |
| **gerçek kod, iki düzeltmeden önce** | **~38–40 g** | **~57–60 saat** |
| + Hessian GPU'da (§6.10) | 15.0 g | 20.4 saat |
| + `TILE_TIMINGS` yeniden ölçüldü (§6.14) | **15.0 g** | **20.1 saat** |
| + telafi bloklanmış (§6.11c, varsayılan kapalı) | 13.7 g | 17.9 saat |
| + fp16 + kron (varsayılan kapalı) | 8.9 g | 12.9 saat |
| + üçü birden (varsayılan kapalı) | **7.5 g** | **10.6 saat** |

"Önce" satırı aralık, çünkü aynı ölçümün iki koşusu 19.65 s ve 22.37 s verdi —
bu makinede %14 koşudan koşuya. Kod artık öyle yapmadığı için kesinleştirmeye
değmez; kaydedilmesi gereken şey aralığın kendisi.

**Neyin altı sürüm boyunca saklanmasına izin verdiği kayda değer:** tam
sürücüyü kimse koşmadı, çünkü `m1_run.py` yok. §8.1'in kritik yol olmasının
sebebi yalnızca "veri yok" değil — **ölçülmeyen maliyet de orada birikiyor.**

---

### 6.11 İki kapı, bir eksik terim, ve kendi kaydımın düzeltilmesi

**a) `_nearest` ızgaranın 21 hücresinin 10'unda 65,536 kodsözcüğünü tarıyordu.**

Hızlı yol tek bir kapıyla açılıyordu — `_LATTICE_MIN_ROWS` (cuda'da 1024) — ve
analitik aramanın **kendi** eşiği (`_ANALYTIC_MIN_ROWS = 384`) yalnız o kapının
*içinde* okunuyordu. Yani **384 ≤ satır < 1024 aralığı analitik yola hiç
ulaşamıyordu.**

Köşe durum değil: LDLQ süpürmesi `_nearest`'e `chunk × lines_per_tile` satır
veriyor — T=1 ve T=2'de 512, T=4'te 816. Tam ızgaranın ince ucu, tile sayısının
en büyük olduğu yer.

Ölçülen doğrudan-yol krossoveri **256**, 384 değil — çünkü 384 *geri düşme*
yolunun eşiği ve orada kafes çözücünün bedeli zaten ödenmiş:

| n | 0.05 | 0.6 | 6.0 |
|---|---|---|---|
| 128 | 0.41× | 0.63× | 0.41× |
| 256 | 0.99× | 1.21× | 1.68× |
| 512 | 2.04× | 1.61× | 3.33× |
| 816 | 2.68× | 5.19× | 6.04× |

Kapı açılınca **taramaya sıfır satır** düşüyor, ve ızgaranın gerçek
şekillerinde süpürme:

| hücre | satır | önce | sonra | |
|---|---|---|---|---|
| T=1 4096×4096 | 512 | 0.332 s | 0.163 s | **2.04×** |
| T=2 4096×4096 | 512 | 0.682 s | 0.294 s | **2.32×** |
| T=4 4096×4096 | 816 | 1.337 s | 0.382 s | **3.50×** |
| T=8 4096×11008 | 560 | 2.701 s | 1.101 s | **2.45×** |

> **Sınıfı tanıdık: `_on_device` ile aynı.** Bir algoritma için kalibre edilmiş
> bir kapı, sonradan gelen daha iyisini sessizce dışarıda bırakıyor, ve belirti
> yanlış cevap değil yavaş cevap. Bu yüzden testler **yolu** izliyor: tarama
> ile analitik zaten yapı gereği aynı cevabı veriyor — boşluğun fark
> edilmemesinin sebebi de o.

**Kesinlik.** Analitik arama exact; float32'de milyonda bir gerçek berabere
başka türlü bozulabiliyor (§6.4'te zaten kabul edilmiş bir takas). Uçtan uca
dört tile'ın üçü bit-birebir aynı kaldı, T=16'da katman hatası **5.8e-5**
oynadı — Kapı B'nin görebildiğinin **550 katı altında**.

**b) Maliyet modelinin yedinci hatası: ileri telafi hiç yazılmamış.**

`run_config` her şeyden önce `prune`'u çağırıyor, `TILE_TIMINGS` ise
`ldlq_quantize_blocks`'tan başlıyor. Arada `forward_compensate` var — `n_in`
uzunluğunda bir Python döngüsü, ve her yinelemesi kalan bütün genişliğe
dokunuyor. Ölçüldü: blok başına **40.7 s**, nokta başına **0.362 saat**,
M1'de **1.58 gün**. Kalibrasyon gibi tile boyutundan bağımsız.

**c) Ve "bloklamak kazandırmıyor" kaydım yanlıştı.**

§7.2'ye "0.90× / 0.87× / 1.06× — kazanç yok" diye yazmıştım. O ölçümü yalnız
(512, 2048) ve (512, 4096)'da yapmışım — **fırlatma bağımlı** rejimde, ve orada
bloklama gerçekten hiçbir şey kaldırmıyor. Gerçek katman genişlikleri **bant
genişliği bağımlı**:

| n_out × n_in | kesin | blok=512 | |
|---|---|---|---|
| 4096 × 4096 | 2431 ms | 665 ms | **3.65×** |
| 11008 × 4096 | 6345 ms | 820 ms | **7.74×** |
| 4096 × 11008 | 18260 ms | 1837 ms | **9.94×** |

Terimin tamamında **6.63×** (0.362 → 0.055 h/nokta). Bit-birebir **değil**
(ertelenen kuyruk tek matmul, 2.7e-6…4.8e-6), o yüzden `compensate_block`
eklendi ve **varsayılan `None`** — kesin düzen.

> **Ders, hızdan daha değerli:** *yanlış rejimde ölçülmüş bir ret, hiç ölçmemekten
> kötüdür* — çünkü bir sonrakinin bakmasını durdurur. Bu kayıt beni sekiz gün
> boyunca yanlış yerde tuttu.

**Bugünkü dağılım** (B=1.5, 7 tile toplamı):

| terim | süre | pay |
|---|---|---|
| codebook | 7.51 h | 36.8% |
| rotasyon | 6.48 h | 31.8% |
| **telafi** | **2.53 h** | **12.4%** |
| kalibrasyon | 2.31 h | 11.3% |
| cholesky | 1.10 h | 5.4% |
| eval | 0.46 h | 2.3% |

**M1 = 15.0 gün** (§6.14'ten sonra). Telafi yazılınca 13.4'ten 15.0'a çıkmıştı;
bloklanırsa 13.6'ya döner (§8.5).

---

### 6.12 Dikiş GPU'da hiç koşmamış — bir yolda beş kusur

08-25. §8.1'i yazmadan önce onun **kullanacağı dikişe** bakıldı:
`sequential_calibrate` → `run_config`. `sequential_calibrate(device="cuda")` —
`m1_run.py`'nin yapmak zorunda olduğu tam çağrı — **çalışmıyordu**, ve arkasında
üst üste beş kusur vardı:

| # | kusur | belirtisi |
|---|---|---|
| 1 | `block_kwargs` cihaza hiç taşınmıyordu | rotary CPU'da kalıyor, blok `apply_rotary_pos_emb` içinde ölüyor |
| 2 | `LayerProblem`'in W'si CPU'ya sabitlenmiş, H bloğu izliyor | cihaz uyuşmazlığı — **ve hiçbir argüman uzlaştırmıyordu** |
| 3 | `dtype` varsayılanı GPU'da float64 | 1/64 hız: blok başına 29.9 s'ye karşı 0.9 s |
| 4 | `inputs` liste sözleşmesi `device` verilince bozuluyor | çağıranın listesi blok 0'da donuyor |
| 5 | `run_config` `W_hat`'ı hesaplayıp atıyor | `compress_fn` ağırlık döndürmek zorunda — iki yarı **bağlanamıyordu** |

**Beşi de 599 testin kör noktasında**, ve hepsi aynı sebeple: CPU'da bunların
hiçbiri bir şey yapmıyor. `.cpu()` zaten CPU'daysa no-op, CPU `block_kwargs`
CPU bloğun yanında zaten doğru, hiçbir şey taşınmıyorsa yeniden bağlama zararsız.

> **Bu kör nokta §14.1'de zaten kayıtlıydı** — iki önceki düzeltmeden. Yine
> vurdu, ve bu kez tam da hızlandırma commit'inin (`8c56f1e`) kendisinde:
> o commit dokunduğu **parçaya** CUDA testi ekledi (`collect_block_statistics`),
> onu **çağıran sürücüye** eklemedi.

**3'ün fiyatı ölçülmedi, hesaplandı** — modelin kendi kayıtlı oranıyla:

| | M1 |
|---|---|
| modelin fiyatladığı (`cuda_f32`) | **15.0 gün** |
| kodun varsayılanının ürettiği (`cuda_f64`) | **50.9 gün** |

25× kazanç gerçek ama **çağıranın `dtype=torch.float32` yazmasına bağlı**.
Varsayılan yine de float64 bırakıldı: float32 kolunun 5.06e-06'sının ölçüldüğü
referans o, ve bu sürücüden geçen **hiçbir kalite sayısı henüz yok** — ucuz kolu
seçmek koşacak olanın kararı, bir yan etki değil. Docstring'e ve §10'a yazıldı.

**Şimdi kapalı, ve testler cevabı değil yolu izliyor** (§14.2): blok 0'a bir
pre-hook takıp rotary'nin *hangi cihazda geldiğini* sayan bir test, `compress_fn`
içinde W/H/act_norm yerleşimini kümeye toplayan bir test, ve dikişin tamamını
koşan bir uçtan uca test. Beşi de HEAD'e karşı **kırmızı** olduğu gösterilerek
kabul edildi.

Kanıt, uçtan uca: gerçek Llama blokları → `sequential_calibrate(device="cuda")`
→ `run_config(return_weight=True)` → ağırlıklar modele geri → `streamed_perplexity`.

---

### 6.13 Üç sabit, tek eşitsizlik, ızgaranın ortası

08-25, ikinci bulgu. Üç sabit birbirinden habersiz ayarlanmış ve aralarında
**hiçbir yerde yazılmayan** bir eşitsizlik varmış:

```
CHUNK_TARGET_ROWS * DECODER_MISS_FRACTION  >  _ANALYTIC_MIN_ROWS
```

Soldan sağa: `auto_chunk` süpürmeyi kaç satıra nişanlıyor, çözücü bunların ne
kadarını devrediyor, ve o artık analitik yola geçecek kadar büyük mü. Ölçülen
üçüncü sayı çözücünün **%34.9**'u çözemediği (üç şekilde de aynı — yapısal).

```
1024 × 0.349 = 357  >  384   →  YANLIŞ
```

Yani süpürmenin **her grubu** 65,536 kodsözcüğünü tarıyordu.

**Ve bu ızgaranın köşesi değil, ortası.** `auto_chunk`'ın doyum tavanı
`ceil(hedef / lines)`, yani `lines` 1024'ü böldüğü her yerde chunk **tam 1024
satıra** oturuyor — B=1.5'te 21 hücrenin **sekizi**:

| hücreler | satır | yol |
|---|---|---|
| T=1, 2, 4 ve T=8'de `down_proj` | 192–816 | doğrudan analitik, temiz |
| **T=8, 16, 32 — sekiz hücre** | **1024** | çözücü + **TAM TARAMA** |
| T=max | 4096–11008 | çözücü + analitik, temiz |

Gerçek katmanda sayıldı (blok 0 `o_proj`, 2048 satır, B=1.5) — grup başına bir
çağrı, her seferinde:

| T | tarama çağrısı | taranan satır | düzeltmeden sonra |
|---|---|---|---|
| 8 | 581 | 184,915 | **0** |
| 16 | 623 | 193,184 | **0** |
| 32 | 645 | 196,712 | **0** |

**İki sabit birden oynadı, çünkü hiçbiri tek başına yetmiyor.** Satır hedefini
büyütmek eriştiği hücreleri kurtarıyor; `down_proj`'a erişemiyor — orada k=7912
chunk'ı **bellekten** 67 tile'a kapatıyor ve satır 1072'de kalıyor. Eşiği
düşürmek de yalnız o hücrede kayda değer.

**Eşik ölçülürken bir tuzak daha çıktı.** Tek ölçekte (a=0.6) krossover 192
görünüyor ve orada 1.13×. Üç ölçekte bakınca 192 diğer ikisinde **kaybediyor**;
her ölçekte kazanan ilk değer **320**. Mevcut `_ANALYTIC_DIRECT_MIN_ROWS`'un
kendi notu aynı şeyi söylüyordu ("256, 192 değil — 192 üçünden birinde
kaybediyor") ve yine de aynı tuzağa düştüm.

| satır | a=0.05 | a=0.6 | a=6.0 |
|---|---|---|---|
| 192 | 0.74× | 0.90× | 0.66× |
| 256 | 0.93× | 1.42× | 1.42× |
| **320** | **1.65×** | **1.78×** | **1.61×** |

**Satır hedefi de bayattı.** `CHUNK_TARGET_ROWS`'un docstring'i "256'dan 1024'e
%3, üstünde alacak bir şey yok" diyordu — analitik aramadan, toplu fit'ten ve
Triton'dan **önce** ölçülmüş, ve yalnız **doyumu** fiyatlıyordu, satır sayısının
hangi arama **yolunu** seçtiğini değil. Gerçek şekillerde ve gerçek tile
sayılarıyla yeniden ölçüldü; plato **2048**'de (toplam 1.20× / 1.24×, aradaki
fark bu ölçümlerin %2–5 yayılımının içinde), ve ötesinde üç şekilde zaten
`CHUNK_BUDGET_BYTES` bağlıyor.

**Hız — gerçek şekiller, gerçek tile sayıları, tek süreçte dönüşümlü, boş kart:**

| şekil | eski 384/1024 | yeni 320/2048 | | taranan satır |
|---|---|---|---|---|
| T=8 k=2816 | 8208 ms | 6992 ms | 1.17× | 479,415 → 0 |
| T=16 k=2944 | 6559 ms | 4750 ms | **1.38×** | 505,066 → 0 |
| T=32 k=3008 | 5662 ms | 3658 ms | **1.55×** | 510,812 → 0 |
| T=16 k=7912 (`down`) | 14439 ms | 12577 ms | 1.15× | 1,329,201 → 0 |
| **toplam** | 34.9 s | 28.0 s | **1.25×** | |

Yayılımlar %1–10. İki uçtaki 1.15–1.17×'in sebebi aynı: orada **bellek**
bağlıyor, satır hedefi erişemiyor — `down_proj` 67 tile'da, T=8 195'te kapanıyor.

**Kalite bedeli — gerçek katmanda, ve neredeyse yok.** T=8 ve T=16 **bit-birebir**
aynı; T=32'de bağıl hata **−0.00002%** (ve lehte). Kapı B'nin ayırabildiği
%3.2'nin 160,000 katı altında. §6.11a aynı takası 5.8e-5 ile almıştı; bu ondan
da küçük.

> **Maliyet modeli o gün güncellenmedi; ertesi gün yeniden ölçüldü** (§6.14).
> Çarpanla yamamak yerine `TILE_TIMINGS`'in tamamı `n_tiles` kaydedilerek
> yeniden alındı — ve doğru olan buymuş: ölçüm, tahmin edilen ~%4'ü değil,
> ızgaranın **şeklini** değiştirdi.

> **Mikro-benchmark yine fazla vaat etti.** 353 satırlık artık kümede analitik,
> taramaya karşı izole ölçümde **2.03×**. Aynı taramaları süpürmeden kaldırmak
> **1.04×** ediyor — tarama, peşinden gelen üçgen çözüm ve geri besleme
> matmul'üyle örtüşüyor, yani izole maliyetinin çoğu yerinde zaten gizli. §6.3'ün
> kuralı bir kez daha, ve bu sefer **sildiğin** çekirdek için.

---

### 6.14 Maliyet modelinin sekizinci hatası: 4 satırın altında hiç örnek yokmuş

`TILE_TIMINGS`'in üç satırı elle ölçülmüştü ve **hangi `n_tiles` ile** ölçüldüğü
hiçbir yerde yazmıyordu. Bu bir ayrıntı değil: `auto_chunk` `n_tiles`'ı chunk'a,
chunk'ı da `_nearest`'in gördüğü satır sayısına çeviriyor — yani **hangi arama
yolunun koştuğunu** o sayı belirliyor. Bir tile süresi, alındığı tile sayısı
olmadan yorumlanamaz.

Bedeli §6.13'te göründü: iki sabit oynadı ve eski satırlar yeni davranışa
**ölçeklenemedi**, çünkü hangi rejimde alındıklarını kimse söyleyemiyordu.

`experiments/m0_tile_timings.py` yazıldı; şekilleri seçmiyor, `accounting`'den
**türetiyor**. B=1.5'te 4096×4096 katmanının bütün tile ekseni:

| T | k | satır | n_tiles | chunk | s/tile | yayılım |
|---|---|---|---|---|---|---|
| 1 | 1024 | 1 | 4096 | 512 | 0.00729 | — |
| 2 | 2048 | 2 | 2048 | 256 | 0.00881 | %1 |
| 4 | 2560 | 4 | 1024 | 204 | 0.00997 | %7 |
| 8 | 2816 | 8 | 512 | 195 | 0.01410 | %5 |
| 16 | 2944 | 16 | 256 | 128 | 0.01882 | %4 |
| 32 | 3008 | 32 | 128 | 64 | 0.03000 | %2 |
| max | 3072 | 4096 | 1 | 1 | 1.95030 | %4 |

**İki örnek şekli bilerek değişti.** `(3072, 128)` emekli — ızgarada 128 satırlı
hücre yok, o nokta kaba ucu temsil etsin diye konmuş bir sondaydı ve T=max'in
gerçek şekli **tek tile'da 4096 satır**, bambaşka bir rejim. T=1, 2, 8 ve 32
eklendi.

**Ve asıl bulgu o eklemede.** Eski küme 4 satırın altında hiçbir şey
taşımıyordu, model de en yakın satır sayısını log uzayında seçtiği için T=1 ve
T=2 **4-satır oranını ödünç alıyordu**. Ağırlık başına maliyet:

| satır | 1 | 2 | 4 | 8 | 16 | 32 | 4096 |
|---|---|---|---|---|---|---|---|
| s/ağırlık | **6.36e-6** | 1.77e-6 | 7.82e-7 | 5.36e-7 | 3.54e-7 | 2.89e-7 | 1.55e-7 |

Uçtan uca **41 kat**, ilk adımda tek başına **8 kat**. Sebebi `fit_scale`: tile
başına bir kez uydurulyor, ve tek satırlık bir tile ona amortize edecek **128
vektör** veriyor, T=4 **1280**.

**Toplam neredeyse kıpırdamadı, şekil kaydı.** M1 **15.0 gün**te kaldı — ince uç
pahalılaştı, kaba uç ucuzladı, ikisi birbirini götürdü. Ama:

- **Tepe ortadan ince uca kaydı.** §6.1 "maliyet ızgaranın ortasında tepe
  yapıyor" diyordu; eğri artık T=1'den itibaren monoton azalıyor.
- **T=1'de bir duvar geri geldi.** Codebook 2.86h, rotasyon + cholesky toplamı
  0.80h — **3.6 kat**. `test_no_single_term_dominates_the_pass_any_more`'un
  "hiçbir terim kaçmıyor" iddiası bu yüzden kırmızıya döndü ve **haklı olarak**:
  o iddia da süresi dolmuş bir olguymuş, ince uç yanlış fiyatlandığı sürece
  doğru duruyordu.
- **Tasarım G, F'ten pahalı oldu** (27.1'e karşı 20.1 saat) — üçüncü kez yer
  değiştirdiler, ve sebebi G'nin uçlarından birinin T=1 olması (§6.6).
- **`τ` süpürmesi 4.5 → 5.5 gün.**

> **Ve T=1 ızgaranın herhangi bir hücresi değil.** Yapısız taban orası — tezin
> `d(T) − d(1)` özdeşliğinde kıyas grubu. Yani modelin en yanlış bildiği hücre,
> tezin en çok konuştuğu hücreymiş.

**Bu hata üçüncü kez provenance düzeltilirken çıktı**, aranarak değil. Diğer
ikisi kalibrasyon ve ileri telafiydi (§6.10, §6.11b). Üçünün ortak yanı: model
bir şeyi *yanlış* hesaplamıyordu — **bilmiyordu**.

---

### 6.15 Dokuzuncu hata: aynı kusur, ikinci sabitte

`SCALE_FIT_MULTIPLIER` de tek bir sayıydı — 1.39 — ve `fit_scale`'in bir tile'a
ne kadar eklediğini tarif ediyordu. §6.14 ile aynı sınıf: satır ekseni boyunca
değişen bir nicelik, tek bir sabitle modellenmiş.

Aynı koşuda ikinci bir kol ölçüldü — her hücre bir kez `scale="per_tile"`, bir
kez `fit_scale`'in bulacağı ölçek **sabit verilerek**, dönüşümlü. (Rastgele bir
sabit ölçek olmaz: kötü ölçeklenmiş girdi kafes çözücüde çok daha fazla ıskalar,
yani süpürmenin iş miktarı değişir ve oran iki farklı şeyi fiyatlar.)

Oran **artıklar üzerinden** okunuyor, ham tile süresi üzerinden değil: cholesky
iki kolda da var ve fitin değiştirdiği bir şey değil, içeride bırakılsa oranı
her genişlikte farklı bir miktarda seyreltirdi — satır-eksenli tablo sessizce
genişlik-eksenli olurdu.

| satır | 1 | 2 | 4 | 8 | 16 | 32 | 4096 |
|---|---|---|---|---|---|---|---|
| ham | 2.22 | 1.75 | 1.63 | 1.55 | 1.57 | 1.57 | 2.16 |
| **artık** | **2.60** | 2.07 | 1.92 | 1.71 | 1.70 | **1.65** | **2.17** |

**Eğri U şeklinde, ve iki ucun sebebi farklı:**

- **İnce uçta** `fit_scale`'in *sabit* bedeli az vektöre bölünüyor — T=1'de bir
  tile 128 vektör tutuyor, T=4'te 1,280.
- **Kaba uçta** fit her vektöre **24 kez** bakıyor, süpürme bir kez. T=max'te
  süpürme verimli (tek tile, grup başına 4096 satır, kart dolu), yani fitin
  orantılı 24 katı artık süpürme yükünün arkasına saklanamıyor.

Yani tek bir sabit **iki ucu birden** kaçırıyor. 1.39, eğrinin düz ortasında
kalibre edilmiş bir sayı.

> **Ve kaydedilmiş gerekçe tersine dönüyor.** Sabitin kendi notu "1.39 dört-satır
> figürü, üçünün **en elverişlisi**, tartışılan kaldıracı abartma konvansiyonuyla
> tutuldu" diyordu. Abartmıyormuş: bugün ölçülen **her** değer onun üstünde,
> %18'den %87'ye kadar. Yani "tile başına fiti atmanın maliyet gerekçesi kalmadı"
> reddi, kaldıracı olduğundan **küçük** gösteren bir sayıyla savunulmuş.

**Kaldıraç gerçekten büyüdü:** fiti tamamen atmak 1.42 gün değil **2.76 gün**
(M1 15.0 → 12.25), ve T=1'de nokta başına **1.66×**.

**Ama erişilebilir olan değişmedi**, ve asıl kayıt bu:

- Fiti **tamamen** kaldırmak (sabit ölçek / `per_layer`) zaten **kalite**
  gerekçesiyle reddedildi (%11 kötü, 08-23). Büyüyen bir sayı kapanmış bir
  seçeneği geri açmıyor.
- §5.8'in önerdiği **örnekleme** ince uçta bir **no-op**: T=1'de tile 128 vektör
  tutuyor, varsayılan sınır 8,192. Sınır, maliyetin yaşadığı yerde hiç ısırmıyor.
- Geriye erişilebilir tek biçim fiti **tile'lar arasında** toplamak kalıyor —
  §7.2'de bit-birebir olmadığı için (indirgeme sırası) alınmamıştı, 2.16×
  ölçülerek. İnce uçta yeniden fiyatlanmayı hak ediyor; **yapılmadı**, kaydedildi.

**Yan sonuç: ölçüm artık tekrarlanıyor.** Aynı betik, aynı makine, iki bağımsız
koşu, yedi şekil — en büyük sapma **%3.4**. Kıyas için 08-24: elle ölçülmüş üç
tile süresinden **ikisi hiç tekrarlanmadı** (%28 ve %65, §6.3). Betiğe
çevirmenin asıl kazancı hız değil, bu.

---

### 6.16 Sürücü koştu, ve model gerçek blokta 5.2× iyimser çıktı

> **BU BAŞLIK 08-25'te DÜZELTİLDİ — §6.18'i okumadan buradan ayrılma.** Model
> gerçek katmanda 5.2× iyimser değil, **1.03×**. Aşağıdaki 65 s'nin kendisi
> yanlıştı: 5.52× ile fp16'nın 1.38×'i terimlere elle uygulanarak çıkarılmıştı
> ve ikisi de fazla kredi verdi. Doğru karşılaştırma 339'a karşı **84**, ve
> boşluğun tamamı **bağlam**. Bölüm olduğu gibi duruyor çünkü o günkü akıl
> yürütme — ve yanlış çıkan tarafı — kayda değer.

08-25. `m1_run.py` yazıldı (§8.1 kapandı) ve gerçek Llama-2-7B üzerinde koştu.
İlk gerçek blok sıkıştırıldı, checkpoint yazıldı, ve **resume gerçek modelde
doğrulandı** — "resuming at block 1 of 32", ardından blok 2, 3, 4.

İlk gerçek sayılar (blok 0, B=1.5, T=16, 4 kalibrasyon penceresi — gösterge
değil, **tesisat kanıtı**):

| katman | rel. hata | katman | rel. hata |
|---|---|---|---|
| q_proj | 0.0988 | gate_proj | 0.2183 |
| k_proj | 0.1065 | up_proj | 0.2219 |
| v_proj | 0.1799 | down_proj | 0.2723 |
| **o_proj** | **0.4199** | | |

**Ve maliyet modeli ilk kez koşarak sınandı.** Blok başına, iki temiz delta
(blok 3 ve 4: 5.6 ve 5.7 dk):

| | modelin dediği | gözlenen | |
|---|---|---|---|
| kaldıraçlar açık | 65 s | **339 s** | **5.2×** |
| kaldıraçlar kapalı | 180 s | 339 s | 1.9× |

**Bu, dokuz hatanın hiçbirinin sınıfı değil.** Öncekilerin yedisi "modelin
bilmediği bir şey"di ve hepsi modeli *okuyarak* bulundu. Bu, liste doğruyken
**oranların** yanlış olması — ve hattı **koşarak** bulundu.

**Kaldıraç çarpanları ölçülmedi, türetildi.** Kron için 5.52×, fp16 için 1.38×
kullandım; birincisi §6.8'in tile-ağırlıklı ortalaması, ikincisi §6.9'un
medyanından çıkardığım bir terim oranı. §6.3'ün kendi kuralı —
*mikro-benchmark'lardan maliyet kurmak burada işlemiyor* — terim oranlarına hiç
uygulanmamıştı.

**Nerede OLMADIĞI da ölçüldü.** Bir bloğun fazları:

| faz | süre | pay |
|---|---|---|
| **`run_config`** | **141.75 s** | **%99.1** |
| `collect_block_statistics` | 0.68 s | %0.5 |
| `output_error` (kayıt için) | 0.41 s | %0.3 |
| blok yeniden ileri | 0.14 s | %0.1 |
| `LayerProblem` kurulumu | 0.02 s | %0.0 |

Üç aday — kalibrasyon, kayıt için hesaplanan `output_error`, problem kurulumu —
**hepsi elendi**. Boşluk `run_config`'in dışında değil, içinde.

**Geriye bağlam kaldı, ve desen şu:**

| aynı yedi katman, aynı şekiller | süre |
|---|---|
| tek tek, **ısınmış** (`m0_pass_breakdown`'ın 2. çağrısı) | 84 s |
| tek blok, **soğuk**, boş kartta ayrı süreçte | 142 s |
| **sürücünün içinde**, blok 3 | **339 s** |

Aritmetik aynı, **bağlam** değişiyor. Sürücüdeki fark kartın o an **7.7 GB**'ta
olmasıyla örtüşüyor; ayrı süreçteki profil 285 MiB'de başladı. Baskın hipotez
**bellek baskısı**: `sequential_calibrate` yedi Hessian'ı (846 MB) blok boyunca
canlı tutuyor, oysa her katman sıkıştırıldıktan sonra kendisininki bırakılabilir.
**Ölçülmedi — sıradaki iş.**

> **Katman ölçümleri de "ısınmış" olduğu için düşük.** `m0_pass_breakdown`
> `clean_seconds` diye **ikinci** çağrıyı raporluyor; birincisi derliyor ve
> önbellek ısıtıyor. Gerçek koşuda her şekil bir kez soğuk karşılanıyor — 84'e
> karşı 142 arasındaki fark bu. **`TILE_TIMINGS` de aynı ısınmış biçimde
> ölçüldü**, yani §6.14'ün sayıları da bu yönde düşük.

**Bir de araç bir kez kendi korumasından geçemedi.** `m0_pass_breakdown`'ın
"sarmalamak cevabı değiştirmemeli" assert'i bir koşuda kırıldı, ertesinde
kırılmadı — **aralıklı**. Gevşetmek yerine **konuşturuldu**: assert duruyor ama
artık farkın büyüklüğünü basıyor. Sebebi: float32-epsilon düzeyinde bir fark
(hatta belirlenimsiz bir indirgeme var) ile büyük bir fark (araç yanlış şeyi
ölçüyor) **zıt tepkiler** gerektiriyor, ve bir tolerans ikisini ayırt edilemez
yapardı. Tekrar görülene kadar açık.

---

### 6.17 Bellek baskısı ölçüldü, ve fp16 sekiz saat sonra geri kapandı

08-25'in ikinci yarısı. §6.16 sürücünün blok başına 339 s harcadığını ve modelin
65 s dediğini bırakmıştı; boşluğun `run_config`'in **içinde** olduğu ve
**bağlamla** büyüdüğü ölçülmüştü (84 s ısınmış → 142 s soğuk ayrı süreçte →
339 s sürücüde). Hipotez bellek baskısıydı.

**Doğrulandı, ve düzeltme ölçüldü.** Aynı yedi katman, aynı sırayla, tek süreçte
dönüşümlü:

| | süre | tepe tahsis |
|---|---|---|
| yedi Hessian blok boyunca tutuluyor (eski) | 122.7 s | 5.40 GiB |
| **her katmanınki bitince bırakılıyor** | **84.2 s** | 5.02 GiB |
| | **1.46×** | |

> **Mekanizma sezgiye aykırı ve taşınmaya değer.** Tepe yalnız 0.38 GiB
> düşüyor ama süre 1.46× iyileşiyor. Kazanç *sığdırmaktan* değil, ayırıcının
> blokları tahliye edip yeniden istemek yerine **yeniden kullanabilmesinden**
> geliyor. Bir bellek tavanı testi bunu göremezdi — ve bir doğruluk testi de
> göremez, çünkü cevap değişmiyor. Koruyan test `weakref` ile **referansı**
> izliyor.

Yan sayı: sıkıştırma tek katmanda **5.4 GiB** tepe yapıyor, kullanılabilir
6.8 GiB'de. Kalibrasyon setini batch başına taşımaya çevirmeseydik (08-25 sabahı)
üstüne 4.0 GiB daha binecekti — **9.4 GiB, kartın tamamından fazla**. Tam koşu
OOM ile ölürdü.

**Ve fp16 geri kapandı.** Sabah üç kaldıraç açılmıştı; fp16 sekiz saat açık
kaldı ve gerçek blok şekillerinde yeniden ölçülünce **hiçbir şey kazandırmadığı**
görüldü:

| katman | fp32 | fp16 | oran |
|---|---|---|---|
| q_proj 4096×4096 | 6.22 s | 6.21 s | **1.002×** |
| gate_proj 11008×4096 | 15.99 s | 15.98 s | **1.000×** |

Onu haklı çıkaran 1.09–1.22× **tek bir katmanda**, üstelik `o_proj`'un 4096
satırının **512'siyle** ölçülmüştü. O boyutta arama geçen sürenin büyük bir
kesri ve fırlatma bağımlı; gerçek genişliklerde süpürme bant genişliği bağımlı
ve fp16'nın kaldıracağı bir şey kalmıyor. **Bir rejimde ölçülmüş, hepsinde
uygulanmış** — bu hafta düzeltilen her sabitle aynı şekil.

Bilanço, üç kalemin üçü de ölçülü:

| kalem | ölçülen |
|---|---|
| GPU'da hız | **1.00×** |
| CPU'da hız | **0.23×** (4.3× yavaş, aritmetik emüle) |
| kalite | ≤%0.90 kötü (§6.9) |

Hiçbir yerde kazandırmayan, bir yerde 4.3× kaybettiren ve kaliteye mal olan bir
kaldıraç. **Varsayılan kapandı; açıkça istenirse hâlâ çalışıyor** — bu projede
her ret erişilebilir kalıyor ki bir sonraki yeniden fiyatlayabilsin.

> **Kalan iki kaldıraç aynı denetimi borçlu.** `rotate_kron`'un 5.52×'i §6.8'in
> tile-ağırlıklı ortalaması, telafi bloklamasının 6.63×'i bir terim oranı, ve
> **ikisi de gerçek blokta ölçülmedi**. Model bir blokta 5.2× iyimser (§6.16) ve
> türetilmiş kaldıraç çarpanları adı konmuş şüpheli.

**Ölçüm aracına dair iki hata daha, ikisi de benim.** Birincisi: §14.2'nin
"kartın boş olduğunu doğrula" kuralı yalnız bir **ön-uçuş** kontrolü, ve
dakikalarca süren bir ölçümde sonradan gelen çekişmeyi hiçbir şey görmüyor.
Üstelik dönüşümlü A/B bunu **gizliyor** — çekişme iki kola da bindiği için
yayılım küçük kalıyor ve her oran 1.00×'e sürükleniyor. Bir fp16 ölçümü tam
olarak böyle kirlendi (başka bir projenin Python'u kartta).

İkincisi: düzeltmeyi önce **saate** bağladım, ve guard ilk gerçek ölçümde
**kendi yüküne** ateş etti — çünkü ölçüm koşarken saat zaten yüksek, kendi
çekirdeklerimizden. Bunu `require_quiet_gpu`'nun docstring'ine ben yazmıştım.
Doğru sinyal **meşgul kart değil, yabancı süreç**: bir PID ya vardır ya yoktur
ve bizim çekirdeklerimiz PID üretmez. `alternating` artık başlangıçta taban
alıyor ve **sonradan geleni** yakalıyor.


### 6.18 İki kaldıraç denetlendi, ve boşluk aritmetikte değilmiş

08-25'in üçüncü diliminde. §6.17 iki kaldıracı **borçlu** bırakmıştı:
`rotate_kron`'un 5.52×'i ve telafi bloklamasının 6.63×'i türetilmişti, fp16 de
öyleydi ve gerçek blokta 1.00× çıkmıştı. Denetim `m0_lever_audit.py` ile
yapıldı — her kaldıraç **iki kez** ölçülüyor: terim tek başına (*izole*), ve
lever çevrilerek tüm `run_config` (*yerinde*).

**İkisi de geçti, ve fp16'nın tam tersi biçimde.** Gerçek Llama-2-7B blok 0,
B=1.5, T=16, 32,768 gerçek token, doğrulanmış boş kart, tek süreçte dönüşümlü:

| katman | hat | kron kapalı | telafi kapalı | kron yerinde/izole | telafi yerinde/izole |
|---|---|---|---|---|---|
| q_proj 4096×4096 | 6.36 s | 8.75 s (1.38×) | 8.19 s (1.29×) | 2.39/2.42 → **%99** | 1.83/1.92 → **%95** |
| gate_proj 11008×4096 | 16.12 s | 23.10 s (1.43×) | 21.98 s (1.36×) | 6.99/6.53 → **%107** | 5.87/5.57 → **%105** |
| down_proj 4096×11008 | 28.68 s | **85.64 s (2.99×)** | 45.95 s (1.60×) | 56.96/56.10 → **%102** | 17.27/17.38 → **%99** |

Altı ölçümün altısı da **±%7 içinde tutuyor**. fp16'da izole 1.16× demişti,
yerinde 1.00× çıkmıştı — fark, terimin ne olduğunda: fp16 aramanın *içindeydi*
ve darboğaz genişlikle değişiyor; bu ikisi ayrı, toplanabilir terimler.

Blok bazında: yedi katman **86.4 s**, kron kapalı 166.8 s (1.93×), telafi kapalı
122.7 s (1.42×), ikisi kapalı **203.2 s (2.35×)**.

**Kalite de tam genişlikte ölçüldü, ilk kez.** §6.8'in −0.03…−0.31%'i `o_proj`'un
4096 satırının 512'siyle alınmıştı. Tam katmanda kron yine **lehte**: q_proj
−0.040%, down_proj −0.012%, gate_proj 0.000%. Telafi bloklaması üç katmanda da
cevabı oynatmıyor — float32 epsilon'u E8P'nin kafes adımının kat kat altında
(§6.11c bunu söylüyordu, artık ölçüldü).

> **Ve asıl bulgu bu: model gerçek katmanda 5.2× iyimser DEĞİL — 1.03×.**

`m0_localize_gap.py` modelin katman başına tahminini aynı katmanın ölçülen
süresine terim terim koyuyor:

| | ölçülen | modelin dediği | |
|---|---|---|---|
| q_proj | 6.36 s | 6.31 s | 1.01× |
| gate_proj | 16.12 s | 16.00 s | 1.01× |
| down_proj | 28.68 s | 26.60 s | 1.08× |
| **bir blok** | **86 s** | **84 s** | **1.03×** |

§6.16'nın 339 s'si duruyor. Yani **boşluğun tamamı bağlam**, aritmetik değil.
§6.17 bunun 1.46×'ini ölçmüştü (Hessian'ları bırakmak); kalan ~2.7× hâlâ
yerelleştirilmedi ve artık aranacağı yer belli: `run_config`'in içi değil,
sürücünün onu koşturduğu ortam.

**§6.16'nın 65 s'si de yanlıştı.** O sayı 5.52× ile fp16'nın 1.38×'ini terimlere
elle uygulayarak çıkarılmıştı; ikisi de fazla kredi verdi. Doğru karşılaştırma
339'a karşı **84**, yani 4.0× — ve dördü de bağlam.

**Modelde iki kusur, ikisi de KÖTÜMSER — bu yüzden kimse şikâyet etmemişti.**
`m0_cost_model` §6.3'ün on hatasının hiçbirine benzemeyen bir şeyi taşıyordu:
listede olan bir terimi, **hattın koşmadığı aritmetikle** fiyatlıyordu.

1. `rotation_seconds`'ın Kronecker yolu **hiç yoktu** — hat 08-25'ten beri
   kron koşuyor, model her zaman yoğun `2k³` yazıyordu. `down_proj`'ta katman
   başına 57.8 s'ye karşı ölçülen 11.7 s.
2. `model_cost`'un `compensate_block` varsayılanı `None`'dı — hat 512 koşuyor.
   Blok başına 34 s.

İkisi de düzeltildi. Model artık kaldıraçlar kapalıyken **180.0 s / 15.0 gün**
diyor (§6.16 ve §8.5'in kayıtlı sayılarını birebir üretiyor, yani değişiklik
gerçekten ek), açıkken **83.4 s** ve B=1.5'te tam fiyatlanmış.

**Bir hatayı da yaparken yakaladım, ve kaydı bu bölümün en taşınır parçası.**
Kron'u iki ölçülmüş genişlikten flop sayısıyla dışarı taşıdım. `k=1024`'te model
**5.4× kötümser** çıktı — çünkü 1024 tam ikinin kuvveti, `kronecker_factors`
`m=1` döndürüyor ve kasma yoğun çarpıma **çöküyor**. §6.8 bunu zaten ölçmüştü
(k=2048'de 0.99×) ve dışarı taşıma önünden geçip gitti. Yani bu hafta düzeltilen
her sabitle aynı şekil, yalnız bu sefer **ben** yapıyordum.

Çözüm yorumlamak değil ölçmek oldu — §6.14'ün `TILE_TIMINGS` için vardığı yer.
`m0_lever_audit.py --rot-sweep` B=1.5 ızgarasının **on dört genişliğinin
hepsini** ölçtü:

| k | çarpanlar | yoğun | kron | |
|---|---|---|---|---|
| 1024 | 1024×1 | 0.82 ms | 0.69 ms | **1.18×** |
| 2048 | 2048×1 | 4.04 ms | 4.13 ms | **0.98×** |
| 2560 | 512×5 | 9.02 ms | 3.93 ms | 2.29× |
| 2944 | 128×23 | 12.51 ms | 3.02 ms | 4.14× |
| 3008 | 64×47 | 13.91 ms | 2.71 ms | **5.14×** |
| 3072 | 1024×3 | 14.61 ms | 8.10 ms | **1.80×** |
| 7912 | 8×989 | 258.1 ms | 45.0 ms | 5.74× |
| 8256 | 64×129 | 275.7 ms | 25.3 ms | **10.88×** |

**En-yakın-k araması da güvenli değil**, ve tablo bunu kendi içinde gösteriyor:
3008 (64×47) 5.14× koşarken %2 daha geniş olan 3072 (1024×3) 1.80× koşuyor. Oran
k'nin **çarpanlarına** bağlı ve sıçramalı. Bu yüzden `ROT_TILE_TIMINGS` **tam-k**
ile okunuyor; ölçülmemiş genişlik yoğun fiyatlanıyor ve `rotate_kron_priced`
bunu **söylüyor**. Sessiz bir indirgeme bu projenin tekrar tekrar kaydettiği
başarısızlık.

**5.52× de yerine oturdu:** o §6.8'in ızgara-ağırlıklı ortalamasıydı ve
**aritmetiği** ölçüyordu. Gather (`H[idx, idx]`) dahil — yani hattın gerçekten
ödediği hâliyle — aynı ağırlıkla **3.53×**, ve T=1'de 2.00×'ten T=max'ta 8.68×'e
uzanıyor. Gather tile başına sabit bir bedel, ve ucuz kolu pahalı koldan çok daha
fazla seyreltiyor; §6.8 ile aradaki bütün fark bu.

> **Kalan iş:** B=1.60 ve B=1.75 süpürülmedi. Başlandı ve başka bir proje
> koşusu karta girince guard ölçümü reddetti; kısmi satırlar **atıldı**, çünkü
> `alternating` tekrarlar arasında örnekliyor ve hangi satırın hâlâ temiz
> olduğunu söyleyemez. `m1_cost` o yüzden `rotate_kron_priced=False` diyor.

---

## 7. Denenip **reddedilenler** — tekrar denenmesin

Bu bölüm kasıtlı olarak uzun. Bir fikrin denenip elendiğini kaydetmemek, onu
ikinci kez denemek demek.

### 7.1 Bilimsel gerekçeyle reddedilenler

| fikir | ölçülen | neden reddedildi |
|---|---|---|
| **Ölçek örneklemesi** (`fit_scale(sample=N)`) | ortalama küçük, **tohum aralığı 15.8 pp** | Kapı B 0.31 σ'lık farkları ayırıyor; bu gürültü onu boğar (§5.8) |
| **Adım azaltma** (`n_steps` 24→6) | +45.6 / +17.8 / +13.6% | Kaliteyi doğrudan bozuyor, ve `n12`'de işaret bile tutarsız |
| **Katman-başı ölçek** (`per_layer`) | %11, yeniden ölçümde T=4'te **+87.9%** | Küçük `T`'de tile'ların ölçekleri gerçekten farklı |
| **Rotasyonu daraltmak** (blok-köşegen RHT) | `R8` ≈ rotasyonsuz (−3.3%) | Rotasyonun işi kalın kuyruğu **geniş** yaymak; 8 koordinat içinde norm değişmez |
| **TF32** | **hattı kırıyor** | Ölçüldü 08-24 ve iş kalite yüzdesine kalmadı: T=4'te döndürülmüş alt-Hessian Cholesky'den geçmiyor, sönümleme payının %85'i gidiyor. Çalıştığı yerde de +%4.8 ile Kapı B'nin %3.2'sini aşan tek kol (§6.9). **Kapandı** |
| **fp16 arama** | gerçek blokta **1.00×**, CPU'da **0.23×**, kalite ≤%0.90 | 08-25'te açıldı, sekiz saat sonra **reddedildi**. Onu haklı çıkaran 1.09–1.22× tek bir katmanın 512 satırıyla ölçülmüştü; gerçek genişliklerde hiçbir şey kazandırmıyor (§6.17). Açıkça istenirse çalışıyor |

### 7.2 Mühendislik gerekçesiyle elenenler

| fikir | ölçülen | neden |
|---|---|---|
| **`torch.compile` (ilk deneme)** | 15 dk asıldı | Triton yoktu. **Sonra çözüldü** — `triton-windows` (§6.5) |
| **Inductor CPU yolu** | `Compiler: cl is not found` | MSVC yok; cihaz başına eager'a düşülüyor |
| **Elementwise'ı elle azaltmak** (18→11 işlem) | 0.94–1.10× | Çözücü **fırlatma** bağımlı, işlem sayısı bağımlı değil: 8K ve 80K satır aynı süre |
| **Mesafe matrisini maddileştirmemek** | 1.00× | Hesap/bant dengeli; kazanç yok |
| **Yalnız kafes alt sınırıyla dal-sınır** | 1.03–1.10× | Kafes sonsuz, küçük α'da sınır zayıf — tam da pahalı uçta |
| **Birleşik alt sınırla dal-sınır** | 1.21–1.77×, kesin | Reddedilmedi, **alınmadı**: analitik arama (§6.4) getirisini büyük ölçüde sildi |
| **Fit'i TILE'LAR ARASINDA toplamak** | 2.16× | Alınmadı: bit-birebir **değil** (indirgeme sırası). Aday ekseninde toplamakla karıştırılmasın — o **alındı** ve bit-birebir (§6.7) |
| ~~**`forward_compensate`'i GPTQ gibi bloklamak**~~ → **bu kayıt YANLIŞTI** | (512,2048) ve (512,4096)'da 0.87–1.06× | Ölçüm doğru, **rejim yanlıştı**. O genişlikler fırlatma bağımlı. Gerçek katmanlarda (4096×4096, 11008×4096, 4096×11008) bant genişliği bağımlı ve bloklama **3.65× / 7.74× / 9.94×**. Terimde 6.63×. §6.11c |
| **Blok başına CPU↔GPU aktarımı** | nokta başına ~220 s | Saatlere karşı ihmal edilebilir |
| **İki kaydırmayı tek çözümde yığmak** | 1.9–2.2× | Alınmadı: Triton füzyonu aynı kazancı zaten topluyor |

### 7.3 Bilinçli olarak yapılmayanlar

| ne | tarih | neden |
|---|---|---|
| **E8P'nin ucuz doğrulama deneyi** | 08-20 | Kullanıcı kararı. Risk §3.2'de açık varsayım olarak taşınıyor — bu projenin en büyük tek riski |
| **Kernel yazmak** | — | Spec §8 kapsam dışı bırakıyor. Roofline alt sınır olarak sunulur, hız iddiası yapılmaz |
| **AQLM-survivor** | 08-20 | VQ ailesinin en zayıf ve en pahalı üyesi; codebook yeniden kalibrasyonu katman başına saatler |
| **Izgarayı daraltmak** | 08-24 | Artık gerekmiyor (12 gün). Gerekseydi bile tile eksenine dokunulmazdı |

---

## 8. Sırada ne var

### 8.1 ~~Bir sonraki oturumun ilk işi~~ — **KAPANDI 08-25**

**`experiments/m1_run.py` yazıldı, gerçek modelde koştu, resume doğrulandı.**
Kalan iş bu bölümde değil: sıradaki engel §8.6'nın ilk maddesi — ve o madde
08-25'te **yeniden tanımlandı**. Model gerçek blokta 5.2× iyimser değil, 1.03×;
boşluğun tamamı **sürücünün bağlamı** (§6.18). Ondan sonra Tasarım F.

*Aşağıdaki tarif, betiğin ne yapması gerektiğini anlatan hâliyle duruyor.*

**`experiments/m1_run.py`: tam model sürücüsü + checkpoint.**

Şu an yok, ve "sıkıştırılmış ppl hiç ölçülmedi" durumunun sebebi bu.
`calibrate.sequential_calibrate` (mevcut, `compress_fn` alıyor) ile
`eval.streamed.streamed_perplexity` (mevcut) arasını bağlayacak; `compress_fn`
içinde `m1_gates.run_config`'in hattı çağrılacak. `hf_llama.load_llama` ve
`capture_block_inputs` hazır.

> **08-25: dikiş artık gerçekten bağlanıyor.** Yukarıdaki paragraf "mevcut"
> diyerek **fazlasını vaat ediyordu** — `sequential_calibrate(device="cuda")` çalışmıyordu,
> `run_config` ağırlık döndürmüyordu, ve arada beş kusur vardı (§6.12).
> Şimdi ikisi de test altında ve zincir uçtan uca koşuyor. Yani `m1_run.py`
> gerçekten ince bir adaptör olabilir; kalanı **checkpoint** ve **sürücünün
> kendi argümanları**.

Sürücünün geçirmesi gereken iki argüman, ikisi de varsayılan değil:
`dtype=torch.float32` (yoksa M1 15 değil 51 gün — §6.12) ve
`return_weight=True`.

**Kesintiye dayanıklılık bunun parçası, sonradan eklenen bir şey değil.**
15 saatlik bir koşu dizüstünde kesilir. Kayıt birimi blok (nokta başına 32);
anahtar `(model, budget, tile, draw, block)`.

> **Dikkat:** `sequential_calibrate` Hessian'ı **sıkıştırılmış** modelden okuyor
> (Spec v6 tuzak 20). Devam ederken bir sonraki bloğun **girdileri de** kayıttan
> gelmeli — yoksa devam eden koşu kesilmeyenden farklı sonuç verir. Doğrulaması
> net: kesip devam ettirilen koşu, kesilmeden koşulanla aynı ppl vermeli
> (`tests/test_hf_llama.py`'nin tiny Llama'sı ile test edilebilir).

### 8.2 Sonra: ilk gerçek koşu

**Tasarım F** — tek bütçe, tek çekiliş, 7 tile, **13.7 saat**, iki kaldıraç
açık ve ikisi de denetlenmiş hâliyle (§6.18). Hattın gerçek modelde uçtan uca
çalıştığını kanıtlar
ve **ilk gerçek U eğrisini** verir.

Kapı B'yi karara bağlamaz (§5.6: verdikt için ≥5 çekiliş, `gate_b` altında
"undetermined" döner). Ama Kapı A için ve eğrinin şeklini görmek için yeterli.

### 8.3 Ön-kaydı dondurmak — artık maliyet engeli yok

İki kutu açık: **`Δ(T)` tahmin eğrisi** ve **`T*_tahmin`**. İkisi de `τ`
süpürmesine bağlı, ve süpürme 29 gündü. **Artık 4.2 gün**
(`m0_cost_model.sweep_cost`) — yani maliyet artık engel değil.

> **Ama süpürme betiği de yok.** Bu belgenin önceki sürümleri `tau_sweep.py`'ye
> mevcut bir şeymiş gibi atıfta bulunuyordu; değil. Modellenen **maliyeti**,
> yazılmış olan **kodu** değil. §8.1 ile aynı durum: engel bilimsel değil,
> yazılmamış betik.

İki betik de aynı dikiş yerini kullanacak (`sequential_calibrate` +
`streamed_perplexity`), o yüzden §8.1'i yazmak §8.3'ün yarısını da yazmış olur.

Sıra önemli: ön-kayıt donmadan M1 başlamaz. Ama F koşusu ön-kayda girmiyor
(tek çekiliş, kapıları karara bağlamıyor), o yüzden paralel gidebilir.

### 8.4 Ölçülmemiş kalan en büyük fikir

**`fit_scale`'i doğru hedefe uydurmak** (§5.8). Şu an ağırlık uzayında
`‖x − αQ(x/α)‖²` minimize ediliyor, oysa hattın hedefi `tr(E H Eᵀ)`.
Örneklemenin T=16 ve T=max'te kaliteyi **iyileştirmesi** bunun belirtisi.

> **Gerekçesi 08-24'te daraldı, 08-25'te kısmen geri geldi.** "Hem daha iyi hem
> daha ucuz olabilir" diyordu; ucuzluk tarafı gitmiş sayılmıştı (tamamen atsan
> 1.4 gün). Fitin payı satır eksenine yayılınca **2.76 gün** oldu ve T=1'de nokta
> başına 1.66× (§6.15) — yani ucuzluk tarafı tamamen ölü değil. Ama asıl gerekçe
> hiç zayıflamadı ve hâlâ o: **yanlış ölçüye göre daha kesin bir α hâlâ yanlış.**

### 8.5 Hassasiyet kaldıraçları — ikisi açık ve denetlendi, biri reddedildi

Üçü de kodda var ve test edilmiş. 08-24'te üçü de kapalıydı, 08-25 sabahı
kullanıcı kararıyla açıldı, ve aynı gün **ölçüm** ikisini doğrulayıp birini
düşürdü. Hiçbiri bit-birebir değil.

| kaldıraç | durum | hız — **gerçek blokta ölçüldü 08-25** | kalite bedeli |
|---|---|---|---|
| `rotate_kron=True` | **açık, DENETLENDİ** (§6.18) | blok **1.93×**; terim, gather dahil, ızgara-ağırlıklı **3.53×** | −0.04…0.00% (**lehte**), tam genişlikte |
| `compensate_block=512` | **açık, DENETLENDİ** (§6.18) | blok **1.42×**; terim **6.83×** | cevabı hiç oynatmıyor |
| ~~`search_dtype=float16`~~ | **reddedildi** (§6.17) | gerçek blokta **1.00×** | ≤%0.90, CPU'da 4.3× yavaş |

> **Borç kapandı ve iki kaldıraç da geçti.** Her biri iki kez ölçüldü — terim
> tek başına, ve lever çevrilerek tüm `run_config` — ve altı karşılaştırmanın
> altısında yerinde tasarruf izole tasarrufun **%95–107**'si. fp16'da bu oran
> 1.16×'ten 1.00×'e düşmüştü; fark, o terimin aramanın *içinde* olması.

Birlikte bir blokta **2.35×** (203.2 → 86.4 s). Tek tek kron 1.93×, telafi
1.42×. Model bunları fiyatladıktan sonra **M1 15.0 → 11.7 gün**; B=1.5'te blok
180.0 → **83.4 s** (2.16×) ve orada tam fiyatlanmış, diğer iki bütçede değil
(§6.18'in kalan işi).

> **Ve 5.52× rakamı emekliye ayrıldı.** O §6.8'in *aritmetiği* ölçen
> ızgara-ağırlıklı ortalamasıydı. Hattın ödediği hâliyle (gather dahil) aynı
> ağırlık **3.53×**, ve tek bir sayı olamaz: T=1'de 2.00×, T=max'ta 8.68×, tam
> ikinin kuvvetinde **1.00×**. `ROT_TILE_TIMINGS` on dört genişliği tek tek
> taşıyor ve tam-k ile okunuyor.

**Kapsam dışı bırakılanlar** (ölçüldü ya da tanımlandı, yapılmadı):
- **TF32** — reddedildi, kalite yüzdesi yüzünden değil: hattı **kırıyor** (§6.9)
- **Hızlı Hadamard dönüşümü** — kron tam ikinin kuvvetinde hiçbir şey
  kazandırmıyor (`m=1`, ayrılacak tek çarpan yok), ve **k=2048 ızgaranın en
  kalabalık genişliği** (T=1'de blok başına 11,008 tile). Gerçek yeni kod
- **LDLQ süpürmesine CUDA graph / `torch.compile`** — §6.5'in açıkça bıraktığı
  yer. §6.11a şekilleri veriden bağımsız yaptığı için artık **mümkün**: analitik
  arama sabit şekilli, kafes çözücünün geri düşmesi değil

### 8.6 Açık kalan kod işleri

- **Axis A için LDLQ** — şu an `NotImplementedError`; Axis B'de indeks ekseni
  girdi kanalları olduğu için Hessian doğrudan uygulanıyor, Axis A'da sweep
  tile'ın sütunları boyunca olmalı
- **Blockwise (tam SparseGPT) maske seçimi** — M3 teslimatı, şu an `upfront`
- **§3.6'nın üç ablasyonu** — grup konvansiyonu (iki koşullu olmalı),
  quantization/maske hatası ayrımı, hizalama
- **Attention koordinasyonu formülü** — `v_proj`↔`o_proj`, GQA, RoPE çiftleri;
  `T=max` için sert kısıt, hâlâ yalnızca ima edilmiş
- **Eval maliyeti** — 238 s yalnız WikiText-2; C4 ve 5 zero-shot görev hiç
  ölçülmedi ve ön-kayıt §4 ikisini de şart koşuyor
- **Sürücünün bağlam bedeli — sıradaki iş, ve artık nerede OLMADIĞI biliniyor.**
  §6.16 modeli bir blokta 5.2× iyimser sanmıştı; §6.18 aynı yedi katmanı sessiz
  bir süreçte ölçtü ve model **1.03×** çıktı. Yani aritmetik doğru fiyatlanmış:
  boşluğun tamamı, 339 s'ye karşı 86, **bağlam**. 1.46×'i ölçüldü (Hessian'ları
  bırakmak, §6.17); kalan ~2.7× için sürücünün kendisi profillenmeli — aday
  listesi kalibrasyon aktivasyonları, blok ağırlıkları ve önceki katmanların
  tahsisleri
- **`ROT_TILE_TIMINGS`'in diğer iki bütçesi** — B=1.60 ve B=1.75 süpürülmedi
  (karta yabancı iş girdi, kısmi satırlar atıldı). O yüzden `m1_cost`
  `rotate_kron_priced=False` diyor ve M1'in 11.7 günü orada yoğun rotasyonla
  fiyatlanıyor, yani **yukarı yanlı**
- **`TILE_TIMINGS` ısınmış ölçüldü** (§6.16) — `m0_pass_breakdown` ikinci çağrıyı
  raporluyor, gerçek koşuda her şekil bir kez soğuk karşılanıyor
- **Fiti tile'lar arasında toplamak** — §7.2'de bit-birebir olmadığı için
  alınmamıştı (2.16×). §6.15 fitin payını ince uçta 2.60× ölçtü, yani o ret
  yeniden fiyatlanmayı hak ediyor; erişilebilir tek biçim o

---

## 9. Açık riskler

**Kapı A'nın düşme olasılığı yüksek.** Prova (`gate_a_dry_run.md`) GPTQ-4bit
survivor'larla her satırın düştüğünü gösterdi. E8P aritmetiği değiştiriyor ama
**gösterilmedi**. Karar tablosunun `✗/✓` dalı hazır: proje durmaz, çerçeve
daralır.

**E8P kalite varsayımı** (§3.2). Projenin en büyük tek riski. Düşerse bant
1.83–2.83'e kayar ve tezin "2 bitin altı" motivasyonu zayıflar.

**`T*`'ın belirsizliği.** Verdikt tarafı çözüldü; eğri iç bölgede düzse küme
büyük çıkar ve *hangi* granülerlik sorusu cevapsız kalır. Başarısızlık değil
ama manşeti zayıflatır.

**Sentetik σ.** Kapı B'nin gücü ve transfer toleransı sentetik katmandan
ölçüldü. Gerçeği ilk M1 bütçesinden gelecek; ön-kayıt §7.4'ün uyarlanabilir
kontrolü bunun için var.

**Maliyet artık birincil risk değil ama sıfır da değil.** 15 gün hâlâ uzun, ve
bütün süreler **bu makineye ve Triton'lu bir kuruluma** ait. Başka donanımda
eğriler yeniden ölçülmeli. Model **dokuz** kez yanıldı, ve 08-24'te iki ölçümü de
geri çekildi (§6.3) — yani bu sayının belirsizliği modelin kendi hata payından
değil, ölçümlerin tekrarlanabilirliğinden geliyor. Ve 08-25 bir sınır daha
gösterdi: model, kodun **varsayılanlarıyla** koşulduğunda ne olacağını yazmıyor
(§6.12'de 15.0 güne karşı 51).

**Kesinti.** 15 saatlik bir koşu bile dizüstünde kesilir ve şu an devam etme
yok. §8.1'in checkpoint'i bu yüzden kritik yolda.

---

## 10. Ortam tuzakları — saatlere mal oldu, tekrar etmesin

| Sorun | Çözüm |
|---|---|
| **HF indirmeleri takılıyor** (0 B/s) | `HF_HUB_DISABLE_XET=1` |
| **Kimliksiz HF istekleri sert kısıtlanıyor** | `hf auth login` (diske yazar, her süreç görür). `$env:HF_TOKEN` yalnız o pencerede geçerli |
| **`snapshot_download` oturumlar arası devam ETMİYOR** | Bir kez başlat, kesme |
| **Arka plan görev bildirimleri güvenilmez** | Wrapper çıkışı işin bitişi değil. Log dosyasına veya süreç listesine bak |
| **`torchvision` ABI uyumsuzluğu transformers'ı komple kırıyor** | torch'u yükseltirken eşleştir, ya da kaldır |
| **`load_dataset("wikitext", ...)` reddediliyor** | `Salesforce/wikitext` — `namespace/name` gerekiyor |
| **Süreç sayarken kendi ölçüm sürecini sayma** | PowerShell filtresini `python -c` içinden çağırınca kendini yakalıyor |
| ~~**`_on_device` önbelleği cihaz DİZESİYLE anahtarlı**~~ → **düzeltildi 08-24** | `"cuda"` ile `"cuda:0"` **farklı nesneler** döndürüyordu ve hızlı yol bir `is` kontrolüyle seçiliyor, yani kısa yazımı kullanan her çağıran sessizce kaba kuvvete düşüyordu. Üç oturum boyunca belgede durdu, kodda düzeltilmedi, ve **08-24'te dört ölçümü daha yanılttı** — ikisi önce GPU çekişmesine, sonra saat düşüşüne yoruldu, ikisi de yanlış. Belirtisi kötü: optimizasyon **1.00× görünüyor**, yani hata değil sonuç gibi okunuyor. Artık `_device_key` anahtarı normalleştiriyor; ölçülen ek maliyet dönüşümlü A/B'de −%0.8/+%0.4 |
| **Kıyaslamada hızlı yolun açık olduğunu doğrula** | `quantize.is_canonical_codebook(cb)` — bir `assert` ile. Kendi codebook kopyasını kuran ya da cihazı kısa yazan bir kıyaslama hâlâ taramayı ölçer, ve bu **doğru** davranış; tek sorun sessiz olmasıydı. Zamanlama yazarken bunu iddia et |
| **Python stdout tamponu arka plan koşularında** | `python -u` |
| **`torch.compile` Windows'ta çalışmıyor sanılıyordu** | `pip install triton-windows==3.7.0.post26` (torch 2.12 → triton 3.7.0). Sonra `has_triton()` True |
| **Inductor CPU'da `cl` (MSVC) istiyor** | CUDA derleniyor, CPU derlenemiyor. `quantize._shift_kernel` cihaz/dtype başına sondalıyor ve eager'a düşüyor — sessiz, çünkü iki yol birebir aynı |
| **`TILESPARSE_NO_COMPILE=1`** | Derlemeyi kapatır; derli/derlisiz karşılaştırma ve toolchain sorunları için |
| **Mutlak süreler koşular arasında karşılaştırılamaz** | Aynı ölçüm iki koşuda %14–37 oynadı, bazen tanımlanabilir bir sebep olmadan. **Yalnız tek süreçte dönüşümlü A/B geçerli.** Bu oturumda bir kez +%37 okuyup değişikliğe yordum; makineymiş |
| **GPU çekişmesi fırlatma kaldıraçlarını GİZLER** | Başka bir iş kartı doldurunca darboğaz GPU'ya geçiyor ve silmeye çalıştığın gecikme zaten gizleniyor — optimizasyon **1.00× okuyor**. Dönüşümlü A/B bunu düzeltmez; hız fazından önce `bench_guard.require_quiet_gpu()` çağır |
| ~~**`nvidia-smi`'nin `utilization.gpu`'suna bak**~~ → **düzeltildi 08-25** | Bu makinede o sayı **yükle ters korele**: boş kartta %42, kendi yükümüzde %25. Sebebi WDDM — kart ekranı da sürüyor, ve listelenen "compute" süreçleri Windows kabuğu, Edge WebView ve **Claude uygulamasının kendisi**. `mem_get_info` daha kötü: **kör**, 3 GiB tutan yabancı süreci hiç görmüyor. Çalışan tek gösterge `clocks.sm` (boşta %42, yabancı yükte %89). Ölçülüp `experiments/bench_guard.py`'ye yazıldı |
| **TF32 hattı kırıyor** | `allow_tf32=True` ile döndürülmüş alt-Hessian Cholesky'den geçmiyor: sönümleme payının %85'i gidiyor. Açma (§6.9) |
| **`sequential_calibrate` varsayılanı GPU'da float64** | Yani 1/64 hız: blok başına 29.9 s'ye karşı 0.9 s, ve yerine geçtiği CPU float64'ten (19.7 s) bile yavaş. Maliyet modeli `cuda_f32`'yi fiyatlıyor; varsayılanla koşmak **M1'i 15 günden 51'e** çıkarır. Sürücü `dtype=torch.float32` geçmeli — varsayılan bilerek değiştirilmedi (§6.12) |
| **CPU'da koşan bir test cihaz varsayılanını sınayamaz** | `.cpu()` zaten CPU'daysa no-op, CPU `block_kwargs` CPU bloğun yanında zaten doğru. Bu oturumlarda **üç kez** vurdu; sonuncusunda tek bir yolda beş kusur biriktirdi (§6.12). Cihaza dokunan her değişikliğin CUDA işaretli bir testi olmalı, ve test **parçanın değil çağıranın** üstünde |

**Donanım:** RTX 5060 Laptop, 8 GB VRAM, sm_120 (Blackwell → cu128+;
`torch 2.12.0+cu130` kurulu). 23.7 GiB RAM, 16 torch thread'i.
7B fp16 (13.5 GB) GPU'ya sığmıyor → **katman-akışlı zorunlu**, ~2.8 GB tepe.

---

## 11. Repo haritası ve çalıştırma

| Modül | İş |
|---|---|
| `accounting.py` | bit bütçeleri, `1−1/T`, `B*`, canlı bant, V:N:M, `rotation_side_bits` |
| `scoring.py` | saliency — iki ağırlık-başı metrik, iki toplama yönü |
| `tiling.py` | tile bölümlemesi, dondurulmuş maske, `align` |
| `prune.py` | maske seçimi + ileriye telafi; **H1 assert'i burada** |
| `compact.py` | survivor'ları tile başına yoğun bloklara topla |
| `rotation.py` | maske-koruyan rotasyon, blok-köşegen varyant |
| `quantize.py` | E8P codebook, kafes çözücü, **analitik arama**, LDLQ, ölçek politikası, füzyon çekirdekleri |
| `calibrate.py` | sıralı kalibrasyon, `LayerProblem` (**dikiş yeri**), `sequential_calibrate` (**sürücü — henüz yalnız testlerden çağrılıyor**; GPU'da `dtype=torch.float32` geçilmeli, §6.12), Hessian biriktirici (cihazda) |
| `hf_llama.py` | HF adaptörü — blok 0 girdilerini yakalar; `to_device` |
| `eval/perplexity.py` | ppl + protokol koruması + yayımlanmış sayı tablosu |
| `eval/streamed.py` | katman-akışlı ppl |
| `experiments/m1_gates.py` | M1'in iki kapısı, `t_star_set`, çekiliş ekseni, `HESSIAN_BLOCK` |
| `experiments/m0_dense_ppl.py` | dense ölçüm + protokol kimliği |
| `experiments/m0_vq_bits.py` | VQ checkpoint maliyeti — manifest'ten, indirmeden |
| `experiments/m0_gate_b_power.py` | Kapı B'nin gücü + hattın gürültüsü |
| `experiments/m0_transfer_pilot.py` | `Δ = Q + τ` transfer sapması → tolerans |
| `experiments/m0_cost_model.py` | ölçülen eğrilerden gerçek koşu maliyeti |
| `experiments/m0_rotation_value.py` | rotasyon gerçek katmanda kazandırıyor mu; blok genişliği süpürmesi |
| `experiments/m0_scale_fit.py` | ölçek uydurmayı ucuzlatmanın kalite bedeli |
| `experiments/m0_precision_levers.py` | fp16 / kron / TF32, tek tek ve sekiz kombinasyonda |
| `experiments/m0_pass_breakdown.py` | bir geçişin fazları — modelin yazmadıklarını bulmak için |
| `experiments/bench_guard.py` | kartın ölçülecek kadar boş olduğunu **fırlatarak** doğrular; dönüşümlü A/B ve yayılım raporu |
| `experiments/m0_chunk_rows.py` | `auto_chunk`'ın satır hedefi ile arama eşiğinin etkileşimi — yol sayımı + zaman |
| `experiments/m0_tile_timings.py` | `TILE_TIMINGS`'i ızgaranın gerçek hücrelerinden **türeterek** ölçer; `n_tiles` kaydeder |
| `experiments/m1_run.py` | **tam model sürücüsü** — kalibre et, sıkıştır, ölç; blok granülerliğinde checkpoint |
| `experiments/m0_lever_audit.py` | bir kaldıracı **iki kez** ölçer — terim tek başına, ve terim yerinde; `--rot-sweep` ile ızgaranın her genişliğinde kron/yoğun |
| `experiments/m0_localize_gap.py` | modelin katman başına tahminini ölçülen süreye terim terim koyar |

**Belgeler:** `docs/spec_v7.md` (şartname) · `preregistration.md` (M1 ön-kaydı,
**dondurulmadı** — iki kutu kaldı, artık maliyet engeli yok) ·
`docs/audit.md` (v6 denetimi) · `docs/gate_a_dry_run.md` (literatür provası) ·
bu belge.

**Henüz yazılmamış betik** (§8.3): `experiments/tau_sweep.py` — `τ` yüzeyi.
`m1_run.py` ile aynı dikişi kullanacak, yani yarısı yazılmış sayılır.
*(`experiments/m1_run.py` **08-25'te yazıldı** — §8.1 kapandı.)*

```bash
python -m pytest tests/ -q                         # 632 test, ~3 dk
HF_HUB_DISABLE_XET=1 python experiments/m0_dense_ppl.py --seqlens 2048 4096 --device cuda
HF_HUB_DISABLE_XET=1 python -u experiments/m0_rotation_value.py \
    --tiles 4 16 max --seqs 16 --rows 512 --solve-device cuda --solve-dtype float32
    # ~30 dk, 51 kol.  --families H ile yalnız kazanan aile
HF_HUB_DISABLE_XET=1 python -u experiments/m0_scale_fit.py \
    --tiles 4 16 max --rows 512                    # ~20 dk, 54 kol
python experiments/m0_cost_model.py                # ~2 dk, sabitler önbelleklenir
# kaldıraç denetimi: bir kez --build (modeli yükler, blok 0'ın yedi Hessian'ını
# diske yazar), sonra istediğin kadar ölçüm; kart boş olmalı, guard fırlatır
python -u experiments/m0_lever_audit.py --build --budget 1.5 --tile 16   # ~25 dk
python -u experiments/m0_lever_audit.py --rot-sweep                      # ~4 dk
python experiments/m0_localize_gap.py                                    # saniyeler
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --budgets 1.5 --draws 5
python experiments/m0_gate_b_power.py --no-noise   # ~15 dk, σ önbellekten
python experiments/m0_transfer_pilot.py --draws 3  # ~8 dk; --reuse ile saniyeler
python experiments/m0_vq_bits.py --all             # ~100 KB ağ, saniyeler
```

---

## 12. Commit geçmişi — ne anlama geliyorlar

| Commit | Ne getirdi |
|---|---|
| `5d7726d` | Hattın tamamı: muhasebeden ppl'e |
| `f94a8af` | Ön-kayıt taslağı; dondurma listesi görünür bir olay olsun diye |
| `6af48d2` | v6 denetimi repoya taşındı — kararların gerekçesi versiyonlansın |
| `94dbdce` | Spec v7: kafes VQ etrafında yeniden kuruldu, 4 aritmetik hata düzeldi |
| `c3c5632` | V:N:M formülü VENOM'dan; **özgünlük iddiasını daralttı** |
| `1e6218f` | HF adaptörü |
| `33d66a4` | Katman-akışlı eval — 7B'yi 8 GB'da ölçmenin yolu |
| `d80ab14` | **İlk gerçek ölçüm**: protokol sorusu çözüldü |
| `a1626c6` | VQ maliyeti checkpoint'ten ölçüldü; SU/SV ayrışması bulundu |
| `3d8658f` | Kapı B'nin gücü ölçüldü; `T*` küme oldu; çekiliş ekseni düzeldi |
| `7d1ee48` | Transfer pilotu: tolerans kuralı, ve modelin büyük `T` önyargısı |
| `797aa2e` | Maliyet modeli — hattın gerçek boyutta koşamadığının tespiti |
| `baa38a7` | Bellek duvarı kapandı, iki yükleyici hatası düzeldi, `fit_scale` modele girdi |
| `31f9761` | **Rotasyon gerçek katmanda −70%**; hat GPU'ya taşındı |
| `0201f93` | E8 kafes çözücü: CPU 3.5×, GPU 1.9×, çıktı birebir aynı |
| `f425880` | Bu belge, bilinenin etrafında yeniden yazıldı |
| `f00fe9c` | Blok genişliği: **geri besleme daraltılır, rotasyon daraltılmaz**; maliyet modelinin Cholesky eğrisi düzeldi (120 → 94 gün) |
| `40c8d9c` | Süpürme tile'lar arasında toplu — bit-birebir aynı, 94 → 48 gün |
| `7da170c` | Ölçek örneklemesi ölçüldü ve **reddedildi**; fp16 eklendi (kapalı) |
| `a33839b` | **Analitik en-yakın-kodsözcüğü**: arama çözülüyor, taranmıyor. 48 → 29 gün |
| `1a27ead` | Analitik aramanın parça boyutu genişletildi (fırlatma bağımlı) |
| `cc3e0f4` | **Triton kuruldu**, iki elementwise zincir füzyonlandı. 29 → 17 gün |
| `8f5f59f` | Bu belge yeniden yazıldı: yapılan / yapılmayan / reddedilen ayrıldı |
| `1efa971` | **Ölçek adayları tek aramada** (§6.7); maliyet modelinin **beşinci hatası** ve iki geri çekilen ölçüm (§6.3). 17 → 12 gün, ve baskın terim codebook'tan **rotasyona** geçti |
| `0a19f90` | **`_on_device` tuzağı kapatıldı** (§10). Üç oturumdur belgede duran, kodda durmayan hata; dört ölçümü bozmuştu |
| `de8a5ec` | **Kronecker kongrüansı gerçek katmanda ölçüldü** (§6.8). Sentetik ölçüm iki mertebe yanılmıştı; hattın kolunda etki lehte, M1 11.98 → 8.17 g. Varsayılan kapalı |
| `383a64a` | **Üç hassasiyet kaldıracı, tek tek ve kombine** (§6.9). TF32 **hattı kırıyor** — §3.3'ün açık kalemi kapandı. `fp16+kron` bağımsız, birlikte M1 11.98 → **6.63 g** |
| `8c56f1e` | **Kalibrasyon Hessian'ı GPU'da** (25×) ve maliyet modelinin **altıncı hatası** (§6.10). `m1_run.py` bugün 40 gün sürerdi, 13.4 değil |
| `98c0413` | **`_nearest`'in ikinci kapısı** (§6.11a, süpürmede 2.0–3.5×) ve **yedinci eksik terim** (§6.11b). Ayrıca §7.2'deki bir ret **yanlış rejimde ölçülmüş** çıktı (§6.11c) |
| `a3d5a05` | **§8.1'in dikişi GPU'da hiç koşmamış** (§6.12). Bir yolda **beş** kusur, beşi de 599 CPU testinin kör noktasında. Zincir artık uçtan uca koşuyor: gerçek Llama blokları → `sequential_calibrate` → `run_config` → `streamed_perplexity` |
| `7872949` | **Süpürme ızgaranın ortasında 65,536 kodsözcüğü tarıyordu** (§6.13). Üç sabit arasındaki yazılmamış eşitsizlik; 21 hücrenin 8'i. Ağırlıklı **1.25×**, kalite bit-birebir. Ayrıca `bench_guard`: boşluk testi artık alışkanlık değil **assert**, çünkü alışkanlığın baktığı sayı boş kartta %42 okuyor |
| `96f973b` | **`TILE_TIMINGS` `n_tiles` kaydedilerek yeniden ölçüldü** ve modelin **sekizinci hatası** çıktı (§6.14): 4 satırın altında hiç örnek yokmuş. Toplam **15.0'da kaldı** ama **tepe ortadan ince uca kaydı**, T=1'de duvar geri geldi, Tasarım G/F üçüncü kez yer değiştirdi |
| `9990d53` | **Üç kaldıraç açıldı**, ve biri `run_config`'ten **erişilemiyormuş** (`compensate_block`). Varsayılanlar adlandırılmış sabitlerde, çözülen değer her kayda yazılıyor |
| `9dba639` | **`m1_run.py` yazıldı — §8.1 kapandı.** Blok granülerliğinde checkpoint; testi "resume çalışıyor" değil **"resume cevapta görünmez"**. Ayrıca fp16'nın CPU'da 4.3× **yavaş** olduğu bulundu |
| `ffb8a06` + `ac45c1b` | **Sürücü gerçek modelde koştu** ve maliyet modeli ilk kez **koşarak** sınandı: blok başına 65 s dediği yerde **339 s** (§6.16). Dokuz hatanın hiçbirinin sınıfı değil — liste doğru, **oranlar** yanlış |
| `515eb35` | **Bellek baskısı ölçüldü** (Hessian'ları bırakmak **1.46×**, §6.17) ve **fp16 geri kapatıldı**: gerçek blokta 1.00×, onu haklı çıkaran ölçüm tek katmanın 512 satırıymış |
| `5b0a4d3` | **`SCALE_FIT_MULTIPLIER` de satır eksenine yayıldı** — **dokuzuncu hata** (§6.15). Eğri U şeklinde (2.60 … 1.65 … 2.17) ve eski 1.39 hepsinin altında, yani "fiti atmanın maliyet gerekçesi yok" reddi kaldıracı **küçük gösteren** bir sayıyla savunulmuş. Ayrıca ölçüm artık **tekrarlanıyor**: iki koşu, en büyük sapma %3.4 |
| `5271044` | **İki kaldıraç denetlendi**: kron ve telafi bloklaması gerçek blokta tuttu (%95–107); model gerçek katmanda 5.2× değil **1.03×** iyimser, yani boşluk aritmetikte değil bağlamda. Modelin iki kötümser kusuru düzeldi; `ROT_TILE_TIMINGS` on dört genişlikte |

**08-25 oturumunun yayı, tek satırda:** gün, sürücünün ilk kez koşup modeli
5.2× yalanlamasıyla başladı; bellek baskısının 1.46×'i ölçüldü, fp16 sekiz saat
sonra kendi ölçümüyle düştü, ve kalan iki kaldıraç denetlenince **model haklı
çıktı** — gerçek katmanda 1.03×. Yani günün asıl bulgusu bir hızlanma değil, bir
**yer değiştirme**: aranan boşluk `run_config`'in aritmetiğinde değil, sürücünün
onu koşturduğu bağlamda. Ve modelin o boşluğu kısmen gizlediği ortaya çıktı,
çünkü hattın koşmadığı yoğun rotasyonu fiyatlıyordu.

**08-24 oturumunun yayı, tek satırda:** hat 17 günden 12'ye indi (`1efa971`),
sonra modelin iki eksik terimi bulununca gerçeğin ~40 olduğu anlaşıldı
(`8c56f1e`, `98c0413`), ve o terimler düzeltilerek **15 güne** inildi. Aradaki
fark hızlanma değil, **modelin doğrulanması** — ve M1'in koşulup koşulmayacağını
söyleyen sayı o.

---

## 13. Çalışma tarzına dair not

Bu projede en pahalı hata sınıfı **sessizce yanlış bir sayı üretmek**. Bu yüzden:

- Golden sabitler elle yazılmaz, türetilir. `tests/golden.py` `accounting.py`'yi
  **import etmez** — golden değerleri çağıran bir test hiçbir şey kanıtlamaz
- Testlerin çoğu davranış değil **iddia** sınıyor
- Doğrulanmamış şeyler açıkça "varsayım" diye işaretlenir
- Bir hipotez ölçümden **önce** yazılır
- Hız iddiaları **uçtan uca** ölçülür, çekirdek mikro-benchmark'ıyla değil —
  maliyet modeli tam olarak bu yüzden dokuz kez yanıldı
- Bir optimizasyonun kabul kriteri **çıktının değişmemesi**; değişiyorsa
  değişimin ne olduğu ölçülür ve karar tablosuna yazılır
- Aleyhe bulgular da kaydedilir (eşleştirme kazancı 1.16×, ayrılabilirlik
  önyargısı, maliyet modelinin **dokuz** hatası, Triton tahminimin 2–3 kat
  iyimser çıkması, Kronecker'ın sentetik ölçümümün iki mertebe yanılması)
- **Denenip elenen fikirler kaydedilir** (§7) — kaydetmemek ikinci kez denemek

Bu belge de aynı disiplinin parçası: ne bilindiğini, ne bilinmediğini ve neyin
denenip bırakıldığını ayrı tutuyor.

---

## 14. Ölçüm hijyeni — bu oturumun asıl çıktısı

Hız kazançları geçici; bunlar değil. 08-24'te **kendi ölçümlerimin dördü**
sessizce yanlıştı ve ikisini önce yanlış teşhis ettim. Dördünün de ortak yanı
şu: **yanlış cevap vermiyorlar, inandırıcı bir cevap veriyorlar.**

### 14.1 Dört tuzak, dördü de gerçekten vurdu

| tuzak | belirtisi | nasıl yakalandı |
|---|---|---|
| **`_on_device` cihaz-dizesi anahtarı** | Optimizasyon **1.00×** okuyor | Kaba kuvvete düştüğünü fark edince. Önce GPU çekişmesine, sonra saat düşüşüne yordum — ikisi de yanlıştı |
| **Mutlak süreleri koşular arası kıyaslamak** | Değişiklik **+%37** okuyor | Tek süreçte dönüşümlü A/B: −%0.8. Makineymiş |
| **Yanlış rejimde ölçülmüş ret** | "Kazanç yok, tekrar deneme" | Ölçümü gerçek genişliklerde tekrarlayınca: **9.94×**. Kayıt beni sekiz gün yanlış yerde tuttu |
| **Cevabı sınayan test** | Test yeşil, hata duruyor | Yolu saymaya geçince. Aynı oturumda **üç kez** oldu |
| **Hiç koşulmamış bir kompozisyon** | Her parça yeşil, birleşimleri çalışmıyor | 08-25: `sequential_calibrate` + `run_config` **beş** kusur taşıyordu ve parça testlerinin hepsi geçiyordu (§6.12) |
| **Değiştirdiğin yolu geçmeyen ölçüm** | Kalite farkı **tam sıfır** — inandırıcı ve boş | 08-25: yönlendirme değişiminin bedelini 512 satırlık bir katmanda ölçtüm; o katman ölü bandın altında kalıyor, yani iki kolda da **sıfır tarama** oldu ve %0.0000 hiçbir şey kanıtlamadı (§6.13) |
| **Ölçülen bir oranı ölçülmemiş bir rejime taşımak** | Model `k=1024`'te **5.4× kötümser** | 08-25: kron/yoğun oranını iki genişlikten flop sayısıyla dışarı taşıdım. 1024 tam ikinin kuvveti, `m=1`, kasma yoğun çarpıma çöküyor — ve §6.8 bunu **zaten ölçmüştü**. Izgaranın on dört genişliğini tek tek ölçünce çıktı (§6.18) |

### 14.2 Kurallar

- **Kıyaslama yazarken `assert quantize.is_canonical_codebook(cb)`.** Hızlı
  yolun açık olduğunu iddia et, varsayma. Bu yüzden dışa açıldı.
- **Hız fazından önce GPU'nun boş olduğunu doğrula — ama `utilization.gpu` ile
  DEĞİL.** Çekişme yalnız gürültü eklemiyor; darboğazı karta taşıyıp *tam da
  silmeye çalıştığın gecikmeyi* gizliyor, ve sonuç 1.00× diye okunuyor. Ama bu
  kural 08-25'e kadar **yanlış göstergeyi** işaret ediyordu ve bir kez yanlış
  karar verdirdi: bu makinede `utilization.gpu` boş kartta **%42**, yabancı
  yükte %99, kendi yükümüzde %25 okuyor. `torch.cuda.mem_get_info` ise **kör** —
  3 GiB tutan yabancı bir süreç onu **bir bayt** oynatmıyor (WDDM çağıran
  bağlamın bütçesini veriyor). Çalışan tek gösterge **`clocks.sm`**: boşta
  %42, yabancı yükte %89. Artık elle bakılmıyor —
  `experiments/bench_guard.require_quiet_gpu()` **fırlatıyor**.
- **Yalnız tek süreçte dönüşümlü A/B.** Bu makinede aynı ölçüm koşudan koşuya
  %14–37 oynuyor.
- **Bir reddi kaydederken hangi rejimde ölçüldüğünü yaz.** Yanlış rejimde
  ölçülmüş bir ret hiç ölçmemekten kötüdür: bir sonrakinin bakmasını durdurur.
- **Test cevabı değil YOLU izlesin.** İki doğru algoritma zaten aynı cevabı
  verir — bir hatanın uzun yaşamasının sebebi tam olarak budur.
- **Yeni testi mutasyonla doğrula.** Eski koda karşı kırmızı olduğu
  gösterilmeden kabul etme. Bu oturumda ilk yazdığım testler üç kez gerçek
  hataları kaçırdı: tek çekiliş kullandıkları için, CPU'da `.to("cpu")` no-op
  olduğu için, ve cevabı sınadıkları için.
- **Parçayı değil ÇAĞIRANI test et.** Bir düzeltmenin dokunduğu fonksiyona test
  yazmak yetmiyor: `8c56f1e` `collect_block_statistics`'e CUDA testi ekledi ve
  onu çağıran `sequential_calibrate` üç kusurla kaldı (§6.12). Kusurun yaşadığı
  yer parça değil, **kompozisyon**.
- **Koşulmamış bir kompozisyon çalışmıyor sayılır.** Bu projede iki kez böyle
  oldu: maliyet modelinin iki eksik terimi de, dikişin beş kusuru da, "her parça
  yeşil ama zinciri kimse koşmadı" durumundan çıktı.
- **Bir kaldıracı iki kez ölç: terimi tek başına, ve terimi yerinde.** Tek
  başına ölçüm kaldıracın *var olduğunu* söyler, yerinde ölçüm *işe yaradığını*.
  fp16'da ikisi ayrıldı (1.16× / 1.00×) ve reddi getiren fark oydu; kron ile
  telafide birleşti (%95–107) ve kabulü getiren de o. Yalnız birine bakmak, iki
  zıt sonucu ayırt edememek demek (§6.18).
- **Bir oranı, oranın kaynağı değişen bir yere taşıma.** Kron'un hızı k'nin
  *çarpanlarına* bağlı ve sıçramalı: 3008 = 64×47 → 5.14×, %2 daha geniş olan
  3072 = 1024×3 → 1.80×. Tek sayı da, en-yakın-k araması da yanlış cevap verir.
  Izgaranın kullandığı her genişlik ölçüldü ve tablo **tam-k** ile okunuyor;
  ölçülmemiş genişlik yoğun fiyatlanıp `rotate_kron_priced=False` diye
  **söyleniyor**.
- **Modelin varsayılanı, kodun koştuğu ARİTMETİK olmalı — sadece parametreleri
  değil.** §6.3'ün on hatası eksik terimlerdi; on birincisi farklı: terim
  listedeydi, ama `rotation_seconds` hattın 08-25'ten beri koşmadığı yoğun
  formu fiyatlıyordu. **Kötümser** olduğu için hiçbir şey şikâyet etmedi, ve
  modelin gerçek bloktaki 5.2× "iyimserliği"nin çoğunu o gizliyordu (§6.18).
- **Bir oranı sabit bir paydaya değil, ölçülen büyüklüğe demirle.** Bir test
  "ölçek fiti M1'in %20'sinden az" diyordu ve kaldıraçlar fiyatlanınca kırmızıya
  döndü — fit bir saniye bile pahalanmadan, çünkü payda 15.0'tan 11.7 güne indi.
  Hareketli paydaya oranlanan bir iddia, ölçtüğü şey hakkında değil geri kalan
  her şey hakkındadır.
- **Yalnız test değil, ÖLÇÜM de yolu izlesin.** "Test cevabı değil yolu
  izlesin" kuralının ölçüm hâli, ve 08-25'te bunu unuttum: bir değişikliğin
  kalite bedelini, değişikliğin hiç tetiklenmediği bir şekilde ölçtüm ve tertemiz
  bir %0.0000 aldım. Bir A/B'de **önce iki kolun gerçekten farklı yollardan
  geçtiğini say**, sonra sayılara bak. Burada sayılacak şey `_brute_force`'a
  düşen satırdı; sıfır/sıfırdı.

### 14.3 Ve modele dair olan

Maliyet modeli dokuz kez yanıldı; **yedisi modelin bilmediği şeydi**, oran hatası
değil. Yani
bu modelde sorulacak soru "oran doğru mu" değil, **"listede ne yok"**. Son ikisi
(kalibrasyon, ileri telafi) hiçbir şey patlamadığı için değil, *ne eksik* diye
arandığı için bulundu — ve ikisi birden M1'i 12 günden ~40'a çıkardı.

Neyin saklanmalarına izin verdiği de kayda değer: **tam sürücüyü kimse
koşmadı.** §8.1 yalnız "gerçek veri yok" diye kritik yolda değil; **ölçülmeyen
maliyet de orada birikiyor.**

08-25'te aynı boşluktan ikinci bir şey çıktı, ve bu sefer maliyet değil
**çalışmayan kod**: sürücünün kullanacağı dikiş GPU'da hiç koşmamıştı ve tek bir
yolda beş kusur biriktirmişti (§6.12). Yani "kimse koşmadı" bu projede iki ayrı
şey üretti — **eksik terim** ve **kırık zincir** — ve ikisi de aynı yerde
duruyordu. Kural buradan çıkıyor: bir kompozisyonun test edilmemiş olması, onun
çalıştığına dair kanıtın *yokluğu* değil, **çalışmadığına dair beklenti**.
````




# Instruction
# How to review this repository

You are reviewing **subfloor**, a research codebase for LLM compression: joint
sparsity + E8P lattice vector quantization below 2 bits per weight, targeted at
Llama-2-7B on an 8 GB laptop GPU.

## What the project is

The pipeline compresses one linear layer at a time, in a fixed order that
`prune()` enforces by raising:

    score -> select mask (in the UNROTATED basis) -> freeze
          -> compact -> rotate -> LDLQ -> compensate

The core claim is an accounting identity: with `W` bits per survivor, density
`d` and tile size `T`, in the bitmap regime `d(T) - d(1) = (1 - 1/T) / W`, which
is constant in the budget. Two gates decide the thesis: Gate A asks whether the
best sparse configuration beats dense 2-bit PTQ; Gate B asks whether the optimal
`T` is interior rather than at an edge.

Read `README.md` first, then `docs/STATUS.md` (Turkish; it is the handover
document and carries the *why* behind every decision).

## The project's own working rules — judge by these, not generic advice

- **A test must watch the path, not the answer.** Tests here check a *claim*
  (where an identity is valid, how the codebook is constructed, that rotation
  preserves the mask), not just an output value.
- **A new test is not accepted until it has been shown red against the old code.**
- **Every timing constant is machine-specific and says so.** Numbers were
  measured on one RTX 5060 laptop; they are not portable and the code says this.
- **Comments and docstrings carry the reasoning.** Density of documentation is
  deliberate, not clutter.
- Test code is 1.4x production code on purpose: most functions carry a
  mathematical claim, and the most expensive failure mode is silently producing
  a wrong number.

Suggestions that ignore a stated reason, or that are generic best-practice
churn, are not useful here.

## Already known — do NOT report these as findings

These are documented by the author in `docs/STATUS.md` sections 7, 8.6, 9 and 10
and in the README's "Gaps". Reporting them back is noise. Only mention one if
you have a genuinely new and specific angle on it.

- The compressed model's perplexity has never been measured; Gate A and Gate B
  have no real data. Every quality number outside the dense baseline is a
  layer-level proxy or synthetic.
- Whether E8P holds its 2-bit *quality* on the compacted survivor submatrix is
  an open assumption (the cost side is verified). This is the project's single
  largest risk, and the cheap experiment testing it was deliberately skipped.
- LDLQ for Axis A raises `NotImplementedError`. Blockwise (SparseGPT-style) mask
  selection is not done. The section 3.6 ablations are not done. The attention
  coordination formula (v_proj/o_proj, GQA, RoPE) is only implied.
- Evaluation cost for C4 and the five zero-shot tasks is unmeasured. The
  preregistration is not frozen. The `tau` sweep script is not written.
- The driver's ~2.7x unexplained context cost (memory ceiling, 5.4 GiB peak).
  `ROT_TILE_TIMINGS` for B=1.60 and B=1.75 were not swept. `TILE_TIMINGS` was
  measured warm.
- Ideas already tried and rejected with measurements: scale sampling, step
  reduction, per-layer scale, block-diagonal RHT, TF32 (breaks Cholesky), fp16
  search, writing custom kernels, AQLM survivors, narrowing the grid, and
  aggregating the scale fit across tiles (not bit-exact).
- Environment traps already recorded: HF download stalls, HF auth, torchvision
  ABI, the `wikitext` namespace, `nvidia-smi utilization.gpu` being inversely
  correlated on this machine (use `clocks.sm`), `sequential_calibrate` defaulting
  to float64 on GPU (the driver must pass float32), CPU tests being unable to
  test device defaults, and `python -u` for background runs.

## What is actually useful to report

Rank these highest:

1. **Correctness bugs** — an input that produces a silently wrong number. Cite
   `file:line`, state the concrete failing input, and say what the wrong output
   is. Do not report a bug you cannot trace through the code.
2. **Robustness gaps** — dtype/device mismatches accepted silently, non-atomic
   file writes, defaults that only work when overridden, exceptions that turn a
   wrong result into an apparent success.
3. **Method and statistics** — Gate A/Gate B design, multiple comparisons,
   whether the bit accounting charges everything a real deployment pays, the
   evaluation protocol, missing baselines, reproducibility of results.
4. **Test gaps** — production branches with no test; tests that would pass
   against broken code; tests that never execute off the author's machine.
5. **Engineering** — what would most help a second contributor.

## How to be trustworthy here

State your confidence. If you are reasoning from reading rather than execution,
say so — you cannot run this code, so a claim like "this raises" or "this
overflows" is a hypothesis until someone runs it. Quote the code you are relying
on. Prefer a few sharp, verifiable findings over a long list. If the README or
`docs/STATUS.md` already states something, summarizing it back is not a finding.
