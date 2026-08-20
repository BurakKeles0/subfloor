"""Tiling, compaction and mask-preserving rotation.

The centrepiece is `test_rotation_preserves_the_frozen_mask` and
`test_line_rotation_leaves_group_obs_saliency_invariant`: together they are the
in-code proof of plan section H1, the invariant the whole approach rests on.
"""

from __future__ import annotations

import math

import pytest
import torch

import compact as C
import rotation as R
import tiling as T

N_OUT, N_IN = 32, 64
DT = torch.float64
TILES_B = [1, 2, 4, 8, 16, 32, T.MAX_TILE]     # must divide N_OUT
TILES_A = [1, 2, 4, 8, 16, 32, 64, T.MAX_TILE]  # must divide N_IN


@pytest.fixture(autouse=True)
def _seed():
    """Reseed before every test so results never depend on execution order."""
    torch.manual_seed(0)


def _mask(axis: str, tile_size, density: float = 0.5) -> T.TileMask:
    n_lines = N_OUT if axis == "B" else N_IN
    n_idx = N_IN if axis == "B" else N_OUT
    n_t = T.n_tiles_for(n_lines, tile_size)
    score = torch.rand((n_t, n_idx), dtype=DT)
    return T.make_topk_mask(score, axis, tile_size, density, N_OUT, N_IN)


# --------------------------------------------------------------------------- #
# Tiling
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tile_size", TILES_B)
def test_tile_geometry_axis_b(tile_size):
    m = _mask("B", tile_size)
    assert m.n_lines == N_OUT and m.n_idx == N_IN
    assert m.n_tiles * m.lines_per_tile == N_OUT
    assert m.expand().shape == (N_OUT, N_IN)
    assert m.is_uniform()


@pytest.mark.parametrize("tile_size", TILES_A)
def test_tile_geometry_axis_a(tile_size):
    """Axis A is the transpose: n_idx = n_out (Spec v6 section 3.2)."""
    m = _mask("A", tile_size)
    assert m.n_lines == N_IN and m.n_idx == N_OUT
    assert m.expand().shape == (N_OUT, N_IN)


def test_t1_is_unstructured_and_tmax_is_structured():
    """The two ends of the family (Spec v6 section 4.1)."""
    fine = _mask("B", 1)
    assert fine.n_tiles == N_OUT                       # every row picks its own
    dense_fine = fine.expand()
    assert not bool((dense_fine[0] == dense_fine[1]).all()), "rows should differ"

    coarse = _mask("B", T.MAX_TILE)
    assert coarse.n_tiles == 1
    dense_coarse = coarse.expand()
    assert bool((dense_coarse == dense_coarse[0]).all()), "one column set for all"


@pytest.mark.parametrize("density", [0.25, 0.5, 0.75])
def test_realized_density_is_reported_not_requested(density):
    """Accounting must be given k/n_idx, never the requested density."""
    m = _mask("B", 4, density)
    k = T.uniform_survivor_count(N_IN, density)
    assert m.density() == pytest.approx(k / N_IN, abs=1e-12)
    assert bool((m.survivors_per_tile() == k).all())


def test_tiling_rejects_ragged_and_bad_input():
    with pytest.raises(ValueError, match="does not divide"):
        T.n_tiles_for(30, 4)
    with pytest.raises(ValueError, match="positive int"):
        T.n_tiles_for(32, 0)
    with pytest.raises(ValueError, match="density"):
        T.uniform_survivor_count(64, 0.0)
    m = _mask("B", 4)
    with pytest.raises(ValueError, match="axis"):
        T.TileMask("C", 4, N_OUT, N_IN, m.support, m.assignment)
    with pytest.raises(ValueError, match="shape"):
        T.TileMask("B", 8, N_OUT, N_IN, m.support, m.assignment)


# --------------------------------------------------------------------------- #
# Compaction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis, tiles", [("B", TILES_B), ("A", TILES_A)])
def test_compact_scatter_roundtrip(axis, tiles):
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    for tile_size in tiles:
        m = _mask(axis, tile_size)
        cw = C.compact(W, m)
        assert torch.equal(C.scatter(cw), m.apply(W))


@pytest.mark.parametrize("tile_size", [1, 4, 16, T.MAX_TILE])
def test_compact_block_shape(tile_size):
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    m = _mask("B", tile_size, 0.5)
    cw = C.compact(W, m)
    assert cw.blocks.shape == (m.n_tiles, m.lines_per_tile, int(m.survivors_per_tile()[0]))
    assert cw.blocks.numel() == m.expand().sum()


def test_compact_gathers_the_right_values():
    W = torch.arange(N_OUT * N_IN, dtype=DT).reshape(N_OUT, N_IN)
    m = _mask("B", 8)
    cw = C.compact(W, m)
    for t in range(m.n_tiles):
        rows = cw.line_index[t]
        cols = cw.idx_index[t]
        assert torch.equal(cw.blocks[t], W[rows][:, cols])


def test_compact_requires_uniform_tiles():
    m = _mask("B", 4)
    support = m.support.clone()
    support[0, support[0].nonzero()[0]] = False           # make tile 0 shorter
    ragged = T.TileMask("B", 4, N_OUT, N_IN, support, m.assignment)
    with pytest.raises(ValueError, match="uniform"):
        C.compact(torch.randn((N_OUT, N_IN), dtype=DT), ragged)


# --------------------------------------------------------------------------- #
# Rotation: orthogonality
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 3, 6, 12, 24, 48, 96])
def test_structured_orthogonal_is_orthogonal(n):
    """Arbitrary n via kron(RHT(2^a), orthogonal(m)) -- QuIP#'s construction."""
    Q = R.structured_orthogonal(n, seed=1, dtype=DT)
    assert Q.shape == (n, n)
    assert torch.allclose(Q @ Q.T, torch.eye(n, dtype=DT), atol=1e-12)


def test_randomized_hadamard_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        R.randomized_hadamard(12)
    assert R.is_power_of_two(64) and not R.is_power_of_two(48)


@pytest.mark.parametrize("axis", ["line", "index"])
def test_rotate_unrotate_is_identity(axis):
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    m = _mask("B", 8)
    cw = C.compact(W, m)
    rot, Q = R.rotate(cw, axis=axis, seed=3)
    back = R.unrotate(rot, Q, axis=axis)
    assert torch.allclose(back.blocks, cw.blocks, atol=1e-12)


def test_rotation_is_not_a_no_op():
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    cw = C.compact(W, _mask("B", 8))
    rot, _ = R.rotate(cw, axis="index", seed=3)
    assert not torch.allclose(rot.blocks, cw.blocks)


# --------------------------------------------------------------------------- #
# THE INVARIANT  (plan section H1)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis", ["line", "index"])
@pytest.mark.parametrize("tile_size", [2, 4, 8, 16, T.MAX_TILE])
def test_rotation_preserves_the_frozen_mask(axis, tile_size):
    """A rotation applied AFTER compaction cannot move a survivor out of its
    tile's index set -- the block spans exactly that set.

    This is what separates the proposed order from QuaRot+Wanda (rotate first,
    then prune), which collapses to 5868 ppl on Llama-2-7B.
    """
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    m = _mask("B", tile_size)
    cw = C.compact(W, m)
    rot, _ = R.rotate(cw, axis=axis, seed=5)

    support_before = m.expand()
    support_after = C.scatter(rot) != 0
    assert torch.equal(support_after, support_before)
    assert float(m.density()) == pytest.approx(
        float(support_after.to(DT).mean()), abs=1e-12
    )


def _group_obs_saliency(block: torch.Tensor, Hinv_SS: torch.Tensor) -> torch.Tensor:
    """Reference implementation of Spec v6 section 4.3's group-OBS error,
    summed over the lines of one tile:  tr(W_S [(H^-1)_SS]^-1 W_S^T).

    Note (H^-1)_SS inverted, NOT (H_SS)^-1 -- Spec v6 section 7, trap 16.
    """
    M = torch.linalg.inv(Hinv_SS)
    return torch.einsum("ij,jk,ik->", block, M, block)


def test_line_rotation_leaves_group_obs_saliency_invariant():
    """Spec v6 section 7.19 claims a tile-local rotation does not change eps_S.
    It is right, and the reason is that the tile's error is a trace:

        sum_i w_i,S M w_i,S^T = tr(W_S M W_S^T) = tr(Q W_S M W_S^T Q^T)

    The spec then files this under "ineffective".  That reading is correct for
    MASK SELECTION and wrong for quantization: an invariant mask is exactly the
    property that makes the rotation legal.
    """
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    m = _mask("B", 8)
    cw = C.compact(W, m)

    A = torch.randn((N_IN, N_IN), dtype=DT)
    Hinv = A @ A.T + N_IN * torch.eye(N_IN, dtype=DT)      # SPD

    rot_line, _ = R.rotate(cw, axis="line", seed=11)
    rot_index, _ = R.rotate(cw, axis="index", seed=11)

    for t in range(cw.n_tiles):
        S = cw.idx_index[t]
        Hinv_SS = Hinv[S][:, S]
        base = _group_obs_saliency(cw.blocks[t], Hinv_SS)
        line = _group_obs_saliency(rot_line.blocks[t], Hinv_SS)
        index = _group_obs_saliency(rot_index.blocks[t], Hinv_SS)

        assert line == pytest.approx(float(base), rel=1e-10), "line axis must be invariant"
        assert index != pytest.approx(float(base), rel=1e-6), "index axis must not be"


# --------------------------------------------------------------------------- #
# Inference cost model
# --------------------------------------------------------------------------- #

def test_overhead_ratios():
    """log2(k)/T for the index axis, log2(T)/k for the line axis: expensive and
    strong against cheap and weak."""
    d, n_idx = 0.27, 11008
    k = d * n_idx
    assert R.index_axis_overhead_ratio(1, d, n_idx) == pytest.approx(math.log2(k))
    assert R.index_axis_overhead_ratio(1, d, n_idx) > 10.0
    assert R.index_axis_overhead_ratio(16, d, n_idx) == pytest.approx(math.log2(k) / 16)
    assert R.index_axis_overhead_ratio(T.MAX_TILE, d, n_idx) == 0.0

    assert R.line_axis_overhead_ratio(16, d, n_idx) == pytest.approx(4.0 / k)
    assert R.line_axis_overhead_ratio(16, d, n_idx) < 2e-3      # 4/2972 = 0.00135
    # At T=16 the line axis is ~536x cheaper than the index axis.
    assert (R.line_axis_overhead_ratio(16, d, n_idx)
            < R.index_axis_overhead_ratio(16, d, n_idx) / 100)


def test_index_overhead_falls_as_tile_grows():
    """The force that pushes T up, and is absent from Delta = Q + tau."""
    d, n_idx = 0.5, 11008
    ratios = [R.index_axis_overhead_ratio(t, d, n_idx) for t in (1, 2, 4, 8, 16, 32)]
    assert all(a > b for a, b in zip(ratios, ratios[1:]))
