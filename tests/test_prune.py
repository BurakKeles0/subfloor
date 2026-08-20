"""Mask selection, the H1 invariant, and forward-only compensation.

`test_t1_axis_b_reproduces_standard_wanda` is Spec v6 section 5.2's equivalence
check.  The spec asks for agreement within 0.05 ppl; at the mask level we can
demand exact equality, which is a stronger statement.
"""

from __future__ import annotations

import pytest
import torch

import prune as P
import scoring as S
import tiling as T

torch.manual_seed(0)

N_OUT, N_IN, N_SAMPLES = 24, 32, 256
DT = torch.float64


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(0)          # see the note in test_scoring.py
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    mixing = torch.randn((N_IN, N_IN), dtype=DT) / (N_IN ** 0.5)
    X = torch.randn((N_SAMPLES, N_IN), dtype=DT) @ mixing
    X[:, 5] *= 10.0
    return W, X, X.norm(dim=0), X.T @ X


# --------------------------------------------------------------------------- #
# The invariant  (plan section H1)
# --------------------------------------------------------------------------- #

def test_refuses_to_prune_a_rotated_matrix(fixture):
    """The whole approach rests on choosing the mask before any rotation.
    Make it an error, not a convention someone has to remember."""
    W, _, act_norm, _ = fixture
    with pytest.raises(ValueError, match="rotated"):
        P.prune(W, axis="B", tile_size=4, density=0.5,
                act_norm=act_norm, already_rotated=True)


def test_blockwise_selection_is_deferred_to_m3(fixture):
    W, _, act_norm, _ = fixture
    with pytest.raises(NotImplementedError, match="M3"):
        P.prune(W, axis="B", tile_size=4, density=0.5,
                act_norm=act_norm, select="blockwise")


# --------------------------------------------------------------------------- #
# Wanda equivalence  (Spec v6 section 5.2)
# --------------------------------------------------------------------------- #

def test_t1_axis_b_reproduces_standard_wanda(fixture):
    """T=1 on Axis B makes every row its own comparison group -- which is
    exactly what standard Wanda does."""
    W, _, act_norm, _ = fixture
    density = 0.5
    k = T.uniform_survivor_count(N_IN, density)

    want = torch.zeros((N_OUT, N_IN), dtype=torch.bool)
    reference = W.abs() * act_norm.unsqueeze(0)          # original Wanda metric
    want.scatter_(1, reference.topk(k, dim=1).indices, True)

    got = P.prune(W, axis="B", tile_size=1, density=density,
                  metric="wanda", act_norm=act_norm, compensate=False)
    assert torch.equal(got.mask.expand(), want)
    assert torch.equal(got.W, W * want.to(DT))


def test_squaring_does_not_change_per_row_ranking(fixture):
    """Why the equivalence holds even though we square: squaring is monotone,
    so it cannot reorder weights WITHIN one comparison group.  It only changes
    how groups pool (tested in test_scoring)."""
    W, _, act_norm, _ = fixture
    l1 = P.prune(W, axis="B", tile_size=1, density=0.5,
                 metric="wanda_l1", act_norm=act_norm)
    l2 = P.prune(W, axis="B", tile_size=1, density=0.5,
                 metric="wanda", act_norm=act_norm)
    assert torch.equal(l1.mask.expand(), l2.mask.expand())


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("axis, tile_size", [
    ("B", 1), ("B", 4), ("B", 8), ("B", T.MAX_TILE),
    ("A", 1), ("A", 4), ("A", 16), ("A", T.MAX_TILE),
])
@pytest.mark.parametrize("density", [0.25, 0.5, 0.75])
def test_density_is_exact_and_mask_is_applied(fixture, axis, tile_size, density):
    W, _, act_norm, _ = fixture
    r = P.prune(W, axis=axis, tile_size=tile_size, density=density,
                act_norm=act_norm)
    n_idx = N_IN if axis == "B" else N_OUT
    k = T.uniform_survivor_count(n_idx, density)
    assert r.density() == pytest.approx(k / n_idx, abs=1e-12)
    assert bool(((r.W == 0) | r.mask.expand()).all())


def test_tmax_prunes_whole_input_channels(fixture):
    """Axis B at T=max is input-channel pruning: a dropped column is dropped
    for every row (Spec v6 section 4.1)."""
    W, _, act_norm, _ = fixture
    r = P.prune(W, axis="B", tile_size=T.MAX_TILE, density=0.5, act_norm=act_norm)
    dead = (r.W == 0).all(dim=0)
    assert int(dead.sum()) == N_IN - T.uniform_survivor_count(N_IN, 0.5)


# --------------------------------------------------------------------------- #
# Compensation
# --------------------------------------------------------------------------- #

def test_compensation_requires_a_hessian(fixture):
    W, _, act_norm, _ = fixture
    with pytest.raises(ValueError, match="requires the Hessian"):
        P.prune(W, axis="B", tile_size=4, density=0.5,
                act_norm=act_norm, compensate=True)


@pytest.mark.parametrize("axis", ["A", "B"])
def test_compensation_keeps_pruned_positions_exactly_zero(fixture, axis):
    W, _, act_norm, H = fixture
    r = P.prune(W, axis=axis, tile_size=4, density=0.5,
                act_norm=act_norm, H=H, compensate=True)
    assert r.compensated
    assert torch.equal(r.W == 0, ~r.mask.expand())


@pytest.mark.parametrize("axis", ["A", "B"])
def test_compensation_reduces_layer_output_error(fixture, axis):
    """The objective that actually matters: ||X W^T - X W_hat^T||^2."""
    W, X, act_norm, H = fixture
    kw = dict(axis=axis, tile_size=4, density=0.5, act_norm=act_norm)
    plain = P.prune(W, **kw)
    comp = P.prune(W, **kw, H=H, compensate=True)
    assert torch.equal(plain.mask.expand(), comp.mask.expand()), "same mask"

    ref = X @ W.T
    err_plain = float((X @ plain.W.T - ref).square().sum())
    err_comp = float((X @ comp.W.T - ref).square().sum())
    assert err_comp < err_plain


def test_compensation_only_pushes_forward():
    """Spec v6 section 4.6 / trap 16.  Dropping the LAST column must leave every
    earlier column untouched -- there is nothing to its right to absorb it.
    Any backward leak would show up here.
    """
    n_out, n_in = 6, 10
    W = torch.randn((n_out, n_in), dtype=DT)
    X = torch.randn((256, n_in), dtype=DT) @ (
        torch.randn((n_in, n_in), dtype=DT) / (n_in ** 0.5)
    )
    Hinv = S.damped_hessian_inverse(X.T @ X, percdamp=0.01)

    keep = torch.ones((n_out, n_in), dtype=torch.bool)
    keep[:, -1] = False
    out = P.forward_compensate(W, keep, Hinv)

    assert torch.allclose(out[:, :-1], W[:, :-1], atol=1e-12)
    assert torch.equal(out[:, -1], torch.zeros(n_out, dtype=DT))


def test_forward_compensate_validates_shapes(fixture):
    W, _, _, H = fixture
    Hinv = S.damped_hessian_inverse(H)
    with pytest.raises(ValueError, match="does not match"):
        P.forward_compensate(W, torch.ones((3, 3), dtype=torch.bool), Hinv)
    with pytest.raises(ValueError, match="Hinv must be"):
        P.forward_compensate(
            W, torch.ones_like(W, dtype=torch.bool), torch.eye(5, dtype=DT)
        )


# --------------------------------------------------------------------------- #
# End to end with the rest of the pipeline
# --------------------------------------------------------------------------- #

def test_prune_then_compact_then_rotate(fixture):
    """The H1 order, start to finish: prune in the untouched basis, freeze,
    compact, rotate -- and the support is still exactly what pruning chose."""
    import compact as C
    import rotation as R

    W, _, act_norm, H = fixture
    r = P.prune(W, axis="B", tile_size=4, density=0.5,
                act_norm=act_norm, H=H, compensate=True)
    cw = C.compact(r.W, r.mask)
    rotated, _ = R.rotate(cw, axis="index", seed=7)

    assert torch.equal(C.scatter(rotated) != 0, r.mask.expand())
    assert C.scatter(rotated).shape == W.shape
