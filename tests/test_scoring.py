"""Saliency metrics, aggregation, and OBS compensation.

`test_matches_spec_formulas_literally` transcribes Spec v6 section 4.3's four
formulas as written -- with explicit loops -- and checks the unified
implementation reproduces them.  That is what licenses the refactor described in
scoring.py's docstring: the axes really are one metric summed two ways.
"""

from __future__ import annotations

import pytest
import torch

import scoring as S
import tiling as T

torch.manual_seed(0)

N_OUT, N_IN, N_SAMPLES = 24, 32, 128
DT = torch.float64


@pytest.fixture(scope="module")
def fixture():
    # Seed HERE, not at import: fixtures build lazily, so a module-level seed
    # would make these values depend on which tests ran first.
    torch.manual_seed(0)
    W = torch.randn((N_OUT, N_IN), dtype=DT)
    # Correlated input channels, as real activations are.  With iid channels
    # H is nearly isotropic and OBS compensation has nothing to exploit --
    # the correlation is the whole reason second-order pruning beats masking.
    mixing = torch.randn((N_IN, N_IN), dtype=DT) / (N_IN ** 0.5)
    X = torch.randn((N_SAMPLES, N_IN), dtype=DT) @ mixing
    X[:, 3] *= 12.0                                   # a fat outlier channel
    act_norm = X.norm(dim=0)
    H = X.T @ X
    Hinv = S.damped_hessian_inverse(H, percdamp=0.01)
    return W, X, act_norm, H, Hinv


# --------------------------------------------------------------------------- #
# The spec's formulas, transcribed literally
# --------------------------------------------------------------------------- #

def test_matches_spec_formulas_literally(fixture):
    W, _, act_norm, _, Hinv = fixture
    tile_size = 4
    hinv_diag = torch.diagonal(Hinv)

    # eps_B(j,t) = ||X_j||^2 * sum_{i in R_t} w_ij^2
    asg_b = T.contiguous_assignment(N_OUT, tile_size)
    want_b = torch.zeros((N_OUT // tile_size, N_IN), dtype=DT)
    for t in range(N_OUT // tile_size):
        rows = (asg_b == t).nonzero().squeeze(1)
        for j in range(N_IN):
            want_b[t, j] = act_norm[j] ** 2 * (W[rows, j] ** 2).sum()
    got_b = S.tile_scores(W, "B", tile_size, "wanda", act_norm=act_norm)
    assert torch.allclose(got_b, want_b, rtol=1e-12)

    # eps_A(i,t) = sum_{j in C_t} (|w_ij| * ||X_j||)^2
    asg_a = T.contiguous_assignment(N_IN, tile_size)
    want_a = torch.zeros((N_IN // tile_size, N_OUT), dtype=DT)
    for t in range(N_IN // tile_size):
        cols = (asg_a == t).nonzero().squeeze(1)
        for i in range(N_OUT):
            want_a[t, i] = ((W[i, cols].abs() * act_norm[cols]) ** 2).sum()
    got_a = S.tile_scores(W, "A", tile_size, "wanda", act_norm=act_norm)
    assert torch.allclose(got_a, want_a, rtol=1e-12)

    # eps_B(j,t) = sum_{i in R_t} w_ij^2 / [H^-1]_jj
    want_obs = torch.zeros((N_OUT // tile_size, N_IN), dtype=DT)
    for t in range(N_OUT // tile_size):
        rows = (asg_b == t).nonzero().squeeze(1)
        for j in range(N_IN):
            want_obs[t, j] = (W[rows, j] ** 2).sum() / hinv_diag[j]
    got_obs = S.tile_scores(W, "B", tile_size, "obs_diag", hinv_diag=hinv_diag)
    assert torch.allclose(got_obs, want_obs, rtol=1e-12)


def test_the_two_axes_share_one_per_weight_metric(fixture):
    """Plan section D2: the axis comparison must not smuggle in a fidelity
    difference.  Here it structurally cannot -- both axes reduce the SAME
    matrix, transposed."""
    W, _, act_norm, _, _ = fixture
    s = S.per_weight_saliency(W, "wanda", act_norm=act_norm)
    a = S.aggregate_to_tiles(s, "A", 4)
    b = S.aggregate_to_tiles(s, "B", 4)
    assert a.shape == (N_IN // 4, N_OUT)
    assert b.shape == (N_OUT // 4, N_IN)
    # Total saliency is conserved whichever way it is summed.
    assert a.sum() == pytest.approx(float(s.sum()), rel=1e-12)
    assert b.sum() == pytest.approx(float(s.sum()), rel=1e-12)


# --------------------------------------------------------------------------- #
# Aggregation edges
# --------------------------------------------------------------------------- #

def test_t1_aggregation_is_the_identity(fixture):
    """T=1 on Axis B: every row is its own tile, so no pooling happens --
    which is exactly what makes T=1 unstructured."""
    W, _, act_norm, _, _ = fixture
    s = S.per_weight_saliency(W, "wanda", act_norm=act_norm)
    assert torch.equal(S.aggregate_to_tiles(s, "B", 1), s)


def test_tmax_aggregation_is_a_full_sum(fixture):
    W, _, act_norm, _, _ = fixture
    s = S.per_weight_saliency(W, "wanda", act_norm=act_norm)
    got = S.aggregate_to_tiles(s, "B", T.MAX_TILE)
    assert got.shape == (1, N_IN)
    assert torch.allclose(got[0], s.sum(dim=0), rtol=1e-12)


def test_scores_feed_make_topk_mask(fixture):
    W, _, act_norm, _, _ = fixture
    score = S.tile_scores(W, "B", 8, "wanda", act_norm=act_norm)
    m = T.make_topk_mask(score, "B", 8, 0.5, N_OUT, N_IN)
    assert m.density() == pytest.approx(0.5, abs=1e-12)
    # The outlier channel survives in every tile.
    assert bool(m.support[:, 3].all())


def test_wanda_l1_can_rank_groups_differently(fixture):
    """Squaring is monotone per weight but not per GROUP -- the ablation in
    plan section E4 is not vacuous."""
    W, _, act_norm, _, _ = fixture
    l2 = S.tile_scores(W, "B", 8, "wanda", act_norm=act_norm)
    l1 = S.tile_scores(W, "B", 8, "wanda_l1", act_norm=act_norm)
    assert not torch.equal(l2.argsort(dim=1), l1.argsort(dim=1))


# --------------------------------------------------------------------------- #
# Hessian and exact group-OBS
# --------------------------------------------------------------------------- #

def test_damped_hessian_inverse(fixture):
    _, _, _, H, Hinv = fixture
    damp = 0.01 * torch.diagonal(H).mean()
    Hd = H + damp * torch.eye(N_IN, dtype=DT)
    assert torch.allclose(Hinv @ Hd, torch.eye(N_IN, dtype=DT), atol=1e-8)
    assert bool((torch.diagonal(Hinv) > 0).all())


def test_hinv_ss_is_not_h_ss_inverse(fixture):
    """Spec v6 section 7, trap 16.  Guarding it with a test because the two
    expressions look interchangeable and are not."""
    _, _, _, H, Hinv = fixture
    Sset = torch.tensor([0, 1, 5, 9])
    right = Hinv[Sset][:, Sset]
    wrong = torch.linalg.inv(H[Sset][:, Sset])
    assert not torch.allclose(right, wrong, rtol=1e-3)


def test_group_obs_error_agrees_with_the_trace_form(fixture):
    W, _, _, _, Hinv = fixture
    Sset = torch.tensor([2, 4, 7, 11, 13])
    W_S = W[:, Sset]
    M = torch.linalg.inv(Hinv[Sset][:, Sset])
    want = 0.5 * torch.trace(W_S @ M @ W_S.T)
    assert S.group_obs_error(W_S, Hinv[Sset][:, Sset]) == pytest.approx(
        float(want), rel=1e-10
    )


# --------------------------------------------------------------------------- #
# Compensation
# --------------------------------------------------------------------------- #

def test_compensation_zeroes_the_removed_set_exactly(fixture):
    """The defining property of the OBS update: W + dW is exactly zero on S."""
    W, _, _, _, Hinv = fixture
    Sset = torch.tensor([2, 4, 7, 11, 13])
    dW = S.group_obs_compensation(W[:, Sset], Hinv, Sset)
    assert torch.allclose((W + dW)[:, Sset], torch.zeros_like(W[:, Sset]), atol=1e-9)


def test_compensation_reduces_layer_output_error(fixture):
    """Functional proof: compensating beats plain masking on the real objective
    ||X W^T - X W_hat^T||^2."""
    W, X, _, _, Hinv = fixture
    Sset = torch.tensor([2, 4, 7, 11, 13])

    naive = W.clone()
    naive[:, Sset] = 0.0

    compensated = W + S.group_obs_compensation(W[:, Sset], Hinv, Sset)

    ref = X @ W.T
    err_naive = (X @ naive.T - ref).square().sum()
    err_comp = (X @ compensated.T - ref).square().sum()
    assert err_comp < err_naive
    # Measured ratio on this fixture is 0.128, i.e. compensation removes ~87%
    # of the masking error.
    assert err_comp < 0.25 * err_naive


def test_compensation_value_comes_from_channel_correlation():
    """Why this matters beyond the unit test: OBS compensation buys nothing when
    input channels are uncorrelated, because there is no redundancy to shift the
    removed weight's job onto.

    This is the mechanism behind Spec v6 section 0.5's compaction warning -- once
    survivors are gathered, a VQ's vectors no longer span adjacent, correlated
    channels, and calibration that relies on that correlation degrades.
    """
    torch.manual_seed(1)
    n_out, n_in, n_s = 16, 32, 256
    W = torch.randn((n_out, n_in), dtype=DT)
    Sset = torch.tensor([1, 3, 6, 10])

    def error_ratio(X: torch.Tensor) -> float:
        Hinv = S.damped_hessian_inverse(X.T @ X, percdamp=0.01)
        naive = W.clone()
        naive[:, Sset] = 0.0
        comp = W + S.group_obs_compensation(W[:, Sset], Hinv, Sset)
        ref = X @ W.T
        return float(
            (X @ comp.T - ref).square().sum() / (X @ naive.T - ref).square().sum()
        )

    iid = torch.randn((n_s, n_in), dtype=DT)
    mixing = torch.randn((n_in, n_in), dtype=DT) / (n_in ** 0.5)
    correlated = torch.randn((n_s, n_in), dtype=DT) @ mixing

    assert error_ratio(iid) > 0.7, "nothing to exploit when channels are iid"
    assert error_ratio(correlated) < 0.3, "correlation is what compensation uses"


# --------------------------------------------------------------------------- #
# FFN coordination (T=max)
# --------------------------------------------------------------------------- #

def test_coordination_requires_normalization():
    """Spec v6 section 7, trap 17: the three FFN terms come from different
    Hessians.  A raw sum is dominated by whichever layer has the largest scale;
    normalizing gives each an equal say."""
    n = 16
    gate = torch.rand(n, dtype=DT) + 0.5
    up = (torch.rand(n, dtype=DT) + 0.5) * 1e5           # different scale
    down = (torch.rand(n, dtype=DT) + 0.5) * 1e-3

    raw = gate + up + down
    coord = S.coordinate_ffn_saliency(gate, up, down)

    assert torch.equal(raw.argsort(), up.argsort()), "raw sum just follows up_proj"
    assert not torch.equal(coord.argsort(), up.argsort())
    assert float(coord.mean()) == pytest.approx(3.0, rel=1e-9)


def test_normalizer_rejects_degenerate_input():
    with pytest.raises(ValueError, match="degenerate"):
        S.normalizer(torch.zeros(8, dtype=DT))
    with pytest.raises(ValueError, match="unknown mode"):
        S.normalizer(torch.ones(8, dtype=DT), mode="nope")


def test_coordination_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="equal length"):
        S.coordinate_ffn_saliency(
            torch.ones(8, dtype=DT), torch.ones(8, dtype=DT), torch.ones(9, dtype=DT)
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_metric_validation(fixture):
    W, _, act_norm, _, Hinv = fixture
    with pytest.raises(ValueError, match="unknown metric"):
        S.per_weight_saliency(W, "nope", act_norm=act_norm)
    with pytest.raises(ValueError, match="requires act_norm"):
        S.per_weight_saliency(W, "wanda")
    with pytest.raises(ValueError, match="requires hinv_diag"):
        S.per_weight_saliency(W, "obs_diag")
    with pytest.raises(ValueError, match="cannot be negative"):
        S.per_weight_saliency(W, "wanda", act_norm=-torch.ones(N_IN, dtype=DT))
    with pytest.raises(ValueError, match="non-positive"):
        S.per_weight_saliency(
            W, "obs_diag", hinv_diag=torch.zeros(N_IN, dtype=DT)
        )
    with pytest.raises(ValueError, match="axis"):
        S.aggregate_to_tiles(W, "C", 4)
