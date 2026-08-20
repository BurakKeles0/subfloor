"""The pilot decides a pre-registered tolerance, so its wiring has to be right.

Every failure here is a quiet one.  A predictor assembled against the wrong
density, or a quantized run compared against an unquantized one at a different
sparsity, still produces a plausible-looking bias -- and that bias would be
frozen into the pre-registration as the tolerance.  `test_t1_predicts_itself...`
is the load-bearing test: at T=1 the predictor is an identity, so anything other
than exactly zero means it is not measuring what it claims to.

Layers are tiny; these check structure, not magnitudes.
"""

from __future__ import annotations

import pytest

import accounting as A
import m0_transfer_pilot as TP
import m1_gates as M
import quantize as Qz
import tiling as Tl
from m0_gate_b_power import redraw_activations


@pytest.fixture
def problem():
    return redraw_activations(32, 64, 128, data_seed=0)


# --------------------------------------------------------------------------- #
# One point
# --------------------------------------------------------------------------- #

def test_t1_predicts_itself_exactly(problem):
    """At T=1 the predictor reduces to Q(d(1)) + 0, and Q(d(1)) IS the
    measurement.  Exactly zero, not approximately: any drift means the two sides
    were run at different densities or with different settings."""
    r = TP.point(problem, 1.5, 1)
    assert r["tau_noquant"] == 0.0
    assert r["tau_quant"] == 0.0
    assert r["prediction_error"] == 0.0
    assert r["delta_predicted"] == r["delta_measured"] == r["Q"]


@pytest.mark.parametrize("tile", [2, 4, 8, 16, Tl.MAX_TILE])
def test_every_comparison_sits_at_one_density(problem, tile):
    """`tau` is defined at EQUAL density.  All four runs behind a point must
    land on the same realized density or the definition is false -- which is why
    `point` raises rather than returning a number it cannot justify.

    Realized is NOT asserted equal to requested: aligning the survivor count to
    eight moves it, by a lot on a layer this small.  That is a separate,
    reported quantity; what has to hold here is that all four runs moved
    together.
    """
    r = TP.point(problem, 1.5, tile)
    if r is None:
        pytest.skip("budget unreachable at this tile size")

    for quantize in (True, False):
        a = M.run_config(problem, budget_bits=1.5, tile_size=tile,
                         quantize=quantize, ldlq=quantize, align=TP.ALIGN)
        b = M.run_config(problem,
                         budget_bits=A.bits_per_position(
                             "unstructured", r["density"], None, problem.n_in,
                             vq_bits=M.E8P_BITS),
                         tile_size=1, quantize=quantize, ldlq=quantize,
                         align=TP.ALIGN)
        assert a["density_realized"] == b["density_realized"]
        assert a["density_realized"] == r["density_realized"]


def test_prediction_error_is_exactly_minus_the_transfer_error(problem):
    """Delta_pred - Delta_meas = (Q + tau_nq) - (Q + tau_q) = tau_nq - tau_q.

    An algebraic identity, so it holds to the last bit -- and it is the cheapest
    check that `Q` really is the same run on both sides of the subtraction.
    """
    for tile in (2, 4, 8, 16, Tl.MAX_TILE):
        r = TP.point(problem, 1.5, tile)
        if r is None:
            continue
        assert r["prediction_error"] == pytest.approx(-r["transfer_error"],
                                                      abs=1e-15)


def test_an_unreachable_budget_gives_nothing_rather_than_a_number(problem):
    """A budget above the dense cost has no density that meets it."""
    assert A.density_for_budget("tile", 10.0, None, problem.n_in,
                                tile_size=16, vq_bits=M.E8P_BITS) is None
    assert TP.point(problem, 10.0, 16) is None


def test_alignment_is_forced_on_both_sides(problem):
    """The bug this guards against: LDLQ forces align=8 while an unquantized run
    defaults to align=1, so the two would be compared at different sparsities
    and the whole equal-density premise would be silently false."""
    q = M.run_config(problem, budget_bits=1.5, tile_size=8, quantize=True)
    n = M.run_config(problem, budget_bits=1.5, tile_size=8, quantize=False)
    assert q["align"] == Qz.E8P_DIM and n["align"] == 1
    assert q["survivors_per_tile"] != n["survivors_per_tile"]

    qa = M.run_config(problem, budget_bits=1.5, tile_size=8, quantize=True,
                      align=TP.ALIGN)
    na = M.run_config(problem, budget_bits=1.5, tile_size=8, quantize=False,
                      ldlq=False, align=TP.ALIGN)
    assert qa["survivors_per_tile"] == na["survivors_per_tile"]
    assert qa["density_realized"] == na["density_realized"]


# --------------------------------------------------------------------------- #
# The tolerance
# --------------------------------------------------------------------------- #

def _fake_pilot(biases: dict, noise: float = 0.001) -> dict:
    return {
        "budget": 1.5,
        "draws": [],
        "per_tile": {
            str(t): {"tile_size": t, "n_draws": 3, "density": 0.7,
                     "delta_measured": 0.3, "delta_predicted": 0.3 + b,
                     "tau_noquant": 0.0 if t == 1 else 0.1,
                     "tau_quant": (0.0 if t == 1 else 0.1) - b,
                     "bias": b, "noise": noise}
            for t, b in biases.items()
        },
    }


def test_tolerance_ignores_the_t1_identity():
    """T=1's error is structurally zero, so including it would only pull the
    maximum down and understate the tolerance."""
    out = TP.tolerance(_fake_pilot({1: 0.0, 4: 0.02, 16: -0.03}), headroom=1.0)
    assert out["max_abs_bias"] == pytest.approx(0.03)
    assert out["worst_tile"] == 16


def test_tolerance_is_sized_to_bias_not_noise():
    """The audit's point, as an assertion: a large bias with a small spread
    must produce a large tolerance."""
    out = TP.tolerance(_fake_pilot({4: 0.04, 8: 0.01}, noise=0.0005),
                       headroom=1.5)
    assert out["tolerance"] == pytest.approx(0.06)
    assert out["bias_over_noise"] == pytest.approx(80.0)


def test_headroom_scales_the_tolerance_and_nothing_else():
    p = _fake_pilot({4: 0.02, 8: 0.01})
    a, b = TP.tolerance(p, headroom=1.0), TP.tolerance(p, headroom=2.0)
    assert b["tolerance"] == pytest.approx(2 * a["tolerance"])
    assert b["max_abs_bias"] == a["max_abs_bias"]


def test_tolerance_refuses_a_pilot_with_only_the_identity_point():
    with pytest.raises(ValueError):
        TP.tolerance(_fake_pilot({1: 0.0}))


def test_identity_check_catches_a_miswired_predictor():
    good = TP.identity_check(_fake_pilot({1: 0.0, 4: 0.02}))
    assert good["exact"]
    bad = _fake_pilot({1: 1e-6, 4: 0.02})
    assert not TP.identity_check(bad)["exact"]


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_pilot_separates_bias_from_noise(problem):
    out = TP.pilot(budget=1.5, n_draws=2, n_out=32, n_in=64, n_samples=128,
                   tiles=(1, 4, Tl.MAX_TILE))
    assert set(out["per_tile"]) == {"1", "4", "max"}
    assert out["per_tile"]["1"]["bias"] == 0.0
    for v in out["per_tile"].values():
        assert v["n_draws"] == 2
        assert v["noise"] >= 0.0


def test_argmin_agreement_flags_a_bias_that_reorders_the_curve():
    """A constant offset is harmless; a sign-changing one is not.

    Built so the predicted and measured optima genuinely differ: T=8 measures
    best but the bias makes T=4 predict best.
    """
    p = _fake_pilot({4: 0.0, 8: 0.0})
    p["per_tile"]["4"].update(delta_measured=0.31, delta_predicted=0.28,
                              bias=-0.03)
    p["per_tile"]["8"].update(delta_measured=0.30, delta_predicted=0.32,
                              bias=+0.02)
    out = TP.argmin_agreement(p)
    assert out["t_star_predicted"] == 4
    assert out["t_star_measured"] == 8
    assert not out["agrees"]
    assert out["bias_changes_sign"]


def test_argmin_agreement_ignores_a_constant_offset():
    p = _fake_pilot({4: 0.02, 8: 0.02})
    p["per_tile"]["4"].update(delta_measured=0.31, delta_predicted=0.33)
    p["per_tile"]["8"].update(delta_measured=0.30, delta_predicted=0.32)
    out = TP.argmin_agreement(p)
    assert out["agrees"] and not out["bias_changes_sign"]


def test_argmin_agreement_excludes_the_t1_identity():
    """T=1 predicts itself perfectly, so leaving it in would let the identity
    win the argmin and hide a disagreement among the tiles that matter."""
    p = _fake_pilot({1: 0.0, 4: 0.01})
    p["per_tile"]["1"].update(delta_measured=0.05, delta_predicted=0.05)
    p["per_tile"]["4"].update(delta_measured=0.31, delta_predicted=0.32)
    assert TP.argmin_agreement(p)["t_star_measured"] == 4
