"""The power analysis has to be trustworthy before its answer is.

It is a simulation that decides how much GPU time M1 is worth, so the failure
mode is a flattering one: a bug that makes Gate B look more powerful than it is
would send M1 out with too few draws and come back undetermined.  The tests
below aim at that -- especially `test_the_shared_draw_effect_changes_nothing`,
which is the pairing argument itself rather than a property of the simulator.

Trial counts are small on purpose; these check structure and direction, not the
third digit of a power estimate.
"""

from __future__ import annotations

import math

import pytest
import torch

import m0_gate_b_power as PW
import m1_gates as M
import tiling as Tl


# --------------------------------------------------------------------------- #
# The truth we simulate under
# --------------------------------------------------------------------------- #

def test_u_curve_has_its_edges_up_and_its_minimum_where_asked():
    mu = PW.u_curve(0.5, t_opt=8)
    assert mu[1] == mu[Tl.MAX_TILE] == PW.BASELINE
    interior = {t: v for t, v in mu.items() if t not in (1, Tl.MAX_TILE)}
    assert min(interior, key=interior.get) == 8
    assert mu[8] == pytest.approx(PW.BASELINE - 0.5)
    # symmetric in log2(T) around the optimum
    assert mu[4] == pytest.approx(mu[16])


def test_a_zero_effect_curve_is_flat():
    mu = PW.u_curve(0.0)
    assert len(set(mu.values())) == 1


def test_spread_controls_how_flat_the_interior_is():
    tight, flat = PW.u_curve(1.0, spread=0.8), PW.u_curve(1.0, spread=3.0)
    # a flat interior keeps the neighbours near the optimum
    assert flat[4] - flat[8] < tight[4] - tight[8]


def test_simulate_records_lays_draws_out_the_way_gate_b_pairs_them():
    """`gate_b` pairs by position within each tile's list, so draw s of every
    tile must land at index s.  If this ordering ever broke, the paired
    bootstrap would silently compare unrelated draws."""
    g = torch.Generator().manual_seed(0)
    recs = PW.simulate_records(PW.u_curve(1.0), 5, 1.0, g)
    by_tile = {}
    for r in recs:
        by_tile.setdefault(r["tile_size"], []).append(r["rel_output_error"])
    assert set(by_tile) == set(PW.TILES)
    assert all(len(v) == 5 for v in by_tile.values())
    assert [r["tile_size"] for r in recs[:5]] == [PW.TILES[0]] * 5


def test_the_shared_draw_effect_changes_nothing():
    """The pairing claim, as a test rather than a comment.

    Noise shared by every tile within a draw -- a calibration sample that is
    simply harder -- shifts all tiles equally, so it cancels in every paired
    difference and moves no mean's rank.  `gate_b` must return exactly the same
    verdict, the same T*, and the same intervals.
    """
    mu = PW.u_curve(1.0)
    for trial in range(5):
        a = M.gate_b(PW.simulate_records(
            mu, 8, 1.0, torch.Generator().manual_seed(trial), draw_effect=0.0))
        b = M.gate_b(PW.simulate_records(
            mu, 8, 1.0, torch.Generator().manual_seed(trial), draw_effect=5.0))
        assert a["verdict"] == b["verdict"]
        assert a["t_star"] == b["t_star"]
        assert a["vs_T1_ci"] == pytest.approx(b["vs_T1_ci"], abs=1e-9)
        assert a["vs_Tmax_ci"] == pytest.approx(b["vs_Tmax_ci"], abs=1e-9)


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #

def test_three_draws_can_never_pass():
    """`gate_b`'s min_seeds guard, seen from the outside: below five draws the
    power is identically zero no matter how large the effect."""
    r = PW.power_at(3, 4.0, n_trials=40)
    assert r["power"] == 0.0
    assert r["p_undetermined"] == 1.0


def test_power_rises_with_the_effect_and_with_the_draws():
    weak = PW.power_at(10, 0.5, n_trials=80, seed=1)
    strong = PW.power_at(10, 3.0, n_trials=80, seed=1)
    assert strong["power"] > weak["power"]

    few = PW.power_at(5, 1.5, n_trials=80, seed=2)
    many = PW.power_at(20, 1.5, n_trials=80, seed=2)
    assert many["power"] > few["power"]


def test_no_effect_almost_never_reads_as_interior():
    """The property the gate exists for.  A bare argmin would call this
    'interior' about half the time; with the Bonferroni correction and the
    paired CIs it should be rare."""
    for n in (5, 20):
        r = PW.power_at(n, 0.0, n_trials=200, seed=7)
        assert r["power"] <= 0.05, f"{n} draws: {r['power']} false positives"


def test_mdd_is_monotone_in_the_number_of_draws():
    """More draws cannot need a larger effect."""
    rows = PW.power_curve(draws=(5, 20), effects=(0.0, 1.0, 2.0, 3.0),
                          n_trials=80, seed=3)
    table = PW.mdd(rows, 0.8)
    found = [(n, v) for n, v in sorted(table.items()) if v is not None]
    for (n0, v0), (n1, v1) in zip(found, found[1:]):
        assert v1 <= v0 + 1e-9, f"{n1} draws needs more than {n0} draws"


def test_mdd_returns_none_when_the_grid_never_reaches_the_target():
    rows = PW.power_curve(draws=(5,), effects=(0.0, 0.1), n_trials=40, seed=4)
    assert PW.mdd(rows, 0.8)[5] is None


# --------------------------------------------------------------------------- #
# Telling two interior tiles apart
# --------------------------------------------------------------------------- #

def test_selection_is_a_coin_flip_when_the_tiles_are_tied():
    r = PW.selection_power(0.0, 10)
    assert r["p_argmin_correct"] == pytest.approx(0.5)
    assert r["p_ci_separates"] < 0.05


def test_selection_matches_the_closed_form_it_claims():
    """Guard against the sqrt(2) going missing: the paired difference of two
    tiles has sd sigma*sqrt(2), not sigma."""
    from statistics import NormalDist
    gap, n = 0.7, 12
    want = NormalDist().cdf(gap * math.sqrt(n) / math.sqrt(2.0))
    assert PW.selection_power(gap, n)["p_argmin_correct"] == pytest.approx(want)


def test_picking_t_star_is_harder_than_passing_the_gate():
    """The trap worth naming: the verdict can be solid while T* is not.

    Separating the optimum from the EDGES is a larger difference than
    separating it from its NEIGHBOUR, so a run can report 'interior' with
    confidence and still have the headline granularity be close to a guess.
    """
    r = PW.power_at(10, 1.5, n_trials=150, spread=3.0, seed=11)
    assert r["power"] > r["p_correct_t_star"]


def test_draws_for_selection_falls_as_the_gap_grows():
    ns = [PW.draws_for_selection(g) for g in (0.25, 0.5, 1.0, 2.0)]
    assert all(n is not None for n in ns)
    assert ns == sorted(ns, reverse=True)


# --------------------------------------------------------------------------- #
# The noise measurement
# --------------------------------------------------------------------------- #

def test_a_calibration_draw_holds_the_layer_fixed():
    """The distinction the pre-registration rests on: a draw is new DATA, not a
    new model.  Redrawing W would fold layer-to-layer variation into sigma."""
    a = PW.redraw_activations(16, 32, 64, data_seed=0)
    b = PW.redraw_activations(16, 32, 64, data_seed=1)
    assert torch.equal(a.W, b.W)
    assert not torch.equal(a.X, b.X)


def test_measure_noise_rejects_an_unknown_axis():
    with pytest.raises(ValueError):
        PW.measure_noise(axis="whatever")
