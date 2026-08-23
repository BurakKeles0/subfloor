"""The cost model decides whether M1 gets run at all, so its arithmetic matters.

An error here is expensive in one direction and merely embarrassing in the
other: too low and weeks of GPU time get committed to a grid that cannot
finish; too high and a feasible experiment gets cancelled.  The tests below fix
the shapes and the scaling rather than the wall-clock numbers, which are
machine-specific by design.

Rates are faked throughout -- benchmarking inside a test suite would measure
whatever else the machine was doing.
"""

from __future__ import annotations

import pytest

import m0_cost_model as CM
import quantize as Qz
import tiling as Tl

#: Round numbers, so every expected value below is checkable by hand.
RATES = {
    "k_benchmarked": 2048,
    "threads": 1,
    "setups": {
        "fake": {
            "cholesky_flops_per_s": 1e12,
            "codebook_flops_per_s_small": 1e10,
            "codebook_flops_per_s_large": 1e11,
            # (k, lines, seconds).  Two tiles of the same width and different
            # line counts, so the fit has to separate the k^3 term from the
            # per-weight one rather than absorbing both.
            "tile_timings": ((1000, 10, 1e-2), (1000, 100, 1e-1)),
        }
    },
}


# --------------------------------------------------------------------------- #
# One layer
# --------------------------------------------------------------------------- #

def test_tile_count_follows_the_scheme():
    """T=1 gives every ROW its own column set; T=max gives the whole matrix one.

    The first is the expensive end, and it is expensive for a reason worth
    naming: per-row column sets are exactly the "row Hessian challenge" that
    SparseGPT exists to avoid.
    """
    assert CM.layer_cost(4096, 11008, 1, 1.5)["n_tiles"] == 4096
    assert CM.layer_cost(4096, 11008, Tl.MAX_TILE, 1.5)["n_tiles"] == 1
    assert CM.layer_cost(4096, 11008, 16, 1.5)["n_tiles"] == 256


def test_k_is_the_aligned_survivor_count_not_the_requested_density():
    """The code factorizes what the mask actually holds, so the cost model has
    to use the aligned count or it will understate every cubic term."""
    c = CM.layer_cost(4096, 11008, 16, 1.5)
    assert c["k"] % Qz.E8P_DIM == 0
    assert c["k"] == Tl.uniform_survivor_count(11008, c["density"],
                                               align=Qz.E8P_DIM)


def test_cholesky_work_falls_as_one_over_t_once_the_density_settles():
    """Between T=16 and T=max the density barely moves (0.719 -> 0.750) while
    the tile count drops 256-fold, so the factorization term collapses.  That is
    the whole shape of the problem: the cost lives at the fine end of the grid.
    """
    fine = CM.layer_cost(4096, 11008, 16, 1.5)
    coarse = CM.layer_cost(4096, 11008, Tl.MAX_TILE, 1.5)
    assert coarse["density"] > fine["density"]           # coarse is denser
    assert coarse["cholesky_flops"] < fine["cholesky_flops"] / 100


def test_the_codebook_term_does_not_depend_on_the_tile_count():
    """One search per line per group, so it is n_out * k/8 either way.  This is
    why the T=max column is not free: the factorization vanishes and the search
    does not."""
    a = CM.layer_cost(4096, 11008, 16, 1.5)
    b = CM.layer_cost(4096, 11008, Tl.MAX_TILE, 1.5)
    ratio = b["codebook_flops"] / a["codebook_flops"]
    assert ratio == pytest.approx(b["k"] / a["k"], rel=1e-9)


def test_streaming_the_sub_hessians_divides_memory_by_the_tile_count():
    c = CM.layer_cost(4096, 11008, 16, 1.5)
    assert c["hessian_bytes"] == c["hessian_bytes_streamed"] * c["n_tiles"]
    assert c["hessian_bytes"] > 100 * 2 ** 30          # over 100 GiB, as written
    assert c["hessian_bytes_streamed"] < 2 ** 30       # under 1 GiB, streamed


def test_an_unreachable_budget_costs_nothing_rather_than_raising():
    assert CM.layer_cost(4096, 11008, 16, 10.0) is None


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

def test_model_cost_counts_every_linear_in_every_block():
    """Seven linears per block, thirty-two blocks.  A missed shape would
    understate the total by a fixed fraction and never look wrong."""
    one = CM.model_cost(16, 1.5, RATES, "fake", n_blocks=1,
                        inventory=((4096, 4096, 1),))
    seven = CM.model_cost(16, 1.5, RATES, "fake", n_blocks=1,
                          inventory=((4096, 4096, 7),))
    assert seven["cholesky_flops"] == pytest.approx(7 * one["cholesky_flops"])

    full = CM.model_cost(16, 1.5, RATES, "fake", n_blocks=32,
                         inventory=((4096, 4096, 1),))
    assert full["cholesky_flops"] == pytest.approx(32 * one["cholesky_flops"])


def test_point_cost_includes_the_measured_evaluation():
    c = CM.model_cost(Tl.MAX_TILE, 1.5, RATES, "fake")
    assert c["point_seconds"] == pytest.approx(
        c["compress_seconds"] + CM.EVAL_SECONDS)
    assert c["eval_seconds"] == CM.EVAL_SECONDS


def test_batching_no_longer_moves_the_clock():
    """`batched` used to pick between two microbenchmarked rates.  Timing now
    comes from END-TO-END per-tile measurements, which already contain whatever
    batching the code does, so the flag is flop bookkeeping and nothing else.

    Kept as a test rather than deleted: silently having a flag that once changed
    the answer and no longer does is how a stale number gets quoted.
    """
    a = CM.model_cost(16, 1.5, RATES, "fake", batched=False)
    b = CM.model_cost(16, 1.5, RATES, "fake", batched=True)
    assert a["compress_seconds"] == pytest.approx(b["compress_seconds"])
    assert a["codebook_flops"] == pytest.approx(b["codebook_flops"])


def test_timing_comes_from_the_measured_tile_fit():
    """One weight costs the fitted constant, times n_out * k per linear.

    The constant is the WORST residual across the fixture's tiles, after the
    Cholesky is removed at its own rate -- worst, not mean, because this model
    has twice been wrong in the flattering direction.
    """
    chol = CM.CHOL_FLOPS_PER_K3 * 1000 ** 3 / RATES["setups"]["fake"][
        "cholesky_flops_per_s"]
    want = max((sec - chol) / (lines * k)
               for k, lines, sec in RATES["setups"]["fake"]["tile_timings"])
    per_weight = CM.codebook_seconds_per_vector(RATES, "fake")
    assert per_weight == pytest.approx(want)

    c = CM.model_cost(16, 1.5, RATES, "fake", n_blocks=1,
                      inventory=((4096, 4096, 1),))
    k = CM.layer_cost(4096, 4096, 16, 1.5)["k"]
    assert c["codebook_seconds"] == pytest.approx(4096 * k * per_weight)


def test_a_setup_without_measured_timings_is_refused():
    """Better to stop than to invent a rate for a device nobody timed."""
    bare = {"setups": {"unmeasured": {"cholesky_flops_per_s": 1e12}}}
    with pytest.raises(ValueError, match="no measured tile timings"):
        CM.codebook_seconds_per_vector(bare, "unmeasured")


def test_peak_memory_is_the_worst_layer_not_the_sum():
    """Layers are compressed one at a time, so the wall is the largest one."""
    c = CM.model_cost(16, 1.5, RATES, "fake")
    worst = max(CM.layer_cost(n_out, n_in, 16, 1.5)["hessian_bytes"]
                for n_out, n_in, _ in CM.LLAMA2_7B)
    assert c["peak_hessian_bytes"] == worst


# --------------------------------------------------------------------------- #
# What fits
# --------------------------------------------------------------------------- #

def test_affordable_never_drops_gate_bs_edges():
    """T=1 and T=max are what Gate B tests against.  Dropping either does not
    make the experiment cheaper, it makes it a different experiment."""
    out = CM.affordable(RATES, "fake", hours=0.0001)
    assert "1" in out["tiles_kept"]
    assert str(Tl.MAX_TILE) in out["tiles_kept"]
    assert not out["fits"]


def test_affordable_drops_the_most_expensive_interior_tile_first():
    out = CM.affordable(RATES, "fake", hours=0.0001)
    per = out["per_tile_hours"]
    dropped = out["tiles_dropped"]
    interior = [t for t in per if t not in ("1", str(Tl.MAX_TILE))]
    assert dropped[0] == max(interior, key=lambda t: per[t])
    assert len(dropped) == len(interior)


def test_affordable_keeps_everything_when_the_budget_is_generous():
    out = CM.affordable(RATES, "fake", hours=1e9)
    assert out["tiles_dropped"] == []
    assert out["fits"]


def test_affordable_reports_the_cost_of_what_it_kept():
    out = CM.affordable(RATES, "fake", hours=1e9)
    assert out["hours"] == pytest.approx(sum(out["per_tile_hours"].values()))


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #

def test_sweep_prices_q_at_t1_and_tau_across_the_grid():
    """`Q` is a T=1 curve; the tau points spread over the tile grid.  Pricing
    tau at the cheapest tile would flatter the estimate by an order of
    magnitude, so it is priced at the grid mean."""
    s = CM.sweep_cost(RATES, "fake", n_tau_points=25, n_q_points=5, n_q_seeds=3)
    q_point = s["per_tile"]["1"]["point_seconds"]
    assert s["q_seconds"] == pytest.approx(15 * q_point)

    tau = [v["point_seconds"] for k, v in s["per_tile"].items() if k != "1"]
    assert s["tau_seconds"] == pytest.approx(25 * sum(tau) / len(tau))
    assert s["total_seconds"] == pytest.approx(s["q_seconds"] + s["tau_seconds"])


def test_sweep_compares_itself_against_the_spec_estimate():
    s = CM.sweep_cost(RATES, "fake")
    assert s["spec_estimate_hours"] == 25.0
    assert s["over_spec"] == pytest.approx(s["total_hours"] / 25.0)


def test_m1_cost_scales_with_the_draw_count():
    a = CM.m1_cost(RATES, "fake", n_draws=5)
    b = CM.m1_cost(RATES, "fake", n_draws=10)
    assert b["seconds"] == pytest.approx(2 * a["seconds"])


# --------------------------------------------------------------------------- #
# The scale-fitting sweep
# --------------------------------------------------------------------------- #

def test_the_scale_fit_multiplier_hits_only_the_codebook_term():
    """`fit_scale` searches the codebook; it does not touch the Hessian.

    The first version of the model left it out entirely and understated every
    codebook figure sixfold, so the term is asserted rather than trusted.
    """
    with_fit = CM.layer_cost(4096, 11008, 16, 1.5, scale_fit=True)
    without = CM.layer_cost(4096, 11008, 16, 1.5, scale_fit=False)
    assert with_fit["cholesky_flops"] == without["cholesky_flops"]
    assert with_fit["codebook_flops"] == pytest.approx(
        CM.SCALE_FIT_MULTIPLIER * without["codebook_flops"])


def test_dropping_the_per_tile_scale_fit_is_worth_real_time():
    """Priced so the option can be compared against the harder fixes rather
    than argued about."""
    a = CM.model_cost(16, 1.5, RATES, "fake", scale_fit=True)
    b = CM.model_cost(16, 1.5, RATES, "fake", scale_fit=False)
    assert b["compress_seconds"] < a["compress_seconds"]
    assert a["cholesky_seconds"] == pytest.approx(b["cholesky_seconds"])
    saved = a["codebook_seconds"] - b["codebook_seconds"]
    assert saved == pytest.approx(
        b["codebook_seconds"] * (CM.SCALE_FIT_MULTIPLIER - 1))


def test_scale_fit_is_on_by_default_because_that_is_what_the_code_does():
    """The default has to describe the pipeline as written, not as we would
    like it -- an optimistic default is how a cost model stops being one."""
    assert CM.layer_cost(4096, 4096, 16, 1.5)["scale_fit"] is True
    assert CM.model_cost(16, 1.5, RATES, "fake")["scale_fit"] is True
