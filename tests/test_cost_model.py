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
            # (k, lines, n_tiles, seconds).  Same width, different line
            # counts, and shaped like the real measurements: per-weight cost
            # FALLS as the batch grows (2e-6 at ten lines, 1e-6 at a hundred),
            # on top of a 1.67e-3 s Cholesky.  Numbers chosen so both residuals
            # come out round and every expectation below is checkable by hand.
            # The tile count rides along because a tile time means nothing
            # without it: `auto_chunk` turns it into the row count that decides
            # which search path `_nearest` takes.
            # The fifth column is the same tile with a FIXED scale, so the
            # ratio between the two is what `scale_fit_multiplier` reads.  It is
            # deliberately different at the two line counts -- 1.5x at ten
            # lines, 1.2x at a hundred -- because a single figure for it was the
            # defect that made this column necessary.
            "tile_timings": ((1000, 10, 64, 2.167e-2, 1.50023e-2),
                             (1000, 100, 64, 1.0167e-1, 8.50028e-2)),
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

    The constant depends on the tile's LINE COUNT, because per-weight cost falls
    with batch size and a tile's line count is its tile size.  A single
    worst-case number would overstate the coarse end of the grid several times
    over -- the end the granularity question is actually about.
    """
    fake = RATES["setups"]["fake"]
    chol = CM.CHOL_FLOPS_PER_K3 * 1000 ** 3 / fake["cholesky_flops_per_s"]
    residuals = {lines: (sec - chol) / (lines * k)
                 for k, lines, _n, sec, _nf in fake["tile_timings"]}

    # exact line counts pick their own sample
    for lines, want in residuals.items():
        assert CM.codebook_seconds_per_vector(RATES, "fake", lines) ==             pytest.approx(want)
    # no line count given -> the conservative worst
    assert CM.codebook_seconds_per_vector(RATES, "fake") ==         pytest.approx(max(residuals.values()))

    c = CM.model_cost(16, 1.5, RATES, "fake", n_blocks=1,
                      inventory=((4096, 4096, 1),))
    k = CM.layer_cost(4096, 4096, 16, 1.5)["k"]
    per_weight = CM.codebook_seconds_per_vector(RATES, "fake", 16)
    assert c["codebook_seconds"] == pytest.approx(4096 * k * per_weight)


def test_a_point_charges_the_calibration_it_actually_does():
    """The model's sixth error, and the largest.

    `sequential_calibrate` walks every block TWICE per point -- once with hooks
    to accumulate the Hessians, once more so the next block sees the compressed
    output -- and neither pass was charged anywhere.  Nothing caught it because
    nothing had ever run the full driver.  At the configuration the code shipped
    with it was 5.59 hours per point, more than the whole compression pass at
    every tile size, and M1 read 12 days when the answer was 40.
    """
    import json
    from pathlib import Path
    rates_file = Path(__file__).resolve().parent.parent / "results" / "m0_rates.json"
    if not rates_file.exists():
        pytest.skip("no measured rates on this machine")
    rates = json.loads(rates_file.read_text(encoding="utf-8"))
    if "cuda_f32" not in rates["setups"]:
        pytest.skip("no cuda rates measured on this machine")

    cal = CM.calibration_seconds(rates, "cuda_f32")
    assert cal > 0
    c = CM.model_cost(4, 1.5, rates, "cuda_f32")
    assert c["calibration_seconds"] == pytest.approx(cal)
    assert c["point_seconds"] == pytest.approx(
        c["compress_seconds"] + cal + c["compensate_seconds"]
        + c["eval_seconds"])

    # Linear in both, because it is one pass over the tokens per block.
    assert CM.calibration_seconds(rates, "cuda_f32", tokens=2 * CM.CALIBRATION_TOKENS)         == pytest.approx(2 * cal)
    assert CM.calibration_seconds(rates, "cuda_f32", n_blocks=2 * CM.N_BLOCKS)         == pytest.approx(2 * cal)

    # And it does not depend on the tile size, which is what re-orders the
    # designs: cost now follows the number of POINTS, not which tiles they use.
    for t in (1, 16, Tl.MAX_TILE):
        assert CM.model_cost(t, 1.5, rates, "cuda_f32")["calibration_seconds"]             == pytest.approx(cal)


def test_a_point_charges_the_compensation_sweep_too():
    """The seventh error, and the third omission in a row.

    `run_config` calls `prune` before anything the model used to charge, and
    `TILE_TIMINGS` starts at `ldlq_quantize_blocks`, so `forward_compensate` --
    a Python loop the length of `n_in` whose every iteration touches the whole
    remaining width -- was priced nowhere.  40.7 s per block, 0.362 h per
    point, 1.58 days of M1.
    """
    import json
    from pathlib import Path
    rates_file = Path(__file__).resolve().parent.parent / "results" / "m0_rates.json"
    if not rates_file.exists():
        pytest.skip("no measured rates on this machine")
    rates = json.loads(rates_file.read_text(encoding="utf-8"))
    if "cuda_f32" not in rates["setups"]:
        pytest.skip("no cuda rates measured on this machine")

    exact = CM.compensate_seconds(rates, "cuda_f32")
    blocked = CM.compensate_seconds(rates, "cuda_f32", compensate_block=512)
    assert exact > 0 and blocked > 0
    # Blocking is the whole reason the second column exists; if it ever stopped
    # paying, `prune(compensate_block=...)` would be carrying a quality cost for
    # nothing.
    assert exact / blocked > 3.0

    # Flat in the tile size, like calibration -- that is what re-orders the
    # designs, since cost then follows the number of POINTS.
    for t in (1, 16, Tl.MAX_TILE):
        assert CM.model_cost(t, 1.5, rates, "cuda_f32")["compensate_seconds"]             == pytest.approx(exact)
    assert CM.model_cost(4, 1.5, rates, "cuda_f32",
                         compensate_block=512)["compensate_seconds"]         == pytest.approx(blocked)


def test_a_layer_shape_nobody_measured_is_charged_nothing():
    """No interpolation.  Five of seven errors were terms the model did not know
    about, and a plausible number for an unmeasured shape is how the next one
    would hide."""
    rates = {"setups": {"fake": {"compensate_timings": ((4096, 4096, 1.0, 0.2),)}}}
    assert CM.compensate_seconds(rates, "fake", inventory=((4096, 4096, 1),),
                                 n_blocks=1) == pytest.approx(1.0)
    # One shape missing from the table zeroes the whole term rather than
    # guessing at it, and the caller sees an obviously wrong 0 instead of a
    # plausibly wrong number.
    assert CM.compensate_seconds(rates, "fake",
                                 inventory=((4096, 4096, 1), (999, 999, 1)),
                                 n_blocks=1) == 0.0


def test_an_unmeasured_setup_is_charged_nothing_rather_than_a_guess():
    """This term was invisible for six versions of the model.  A fabricated
    default would put it back, silently and in the optimistic direction."""
    bare = {"setups": {"unmeasured": {"cholesky_flops_per_s": 1e12}}}
    assert CM.calibration_seconds(bare, "unmeasured") == 0.0


def test_the_cholesky_is_subtracted_at_the_width_it_was_measured_under():
    """The model's fifth error, and the second of them to be optimistic.

    `TILE_TIMINGS`'s cuda row was re-measured with `hessian_block=512`, so the
    factorization inside those tiles was (k/512) small ones rather than one of
    width k.  Subtracting a FULL-width Cholesky from that measurement takes out
    time the tile never spent, and what is left over-attributes nothing and
    under-attributes the codebook: 34% at (2560, 4), 24% at (2944, 16), 9% at
    (3072, 128).  Worst at the fine granularities, which is where the grid's
    cost lives and which the granularity question is about.

    The two rows really were taken under different arrangements -- `cpu_f64` is
    still the full-width one-tile form -- so this cannot be a single constant.
    """
    import copy
    blocked = copy.deepcopy(RATES)
    blocked["setups"]["fake"]["tile_timing_block"] = 250      # k=1000 -> 4 parts

    full = CM.codebook_seconds_per_vector(RATES, "fake", 10)
    got = CM.codebook_seconds_per_vector(blocked, "fake", 10)
    assert got > full, "subtracting less Cholesky must charge the codebook more"

    k, lines, _n, sec, _nf = RATES["setups"]["fake"]["tile_timings"][0]
    want = (sec - CM.cholesky_seconds(k, blocked, "fake", block=250)) / (lines * k)
    assert got == pytest.approx(want)

    # A setup that names no measurement width keeps the full-width assumption,
    # so nothing that was already right silently moves.
    assert CM.codebook_seconds_per_vector(RATES, "fake", 10) == pytest.approx(
        (sec - CM.cholesky_seconds(k, RATES, "fake")) / (lines * k))


def test_the_real_cuda_timings_declare_their_measurement_width():
    """The cuda row is the one the pipeline runs under, and it is blocked.  If
    someone re-measures it full-width and forgets this map, the codebook
    constant silently gains 34% -- so the map has to be part of the table."""
    assert CM.TILE_TIMING_BLOCK["cuda_f32"] == 512
    assert CM.TILE_TIMING_BLOCK["cpu_f64"] is None
    assert set(CM.TILE_TIMING_BLOCK) == set(CM.TILE_TIMINGS)


def test_the_per_weight_constant_falls_with_the_line_count():
    """Bigger tiles amortize the codebook load and the decoder's fixed cost.
    If this ever inverted, the model would be reading its samples backwards."""
    fake = RATES["setups"]["fake"]
    counts = sorted(lines for _, lines, _, _, _ in fake["tile_timings"])
    values = [CM.codebook_seconds_per_vector(RATES, "fake", n) for n in counts]
    assert values == sorted(values, reverse=True)


def test_line_counts_between_samples_snap_to_the_nearest_octave():
    """The samples are octaves apart, so interpolating between them would
    invent precision the measurements do not have."""
    fake = RATES["setups"]["fake"]
    small = CM.codebook_seconds_per_vector(RATES, "fake", 10)
    large = CM.codebook_seconds_per_vector(RATES, "fake", 100)
    assert small == CM.codebook_seconds_per_vector(RATES, "fake", 11)
    assert large == CM.codebook_seconds_per_vector(RATES, "fake", 90)
    assert small != large


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
        CM.NOMINAL_SCALE_FIT_MULTIPLIER * without["codebook_flops"])


def test_dropping_the_per_tile_scale_fit_is_worth_real_time():
    """Priced so the option can be compared against the harder fixes rather
    than argued about."""
    a = CM.model_cost(16, 1.5, RATES, "fake", scale_fit=True)
    b = CM.model_cost(16, 1.5, RATES, "fake", scale_fit=False)
    assert b["compress_seconds"] < a["compress_seconds"]
    assert a["cholesky_seconds"] == pytest.approx(b["cholesky_seconds"])
    saved = a["codebook_seconds"] - b["codebook_seconds"]
    mult = CM.scale_fit_multiplier(RATES, "fake", 16)
    assert saved == pytest.approx(b["codebook_seconds"] * (mult - 1))


def test_the_scale_fit_multiplier_varies_with_the_line_count():
    """The defect that made the fifth column necessary, asserted directly.

    `fit_scale` runs once per tile over the vectors that tile holds, so its
    share of a tile is set by how many there are -- 128 at T=1 against 5,888 at
    T=16 on the real grid.  A single constant prices one of those right and the
    other wrong, and until 2026-08-25 the constant was the four-line figure,
    which is the end the grid's cost actually lives at.

    Asserted on the fixture rather than on the real table so it states the
    RELATIONSHIP -- fewer lines, larger share -- instead of pinning a machine's
    numbers.
    """
    small = CM.scale_fit_multiplier(RATES, "fake", 10)
    large = CM.scale_fit_multiplier(RATES, "fake", 100)
    assert small == pytest.approx(1.5, rel=1e-3)
    assert large == pytest.approx(1.2, rel=1e-3)
    assert small > large, (
        "the per-tile fit's share must fall as a tile holds more vectors; if "
        "this inverted, the model is reading its samples backwards"
    )

    # `lines=None` keeps the old convention: report the WORST, so a rejection
    # argued on this number stays robust.
    assert CM.scale_fit_multiplier(RATES, "fake") == pytest.approx(small)


def test_the_scale_fit_multiplier_is_read_on_residuals_not_raw_times():
    """The Cholesky is in both arms and is not part of what the fit changes.
    Leaving it in would dilute the ratio by a different amount at every width,
    which is how a per-line-count table would quietly become a per-width one.
    """
    fake = RATES["setups"]["fake"]
    k, lines, _n, sec, no_fit = fake["tile_timings"][0]
    chol = CM.cholesky_seconds(k, RATES, "fake")
    got = CM.scale_fit_multiplier(RATES, "fake", lines)
    assert got == pytest.approx((sec - chol) / (no_fit - chol))
    assert got != pytest.approx(sec / no_fit), (
        "ratio taken on raw tile times; the Cholesky has to come out of both"
    )


def test_scale_fit_is_on_by_default_because_that_is_what_the_code_does():
    """The default has to describe the pipeline as written, not as we would
    like it -- an optimistic default is how a cost model stops being one."""
    assert CM.layer_cost(4096, 4096, 16, 1.5)["scale_fit"] is True
    assert CM.model_cost(16, 1.5, RATES, "fake")["scale_fit"] is True


# --------------------------------------------------------------------------- #
# BLOCK-DIAGONAL FEEDBACK  (docs/STATUS.md section 6.3, corrected)
# --------------------------------------------------------------------------- #

def test_the_factorization_is_per_tile_because_of_the_column_set():
    """The correction that motivated the block-width sweep.

    STATUS 6.3 reads the k^3 term as "each tile has its own column set AND its
    own rotation".  Only the first clause is true: `rotation.rotate` defaults to
    `share_across_tiles=True`, so one rotation serves the whole layer and no
    rotation width can make the factorization shared.  What the model charges
    is n_tiles factorizations, and n_tiles comes from the tiling alone.
    """
    c = CM.layer_cost(4096, 11008, 16, 1.5, hessian_block=None)
    assert c["n_tiles"] == 4096 // 16
    assert c["cholesky_flops"] == pytest.approx(
        c["n_tiles"] * CM.CHOL_FLOPS_PER_K3 * c["k"] ** 3)


@pytest.mark.parametrize("block", [2048, 512, 128])
def test_block_diagonal_feedback_turns_k_cubed_into_k_times_b_squared(block):
    full = CM.layer_cost(4096, 11008, 16, 1.5, hessian_block=None)
    blocked = CM.layer_cost(4096, 11008, 16, 1.5, hessian_block=block)
    k = full["k"]
    assert blocked["k"] == k
    # Exact, ragged tail included -- not k*b^2, which would round the tail away.
    expected = sum(min(block, k - o) ** 3 for o in range(0, k, block))
    assert blocked["cholesky_flops"] == pytest.approx(
        full["n_tiles"] * CM.CHOL_FLOPS_PER_K3 * expected)
    assert blocked["cholesky_flops"] < full["cholesky_flops"] / 10
    # The lever touches the factorization and nothing else.
    assert blocked["codebook_flops"] == full["codebook_flops"]


def test_block_width_wider_than_k_is_the_unconstrained_cost():
    full = CM.layer_cost(4096, 4096, Tl.MAX_TILE, 1.5, hessian_block=None)
    assert CM.layer_cost(4096, 4096, Tl.MAX_TILE, 1.5,
                         hessian_block=10 ** 6)["cholesky_flops"] == pytest.approx(
        full["cholesky_flops"])


def test_the_cost_curve_flattens_long_before_width_eight():
    """Why the sweep is run at 512 and not at 8.

    Section 6.3 proposed groups of eight.  Going from 512 down to 8 buys under
    2% more of M1's runtime while multiplying the dropped Hessian couplings by
    sixty-four -- the worst available trade.  The sweep spans the knee so the
    measurement can be read against it.
    """
    days = {b: CM.m1_cost(RATES, "fake", scale_fit=False, hessian_block=b)["days"]
            for b in (None, 2048, 512, 8)}
    total, extra = days[None] - days[8], days[512] - days[8]
    assert days[2048] < 0.6 * days[None]
    assert extra < 0.05 * total             # 512 already collects the saving

    # Machine-independent, because it is arithmetic rather than a rate: at the
    # widest k in the grid, a width of 512 has already dropped 99.6% of the
    # factorization, and everything narrower is fighting over the last 0.4%.
    k = CM.layer_cost(4096, 11008, 16, 1.5)["k"]
    kept = lambda b: sum(min(b, k - o) ** 3 for o in range(0, k, b)) / k ** 3
    assert kept(512) < 0.005
    assert kept(512) - kept(8) < 0.005


def test_m1_records_the_block_width_it_priced():
    """A cost claim that does not carry its assumptions is how the model was
    wrong three times already."""
    assert CM.m1_cost(RATES, "fake")["hessian_block"] == CM.DEFAULT_HESSIAN_BLOCK
    assert CM.m1_cost(RATES, "fake", hessian_block=None)["hessian_block"] is None
    # The default must track the pipeline, not history: `m1_gates` confines the
    # feedback to this width, so pricing an unconfined run by default would
    # quote a configuration nobody runs.
    import m1_gates
    assert CM.DEFAULT_HESSIAN_BLOCK == m1_gates.HESSIAN_BLOCK


def test_sampling_the_per_tile_scale_is_inert_where_the_cost_lives():
    """`docs/STATUS.md` reads the 68-day figure as "with the scale fit sampled".
    It is not that: `scale_fit=False` prices removing the fit entirely.

    The sweep only visits the vectors a tile actually has, so a cap of 8192 is
    inert at every tile size below T=max -- which is to say, at every tile size
    that costs anything.  The lever is real but it is not this lever, and the
    caps that would bite are small enough to need a measured quality cost.
    """
    for tile, expected in ((1, False), (4, False), (16, False),
                           (Tl.MAX_TILE, True)):
        c = CM.layer_cost(4096, 4096, tile, 1.5)
        assert CM.scale_sample_bites(c["lines_per_tile"], c["k"], 8192) is expected

    # It does bite once the cap is small -- that is where the measurement goes.
    c = CM.layer_cost(4096, 4096, 4, 1.5)
    assert CM.scale_sample_bites(c["lines_per_tile"], c["k"], 256) is True


# --------------------------------------------------------------------------- #
# THE CHOLESKY CORRECTION  (2026-08-23)
# --------------------------------------------------------------------------- #
# The model charged every width the flop/s measured at k=2048, from a benchmark
# that warmed `cholesky` but not `cholesky_inverse`.  Both were wrong in the
# same direction: 1.6x from the missing warmup, 2.6x more from the rate's
# k-dependence, 9.4x together at the widths that matter.  The 120-day M1 figure
# came out of that, and it is 94 once measured.

REAL = {"setups": {"cuda_f32": {"cholesky_flops_per_s": 1e9,   # deliberately absurd
                                "codebook_flops_per_s_small": 1e10,
                                "codebook_flops_per_s_large": 1e11}}}


def test_the_measured_curve_beats_the_flat_rate_when_one_exists():
    """A setup with a measured curve must ignore `cholesky_flops_per_s`
    entirely -- otherwise a stale rate in a results file silently wins."""
    got = CM.cholesky_seconds(4096, REAL, "cuda_f32")
    assert got == pytest.approx(CM.CHOL_TIMINGS["cuda_f32"][2][1], rel=1e-9)
    # ... and a setup without one still works, so fixtures keep functioning.
    assert CM.cholesky_seconds(1000, RATES, "fake") == pytest.approx(
        CM.CHOL_FLOPS_PER_K3 * 1000 ** 3 / 1e12)


def test_the_cholesky_rate_is_not_one_number():
    """Small factorizations cannot fill the card.  If this ever flattened, the
    curve would have stopped describing the kernel."""
    rate = lambda k, s: CM.CHOL_FLOPS_PER_K3 * k ** 3 / s
    rates = [rate(k, s) for k, s in CM.CHOL_TIMINGS["cuda_f32"]]
    assert rates == sorted(rates)                 # monotone in k
    assert rates[-1] > 5 * rates[0]


def test_block_diagonal_seconds_are_the_sum_over_blocks():
    k, b = 4096, 512
    got = CM.cholesky_seconds(k, REAL, "cuda_f32", block=b)
    want = sum(CM.cholesky_seconds(min(b, k - o), REAL, "cuda_f32")
               for o in range(0, k, b))
    assert got == pytest.approx(want)
    assert got < CM.cholesky_seconds(k, REAL, "cuda_f32") / 10


def test_the_rotation_apply_is_charged_now_that_it_outweighs_the_cholesky():
    """`tile_hessian_stream` rotates every tile's sub-Hessian, 2*k^3 at matmul
    rates.  It went unmodelled while the Cholesky dwarfed it; at k=7912 it is
    the larger of the two, and with the feedback blocked it is larger still."""
    assert CM.rotation_seconds(8192, REAL, "cuda_f32") > CM.cholesky_seconds(
        8192, REAL, "cuda_f32")
    assert CM.rotation_seconds(4096, REAL, "cuda_f32") == pytest.approx(
        CM.ROT_TIMINGS["cuda_f32"][2][1], rel=1e-9)
    assert CM.rotation_seconds(4096, RATES, "fake") == 0.0     # no curve, no charge


def test_compress_time_is_the_three_terms():
    c = CM.model_cost(16, 1.5, RATES, "fake")
    assert c["compress_seconds"] == pytest.approx(
        c["cholesky_seconds"] + c["rotation_seconds"] + c["codebook_seconds"])


def test_no_single_term_dominates_the_pass_any_more():
    """Where the optimisation ran out, and why that is itself the finding.

    STATUS 6.3 read the factorization as the structural wall.  It was not -- the
    codebook sweep outweighed the Cholesky and the rotation together by over 5x.
    Confining the feedback, chunking the sweep, replacing the scan with
    arithmetic and fusing the elementwise chains have since cut the codebook
    term about eightfold, and the ratio came down with it: 5x, then 3.6x, now
    about 1.7x.  The two remaining levers are within 10% of each other in days
    saved.

    So the invariant worth holding is no longer "the codebook is the wall".  It
    is that NOTHING is: the terms have converged, and the next optimisation has
    to be chosen by measuring which is largest rather than by assuming.  That
    deserves a test because the assumption was wrong twice.

    2026-08-24: it was wrong a third time, and this test caught it.  Batching
    `fit_scale`'s candidate scales cut the codebook term again (3.78x per tile
    at four lines), and the assertion that the codebook is still the LARGEST
    term went red.  It is not: at T=4 the sub-Hessian rotation is now 1.92h
    against the codebook's 1.34h.  That assertion was a fact with an expiry
    date, not an invariant, so it has been replaced by the invariant the
    docstring was already claiming -- no term runs away from the others -- plus
    a check on WHICH one leads, so the next person reads the answer instead of
    inheriting mine.

    2026-08-25: AND THE REPLACEMENT EXPIRED TOO, which is worth more than the
    number it was guarding.  `TILE_TIMINGS` was re-measured with the tile counts
    recorded, and with a sample below four lines for the first time.  At T=1 the
    codebook term is 2.86h against 0.46 of rotation and 0.34 of Cholesky -- 3.6x
    the other two together.  So "nothing is the wall" was itself a fact with an
    expiry date: it held while the fine end was priced off the four-line rate,
    and the fine end is not like the four-line rate at all.

    The wall at T=1 is `fit_scale`'s per-tile fixed cost.  A one-line tile gives
    it 128 vectors to amortize over; a T=4 tile gives it 1280.  T=1 is also the
    unstructured baseline the whole thesis compares against, so this is not a
    corner of the grid to wave at.

    What is asserted now is the shape rather than a slogan: the middle and
    coarse end are balanced, the fine end is not, and the fine end's leader is
    the codebook.
    """
    import json
    from pathlib import Path
    rates_file = Path(__file__).resolve().parent.parent / "results" / "m0_rates.json"
    if not rates_file.exists():
        pytest.skip("no measured rates on this machine")
    rates = json.loads(rates_file.read_text(encoding="utf-8"))
    if "cuda_f32" not in rates["setups"]:
        pytest.skip("no cuda rates measured on this machine")

    # From T=4 outward no term runs away from the rest.  Stated on the terms
    # themselves rather than on any one of them, so it survives the lead
    # changing hands again -- which it has done three times.
    for tile in (4, 16):
        c = CM.model_cost(tile, 1.5, rates, "cuda_f32")
        terms = {k: c[k + "_seconds"]
                 for k in ("codebook", "rotation", "cholesky")}
        top = max(terms, key=terms.__getitem__)
        rest = sum(terms.values()) - terms[top]
        assert terms[top] < 3 * rest, (
            f"{top} has run away from the other terms at T={tile}; the model "
            "was rebuilt around no single wall, so this means one is back"
        )

    # And at the fine end one HAS come back.  Pinned as a measured fact, with
    # its cause, so the next person reads it instead of inheriting the slogan.
    fine = CM.model_cost(1, 1.5, rates, "cuda_f32")
    others = fine["rotation_seconds"] + fine["cholesky_seconds"]
    assert fine["codebook_seconds"] > 3 * others, (
        "the codebook no longer dominates at T=1; it was 3.6x the rotation and "
        "Cholesky together as of 2026-08-25, driven by `fit_scale`'s fixed cost "
        "over a one-line tile's 128 vectors -- if that has changed, the fine "
        "end of the grid has a different cost story and section 6.1 is stale"
    )

    # Which term leads is a measured fact and it has changed twice.  Pinning it
    # is the point: the ROTATION is now the largest at the grid's expensive
    # middle, and it is the one term still computed as a dense GEMM against a
    # matrix that is a Kronecker product (`rotation.structured_orthogonal`).
    at_four = CM.model_cost(4, 1.5, rates, "cuda_f32")
    assert at_four["rotation_seconds"] > at_four["codebook_seconds"], (
        "the codebook has retaken the lead at T=4; the sub-Hessian rotation was "
        "the largest term as of 2026-08-24 and the next lever was priced on that"
    )

    # The two levers that cost quality used to be comparable -- 8.8 days against
    # 8.3 -- which was the argument for measuring rather than guessing which to
    # pull.  Batching the candidate fit took the scale lever's cost away with
    # it: dropping the per-tile fit now saves about a seventh of what confining
    # the feedback already saves, and confining the feedback is a decision
    # already taken (and one that IMPROVES quality, 2026-08-23).
    #
    # So the state to hold is that no large quality-costing lever is left: the
    # cheap wins have been taken, and anything further has to be paid for in
    # accuracy or in a structural change like the rotation's Kronecker form.
    base = CM.m1_cost(rates, "cuda_f32")["days"]
    unblocked = CM.m1_cost(rates, "cuda_f32", hessian_block=None)["days"]
    no_fit = CM.m1_cost(rates, "cuda_f32", scale_fit=False)["days"]
    scale_lever, block_lever = base - no_fit, unblocked - base
    assert scale_lever < 0.5 * block_lever, (
        "the scale fit has grown back into a major cost; it was 83% of a tile "
        "before the candidates were batched and 28% after"
    )
    assert scale_lever / base < 0.2, (
        "dropping the per-tile scale fit is supposed to be a minor saving now; "
        "if it is large again the fit is being paid for once per candidate"
    )
