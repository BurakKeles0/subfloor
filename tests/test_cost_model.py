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
            # (k, lines, seconds).  Same width, different line counts, and
            # shaped like the real measurements: per-weight cost FALLS as the
            # batch grows (2e-6 at ten lines, 1e-6 at a hundred), on top of a
            # 1.67e-3 s Cholesky.  Numbers chosen so both residuals come out
            # round and every expectation below is checkable by hand.
            "tile_timings": ((1000, 10, 2.167e-2), (1000, 100, 1.0167e-1)),
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
                 for k, lines, sec in fake["tile_timings"]}

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


def test_the_per_weight_constant_falls_with_the_line_count():
    """Bigger tiles amortize the codebook load and the decoder's fixed cost.
    If this ever inverted, the model would be reading its samples backwards."""
    fake = RATES["setups"]["fake"]
    counts = sorted(lines for _, lines, _ in fake["tile_timings"])
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


def test_the_codebook_sweep_is_the_wall_not_the_factorization():
    """The finding that reorders everything.

    STATUS 6.3 reads the factorization as the structural wall and a block width
    as the fix.  Measured, the factorization is a fraction of the pass and the
    scale-fitting sweep is most of it.  Evaluated at the configuration the
    pipeline actually runs -- feedback confined to 512, sweep chunked, unsettled
    rows resolved analytically -- the codebook term still outweighs the Cholesky
    and the rotation together at every tile size, and dropping the per-tile
    scale fit is still worth more than twice what confining the feedback is.

    Both margins have NARROWED -- the first from over 5x to 3.6x, the second
    from 4x to 2.2x -- and that is progress rather than drift: the codebook term
    fell 2.6x when the scan went away while the rotation did not move.  Once
    they approach parity the rotation becomes worth attacking (TF32 buys 1.66x
    on it), so the thresholds below are loose enough to keep passing and tight
    enough to notice.

    The block width is still worth taking: it is free, it is what makes the
    chunked sweep affordable, and the measurement says it IMPROVES quality.  It
    is simply not the lever that decides whether M1 runs.
    """
    import json
    from pathlib import Path
    rates_file = Path(__file__).resolve().parent.parent / "results" / "m0_rates.json"
    if not rates_file.exists():
        pytest.skip("no measured rates on this machine")
    rates = json.loads(rates_file.read_text(encoding="utf-8"))
    if "cuda_f32" not in rates["setups"]:
        pytest.skip("no cuda rates measured on this machine")

    for tile in (1, 4, 16):
        c = CM.model_cost(tile, 1.5, rates, "cuda_f32", hessian_block=512)
        assert c["codebook_seconds"] >= 2 * (c["cholesky_seconds"]
                                             + c["rotation_seconds"])

    base = CM.m1_cost(rates, "cuda_f32")["days"]
    blocked = CM.m1_cost(rates, "cuda_f32", hessian_block=512)["days"]
    no_fit = CM.m1_cost(rates, "cuda_f32", scale_fit=False,
                        hessian_block=512)["days"]
    assert blocked > 0.7 * base                  # the block width: a minor saving
    assert (base - no_fit) > 2 * (base - blocked)   # the scale fit: still larger
