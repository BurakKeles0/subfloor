"""The M1 driver, and above all the guard on Gate B.

`test_gate_b_does_not_cry_interior_on_noise` is the point of the whole file.
With three calibration draws, the argmin of a flat-but-noisy curve lands in the
interior most of the time.  A gate defined as `argmin` would therefore "pass"
on data containing no effect at all -- exactly the false positive plan section
B5 exists to prevent.
"""

from __future__ import annotations

import pytest
import torch

import m1_gates as M
import tiling as Tl


def _records(errors_by_tile: dict) -> list[dict]:
    return [
        {"tile_size": t, "rel_output_error": e}
        for t, errs in errors_by_tile.items()
        for e in errs
    ]


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def test_bootstrap_ci_brackets_the_mean():
    v = [1.0, 1.1, 0.9, 1.05, 0.95]
    lo, hi = M.bootstrap_ci(v, n_boot=2000, seed=0)
    assert lo < sum(v) / len(v) < hi


def test_bootstrap_ci_narrows_with_more_data():
    g = torch.Generator().manual_seed(0)
    small = (torch.randn(4, generator=g) * 0.1 + 1).tolist()
    large = (torch.randn(200, generator=g) * 0.1 + 1).tolist()
    w_small = M.bootstrap_ci(small, n_boot=4000)[1] - M.bootstrap_ci(small, n_boot=4000)[0]
    w_large = M.bootstrap_ci(large, n_boot=4000)[1] - M.bootstrap_ci(large, n_boot=4000)[0]
    assert w_large < w_small


def test_pairing_cancels_shared_draw_noise():
    """Why the bootstrap is paired: every tile size sees the same calibration
    draw, so the draw-to-draw component is common and subtracts out.  Ignoring
    the pairing throws that away and widens the interval for nothing."""
    g = torch.Generator().manual_seed(0)
    shared = torch.randn(12, generator=g) * 1.0          # draw-to-draw noise
    a = (shared + 0.10).tolist()
    b = (shared + 0.15).tolist()

    lo_p, hi_p = M.paired_bootstrap_ci(a, b, n_boot=4000)
    lo_a, hi_a = M.bootstrap_ci(a, n_boot=4000)
    lo_b, hi_b = M.bootstrap_ci(b, n_boot=4000)

    assert (hi_p - lo_p) < (hi_a - lo_a) / 5
    assert lo_p < -0.05 < hi_p                            # true difference
    assert lo_a < hi_b, "unpaired intervals overlap and settle nothing"


def test_paired_bootstrap_rejects_ragged_input():
    with pytest.raises(ValueError, match="equal lengths"):
        M.paired_bootstrap_ci([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------- #
# Gate B
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("noise_seed", range(6))
def test_gate_b_does_not_cry_interior_on_noise(noise_seed):
    """No effect in the data: every tile size has the same true error.  Run over
    several noise draws, because a guard that only holds for one seed is not a
    guard.

    Without the Bonferroni correction and the minimum-draw rule this fails --
    which is precisely why both are in `gate_b`.
    """
    g = torch.Generator().manual_seed(noise_seed)
    tiles = [1, 2, 4, 8, 16, Tl.MAX_TILE]
    noisy = {t: (0.5 + torch.randn(12, generator=g) * 0.02).tolist() for t in tiles}
    verdict = M.gate_b(_records(noisy))["verdict"]
    assert verdict != "interior", "argmin of noise must not pass the gate"


def test_gate_b_refuses_to_rule_on_three_draws():
    """Spec v6 section 6 asks for seeds >= 3.  Three is enough to report a mean
    and not enough to decide this gate -- a percentile bootstrap over three
    numbers has no coverage guarantee.  Say so instead of producing a verdict.
    """
    tiles = {1: [0.60, 0.61, 0.59], 4: [0.40, 0.41, 0.39],
             Tl.MAX_TILE: [0.58, 0.59, 0.57]}
    out = M.gate_b(_records(tiles))
    assert out["verdict"] == "undetermined"
    assert "too few" in out["reason"]


def test_gate_b_finds_a_real_interior_optimum():
    g = torch.Generator().manual_seed(0)

    def draws(mu):
        return (mu + torch.randn(10, generator=g) * 0.01).tolist()

    tiles = {1: draws(0.60), 2: draws(0.50), 4: draws(0.40),
             8: draws(0.45), Tl.MAX_TILE: draws(0.58)}
    out = M.gate_b(_records(tiles))
    assert out["verdict"] == "interior"
    assert out["t_star"] == 4
    assert out["beats_fine"] and out["beats_coarse"]
    assert out["alpha_effective"] < 0.05, "Bonferroni over the interior candidates"


def test_gate_b_reports_edge_when_the_optimum_is_at_an_edge():
    """T=max winning means structured pruning wins -- a real result, and not
    the thesis (Spec v6's decision table)."""
    g = torch.Generator().manual_seed(1)

    def draws(mu):
        return (mu + torch.randn(10, generator=g) * 0.01).tolist()

    tiles = {1: draws(0.60), 4: draws(0.50), 16: draws(0.45),
             Tl.MAX_TILE: draws(0.30)}
    out = M.gate_b(_records(tiles))
    assert out["verdict"] == "edge"


def test_gate_b_needs_both_edges():
    out = M.gate_b(_records({4: [0.4], 8: [0.5]}))
    assert out["verdict"] == "undetermined"
    assert "edges" in out["reason"]


# --------------------------------------------------------------------------- #
# Gate A
# --------------------------------------------------------------------------- #

def test_gate_a_passes_only_when_the_ci_clears_the_wall():
    wall = {"rel_output_error": 0.50, "bits_realized": 2.0}
    clear = M.gate_a(_records({4: [0.30, 0.31, 0.29]}), wall)
    assert clear["verdict"] == "pass" and clear["best_tile"] == 4

    marginal = M.gate_a(_records({4: [0.49, 0.52, 0.48]}), wall)
    assert marginal["verdict"] == "fail", "overlapping the wall is not a pass"


def test_gate_a_states_the_budget_mismatch():
    """The sparse configs sit below 2 bits, so this is not budget-matched and
    the record has to say so."""
    out = M.gate_a(_records({4: [0.3]}), {"rel_output_error": 0.5, "bits_realized": 2.0})
    assert "budget-matched" in out["note"]
    assert out["wall_bits"] == 2.0


# --------------------------------------------------------------------------- #
# run_config
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def problem():
    return M.synthetic_problem(n_out=32, n_in=64, n_samples=128, seed=0)


@pytest.fixture(scope="module")
def big_problem():
    """Wide enough for the effects below to be measurable: at n_in=64 the
    align-to-8 rounding is 12.5% granular and the Hessian is estimated from too
    few samples for its dimension."""
    return M.synthetic_problem(n_out=64, n_in=128, n_samples=256, seed=0)


@pytest.mark.parametrize("tile_size, want_density", [
    (1, 0.25), (2, 0.5), (4, 0.625), (8, 0.6875), (16, 0.71875), (Tl.MAX_TILE, 0.75),
])
def test_run_config_hits_the_budget_exactly(problem, tile_size, want_density):
    """B=1.5 with E8P survivors gives exact dyadic densities, so the realized
    bits should land on the budget with zero offset.

    `ldlq=False` here because LDLQ aligns the survivor count to 8, which moves
    the realized density on purpose -- see the next test.
    """
    r = M.run_config(problem, budget_bits=1.5, tile_size=tile_size,
                     ldlq=False, seed=0)
    assert "skipped" not in r
    assert r["density_requested"] == pytest.approx(want_density, abs=1e-12)
    assert r["density_realized"] == pytest.approx(want_density, abs=1e-12)
    assert r["bits_realized"] == pytest.approx(1.5, abs=1e-12)
    assert r["offset"] == pytest.approx(0.0, abs=1e-12)
    assert not r["flagged"]


def test_alignment_moves_the_density_and_says_so(problem):
    """LDLQ needs k % 8 == 0, so the survivor count is rounded -- and the
    accounting reports the resulting offset instead of hiding it.

    On this 64-wide fixture the rounding is coarse (8/64 = 12.5% granularity).
    At a real n_idx = 11008 it is 0.07%, i.e. essentially free -- which is why
    tensor-core alignment is worth taking (plan section E3).
    """
    plain = M.run_config(problem, budget_bits=1.5, tile_size=8, ldlq=False, seed=0)
    aligned = M.run_config(problem, budget_bits=1.5, tile_size=8, ldlq=True, seed=0)

    assert plain["offset"] == pytest.approx(0.0, abs=1e-12)
    assert aligned["density_realized"] != plain["density_realized"]
    assert aligned["offset"] != 0.0
    assert aligned["flagged"], "a >1% offset must be flagged, not swallowed"

    k = aligned["density_realized"] * aligned["n_idx"]
    assert k % 8 == pytest.approx(0.0, abs=1e-9)


def test_ldlq_is_axis_b_only(problem):
    with pytest.raises(NotImplementedError, match="Axis B"):
        M.run_config(problem, budget_bits=1.5, tile_size=8, axis="A", ldlq=True)


def test_run_config_records_the_provenance_protocol_6_wants(problem):
    r = M.run_config(problem, budget_bits=1.5, tile_size=16, seed=0)
    for key in ("bits_realized", "q_over_scales_with_density", "budget_bits",
                "offset", "n_idx", "density_realized", "vq_bits", "seed",
                "in_bitmap_regime"):
        assert key in r, f"protocol section 6 requires {key} in every record"


def test_run_config_skips_unreachable_budgets(problem):
    """At 2.14 bits an E8P tile family runs past d=1 -- which is why the anchors
    moved (plan H2)."""
    r = M.run_config(problem, budget_bits=2.140625, tile_size=16, seed=0)
    assert r["skipped"]


def test_run_config_can_hand_back_the_compressed_weight(problem):
    """`calibrate.sequential_calibrate`'s `compress_fn` has to RETURN a weight,
    and `run_config` computed one and dropped it -- so the two halves of the
    seam could not be connected at all, which is half of why
    `experiments/m1_run.py` does not exist (`docs/STATUS.md` section 8.1).

    Off by default on purpose: the grid runs thousands of configs and a weight
    per record would hold the whole sweep in memory.
    """
    plain = M.run_config(problem, budget_bits=1.5, tile_size=16, seed=0)
    assert "W_hat" not in plain

    with_w = M.run_config(problem, budget_bits=1.5, tile_size=16, seed=0,
                          return_weight=True)
    W_hat = with_w["W_hat"]
    assert W_hat.shape == problem.W.shape
    assert W_hat.device == problem.W.device

    # It has to be the weight the record was MEASURED on, not a second pass:
    # a re-derived weight would be the thing this test cannot see going stale.
    assert problem.output_error(W_hat) == pytest.approx(
        with_w["rel_output_error"], rel=1e-12)
    assert with_w["rel_output_error"] == pytest.approx(
        plain["rel_output_error"], rel=1e-12)


def test_compensation_helps_until_the_quantizer_takes_it_back(big_problem):
    """A FINDING with a direct consequence for the milestone plan.

    Measured on this fixture at B=1.5:

                    no quantizer    with 2-bit E8P
        T=4            -19.1%           +2.5%
        T=8            -20.7%           +4.4%
        T=16           -16.1%           -7.6%

    OBS compensation works by pushing a removed weight's job onto the survivors,
    which makes those survivors LARGER and more spread out.  A 2-bit quantizer
    then damages them more than it would have.  Sequentially, the second step
    undoes much of what the first bought -- and at two of three tile sizes it
    ends up net negative.

    This is the mechanism behind the "Progressive Intensity Hypothesis"
    (arXiv:2603.18426) that puts quantization BEFORE pruning, against Spec v6's
    `prune_then_quantize` default.  The fix is quantization-aware compensation
    (SparseGPT's joint mode, OBR) rather than the two run back to back, and it
    is why the order ablation moved from M3 to M2 in plan section H4.
    """
    def err(compensate: bool, quantize: bool, ldlq: bool = False) -> float:
        return M.run_config(
            big_problem, budget_bits=1.5, tile_size=8, compensate=compensate,
            rotate_axis="index", quantize=quantize, ldlq=ldlq, seed=0,
        )["rel_output_error"]

    gain_alone = err(True, False) / err(False, False)
    gain_plain = err(True, True) / err(False, True)
    gain_ldlq = err(True, True, ldlq=True) / err(False, True, ldlq=True)

    assert gain_alone < 0.85, "compensation must help the pruning objective"
    assert gain_plain > gain_alone + 0.15, (
        "plain rounding should take most of the compensation gain back; "
        f"got {gain_alone:.3f} -> {gain_plain:.3f}"
    )
    # ...and LDLQ hands it back: measured -5.8% (T=4) and -13.9% (T=8) against
    # +2.5% / +7.5% with plain rounding.  Pushing quantization error into the
    # Hessian's cheap directions is what stops the two steps fighting.
    assert gain_ldlq < 1.0, "with LDLQ, compensation should help again"
    assert gain_ldlq < gain_plain


def test_rotation_pays_off_only_with_hessian_aware_rounding(big_problem):
    """Rotation's benefit is entirely conditional on LDLQ -- the sign flips.

    Measured at B=1.5 on the 64x128 fixture, rotation's effect on the
    activation-weighted error:

        T=4    plain +2.6%    LDLQ -29.5%
        T=8    plain +4.6%    LDLQ -23.2%
        T=16   plain +0.1%    LDLQ -31.0%
        T=32   plain +3.9%    LDLQ -27.0%

    An RHT makes the quantization error isotropic, which is the wrong shape
    unless the Hessian is isotropic too.  Rounding against the rotated
    sub-Hessian is what turns the rotation from a cost into a gain -- and it is
    what licenses treating rotation as a force that pushes T up (plan C / I3).

    Ratios are taken WITHIN each rounding mode, since `ldlq=True` aligns the
    survivor count to 8 and therefore sits at a slightly different density.
    """
    def err(ldlq: bool, rotate: str | None) -> float:
        return M.run_config(
            big_problem, budget_bits=1.5, tile_size=8, compensate=True,
            rotate_axis=rotate, ldlq=ldlq, seed=0,
        )["rel_output_error"]

    plain_ratio = err(False, "index") / err(False, None)
    ldlq_ratio = err(True, "index") / err(True, None)

    assert plain_ratio > 1.0, "without LDLQ the rotation should not pay"
    assert ldlq_ratio < 0.9, "with LDLQ the rotation should clearly pay"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_gate_run_produces_a_complete_report(problem):
    run = M.GateRun(budgets=(1.5,), tiles=(1, 4, Tl.MAX_TILE), seeds=(0, 1))
    out = run.run(problem)
    assert out["meta"]["survivor_quantizer"] == "E8P"
    assert out["meta"]["vq_bits"] == 2.0
    assert out["wall"]["bits_realized"] == 2.0
    blk = out["budgets"]["1.5"]
    assert blk["live"]
    assert blk["gate_a"]["verdict"] in ("pass", "fail")
    assert blk["gate_b"]["verdict"] in ("interior", "edge", "undetermined")
    assert len(blk["records"]) == 3 * 2


def test_cli_refuses_to_pretend_it_can_load_a_model():
    assert M.main([]) == 2


# --------------------------------------------------------------------------- #
# T* as a set, not a point
# --------------------------------------------------------------------------- #

def test_t_star_set_collapses_to_one_tile_when_the_optimum_is_sharp():
    """A sharp optimum earns a point estimate."""
    tiles = {1: [0.50] * 8, 2: [0.44] * 8, 4: [0.36] * 8, 8: [0.20] * 8,
             16: [0.36] * 8, 32: [0.45] * 8, Tl.MAX_TILE: [0.52] * 8}
    out = M.t_star_set(_records(tiles))
    assert out["t_star"] == 8
    assert out["set"] == [8]


def test_t_star_set_keeps_every_tile_a_flat_interior_cannot_separate():
    """The case the power analysis says to expect: a flat bottom.

    T=4, 8 and 16 differ by less than the noise, so the argmin among them is
    close to arbitrary and the honest report is all three.
    """
    g = torch.Generator().manual_seed(4)
    base = {1: 0.50, 2: 0.42, 4: 0.301, 8: 0.300, 16: 0.302, 32: 0.44,
            Tl.MAX_TILE: 0.52}
    tiles = {t: (v + 0.02 * torch.randn(8, generator=g)).tolist()
             for t, v in base.items()}
    out = M.t_star_set(_records(tiles))
    assert set(out["set"]) >= {4, 8, 16}
    assert 2 not in out["set"] or 32 not in out["set"] or len(out["set"]) >= 3


def test_t_star_set_never_reports_an_edge():
    """The edges are what Gate B tests AGAINST; they are not granularity
    candidates and must never appear in the set."""
    tiles = {1: [0.10] * 6, 4: [0.40] * 6, 8: [0.41] * 6,
             Tl.MAX_TILE: [0.11] * 6}
    out = M.t_star_set(_records(tiles))
    assert 1 not in out["set"] and Tl.MAX_TILE not in out["set"]
    assert out["t_star"] in (4, 8)


def test_t_star_set_handles_a_grid_with_no_interior():
    out = M.t_star_set(_records({1: [0.3] * 5, Tl.MAX_TILE: [0.4] * 5}))
    assert out["t_star"] is None and out["set"] == []


def test_the_driver_reports_the_set_alongside_the_verdict():
    tiles = {1: [0.50] * 6, 2: [0.44] * 6, 4: [0.36] * 6, 8: [0.20] * 6,
             16: [0.36] * 6, 32: [0.45] * 6, Tl.MAX_TILE: [0.52] * 6}
    recs = _records(tiles)
    assert M.gate_b(recs)["t_star"] == M.t_star_set(recs)["t_star"]


# --------------------------------------------------------------------------- #
# Which axis the draws are over
# --------------------------------------------------------------------------- #

def test_a_list_of_problems_is_a_calibration_draw_axis(problem):
    """Gate B's intervals are over calibration draws, so the driver has to
    replicate over new DATA when it is given any."""
    from calibrate import synthetic_problem

    probs = [synthetic_problem(problem.n_out, problem.n_in, seed=d)
             for d in range(3)]
    out = M.GateRun(budgets=(1.5,), tiles=(1, 4, Tl.MAX_TILE)).run(probs)
    assert out["meta"]["draw_axis"] == "calibration"
    assert out["meta"]["n_draws"] == 3
    assert len(out["budgets"]["1.5"]["records"]) == 3 * 3
    assert out["budgets"]["1.5"]["gate_b"]["draw_axis"] == "calibration"


def test_a_single_problem_falls_back_to_the_rotation_seed_and_says_so(problem):
    """The fallback is legitimate but weaker -- rotation-seed noise measured at
    roughly half the calibration noise -- so it must never pass unlabelled."""
    out = M.GateRun(budgets=(1.5,), tiles=(1, 4, Tl.MAX_TILE),
                    seeds=(0, 1)).run(problem)
    assert out["meta"]["draw_axis"] == "rotation_seed"
    assert out["meta"]["n_draws"] == 2
    assert all(r["draw_axis"] == "rotation_seed"
               for r in out["budgets"]["1.5"]["records"] if "skipped" not in r)


def test_calibration_draws_actually_differ(problem):
    """Guard against the draws collapsing to identical runs: if every draw gave
    the same number, the CIs would be zero-width and Gate B would pass on
    anything."""
    from calibrate import synthetic_problem

    probs = [synthetic_problem(problem.n_out, problem.n_in, seed=d)
             for d in range(3)]
    out = M.GateRun(budgets=(1.5,), tiles=(4,)).run(probs)
    errs = [r["rel_output_error"] for r in out["budgets"]["1.5"]["records"]]
    assert len(set(errs)) == 3


def test_gate_run_rejects_an_empty_draw_list():
    with pytest.raises(ValueError):
        M.GateRun().run([])


def test_tile_hessian_stream_matches_the_stacked_form(problem):
    """The streaming path is what real layers use, so it has to agree with the
    stacked one tile for tile -- including the rotation into the block's basis."""
    import compact as C
    import prune as P
    import rotation as R

    pruned = P.prune(problem.W, axis="B", tile_size=4, density=0.625,
                     metric="wanda", act_norm=problem.act_norm, H=problem.H,
                     compensate=True, align=8)
    cw = C.compact(pruned.W, pruned.mask)
    _, Qm = R.rotate(cw, axis="index", seed=0)

    for Q_arg in (None, Qm):
        stacked = M.tile_hessians(problem, cw, Q_arg)
        stream = M.tile_hessian_stream(problem, cw, Q_arg)
        for t in range(stacked.shape[0]):
            assert torch.equal(stacked[t], stream(t))


def test_streaming_does_not_change_what_run_config_reports(problem):
    """run_config now streams; the numbers it produces must be the ones the
    stacked path produced, or every measurement before today is orphaned."""
    import compact as C
    import prune as P
    import quantize as Qz
    import rotation as R

    r = M.run_config(problem, budget_bits=1.5, tile_size=4)
    pruned = P.prune(problem.W, axis="B", tile_size=4,
                     density=r["density_requested"], metric="wanda",
                     act_norm=problem.act_norm, H=problem.H, compensate=True,
                     align=Qz.E8P_DIM)
    cw = C.compact(pruned.W, pruned.mask)
    rot, Qm = R.rotate(cw, axis="index", seed=0)
    qb = Qz.ldlq_quantize_blocks(rot.blocks, M.tile_hessians(problem, cw, Qm))
    W_hat = C.scatter(R.unrotate(rot.with_blocks(qb.values), Qm, axis="index"))
    assert problem.output_error(W_hat) == pytest.approx(r["rel_output_error"],
                                                        rel=1e-12)


# --------------------------------------------------------------------------- #
# THE ROTATION'S KRONECKER STRUCTURE
# --------------------------------------------------------------------------- #

def test_rotate_kron_is_the_same_rotation_not_a_different_one(problem):
    """`rotate_kron` changes the ARITHMETIC of `Q H Q^T`, nothing else.

    So the two arms have to land within float's own error of each other.  They
    are not bit-identical -- the association order differs -- which is why this
    is a caller's choice priced in `experiments/m0_rotation_value.py` rather
    than a default.  What must never happen is a real divergence: that would
    mean the factors are not the rotation the blocks were rotated by, and no
    other test would catch it because both answers look reasonable.
    """
    for t in (2, 4, Tl.MAX_TILE):
        dense = M.run_config(problem, budget_bits=1.5, tile_size=t)
        kron = M.run_config(problem, budget_bits=1.5, tile_size=t,
                            rotate_kron=True)
        if "skipped" in dense:
            continue
        assert dense["rotate_kron"] is False and kron["rotate_kron"] is True
        assert kron["rel_output_error"] == pytest.approx(
            dense["rel_output_error"], rel=1e-6)


def test_rotate_kron_refuses_a_rotation_that_is_not_the_kronecker_one(problem):
    """A block-diagonal rotation is a different matrix, and its factors are per
    block.  Silently using the full-width factors would rotate the Hessian by
    something the blocks were never rotated by."""
    with pytest.raises(ValueError, match="block-diagonal one is a different"):
        M.run_config(problem, budget_bits=1.5, tile_size=4,
                     rotate_kron=True, rotate_block=8)
    with pytest.raises(ValueError, match="full index-axis rotation"):
        M.run_config(problem, budget_bits=1.5, tile_size=4,
                     rotate_kron=True, rotate_axis="line")
