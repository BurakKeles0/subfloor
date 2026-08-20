"""Spec v6 section 3.4 acceptance tests, plus the v6 audit findings.

Every assertion compares `accounting.py` (general dispatch, floats) against
`golden.py` (closed-form algebra, exact rationals).  Two independent routes.

Nothing in this file may contain a hand-typed decimal constant for a quantity
that can be derived.  That rule is itself tested -- see
`test_spec_v6_errata_are_not_reproduced`.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

import pytest

import accounting as A
import golden as G

TOL = 1e-12
N = G.N_IDX_FFN


def approx(x: F | float) -> pytest.approx:
    return pytest.approx(float(x), abs=TOL)


# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("wb, over_128", [(4, 532), (3, 403), (2, 274)])
def test_weight_cost(wb, over_128):
    """W: 4-bit = 532/128, 3-bit = 403/128, 2-bit = 274/128."""
    assert A.weight_cost(wb) == approx(F(over_128, 128))
    assert G.W_OVER_128[wb] == over_128


def test_anchors_are_full_dense_costs():
    """Kural 1 -- never anchor to a round number."""
    assert A.anchor_budget_to("dense", 3) == approx(G.ANCHOR_1)
    assert A.anchor_budget_to("dense", 2) == approx(G.ANCHOR_2)
    assert float(G.ANCHOR_1) == 3.1484375
    assert float(G.ANCHOR_2) == 2.140625


def test_q_overhead_convention_flips_the_2_4_comparison():
    """Spec v6 section 3.1: with q_over NOT scaling, 2:4 @ 4-bit lands ABOVE
    dense 3-bit; if it scaled it would land below.  Print this in every table."""
    assert A.Q_OVERHEAD_SCALES_WITH_DENSITY["nm"] is False
    assert "vq" not in A.Q_OVERHEAD_SCALES_WITH_DENSITY

    got = A.bits_per_position("nm", None, 4, N, nm=(2, 4))
    assert got == approx(G.NM_2_4_AT_4BIT)
    assert got > float(G.ANCHOR_1)
    assert float(G.NM_2_4_AT_4BIT_IF_SCALED) < float(G.ANCHOR_1)


# --------------------------------------------------------------------------- #
# The identity  (Spec v6 section 3.4, with its validity domain asserted)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("budget", [F(403, 128), F(274, 128), F(7, 4), F(3, 2)])
@pytest.mark.parametrize("wb", [2, 3, 4])
@pytest.mark.parametrize("tile", [2, 4, 8, 16, 32])
def test_identity_holds_in_the_bitmap_regime(budget, wb, tile):
    """d(T) - d(1) = (1 - 1/T) / W, independent of B.

    Spec v6 section 7, trap 1: the identity is asserted only where it is valid.
    """
    b = float(budget)
    assert A.in_bitmap_regime(b, wb, N, tile_size=1)
    assert A.in_bitmap_regime(b, wb, N, tile_size=tile)

    d1 = A.density_for_budget("unstructured", b, wb, N, tile_size=1)
    dt = A.density_for_budget("tile", b, wb, N, tile_size=tile)
    if d1 is None or dt is None:
        pytest.skip(f"d > 1 at B={b}, wb={wb}: budget unreachable, not a bug")

    assert dt - d1 == approx(G.identity(tile, wb))
    assert A.tile_density_advantage(tile, wb) == approx(G.identity(tile, wb))


@pytest.mark.parametrize("wb, expected", list(G.IDENTITY_T16.items()))
def test_identity_t16_golden_values(wb, expected):
    """T=16: 4-bit -> 120/532, 3-bit -> 120/403, 2-bit -> 120/274.

    The leverage GROWS as wb falls (0.2256 -> 0.2978 -> 0.4380).
    """
    assert A.tile_density_advantage(16, wb) == approx(expected)


@pytest.mark.parametrize("tile, frac", [(2, F(1, 2)), (4, F(3, 4)), (8, F(7, 8)),
                                        (16, F(15, 16)), (32, F(31, 32))])
def test_identity_fraction_of_reachable_gain(tile, frac):
    """[d(T)-d(1)] / [d(inf)-d(1)] = 1 - 1/T, free of both B and W.

    This is the ONLY justification for the T grid (Spec v6 section 5.3): T=4
    already buys 75% of the reachable gain at a far lower constraint cost, which
    is why M1's {1,16,max} would have been a false-stop.
    """
    for wb in (2, 3, 4):
        ratio = A.tile_density_advantage(tile, wb) / A.tile_density_advantage("max", wb)
        assert ratio == approx(frac)
        assert G.identity_fraction_of_max(tile) == frac


# --------------------------------------------------------------------------- #
# M1 tables  (Spec v6 section 5.2)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tile", [1, 4, 16, "max"])
def test_anchor_1_table(tile):
    """AUDIT A1: the spec's anchor-1 table had two wrong cells."""
    d = A.density_for_budget(
        "unstructured" if tile == 1 else "tile", float(G.ANCHOR_1), 4, N,
        tile_size=tile,
    )
    assert d == approx(G.ANCHOR_1_DENSITIES[tile])
    assert d == approx(G.ANCHOR_1_AS_OVER_532[tile])


@pytest.mark.parametrize("tile", [1, 4, 16, "max"])
def test_anchor_2_table(tile):
    d = A.density_for_budget(
        "unstructured" if tile == 1 else "tile", float(G.ANCHOR_2), 4, N,
        tile_size=tile,
    )
    assert d == approx(G.ANCHOR_2_DENSITIES[tile])
    assert d == approx(G.ANCHOR_2_AS_OVER_532[tile])


def test_anchor_2_tile16_is_exactly_one_half():
    """The cleanest cell in the spec: 266/532."""
    d = A.density_for_budget("tile", float(G.ANCHOR_2), 4, N, tile_size=16)
    assert d == pytest.approx(0.5, abs=TOL)


def test_nm_offsets():
    """Kural 2: fixed-density schemes carry a signed offset, flagged past 1%."""
    at4 = A.bits_per_position("nm", None, 4, N, nm=(2, 4))
    at2 = A.bits_per_position("nm", None, 2, N, nm=(2, 4))
    assert at4 == approx(G.NM_2_4_AT_4BIT)
    assert at2 == approx(G.ANCHOR_2)                     # offset exactly zero

    offset_pct = (at4 - float(G.ANCHOR_1)) / float(G.ANCHOR_1)
    assert offset_pct == approx(G.OFFSET_NM_2_4_AT_4BIT)
    assert offset_pct == pytest.approx(0.00248139, abs=1e-8)
    assert abs(offset_pct) < A.OFFSET_FLAG_THRESHOLD     # 0.248% -> not flagged


# --------------------------------------------------------------------------- #
# B* wall  (Spec v6 section 0.3.1)
# --------------------------------------------------------------------------- #

def test_b_star_values():
    assert A.b_star(4, 11008) == approx(G.b_star(4, 11008))
    assert A.b_star(4, 4096) == approx(G.b_star(4, 4096))
    assert A.b_star(4, 4096) == pytest.approx(1 + 4.15625 / 12, abs=TOL)
    assert A.d_star(11008) == approx(G.d_star(11008))
    assert A.b_star(4, 11008, tile_size="max") is None    # no index, no crossover


def test_b_star_continuity():
    """At B* the bitmap and list branches meet exactly at d = 1/log2(n_idx)."""
    bs = A.b_star(4, N)
    d = A.density_for_budget("unstructured", bs, 4, N, tile_size=1)
    assert d == pytest.approx(A.d_star(N), abs=1e-9)

    adv = A.density_for_budget("tile", bs, 4, N, tile_size=16) - d
    assert adv == approx(G.identity(16, 4))               # advantage fully preserved


@pytest.mark.parametrize("budget", [1.20, 1.00])
def test_advantage_erodes_below_b_star(budget):
    """AUDIT A3: the spec's erosion table was off in the 6th digit."""
    assert budget < A.b_star(4, N)
    d_unstr = A.density_for_budget("unstructured", budget, 4, N, tile_size=1)
    d_tile = A.density_for_budget("tile", budget, 4, N, tile_size=16)
    adv = d_tile - d_unstr
    assert adv == approx(G.advantage_below_b_star(budget, 4, N))
    assert adv < float(G.identity(16, 4))                 # strictly eroded


def test_tile_never_enters_the_list_regime_in_the_planned_band():
    """Scope note: min(bitmap, list) is a correctness fix and a finding, not a
    change to the experiment design.  Do not over-engineer around it."""
    for tile in (2, 4, 8, 16, 32):
        assert A.b_star(4, N, tile_size=tile) < 1.0
        d = A.density_for_budget("tile", 1.0, 4, N, tile_size=tile)
        assert d * math.log2(N) > 1.0                     # still bitmap at B=1.0


# --------------------------------------------------------------------------- #
# Index model  (Spec v6 section 3.2)
# --------------------------------------------------------------------------- #

def test_index_cascade():
    assert A.index_bits("unstructured", 0.50, N) == pytest.approx(1.0, abs=TOL)
    assert A.index_bits("unstructured", 0.05, N) == approx(0.05 * G.L(N))
    assert A.index_bits("tile", 0.50, N, tile_size=16) == pytest.approx(1 / 16, abs=TOL)
    assert A.index_bits("structured", 0.5, N) == 0.0
    assert A.index_bits("dense", 1.0, N) == 0.0


def test_index_is_layer_independent_in_the_bitmap_regime():
    """n_idx only matters below B*; in the primary band the index is 1/T."""
    for n_idx in (4096, 11008, 13824):
        assert A.density_for_budget(
            "tile", float(G.ANCHOR_2), 4, n_idx, tile_size=16
        ) == pytest.approx(0.5, abs=TOL)


def test_corrected_floor_and_list_regime_density():
    """AUDIT A2 + Spec v6 section 3.4's two corrected tests.

    'Unstructured cannot go below 1.0 bit' is a claim about the BITMAP, not
    about unstructured sparsity (section 7, trap 5).
    """
    assert A.scheme_floor("unstructured", 4, N) == 0.0
    assert A.scheme_floor("tile", 4, N, tile_size=16) == 0.0

    d = A.density_for_budget("unstructured", 0.60, 4, N, tile_size=1)
    assert d is not None
    assert d == approx(G.d_list(0.60, 4, N))
    assert d == pytest.approx(0.0341248, abs=1e-7)


def test_entropy_index_is_a_bound_not_a_scheme():
    """H(d) is reachable only by giving up random access."""
    d = 0.12
    assert A.index_bits("unstructured", d, N, index_model="info_theoretic") == \
        pytest.approx(A.entropy_bits(d), abs=TOL)
    assert A.entropy_bits(d) < A.index_bits("unstructured", d, N)
    assert A.entropy_bits(0.0) == 0.0 and A.entropy_bits(1.0) == 0.0


def test_info_theoretic_budget_inversion_round_trips():
    """The non-analytic index model still inverts (bisection fallback)."""
    d = A.density_for_budget(
        "tile", float(G.ANCHOR_2), 4, N, tile_size=16, index_model="info_theoretic"
    )
    back = A.bits_per_position(
        "tile", d, 4, N, tile_size=16, index_model="info_theoretic"
    )
    assert back == pytest.approx(float(G.ANCHOR_2), abs=1e-9)


# --------------------------------------------------------------------------- #
# AUDIT B1: a block-local fixed-count index keeps random access AND beats bitmap
# --------------------------------------------------------------------------- #

def test_block_local_index_beats_the_bitmap_at_low_density():
    """The headline advantage is quoted against a bitmap.  A fixed-count block
    code is decodable in O(1) per block -- random access survives -- and costs
    less than 1.0 bit well inside the exploratory band.

    Consequence: section 3.2's 'H(d) is not reachable' paragraph is too strong,
    and the N:M row's combinatorial entry belongs in the `practical` column.
    """
    assert A.nm_index_bits(2, 4) == 1.0                        # bitmap-equivalent
    assert A.nm_index_bits(1, 4) == 0.5                        # cheaper than bitmap
    assert A.nm_index_bits(2, 8, packing="combinatorial") == \
        approx(G.nm_index_combinatorial(2, 8))
    assert A.nm_index_bits(2, 8, packing="combinatorial") < 1.0

    # ...but it does NOT bite in the primary band: d * log2(M) < 1 needs
    # d < 1/log2(M), and the primary band runs d = 0.27 - 0.76.
    for m in (16, 32, 128):
        assert 1.0 / math.log2(m) < 0.274


def test_primary_band_results_are_unaffected_by_b1():
    """Bitmap is optimal at every density M1/M2 actually visit."""
    for budget in (float(G.ANCHOR_1), float(G.ANCHOR_2), 1.75):
        for tile in (1, 4, 16):
            scheme = "unstructured" if tile == 1 else "tile"
            d = A.density_for_budget(scheme, budget, 4, N, tile_size=tile)
            if d is None:
                continue
            assert d * math.log2(N) >= 1.0


# --------------------------------------------------------------------------- #
# Live band  (Spec v6 section 3.5)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("wb", [2, 3, 4])
def test_live_band(wb):
    lo, hi = A.live_band(wb)
    g_lo, g_hi = G.LIVE_BANDS[wb]
    assert lo == approx(g_lo)
    assert hi == approx(g_hi)


def test_wb2_is_dead_across_the_whole_primary_band():
    """The degenerate cell the filter exists for: at anchor 2, wb=2 has W == B,
    so d(T=max) is exactly 1.0.  'T=max wins' there means '2-bit quantization
    wins', not 'structured pruning wins' (section 7, trap 14)."""
    cfg = A.Config(scheme="tile", weight_bits=2, n_idx=N, tile_size=16,
                   budget_bits=float(G.ANCHOR_2))
    assert not A.is_live(cfg)

    diag = A.live_diagnostics(cfg)
    assert diag["d_coarse"] == pytest.approx(1.0, abs=TOL)
    assert diag["d_fine"] == approx(G.DEGENERATE_WB2_ANCHOR2[1])
    assert "coarse end already dense" in diag["reasons"][0]

    lo, hi = A.live_band(2)
    for b in (2.0, 2.1, 2.2, 2.3):
        assert not (lo < b < hi)


@pytest.mark.parametrize("wb", [3, 4])
def test_primary_band_is_live_for_wb_3_and_4(wb):
    for b in (2.0, 2.141, 2.3):
        cfg = A.Config(scheme="tile", weight_bits=wb, n_idx=N, tile_size=16,
                       budget_bits=b)
        assert A.is_live(cfg)


# --------------------------------------------------------------------------- #
# AUDIT D1: is_live is a granularity question, not a reportability question
# --------------------------------------------------------------------------- #

def test_aqlm_survivor_densities():
    assert float(G.AQLM_VQ_BITS) == pytest.approx(2.186, abs=1e-6)
    for tile in (1, 16, "max"):
        scheme = "unstructured" if tile == 1 else "tile"
        d = A.density_for_budget(
            scheme, float(G.ANCHOR_2), None, N,
            tile_size=tile, vq_bits=float(G.AQLM_VQ_BITS),
        )
        assert d == approx(G.AQLM_ANCHOR_2[tile])

    d1 = A.density_for_budget("unstructured", float(G.ANCHOR_2), None, N,
                              tile_size=1, vq_bits=float(G.AQLM_VQ_BITS))
    d16 = A.density_for_budget("tile", float(G.ANCHOR_2), None, N,
                               tile_size=16, vq_bits=float(G.AQLM_VQ_BITS))
    assert d16 - d1 == approx(G.AQLM_IDENTITY_T16)       # identity holds for VQ too


def test_aqlm_survivor_fails_is_live_but_is_still_a_gate_a_row():
    """AUDIT D1: Spec v6 calls AQLM-survivor 'Gate A's strongest candidate', yet
    its d(T=max) = 0.979 fails the section 3.5 filter.  Both are true -- the
    filter answers a different question.  Do not silently drop the row."""
    cfg = A.Config(scheme="tile", weight_bits=None, vq_bits=float(G.AQLM_VQ_BITS),
                   n_idx=N, tile_size=16, budget_bits=float(G.ANCHOR_2),
                   label="AQLM-survivor + tile-16")
    diag = A.live_diagnostics(cfg)

    assert diag["d_coarse"] == pytest.approx(0.979243, abs=1e-6)
    assert diag["live"] is False                          # not a granularity probe
    assert diag["d_fine"] > A.LIVE_DENSITY_MIN            # fine end is genuinely sparse
    assert "coarse end already dense" in diag["reasons"][0]

    # The row is still constructible and reportable at its own cost.
    assert cfg.bits_per_position() == pytest.approx(float(G.ANCHOR_2), abs=1e-9)


def test_gate_a_table_can_opt_out_of_the_live_filter():
    rows_filtered = A.budget_matched_grid(
        float(G.ANCHOR_2), n_idx=N, vq_bits_grid=(float(G.AQLM_VQ_BITS),),
        apply_live_filter=True,
    )
    rows_all = A.budget_matched_grid(
        float(G.ANCHOR_2), n_idx=N, vq_bits_grid=(float(G.AQLM_VQ_BITS),),
        apply_live_filter=False,
    )
    assert len(rows_all) > len(rows_filtered)
    assert any(r["vq_bits"] is not None for r in rows_all)
    assert all(r["live"] for r in rows_filtered)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scheme", ["nm", "vnm", "vq_dense", "dense"])
def test_density_for_budget_raises_on_fixed_density_schemes(scheme):
    """Spec v6 section 7, trap 21.  Note the signature takes no `nm`: there is
    no density to solve for, so the argument would be meaningless."""
    with pytest.raises(ValueError, match="fixed density"):
        A.density_for_budget(scheme, 2.140625, 4, N)


def test_vnm_requires_the_full_triple():
    with pytest.raises(ValueError, match=r"vnm=\(V, N, M\)"):
        A.index_bits("vnm", 0.25, N, nm=(2, 8))


# --------------------------------------------------------------------------- #
# V:N:M  (VENOM, arXiv:2310.02065) -- the M0 exit condition, now filled in
# --------------------------------------------------------------------------- #

def test_vnm_matches_the_two_metadata_structures():
    """m-indices at 2 bits per nonzero, plus a column-loc entry per selected
    column amortized over the V rows that share it:

        2N/M  +  4*ceil(log2 M)/(V*M)
    """
    v, n, m = 64, 2, 8
    want = 2 * n / m + 4 * math.ceil(math.log2(m)) / (v * m)
    assert A.vnm_index_bits(v, n, m) == pytest.approx(want, abs=TOL)
    assert A.vnm_index_bits(64, 2, 8) == pytest.approx(0.5234375, abs=TOL)
    assert A.vnm_index_bits(64, 2, 16) == pytest.approx(0.265625, abs=TOL)


def test_vnm_reduces_to_plain_2_4():
    """At M = 4 the vector stage picks 4 of 4 -- no information -- so the cost
    collapses to the native 2:4 metadata, exactly 1.0 bit."""
    assert A.vnm_index_bits(64, 2, 4, packing="combinatorial") == 1.0
    assert A.nm_index_bits(2, 4) == 1.0


def test_the_V_in_vnm_is_a_row_tile():
    """THE structural point (spec v7 section 0.4).

    VENOM's column-loc is one column selection shared by V rows -- which is this
    project's Axis B at T=V.  So its index amortizes as 1/V, the same mechanism
    as our 1/T, and VENOM is far closer prior work than Spec v6 credited.

    The m-indices term does NOT amortize, because it is per-nonzero: that is
    what a fixed 2:4 pattern inside the block costs, and it is the part our free
    within-tile density removes.
    """
    n, m = 2, 8
    base = 2 * n / m                                   # the un-amortizable part
    costs = {v: A.vnm_index_bits(v, n, m) for v in (1, 2, 4, 8, 16, 32, 64)}

    assert all(a > b for a, b in zip(costs.values(), list(costs.values())[1:]))
    for v, c in costs.items():
        assert (c - base) * v == pytest.approx(
            (costs[1] - base), abs=1e-9
        ), "the column-loc term must fall exactly as 1/V"
    assert min(costs.values()) > base                  # never reaches the floor


def test_vnm_beats_a_bitmap_at_the_same_density():
    """A concrete instance of section 3.2's correction: a bitmap is not the
    floor for a random-accessible index.  V:2:16 sits at d=0.125 and costs
    0.266 bits, against a bitmap's 1.0."""
    v, n, m = 64, 2, 16
    density = n / m
    assert density == 0.125
    assert A.vnm_index_bits(v, n, m) < A.index_bits("unstructured", density, N)


def test_vnm_bits_per_position_and_density():
    """Density is fixed by N/M, and q_over does not scale with it."""
    got = A.bits_per_position("vnm", None, 4, N, vnm=(64, 2, 8))
    want = 0.25 * 4 + A.q_overhead(4) + A.vnm_index_bits(64, 2, 8)
    assert got == pytest.approx(want, abs=TOL)
    assert A.Q_OVERHEAD_SCALES_WITH_DENSITY["vnm"] is False

    with pytest.raises(ValueError, match="implies density"):
        A.bits_per_position("vnm", 0.4, 4, N, vnm=(64, 2, 8))


def test_vnm_validates():
    with pytest.raises(ValueError, match="0 < N <= 4"):
        A.vnm_index_bits(64, 5, 8)
    with pytest.raises(ValueError, match="multiple of 4"):
        A.vnm_index_bits(64, 2, 6)
    with pytest.raises(ValueError, match="V must be positive"):
        A.vnm_index_bits(0, 2, 8)
    with pytest.raises(ValueError, match="unknown packing"):
        A.vnm_index_bits(64, 2, 8, packing="nope")


def test_dense_rejects_a_non_unit_density():
    with pytest.raises(ValueError, match="density=1.0"):
        A.bits_per_position("dense", 0.5, 4, N)


def test_nm_rejects_an_inconsistent_density():
    with pytest.raises(ValueError, match="implies density"):
        A.bits_per_position("nm", 0.4, 4, N, nm=(2, 4))


def test_weight_bits_and_vq_bits_are_mutually_exclusive():
    with pytest.raises(ValueError, match="not both"):
        A.bits_per_position("unstructured", 0.5, 4, N, tile_size=1, vq_bits=2.186)


def test_unreachable_budgets_return_none():
    """wb=2 cannot reach anchor 1 at any density: d would exceed 1."""
    assert A.density_for_budget("unstructured", float(G.ANCHOR_1), 2, N,
                                tile_size=1) is None
    assert A.density_for_budget("tile", float(G.ANCHOR_1), 2, N,
                                tile_size=16) is None


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

def test_budget_matched_grid_reproduces_the_anchor_1_family():
    rows = A.budget_matched_grid(
        float(G.ANCHOR_1), n_idx=N, weight_bits_grid=(4,),
        tile_grid=(4, 16), nm_variants=((2, 4),), apply_live_filter=True,
    )
    by_label = {r["label"]: r for r in rows}
    assert by_label["4-bit + unstructured"]["density"] == \
        approx(G.ANCHOR_1_DENSITIES[1])
    assert by_label["4-bit + tile-4"]["density"] == approx(G.ANCHOR_1_DENSITIES[4])
    assert by_label["4-bit + tile-16"]["density"] == approx(G.ANCHOR_1_DENSITIES[16])
    assert by_label["4-bit + T=max (structured)"]["density"] == \
        approx(G.ANCHOR_1_DENSITIES["max"])
    assert by_label["4-bit + 2:4"]["offset_pct"] == approx(G.OFFSET_NM_2_4_AT_4BIT)

    for r in rows:
        assert r["anchor"] == float(G.ANCHOR_1)
        assert r["n_idx"] == N
        assert r["in_bitmap_regime"]
        if r["scheme"] != "nm":
            assert r["bits_per_position"] == pytest.approx(float(G.ANCHOR_1), abs=1e-9)


def test_grid_drops_dead_cells_and_flags_large_offsets():
    rows = A.budget_matched_grid(float(G.ANCHOR_2), n_idx=N,
                                 weight_bits_grid=(2, 3, 4))
    assert all(r["weight_bits"] != 2 for r in rows), "wb=2 is dead at anchor 2"
    assert all(not r["flagged"] for r in rows)


# --------------------------------------------------------------------------- #
# Roofline  (Spec v6 section 0.6)
# --------------------------------------------------------------------------- #

def test_roofline_bytes():
    n_params = 45_100_000
    dense4 = A.Config(scheme="dense", weight_bits=4, n_idx=N)
    tile16 = A.Config(scheme="tile", weight_bits=4, n_idx=N, tile_size=16,
                      budget_bits=float(G.ANCHOR_2))
    assert A.roofline_bytes(dense4, n_params) == \
        math.ceil(n_params * float(G.W(4)) / 8)
    assert A.roofline_bytes(tile16, n_params) < A.roofline_bytes(dense4, n_params)


# --------------------------------------------------------------------------- #
# AUDIT A4: the 70B envelope
# --------------------------------------------------------------------------- #

def test_deployment_threshold_needs_a_stated_context_length():
    """Spec v6 section 0.1 assumes 4k and concludes '~2.2-2.3 bit'.  Its own line
    items give 2.46-2.65 at 4k.  The 2.2-2.3 figure belongs to 8k-16k.

    This matters: at 4k the PTQ floor (2.0-2.2) comfortably fits, and the
    motivation for going below it evaporates.
    """
    assert G.max_bits_per_position(24.0, 4096) == pytest.approx(2.645, abs=5e-3)
    assert G.max_bits_per_position(22.5, 4096) == pytest.approx(2.456, abs=5e-3)
    assert not (2.2 <= G.max_bits_per_position(22.5, 4096) <= 2.3)

    # The spec's stated threshold is recovered at 8k, not 4k.
    assert G.max_bits_per_position(22.5, 8192) == pytest.approx(2.299, abs=5e-3)
    assert 2.0 <= G.max_bits_per_position(23.0, 16384) <= 2.2

    # Spec section 0.1's own table, reproduced.
    assert G.LINEAR_GIB_PER_BIT == pytest.approx(7.969, abs=1e-3)
    assert G.EMBED_GIB == pytest.approx(0.98, abs=5e-3)
    assert G.fixed_load_gib(4096) == pytest.approx(2.93, abs=1e-2)


# --------------------------------------------------------------------------- #
# AUDIT C: RHT affordability is a function of T
# --------------------------------------------------------------------------- #

def test_rht_overhead_scales_as_log_over_t():
    """A frozen mask cannot be destroyed by a rotation, so RHT is legal on the
    compacted survivors -- but each tile owns a different column set and needs
    its own transform.  Cost relative to the GEMV is log2(d*n)/T: unaffordable
    at T=1, nearly free at T=max.

    This is a second force pushing T up that Delta = Q + tau does not contain.
    """
    d = 0.27
    r1 = G.rht_overhead_ratio(1, d, N)
    r16 = G.rht_overhead_ratio(16, d, N)
    r64 = G.rht_overhead_ratio(64, d, N)
    assert r1 > 10.0
    assert 0.5 < r16 < 1.0
    assert r64 < 0.25
    assert G.rht_overhead_ratio("max", d, N) == 0.0
    assert r1 / r16 == pytest.approx(16.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# The errata themselves
# --------------------------------------------------------------------------- #

def test_spec_v6_errata_are_not_reproduced():
    """Guard against someone 'fixing' the code to match the spec document.

    Each entry records a value Spec v6 printed that is arithmetically wrong.
    Delete a row only when the document itself has been corrected.
    """
    for name, e in G.SPEC_V6_ERRATA.items():
        assert e["spec"] != pytest.approx(e["true"], abs=TOL), (
            f"{name}: spec value {e['spec']} now agrees with the derivation -- "
            f"either the document was fixed (delete this row) or the derivation "
            f"regressed ({e['where']})"
        )

    assert A.density_for_budget("unstructured", float(G.ANCHOR_1), 4, N,
                                tile_size=1) is not None
    tile4 = A.density_for_budget("tile", float(G.ANCHOR_1), 4, N, tile_size=4)
    tmax = A.density_for_budget("structured", float(G.ANCHOR_1), 4, N)
    assert tile4 == approx(F(371, 532))
    assert tmax == approx(F(403, 532))
    assert tile4 != pytest.approx(0.696992, abs=1e-6)
    assert tmax != pytest.approx(0.757954, abs=1e-6)


def test_no_hardcoded_log2_in_accounting():
    """A3's root cause: a transcendental constant typed by hand."""
    src = (__import__("pathlib").Path(A.__file__)).read_text(encoding="utf-8")
    assert "13.4262" not in src
    assert "0.696992" not in src and "0.757954" not in src


# --------------------------------------------------------------------------- #
# E8P survivor branch  (plan section H2)
# --------------------------------------------------------------------------- #

VQ = float(G.E8P_VQ_BITS)


def test_e8p_live_band_is_entirely_below_two_bits():
    """Halving W moves the whole live band under the PTQ floor -- the regime the
    thesis is about, and where dense PTQ has no answer."""
    lo, hi = A.live_band(None, vq_bits=VQ)
    assert (lo, hi) == (approx(G.E8P_LIVE_BAND[0]), approx(G.E8P_LIVE_BAND[1]))
    assert (lo, hi) == (pytest.approx(1.4, abs=TOL), pytest.approx(1.8, abs=TOL))
    assert hi < 2.0


def test_e8p_and_gptq4_live_bands_do_not_overlap():
    """Structural consequence (plan H2): the survivor quantizer decides the
    budget regime, so the two families can never be budget-matched in a live
    cell.  M2's weight_bits axis is largely moot once survivors go to VQ.
    """
    e8p_lo, e8p_hi = A.live_band(None, vq_bits=VQ)
    g4_lo, g4_hi = A.live_band(4)
    assert e8p_hi < g4_lo, "bands must be disjoint"


@pytest.mark.parametrize("tile", [1, 2, 4, 8, 16, 32, "max"])
def test_e8p_density_table_at_1p5(tile):
    """Every density at B=1.5 is an exact dyadic rational."""
    scheme = {1: "unstructured", "max": "structured"}.get(tile, "tile")
    d = A.density_for_budget(scheme, 1.5, None, N, tile_size=tile, vq_bits=VQ)
    assert d == approx(G.E8P_AT_1P5[tile])
    assert G.E8P_AT_1P5[tile].denominator <= 64      # dyadic, no rounding


def test_e8p_doubles_the_leverage():
    """(1 - 1/T)/W with W halved: 15/32 against 30/133."""
    adv = A.tile_density_advantage(16, None, vq_bits=VQ)
    assert adv == approx(G.E8P_IDENTITY_T16)
    assert adv == pytest.approx(0.46875, abs=TOL)
    assert adv / A.tile_density_advantage(16, 4) == pytest.approx(
        float(G.W(4)) / VQ, abs=1e-9
    )


def test_e8p_b_star():
    assert A.b_star(None, N, vq_bits=VQ) == approx(G.b_star_vq(G.E8P_VQ_BITS, N))
    assert A.b_star(None, N, vq_bits=VQ) == pytest.approx(1.148962, abs=1e-6)


def test_old_anchor_2_is_unreachable_under_e8p():
    """Why the anchors had to move: at B=2.140625 the tile family runs past
    d=1, so the cell stops being a sparsity experiment at all."""
    for tile in (16, 32, "max"):
        scheme = "structured" if tile == "max" else "tile"
        assert A.density_for_budget(
            scheme, float(G.ANCHOR_2), None, N, tile_size=tile, vq_bits=VQ
        ) is None


@pytest.mark.parametrize("budget", [float(b) for b in G.E8P_BUDGETS])
def test_new_primary_band_is_live(budget):
    cfg = A.Config(scheme="tile", vq_bits=VQ, n_idx=N, tile_size=16,
                   budget_bits=budget)
    assert A.is_live(cfg)
    rows = A.budget_matched_grid(
        budget, n_idx=N, weight_bits_grid=(), vq_bits_grid=(VQ,)
    )
    assert rows, "grid must not be empty in the primary band"
    for r in rows:
        assert r["live"]
        assert r["offset"] == pytest.approx(0.0, abs=1e-9)
