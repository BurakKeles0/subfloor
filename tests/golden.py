"""Golden values for the accounting layer -- derived, never typed.

WHY THIS FILE IS NOT AN IMPORT OF `accounting`
----------------------------------------------
A golden file that calls the code under test proves nothing.  This module
therefore re-derives every constant along an INDEPENDENT route:

  * `accounting.py` uses a general dispatch over schemes with float arithmetic.
  * this module uses the closed-form algebra written straight out of Spec v6,
    in EXACT rational arithmetic (`fractions.Fraction`) wherever the quantity is
    rational.

Two routes, one answer.  A disagreement is a real bug in one of them -- most
usefully, it catches a wrong q_over convention, a wrong branch in the
min(bitmap, list) cascade, or a tile-amortization applied to the wrong term.

`Fraction` also removes the entire error class the v6 audit found: you cannot
mistype 0.696992 for 371/532 when the value is spelled as the division.  The
only float constants below are the ones that genuinely involve log2(n_idx), and
even those are written as expressions.

Spec v6 section references are given per block so a reviewer can check each
number against the document.
"""

from __future__ import annotations

import math
from fractions import Fraction as F

# --------------------------------------------------------------------------- #
# Conventions  (Spec v6 section 3.1)
# --------------------------------------------------------------------------- #

GROUP_SIZE = 128
SCALE_BITS = 16
N_IDX_FFN = 11008          # Llama-2-7B FFN intermediate size
N_IDX_ATTN = 4096          # Llama-2-7B hidden size


def W(wb: int) -> F:
    """W = wb + q_over, exact.  One FP16 scale + one wb-bit zero per group.

    4-bit -> 532/128, 3-bit -> 403/128, 2-bit -> 274/128.
    """
    return F(wb) + F(SCALE_BITS + wb, GROUP_SIZE)


#: W as the spec writes it: an integer over 128.
W_OVER_128 = {wb: W(wb) * GROUP_SIZE for wb in (2, 3, 4)}   # {4: 532, 3: 403, 2: 274}

#: Kural 1 -- budgets anchor to the FULL cost of a dense baseline, never a round
#: number (Spec v6 section 7, trap 6).
ANCHOR_1 = W(3)            # 403/128 = 3.1484375  -- dense 3-bit GPTQ
ANCHOR_2 = W(2)            # 274/128 = 2.140625   -- dense 2-bit VQ class

#: Spec v6 section 5.3.  Primary band is 2.0-2.3; below 1.75 is exploratory.
BUDGET_SWEEP = (ANCHOR_1, F(26, 10), ANCHOR_2, F(19, 10), F(7, 4), F(3, 2))

TILE_GRID = (1, 2, 4, 8, 16, 32)


# --------------------------------------------------------------------------- #
# Density under a bitmap index  (Spec v6 section 0.3)
# --------------------------------------------------------------------------- #

def d_bitmap(budget: F, wb: int, tile_size: int | str) -> F:
    """d(T) = (B - 1/T) / W, exact.

    Valid ONLY for B >= B*(T).  Spec v6 section 7, trap 1: never use the identity
    without asserting its domain.
    """
    inv_t = F(0) if tile_size == "max" else F(1, tile_size)
    return (budget - inv_t) / W(wb)


def identity(tile_size: int | str, wb: int) -> F:
    """d(T) - d(1) = (1 - 1/T) / W, exact and independent of the budget."""
    inv_t = F(0) if tile_size == "max" else F(1, tile_size)
    return (1 - inv_t) / W(wb)


def identity_fraction_of_max(tile_size: int | str) -> F:
    """[d(T) - d(1)] / [d(inf) - d(1)] = 1 - 1/T.

    Independent of BOTH the budget and W.  This is why the T grid is
    {1,2,4,8,16,32}: T=2 buys 50% of the reachable gain, T=4 buys 75%, T=8
    87.5%, T=16 93.75%, T=32 96.9% (Spec v6 section 5.3).
    """
    inv_t = F(0) if tile_size == "max" else F(1, tile_size)
    return 1 - inv_t


#: Spec v6 section 5.2, M1 anchor-1 table.  4-bit weights.
#:
#: ERRATUM (audit A1): the spec printed 0.696992 for tile-4 and 0.757954 for
#: T=max.  Both are wrong; the correct values are 371/532 and 403/532.  The
#: anchor-2 table in the spec is correct throughout.
ANCHOR_1_DENSITIES = {t: d_bitmap(ANCHOR_1, 4, t) for t in (*TILE_GRID, "max")}
ANCHOR_2_DENSITIES = {t: d_bitmap(ANCHOR_2, 4, t) for t in (*TILE_GRID, "max")}

#: The same tables spelled as the spec spells them, as a cross-check on the
#: division above.  If these disagree with ANCHOR_*_DENSITIES the arithmetic in
#: this file is itself wrong.
ANCHOR_1_AS_OVER_532 = {1: F(275, 532), 4: F(371, 532), 16: F(395, 532),
                        "max": F(403, 532)}
ANCHOR_2_AS_OVER_532 = {1: F(146, 532), 4: F(242, 532), 16: F(266, 532),
                        "max": F(274, 532)}

#: Spec v6 section 3.4, the identity test.  120/532 = 30/133 etc.
IDENTITY_T16 = {4: F(120, 532), 3: F(120, 403), 2: F(120, 274)}


# --------------------------------------------------------------------------- #
# The B* wall  (Spec v6 section 0.3.1)
# --------------------------------------------------------------------------- #
# These involve log2(n_idx) and are therefore irrational.  Written as
# expressions, never as decimals.
#
# ERRATUM (audit A3): the spec used log2(11008) = 13.4262102.  The true value is
# 13.4262648.  Everything derived from it in section 3.4 was off in the 6th-7th
# significant digit, which matters because those values are asserted to 1e-12.

def L(n_idx: int) -> float:
    """log2(n_idx) -- the width of one fixed-width index entry."""
    return math.log2(n_idx)


def b_star(wb: int, n_idx: int, tile_size: int | str = 1) -> float:
    """B*(T) = 1/T + W / log2(n_idx).

    The budget at which the bitmap and fixed-width-list branches meet.  At and
    above it the identity is exact; below it the unstructured index gets cheaper
    than a bitmap and the tile advantage erodes.
    """
    inv_t = 0.0 if tile_size == "max" else 1.0 / tile_size
    return inv_t + float(W(wb)) / L(n_idx)


def d_star(n_idx: int) -> float:
    """1 / log2(n_idx): the density at which a list index costs exactly 1 bit."""
    return 1.0 / L(n_idx)


def d_list(budget: float, wb: int, n_idx: int) -> float:
    """Unstructured density in the LIST regime: d = B / (W + log2(n_idx)).

    ERRATUM (audit A2 / spec section 3.4): the spec asserts
    density_for_budget('unstructured', 0.60, 4, 11008) == 0.0341271.
    The correct value is 0.0341248.
    """
    return budget / (float(W(wb)) + L(n_idx))


def advantage_below_b_star(budget: float, wb: int, n_idx: int,
                           tile_size: int = 16) -> float:
    """d(tile) - d(unstructured) when the unstructured branch has gone to a list.

    Above B* this equals `identity(T, wb)` exactly.  Below it, it shrinks --
    the closed-form wall that bounds how far the exploratory band means anything.
    """
    inv_t = 1.0 / tile_size
    return (budget - inv_t) / float(W(wb)) - d_list(budget, wb, n_idx)


# --------------------------------------------------------------------------- #
# N:M  (Spec v6 section 3.2)
# --------------------------------------------------------------------------- #

def nm_index_fixed_width(n: int, m: int) -> F:
    """ceil(log2(M)) * N / M, exact.  2:4 -> 1.0, 4:8 -> 1.5, 1:4 -> 0.5."""
    return F(math.ceil(math.log2(m))) * F(n, m)


def nm_index_combinatorial(n: int, m: int) -> float:
    """log2(C(M, N)) / M.

    AUDIT (plan section B1): this is a PRACTICAL encoding, not merely an
    information-theoretic bound.  One block of M positions holding exactly N
    survivors decodes in O(1) without touching any other block, so random access
    survives.  At d = 0.25 it costs 0.601 bits against the bitmap's 1.0 -- which
    is why the headline tile advantage must be quoted against the best
    random-accessible index, not against a bitmap strawman.
    """
    return math.log2(math.comb(m, n)) / m


def nm_bits(n: int, m: int, wb: int) -> F:
    """Full cost of an N:M config.  q_over does NOT scale with density here
    (Spec v6 section 3.1), which is what puts 2:4 @ 4-bit ABOVE dense 3-bit.

    2:4 @ 4-bit -> 101/32   = 3.15625   (offset +0.248% vs anchor 1)
    2:4 @ 2-bit -> 274/128  = 2.140625  (offset  0.000% vs anchor 2)
    """
    return F(n, m) * wb + F(SCALE_BITS + wb, GROUP_SIZE) + nm_index_fixed_width(n, m)


NM_2_4_AT_4BIT = nm_bits(2, 4, 4)          # 101/32
NM_2_4_AT_2BIT = nm_bits(2, 4, 2)          # == ANCHOR_2, exactly
OFFSET_NM_2_4_AT_4BIT = (NM_2_4_AT_4BIT - ANCHOR_1) / ANCHOR_1   # +0.248139%

#: If q_over DID scale with density, 2:4 @ 4-bit would land at 3.078125, i.e.
#: BELOW dense 3-bit -- the sign of the whole M1 comparison flips on this
#: convention.  It must be printed in every table (Spec v6 section 3.1).
NM_2_4_AT_4BIT_IF_SCALED = F(2, 4) * W(4) + nm_index_fixed_width(2, 4)


# --------------------------------------------------------------------------- #
# VQ  (Spec v6 section 3.2)
# --------------------------------------------------------------------------- #

def vq_bits(idx_bits: int, dim: int, entry_bits: int,
            weights_per_codebook: int) -> F:
    """idx_bits/dim + 2^idx_bits * dim * entry_bits / weights_per_codebook."""
    return F(idx_bits, dim) + F(
        2 ** idx_bits * dim * entry_bits, weights_per_codebook
    )


#: AQLM 1x16, dim=8, FP16 entries, amortized over a Llama-2-7B FFN block (45.1M).
#: = 2.0 + 0.186000177 = 2.186000177
#: The spec quotes 2.186 and requires the real value to be measured from the
#: checkpoint's file size before it anchors anything.
AQLM_VQ_BITS = vq_bits(16, 8, 16, 45_100_000)

#: AQLM-survivor at anchor 2.  Note d(T=max) = 0.979: this config FAILS the
#: section 3.5 live filter at the coarse edge while remaining a valid Gate A row
#: (audit D1).
AQLM_ANCHOR_2 = {
    t: (ANCHOR_2 - (F(0) if t == "max" else F(1, t))) / AQLM_VQ_BITS
    for t in (*TILE_GRID, "max")
}
AQLM_IDENTITY_T16 = (1 - F(1, 16)) / AQLM_VQ_BITS


# --------------------------------------------------------------------------- #
# Live band  (Spec v6 section 3.5)
# --------------------------------------------------------------------------- #

LIVE_D_MIN = F(1, 5)       # NOT Fraction(0.2) -- that is not 1/5 in binary
LIVE_D_MAX = F(9, 10)


def live_band(wb: int) -> tuple[F, F]:
    """(B_min, B_max) = (1 + 0.2*W, 0.9*W), exact.

      wb=4 -> (293/160,  1197/320)  = (1.83125,   3.740625)
      wb=3 -> (1043/640, 3627/1280) = (1.6296875, 2.83359375)
      wb=2 -> (457/320,  1233/640)  = (1.428125,  1.9265625)

    wb=2 is DEAD across the entire primary band (2.0-2.3).
    """
    return 1 + LIVE_D_MIN * W(wb), LIVE_D_MAX * W(wb)


LIVE_BANDS = {wb: live_band(wb) for wb in (2, 3, 4)}

#: The degenerate cell the filter exists to catch: wb=2 at anchor 2 has W == B,
#: so d(T=max) is exactly 1.0.  "T=max wins" there would mean "2-bit
#: quantization wins", not "structured pruning wins".
DEGENERATE_WB2_ANCHOR2 = {
    1: d_bitmap(ANCHOR_2, 2, 1),        # 0.532853
    16: d_bitmap(ANCHOR_2, 2, 16),      # 0.970803
    32: d_bitmap(ANCHOR_2, 2, 32),      # 0.985401
    "max": d_bitmap(ANCHOR_2, 2, "max"),  # exactly 1
}


# --------------------------------------------------------------------------- #
# Errata: what Spec v6 printed vs what is true
# --------------------------------------------------------------------------- #
# Kept so the test suite can assert we are NOT reproducing the spec's values.
# Delete a row only when the spec document itself has been corrected.

SPEC_V6_ERRATA = {
    "anchor1_tile4_density": {
        "spec": 0.696992,
        "true": float(F(371, 532)),          # 0.6973684210526315
        "where": "section 5.2, M1 anchor-1 table, row 3",
    },
    "anchor1_tmax_density": {
        "spec": 0.757954,
        "true": float(F(403, 532)),          # 0.7575187969924813
        "where": "section 5.2, M1 anchor-1 table, row 5",
    },
    "log2_11008": {
        "spec": 13.4262102,
        "true": math.log2(11008),            # 13.426264754702098
        "where": "section 3.4, index model block",
    },
    "density_for_budget_0p60": {
        "spec": 0.0341271,
        "true": d_list(0.60, 4, N_IDX_FFN),  # 0.03412481...
        "where": "section 3.4, corrected-tests block",
    },
    "b_star_11008": {
        "spec": 1.3095620,
        "true": b_star(4, N_IDX_FFN),        # 1.3095612...
        "where": "section 3.4, B* continuity block",
    },
    "advantage_at_B_1p20": {
        "spec": 0.205428,
        "true": advantage_below_b_star(1.20, 4, N_IDX_FFN),   # 0.2054344...
        "where": "section 0.3.1, erosion table",
    },
    "advantage_at_B_1p00": {
        "spec": 0.168684,
        "true": advantage_below_b_star(1.00, 4, N_IDX_FFN),   # 0.1686891...
        "where": "section 0.3.1, erosion table",
    },
}


# --------------------------------------------------------------------------- #
# 70B deployment envelope  (Spec v6 section 0.1)
# --------------------------------------------------------------------------- #
# ERRATUM (audit A4): the spec assumes 4k context and concludes "the real
# threshold is ~2.2-2.3 bit".  Its own line items give 2.46-2.65 at 4k.  The
# 2.2-2.3 figure corresponds to 8k-16k context.  Since the entire motivation
# rests on the budget landing at or below the PTQ floor, the context length that
# produces the threshold has to be stated.

GIB = 2 ** 30
LLAMA70B_LINEAR_PARAMS = 68.45e9
LLAMA70B_EMBED_PARAMS = 2 * 32000 * 8192          # embed_tokens + lm_head
LLAMA70B_KV_BYTES_PER_TOKEN = 2 * 8 * 128 * 80 * 2  # GQA: 2*(kv heads)*(head dim)*(layers)*fp16
FRAMEWORK_OVERHEAD_GIB = 0.7

LINEAR_GIB_PER_BIT = LLAMA70B_LINEAR_PARAMS / 8 / GIB          # 7.9686
EMBED_GIB = LLAMA70B_EMBED_PARAMS * 2 / GIB                    # 0.9766


def fixed_load_gib(ctx: int) -> float:
    """Everything that is NOT a linear weight: embeddings + KV cache + framework."""
    return EMBED_GIB + LLAMA70B_KV_BYTES_PER_TOKEN * ctx / GIB + FRAMEWORK_OVERHEAD_GIB


def max_bits_per_position(capacity_gib: float, ctx: int) -> float:
    """Largest weight budget that fits, given usable VRAM and context length."""
    return (capacity_gib - fixed_load_gib(ctx)) / LINEAR_GIB_PER_BIT


#: (capacity_gib, ctx) -> max bits.  The 2.0-2.2 PTQ floor is only binding at
#: 8k+ context, which is where the motivation belongs.
DEPLOYMENT_ENVELOPE = {
    (cap, ctx): max_bits_per_position(cap, ctx)
    for cap in (24.0, 23.0, 22.5)
    for ctx in (4096, 8192, 16384)
}


# --------------------------------------------------------------------------- #
# RHT affordability  (audit section C -- not in Spec v6)
# --------------------------------------------------------------------------- #

def rht_overhead_ratio(tile_size: int | str, density: float, n_idx: int) -> float:
    """Cost of per-tile randomized Hadamard on the COMPACTED survivor matrix,
    relative to the GEMV it sits on:  log2(d * n) / T.

    A mask that is already frozen cannot be destroyed by a rotation, so RHT is
    legal on the compacted survivors -- but each tile owns a different column
    set, so it needs its own transform.  That makes incoherence processing
    unaffordable at T=1 (11.5x) and nearly free at T=max, i.e. granularity
    controls whether QuIP#/QTIP-class quantization is available to a sparse
    matrix at all.  This is a second force pushing T up that the Delta = Q + tau
    model does not contain.
    """
    if tile_size == "max":
        return 0.0
    return math.log2(density * n_idx) / tile_size


# --------------------------------------------------------------------------- #
# E8P survivor branch  (plan section H2 -- supersedes the 4-bit anchors)
# --------------------------------------------------------------------------- #
# Decision of 2026-08-20: survivors are quantized with a lattice VQ (QuIP# E8P)
# rather than GPTQ-4bit.  E8P is a STRUCTURED codebook, so Spec v6 section 3.2's
# amortization term is zero and W collapses from 532/128 to exactly 2.
#
# WARNING (plan H5/H7): vq_bits = 2.0 is the paper-arithmetic value and the
# assumption that E8P holds its quality on a COMPACTED SURVIVOR submatrix is
# explicitly untested.  Measure it from the checkpoint file size before it
# anchors anything.

E8P_VQ_BITS = F(2)

#: The whole live band moves below 2 bits -- which is the regime the thesis is
#: about and where dense PTQ has no answer (QuIP# 2-bit 6.66, QuaRot-GPTQ 2-bit
#: 22.07).  The two families do NOT overlap: GPTQ-4bit is live over 1.83-3.74.
E8P_LIVE_BAND = (1 + LIVE_D_MIN * E8P_VQ_BITS, LIVE_D_MAX * E8P_VQ_BITS)  # 7/5, 9/5

#: Primary band after re-anchoring.  All three are live for E8P survivors.
E8P_BUDGETS = (F(7, 4), F(8, 5), F(3, 2))


def d_bitmap_vq(budget: F, vq: F, tile_size: int | str) -> F:
    """d(T) = (B - 1/T) / vq_bits, exact.  The VQ branch carries no q_over."""
    inv_t = F(0) if tile_size == "max" else F(1, tile_size)
    return (budget - inv_t) / vq


def identity_vq(tile_size: int | str, vq: F) -> F:
    """(1 - 1/T) / vq_bits.  Halving W doubles the leverage."""
    inv_t = F(0) if tile_size == "max" else F(1, tile_size)
    return (1 - inv_t) / vq


#: B = 1.5 with E8P: every density is an exact dyadic rational.
#: T=1 -> 1/4, T=2 -> 1/2, T=4 -> 5/8, T=8 -> 11/16, T=16 -> 23/32,
#: T=32 -> 47/64, T=max -> 3/4
E8P_AT_1P5 = {t: d_bitmap_vq(F(3, 2), E8P_VQ_BITS, t) for t in (*TILE_GRID, "max")}

#: 15/32 = 0.46875, against 30/133 = 0.2255639 for GPTQ-4bit survivors.
E8P_IDENTITY_T16 = identity_vq(16, E8P_VQ_BITS)


def b_star_vq(vq: F, n_idx: int, tile_size: int | str = 1) -> float:
    """B*(T) = 1/T + vq_bits / log2(n_idx) for the VQ branch."""
    inv_t = 0.0 if tile_size == "max" else 1.0 / tile_size
    return inv_t + float(vq) / L(n_idx)


def live_band_vq(vq: F) -> tuple[F, F]:
    return 1 + LIVE_D_MIN * vq, LIVE_D_MAX * vq
