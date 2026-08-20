"""Bit-budget accounting for the tile-sparsity study.

Implements Spec v6 section 3 (Muhasebe).  Every budget and every density the
experiments are anchored to must come from this module -- nothing downstream may
hard-code a bit budget or a density.

Audit corrections baked in (plan file sections A1-A3, B1, D1):

  * Golden constants are DERIVED, never typed.  Spec v6 section 5.2's anchor-1
    table had two wrong cells (tile-4, T=max) and section 3.4 carried a stale
    value of log2(11008).  Both error classes are structurally impossible here.

  * `nm_index_bits(..., packing="combinatorial")` is exposed as a PRACTICAL
    encoding, not an information-theoretic bound.  A fixed-count block code is
    decodable in O(1) per block, so it keeps random access -- which is the whole
    justification for the `practical` column.  The default `index_model`
    ("practical") is left exactly as Spec v6 froze it, so the pre-registered
    accounting is unchanged; the correction is available and testable next to it
    rather than silently replacing it.

  * `is_live` answers "is this cell a usable GRANULARITY probe".  That is not the
    same question as "is this row reportable" -- see `live_diagnostics`.  Spec
    v6's own strongest Gate A candidate (AQLM-survivor) fails `is_live` at the
    T=max edge while remaining a perfectly valid Gate A row.

Conventions frozen by the spec:
  W(wb)   = wb + q_over,  q_over = (scale_bits + wb) / group_size
            4-bit -> 532/128, 3-bit -> 403/128, 2-bit -> 274/128
  index   = min(1, d * log2(n_idx)) / T     (bitmap / fixed-width-list cascade)
  B*(T)   = 1/T + W / log2(n_idx)           (the two index branches meet here)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

__all__ = [
    "SCHEMES",
    "FIXED_DENSITY_SCHEMES",
    "Q_OVERHEAD_SCALES_WITH_DENSITY",
    "MAX_TILE",
    "Config",
    "q_overhead",
    "weight_cost",
    "vq_bits_from_spec",
    "entropy_bits",
    "nm_index_bits",
    "vnm_index_bits",
    "index_bits",
    "bits_per_position",
    "anchor_budget_to",
    "density_for_budget",
    "scheme_floor",
    "b_star",
    "d_star",
    "in_bitmap_regime",
    "tile_density_advantage",
    "live_band",
    "live_diagnostics",
    "is_live",
    "budget_matched_grid",
    "roofline_bytes",
]

# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #

SCHEMES = ("dense", "unstructured", "tile", "structured", "nm", "vnm")

#: Schemes whose density is fixed by the scheme itself, so `density_for_budget`
#: is meaningless for them (Spec v6 section 3.3, Kural 2).  "vq_dense" is not a
#: sparsity scheme but is accepted as a name so callers get the same error.
#: "dense" is added to the spec's three because it is equally fixed (d == 1).
FIXED_DENSITY_SCHEMES = frozenset({"nm", "vnm", "vq_dense", "dense"})

#: Spec v6 section 3.1.  Whether the quantization overhead (group scale +
#: zero-point) is charged per SURVIVING weight (True -> d * (wb + q_over)) or on
#: top of the surviving weights (False -> d * wb + q_over).
#:
#: This convention flips the sign of the M1 2:4 comparison and MUST be printed in
#: every table:  True  -> 2:4 @ 4-bit = 3.078125 (below dense 3-bit)
#:               False -> 2:4 @ 4-bit = 3.156250 (above dense 3-bit)
#:
#: There is deliberately no "vq" key: the VQ branch does not use q_over at all,
#: it uses vq_bits (section 3.2).
Q_OVERHEAD_SCALES_WITH_DENSITY = {
    "dense": True,
    "unstructured": True,
    "tile": True,
    "structured": True,
    "nm": False,
    "vnm": False,
}

#: Sentinel for the coarse edge of the tile family, T = n (one column set for the
#: whole matrix).  Numerically identical to scheme="structured".
MAX_TILE = "max"

DEFAULT_GROUP_SIZE = 128
DEFAULT_SCALE_BITS = 16

#: Spec v6 section 3.5.  A budget cell is a usable granularity probe only if the
#: fine end of the family is actually sparse and the coarse end is not already
#: dense.  Outside this band the cell measures quantization, not granularity.
LIVE_DENSITY_MIN = 0.2   # required: d(T=1)   >  LIVE_DENSITY_MIN
LIVE_DENSITY_MAX = 0.9   # required: d(T=max) <  LIVE_DENSITY_MAX

#: Spec v6 section 3.3, Kural 2: a fixed-density scheme reported at its own
#: natural cost gets a signed offset column, flagged past this threshold.
OFFSET_FLAG_THRESHOLD = 0.01


# --------------------------------------------------------------------------- #
# Weight cost
# --------------------------------------------------------------------------- #

def q_overhead(
    weight_bits: int,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """Quantization overhead per weight: one FP16 scale + one wb-bit zero-point
    per group.

    >>> q_overhead(4)
    0.15625
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return (scale_bits + weight_bits) / group_size


def weight_cost(
    weight_bits: int,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """W = wb + q_over -- the cost of ONE SURVIVING weight.

    This is the denominator of the `1 - 1/T` identity, so it is the single most
    load-bearing quantity in the spec.

    >>> weight_cost(4) * 128
    532.0
    """
    return weight_bits + q_overhead(
        weight_bits, group_size=group_size, scale_bits=scale_bits
    )


def vq_bits_from_spec(
    idx_bits: int,
    dim: int,
    *,
    entry_bits: int = 16,
    weights_per_codebook: float | None = None,
) -> float:
    """Spec v6 section 3.2, VQ branch.

        vq_bits = idx_bits / dim + codebook_amortization

    `weights_per_codebook=None` means a structured codebook with no per-model
    storage (QuIP# E8P), i.e. zero amortization.

    AQLM 1x16, dim=8, over a Llama-2-7B FFN block (45.1M weights):

    >>> round(vq_bits_from_spec(16, 8, weights_per_codebook=45.1e6), 6)
    2.186

    WARNING: this is the paper-arithmetic value.  Spec v6 requires the real cost
    to be measured from the checkpoint's file size before it anchors anything.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    base = idx_bits / dim
    if weights_per_codebook is None:
        return base
    if weights_per_codebook <= 0:
        raise ValueError("weights_per_codebook must be positive or None")
    amortization = (2 ** idx_bits) * dim * entry_bits / weights_per_codebook
    return base + amortization


def _weight_terms(
    weight_bits: int | None,
    vq_bits: float | None,
    group_size: int,
    scale_bits: int,
) -> tuple[float, float]:
    """Return (payload_bits, q_over) per surviving weight.

    The VQ branch carries no q_over, which is why it has no entry in
    Q_OVERHEAD_SCALES_WITH_DENSITY.
    """
    if vq_bits is not None:
        if weight_bits is not None:
            raise ValueError("pass either weight_bits or vq_bits, not both")
        if vq_bits <= 0:
            raise ValueError(f"vq_bits must be positive, got {vq_bits}")
        return float(vq_bits), 0.0
    if weight_bits is None:
        raise ValueError("one of weight_bits / vq_bits is required")
    if weight_bits <= 0:
        raise ValueError(f"weight_bits must be positive, got {weight_bits}")
    return float(weight_bits), q_overhead(
        weight_bits, group_size=group_size, scale_bits=scale_bits
    )


def total_weight_cost(
    weight_bits: int | None = None,
    *,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """W, resolved for either the scalar-quantization or the VQ branch."""
    payload, q_over = _weight_terms(weight_bits, vq_bits, group_size, scale_bits)
    return payload + q_over


# --------------------------------------------------------------------------- #
# Index cost
# --------------------------------------------------------------------------- #

def entropy_bits(density: float) -> float:
    """Binary entropy H(d) in bits.  H(0) = H(1) = 0."""
    if density <= 0.0 or density >= 1.0:
        return 0.0
    return -(
        density * math.log2(density) + (1.0 - density) * math.log2(1.0 - density)
    )


def nm_index_bits(n: int, m: int, *, packing: str = "fixed_width") -> float:
    """Index cost of an N:M mask, in bits per position.

    packing="fixed_width"   : ceil(log2(M)) * N / M      (Spec v6, `practical`)
    packing="combinatorial" : log2(C(M, N)) / M          (Spec v6, `info_theoretic`)

    AUDIT (plan section B1): the combinatorial packing is *also practical*.  A
    block of M positions with exactly N survivors is decodable in O(1) without
    touching any other block, so random access survives.  Spec v6 files it under
    `info_theoretic`, which understates how cheap a random-accessible index can
    be and therefore overstates the tile advantage.

    >>> nm_index_bits(2, 4)
    1.0
    >>> round(nm_index_bits(2, 8, packing="combinatorial"), 6)
    0.600919
    """
    if not (0 <= n <= m) or m <= 0:
        raise ValueError(f"need 0 <= N <= M and M > 0, got N={n}, M={m}")
    if packing == "fixed_width":
        if n == 0:
            return 0.0
        return math.ceil(math.log2(m)) * n / m
    if packing == "combinatorial":
        return math.log2(math.comb(m, n)) / m
    raise ValueError(f"unknown packing {packing!r}")


def vnm_index_bits(
    v: int, n: int, m: int, *, native_m: int = 4, packing: str = "fixed_width"
) -> float:
    """Index cost of VENOM's V:N:M format, in bits per position.

    Reconstructed from Castro et al., SC'23 (arXiv:2310.02065).  An R x K matrix
    is cut into V x M blocks and pruned in two stages:

      1. vector-wise -- each block selects `native_m` (=4) of its M columns, and
         all V rows of the block share that selection;
      2. within those 4 columns the hardware's native 2:4 applies, so each row
         keeps N of them.

    Two metadata structures follow, and the paper gives their shapes:

      m-indices   R x K/M x N,     2 bits each   -> 2N/M per position
      column-loc  R/V x K/M x 4,   one column id -> 4*ceil(log2 M)/(V*M)

    THE STRUCTURAL POINT: `V` is a row-tile.  A group of V rows sharing one
    column selection is exactly this project's Axis B at T=V, with the extra
    constraint that the selection is block-local (4 out of each M) rather than
    free across the row.  VENOM is therefore much closer prior work than Spec v6
    credited, and it is also a concrete instance of the block-local index that
    section 3.2 says a bitmap is not the floor of.

    The 2-bit width of m-indices is stated in the paper.  The width of a
    column-loc entry is INFERRED as ceil(log2 M) from the array's shape and
    meaning; it is not quoted.  At M == native_m the vector stage is degenerate
    (choosing 4 of 4) and the honest cost is zero -- `packing="combinatorial"`
    reports that, `fixed_width` reports what VENOM's array actually stores.

    >>> round(vnm_index_bits(64, 2, 8), 6)
    0.523438
    >>> vnm_index_bits(64, 2, 4, packing="combinatorial")   # plain 2:4
    1.0
    """
    if not (0 < n <= native_m):
        raise ValueError(f"need 0 < N <= {native_m}, got N={n}")
    if m < native_m or m % native_m:
        raise ValueError(f"M must be a multiple of {native_m}, got M={m}")
    if v < 1:
        raise ValueError(f"V must be positive, got {v}")

    per_nonzero = math.log2(native_m)                 # 2 bits for the native 2:4
    m_indices = per_nonzero * n / m

    if packing == "fixed_width":
        column_loc = native_m * math.ceil(math.log2(m)) / (v * m)
    elif packing == "combinatorial":
        column_loc = math.log2(math.comb(m, native_m)) / (v * m)
    else:
        raise ValueError(f"unknown packing {packing!r}")
    return m_indices + column_loc


def _elementwise_index(density: float, n_idx: int, model: str) -> float:
    """Index cost of a free (unstructured) mask over `n_idx` positions, before
    any tile amortization.

    `practical` is the min(bitmap, fixed-width list) cascade: 1.0 bit per
    position, or d*log2(n_idx) bits when the survivors are sparse enough that
    listing them is cheaper.  Both keep O(1) random access.
    """
    if model == "practical":
        if n_idx is None or n_idx <= 1:
            raise ValueError(f"n_idx must be > 1, got {n_idx}")
        return min(1.0, density * math.log2(n_idx))
    if model == "info_theoretic":
        return entropy_bits(density)
    raise ValueError(f"unknown index_model {model!r}")


def _inv_tile(scheme: str, tile_size: int | str | None) -> float:
    """1/T, the factor the index is amortized by.

    0.0 means "no index at all" (T = n, the structured edge of the family).
    """
    if scheme == "structured":
        return 0.0
    if scheme == "unstructured":
        return 1.0
    if scheme == "tile":
        if tile_size is None:
            raise ValueError("scheme='tile' requires tile_size")
        if tile_size == MAX_TILE:
            return 0.0
        if not isinstance(tile_size, int) or tile_size < 1:
            raise ValueError(
                f"tile_size must be a positive int or {MAX_TILE!r}, got {tile_size!r}"
            )
        return 1.0 / tile_size
    raise ValueError(f"scheme {scheme!r} has no tile amortization")


def index_bits(
    scheme: str,
    density: float,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    nm: tuple[int, int] | None = None,
    vnm: tuple[int, int, int] | None = None,
    index_model: str = "practical",
    nm_packing: str = "fixed_width",
) -> float:
    """Index cost in bits per position (Spec v6 section 3.2).

    Note that `n_idx` is layer-dependent (Axis B: n_idx = d_in, Axis A:
    n_idx = n_out).  In the bitmap regime the index is 1/T and therefore layer
    INDEPENDENT; the layer dependence only switches on below B*.
    """
    if scheme in ("dense", "structured"):
        return 0.0
    if scheme == "nm":
        if nm is None:
            raise ValueError("scheme='nm' requires nm=(N, M)")
        return nm_index_bits(nm[0], nm[1], packing=nm_packing)
    if scheme == "vnm":
        if vnm is None:
            raise ValueError("scheme='vnm' requires vnm=(V, N, M)")
        return vnm_index_bits(*vnm, packing=nm_packing)
    if scheme in ("unstructured", "tile"):
        return _elementwise_index(density, n_idx, index_model) * _inv_tile(
            scheme, tile_size
        )
    raise ValueError(f"unknown scheme {scheme!r}")


# --------------------------------------------------------------------------- #
# Bits per position
# --------------------------------------------------------------------------- #

def bits_per_position(
    scheme: str,
    density: float | None = None,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    nm: tuple[int, int] | None = None,
    vnm: tuple[int, int, int] | None = None,
    vq_bits: float | None = None,
    index_model: str = "practical",
    nm_packing: str = "fixed_width",
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """Total bits per weight POSITION (not per surviving weight).

    >>> bits_per_position("dense", 1.0, 4)
    4.15625
    >>> round(bits_per_position("tile", 0.5, 4, 11008, tile_size=16), 6)
    2.140625
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    if scheme == "dense":
        if density is not None and abs(density - 1.0) > 1e-12:
            raise ValueError(f"scheme='dense' implies density=1.0, got {density}")
        density = 1.0
    elif scheme in ("nm", "vnm"):
        if scheme == "vnm":
            if vnm is None:
                raise ValueError("scheme='vnm' requires vnm=(V, N, M)")
            implied = vnm[1] / vnm[2]
        else:
            if nm is None:
                raise ValueError("scheme='nm' requires nm=(N, M)")
            implied = nm[0] / nm[1]
        if density is not None and abs(density - implied) > 1e-12:
            raise ValueError(
                f"scheme={scheme!r} with nm={nm} implies density={implied}, "
                f"got {density}"
            )
        density = implied
    if density is None:
        raise ValueError(f"scheme={scheme!r} requires an explicit density")
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density must be in [0, 1], got {density}")

    payload, q_over = _weight_terms(weight_bits, vq_bits, group_size, scale_bits)

    if Q_OVERHEAD_SCALES_WITH_DENSITY[scheme]:
        weight_term = density * (payload + q_over)
    else:
        weight_term = density * payload + q_over

    idx = index_bits(
        scheme,
        density,
        n_idx,
        tile_size=tile_size,
        nm=nm,
        vnm=vnm,
        index_model=index_model,
        nm_packing=nm_packing,
    )
    return weight_term + idx


def anchor_budget_to(
    scheme: str,
    weight_bits: int | None = None,
    **kw,
) -> float:
    """Kural 1: every budget is anchored to the FULL cost of a dense baseline.

    Never anchor to a round number -- Spec v6 section 7, trap 6.

    >>> anchor_budget_to("dense", 3)
    3.1484375
    """
    kw.pop("density", None)
    return bits_per_position(scheme, 1.0, weight_bits, **kw)


# --------------------------------------------------------------------------- #
# Inverting the budget
# --------------------------------------------------------------------------- #

def _solve_density_numeric(
    budget_bits: float,
    scheme: str,
    weight_bits: int | None,
    n_idx: int | None,
    kw: dict,
) -> float | None:
    """Bisection fallback for index models with no closed-form inverse
    (e.g. info_theoretic, where the index is H(d))."""
    def cost(d: float) -> float:
        return bits_per_position(
            scheme, d, weight_bits, n_idx, **kw
        )

    lo, hi = 0.0, 1.0
    if cost(hi) < budget_bits:
        return None                      # budget exceeds even the dense cost
    if cost(1e-15) > budget_bits:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if cost(mid) <= budget_bits:
            lo = mid
        else:
            hi = mid
    return lo


def density_for_budget(
    scheme: str,
    budget_bits: float,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    vq_bits: float | None = None,
    index_model: str = "practical",
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float | None:
    """Largest density reachable at `budget_bits`, or None if unreachable.

    Returns None when the budget cannot be met at any density in (0, 1] -- either
    it is below the scheme's floor or it exceeds the dense cost.

    Raises ValueError for fixed-density schemes (Spec v6 section 3.3, Kural 2):
    those are reported at their own natural cost with a signed offset column, not
    solved for.

    Closed form for `practical`; the min(bitmap, list) cascade makes the two
    branches mutually exclusive across B*(T), so the dispatch below is exact:

        bitmap : d = (B - 1/T) / W          valid iff d * log2(n_idx) >= 1
        list   : d = B / (W + log2(n_idx)/T) valid iff d * log2(n_idx) <  1
    """
    if scheme in FIXED_DENSITY_SCHEMES:
        raise ValueError(
            f"scheme={scheme!r} has a fixed density; report it at its own cost "
            "with a signed offset column (Spec v6 section 3.3, Kural 2) instead "
            "of solving for density."
        )
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    kw = dict(
        tile_size=tile_size,
        vq_bits=vq_bits,
        index_model=index_model,
        group_size=group_size,
        scale_bits=scale_bits,
    )

    if index_model != "practical":
        d = _solve_density_numeric(budget_bits, scheme, weight_bits, n_idx, kw)
        return d if (d is not None and 0.0 < d <= 1.0) else None

    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    inv_t = _inv_tile(scheme, tile_size)

    if inv_t == 0.0:
        d = budget_bits / W                      # structured / T=max: no index
    else:
        L = math.log2(n_idx)
        d_bitmap = (budget_bits - inv_t) / W
        if d_bitmap * L >= 1.0:
            d = d_bitmap
        else:
            d = budget_bits / (W + L * inv_t)

    return d if 0.0 < d <= 1.0 else None


def scheme_floor(
    scheme: str,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = None,
    nm: tuple[int, int] | None = None,
    vnm: tuple[int, int, int] | None = None,
    vq_bits: float | None = None,
    index_model: str = "practical",
    nm_packing: str = "fixed_width",
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """Infimum of bits_per_position over achievable densities.

    Under the min(bitmap, list) cascade BOTH the weight term and the index term
    vanish as d -> 0, so density-scaling schemes have NO hard bit floor.  This is
    the section 3.4 correction: `scheme_floor("unstructured", 4, 11008) == 0.0`,
    not 1.0.  "Unstructured cannot go below 1.0 bit" is a statement about the
    BITMAP, not about unstructured sparsity (Spec v6 section 7, trap 5).
    """
    if scheme == "dense":
        return total_weight_cost(
            weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
        )
    if scheme in ("nm", "vnm"):
        return bits_per_position(
            scheme,
            None,
            weight_bits,
            n_idx,
            nm=nm,
            vnm=vnm,
            vq_bits=vq_bits,
            index_model=index_model,
            nm_packing=nm_packing,
            group_size=group_size,
            scale_bits=scale_bits,
        )
    if scheme in ("unstructured", "tile", "structured"):
        _inv_tile(scheme, tile_size)     # validate tile_size even though unused
        return 0.0
    raise ValueError(f"unknown scheme {scheme!r}")


# --------------------------------------------------------------------------- #
# The B* wall and the 1 - 1/T identity
# --------------------------------------------------------------------------- #

def b_star(
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = 1,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float | None:
    """Budget at which the bitmap and list index branches meet:

        B*(T) = 1/T + W / log2(n_idx)

    At and above B*(T) the `1 - 1/T` identity holds exactly.  Below it the
    unstructured index gets cheaper than a bitmap and the tile advantage ERODES
    -- B* is the lowest budget at which the advantage is fully preserved, and it
    is a closed-form limit on how far the exploratory band can be pushed.

    Returns None for T = max (no index, hence no crossover).

    >>> round(b_star(4, 4096), 7)
    1.3463542
    """
    scheme = "tile" if tile_size not in (1, None) else "unstructured"
    inv_t = _inv_tile(scheme, tile_size if scheme == "tile" else None)
    if inv_t == 0.0:
        return None
    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    return inv_t + W / math.log2(n_idx)


def d_star(n_idx: int) -> float:
    """Density at which a fixed-width index costs exactly one bit: 1 / log2(n_idx)."""
    return 1.0 / math.log2(n_idx)


def in_bitmap_regime(
    budget_bits: float,
    weight_bits: int | None = None,
    n_idx: int | None = None,
    *,
    tile_size: int | str | None = 1,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> bool:
    """Validity domain of the `1 - 1/T` identity (Spec v6 section 7, trap 1).

    Never use the identity without asserting this.
    """
    bs = b_star(
        weight_bits,
        n_idx,
        tile_size=tile_size,
        vq_bits=vq_bits,
        group_size=group_size,
        scale_bits=scale_bits,
    )
    return True if bs is None else budget_bits >= bs


def tile_density_advantage(
    tile_size: int | str,
    weight_bits: int | None = None,
    *,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> float:
    """The headline identity:  d(T) - d(1) = (1 - 1/T) / W.

    Independent of the budget B.  The ABSOLUTE advantage is constant; the ratio
    d(T)/d(1) grows only because the denominator shrinks (Spec v6 section 7,
    trap 2 -- never write the headline as "the ratio grows").

    Valid only for B >= B*(1); check with `in_bitmap_regime`.
    """
    inv_t = _inv_tile("tile", tile_size)
    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    return (1.0 - inv_t) / W


# --------------------------------------------------------------------------- #
# Live-band filter
# --------------------------------------------------------------------------- #

def live_band(
    weight_bits: int | None = None,
    *,
    vq_bits: float | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    scale_bits: int = DEFAULT_SCALE_BITS,
) -> tuple[float, float]:
    """(B_min, B_max) over which a budget cell is a usable granularity probe.

        d(T=1)   > 0.2  =>  B > 1 + 0.2 * W
        d(T=max) < 0.9  =>  B < 0.9 * W

    >>> lo, hi = live_band(2)
    >>> lo * 320, hi * 640
    (457.0, 1233.0)
    """
    W = total_weight_cost(
        weight_bits, vq_bits=vq_bits, group_size=group_size, scale_bits=scale_bits
    )
    return 1.0 + LIVE_DENSITY_MIN * W, LIVE_DENSITY_MAX * W


def live_diagnostics(config: "Config") -> dict:
    """Why a cell is or is not a granularity probe.

    AUDIT (plan section D1): `is_live` is a question about the GRANULARITY AXIS,
    not about whether a row may be reported.  A config that fails at the T=max
    edge -- AQLM-survivor at anchor 2 has d(T=max) = 0.979 -- is still a
    perfectly valid Gate A row.  Split the two uses; do not silently drop rows.
    """
    if config.budget_bits is None:
        raise ValueError("live_diagnostics needs config.budget_bits")
    common = dict(
        n_idx=config.n_idx,
        vq_bits=config.vq_bits,
        index_model=config.index_model,
        group_size=config.group_size,
        scale_bits=config.scale_bits,
    )
    d_fine = density_for_budget(
        "unstructured", config.budget_bits, config.weight_bits, tile_size=1, **common
    )
    d_coarse = density_for_budget(
        "structured", config.budget_bits, config.weight_bits, **common
    )
    lo, hi = live_band(
        config.weight_bits,
        vq_bits=config.vq_bits,
        group_size=config.group_size,
        scale_bits=config.scale_bits,
    )
    fine_ok = d_fine is not None and d_fine > LIVE_DENSITY_MIN
    coarse_ok = d_coarse is not None and d_coarse < LIVE_DENSITY_MAX
    reasons = []
    if not fine_ok:
        reasons.append(
            f"fine end not sparse enough: d(T=1)="
            f"{'None' if d_fine is None else f'{d_fine:.6f}'} <= {LIVE_DENSITY_MIN}"
        )
    if not coarse_ok:
        reasons.append(
            f"coarse end already dense: d(T=max)="
            f"{'None' if d_coarse is None else f'{d_coarse:.6f}'} >= {LIVE_DENSITY_MAX}"
        )
    return {
        "live": fine_ok and coarse_ok,
        "d_fine": d_fine,
        "d_coarse": d_coarse,
        "band": (lo, hi),
        "reasons": reasons,
    }


def is_live(config: "Config") -> bool:
    """Spec v6 section 3.5.  True iff the cell measures granularity rather than
    quantization.

    wb=2 is DEAD across the whole primary band (2.0-2.3): at anchor 2 its
    d(T=max) is exactly 1.0, so "T=max wins" there would mean "2-bit quantization
    wins", not "structured pruning wins".  Reporting such a cell invites exactly
    that misreading (Spec v6 section 7, trap 14).
    """
    return live_diagnostics(config)["live"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Config:
    """One accounting cell.  Everything a results/*.json row must carry."""

    scheme: str
    weight_bits: int | None = None
    density: float | None = None
    n_idx: int | None = None
    tile_size: int | str | None = None
    nm: tuple[int, int] | None = None
    vq_bits: float | None = None
    budget_bits: float | None = None
    index_model: str = "practical"
    nm_packing: str = "fixed_width"
    group_size: int = DEFAULT_GROUP_SIZE
    scale_bits: int = DEFAULT_SCALE_BITS
    label: str = ""

    def resolved_density(self) -> float | None:
        """Density, solved from budget_bits when the scheme allows it."""
        if self.density is not None:
            return self.density
        if self.scheme == "dense":
            return 1.0
        if self.nm is not None:
            return self.nm[0] / self.nm[1]
        if self.budget_bits is None:
            return None
        return density_for_budget(
            self.scheme,
            self.budget_bits,
            self.weight_bits,
            self.n_idx,
            tile_size=self.tile_size,
            vq_bits=self.vq_bits,
            index_model=self.index_model,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
        )

    def bits_per_position(self) -> float:
        d = self.resolved_density()
        if d is None:
            raise ValueError(f"{self.label or self.scheme}: density is unresolvable")
        return bits_per_position(
            self.scheme,
            d,
            self.weight_bits,
            self.n_idx,
            tile_size=self.tile_size,
            nm=self.nm,
            vq_bits=self.vq_bits,
            index_model=self.index_model,
            nm_packing=self.nm_packing,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
        )

    def weight_cost(self) -> float:
        return total_weight_cost(
            self.weight_bits,
            vq_bits=self.vq_bits,
            group_size=self.group_size,
            scale_bits=self.scale_bits,
        )


# --------------------------------------------------------------------------- #
# Budget-matched grid
# --------------------------------------------------------------------------- #

def default_candidates(
    weight_bits: int,
    n_idx: int,
    *,
    tile_grid: Sequence[int] = (2, 4, 8, 16, 32),
    nm_variants: Sequence[tuple[int, int]] = ((2, 4), (4, 8)),
    vq_bits: float | None = None,
) -> list[Config]:
    """The standard family at one weight_bits: unstructured, the tile ladder,
    the structured edge, and the N:M lattice."""
    kw = dict(weight_bits=weight_bits, n_idx=n_idx, vq_bits=vq_bits)
    if vq_bits is not None:
        kw["weight_bits"] = None
    tag = f"vq{vq_bits:g}" if vq_bits is not None else f"{weight_bits}-bit"
    out = [Config(scheme="unstructured", tile_size=1, label=f"{tag} + unstructured", **kw)]
    out += [
        Config(scheme="tile", tile_size=t, label=f"{tag} + tile-{t}", **kw)
        for t in tile_grid
    ]
    out.append(Config(scheme="structured", label=f"{tag} + T=max (structured)", **kw))
    out += [
        Config(scheme="nm", nm=v, label=f"{tag} + {v[0]}:{v[1]}", **kw)
        for v in nm_variants
    ]
    return out


def budget_matched_grid(
    budget_bits: float,
    tol: float = 0.02,
    *,
    n_idx: int = 11008,
    weight_bits_grid: Sequence[int] = (2, 3, 4),
    tile_grid: Sequence[int] = (2, 4, 8, 16, 32),
    nm_variants: Sequence[tuple[int, int]] = ((2, 4), (4, 8)),
    vq_bits_grid: Sequence[float] = (),
    candidates: Iterable[Config] | None = None,
    apply_live_filter: bool = True,
) -> list[dict]:
    """Every config that sits at `budget_bits`, with a signed offset column.

    `tol` is the inclusion tolerance for FIXED-density schemes, which cannot be
    solved onto the budget and are reported at their own natural cost.  Rows past
    OFFSET_FLAG_THRESHOLD (1%) are flagged.

    `apply_live_filter` drops cells that are not granularity probes (section 3.5).
    Do NOT sweep the full product without it -- degenerate cells get misread.
    Set it False when building a Gate A table, where a coarse-end-dense row is
    still meaningful (see `live_diagnostics`).
    """
    if candidates is None:
        cands: list[Config] = []
        for wb in weight_bits_grid:
            cands += default_candidates(
                wb, n_idx, tile_grid=tile_grid, nm_variants=nm_variants
            )
        for vb in vq_bits_grid:
            cands += default_candidates(
                None, n_idx, tile_grid=tile_grid, nm_variants=(), vq_bits=vb
            )
    else:
        cands = list(candidates)

    rows: list[dict] = []
    for c in cands:
        c = replace(c, budget_bits=budget_bits)
        fixed = c.scheme in FIXED_DENSITY_SCHEMES
        try:
            density = c.resolved_density()
        except (ValueError, NotImplementedError):
            continue
        if density is None or not (0.0 < density <= 1.0):
            continue
        try:
            bits = bits_per_position(
                c.scheme, density, c.weight_bits, c.n_idx,
                tile_size=c.tile_size, nm=c.nm, vq_bits=c.vq_bits,
                index_model=c.index_model, nm_packing=c.nm_packing,
                group_size=c.group_size, scale_bits=c.scale_bits,
            )
        except (ValueError, NotImplementedError):
            continue

        offset = bits - budget_bits
        offset_pct = offset / budget_bits
        if fixed and abs(offset_pct) > tol:
            continue

        diag = live_diagnostics(c)
        if apply_live_filter and not diag["live"]:
            continue

        rows.append(
            {
                "label": c.label or c.scheme,
                "scheme": c.scheme,
                "weight_bits": c.weight_bits,
                "vq_bits": c.vq_bits,
                "tile_size": c.tile_size,
                "nm": c.nm,
                "density": density,
                "bits_per_position": bits,
                "offset": offset,
                "offset_pct": offset_pct,
                "flagged": abs(offset_pct) > OFFSET_FLAG_THRESHOLD,
                "n_idx": c.n_idx,
                "q_over_scales_with_density": Q_OVERHEAD_SCALES_WITH_DENSITY[c.scheme],
                "anchor": budget_bits,
                "live": diag["live"],
                "live_reasons": diag["reasons"],
                "in_bitmap_regime": in_bitmap_regime(
                    budget_bits, c.weight_bits, c.n_idx,
                    tile_size=c.tile_size if c.scheme == "tile" else 1,
                    vq_bits=c.vq_bits,
                ),
            }
        )
    rows.sort(key=lambda r: r["density"])
    return rows


# --------------------------------------------------------------------------- #
# Roofline
# --------------------------------------------------------------------------- #

def roofline_bytes(config: Config, n_params: int) -> int:
    """Weight bytes moved for one batch=1 decode step through `n_params`
    positions.

    Spec v6 section 0.6: at batch=1 decode, time ~ bytes moved / bandwidth, so
    this is the roofline LOWER BOUND for the scheme.  It deliberately excludes
    activations, the KV cache, and any gather overhead -- a scheme with a lower
    bound here has not been shown to be faster, only to move fewer weight bytes.
    """
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    return math.ceil(n_params * config.bits_per_position() / 8)


if __name__ == "__main__":  # pragma: no cover
    import doctest

    failures, _ = doctest.testmod()
    raise SystemExit(1 if failures else 0)
