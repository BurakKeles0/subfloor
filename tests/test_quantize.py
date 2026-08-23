"""E8P codebook fidelity and lattice quantization.

The codebook tests are the important ones: `vq_bits = 2.0` and
`codebook_amortization = 0` are load-bearing accounting claims, and they are
true only if the codebook really is 2^16 codewords over 8 dimensions built from
a fixed 256-entry table.  These check that rather than assuming it.
"""

from __future__ import annotations

import math

import pytest
import torch

import compact as C
import prune as P
import quantize as Q
import rotation as R

DT = torch.float64


@pytest.fixture(autouse=True)
def _seed():
    """Reseed before every test so results never depend on execution order."""
    torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# Codebook fidelity
# --------------------------------------------------------------------------- #

def test_source_codebook_is_227_plus_29():
    """The count that confirms the reconstruction: non-negative half-integer
    patterns with norm^2 <= 10 number exactly 227, and 29 padding patterns at
    norm^2 == 12 close the table at 256."""
    S = Q.source_codebook(DT)
    assert S.shape == (256, 8)

    n2 = S.square().sum(dim=1)
    assert int((n2 <= 10 + 1e-9).sum()) == 227
    assert int((n2 - 12).abs().lt(1e-9).sum()) == 29

    # Half-integers, and therefore never zero -- which is what makes every sign
    # flip produce a distinct vector.
    assert torch.allclose((S * 2).remainder(2.0), torch.ones_like(S))
    assert bool((S > 0).all())


def test_codebook_is_2_to_the_16_distinct_codewords():
    cb = Q.e8p_codebook(DT)
    assert cb.shape == (65536, 8)
    assert cb.shape[0] == 2 ** Q.E8P_INDEX_BITS
    uniq = torch.unique(torch.round(cb * 4).to(torch.int64), dim=0)
    assert uniq.shape[0] == 65536, "codewords must be distinct"


def test_every_codeword_lies_in_the_lattice():
    """Undo the +-1/4 shift and the result must be an all-half-integer vector
    with an even coordinate sum -- i.e. a point of E8."""
    cb = Q.e8p_codebook(DT)
    assert bool(Q.in_e8_plus_quarter(cb).all())


def test_eighth_sign_carries_the_parity():
    """Only seven signs are stored; the eighth is whatever lands in the lattice.
    So exactly 2^7 codewords share each source pattern and shift."""
    cb = Q.e8p_codebook(DT)
    plus = cb[: cb.shape[0] // 2] - 0.25
    sums = plus.sum(dim=1)
    assert torch.allclose(sums.remainder(2.0), torch.zeros_like(sums), atol=1e-9)
    assert plus.shape[0] == 256 * 2 ** 7


def test_bit_rate_is_exactly_two():
    """The whole reason for this quantizer: W drops from 4.15625 to 2.0."""
    assert Q.E8P_BITS_PER_WEIGHT == 2.0
    assert Q.E8P_INDEX_BITS / Q.E8P_DIM == 2.0

    import accounting as A
    assert A.vq_bits_from_spec(16, 8, weights_per_codebook=None) == 2.0
    # A 1 KiB table shared by every model amortizes to nothing measurable.
    assert A.vq_bits_from_spec(
        16, 8, entry_bits=16, weights_per_codebook=45.1e6
    ) > 2.0                                        # AQLM, by contrast, does not


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #

def test_codewords_quantize_to_themselves():
    cb = Q.e8p_codebook(DT)
    sample = cb[torch.randperm(cb.shape[0])[:64]]
    deq, idx, _ = Q.quantize_vectors(sample, scale=1.0)
    assert torch.allclose(deq, sample, atol=1e-12)
    assert torch.equal(cb[idx], sample)


def test_nearest_is_really_nearest():
    """Brute-force the argmin independently for a few vectors."""
    cb = Q.e8p_codebook(DT)
    x = torch.randn((8, 8), dtype=DT)
    _, idx, _ = Q.quantize_vectors(x, scale=1.0)
    for r in range(x.shape[0]):
        d = (cb - x[r]).square().sum(dim=1)
        assert int(d.argmin()) == int(idx[r])


def test_fit_scale_beats_an_arbitrary_scale():
    x = torch.randn((256, 8), dtype=DT) * 0.037          # weight-like magnitude
    cb = Q.e8p_codebook(DT)
    a = Q.fit_scale(x, cb)
    fitted, _, _ = Q.quantize_vectors(x, scale=a)
    naive, _, _ = Q.quantize_vectors(x, scale=1.0)
    assert Q.quantization_snr(x, fitted) > Q.quantization_snr(x, naive)


def test_beats_two_bit_scalar_quantization():
    """A lattice in 8 dimensions should beat rounding each weight to one of four
    levels at the same 2 bits/weight -- that is the entire premise."""
    x = torch.randn((512, 8), dtype=DT)
    vq, _, _ = Q.quantize_vectors(x)

    # 2-bit scalar RTN with a per-group scale, same bit budget.
    step = x.abs().max() / 1.5
    scalar = torch.clamp(torch.round(x / step), -2, 1) * step

    assert Q.quantization_snr(x, vq) > Q.quantization_snr(x, scalar) + 3.0


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("k", [8, 16, 24])
def test_quantize_blocks_shape_and_rate(k):
    blocks = torch.randn((3, 4, k), dtype=DT) * 0.05
    qb = Q.quantize_blocks(blocks)
    assert qb.values.shape == blocks.shape
    assert qb.padding == 0
    assert qb.indices.shape == (3, 4 * k // 8)
    assert qb.scales.shape == (3,)
    assert qb.bits_per_weight == 2.0
    assert Q.quantization_snr(blocks, qb.values) > 5.0


def test_padding_is_handled_and_dropped():
    blocks = torch.randn((2, 4, 20), dtype=DT) * 0.05     # 20 is not a multiple of 8
    qb = Q.quantize_blocks(blocks)
    assert qb.padding == 4
    assert qb.values.shape == blocks.shape


def test_quantize_blocks_validates():
    with pytest.raises(ValueError, match="3-D"):
        Q.quantize_blocks(torch.randn((4, 8), dtype=DT))
    with pytest.raises(ValueError, match=r"\[n, 8\]"):
        Q.quantize_vectors(torch.randn((4, 7), dtype=DT))


# --------------------------------------------------------------------------- #
# Why rotation is in the pipeline at all
# --------------------------------------------------------------------------- #

def test_rotation_improves_lattice_fit_on_outlier_heavy_blocks():
    """A lattice quantizer wants roughly isotropic input.  Rotation is what
    supplies it -- and survivors, being the fat tail of the weight
    distribution, are exactly the case that needs it (plan H5).
    """
    n_tiles, lpt, k = 4, 16, 64
    blocks = torch.randn((n_tiles, lpt, k), dtype=DT) * 0.01
    blocks[:, :, ::16] *= 40.0                            # a few fat channels

    cw = C.CompactWeights(
        blocks=blocks,
        line_index=torch.arange(n_tiles * lpt).view(n_tiles, lpt),
        idx_index=torch.arange(k).unsqueeze(0).expand(n_tiles, k).contiguous(),
        mask=None,
    )
    rotated, _ = R.rotate(cw, axis="index", seed=2)

    snr_plain = Q.quantization_snr(blocks, Q.quantize_blocks(blocks).values)
    snr_rot = Q.quantization_snr(
        rotated.blocks, Q.quantize_blocks(rotated.blocks).values
    )
    # Measured on this fixture: 5.4 -> 7.6 dB, so rotation buys ~2.2 dB at
    # 2 bits/weight.  Real, but worth keeping in proportion -- it is a
    # meaningful gain, not a rescue.
    assert snr_rot > snr_plain + 1.5, (
        f"rotation should help the lattice: {snr_plain:.1f} -> {snr_rot:.1f} dB"
    )


def test_full_pipeline_prune_compact_rotate_quantize():
    """The H1 order end to end, with the mask still intact at the far side."""
    n_out, n_in = 16, 64
    W = torch.randn((n_out, n_in), dtype=DT)
    X = torch.randn((256, n_in), dtype=DT) @ (
        torch.randn((n_in, n_in), dtype=DT) / math.sqrt(n_in)
    )

    r = P.prune(W, axis="B", tile_size=4, density=0.5,
                act_norm=X.norm(dim=0), H=X.T @ X, compensate=True)
    cw = C.compact(r.W, r.mask)
    rot, Qm = R.rotate(cw, axis="index", seed=1)
    qb = Q.quantize_blocks(rot.blocks)

    back = R.unrotate(rot.with_blocks(qb.values), Qm, axis="index")
    recon = C.scatter(back)

    assert torch.equal(recon != 0, r.mask.expand())
    assert Q.quantization_snr(r.W, recon) > 5.0


# --------------------------------------------------------------------------- #
# LDLQ -- Hessian-aware rounding
# --------------------------------------------------------------------------- #

def _weighted_error(block: torch.Tensor, hat: torch.Tensor, H: torch.Tensor) -> float:
    """tr(E H E^T) -- the objective, as opposed to ||E||^2."""
    E = block - hat
    return float(torch.einsum("ij,jk,ik->", E, H, E))


def test_ldlq_reduces_to_plain_rounding_when_the_hessian_is_isotropic():
    """With H = I there is no cheap direction to push error into, so the error
    feedback term vanishes and LDLQ must agree with nearest-neighbour exactly.
    Any disagreement means the propagation is firing when it should not."""
    block = torch.randn((12, 32), dtype=DT) * 0.05
    H = torch.eye(32, dtype=DT)

    cb = Q.e8p_codebook(DT)
    scale = Q.fit_scale(block.reshape(-1, 8), cb)
    ldlq = Q.ldlq_quantize(block, H, scale=scale)

    plain, _, _ = Q.quantize_vectors(block.reshape(-1, 8), scale=scale)
    assert torch.allclose(ldlq.values, plain.reshape(block.shape), atol=1e-12)


def test_ldlq_beats_plain_rounding_on_the_weighted_objective():
    """The point of the exercise: minimize tr(E H E^T), not ||E||^2."""
    torch.manual_seed(3)
    n_lines, k = 16, 64
    block = torch.randn((n_lines, k), dtype=DT) * 0.05
    A = torch.randn((k, k), dtype=DT) / math.sqrt(k)
    X = torch.randn((512, k), dtype=DT) @ A
    H = X.T @ X

    cb = Q.e8p_codebook(DT)
    scale = Q.fit_scale(block.reshape(-1, 8), cb)
    ldlq = Q.ldlq_quantize(block, H, scale=scale)
    plain, _, _ = Q.quantize_vectors(block.reshape(-1, 8), scale=scale)
    plain = plain.reshape(block.shape)

    assert _weighted_error(block, ldlq.values, H) < _weighted_error(block, plain, H)
    # It buys this by accepting MORE plain-L2 error, in cheaper directions.
    assert (block - ldlq.values).square().sum() > 0


def test_ldlq_needs_the_index_axis_aligned_to_eight():
    with pytest.raises(ValueError, match="multiple of 8"):
        Q.ldlq_quantize(torch.randn((4, 20), dtype=DT), torch.eye(20, dtype=DT))


def test_ldlq_validates_the_hessian_shape():
    with pytest.raises(ValueError, match="to match the block"):
        Q.ldlq_quantize(torch.randn((4, 16), dtype=DT), torch.eye(8, dtype=DT))
    with pytest.raises(ValueError, match="2-D"):
        Q.ldlq_quantize(torch.randn((2, 4, 16), dtype=DT), torch.eye(16, dtype=DT))


def test_ldlq_quantize_blocks_shapes():
    n_tiles, lpt, k = 3, 8, 32
    blocks = torch.randn((n_tiles, lpt, k), dtype=DT) * 0.05
    H = torch.eye(k, dtype=DT).unsqueeze(0).expand(n_tiles, k, k).contiguous()
    qb = Q.ldlq_quantize_blocks(blocks, H)
    assert qb.values.shape == blocks.shape
    assert qb.indices.shape == (n_tiles, lpt * k // 8)
    assert qb.bits_per_weight == 2.0
    assert qb.padding == 0

    with pytest.raises(ValueError, match="hessians must be"):
        Q.ldlq_quantize_blocks(blocks, torch.eye(k, dtype=DT))


def _rotation_gain(block: torch.Tensor, H: torch.Tensor, k: int) -> dict:
    V = R.structured_orthogonal(k, seed=1, dtype=DT)
    block_rot, H_rot = block @ V.T, V @ H @ V.T
    cb = Q.e8p_codebook(DT)

    def err(b, h, use_ldlq):
        s = Q.fit_scale(b.reshape(-1, 8), cb)
        if use_ldlq:
            hat = Q.ldlq_quantize(b, h, scale=s).values
        else:
            q, _, _ = Q.quantize_vectors(b.reshape(-1, 8), scale=s)
            hat = q.reshape(b.shape)
        return _weighted_error(b, hat, h)

    return {
        "nn_gain": err(block_rot, H_rot, False) / err(block, H, False),
        "ldlq_gain": err(block_rot, H_rot, True) / err(block, H, True),
        "best": min(err(block, H, True), err(block_rot, H_rot, True)),
        "nn_plain": err(block, H, False),
    }


def test_rotation_pays_only_when_the_weights_are_not_already_isotropic():
    """What rotation is actually for -- and what it is NOT for.

    Measured on 16x64 blocks against a correlated Hessian:

        Gaussian weights      rotation: NN +17.5%   LDLQ  +4.8%   (a loss)
        heavy-tailed weights  rotation: NN -61.7%   LDLQ -39.0%   (a large win)

    An RHT exists to spread outliers.  Gaussian weights have none, so rotating
    them only costs.  Survivors are the fat tail of the weight distribution by
    construction, which is exactly the case that benefits -- and the reason the
    rotation belongs in this pipeline at all (plan sections C, H5).

    Note this is narrower than "rotation only pays with LDLQ".  On heavy-tailed
    blocks the rotation helps plain rounding too.  What LDLQ adds is the lower
    absolute error, and the reliability of the gain in the full pipeline.
    """
    torch.manual_seed(5)
    k = 64
    A = torch.randn((k, k), dtype=DT) / math.sqrt(k)
    X = torch.randn((512, k), dtype=DT) @ A
    H = X.T @ X

    gaussian = torch.randn((16, k), dtype=DT) * 0.05
    heavy = torch.randn((16, k), dtype=DT) * 0.01 * torch.exp(
        torch.randn((16, k), dtype=DT) * 1.2
    )

    g = _rotation_gain(gaussian, H, k)
    h = _rotation_gain(heavy, H, k)

    assert g["nn_gain"] > 1.0 and g["ldlq_gain"] > 1.0, (
        "rotating already-isotropic weights should not help"
    )
    assert h["nn_gain"] < 0.8 and h["ldlq_gain"] < 0.8, (
        "rotating heavy-tailed weights should help substantially"
    )
    assert h["best"] < h["nn_plain"], "LDLQ+rotation should be the best combination"


# --------------------------------------------------------------------------- #
# Streaming the sub-Hessians
# --------------------------------------------------------------------------- #

def test_streamed_hessians_give_bit_identical_results():
    """The memory fix must be a pure refactor.

    LDLQ consumes tiles in order, so handing it a callable instead of a stacked
    tensor changes nothing about the arithmetic -- and at real widths it is the
    difference between 119 GiB and 239 MiB.  Identical to the last bit, not
    approximately: any drift would mean the callable is serving a different
    tile than the index it was given.
    """
    g = torch.Generator().manual_seed(0)
    blocks = torch.randn(5, 6, 16, generator=g, dtype=torch.float64)
    hs = []
    for _ in range(5):
        a = torch.randn(16, 16, generator=g, dtype=torch.float64)
        hs.append(a @ a.T + torch.eye(16, dtype=torch.float64))
    stacked = torch.stack(hs)

    a = Q.ldlq_quantize_blocks(blocks, stacked)
    b = Q.ldlq_quantize_blocks(blocks, lambda t: stacked[t])
    assert torch.equal(a.values, b.values)
    assert torch.equal(a.indices, b.indices)
    assert torch.equal(a.scales, b.scales)


def test_a_streamed_hessian_of_the_wrong_shape_is_rejected():
    """Named per tile, because a callable cannot be shape-checked up front and a
    silently wrong tile would just produce worse numbers."""
    g = torch.Generator().manual_seed(0)
    blocks = torch.randn(3, 4, 16, generator=g, dtype=torch.float64)
    good = torch.eye(16, dtype=torch.float64)
    bad = torch.eye(8, dtype=torch.float64)
    with pytest.raises(ValueError, match="tile 1"):
        Q.ldlq_quantize_blocks(blocks, lambda t: good if t == 0 else bad)


# --------------------------------------------------------------------------- #
# Lattice decoding instead of scanning
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scale", [0.5, 1.0, 1.6, 3.0])
def test_lattice_decoding_agrees_with_the_scan_exactly(scale):
    """The only claim that matters: same answer, not a close one.

    The codebook is a lattice INTERSECTED with a norm ball plus 29 arbitrary
    padding patterns, so decoding to the nearest lattice point is not by itself
    the nearest codeword.  `nearest_e8p` only commits where it can prove the
    result and hands the rest back to the scan; if that logic were wrong it
    would show up here as a slightly-worse codeword, not a crash.
    """
    torch.manual_seed(0)
    cb = Q._on_device(torch.float64, "cpu")
    x = torch.randn(4000, 8, dtype=torch.float64) * scale
    fast_i, fast_c = Q._nearest(x, cb)
    slow_i, slow_c = Q._brute_force(x, cb)
    assert torch.equal(fast_i, slow_i)
    assert torch.equal(fast_c, slow_c)


def test_the_exactness_flag_is_honest():
    """Where the decoder claims exactness it must BE exact, and where it does
    not the caller must fall back -- a flag that over-claims would quietly
    degrade every quantized weight."""
    torch.manual_seed(1)
    cb = Q._on_device(torch.float64, "cpu")
    x = torch.randn(4000, 8, dtype=torch.float64) * 1.3
    idx, code, exact = Q.nearest_e8p(x)
    slow_i, _ = Q._brute_force(x, cb)
    assert exact.any() and not exact.all(), "need both branches represented"
    assert torch.equal(idx[exact], slow_i[exact])


def test_every_codeword_decodes_to_itself():
    """A codeword is its own nearest codeword, so the decoder must return it
    unchanged -- including the 29 padding patterns, which are the ones a
    norm-ball test would wrongly reject."""
    cb = Q._on_device(torch.float64, "cpu")
    idx, code, exact = Q.nearest_e8p(cb)
    assert bool(exact.all())
    assert torch.equal(code, cb)
    assert torch.equal(idx, torch.arange(cb.shape[0]))


def test_the_decoder_lands_in_the_lattice():
    """Half-integers with an even coordinate sum -- D8 + 1/2.  Both properties
    matter: the parity fix is the only reason the second-nearest coordinate is
    ever chosen."""
    torch.manual_seed(2)
    y = torch.randn(2000, 8, dtype=torch.float64) * 2.0
    h = Q._nearest_halfinteger_even(y)
    assert torch.all(((h - 0.5) % 1.0).abs() < 1e-12)
    assert torch.all((h.sum(dim=-1) % 2).abs() < 1e-12)
    # and it beats naive rounding, which ignores the parity constraint
    naive = torch.floor(y) + 0.5
    bad = (naive.sum(dim=-1) % 2).abs() > 1e-12
    assert bad.any()
    assert torch.all((y - h).square().sum(-1) >= (y - naive).square().sum(-1) - 1e-12)


def test_the_source_index_table_round_trips_and_rejects_outsiders():
    table = Q._source_index_table()
    S = Q.source_codebook(torch.float64)
    powers = Q._LEVELS ** torch.arange(Q.E8P_DIM, dtype=torch.int64)
    keys = (((S - 0.5).round().to(torch.int64)) * powers).sum(dim=1)
    assert torch.equal(table[keys], torch.arange(S.shape[0]))
    assert int((table >= 0).sum()) == S.shape[0]


def test_the_codebook_cache_returns_one_object_per_device():
    """The dispatch is an identity check, so a fresh `.to()` copy each call
    would silently disable the fast path -- which is exactly what happened
    before this cache existed."""
    a = Q._on_device(torch.float64, "cpu")
    b = Q._on_device(torch.float64, "cpu")
    assert a is b


def test_small_batches_keep_the_scan_and_still_agree():
    """Below the crossover the decoder's fixed cost is not worth paying, so the
    scan runs instead.  Correctness must not depend on which side of the
    threshold a call lands."""
    torch.manual_seed(3)
    cb = Q._on_device(torch.float64, "cpu")
    tiny = torch.randn(8, 8, dtype=torch.float64)
    assert tiny.shape[0] < Q._LATTICE_MIN_ROWS["cpu"]
    assert torch.equal(Q._nearest(tiny, cb)[0], Q._brute_force(tiny, cb)[0])


def test_a_foreign_codebook_still_goes_through_the_scan():
    """`_nearest` is not E8P-only; anything that is not the cached table has to
    fall through, or a caller with its own codebook would get E8P answers."""
    torch.manual_seed(4)
    other = torch.randn(512, 8, dtype=torch.float64)
    x = torch.randn(200, 8, dtype=torch.float64)
    idx, code = Q._nearest(x, other)
    assert idx.max() < other.shape[0]
    assert torch.equal(code, other[idx])


# --------------------------------------------------------------------------- #
# BLOCK-DIAGONAL FEEDBACK  (docs/STATUS.md section 6.3)
# --------------------------------------------------------------------------- #
# The project's largest measured cost is the LDLQ factorization: it runs once
# per tile because each tile owns a different column set, at k^3 each, and at
# T=4 that is 2.6e16 flops per pass over Llama-2-7B.  `hessian_block=b` is the
# only lever that changes the exponent, taking it to k*b^2.
#
# It buys that by dropping the sub-Hessian couplings that reach past width b,
# which is an approximation of the OBJECTIVE, not of the arithmetic.  These
# tests pin exactly what is dropped, so the price is measurable rather than
# hopeful; `experiments/m0_rotation_value.py` measures it.

def _spd(k: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    A = torch.randn((k, k), dtype=DT) / math.sqrt(k)
    X = torch.randn((8 * k, k), dtype=DT) @ A
    return X.T @ X / X.shape[0] + torch.eye(k, dtype=DT)


def _blockdiag_part(H: torch.Tensor, block: int) -> torch.Tensor:
    out = torch.zeros_like(H)
    for off, width in R.block_partition(H.shape[-1], block):
        out[off:off + width, off:off + width] = H[off:off + width,
                                                  off:off + width]
    return out


@pytest.mark.parametrize("block", [64, 128])
def test_full_width_feedback_is_the_unconstrained_sweep(block):
    """`hessian_block >= k` must be bit-identical to `None`, so the widest arm
    of a sweep is provably the same computation the -70% result used."""
    k = 64
    H, W = _spd(k), torch.randn((16, k), dtype=DT) * 0.05
    ref = Q.ldlq_quantize(W, H)
    got = Q.ldlq_quantize(W, H, hessian_block=block)
    assert torch.equal(got.values, ref.values)
    assert torch.equal(got.indices, ref.indices)


@pytest.mark.parametrize("block", [8, 16, 32])
def test_feedback_block_is_exactly_ldlq_against_the_block_diagonal_hessian(block):
    """The claim, stated so it can fail.

    Zeroing H's off-block entries leaves its DIAGONAL untouched, so the damping
    term -- percdamp times the mean of the whole diagonal -- is identical in
    both runs.  The two therefore differ in nothing but the dropped couplings,
    and must agree exactly."""
    k = 64
    H, W = _spd(k), torch.randn((16, k), dtype=DT) * 0.05
    got = Q.ldlq_quantize(W, H, hessian_block=block)
    ref = Q.ldlq_quantize(W, _blockdiag_part(H, block))
    assert torch.equal(got.values, ref.values)


@pytest.mark.parametrize("block", [8, 16, 32])
def test_no_error_crosses_a_block_boundary(block):
    """Operational proof of independence: perturb only the couplings that reach
    across a boundary and the output must not move at all.  Without the
    constraint the same perturbation does move it -- otherwise the test would
    pass for a quantizer that ignores H entirely."""
    k = 64
    H, W = _spd(k), torch.randn((16, k), dtype=DT) * 0.05
    noise = _spd(k, seed=7) - _blockdiag_part(_spd(k, seed=7), block)
    perturbed = H + 0.5 * (noise + noise.T) / 2

    assert torch.equal(Q.ldlq_quantize(W, H, hessian_block=block).values,
                       Q.ldlq_quantize(W, perturbed, hessian_block=block).values)
    assert not torch.equal(Q.ldlq_quantize(W, H).values,
                           Q.ldlq_quantize(W, perturbed).values)


def test_feedback_block_must_not_split_a_codeword():
    """A boundary inside an E8P group would mean eight coordinates quantized as
    one vector while their feedback came from two different factorizations."""
    k = 64
    H, W = _spd(k), torch.randn((16, k), dtype=DT) * 0.05
    with pytest.raises(ValueError, match="multiple of the quantizer group"):
        Q.ldlq_quantize(W, H, hessian_block=12)


def test_feedback_block_reaches_the_tile_loop():
    """`ldlq_quantize_blocks` must pass it through -- the streaming path is the
    only one a real layer ever takes."""
    n_tiles, lpt, k, block = 3, 4, 32, 8
    blocks = torch.randn((n_tiles, lpt, k), dtype=DT) * 0.05
    H = _spd(k)
    got = Q.ldlq_quantize_blocks(blocks, lambda t: H, hessian_block=block)
    ref = Q.ldlq_quantize_blocks(blocks, lambda t: _blockdiag_part(H, block))
    assert torch.equal(got.values, ref.values)
    assert not torch.equal(got.values,
                           Q.ldlq_quantize_blocks(blocks, lambda t: H).values)


# --------------------------------------------------------------------------- #
# CHUNKED SWEEP
# --------------------------------------------------------------------------- #
# The sweep is not compute-bound: measured on this machine a group costs
# 0.248 ms of wall time against 0.0034 ms of arithmetic, because a [lines, 8]
# search against 65536 codewords cannot fill a GPU and there are k/8 of them in
# a row.  Tiles are independent given their own Hessians, so the group loop can
# be hoisted out of the tile loop and C tiles quantized at each group together.
# Measured 5-12x, most at lines=4 -- the grid's most expensive column.
#
# Because the arithmetic per tile is untouched, the ONLY acceptable outcome is
# bit-identical output.  That is what these check; a speedup that changed a
# number would be a different pipeline, not a faster one.

@pytest.mark.parametrize("hessian_block", [None, 64])
@pytest.mark.parametrize("chunk", [2, 5, 12, 48])
def test_chunking_the_sweep_changes_nothing(chunk, hessian_block):
    torch.manual_seed(0)
    n_tiles, lines, k = 12, 8, 256
    blocks = torch.randn((n_tiles, lines, k), dtype=DT) * 0.05
    hs = torch.stack([_spd(k, seed=t) for t in range(n_tiles)])

    ref = Q.ldlq_quantize_blocks(blocks, lambda t: hs[t], scale=0.05,
                                 hessian_block=hessian_block, chunk=1)
    got = Q.ldlq_quantize_blocks(blocks, lambda t: hs[t], scale=0.05,
                                 hessian_block=hessian_block, chunk=chunk)
    assert torch.equal(got.values, ref.values)
    assert torch.equal(got.indices, ref.indices)
    assert torch.equal(got.scales, ref.scales)


def test_chunking_preserves_the_per_tile_scale():
    """A chunk fits one alpha per member, not one for the chunk.  Getting this
    wrong would silently turn `per_tile` into something between per-tile and
    per-layer -- and per-layer was measured 11% worse."""
    torch.manual_seed(1)
    n_tiles, lines, k = 6, 8, 128
    blocks = torch.stack([torch.randn((lines, k), dtype=DT) * (0.01 * (t + 1))
                          for t in range(n_tiles)])
    hs = torch.stack([_spd(k, seed=t) for t in range(n_tiles)])

    ref = Q.ldlq_quantize_blocks(blocks, lambda t: hs[t], chunk=1)
    got = Q.ldlq_quantize_blocks(blocks, lambda t: hs[t], chunk=n_tiles)
    assert torch.equal(got.scales, ref.scales)
    assert len(set(ref.scales.tolist())) > 1        # the tiles really do differ
    assert torch.equal(got.values, ref.values)


def test_a_ragged_final_chunk_is_handled():
    """n_tiles need not divide the chunk, and the tail must not be dropped."""
    torch.manual_seed(2)
    n_tiles, lines, k = 7, 4, 64
    blocks = torch.randn((n_tiles, lines, k), dtype=DT) * 0.05
    hs = torch.stack([_spd(k, seed=t) for t in range(n_tiles)])
    got = Q.ldlq_quantize_blocks(blocks, lambda t: hs[t], scale=0.05, chunk=3)
    ref = Q.ldlq_quantize_blocks(blocks, lambda t: hs[t], scale=0.05, chunk=1)
    assert got.values.shape == blocks.shape
    assert got.indices.shape == (n_tiles, lines * k // 8)
    assert torch.equal(got.values, ref.values)


def test_chunk_is_validated():
    blocks = torch.randn((2, 4, 64), dtype=DT) * 0.05
    with pytest.raises(ValueError, match="chunk must be positive"):
        Q.ldlq_quantize_blocks(blocks, lambda t: _spd(64), chunk=0)


def test_auto_chunk_is_bounded_by_memory_and_by_saturation():
    """Two ceilings.  Memory is usually the binding one, and it is why
    `hessian_block` is what makes a useful chunk affordable at all."""
    k, lines, item = 7912, 16, 4
    confined = Q.auto_chunk(1000, lines, k, item, hessian_block=512)
    full = Q.auto_chunk(1000, lines, k, item, hessian_block=None)
    assert full == 4                       # k^2 * 4 = 250 MiB, four in a GiB
    assert confined == 64                  # k*512 * 4 = 16 MiB, so saturation binds
    assert confined > 8 * full

    # Saturation caps it even when memory would allow more.
    assert Q.auto_chunk(10_000, 4096, 1024, item, hessian_block=512) == 1
    assert Q.auto_chunk(10_000, 16, 512, item, hessian_block=512) == 64

    # Never more tiles than exist, never fewer than one.
    assert Q.auto_chunk(3, 16, 512, item, hessian_block=512) == 3
    assert Q.auto_chunk(1000, 1, 8192, item, hessian_block=None,
                        budget_bytes=1) == 1
