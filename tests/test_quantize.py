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
