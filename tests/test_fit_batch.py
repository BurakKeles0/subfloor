"""Fitting the tiles' scales together must not change any tile's scale.

`docs/STATUS.md` section 7.2 turned this lever down as "not bit-identical: it
reduces every tile's error together".  That is a statement about an
implementation, not about the idea -- and this project has been wrong in both
directions on exactly that distinction: `compensate_block` was rejected on the
same grounds and costs nothing, while `search_dtype=float16` really did move the
answer.

So what is asserted here is the property the rejection was about: each tile
keeps its own alpha, computed from its own vectors, reduced over its own
[n, 8].  The speed is measured elsewhere; a test that needed a GPU to be idle
could only run when the thing it protects is already unnecessary.
"""

from __future__ import annotations

import pytest
import torch

import calibrate as Cal
import m1_gates as M
import quantize as Qz


def _blocks(n_tiles: int, n: int, seed: int = 0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n_tiles, n, Qz.E8P_DIM, generator=g, dtype=dtype) * 0.05)


@pytest.mark.parametrize("n_tiles,n", [(8, 128), (16, 64), (3, 736), (32, 40)])
def test_batching_across_tiles_reproduces_every_tile_scale(n_tiles, n):
    """The whole question, asked directly.

    `n=40` is in the window where batching CHANGES THE ROUTING inside
    `_nearest`: on cuda a tile's 24 candidates are 960 rows, below the lattice
    decoder's 1024-row floor, so alone they take the analytic search and packed
    together they take the decoder.  Both are exact -- `_nearest`'s docstring
    says so -- and this is the test that holds it, because if it stopped being
    true this lever would silently start moving the answer at the fine end.
    """
    cb = Qz._on_device(torch.float32, "cpu")
    x = _blocks(n_tiles, n)
    one = [Qz.fit_scale(x[t], cb, seed_rng=t) for t in range(n_tiles)]
    many = Qz.fit_scales(x, cb, seed_rng=0)
    assert many == one


def test_a_zero_tile_gets_the_same_answer_either_way():
    """`fit_scale` short-circuits an all-zero tile to 1.0 and never searches.

    The batched form has to reproduce that WITHOUT letting the tile into the
    stack -- its seed would be 0/rms and every candidate 0, which divides.
    """
    cb = Qz._on_device(torch.float32, "cpu")
    x = _blocks(4, 128)
    x[2] = 0.0
    one = [Qz.fit_scale(x[t], cb, seed_rng=t) for t in range(4)]
    many = Qz.fit_scales(x, cb, seed_rng=0)
    assert one[2] == 1.0 and many[2] == 1.0
    assert many == one


def test_sampling_draws_the_same_subset_per_tile():
    """`scale_sample` seeds the RNG per tile (`scale_seed + t`).

    Getting that wrong is invisible in the alpha's magnitude and fatal to the
    comparison: a shared subset correlates the tiles' scales in a way a full fit
    never does, which is the reason the per-tile offset exists at all.
    """
    cb = Qz._on_device(torch.float32, "cpu")
    x = _blocks(6, 512)
    one = [Qz.fit_scale(x[t], cb, sample=64, seed_rng=7 + t) for t in range(6)]
    many = Qz.fit_scales(x, cb, sample=64, seed_rng=7)
    assert many == one
    # And it really is per tile: one offset off, and the answers move.
    shifted = Qz.fit_scales(x, cb, sample=64, seed_rng=8)
    assert shifted != many


def test_the_sweep_takes_the_batched_path_and_lands_in_the_same_place():
    """The caller, not the part.  `batch_fit` has to REACH the fit.

    A record can say `batch_fit=True` while the argument goes nowhere, which is
    how `compensate_block` stayed unreachable from the driver for a day
    (section 6.12).  So the path is counted, and the output is required to be
    bit-identical -- which, unlike the compensation lever, it is.
    """
    problem = Cal.synthetic_problem(64, 128, 256)
    seen = {"one": 0, "many": 0}
    real_one, real_many = Qz.fit_scale, Qz.fit_scales

    def spy_one(*a, **k):
        seen["one"] += 1
        return real_one(*a, **k)

    def spy_many(*a, **k):
        seen["many"] += 1
        return real_many(*a, **k)

    Qz.fit_scale, Qz.fit_scales = spy_one, spy_many
    try:
        off = M.run_config(problem, budget_bits=1.5, tile_size=4,
                           batch_fit=False, return_weight=True)
        n_per_tile, seen["one"] = seen["one"], 0
        on = M.run_config(problem, budget_bits=1.5, tile_size=4,
                          batch_fit=True, return_weight=True)
    finally:
        Qz.fit_scale, Qz.fit_scales = real_one, real_many

    assert n_per_tile > 1, "the per-tile arm fitted once; nothing to batch"
    assert seen["many"] >= 1 and seen["one"] == 0, (
        f"batch_fit=True did not reach fit_scales: {seen}")
    assert off["batch_fit"] is False and on["batch_fit"] is True
    assert torch.equal(off["W_hat"], on["W_hat"])
    assert off["rel_output_error"] == on["rel_output_error"]


def test_the_pipeline_default_is_off_and_says_so():
    """Off until it is decided, and the decision is the user's.

    Recorded rather than assumed: the measurement says it costs nothing and
    buys time at the fine end, but every quality number in this project was
    taken with it off and turning it on is a change of configuration.
    """
    problem = Cal.synthetic_problem(64, 128, 256)
    assert M.PIPELINE_BATCH_FIT is False
    assert M.run_config(problem, budget_bits=1.5,
                        tile_size=4)["batch_fit"] is False


def test_a_stack_that_is_not_three_dimensional_is_refused():
    cb = Qz._on_device(torch.float32, "cpu")
    with pytest.raises(ValueError, match=r"\[n_tiles, n, width\]"):
        Qz.fit_scales(torch.zeros(128, Qz.E8P_DIM), cb)
