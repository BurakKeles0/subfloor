"""Sequential calibration.

`test_next_block_sees_the_compressed_model` is the one that matters.  Spec v6
trap 20 forbids calibrating from the dense model, and the difference is
invisible unless you go looking: both orders run, both produce Hessians, and
only one of them describes the model you are actually building.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import calibrate as Cal

DT = torch.float64


class TinyBlock(nn.Module):
    """Two linears with a nonlinearity, enough to have real statistics."""

    def __init__(self, d_in: int, d_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(torch.relu(self.fc1(x)))


def _blocks(n: int = 2, d: int = 16, h: int = 24, seed: int = 0) -> list[TinyBlock]:
    torch.manual_seed(seed)
    return [TinyBlock(d, h).double() for _ in range(n)]


def _inputs(n_batches: int = 2, batch: int = 4, seqlen: int = 6, d: int = 16):
    torch.manual_seed(1)
    return [torch.randn((batch, seqlen, d), dtype=DT) for _ in range(n_batches)]


# --------------------------------------------------------------------------- #
# LayerProblem
# --------------------------------------------------------------------------- #

def test_output_error_agrees_whether_it_goes_through_x_or_h():
    """Real layers cannot keep X, so the objective is computed from H.  The two
    routes must give the same number:  ||X E^T||_F^2 = tr(E H E^T)."""
    torch.manual_seed(0)
    W = torch.randn((8, 12), dtype=DT)
    X = torch.randn((64, 12), dtype=DT) @ (torch.randn((12, 12), dtype=DT) / 3.5)
    W_hat = W + torch.randn_like(W) * 0.05

    from_x = Cal.LayerProblem(W, X)
    from_h = Cal.LayerProblem.from_statistics(W, X.T @ X)

    assert from_x.output_error(W_hat) == pytest.approx(
        from_h.output_error(W_hat), rel=1e-10
    )
    # And the direct definition, for good measure.
    ref = X @ W.T
    direct = float(((X @ W_hat.T - ref).square().sum() / ref.square().sum()).sqrt())
    assert from_h.output_error(W_hat) == pytest.approx(direct, rel=1e-10)


def test_from_statistics_derives_act_norm():
    torch.manual_seed(0)
    X = torch.randn((64, 12), dtype=DT)
    p = Cal.LayerProblem.from_statistics(torch.randn((8, 12), dtype=DT), X.T @ X)
    assert torch.allclose(p.act_norm, X.norm(dim=0), atol=1e-10)


def test_layer_problem_validates():
    W = torch.randn((8, 12), dtype=DT)
    with pytest.raises(ValueError, match="either X or"):
        Cal.LayerProblem(W)
    with pytest.raises(ValueError, match="input channels"):
        Cal.LayerProblem(W, torch.randn((4, 7), dtype=DT))
    with pytest.raises(ValueError, match="to match W"):
        Cal.LayerProblem.from_statistics(W, torch.eye(7, dtype=DT))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_hessian_accumulator_matches_the_direct_product():
    torch.manual_seed(0)
    acc = Cal.HessianAccumulator(5, dtype=DT)
    chunks = [torch.randn((3, 7, 5), dtype=DT) for _ in range(3)]
    for c in chunks:
        acc.update(c)
    flat = torch.cat([c.reshape(-1, 5) for c in chunks])
    assert torch.allclose(acc.H, flat.T @ flat, atol=1e-10)
    assert acc.n_tokens == flat.shape[0]
    assert torch.allclose(acc.act_norm, flat.norm(dim=0), atol=1e-10)


def test_find_linears_walks_nested_modules():
    block = TinyBlock(4, 6)
    found = Cal.find_linears(block)
    assert set(found) == {"fc1", "fc2"}
    assert found["fc1"].in_features == 4
    assert found["fc2"].in_features == 6


def test_collect_block_statistics_captures_each_linear_input():
    """fc2 sees the hidden activation, not the block input -- so its Hessian
    must be the hidden width, and must match a hand-computed forward pass."""
    block = _blocks(1)[0]
    inputs = _inputs()
    accs = Cal.collect_block_statistics(block, inputs)

    assert accs["fc1"].H.shape == (16, 16)
    assert accs["fc2"].H.shape == (24, 24)

    with torch.no_grad():
        hidden = torch.cat([torch.relu(block.fc1(b)).reshape(-1, 24) for b in inputs])
    assert torch.allclose(accs["fc2"].H, hidden.T @ hidden, atol=1e-9)


def test_collect_rejects_a_block_with_no_linears():
    with pytest.raises(ValueError, match="no nn.Linear"):
        Cal.collect_block_statistics(nn.ReLU(), _inputs())


# --------------------------------------------------------------------------- #
# THE TRAP  (Spec v6 section 7, trap 20)
# --------------------------------------------------------------------------- #

def test_next_block_sees_the_compressed_model():
    """Statistics for block 1 must be gathered from the output of the COMPRESSED
    block 0, not the dense one.

    Compressing block 0 differently has to change what block 1 sees.  If the
    loop ran the dense block to produce the next inputs, the two runs below
    would produce identical Hessians and the bug would be silent.
    """
    def run(scale: float) -> torch.Tensor:
        blocks = _blocks(2)
        seen: dict[str, torch.Tensor] = {}

        def compress(i, name, problem):
            if i == 1:
                seen[name] = problem.H.clone()
            return problem.W * (scale if i == 0 else 1.0)

        Cal.sequential_calibrate(blocks, list(_inputs()), compress)
        return seen["fc1"]

    untouched = run(1.0)
    squashed = run(0.05)

    assert not torch.allclose(untouched, squashed, rtol=1e-3), (
        "block 1's Hessian did not change when block 0 was compressed -- "
        "calibration is reading the dense model (trap 20)"
    )


def test_compression_actually_lands_on_the_weights():
    blocks = _blocks(1)
    before = blocks[0].fc1.weight.data.clone()
    Cal.sequential_calibrate(
        blocks, list(_inputs()), lambda i, n, p: p.W * 0.5
    )
    assert torch.allclose(blocks[0].fc1.weight.data, before * 0.5, atol=1e-10)


def test_records_carry_the_per_layer_error():
    blocks = _blocks(2)
    records = Cal.sequential_calibrate(
        blocks, list(_inputs()), lambda i, n, p: p.W
    )
    assert len(records) == 4                       # 2 blocks x 2 linears
    assert [r["name"] for r in records[:2]] == ["fc1", "fc2"]
    for r in records:
        assert r["rel_output_error"] == pytest.approx(0.0, abs=1e-12)
        assert r["n_tokens"] == 48                 # 2 batches x 4 x 6
        assert r["layer"].startswith("blocks.")


def test_lossy_compression_shows_up_in_the_error():
    blocks = _blocks(1)
    records = Cal.sequential_calibrate(
        blocks, list(_inputs()), lambda i, n, p: p.W * 0.5
    )
    assert all(r["rel_output_error"] > 0.1 for r in records)


def test_compress_fn_must_return_the_same_shape():
    blocks = _blocks(1)
    with pytest.raises(ValueError, match="expected"):
        Cal.sequential_calibrate(
            blocks, list(_inputs()), lambda i, n, p: p.W[:, :-1]
        )


def test_a_finished_layer_s_hessian_is_released_before_the_next_one():
    """The fix that made a block 1.46x faster, and the only kind of test that
    can hold it.

    A block's seven accumulators are 846 MB at Llama-2-7B's widths and one
    layer's compression peaks at 5.4 GiB against 6.8 usable, so holding all
    seven for the whole block leaves the allocator evicting and re-requesting
    instead of reusing.  Measured on a real block: 122.7 s holding against
    84.2 s releasing, while the peak moved only 5.40 -> 5.02 GiB.  The win is
    the room to reuse, not the peak -- which is why a memory-ceiling test would
    not see it either.

    Nothing about the ANSWER changes, so no correctness test can catch a
    regression here.  What can is the reference itself: take a weakref to one
    layer's Hessian and require it to be dead by the time a later layer is
    compressed.
    """
    import gc
    import weakref

    blocks = _blocks(2)
    seen: dict[str, object] = {}
    order: list[str] = []

    def compress(i, name, problem):
        order.append(name)
        # Both indices are in the SAME block -- a check that straddled two
        # blocks would pass on its own, since each block builds a fresh
        # accumulator dict and the previous one goes out of scope regardless.
        # That vacuous version was written first and survived the mutation.
        if len(order) == 1:
            seen["first"] = weakref.ref(problem.H)
        elif len(order) == 2:
            gc.collect()
            assert seen["first"]() is None, (
                f"the Hessian of {order[0]} was still alive while {name} was "
                "being compressed in the same block; every accumulator is held "
                "at once and the allocator has nothing to reuse"
            )
            seen["checked"] = True
        return problem.W

    Cal.sequential_calibrate(blocks, list(_inputs()), compress)
    assert seen.get("checked"), "the block had too few layers to check"


def test_sequential_calibrate_validates_inputs():
    with pytest.raises(ValueError, match="no blocks"):
        Cal.sequential_calibrate([], _inputs(), lambda i, n, p: p.W)
    with pytest.raises(ValueError, match="no calibration batches"):
        Cal.sequential_calibrate(_blocks(1), [], lambda i, n, p: p.W)


# --------------------------------------------------------------------------- #
# End to end with the compression pipeline
# --------------------------------------------------------------------------- #

def test_pipeline_plugs_into_calibration():
    """The seam: run_config consumes a LayerProblem, and calibration produces
    them one layer at a time."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    import m1_gates as M

    blocks = _blocks(2, d=32, h=32)
    calls = []

    def compress(i, name, problem):
        r = M.run_config(problem, budget_bits=1.5, tile_size=4,
                         ldlq=False, seed=0)
        calls.append(r)
        # run_config reports metrics; rebuild the weight through the same path.
        return problem.W

    records = Cal.sequential_calibrate(blocks, list(_inputs(d=32)), compress)
    assert len(records) == 4
    assert all("skipped" not in c for c in calls)
    assert all(c["bits_realized"] == pytest.approx(1.5, abs=1e-9) for c in calls)


def test_block_offset_reaches_compress_fn_and_the_layer_name():
    """A resumed run is handed `blocks[start:]`, so the loop index is not the
    block's index in the model.  Every number a caller can see has to be the
    model's.

    This was wrong in two places at once and neither could fail loudly.  The
    record's `block` field was absolute while its `layer` field was built from
    the loop index, so one record carried two different block numbers; and
    `compress_fn` was handed the loop index, so a `compress_fn` that branches
    on "is this block 0" -- which is how `experiments/m1_run.py` decides where
    to take the dense-E8P early warning for section 3.2, the check on this
    project's largest single risk -- fired on whichever block a session
    happened to resume at, and labelled the result block 0.

    On the cloud path every finishing session is a resumed session, so the
    diagnostic was always attached to the wrong block there.
    """
    blocks = _blocks(2)
    seen = []

    def compress(i, name, problem):
        seen.append((i, name, problem.name))
        return problem.W

    records = Cal.sequential_calibrate(blocks, list(_inputs()), compress,
                                       block_offset=17)

    assert [i for i, _, _ in seen] == [17, 17, 18, 18]
    assert [r["block"] for r in records] == [17, 17, 18, 18]
    # The layer name is the same index, spelled: one record, one block number.
    for r in records:
        assert r["layer"].startswith(f"blocks.{r['block']}.")
    assert [p for _, _, p in seen] == [r["layer"] for r in records]


# --------------------------------------------------------------------------- #
# Calibration windows
# --------------------------------------------------------------------------- #

class _FakeTokenizer:
    """Whitespace tokenizer, enough to exercise the windowing logic."""

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[abs(hash(w)) % 32000 for w in text.split()]])
        return type("Enc", (), {"input_ids": ids})()


def test_wikitext_windows_come_from_the_joined_stream(monkeypatch):
    """The bug this replaces: WikiText rows are single LINES, so requiring one
    row to exceed 2048 tokens found zero windows and the loader raised.  The
    reference implementations join the split first, which is also what
    `eval.perplexity.load_eval_tokens` does for the test split."""
    lines = [f"line {i} " + " ".join(f"w{i}_{j}" for j in range(20))
             for i in range(500)]

    def fake_load_dataset(*args, **kwargs):
        return {"text": lines}

    monkeypatch.setitem(
        __import__("sys").modules, "datasets",
        type("M", (), {"load_dataset": staticmethod(fake_load_dataset)}))

    out = Cal.load_calibration_tokens(_FakeTokenizer(), n_samples=5, seqlen=64,
                                    seed=0, dataset="wikitext2")
    assert out.shape == (5, 64)


def test_calibration_windows_are_reproducible_and_the_seed_is_the_draw(monkeypatch):
    """Spec section 6: the seed IS the calibration draw.  Same seed must give
    the same windows, a different seed different ones -- otherwise the paired
    comparisons Gate B relies on are not paired at all."""
    lines = [f"line {i} " + " ".join(f"w{i}_{j}" for j in range(20))
             for i in range(500)]
    monkeypatch.setitem(
        __import__("sys").modules, "datasets",
        type("M", (), {"load_dataset": staticmethod(lambda *a, **k: {"text": lines})}))

    kw = dict(n_samples=4, seqlen=32, dataset="wikitext2")
    a = Cal.load_calibration_tokens(_FakeTokenizer(), seed=0, **kw)
    b = Cal.load_calibration_tokens(_FakeTokenizer(), seed=0, **kw)
    c = Cal.load_calibration_tokens(_FakeTokenizer(), seed=1, **kw)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)


def test_a_corpus_shorter_than_one_window_says_so(monkeypatch):
    monkeypatch.setitem(
        __import__("sys").modules, "datasets",
        type("M", (), {"load_dataset": staticmethod(lambda *a, **k: {"text": ["a b c"]})}))
    with pytest.raises(RuntimeError, match="need >"):
        Cal.load_calibration_tokens(_FakeTokenizer(), n_samples=1, seqlen=64,
                                  dataset="wikitext2")


def test_an_unknown_calibration_dataset_is_rejected():
    with pytest.raises(ValueError, match="unknown dataset"):
        Cal.load_calibration_tokens(_FakeTokenizer(), dataset="nope")


# --------------------------------------------------------------------------- #
# WHERE THE HESSIAN IS ACCUMULATED
# --------------------------------------------------------------------------- #
# `collect_block_statistics` used to build every accumulator on the CPU and copy
# each activation off the device it was already on, so the largest matmul in a
# full-model run happened on the slower processor.  Measured on one Llama-2-7B
# block over 16,384 tokens: 19.7 s that way against 0.91 s on the GPU in
# float32 -- and this term, which the cost model had never charged at all, is
# 5.6 hours per M1 point as it stood and 0.26 after.

def test_the_accumulator_follows_its_hessian_not_its_input():
    """`update` must move the ACTIVATION to H, never H to the activation.

    The other way round reallocates a k-by-k tensor per batch, and on the sizes
    that matter (11008^2) that is most of a gigabyte each time.
    """
    acc = Cal.HessianAccumulator(4, device="cpu", dtype=torch.float64)
    x = torch.randn((6, 4), dtype=torch.float32)
    acc.update(x)
    assert acc.H.device.type == "cpu" and acc.H.dtype is torch.float64
    assert acc.n_tokens == 6
    assert torch.allclose(acc.H, x.double().T @ x.double(), atol=1e-12)


def test_statistics_land_on_the_blocks_own_device_by_default():
    block = _blocks(1)[0]
    accs = Cal.collect_block_statistics(block, _inputs(), dtype=DT)
    want = next(block.parameters()).device
    assert accs and all(a.H.device == want for a in accs.values())


def test_an_explicit_device_still_wins():
    """The default is a convenience, not a policy: seven Hessians of a 7B block
    are 1.73 GiB at float64, so a caller short of VRAM has to be able to put
    them somewhere else."""
    block = _blocks(1)[0]
    accs = Cal.collect_block_statistics(block, _inputs(), dtype=DT, device="cpu")
    assert accs and all(a.H.device.type == "cpu" for a in accs.values())


def test_a_narrow_compute_dtype_does_not_change_where_the_sum_is_kept():
    """`compute_dtype` narrows the PRODUCT only.  Measured, that is also all it
    can do for accuracy -- float32 products into a float64 H agree with float64
    throughout to 5.08e-06, against 5.06e-06 for a plain float32 accumulator.
    The rounding is in the multiply, and a wider accumulator cannot undo it."""
    x = torch.randn((512, 16), dtype=torch.float32) * 3.0
    wide = Cal.HessianAccumulator(16, dtype=torch.float64)
    mixed = Cal.HessianAccumulator(16, dtype=torch.float64,
                                   compute_dtype=torch.float32)
    narrow = Cal.HessianAccumulator(16, dtype=torch.float32)
    for a in (wide, mixed, narrow):
        a.update(x)
    assert mixed.H.dtype is torch.float64 and narrow.H.dtype is torch.float32
    # The mixed accumulator is no closer to the float64 answer than the narrow
    # one -- that is the measurement, and it is why neither is the default.
    ref = wide.H
    d_mixed = float((mixed.H - ref).abs().max() / ref.abs().max())
    d_narrow = float((narrow.H.double() - ref).abs().max() / ref.abs().max())
    assert d_mixed == pytest.approx(d_narrow, rel=0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_the_default_device_is_the_one_that_actually_distinguishes():
    """The two tests above cannot fail on a CPU-only run, and that is the point.

    `next(block.parameters()).device` IS "cpu" there, so the old hard-wired
    "cpu" and the new default agree and the assertions hold either way -- the
    same shape of blind spot `quantize.is_canonical_codebook` had.  With the
    block on a GPU they diverge, and this is where the 25x lives.
    """
    block = _blocks(1)[0].to("cuda")
    inputs = [x.to("cuda") for x in _inputs()]

    accs = Cal.collect_block_statistics(block, inputs, dtype=DT)
    assert accs and all(a.H.is_cuda for a in accs.values())

    on_cpu = Cal.collect_block_statistics(block, inputs, dtype=DT, device="cpu")
    assert all(a.H.device.type == "cpu" for a in on_cpu.values())

    # Same answer either way: where it is accumulated is a speed decision, and
    # at float64 it is not even a precision one.
    for name, a in accs.items():
        assert torch.equal(a.H.cpu(), on_cpu[name].H)
