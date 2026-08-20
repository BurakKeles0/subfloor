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
