"""The three precision levers, and the scaffolding that measures them together.

The experiment's own numbers come from a real layer and are not reproducible in
a test.  What is testable is the part that would silently corrupt them: whether
the arms are the arms they claim to be, whether the TF32 toggle puts the global
back, and whether the baseline row really is a baseline.

That middle one earns its test.  TF32 is process-global state, so a leaked flag
does not fail anything -- it just makes every later measurement in the session
quietly about a different precision, including ones in other files.
"""

from __future__ import annotations

import pytest
import torch

import m0_precision_levers as P
import m1_gates as M
import tiling as Tl


@pytest.fixture(scope="module")
def problem():
    return M.synthetic_problem(n_out=32, n_in=128, n_samples=256, seed=0)


def test_all_eight_combinations_are_present_and_ordered():
    arms = P.all_arms()
    assert len(arms) == 8
    assert [a.name for a in arms][0] == "-"
    # Singles before pairs before the triple: the table is meant to be read in
    # that order, each row against the ones above it.
    counts = [sum((a.fp16, a.kron, a.tf32)) for a in arms]
    assert counts == sorted(counts)
    assert {a.name for a in arms} == {
        "-", "fp16", "kron", "tf32", "fp16+kron", "fp16+tf32", "kron+tf32",
        "fp16+kron+tf32",
    }


def test_an_arm_asks_for_exactly_the_levers_it_names():
    assert P.Arm().kwargs == {"search_dtype": None, "rotate_kron": False}
    assert P.Arm(fp16=True).kwargs["search_dtype"] is torch.float16
    assert P.Arm(kron=True).kwargs["rotate_kron"] is True
    # tf32 is global state, not a run_config argument, so it must NOT leak into
    # the kwargs -- passing it there would be a silent TypeError-free no-op.
    assert "tf32" not in P.Arm(tf32=True).kwargs


def test_the_tf32_toggle_puts_the_global_back():
    before = (torch.backends.cuda.matmul.allow_tf32,
              torch.get_float32_matmul_precision())
    for on in (True, False):
        with P.tf32(on):
            assert torch.backends.cuda.matmul.allow_tf32 is on
        assert (torch.backends.cuda.matmul.allow_tf32,
                torch.get_float32_matmul_precision()) == before


def test_the_tf32_toggle_puts_the_global_back_after_an_exception():
    """The case that matters: a run that raises must not leave the process in a
    different precision than it found it.  Without this the next measurement --
    possibly in another test file -- is about something else and says so
    nowhere."""
    before = (torch.backends.cuda.matmul.allow_tf32,
              torch.get_float32_matmul_precision())
    with pytest.raises(RuntimeError):
        with P.tf32(True):
            raise RuntimeError("boom")
    assert (torch.backends.cuda.matmul.allow_tf32,
            torch.get_float32_matmul_precision()) == before


def test_quality_reports_every_arm_against_the_no_lever_baseline(problem):
    rows = P.quality(problem, tiles=(4, Tl.MAX_TILE), progress=lambda s: None)
    tiles = {r["tile_size"] for r in rows}
    assert rows and len(rows) == 8 * len(tiles)
    for t in tiles:
        base = [r for r in rows if r["tile_size"] == t and r["arm"] == "-"]
        assert len(base) == 1 and base[0]["vs_none"] == 0.0
        # Every arm is measured on the same layer at the same budget, so a
        # missing cell means a skipped config leaked through rather than a
        # lever failing.
        assert len({r["arm"] for r in rows if r["tile_size"] == t}) == 8


def test_speed_is_relative_to_the_same_baseline(problem):
    rows = P.speed(problem, tiles=(4,), passes=2, progress=lambda s: None)
    assert len(rows) == 8
    base = [r for r in rows if r["arm"] == "-"]
    assert len(base) == 1 and base[0]["speedup"] == pytest.approx(1.0)
    assert all(r["seconds"] > 0 for r in rows)


def test_the_gate_b_threshold_is_the_one_the_experiment_downstream_can_see():
    """0.032 is not a tolerance anyone picked.  It is the smallest difference
    Gate B resolves (docs/STATUS.md 5.6, 5.8), so it is the only number that
    makes "is this lever safe" answerable rather than a matter of taste."""
    assert P.GATE_B_RESOLUTION == 0.032


def test_disjoint_savings_beat_the_product_of_their_speedups():
    """The null the composition table is read against, and it is not the
    product.

    A lever that removes fraction `a` of the time gives `1/(1-a)`.  Two on
    disjoint work remove `a+b` and give `1/(1-a-b)`, which is strictly more
    than `1/((1-a)(1-b))`.  Calling the product the prediction would make every
    independent pair look synergistic and hide the one pair that is genuinely
    overlapping.
    """
    a, b = 1 / (1 - 0.30), 1 / (1 - 0.20)        # 30% and 20% of the runtime
    assert P._disjoint(a, b) == pytest.approx(1 / (1 - 0.50))
    assert P._disjoint(a, b) > a * b
    # Three at once, and the identity element.
    c = 1 / (1 - 0.10)
    assert P._disjoint(a, b, c) == pytest.approx(1 / (1 - 0.60))
    assert P._disjoint(a, 1.0) == pytest.approx(a)


def test_disjoint_refuses_to_promise_more_time_than_exists():
    """Fractions that sum past the whole runtime mean the inputs were never
    disjoint.  Returning a negative or wildly large speedup would print as a
    number and read as a result."""
    huge = 1 / (1 - 0.7)
    assert P._disjoint(huge, huge) == float("inf")
