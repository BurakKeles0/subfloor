"""The audit that decides whether a lever's recorded factor is real.

Every number this experiment produces rests on one claim: that its three arms
differ where the lever is and NOWHERE ELSE.  If that fails, a timing difference
is a difference between two problems rather than between two arithmetics -- and
it would still look like a lever, which is the whole difficulty.

So these tests are about the ISOLATION, not about any timing.  A test needing a
quiet card could only run when the thing it protects is already unnecessary.

`docs/STATUS.md` section 14.2: watch the path, not the answer.  Here the answer
is a wall clock, and a wall clock will happily report a difference between two
runs that never took different code.
"""

from __future__ import annotations

import pytest
import torch

import calibrate as Cal
import m0_lever_audit as LA
import m1_gates as M


@pytest.fixture(scope="module")
def problem():
    """Small, but wide enough that the mask is not degenerate."""
    return Cal.synthetic_problem(64, 128, 256)


# --------------------------------------------------------------------------- #
# What actually ran
# --------------------------------------------------------------------------- #

def test_the_trace_counts_the_levers_where_they_act(problem):
    """The pipeline arm builds Kronecker factors, rotates with them, blocks.

    Mutation check built in: every assertion here is on a NON-ZERO count.  A
    `run_traced` that failed to install its spies -- the obvious way for this
    file to become decorative -- returns zeros and fails on the first line.
    """
    record, trace = LA.run_traced(problem, budget_bits=1.5, tile_size=4)

    assert trace.kron_built == 1, "the Kronecker factors were never built"
    assert trace.rotations_kron > 0, "no sub-Hessian took the Kronecker path"
    assert trace.rotations_dense == 0, "something rotated densely anyway"
    assert trace.compensate_blocks == (M.PIPELINE_COMPENSATE_BLOCK,)

    # And the record agrees, which is the weaker of the two claims: a record can
    # say `compensate_block=512` while the argument goes nowhere, which is how
    # that lever stayed unreachable from the driver for a day.
    assert record["rotate_kron"] is M.PIPELINE_ROTATE_KRON
    assert record["compensate_block"] == M.PIPELINE_COMPENSATE_BLOCK


def test_each_arm_moves_its_own_lever_and_only_its_own(problem):
    """The property the whole experiment rests on, measured rather than assumed.

    `no_kron` must rotate the SAME number of sub-Hessians as `pipeline`, only
    densely.  If it rotated a different number the two arms would be compressing
    different problems and the timing delta would not be the lever -- and
    nothing in the wall clock would say so.
    """
    traces = {name: LA.run_traced(problem, budget_bits=1.5, tile_size=4, **kw)[1]
              for name, kw in LA.ARMS.items()}
    LA.check_paths(traces)                      # the experiment's own gate

    pipe, nk, nb = traces["pipeline"], traces["no_kron"], traces["no_block"]
    assert nk.rotations_dense == pipe.rotations_kron > 0
    assert nk.kron_built == 0
    assert nk.compensate_blocks == pipe.compensate_blocks
    assert nb.compensate_blocks == (None,)
    assert nb.rotations_kron == pipe.rotations_kron


def test_the_spies_are_removed_even_when_the_run_dies(problem):
    """A leaked spy would silently double-count every later arm."""
    import prune as P
    import rotation as R

    before = (R.kronecker_factors, R.rotate_hessian, P.forward_compensate)
    with pytest.raises(TypeError):
        LA.run_traced(problem, budget_bits=1.5, tile_size=4, not_a_kwarg=1)
    assert (R.kronecker_factors, R.rotate_hessian, P.forward_compensate) == before


# --------------------------------------------------------------------------- #
# What the gate refuses
# --------------------------------------------------------------------------- #

def _traces(**over):
    base = {
        "pipeline": LA.Trace(1, 16, 0, (M.PIPELINE_COMPENSATE_BLOCK,)),
        "no_kron": LA.Trace(0, 0, 16, (M.PIPELINE_COMPENSATE_BLOCK,)),
        "no_block": LA.Trace(1, 16, 0, (None,)),
    }
    base.update(over)
    return base


def test_a_correct_arm_set_passes():
    LA.check_paths(_traces())


@pytest.mark.parametrize("name,trace,because", [
    ("pipeline", LA.Trace(0, 0, 16, (512,)), "pipeline never took the lever"),
    ("pipeline", LA.Trace(1, 16, 0, (None,)), "pipeline never blocked"),
    ("no_kron", LA.Trace(1, 16, 0, (512,)), "the lever did not move"),
    ("no_kron", LA.Trace(0, 0, 12, (512,)), "a different number of tiles"),
    ("no_kron", LA.Trace(0, 0, 16, (None,)), "two levers moved at once"),
    ("no_block", LA.Trace(1, 16, 0, (512,)), "the lever did not move"),
    ("no_block", LA.Trace(0, 0, 16, (None,)), "two levers moved at once"),
])
def test_the_gate_refuses_arms_that_do_not_isolate(name, trace, because):
    """Each of these would produce a perfectly readable, meaningless number.

    That is the point.  None of them raises anywhere else in the pipeline: the
    run completes, the clock reports a difference, and the difference is not the
    lever.  Timing is the one measurement with no natural error signal.
    """
    with pytest.raises(RuntimeError, match="do not isolate"):
        LA.check_paths(_traces(**{name: trace}))


# --------------------------------------------------------------------------- #
# The machine
# --------------------------------------------------------------------------- #

def test_a_process_that_arrives_after_the_baseline_stops_the_run(monkeypatch):
    """The signal that survives our own load, per section 6.17.

    Not the clock: during a measurement the clock is high because WE are working
    the card, so a clock-keyed mid-run check fires on itself.  It did, twice --
    once inside `bench_guard` and once in the first draft of this experiment,
    which was refused at 88% of maximum by its own preceding arm.
    """
    monkeypatch.setattr(LA, "foreign_compute_pids",
                        lambda: [(111, "theirs.exe"), (222, "mine.exe")])

    # Both known at the start: tolerated, whatever they are doing.
    assert LA.no_newcomers({111: "theirs.exe", 222: "mine.exe"}) == []

    with pytest.raises(RuntimeError, match="refusing to time"):
        LA.no_newcomers({111: "theirs.exe"})

    assert LA.no_newcomers({111: "theirs.exe"}, strict=False) == [(222, "mine.exe")]


def test_host_memory_is_read_rather_than_guessed():
    """The card is not the only thing that can be busy.

    Stage one materializes a 13.5 GB checkpoint, and short host memory does not
    raise -- it pages, which reads as a slow machine rather than a wrong one.
    """
    avail, total = LA.host_memory_gib()
    assert total > 0.0, "host memory could not be read on this platform"
    assert 0.0 < avail <= total

    LA.require_host_memory(0.001)               # plenty
    with pytest.raises(RuntimeError, match="host RAM available"):
        LA.require_host_memory(total * 10)


def test_the_three_default_layers_are_the_three_distinct_shapes():
    """Timing all seven would repeat two shapes five times between them.

    Pinned because the saving is what makes the audit affordable, and because
    `m0_cost_model.COMPENSATE_TIMINGS` is keyed on exactly these three.
    """
    import m0_cost_model as CM

    shapes = {(n_out, n_in) for n_out, n_in, _ in CM.LLAMA2_7B}
    assert len(LA.DEFAULT_LAYERS) == len(shapes) == 3
    assert set(LA.DEFAULT_LAYERS) <= set(LA.ALL_LAYERS)
    assert {(n_out, n_in) for n_out, n_in, _e, _b
            in CM.COMPENSATE_TIMINGS["cuda_f32"]} == shapes
