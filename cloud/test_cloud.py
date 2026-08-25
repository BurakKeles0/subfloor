"""The budget has to fire, and it has to fire in the right place.

A wall-clock budget that silently stops working is worse than no budget: the
session gets killed mid-block instead of at a boundary, and on a free tier that
is the difference between losing four minutes and losing four hours.  So what is
tested is the DECISION -- given a checkpoint state and a clock, does it stop --
rather than any particular run.

Run with `python -m pytest cloud/ -q`.  Not under `tests/`, because nothing here
is part of the experiment: `cloud/` is additive and the pipeline does not import
it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_point as RP                              # noqa: E402
import m1_run as R                                  # noqa: E402


@pytest.fixture
def spec():
    return R.PointSpec(budget_bits=1.5, tile_size=16, draw=0)


def _write_state(root: Path, spec: R.PointSpec, next_block: int) -> None:
    d = root / spec.slug()
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps(
        {"spec": asdict(spec), "next_block": next_block, "records": []}),
        encoding="utf-8")


# --------------------------------------------------------------------------- #
# Reading the state
# --------------------------------------------------------------------------- #

def test_a_point_that_has_not_started_reads_as_minus_one(tmp_path, spec):
    assert RP._state_block(tmp_path, spec) == -1


def test_the_block_count_comes_from_the_checkpoint(tmp_path, spec):
    _write_state(tmp_path, spec, 7)
    assert RP._state_block(tmp_path, spec) == 7


def test_a_half_written_state_reads_as_not_started(tmp_path, spec):
    """`save_block` writes `state.json` last precisely so a crash leaves either
    a complete state or none.  A truncated one still has to be survivable, and
    the safe reading is "not started" -- which re-does one block rather than
    skipping one."""
    d = tmp_path / spec.slug()
    d.mkdir(parents=True)
    (d / "state.json").write_text('{"next_block": 3, "spe', encoding="utf-8")
    assert RP._state_block(tmp_path, spec) == -1


# --------------------------------------------------------------------------- #
# When the budget fires
# --------------------------------------------------------------------------- #

def _progress(tmp_path, spec, *, now, deadline, seen=-1):
    counter = {"budget": False, "blocks": 0, "seen": seen}
    clock = {"t": now}
    fn = RP.make_progress(tmp_path, spec, deadline=deadline, started=0.0,
                          seen=seen, counter=counter,
                          log=lambda m: None, clock=lambda: clock["t"])
    return fn, counter, clock


def test_it_does_not_fire_while_the_budget_holds(tmp_path, spec):
    fn, counter, _ = _progress(tmp_path, spec, now=100.0, deadline=1000.0)
    _write_state(tmp_path, spec, 1)
    fn("block 1/32 done")
    assert counter["blocks"] == 1 and not counter["budget"]


def test_it_fires_at_a_block_boundary_once_the_budget_is_gone(tmp_path, spec):
    fn, counter, _ = _progress(tmp_path, spec, now=2000.0, deadline=1000.0)
    _write_state(tmp_path, spec, 1)
    with pytest.raises(RP.BudgetSpent):
        fn("block 1/32 done")
    assert counter["budget"] and counter["seen"] == 1


def test_an_expired_budget_alone_is_not_enough(tmp_path, spec):
    """THE POINT OF THE WHOLE MECHANISM.  Past the deadline but mid-block, there
    is no complete checkpoint to stop at, so stopping would throw the block
    away.  It waits for the state to advance."""
    fn, counter, _ = _progress(tmp_path, spec, now=2000.0, deadline=1000.0)
    fn("  evaluating wikitext2")            # no state file at all
    _write_state(tmp_path, spec, 4)
    _ = RP._state_block(tmp_path, spec)
    fn2, counter2, _ = _progress(tmp_path, spec, now=2000.0, deadline=1000.0,
                                 seen=4)
    fn2("  some message, block 4 already counted")   # state did not advance
    assert not counter["budget"] and not counter2["budget"]


def test_it_watches_the_state_and_not_the_message(tmp_path, spec):
    """Mutation-proofing by construction: a message that says a block finished,
    with a checkpoint that says otherwise, must not trigger anything.

    If this ever starts passing on the strength of the string, the budget has
    been rewired to another module's log wording and will break the next time
    that wording changes."""
    fn, counter, _ = _progress(tmp_path, spec, now=2000.0, deadline=1000.0)
    fn("  block 17/32 done (99.9 min)")     # convincing, and a lie
    assert counter["blocks"] == 0 and not counter["budget"]


def test_a_resumed_session_does_not_count_the_blocks_it_inherited(tmp_path, spec):
    """Resuming at block 10 must not read that as ten blocks of progress and
    stop immediately -- the session has done no work yet."""
    _write_state(tmp_path, spec, 10)
    fn, counter, _ = _progress(tmp_path, spec, now=2000.0, deadline=1000.0,
                               seen=RP._state_block(tmp_path, spec))
    fn("  resuming at block 10 of 32")
    assert counter["blocks"] == 0 and not counter["budget"]
    _write_state(tmp_path, spec, 11)
    with pytest.raises(RP.BudgetSpent):
        fn("block 11/32 done")
    assert counter["blocks"] == 1


# --------------------------------------------------------------------------- #
# The preflight's own claims
# --------------------------------------------------------------------------- #

def test_the_preflight_names_every_threshold_it_did_not_measure(tmp_path):
    """Section 6.13 cost ten of twenty-one grid cells to a threshold that was
    right for one card.  The preflight cannot re-measure them, so the least it
    owes is to print them -- and this checks they all still exist under the
    names it prints, which is what would break silently in a refactor."""
    import quantize as Qz
    for name in ("_LATTICE_MIN_ROWS", "_ANALYTIC_MIN_ROWS",
                 "_ANALYTIC_DIRECT_MIN_ROWS", "CHUNK_TARGET_ROWS",
                 "DECODER_MISS_FRACTION"):
        assert hasattr(Qz, name), f"preflight prints {name}, which is gone"


def test_the_checkpoint_estimate_matches_what_a_point_really_writes():
    """13 GiB is not a guess: 32 blocks of a Llama-2-7B block's parameters at
    float16.  If the estimate drifts from the arithmetic, the storage check
    starts passing on machines where the run will die at block 30."""
    import preflight as PF

    params = 4 * 4096 * 4096 + 3 * 11008 * 4096          # attn + mlp
    per_block_gib = params * 2 / 2 ** 30
    assert PF.POINT_CHECKPOINT_GIB >= 32 * per_block_gib
    assert PF.POINT_CHECKPOINT_GIB <= 32 * per_block_gib + 2.0
