"""The checkpoint's crash contract, at the level of the files themselves.

`tests/test_hf_llama.py` owns the end-to-end claim -- that a resumed run lands
on the uninterrupted one's perplexity.  This file owns the narrower one it rests
on: that an interruption *inside a write* cannot leave a checkpoint which
resumes into the wrong activations.

The distinction matters because the two failures do not look alike.  A run that
resumes from a half-written `inputs.pt` does not crash; it calibrates the next
block against activations no version of the model ever produced, and returns a
perplexity.  So the test cannot watch the answer -- there is an answer either
way.  It watches the path: what is on disk after a write dies partway.

No `transformers` here on purpose.  `Checkpoint` needs torch and a state dict
and nothing else, and a test gated behind an optional dependency is a test that
does not run on the machine most likely to be interrupted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import m1_run as R


SPEC = R.PointSpec(model="tiny", budget_bits=1.5, tile_size=4, draw=0)

#: The protocol beside it.  `eval_datasets` is a tuple because that is what
#: `run_point` passes, and `state.json` hands it back a list.
PROTOCOL = {"calib_samples": 8, "calib_seqlen": 2048, "eval_seqlen": 4096,
            "eval_datasets": ("wikitext2",)}


def _block(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return torch.nn.Linear(4, 4)


def _inputs(fill: float) -> list[torch.Tensor]:
    return [torch.full((2, 3), fill), torch.full((2, 3), fill + 1)]


def _explode_on(monkeypatch, stem: str):
    """Make `torch.save` truncate its target and then die, as a killed process
    does -- but only for the file whose name starts with `stem`.

    Truncating first is the whole point.  `torch.save` opens the path for
    writing, so the old contents are gone before any of the new ones land; a
    mock that merely raised would test a failure mode that cannot happen.
    """
    real = torch.save

    def exploding(obj, path, *args, **kwargs):
        p = Path(path)
        if p.name.startswith(stem):
            p.write_bytes(b"\x00" * 16)
            raise OSError("no space left on device")
        return real(obj, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", exploding)


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #

def test_atomic_leaves_the_target_untouched_when_the_write_dies(tmp_path):
    target = tmp_path / "thing.json"
    target.write_text("the previous complete value", encoding="utf-8")

    def write_then_die(p: Path) -> None:
        p.write_text("half of the ne", encoding="utf-8")
        raise OSError("no space left on device")

    with pytest.raises(OSError):
        R._atomic(target, write_then_die)

    assert target.read_text(encoding="utf-8") == "the previous complete value"


def test_atomic_does_not_leave_a_temporary_behind_on_success(tmp_path):
    target = tmp_path / "thing.json"
    R._atomic(target, lambda p: p.write_text("value", encoding="utf-8"))

    assert target.read_text(encoding="utf-8") == "value"
    assert list(tmp_path.iterdir()) == [target]


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

def test_a_crash_writing_the_activations_leaves_the_last_pair_consistent(
        tmp_path, monkeypatch):
    """`inputs.pt` is gigabytes and is rewritten every block, so it is the file
    a run is most likely to be killed inside.

    Written in place, that leaves `state.json` still naming block 1 as next
    while the activations beside it are half block 0's and half block 1's --
    and resuming from that pair is the silent wrongness the checkpoint exists
    to prevent.  Written through a rename, the previous complete pair survives
    and the run resumes one block earlier, which costs time and nothing else.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC)
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}])

    _explode_on(monkeypatch, "inputs")
    with pytest.raises(OSError):
        ckpt.save_block(1, _block(1), _inputs(99.0), [{"block": 0},
                                                      {"block": 1}])

    state = json.loads(ckpt.state_path.read_text(encoding="utf-8"))
    assert state["next_block"] == 1
    restored = ckpt.restore_inputs()
    assert all(torch.equal(a, b) for a, b in zip(restored, _inputs(1.0)))


def test_a_crash_writing_the_state_leaves_the_previous_state_readable(
        tmp_path, monkeypatch):
    """A truncated `state.json` is not a wrong answer, it is no answer: the
    next run's `json.loads` raises and the point cannot be resumed at all.
    Hours of compressed blocks are still on disk with nothing pointing at them.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC)
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}])

    real_write_text = Path.write_text

    def exploding(self, data, *args, **kwargs):
        if self.name.startswith("state"):
            real_write_text(self, '{"spec": {"mod', *args, **kwargs)
            raise OSError("no space left on device")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", exploding)
    with pytest.raises(OSError):
        ckpt.save_block(1, _block(1), _inputs(99.0), [{"block": 0},
                                                      {"block": 1}])
    monkeypatch.undo()

    assert ckpt.load()["next_block"] == 1


def test_the_state_carries_the_diagnostics_and_the_elapsed_time(tmp_path):
    """A point is assembled across sessions, and on the cloud it always is.

    `records` was restored on resume and these two were not, so both belonged
    to whichever session happened to finish last: the section 3.2 early warning
    could come back `[]` from a session that only ran the evaluation, and the
    wall clock -- which this run is also supposed to MEASURE, since nobody has
    priced a point on a 16 GiB card -- counted the last leg alone.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC)
    diags = [{"block": 0, "name": "q_proj", "ratio_to_dense": 1.2}]
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}],
                    diagnostics=diags, seconds=1234.5)

    state = ckpt.load()
    assert state["diagnostics"] == diags
    assert state["seconds"] == pytest.approx(1234.5)


def test_a_checkpoint_written_before_these_fields_existed_still_resumes(
        tmp_path):
    """Whatever is on the card when this lands has to keep working; a resume
    that raises KeyError would strand hours of compressed blocks.

    `protocol` is the third such field, and the one that could refuse rather
    than raise: the state below predates it, so there is nothing to compare
    and the resume has to go ahead anyway.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC, PROTOCOL)
    ckpt.dir.mkdir(parents=True, exist_ok=True)
    ckpt.state_path.write_text(json.dumps({
        "spec": {"model": "tiny", "budget_bits": 1.5, "tile_size": 4,
                 "draw": 0},
        "next_block": 3,
        "records": [{"block": 0}],
    }, indent=2), encoding="utf-8")

    state = ckpt.load()
    assert state["next_block"] == 3
    assert state.get("diagnostics", []) == []
    assert float(state.get("seconds", 0.0)) == 0.0
    assert "protocol" not in state


def test_resuming_a_different_configuration_is_refused(tmp_path):
    """`Checkpoint.load` is the only thing standing between one point's
    blocks and another point's perplexity, and nothing in this file checked
    that it stands.  `tests/test_hf_llama.py` has the end-to-end version,
    but it is behind `importorskip("transformers")` -- so on a machine
    without it the guard went untested, and this file exists precisely
    because that is the machine most likely to be interrupted.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC)
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}])

    other = R.Checkpoint(tmp_path, R.PointSpec(
        model="tiny", budget_bits=2.0, tile_size=4, draw=0))
    other.dir = ckpt.dir           # what a hand-moved directory looks like

    with pytest.raises(ValueError, match="across configurations"):
        other.load()


def test_resuming_with_a_different_protocol_is_refused(tmp_path):
    """The expensive one, because it needs no hand-moved directory.

    The directory is named after the spec alone, so a half-finished point
    relaunched with a different `--calib-seqlen` lands in its own checkpoint
    and used to continue: the activations come back at the old length while
    the causal mask and the rotary embeddings are rebuilt at the new one,
    and the point finishes as one perplexity over two protocols with nothing
    in the result file to say so.  `cloud/README.md` has carried this as a
    trap the operator must remember; remembering is not a mechanism.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC, PROTOCOL)
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}])

    longer = R.Checkpoint(tmp_path, SPEC, {**PROTOCOL, "calib_seqlen": 4096})
    # The collision is the point: same spec, same slug, same directory.  If
    # the protocol were in the slug this would be a different path and the
    # checkpoint on the card would have lost its.
    assert longer.dir == ckpt.dir

    with pytest.raises(ValueError, match="across protocols"):
        longer.load()


def test_an_unchanged_protocol_survives_the_round_trip_through_the_state(
        tmp_path):
    """The guard's own failure mode, and the reason it is worth a test of
    its own: `eval_datasets` is a tuple in memory and a list once it has
    been through `state.json`, so a comparison made against the in-memory
    form refuses every correct resume.  That trades a silently wrong answer
    for a checkpoint nobody can continue -- on a cloud disk that holds one
    point and nothing else, the worse trade.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC, PROTOCOL)
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}])

    state = R.Checkpoint(tmp_path, SPEC, PROTOCOL).load()
    assert state["next_block"] == 1
    # And it really did come back from disk in the other shape.
    assert state["protocol"]["eval_datasets"] == ["wikitext2"]


def test_a_finished_block_leaves_no_temporary_files(tmp_path):
    """`clear()` unlinks whatever it finds, so a stray `.tmp` would be removed
    rather than kept -- but it would also be gigabytes sitting beside the file
    it duplicates, on the card-sized volumes `cloud/README.md` budgets for.
    """
    ckpt = R.Checkpoint(tmp_path, SPEC)
    ckpt.save_block(0, _block(0), _inputs(1.0), [{"block": 0}])

    assert not [p for p in ckpt.dir.iterdir() if p.suffix == ".tmp"]
