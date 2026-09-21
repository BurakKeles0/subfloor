"""M1's driver: compress a whole model at one configuration, then measure it.

WHAT THIS CLOSES.  `docs/STATUS.md` section 3.3 has said "the compressed model's
perplexity has never been measured" since the project started, and section 8.1
has named this file as the reason.  Not a scientific obstacle -- the script did
not exist.  Everything it needs has: `calibrate.sequential_calibrate` walks the
blocks, `m1_gates.run_config` is the pipeline, `eval.streamed.streamed_perplexity`
scores a model too large for the card, and `hf_llama` joins them to a real
checkpoint.  On 2026-08-25 that chain was run end to end for the first time and
five defects fell out of it; this file is what makes running it routine.

ONE POINT is one (budget, tile size, calibration draw): calibrate and compress
all 32 blocks in order, then evaluate.  The order is Spec v6 trap 20 -- each
block's Hessians come from the COMPRESSED model above it, so a block is
compressed and only then re-run to produce the next one's inputs.

CHECKPOINTING IS NOT AN ADD-ON.  A point is hours and a laptop closes.  The unit
is the block, because that is the only moment the state is consistent: the block
is final and the activations are exactly what the next one will see.  Anything
finer would have to checkpoint mid-Hessian; anything coarser loses the point.

    resume/                       one directory per point
      state.json                  the protocol, which block is next, and the
                                  records so far
      inputs.pt                   the NEXT block's inputs -- overwritten
      block-<i>.pt                that block's compressed weights -- appended

The activations are what makes this correct rather than merely fast.  They are
the compressed model's output, and they cannot be recomputed from a fresh
checkpoint without redoing every block above.  A resume that restored only the
weights and re-ran the dense model would calibrate block 17 against activations
no version of the model ever produced -- and would not fail, it would just be
quietly wrong.  `test_a_resumed_run_lands_where_an_uninterrupted_one_does` is
the check that this is not happening.

TWO ARGUMENTS THAT ARE NOT DEFAULTS, and both are load-bearing:
`dtype=torch.float32` on the calibration (float64 on a GPU is 1/64 rate and
worth 36 days of M1) and `return_weight=True` on `run_config` (without it the
pipeline computes the compressed weight and drops it).  Both are passed here.

WHAT IT REPORTS BESIDES PERPLEXITY.  Per layer: the relative output error and
the SNR, which are free -- the pipeline already computes them.  And for block 0,
TWO references, because section 3.2's early-warning rule -- the check on this
project's single largest risk -- needs the reference its own sentence describes,
and the one it had was not it.

`m1_gates.dense_wall` is the whole layer as one tile, no rotation, no LDLQ.
`ratio_to_dense` against it says the full pipeline beats a bare quantizer.  That
is worth knowing and it is not section 3.2's question, which is whether the SAME
pipeline keeps its quality once the submatrix is a compacted set of survivors
(`docs/STATUS.md` section 5.11).  Only a reference that changes SPARSITY and
nothing else answers that, so block 0 also runs `m1_gates.pipeline_dense`: the
identical pipeline at density 1.0, same tile size, same seed, same problem.
`ratio_to_pipeline_dense` is what decides `assumption_broken`.

WHAT THE HONEST REFERENCE COSTS, and why it stays at block 0.  It is a second
full pipeline pass over block 0's seven layers, at the most expensive density
there is.  The one d=1.000 block anyone has measured is section 5.14's: 11.46
min on the laptop, 6.28 h for all 32.  So block 0 costs about 11.5 min, roughly
5% of the ~4 h section 5.14 budgets for the next point, and running it on every
block would add ~6.1 h and more than double that run.  Block 0 only, which is
where the check already was.  The point that IS the reference (realized density
1.0) reuses its own record and pays nothing.

The MULTIPLE over a sparse block is deliberately not stated, because nothing in
the repo measures it.  The only laptop d<1 per-block number is section 6.16's
339 s, and that run predates section 6.17's measured 1.46x Hessian-release fix
and used four calibration windows instead of 128; dividing 11.46 min by it
would be section 14.2's "do not carry a ratio to a place where its source
changed".  On the T4 (section 5.11: 168 s per block at d=0.719) no d=1 block
has ever been run, so a cloud point's added wall clock is unmeasured.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import calibrate as Cal              # noqa: E402
import hf_llama as HF                # noqa: E402
import m1_gates as M                 # noqa: E402
import perplexity as PPL             # noqa: E402
import streamed as ST                # noqa: E402
import tiling as Tl                  # noqa: E402

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"

#: Where a finished point's JSON lands when `--out` is not given.  It has a
#: default because the checkpoint is cleared on success: without one, the only
#: record of a multi-hour run was the terminal it was launched from.
DEFAULT_OUT_DIR = Path("results/m1_points")

#: The preregistration's calibration set: 128 windows at the primary seqlen.
CALIB_SAMPLES = 128
CALIB_SEQLEN = 4096

#: The evaluation window.  Named for the same reason `CALIB_SEQLEN` is: it is
#: read in three places -- this module's signature, this module's argparse, and
#: `cloud/run_point.py`, which must not drift from it -- and a default written
#: as three separate literals is a default nothing can pin.
EVAL_SEQLEN = 4096

#: Both are required for the gates (preregistration section 4).  The five
#: zero-shot tasks are reported separately and do NOT enter the gates, so they
#: are not run per point -- once, at the end, on the chosen configuration.
EVAL_DATASETS = ("wikitext2", "c4")

#: Section 3.2: a compressed layer whose error exceeds this multiple of the
#: dense E8P reference means the survivor-submatrix assumption has failed.  The
#: fallback is rotation + GPTQ-3bit and the band moves to 1.83-2.83.
#:
#: THE NUMBER IS INHERITED, NOT RECALIBRATED, and the denominator under it has
#: changed.  Section 3.2 was written when "the dense E8P reference" meant
#: `dense_wall`, a bare quantizer; `_early_warning` now decides against
#: `pipeline_dense`, whose block-0 errors are 1.4-7.1x SMALLER than
#: `dense_wall`'s (both sets are in the two files under `results/m1_points/`).
#: "Twice the reference" is a different bar per layer than it was, and 2.0 is
#: carried over because nothing measured in this repo calibrates a different
#: one -- not because it was re-derived.
#:
#: AND IT IS NOT A GATE A PREDICTOR.  At B=1.5, T=16 the largest block-0 ratio
#: against `pipeline_dense` is 1.469 (`mlp.up_proj`, from those same two files;
#: section 5.13 rung 4 got 1.463 on its own cache) -- the rule stays silent --
#: while that density costs 3.4x perplexity and fails Gate A (section 5.14).
#: So `assumption_broken: false` means "sparsity did not more than double block
#: 0's layer error", and says nothing about whether the point is any good.
#: Sections 5.13 and 5.14: layer error is a diagnostic, the decision is read
#: from perplexity.
EARLY_WARNING_RATIO = 2.0


@dataclass(frozen=True)
class PointSpec:
    """One grid point, and the checkpoint's directory name (`docs/STATUS.md`
    section 8.1).

    It is not the whole checkpoint key, and deliberately so.  The protocol
    that produced a point -- how many calibration windows, how long, at what
    evaluation length -- belongs to the point as much as its budget does, but
    folding it in here would rewrite every slug, and a slug is a path twice
    over: the checkpoint directory under `results/m1_resume/` and the
    finished point's file under `results/m1_points/`.  Rewriting it would
    orphan whatever checkpoint is on the card and stop the results already
    measured from naming the runs that produced them.  The protocol is
    carried beside the spec instead; see `Checkpoint.__init__`.
    """
    model: str = DEFAULT_MODEL
    budget_bits: float = 1.5
    tile_size: int | str = 16
    draw: int = 0

    def slug(self) -> str:
        name = self.model.rsplit("/", 1)[-1]
        return f"{name}_b{self.budget_bits}_t{self.tile_size}_d{self.draw}"


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #

def _atomic(path: Path, write: Callable[[Path], None]) -> None:
    """Write through a sibling temporary file, then rename onto `path`.

    `os.replace` is atomic on POSIX and on Windows, and the sibling keeps it on
    one volume, which is what the guarantee requires.  A failed write leaves the
    temporary behind and `path` as it was; it is removed on the next attempt.
    """
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class Checkpoint:
    """Block-granular resume for one point.

    Deliberately three files rather than one blob.  The activations are large
    and rewritten every block; the weights are large and written once each; the
    state is tiny and must survive a crash mid-write of either.  Bundling them
    would mean rewriting gigabytes to advance a counter.
    """

    def __init__(self, root: Path, spec: PointSpec,
                 protocol: dict | None = None) -> None:
        self.dir = root / spec.slug()
        self.spec = spec
        # Round-tripped through JSON because that is the form it comes back
        # in: `eval_datasets` is a tuple here and a list once it has been
        # through `state.json`, and comparing the two would refuse every
        # correct resume -- a checkpoint nobody can continue, which on the
        # 20 GB cloud disk `cloud/README.md` budgets for is the worse of the
        # two failures.  Empty when the caller names no protocol, which is
        # what makes this argument optional for the tests that predate it.
        self.protocol = json.loads(json.dumps(protocol or {}))

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def inputs_path(self) -> Path:
        return self.dir / "inputs.pt"

    def block_path(self, i: int) -> Path:
        return self.dir / f"block-{i:03d}.pt"

    def load(self) -> dict | None:
        """The point's state, or None if it has not started."""
        if not self.state_path.exists():
            return None
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        if state.get("spec") != asdict(self.spec):
            raise ValueError(
                f"{self.dir} holds a checkpoint for {state.get('spec')}, not "
                f"{asdict(self.spec)} -- refusing to resume across configurations"
            )
        # Compared only when the stored state names one.  A checkpoint
        # written before this field existed cannot answer the question, and
        # refusing it would strand exactly the hours of compressed blocks
        # this class exists to save.
        stored = state.get("protocol")
        if stored is not None and stored != self.protocol:
            raise ValueError(
                f"{self.dir} holds a checkpoint calibrated with {stored}, "
                f"not {self.protocol} -- refusing to resume across protocols"
            )
        return state

    def save_block(self, index: int, block: torch.nn.Module,
                   inputs: list[torch.Tensor], records: list[dict],
                   diagnostics: list[dict] | None = None,
                   seconds: float = 0.0) -> None:
        """Persist one finished block.  Weights first, state last.

        The order is the crash contract: `state.json` is what says a block is
        done, so it is written only once the weights and activations it refers
        to are on disk.  A crash between them leaves a block file nothing points
        at, which the next run overwrites -- the harmless direction.

        The two files that are REWRITTEN every block go through a temporary
        path and `os.replace` (`_atomic`).  In place they are truncated the
        moment the write opens, and `inputs.pt` is gigabytes: a crash inside
        that window leaves `state.json` still saying block `i` is next while
        the activations it names are half this block's and half the last one's.
        That does not fail on resume -- it calibrates the next block against
        activations no version of the model ever produced, which is exactly the
        silent wrongness this checkpoint exists to prevent (see the module
        docstring).  Renaming makes the failure land on the harmless side: the
        previous complete pair survives untouched.

        `diagnostics` and `seconds` are carried for the same reason `records`
        is: the point's answer is assembled across sessions, and a cloud point
        is several by construction.  Left out, both belonged to whichever
        session happened to finish -- the E8P early warning could be `[]` and
        the wall clock counted only the last leg.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in block.state_dict().items()},
                   self.block_path(index))
        _atomic(self.inputs_path,
                lambda p: torch.save([t.detach().cpu() for t in inputs], p))
        _atomic(self.state_path, lambda p: p.write_text(json.dumps({
            "spec": asdict(self.spec),
            "protocol": self.protocol,
            "next_block": index + 1,
            "records": records,
            "diagnostics": list(diagnostics or []),
            "seconds": seconds,
        }, indent=2), encoding="utf-8"))

    def restore_inputs(self) -> list[torch.Tensor]:
        return list(torch.load(self.inputs_path, weights_only=True))

    def apply_saved_blocks(self, blocks, upto: int) -> None:
        """Put the compressed weights back before evaluating.

        Resuming the COMPRESSION needs only the activations -- they already
        carry what the blocks above did.  Resuming the MEASUREMENT needs the
        weights, and they are the thing a fresh `load_llama` does not have.
        """
        for i in range(upto):
            path = self.block_path(i)
            if not path.exists():
                raise FileNotFoundError(
                    f"block {i} is marked done but {path} is missing; the "
                    "checkpoint cannot be completed"
                )
            blocks[i].load_state_dict(torch.load(path, weights_only=True))

    def clear(self) -> None:
        if self.dir.exists():
            for f in self.dir.iterdir():
                f.unlink()
            self.dir.rmdir()


# --------------------------------------------------------------------------- #
# One point
# --------------------------------------------------------------------------- #

def _early_warning(problem: Cal.LayerProblem, record: dict,
                   spec: PointSpec) -> dict:
    """Section 3.2's rule, on one layer, against two references.

    The claim under test is that E8P keeps its quality on a compacted submatrix
    of survivors, which are the fat tail of the distribution by construction.
    Testing it needs a reference that differs from the run in SPARSITY and in
    nothing else.  `m1_gates.dense_wall` is not that: it is the whole layer as
    one tile with no rotation and no LDLQ, so `ratio_to_dense` says only that
    the full pipeline beats a bare quantizer (`docs/STATUS.md` section 5.11).
    It is kept -- it is nearly free, it is what the two finished points carry,
    and "we beat a bare quantizer" is a real diagnostic -- but it decides
    nothing.

    `ratio_to_pipeline_dense` decides.  Its denominator is the identical
    pipeline at density 1.0, same tile size, same seed, same `LayerProblem`:
    section 5.11 named that reference and section 5.14 ran it at the model
    level.  A point that already IS the reference reuses its own record rather
    than recomputing itself for eleven minutes.

    `pipeline_dense_density` travels with the ratio on purpose.  Section 14.2's
    rule is to count that the two arms really took different paths before
    reading the number between them; a reference that quietly came back at the
    run's own density would report a reassuring 1.0 and mean nothing.
    """
    wall = M.dense_wall(problem)
    dense = (record if record["density_realized"] >= 1.0
             else M.pipeline_dense(problem, spec.tile_size, seed=spec.draw))
    if "rel_output_error" not in dense:
        # Cannot happen while `bits_per_position` and `density_for_budget` are
        # inverses, and it is checked anyway: a skipped reference would leave
        # a zero denominator, and an infinite ratio reads as `assumption_broken`
        # -- the project's largest risk answered by an arithmetic accident.
        raise RuntimeError(
            f"the d=1 reference was skipped ({dense.get('skipped')}); section "
            "3.2's early warning has no reference and will not be guessed"
        )
    err = record["rel_output_error"]
    ratio = err / max(dense["rel_output_error"], 1e-30)
    return {
        "dense_e8p_error": wall["rel_output_error"],
        "dense_e8p_snr_db": wall["snr_db"],
        "ratio_to_dense": err / max(wall["rel_output_error"], 1e-30),
        # `bits_realized`, not `budget_bits`.  They agree on the reference arm,
        # which solves d=1.0 exactly; they do NOT agree on the reuse branch,
        # where `budget_bits` is the RUN's budget and only `bits_realized` is
        # what the dense arm actually cost.  Recording the request instead of
        # the outcome is the mistake `run_config` already refuses to make.
        "pipeline_dense_bits": dense["bits_realized"],
        "pipeline_dense_density": dense["density_realized"],
        "pipeline_dense_error": dense["rel_output_error"],
        # The three fields that say the reference differed in SPARSITY ALONE.
        # Density is not enough on its own: a reference taken at another tile
        # size or another seed would also come back at d=1.0 and would not be
        # the same line (section 14.2 -- count that the arms took different
        # paths, and that they took the SAME path everywhere else).
        "pipeline_dense_tile_size": dense.get("tile_size"),
        "pipeline_dense_seed": dense.get("seed"),
        "pipeline_dense_reused_point": dense is record,
        "ratio_to_pipeline_dense": ratio,
        "early_warning_ratio": EARLY_WARNING_RATIO,
        "assumption_broken_reference": "pipeline_dense",
        "assumption_broken": ratio > EARLY_WARNING_RATIO,
    }


def run_point(
    spec: PointSpec = PointSpec(),
    *,
    device: str = "cuda",
    resume_root: Path | None = None,
    calib_samples: int = CALIB_SAMPLES,
    calib_seqlen: int = CALIB_SEQLEN,
    calib_batch: int = 1,
    eval_datasets=EVAL_DATASETS,
    eval_seqlen: int = EVAL_SEQLEN,
    max_eval_windows: int | None = None,
    stop_after_block: int | None = None,
    progress=print,
) -> dict:
    """Compress the whole model at `spec`, then measure it.

    `stop_after_block` exists for the resume test and for nothing else: it
    aborts mid-run the way a closed laptop would, leaving a checkpoint behind.
    """
    t0 = time.time()
    harness = HF.load_llama(spec.model, dtype=torch.float16)
    blocks = harness.blocks
    # What `spec` does not say.  None of it used to reach the checkpoint or
    # the output file, and both consequences were real: a finished point's
    # only trace of its calibration length was `records[i].n_tokens`, and a
    # point relaunched with a different `--calib-seqlen` resumed without
    # complaint -- the activations come off the disk at the old length while
    # the causal mask and the rotary embeddings are rebuilt at the new one,
    # so the point finishes as one perplexity over two protocols.
    # `cloud/README.md` documents that as a trap the operator must remember.
    #
    # THE EVALUATION FIELDS ARE IN HERE TOO, and that is a choice rather than
    # an oversight.  None of them touches a compressed block on disk, so the
    # argument above does not reach them -- but this same dict is the result
    # file's whole provenance, and a point whose evaluation window changed
    # halfway is a point whose `protocol` field is a guess.  The cost is that
    # the window is free to pick at launch and fixed for the life of the
    # checkpoint: an operator twenty blocks in cannot change it without
    # starting the point over.  That is the strict direction, and it is the
    # one this project has taken everywhere else a number could be quietly
    # wrong instead of loudly absent.
    protocol = {
        "calib_samples": calib_samples,
        "calib_seqlen": calib_seqlen,
        "eval_seqlen": eval_seqlen,
        "eval_datasets": eval_datasets,
        "max_eval_windows": max_eval_windows,
    }
    ckpt = Checkpoint(resume_root, spec, protocol) if resume_root else None

    state = ckpt.load() if ckpt else None
    start_block = state["next_block"] if state else 0
    records: list[dict] = list(state["records"]) if state else []
    # Restored, not restarted.  `.get` rather than `[]` so a checkpoint written
    # before these two were carried still resumes -- it loses them, which is
    # what it had anyway.
    diagnostics: list[dict] = list(state.get("diagnostics", [])) if state else []
    seconds_before: float = float(state.get("seconds", 0.0)) if state else 0.0

    if start_block >= len(blocks):
        progress(f"  all {len(blocks)} blocks already compressed; evaluating")
        inputs, block_kwargs = [], {}
    elif state:
        progress(f"  resuming at block {start_block} of {len(blocks)}")
        inputs = ckpt.restore_inputs()
        # `block_kwargs` is the causal mask and the rotary embeddings for this
        # window shape.  Rebuilt rather than stored: it is a pure function of
        # the token shape, and a stored copy is one more thing that can
        # disagree with the activations it is supposed to accompany.
        _, block_kwargs = HF.capture_block_inputs(
            harness.model, _dummy_ids(inputs, calib_seqlen))
    else:
        progress(f"  tokenizing {calib_samples} x {calib_seqlen} calibration tokens")
        tokens = Cal.load_calibration_tokens(
            harness.tokenizer, n_samples=calib_samples, seqlen=calib_seqlen,
            seed=spec.draw, dataset="c4")
        batches = [tokens[i:i + calib_batch]
                   for i in range(0, tokens.shape[0], calib_batch)]
        inputs, block_kwargs = HF.capture_block_inputs(harness.model, batches)

    if ckpt and start_block > 0:
        ckpt.apply_saved_blocks(blocks, start_block)

    def compress(i: int, name: str, problem: Cal.LayerProblem) -> torch.Tensor:
        r = M.run_config(problem, budget_bits=spec.budget_bits,
                         tile_size=spec.tile_size, seed=spec.draw,
                         return_weight=True)
        if "W_hat" not in r:
            raise RuntimeError(
                f"block {i} {name}: {r.get('skipped')} -- the budget is "
                f"unreachable at tile size {spec.tile_size}"
            )
        if i == 0:
            diagnostics.append({"block": i, "name": name,
                                **_early_warning(problem, r, spec)})
        return r["W_hat"]

    def block_done(i: int, ins, recs) -> None:
        done = seconds_before + (time.time() - t0)
        if ckpt:
            ckpt.save_block(i, blocks[i], ins, recs,
                            diagnostics=diagnostics, seconds=done)
        progress(f"  block {i + 1}/{len(blocks)} done ({done / 60:.1f} min)")
        if stop_after_block is not None and i >= stop_after_block:
            raise _StopRun()

    if start_block < len(blocks):
        try:
            recs = Cal.sequential_calibrate(
                blocks[start_block:], inputs, compress,
                block_kwargs=block_kwargs,
                dtype=torch.float32,          # NOT the default; see the docstring
                device=device,
                progress=None,
                on_block_done=block_done,
                block_offset=start_block,
            )
        except _StopRun:
            progress(f"  stopped after block {stop_after_block} (checkpoint kept)")
            return {"spec": asdict(spec), "stopped_after_block": stop_after_block,
                    "records": records, "diagnostics": diagnostics}
        records = records + [r for r in recs if r["block"] >= start_block]

    ppl = {}
    for dataset in eval_datasets:
        progress(f"  evaluating {dataset}")
        stream = PPL.load_eval_tokens(harness.tokenizer, dataset=dataset)
        r = ST.streamed_perplexity(
            harness.model, stream, seqlen=eval_seqlen, device=device,
            dataset=dataset, model_name=spec.model,
            max_windows=max_eval_windows)
        ppl[dataset] = r.perplexity
        progress(f"    {dataset}: {r.perplexity:.4f} over {r.n_windows} windows")

    # `seconds` is the POINT's cost, summed over every session that built it --
    # a cloud point is several by construction, and this run is also the
    # measurement of what a 16 GiB card does with one (`cloud/README.md`).
    # Reported from `t0` alone it counted only the last leg, which for a
    # resumed point could be the evaluation and nothing else.
    out = {
        "spec": asdict(spec),
        "protocol": protocol,
        "seconds": seconds_before + (time.time() - t0),
        "seconds_this_session": time.time() - t0,
        "perplexity": ppl,
        "records": records,
        "diagnostics": diagnostics,
        "levers": {
            "rotate_kron": M.PIPELINE_ROTATE_KRON,
            "search_dtype": str(M.PIPELINE_SEARCH_DTYPE),
            "compensate_block": M.PIPELINE_COMPENSATE_BLOCK,
        },
    }
    if ckpt:
        ckpt.clear()
    return out


class _StopRun(Exception):
    """Interrupt a run the way a closed laptop would."""


def _dummy_ids(inputs, seqlen):
    """Token batches shaped like the run's, only to rebuild `block_kwargs`.

    The kwargs are the causal mask and the rotary embeddings, which depend on
    the window SHAPE and nothing else -- so any ids of the right shape produce
    the ones the run was using.  The hidden states they come with are discarded;
    the real ones come off the checkpoint.
    """
    batch = inputs[0].shape[0]
    return [torch.zeros((batch, seqlen), dtype=torch.long)]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tile", default="16",
                    help="tile size, or 'max' for the structured end")
    ap.add_argument("--draw", type=int, default=0,
                    help="calibration draw -- the axis Gate B's CIs are over")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume-root", type=Path, default=Path("results/m1_resume"),
                    help="checkpoint directory; --no-resume disables it")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES)
    ap.add_argument("--calib-seqlen", type=int, default=CALIB_SEQLEN)
    ap.add_argument("--eval-seqlen", type=int, default=EVAL_SEQLEN)
    ap.add_argument("--datasets", nargs="*", default=list(EVAL_DATASETS))
    ap.add_argument("--max-eval-windows", type=int, default=None)
    ap.add_argument("--stop-after-block", type=int, default=None,
                    help="abort mid-run, for the resume test")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the finished point's JSON goes; defaults to "
                         f"{DEFAULT_OUT_DIR}/<slug>.json")
    args = ap.parse_args(argv)

    tile = Tl.MAX_TILE if args.tile == "max" else int(args.tile)
    spec = PointSpec(model=args.model, budget_bits=args.budget,
                     tile_size=tile, draw=args.draw)
    print(f"point: {spec.slug()}")

    out = run_point(
        spec, device=args.device,
        resume_root=None if args.no_resume else args.resume_root,
        calib_samples=args.calib_samples, calib_seqlen=args.calib_seqlen,
        eval_datasets=tuple(args.datasets), eval_seqlen=args.eval_seqlen,
        max_eval_windows=args.max_eval_windows,
        stop_after_block=args.stop_after_block,
    )
    # Written unconditionally, and this is not a convenience.  A point is hours
    # and `run_point` clears the checkpoint once it has evaluated, so with no
    # `--out` the perplexity, the per-layer records and block 0's E8P
    # early-warning diagnostic -- the check on this project's largest single
    # risk -- survived only as stdout.  An interrupted run writes nothing: it
    # has no result yet, and clobbering a finished point's JSON with a partial
    # one is the failure this avoids.
    if "stopped_after_block" not in out:
        dest = args.out or (DEFAULT_OUT_DIR / f"{spec.slug()}.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, default=str),
                        encoding="utf-8")
        print(f"written: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
