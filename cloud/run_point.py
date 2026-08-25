"""One grid point on a machine that can be taken away from you.

A wrapper around `experiments/m1_run.py`, adding the two things a free-tier
session needs and nothing else.  IT DOES NOT TOUCH THE PIPELINE -- every file
under this directory is additive, and the run it produces is the same run
`m1_run.py` produces locally.  That is the point: a cloud fork that drifts is a
second implementation of the experiment.

WHAT IT ADDS

  1. A WALL-CLOCK BUDGET that stops at a block boundary.  Kaggle cuts a session
     at 12 hours and Colab's free tier cuts one whenever it likes; a run killed
     mid-block loses that block's work, and at T=1 a block is not cheap.
  2. A resume that is the DEFAULT rather than a flag, because on this kind of
     machine the second run is the normal case, not the exception.

HOW THE BUDGET KNOWS A BLOCK IS DONE, and why not the obvious way.  `run_point`
takes a `progress` callable and prints "block i/32 done" through it, so the
obvious hook is to match that string.  It is the wrong hook: the message is a
human-facing detail of another module and a wording change would silently turn
the budget off.  What this watches instead is `state.json`'s `next_block` --
the thing `Checkpoint.save_block` writes LAST, after the weights and the
activations are on disk.  So the budget fires exactly when there is a complete
checkpoint to stop at, and it is reading the state rather than a sentence about
the state.

EXIT CODES, because a notebook cell needs to tell three outcomes apart:

    0   the point finished, perplexity included
   42   the budget ran out; the checkpoint is complete, run again to continue
    1   something broke

STORAGE IS THE CONSTRAINT NOBODY EXPECTS.  A point's checkpoint grows to about
12 GiB -- 32 blocks of compressed weights at 0.38 GiB each -- and every one of
them is needed at the end, because the evaluation runs on the assembled
compressed model.  Add the activations (`inputs.pt`, 4.0 GiB at the
preregistration's 128 x 4096, 128 MiB at 4 x 2048) and the 13 GB checkpoint
cache.  `--resume-root` must point somewhere that holds all of it AND survives
the session; see `cloud/README.md` for what that means on each platform.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import torch                                    # noqa: E402
import m1_run as R                              # noqa: E402


class BudgetSpent(Exception):
    """The wall clock ran out at a block boundary.  Not an error."""


def _state_block(resume_root: Path, spec: R.PointSpec) -> int:
    """How many blocks the checkpoint says are done.  -1 if it has not started.

    Read from disk every time rather than cached: the file is a few kilobytes
    and it is the only thing that actually knows.
    """
    path = resume_root / spec.slug() / "state.json"
    if not path.exists():
        return -1
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["next_block"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return -1


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 2 ** 30


def make_progress(resume_root: Path, spec: R.PointSpec, *, deadline: float,
                  started: float, seen: int, counter: dict,
                  log=print, clock=time.time):
    """The hook `m1_run.run_point` calls, with the budget inside it.

    Split out so it can be tested without a GPU and a 13 GiB checkpoint.

    WHAT IT WATCHES IS THE CHECKPOINT STATE, NOT THE MESSAGE.  `save_block`
    writes `state.json` last, after the weights and the activations are on
    disk, so an advance of `next_block` means there is a COMPLETE checkpoint to
    stop at.  Matching the "block i/32 done" string instead would tie this
    budget to another module's log wording, and a reworded line would switch it
    off in silence -- the failure mode `docs/STATUS.md` section 14.2 keeps
    recording under "watch the path, not the answer".

    Two conditions, and both are necessary: the state advanced AND the clock is
    past the deadline.  Past the deadline mid-block there is nothing complete to
    stop at, so stopping would throw that block away.
    """
    box = {"seen": seen}

    def progress(msg: str) -> None:
        elapsed = (clock() - started) / 60.0
        log(f"[{elapsed:6.1f} min] {msg}")
        now = _state_block(resume_root, spec)
        if now > box["seen"]:
            box["seen"] = now
            counter["blocks"] += 1
            counter["seen"] = now
            left = (deadline - clock()) / 60.0
            log(f"           checkpoint at block {now}/32, "
                f"{left:.0f} min of budget left")
            if clock() >= deadline:
                counter["budget"] = True
                raise BudgetSpent()

    return progress


def run(spec: R.PointSpec, *, resume_root: Path, hours: float,
        calib_samples: int, calib_seqlen: int, eval_datasets,
        max_eval_windows: int | None, device: str = "cuda",
        min_free_gib: float = 18.0) -> tuple[int, dict | None]:
    deadline = time.time() + hours * 3600.0
    started = time.time()
    seen = _state_block(resume_root, spec)
    fired = {"budget": False, "blocks": 0, "seen": seen}

    free = free_gib(resume_root)
    print(f"checkpoint root {resume_root}  ({free:.1f} GiB free)")
    if free < min_free_gib:
        print(f"  [WARNING] a point wants about {min_free_gib:.0f} GiB here "
              f"(32 blocks x 0.38 GiB, plus the activations).  It will run and "
              f"then fail at the block that does not fit, which is the "
              f"expensive way to find out.")
    if seen >= 0:
        print(f"  resuming: {seen} of 32 blocks already done")

    progress = make_progress(
        resume_root, spec, deadline=deadline, started=started, seen=seen,
        counter=fired, log=lambda m: print(m, flush=True))

    try:
        out = R.run_point(
            spec, device=device, resume_root=resume_root,
            calib_samples=calib_samples, calib_seqlen=calib_seqlen,
            eval_datasets=eval_datasets, max_eval_windows=max_eval_windows,
            progress=progress)
    except BudgetSpent:
        print(f"\nbudget spent after {fired['blocks']} block(s) this session; "
              f"{fired['seen']}/32 done in total.  The checkpoint is complete "
              f"-- run the same command again to continue.")
        return 42, None
    return 0, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=float, default=1.5,
                    help="bits per weight (the grid's B)")
    ap.add_argument("--tile", default="16", help="tile size, or 'max'")
    ap.add_argument("--draw", type=int, default=0, help="calibration draw")
    ap.add_argument("--model", default=R.DEFAULT_MODEL)
    ap.add_argument("--resume-root", type=Path, required=True,
                    help="PERSISTENT storage; needs ~18 GiB for one point")
    ap.add_argument("--hours", type=float, default=11.0,
                    help="stop at the first block boundary past this")
    ap.add_argument("--calib-samples", type=int, default=R.CALIB_SAMPLES)
    ap.add_argument("--calib-seqlen", type=int, default=2048,
                    help="2048, not m1_run's 4096: C4 cannot supply 4096-token "
                         "windows at the sampler's try budget (0.33%% of "
                         "documents are that long)")
    ap.add_argument("--datasets", nargs="*", default=["wikitext2"],
                    help="evaluation sets; the preregistration wants c4 too, "
                         "and it has never been timed")
    ap.add_argument("--max-eval-windows", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None,
                    help="where the finished point's JSON goes")
    args = ap.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("no CUDA device; this needs a GPU runtime", file=sys.stderr)
        return 1

    tile = args.tile if args.tile == "max" else int(args.tile)
    spec = R.PointSpec(model=args.model, budget_bits=args.budget,
                       tile_size=tile, draw=args.draw)
    print(f"point: {spec.slug()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu:   {p.name}, {p.total_memory / 2 ** 30:.1f} GiB")

    code, out = run(
        spec, resume_root=args.resume_root, hours=args.hours,
        calib_samples=args.calib_samples, calib_seqlen=args.calib_seqlen,
        eval_datasets=tuple(args.datasets),
        max_eval_windows=args.max_eval_windows, device=args.device)

    if code == 0 and out is not None:
        dest = args.out or (args.resume_root / f"{spec.slug()}.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nfinished in {out['seconds'] / 3600:.2f} h")
        for name, ppl in out["perplexity"].items():
            print(f"  {name}: {ppl:.4f}")
        print(f"wrote {dest}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
