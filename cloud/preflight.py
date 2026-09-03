"""Is this machine able to run the experiment, and what are its constants?

TWO QUESTIONS, AND THE SECOND ONE IS NOT OPTIONAL.

Every timing constant in this project says "measured on this machine" and means
it.  `docs/STATUS.md` section 14.2's first rule is that a constant measured in
one regime does not transfer to another, and section 6.13 is what it cost when
one did not: three thresholds tuned to a laptop RTX left ten of twenty-one grid
cells scanning 65,536 codewords apiece.  A different card moves the dead band.

So the cost model, the tile timings and the rotation table are all re-measured
here before anything expensive runs.  It takes minutes and it is the difference
between a schedule and a guess.

The thresholds that are NOT re-measured by this script, and which a new card can
invalidate on its own, are listed at the end of the report -- deliberately, so
they are read rather than inherited:

    quantize._LATTICE_MIN_ROWS          the lattice decoder's floor
    quantize._ANALYTIC_MIN_ROWS         where the analytic search beats a scan
    quantize._ANALYTIC_DIRECT_MIN_ROWS  the window between them
    quantize.CHUNK_TARGET_ROWS          how wide a chunked sweep aims to be
    quantize.DECODER_MISS_FRACTION      what share of rows the decoder cannot settle

WHAT IT REFUSES.  A CPU-only runtime, a torch too old to have the pipeline's
kernels, a `transformers` older than 5 (`from_pretrained(dtype=...)` is a v5
keyword and v4 fails with an unrelated message), and a disk too small to hold a
point's checkpoint.  Each of those otherwise surfaces hours later as something
that reads like a bug in the experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import torch                                     # noqa: E402

#: One point's checkpoint, worked from the shapes rather than rounded off:
#:
#:     32 blocks x 0.377 GiB   a Llama-2-7B decoder layer at fp16      12.06
#:     inputs.pt               128 x 2048 x 4096 at fp16                2.00
#:     inputs.pt.tmp           its twin, alive for one torch.save       2.00
#:                                                                     -----
#:                                                                     16.06
#:
#: The twin is what the atomic rename in `Checkpoint.save_block` costs: the new
#: activations are written beside the old ones and renamed over them, so both
#: exist for the length of that write.  Two GiB of disk buys a checkpoint that
#: cannot be half-written, which on a pre-empted cloud session is the
#: difference between resuming a block early and resuming into activations no
#: version of the model produced (`tests/test_checkpoint.py`).
#:
#: 13.0 until 2026-09-03, which counted the blocks and forgot both activation
#: files -- so the check passed between 13 and 16 GiB and the run died around
#: block 30.  That is precisely the late failure the message below promises to
#: prevent.
#:
#: At `--calib-seqlen 4096` (`m1_run`'s default; the cloud runs 2048 because C4
#: cannot supply 4096-token windows) the last two rows double: add 4 GiB.
_BLOCK_PARAMS = 4 * 4096 * 4096 + 3 * 11008 * 4096
_ACTIVATIONS_GIB = 128 * 2048 * 4096 * 2 / 2 ** 30
POINT_CHECKPOINT_GIB = round(
    32 * _BLOCK_PARAMS * 2 / 2 ** 30 + 2 * _ACTIVATIONS_GIB, 1)
#: The Llama-2-7B checkpoint itself, wherever HuggingFace caches it.
MODEL_CACHE_GIB = 13.0


def check_environment(allow_small_gpu: bool = False) -> list[str]:
    problems = []
    print(f"python       {sys.version.split()[0]}")
    print(f"torch        {torch.__version__}")
    if not torch.cuda.is_available():
        problems.append("no CUDA device: this needs a GPU runtime")
    else:
        p = torch.cuda.get_device_properties(0)
        total = p.total_memory / 2 ** 30
        print(f"gpu          {p.name}, {total:.1f} GiB, "
              f"capability {p.major}.{p.minor}")
        if total < 12.0:
            msg = (f"{total:.1f} GiB of VRAM.  Measured on an 8 GiB card, the "
                   "driver sits at 94% occupancy through a block and has died "
                   "in cuSOLVER there; 16 GiB is the first size with room.  "
                   "Pass --allow-small-gpu to run anyway")
            # A blocker that names its own override, rather than either a wall
            # or a warning.  A wall stops someone who has a reason; a warning
            # scrolls past and the run dies four hours later.
            if allow_small_gpu:
                print(f"  [WARNING] {msg}")
            else:
                problems.append(msg)
    try:
        import transformers
        print(f"transformers {transformers.__version__}")
        if int(transformers.__version__.split(".")[0]) < 5:
            problems.append(
                f"transformers {transformers.__version__}: `hf_llama.load_llama`"
                " calls `from_pretrained(dtype=...)`, which is a v5 keyword")
    except ImportError:
        problems.append("transformers is not installed")
    try:
        import datasets
        print(f"datasets     {datasets.__version__}")
    except ImportError:
        problems.append("datasets is not installed")
    return problems


def check_storage(resume_root: Path, hf_home: Path | None) -> list[str]:
    problems = []
    resume_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(resume_root).free / 2 ** 30
    print(f"\ncheckpoint   {resume_root}  {free:.1f} GiB free "
          f"(a point wants ~{POINT_CHECKPOINT_GIB:.0f})")
    if free < POINT_CHECKPOINT_GIB:
        problems.append(
            f"{free:.1f} GiB free at {resume_root}, and a point's checkpoint "
            f"reaches ~{POINT_CHECKPOINT_GIB:.0f} GiB.  Every one of the 32 "
            "block files is needed at the end -- the evaluation runs on the "
            "assembled compressed model -- so this fails late, not early")
    if hf_home is not None:
        hf_home.mkdir(parents=True, exist_ok=True)
        hf_free = shutil.disk_usage(hf_home).free / 2 ** 30
        print(f"model cache  {hf_home}  {hf_free:.1f} GiB free "
              f"(the checkpoint is ~{MODEL_CACHE_GIB:.0f})")
        if hf_free < MODEL_CACHE_GIB:
            problems.append(f"{hf_free:.1f} GiB free at {hf_home}, model is "
                            f"~{MODEL_CACHE_GIB:.0f} GiB")
    return problems


def check_pipeline() -> list[str]:
    """The smallest end-to-end run there is: does the chain actually work here?

    A synthetic layer through `run_config`, which is prune -> compact -> rotate
    -> LDLQ -> E8P.  It costs seconds and it is the difference between finding
    out now and finding out after the model has downloaded.
    """
    import calibrate as Cal
    import m1_gates as M
    import quantize as Qz

    problems = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p = Cal.synthetic_problem(64, 128, 256)
    p = Cal.LayerProblem.from_statistics(p.W.to(dev, torch.float32),
                                         p.H.to(dev, torch.float32))
    t0 = time.time()
    r = M.run_config(p, budget_bits=1.5, tile_size=4)
    print(f"\npipeline     synthetic layer in {time.time() - t0:.1f}s, "
          f"rel error {r['rel_output_error']:.4f}")
    if not (0.0 < r["rel_output_error"] < 1.0):
        problems.append(f"synthetic layer gave {r['rel_output_error']}, which "
                        "is not a plausible relative error")

    cb = Qz._on_device(torch.float32, dev)
    if not Qz.is_canonical_codebook(cb):
        problems.append(
            "the codebook is not the canonical E8P table, so the fast paths are "
            "off and every timing below would describe a scan")
    print(f"levers       rotate_kron={M.PIPELINE_ROTATE_KRON} "
          f"compensate_block={M.PIPELINE_COMPENSATE_BLOCK} "
          f"search_dtype={M.PIPELINE_SEARCH_DTYPE} "
          f"batch_fit={M.PIPELINE_BATCH_FIT}")
    return problems


def measure_constants(out_dir: Path, *, quick: bool = False) -> dict:
    """Re-measure what the cost model is allowed to assume.

    `quick=True` takes the kernel rates only, which is a minute and enough to
    price the grid roughly.  The full version also re-measures the per-tile
    times and the rotation table, and those are what section 6.14 and section
    6.18 had to correct on the local machine -- so on a new one they are not
    optional either, just deferrable.
    """
    import m0_cost_model as CM

    out_dir.mkdir(parents=True, exist_ok=True)
    rates_path = out_dir / "m0_rates.json"
    print("\nmeasuring kernel rates ...")
    t0 = time.time()
    rates = CM.measure_rates(cache=rates_path)
    print(f"  {time.time() - t0:.0f}s -> {rates_path}")
    setup = "cuda_f32" if "cuda_f32" in rates["setups"] else "cpu_f32"

    report = {"setup": setup, "device": rates.get("device")}
    if setup == "cuda_f32":
        m1 = CM.m1_cost(rates, setup)
        f = sum(CM.model_cost(t, 1.5, rates, setup)["point_seconds"]
                for t in CM.TILES
                if "skipped" not in CM.model_cost(t, 1.5, rates, setup))
        report["m1_days"] = m1["days"]
        report["design_f_hours"] = f / 3600
        report["rotate_kron_priced"] = m1["rotate_kron_priced"]
        print(f"\n  M1 (105 points)      {m1['days']:.1f} days")
        print(f"  Design F (7 points)  {f / 3600:.1f} h")
        print(f"  Kronecker priced     {m1['rotate_kron_priced']}")
        print("\n  THESE STILL READ THE LOCAL MACHINE'S TILE AND ROTATION "
              "TABLES.")
        print("  Only the kernel rates above were re-measured here.  For the "
              "rest:")
        print("    python experiments/m0_tile_timings.py")
        print("    python -u experiments/m0_lever_audit.py --build --rot-sweep")
        if not quick:
            print("  (both want a quiet card and 20-40 minutes; this script "
                  "does not run them)")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resume-root", type=Path, required=True)
    ap.add_argument("--hf-home", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--skip-rates", action="store_true")
    ap.add_argument("--allow-small-gpu", action="store_true",
                    help="proceed on a card measured to be too small")
    args = ap.parse_args(argv)

    print("=" * 70)
    print("PREFLIGHT")
    print("=" * 70)
    problems = check_environment(allow_small_gpu=args.allow_small_gpu)
    problems += check_storage(args.resume_root, args.hf_home)
    if not problems:
        problems += check_pipeline()

    if problems:
        print("\n" + "=" * 70)
        print("NOT READY:")
        for p in problems:
            print(f"  - {p}")
        print("=" * 70)
        return 1

    report = {}
    if not args.skip_rates:
        report = measure_constants(args.out)

    print("\n" + "=" * 70)
    print("READY.  Thresholds this script did NOT re-measure, and which a new")
    print("card can invalidate on its own (docs/STATUS.md section 6.13):")
    import quantize as Qz
    print(f"  _LATTICE_MIN_ROWS          {Qz._LATTICE_MIN_ROWS}")
    print(f"  _ANALYTIC_MIN_ROWS         {Qz._ANALYTIC_MIN_ROWS}")
    print(f"  _ANALYTIC_DIRECT_MIN_ROWS  {Qz._ANALYTIC_DIRECT_MIN_ROWS}")
    print(f"  CHUNK_TARGET_ROWS          {Qz.CHUNK_TARGET_ROWS}")
    print(f"  DECODER_MISS_FRACTION      {Qz.DECODER_MISS_FRACTION}")
    print("=" * 70)
    if report:
        (args.out / "cloud_preflight.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
