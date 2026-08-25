"""Do the two remaining levers survive a real block, or only a microbenchmark?

`docs/STATUS.md` section 8.5 carries three pipeline levers.  One of them, fp16
search, was on for eight hours on 2026-08-25 and then withdrawn by measurement:
the 1.09-1.22x that justified it came from ONE layer at 512 of its 4096 rows,
and re-measured at the widths a real block has it read 1.00x.  The other two
have never been through that check:

    rotate_kron       5.52x on the rotation term      (section 6.8)
    compensate_block  6.63x on the compensation term  (`COMPENSATE_TIMINGS`)

Both are DERIVED in exactly the sense fp16's factor was -- measured on the term
ALONE and then handed to the cost model as a ratio.  The model is 5.2x
optimistic on a real block (section 6.16) and derived lever factors are the
named suspect.

SO THIS FILE MEASURES EACH LEVER TWICE, ON PURPOSE, IN ONE PROCESS:

    micro     the term by itself, at THIS layer's real widths
    in situ   the whole of `run_config`, with the lever flipped and
              nothing else changed

A lever whose two numbers agree is one the cost model may keep charging as a
term ratio.  One whose in-situ saving is smaller saves less than the model
believes, and that difference is precisely the gap section 6.16 is hunting.
Neither number alone can say which -- which is why the micro arm is measured
here rather than quoted from the record.  Quoting it is how fp16 lasted a day.

WHAT IS CONTROLLED, and every item is a trap this project has already paid for:

  * ONE PROCESS, ROUND ROBIN.  The same measurement moves 14-37% between runs
    on this machine, so `bench_guard.alternating` interleaves the arms.
  * THE PATH IS COUNTED BEFORE THE CLOCK IS READ.  Section 14.2: a measurement
    that does not traverse the changed path returns a believable number and
    proves nothing.  Every arm runs once under spies on
    `rotation.kronecker_factors`, `rotation.rotate_hessian` and
    `prune.forward_compensate`, and the run is abandoned unless the arms differ
    where they should AND agree everywhere else -- same tile count, same number
    of rotations, same compensation call.
  * A FOREIGN PROCESS ARRIVING MID-RUN ABORTS IT.  Contention lands on both
    arms, the spread stays small, and every ratio drifts toward 1.00x.
  * HOST RAM IS CHECKED TOO, not just the card.  Stage one materializes a
    13.5 GB checkpoint in host memory, and on a machine that is also running
    something else that is the step which fails -- by paging, which looks like
    slowness rather than an error.

TWO STAGES, because the first is expensive and the second wants repeating.
Stage one loads the model once, takes block 0's real Hessians for every linear
and writes (W, H) per layer to a cache.  Stage two reads one layer at a time and
measures.  Re-running the audit after that costs no model load at all.

WHAT THIS IS NOT.  Block 0 only, and its statistics come from the dense model --
there is nothing above block 0 to compress.  That makes the widths, the tile
counts and the timings exactly right and leaves the quality column weaker
evidence than a full sequential run, which spec v6 trap 20 requires.  Times are
what this file is for; quality is a free by-product and is labelled as one.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accounting as A               # noqa: E402
import calibrate as Cal              # noqa: E402
import compact as C                  # noqa: E402
import hf_llama as HF                # noqa: E402
import m1_gates as M                 # noqa: E402
import prune as P                    # noqa: E402
import quantize as Qz                # noqa: E402
import rotation as R                 # noqa: E402
import tiling as Tl                  # noqa: E402
from bench_guard import (alternating, foreign_compute_pids,     # noqa: E402
                         require_quiet_gpu)

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"

#: Every linear in a block, in the order the pipeline compresses them.
ALL_LAYERS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
              "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
              "mlp.down_proj")

#: The three DISTINCT shapes a Llama-2-7B block has.  q/k/v/o are one shape and
#: gate/up another, so timing all seven would repeat two of them five times over
#: -- and `COMPENSATE_TIMINGS` is keyed on exactly these three.  `o_proj` is
#: worth adding when the question is quality rather than time, because that is
#: the layer sections 6.8 and 6.9 measured, at 512 of its 4096 rows.
DEFAULT_LAYERS = ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj")

#: The arms.  `pipeline` passes nothing, so it inherits whatever the pipeline
#: currently runs with and the record says which that was.  Pinning the values
#: here would make this experiment stop tracking the thing it audits.
ARMS: dict[str, dict] = {
    "pipeline": {},
    "no_kron": {"rotate_kron": False},
    "no_block": {"compensate_block": None},
}

#: What the cost model is told each lever is worth, so the report can put the
#: claim next to the measurement instead of leaving the reader to look it up.
CLAIMED = {"rotation": 5.52, "compensate": 6.63}


# --------------------------------------------------------------------------- #
# The machine, not just the card
# --------------------------------------------------------------------------- #

class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def host_memory_gib() -> tuple[float, float]:
    """(available, total) physical RAM.  (0, 0) where it cannot be read.

    Here because the card is not the only thing that can be busy, and because
    `psutil` is not a dependency of this project and is not going to become one
    for two numbers.  The stage that host memory decides is the model load: it
    does not raise when the memory is short, it pages, and paging reads as a
    slow machine rather than a wrong one.
    """
    if not sys.platform.startswith("win"):
        import os
        try:
            return (os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30,
                    os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2 ** 30)
        except (ValueError, AttributeError, OSError):
            return (0.0, 0.0)
    st = _MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return (0.0, 0.0)
    return (st.ullAvailPhys / 2 ** 30, st.ullTotalPhys / 2 ** 30)


def require_host_memory(need_gib: float, *, strict: bool = True) -> float:
    """Refuse a load that would page, and say by how much it would."""
    avail, total = host_memory_gib()
    if total == 0.0:
        return 0.0
    if avail < need_gib:
        msg = (f"{avail:.1f} GiB of host RAM available, {need_gib:.1f} needed "
               "for the checkpoint.  Loading anyway pages to disk: it will "
               "finish, eventually, and nothing downstream will say why it was "
               "slow")
        if strict:
            raise RuntimeError(msg)
        print(f"  [WARNING] {msg}")
    return avail


# --------------------------------------------------------------------------- #
# Stage one: block 0's real problems, once
# --------------------------------------------------------------------------- #

def build_block_problems(
    model_name: str = DEFAULT_MODEL, *,
    names: tuple[str, ...] = ALL_LAYERS,
    n_seqs: int = 16, seqlen: int = 2048, batch: int = 4,
    dataset: str = "wikitext2", device: str | None = None,
    dtype: torch.dtype = torch.float32,
    progress=print,
) -> dict[str, dict]:
    """Every linear of block 0, with the Hessian its own calibration produces.

    ONE model load for all seven rather than `m0_rotation_value.build_problem`
    once per layer.  That function is right for a study of one layer; here the
    seven Hessians are the point, and loading a 13.5 GB checkpoint seven times
    would be most of the wall clock.

    Accumulated in float32 ON THE CARD, which is what `m1_run.py` passes and
    what the pipeline consumes.  float64 on a consumer GPU runs at roughly 1/64
    rate and is worth 36 days of M1 (section 6.10).

    Returns CPU tensors, ready to cache.  The audit wants them one at a time:
    seven Hessians resident at once is 0.87 GiB of a 6.8 GiB card, and section
    6.17 measured that as 1.46x of the compression's wall clock.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    require_host_memory(13.0, strict=False)

    progress(f"loading {model_name} (cpu, fp16) ...")
    t0 = time.time()
    harness = HF.load_llama(model_name, dtype=torch.float16)
    progress(f"  {time.time() - t0:.0f}s, {host_memory_gib()[0]:.1f} GiB RAM left")

    progress(f"tokenizing {n_seqs} x {seqlen} calibration tokens ...")
    tokens = Cal.load_calibration_tokens(harness.tokenizer, n_samples=n_seqs,
                                         seqlen=seqlen, seed=0, dataset=dataset)
    batches = [tokens[i:i + batch] for i in range(0, tokens.shape[0], batch)]

    progress("capturing block 0 inputs ...")
    hidden, block_kwargs = HF.capture_block_inputs(harness.model, batches)

    block = harness.blocks[0]
    linears = Cal.find_linears(block)
    missing = set(names) - set(linears)
    if missing:
        raise KeyError(f"block 0 has no {sorted(missing)}; it has "
                       f"{sorted(linears)}")
    weights = {name: linears[name].weight.detach().to(torch.float32).cpu().clone()
               for name in names}

    progress(f"accumulating {len(names)} Hessians on {device} ...")
    t0 = time.time()
    block.to(device)
    hidden = [h.to(device) for h in hidden]
    # Nested: the rotary embeddings arrive as a tuple, so this has to recurse.
    block_kwargs = HF.to_device(block_kwargs, device)
    accs = Cal.collect_block_statistics(block, hidden, block_kwargs=block_kwargs,
                                        names=list(names), dtype=dtype)
    progress(f"  {time.time() - t0:.0f}s, "
             f"{next(iter(accs.values())).n_tokens:,} tokens")

    out = {}
    for name in names:
        out[name] = {
            "W": weights[name],
            "H": accs[name].H.to(torch.float32).cpu().clone(),
            "n_tokens": int(accs[name].n_tokens),
            "model": model_name,
            "block": 0,
            "layer": name,
            "calibration": {"n_seqs": n_seqs, "seqlen": seqlen,
                            "dataset": dataset},
        }
        # Dropped as it is written, for the reason section 6.17 measured: the
        # allocator wants the room back more than the card wants the ceiling.
        accs.pop(name)

    del harness, hidden, block, linears, accs, weights, tokens, batches
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def cache_problems(cache: Path, problems: dict[str, dict], progress=print) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, blob in problems.items():
        path = cache / f"{name.replace('.', '_')}.pt"
        torch.save(blob, path)
        progress(f"  wrote {path.name} ({path.stat().st_size / 2 ** 20:.0f} MiB)")


def load_problem(cache: Path, name: str, *, device: str = "cuda",
                 dtype: torch.dtype = torch.float32) -> Cal.LayerProblem:
    path = cache / f"{name.replace('.', '_')}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run once with --build, which loads the model")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return Cal.LayerProblem.from_statistics(
        blob["W"].to(device, dtype), blob["H"].to(device, dtype),
        name=f"{blob['model']}:layers.{blob['block']}.{name}",
        n_tokens=blob["n_tokens"],
    )


# --------------------------------------------------------------------------- #
# Counting the path before reading the clock
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Trace:
    """Where each lever actually acted during one `run_config`."""
    kron_built: int = 0
    rotations_kron: int = 0
    rotations_dense: int = 0
    compensate_blocks: tuple = ()

    @property
    def rotations(self) -> int:
        return self.rotations_kron + self.rotations_dense

    def __str__(self) -> str:
        return (f"factors={self.kron_built} rot={self.rotations_kron}K/"
                f"{self.rotations_dense}D comp={list(self.compensate_blocks)}")


def run_traced(problem: Cal.LayerProblem, **kw) -> tuple[dict, Trace]:
    """`run_config`, with every lever counted where it acts.

    Watching the returned record is not enough, and the project has the scar: a
    record can say `compensate_block=512` while the argument goes nowhere, which
    is how that lever stayed unreachable from the driver for a day.  So the
    count comes from the functions themselves.
    """
    seen: dict = {"kron": 0, "rk": 0, "rd": 0, "blocks": []}
    real_factors = R.kronecker_factors
    real_rotate = R.rotate_hessian
    real_comp = P.forward_compensate

    def spy_factors(*a, **k):
        seen["kron"] += 1
        return real_factors(*a, **k)

    def spy_rotate(H, Q=None, *, factors=None):
        seen["rk" if factors is not None else "rd"] += 1
        return real_rotate(H, Q, factors=factors)

    def spy_comp(W, keep, Hinv, block=None):
        seen["blocks"].append(block)
        return real_comp(W, keep, Hinv, block=block)

    R.kronecker_factors, R.rotate_hessian, P.forward_compensate = (
        spy_factors, spy_rotate, spy_comp)
    try:
        record = M.run_config(problem, **kw)
    finally:
        R.kronecker_factors, R.rotate_hessian, P.forward_compensate = (
            real_factors, real_rotate, real_comp)
    return record, Trace(seen["kron"], seen["rk"], seen["rd"],
                         tuple(seen["blocks"]))


def check_paths(traces: dict[str, Trace]) -> None:
    """Abandon the run unless the arms differ where they should and only there.

    Both halves matter.  An arm that never takes the changed path measures
    nothing -- section 14.2's rule, learned from a quality difference of exactly
    0.0000% on a layer whose routing never moved.  And an arm that differs
    ELSEWHERE is not a lever measurement at all: if the tile counts diverge, the
    two arms are compressing different problems and the delta is not the lever.
    """
    pipe = traces["pipeline"]
    problems = []

    if pipe.kron_built != 1 or pipe.rotations_kron == 0 or pipe.rotations_dense:
        problems.append(f"pipeline did not take the Kronecker path: {pipe}")
    if pipe.compensate_blocks != (M.PIPELINE_COMPENSATE_BLOCK,):
        problems.append(f"pipeline did not block the compensation: {pipe}")

    if "no_kron" in traces:
        nk = traces["no_kron"]
        if nk.kron_built or nk.rotations_kron:
            problems.append(f"no_kron still built Kronecker factors: {nk}")
        if nk.rotations_dense != pipe.rotations:
            problems.append(
                f"no_kron rotated {nk.rotations_dense} sub-Hessians against the "
                f"pipeline's {pipe.rotations}; not the same problem")
        if nk.compensate_blocks != pipe.compensate_blocks:
            problems.append(f"no_kron moved the compensation too: {nk}")

    if "no_block" in traces:
        nb = traces["no_block"]
        if nb.compensate_blocks != (None,):
            problems.append(f"no_block did not reach the exact sweep: {nb}")
        if nb.rotations_kron != pipe.rotations_kron or nb.rotations_dense:
            problems.append(f"no_block moved the rotation too: {nb}")

    if problems:
        raise RuntimeError(
            "the arms do not isolate the levers, so nothing timed here would "
            "mean anything:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# The same two terms, measured alone
# --------------------------------------------------------------------------- #

def micro(problem: Cal.LayerProblem, *, budget_bits: float, tile_size,
          seed: int = 0, reps: int = 5, warmup: int = 2,
          strict: bool = True, progress=print) -> dict:
    """Each lever's term BY ITSELF, at this layer's real widths.

    This is the arrangement both recorded factors came from -- 5.52x is a
    tile-weighted average of `rotate_hessian` timings and 6.63x is the ratio of
    `COMPENSATE_TIMINGS`' two columns -- reproduced here so the comparison with
    the in-situ delta is between two numbers taken minutes apart on one card,
    rather than between a measurement and a quotation.
    """
    scheme = {1: "unstructured", Tl.MAX_TILE: "structured"}.get(tile_size, "tile")
    density = A.density_for_budget(scheme, budget_bits, None, problem.n_in,
                                   tile_size=tile_size,
                                   vq_bits=Qz.E8P_BITS_PER_WEIGHT)
    if density is None or not 0.0 < density <= 1.0:
        return {"skipped": "budget unreachable at this tile size"}

    grabbed: dict = {}
    real_comp = P.forward_compensate

    def spy(W, keep, Hinv, block=None):
        grabbed["args"] = (W, keep, Hinv)
        return real_comp(W, keep, Hinv, block=block)

    P.forward_compensate = spy
    try:
        pruned = P.prune(
            problem.W, axis="B", tile_size=tile_size, density=density,
            metric="wanda", act_norm=problem.act_norm, H=problem.H,
            compensate=True, compensate_block=M.PIPELINE_COMPENSATE_BLOCK,
            align=Qz.E8P_DIM,
        )
    finally:
        P.forward_compensate = real_comp
    if "args" not in grabbed:
        raise RuntimeError("prune never compensated; there is nothing to time")

    W_c, keep, Hinv = grabbed["args"]
    progress("  micro: compensation sweep ...")
    comp = alternating(
        {"exact": lambda: real_comp(W_c, keep, Hinv, block=None),
         "blocked": lambda: real_comp(W_c, keep, Hinv,
                                      block=M.PIPELINE_COMPENSATE_BLOCK)},
        reps=reps, warmup=warmup, strict=strict,
    )
    grabbed.clear()
    del W_c, keep, Hinv
    gc.collect()

    cw = C.compact(pruned.W, pruned.mask)
    rotated, Qm = R.rotate(cw, axis="index", seed=seed)
    factors = R.kronecker_factors(cw.k, seed, rotated.blocks.dtype,
                                  rotated.blocks.device)
    dense_one = M.tile_hessian_stream(problem, cw, Qm, factors=None)
    kron_one = M.tile_hessian_stream(problem, cw, Qm, factors=factors)
    progress("  micro: sub-Hessian rotation ...")
    # One tile.  Both arms gather `H[idx, idx]` first, exactly as the sweep
    # does, so what separates them is the rotation and nothing else.
    rot = alternating({"dense": lambda: dense_one(0), "kron": lambda: kron_one(0)},
                      reps=reps, warmup=warmup, strict=strict)

    n_tiles = int(cw.n_tiles)
    out = {
        "k": int(cw.k),
        "n_tiles": n_tiles,
        "lines_per_tile": int(cw.lines_per_tile),
        "density_requested": density,
        "density_realized": pruned.mask.density(),
        "compensate": {
            "exact_s": comp["exact"]["median"],
            "blocked_s": comp["blocked"]["median"],
            "ratio": comp["exact"]["median"] / comp["blocked"]["median"],
            "saving_s": comp["exact"]["median"] - comp["blocked"]["median"],
            "spread": max(comp["exact"]["spread"], comp["blocked"]["spread"]),
        },
        "rotation": {
            "dense_per_tile_s": rot["dense"]["median"],
            "kron_per_tile_s": rot["kron"]["median"],
            "ratio": rot["dense"]["median"] / rot["kron"]["median"],
            "saving_s": n_tiles * (rot["dense"]["median"] - rot["kron"]["median"]),
            "dense_layer_s": n_tiles * rot["dense"]["median"],
            "kron_layer_s": n_tiles * rot["kron"]["median"],
            "spread": max(rot["dense"]["spread"], rot["kron"]["spread"]),
        },
    }
    del pruned, cw, rotated, Qm, factors, dense_one, kron_one
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# The rotation term at every width the grid really uses
# --------------------------------------------------------------------------- #

def rot_sweep(problem: Cal.LayerProblem, *, budget_bits: float, tiles,
              seed: int = 0, reps: int = 5, warmup: int = 2,
              strict: bool = True, progress=print) -> list[dict]:
    """Dense against Kronecker, one tile, at each tile size's `k`.

    WHY THIS EXISTS AND WHY IT IS NOT OPTIONAL.  The audit measures the lever at
    T=16, which gives two widths.  Pricing the rest of the grid from those two
    requires extrapolating, and the first attempt did it by flop count and got
    k=1024 wrong by 5.4x IN THE PESSIMISTIC DIRECTION -- because 1024 is a pure
    power of two, `kronecker_factors` returns `m=1`, and the contraction
    degenerates to the dense product.  Section 6.8 had measured exactly that
    (0.99x at k=2048) and the extrapolation walked straight past it.

    Which is this project's most-repeated failure with a new subject: a constant
    measured in one regime and applied in all of them.  So the widths are
    measured instead of interpolated, the same answer section 6.14 reached about
    `TILE_TIMINGS` when it turned out to have no sample below four lines.

    The mask comes from a real `prune`, with the compensation off: compensating
    changes the weights, not the survivor set, so the index sets -- and with them
    the gather's locality, which is inside these timings -- are the pipeline's.
    """
    rows = []
    for tile_size in tiles:
        scheme = {1: "unstructured", Tl.MAX_TILE: "structured"}.get(tile_size, "tile")
        density = A.density_for_budget(scheme, budget_bits, None, problem.n_in,
                                       tile_size=tile_size,
                                       vq_bits=Qz.E8P_BITS_PER_WEIGHT)
        if density is None or not 0.0 < density <= 1.0:
            progress(f"    T={tile_size}: budget unreachable")
            continue
        pruned = P.prune(problem.W, axis="B", tile_size=tile_size,
                         density=density, metric="wanda",
                         act_norm=problem.act_norm, H=None, compensate=False,
                         align=Qz.E8P_DIM)
        cw = C.compact(pruned.W, pruned.mask)
        rotated, Qm = R.rotate(cw, axis="index", seed=seed)
        factors = R.kronecker_factors(cw.k, seed, rotated.blocks.dtype,
                                      rotated.blocks.device)
        dense_one = M.tile_hessian_stream(problem, cw, Qm, factors=None)
        kron_one = M.tile_hessian_stream(problem, cw, Qm, factors=factors)
        t = alternating({"dense": lambda: dense_one(0),
                         "kron": lambda: kron_one(0)},
                        reps=reps, warmup=warmup, strict=strict)
        k = int(cw.k)
        a = (k & -k).bit_length() - 1
        rows.append({
            "n_in": problem.n_in, "tile_size": str(tile_size), "k": k,
            "factors": f"{1 << a}x{k >> a}", "n_tiles": int(cw.n_tiles),
            "dense_s": t["dense"]["median"], "kron_s": t["kron"]["median"],
            "ratio": t["dense"]["median"] / t["kron"]["median"],
            "spread": max(t["dense"]["spread"], t["kron"]["spread"]),
        })
        r = rows[-1]
        progress(f"    T={str(tile_size):<4} k={k:<5} = {r['factors']:<9} "
                 f"dense {r['dense_s'] * 1e3:8.3f} ms  kron {r['kron_s'] * 1e3:8.3f} ms"
                 f"  {r['ratio']:6.2f}x  spread {r['spread']:.1%}")
        del pruned, cw, rotated, Qm, factors, dense_one, kron_one
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #
# One layer, both ways
# --------------------------------------------------------------------------- #

def no_newcomers(baseline: dict[int, str], *, strict: bool = True) -> list:
    """Has a foreign process ARRIVED on the card since the session started?

    A per-layer check that is deliberately NOT `require_quiet_gpu`.  That one
    reads the SM clock, and its own docstring says to call it before the speed
    phase and never during -- because once our kernels are in flight the clock
    is high on our account.  Calling it between two layers is "during": the
    first draft of this file did exactly that and was refused at 88% of maximum,
    by its own previous arm.  Section 6.17 records the identical mistake being
    made in `bench_guard` itself a few hours earlier, which is the argument for
    writing the reason down here rather than just reordering the calls.

    A foreign PID is the signal that survives our own load, so it is the one a
    mid-session check can use.
    """
    newcomers = [(p, n) for p, n in foreign_compute_pids() if p not in baseline]
    if newcomers:
        who = ", ".join(f"{p} ({n.rsplit(chr(92), 1)[-1]})" for p, n in newcomers)
        msg = f"another process is on the card: {who}"
        if strict:
            raise RuntimeError("refusing to time: " + msg)
        print(f"  [WARNING] {msg}")
    return newcomers


def audit_layer(problem: Cal.LayerProblem, *, budget_bits: float, tile_size,
                arms: dict[str, dict] | None = None, reps: int = 3,
                warmup: int = 1, micro_reps: int = 5, strict: bool = True,
                baseline: dict[int, str] | None = None,
                progress=print) -> dict:
    arms = ARMS if arms is None else arms
    base = {"budget_bits": budget_bits, "tile_size": tile_size}
    no_newcomers(baseline or {}, strict=strict)
    avail, total = host_memory_gib()
    progress(f"  host RAM {avail:.1f}/{total:.1f} GiB free")

    # One discarded pass before anything is recorded.  Without it the FIRST arm
    # pays for `torch.compile` on two elementwise kernels and reads several
    # times the others -- 10.8 s against 1.6 s on the fixture this was found on
    # -- and that cost belongs to no arm.  It is charged to none of them.
    progress("  warming the compiled kernels (discarded) ...")
    M.run_config(problem, **base, **arms["pipeline"])

    progress(f"  paths: {len(arms)} arms under spies ...")
    records, traces = {}, {}
    for name, kw in arms.items():
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        rec, tr = run_traced(problem, **base, **kw)
        rec["first_pass_seconds"] = time.perf_counter() - t0
        rec["peak_gib"] = (torch.cuda.max_memory_allocated() / 2 ** 30
                           if torch.cuda.is_available() else 0.0)
        records[name], traces[name] = rec, tr
        progress(f"    {name:<9} {tr}  {rec['first_pass_seconds']:6.1f}s  "
                 f"peak {rec['peak_gib']:.2f} GiB  "
                 f"err {rec['rel_output_error']:.6f}")
    check_paths(traces)

    def make(kw):
        def fn():
            M.run_config(problem, **base, **kw)
        return fn

    progress(f"  speed: {warmup} warmup + {reps} reps, round robin ...")
    timed = alternating({n: make(kw) for n, kw in arms.items()},
                        reps=reps, warmup=warmup, strict=strict)
    for name, t in timed.items():
        progress(f"    {name:<9} {t['median']:7.2f}s  spread {t['spread']:.1%}")

    mic = micro(problem, budget_bits=budget_bits, tile_size=tile_size,
                reps=micro_reps, warmup=2, strict=strict, progress=progress)

    pipe = timed["pipeline"]["median"]
    levers = {}
    for arm, term in (("no_kron", "rotation"), ("no_block", "compensate")):
        if arm not in timed or "skipped" in mic:
            continue
        observed = timed[arm]["median"] - pipe
        predicted = mic[term]["saving_s"]
        off_err = records[arm]["rel_output_error"]
        levers[term] = {
            "arm": arm,
            "off_s": timed[arm]["median"],
            "on_s": pipe,
            "observed_saving_s": observed,
            "micro_saving_s": predicted,
            # How much of what the term alone promised the pass actually keeps.
            "kept": observed / predicted if predicted else float("nan"),
            "term_ratio_micro": mic[term]["ratio"],
            "claimed_term_ratio": CLAIMED[term],
            "layer_speedup": timed[arm]["median"] / pipe if pipe else float("nan"),
            "share_of_pass": observed / timed[arm]["median"] if observed else 0.0,
            "quality_off": off_err,
            "quality_on": records["pipeline"]["rel_output_error"],
            "quality_pct": ((records["pipeline"]["rel_output_error"] / off_err - 1.0)
                            * 100.0 if off_err else float("nan")),
        }

    return {
        "layer": problem.name,
        "n_out": problem.n_out, "n_in": problem.n_in,
        "n_tokens": problem.n_tokens,
        "budget_bits": budget_bits,
        "tile_size": str(tile_size),
        "resolved": {k: records["pipeline"].get(k)
                     for k in ("rotate_kron", "compensate_block", "search_dtype",
                               "hessian_block", "chunk", "density_realized",
                               "survivors_per_tile", "bits_realized")},
        "traces": {n: str(t) for n, t in traces.items()},
        "timed": {n: {**{k: v for k, v in t.items() if k != "samples"},
                      "samples": [round(s, 4) for s in t["samples"]]}
                  for n, t in timed.items()},
        "first_pass_seconds": {n: r["first_pass_seconds"]
                               for n, r in records.items()},
        "peak_gib": {n: r["peak_gib"] for n, r in records.items()},
        "quality": {n: r["rel_output_error"] for n, r in records.items()},
        "micro": mic,
        "levers": levers,
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def report(out: dict) -> None:
    print()
    print("=" * 78)
    print("LEVER AUDIT -- the term alone against the same term in situ, "
          f"B={out['budget_bits']}, T={out['tile_size']}")
    print("=" * 78)
    for layer in out["layers"]:
        if "error" in layer:
            print(f"\n{layer['layer']}: FAILED -- {layer['error']}")
            continue
        m = layer["micro"]
        print(f"\n{layer['layer'].rsplit('.', 1)[-1]}  "
              f"{layer['n_out']}x{layer['n_in']}  "
              f"k={m.get('k')} tiles={m.get('n_tiles')}")
        pipe = layer["timed"]["pipeline"]
        print(f"  pipeline {pipe['median']:.2f}s "
              f"(spread {pipe['spread']:.1%}, "
              f"first pass {layer['first_pass_seconds']['pipeline']:.1f}s, "
              f"peak {layer['peak_gib']['pipeline']:.2f} GiB)")
        for term, lv in layer["levers"].items():
            print(f"  {term:<11} off {lv['off_s']:8.2f}s -> on {lv['on_s']:8.2f}s"
                  f"    layer {lv['layer_speedup']:.3f}x")
            print(f"  {'':<11} term ratio  claimed {lv['claimed_term_ratio']:5.2f}x"
                  f"    micro here {lv['term_ratio_micro']:5.2f}x")
            print(f"  {'':<11} saving      micro {lv['micro_saving_s']:8.2f}s"
                  f"    in situ {lv['observed_saving_s']:8.2f}s"
                  f"    kept {lv['kept']:.0%}")
            print(f"  {'':<11} quality     {lv['quality_pct']:+.3f}%  "
                  f"({lv['quality_on']:.6f} on / {lv['quality_off']:.6f} off)")
    print()


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache", type=Path, default=Path("results/block0_problems"),
                    help="where stage one leaves (W, H) for each layer")
    ap.add_argument("--build", action="store_true",
                    help="stage one: load the model and fill the cache")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--build-layers", nargs="*", default=list(ALL_LAYERS))
    ap.add_argument("--layers", nargs="*", default=list(DEFAULT_LAYERS))
    ap.add_argument("--rot-sweep", action="store_true",
                    help="instead of the audit: dense vs Kronecker at every "
                         "width the grid uses, for m0_cost_model")
    ap.add_argument("--rot-layers", nargs="*",
                    default=["self_attn.q_proj", "mlp.down_proj"],
                    help="one layer per distinct n_in; k depends on n_in alone")
    ap.add_argument("--rot-tiles", nargs="*",
                    default=[str(t) for t in M.DEFAULT_TILES],
                    help="the grid's tile sizes; 'max' for the structured edge")
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tile", default="16")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--micro-reps", type=int, default=5)
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-strict", action="store_true",
                    help="report contention instead of refusing to time")
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_lever_audit.json"))
    args = ap.parse_args(argv)

    tile = Tl.MAX_TILE if args.tile == "max" else int(args.tile)
    strict = not args.no_strict

    # THE ONE clock-based check, and it happens here because here is the only
    # moment it can mean anything: before this process has run a single kernel.
    # Everything after is guarded on foreign PIDs instead -- see `no_newcomers`.
    baseline: dict[int, str] = {}
    if torch.cuda.is_available() and not args.build_only:
        print(f"pre-flight: {require_quiet_gpu(strict=strict)}")
        baseline = dict(foreign_compute_pids())
        if baseline:
            print(f"  baseline foreign processes (tolerated): "
                  f"{sorted(baseline)}")
    avail, total = host_memory_gib()
    print(f"pre-flight: host RAM {avail:.1f}/{total:.1f} GiB free")

    if args.build or args.build_only:
        problems = build_block_problems(
            args.model, names=tuple(args.build_layers), n_seqs=args.seqs,
            seqlen=args.seqlen, device=args.device)
        cache_problems(args.cache, problems)
        del problems
        gc.collect()
        if args.build_only:
            return 0

    out = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "budget_bits": args.budget,
        "tile_size": args.tile,
        "device": (torch.cuda.get_device_name(0)
                   if torch.cuda.is_available() else "cpu"),
        "reps": args.reps, "warmup": args.warmup,
        "pipeline_defaults": {
            "rotate_kron": M.PIPELINE_ROTATE_KRON,
            "compensate_block": M.PIPELINE_COMPENSATE_BLOCK,
            "search_dtype": str(M.PIPELINE_SEARCH_DTYPE),
        },
        "layers": [],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.rot_sweep:
        out["rot_sweep"] = []
        for name in args.rot_layers:
            print(f"\n--- rotation sweep: {name} ---")
            problem = load_problem(args.cache, name, device=args.device)
            no_newcomers(baseline, strict=strict)
            out["rot_sweep"] += rot_sweep(
                problem, budget_bits=args.budget,
                tiles=[Tl.MAX_TILE if t == "max" else int(t)
                       for t in args.rot_tiles],
                reps=args.micro_reps, strict=strict)
            del problem
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
        return 0
    for name in args.layers:
        print(f"\n--- {name} ---")
        problem = load_problem(args.cache, name, device=args.device)
        try:
            out["layers"].append(audit_layer(
                problem, budget_bits=args.budget, tile_size=tile,
                reps=args.reps, warmup=args.warmup,
                micro_reps=args.micro_reps, strict=strict,
                baseline=baseline))
        except Exception as exc:                       # noqa: BLE001
            out["layers"].append(
                {"layer": name, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}")
        del problem
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Written after EVERY layer, not at the end.  The expensive layer is
        # last, it is the one a returning foreign process is most likely to
        # interrupt, and losing the two cheap ones with it would mean paying
        # for them again.
        args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    report(out)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
