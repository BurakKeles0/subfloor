"""Is the card quiet enough to time on, and how much did the answer wobble.

`docs/STATUS.md` section 14.2 has carried the rule "verify the GPU is idle
before the speed phase" since it cost a measurement.  It stayed a DISCIPLINE --
look at `nvidia-smi` yourself -- and on 2026-08-25 the discipline failed in the
most embarrassing way available: the number it says to look at is wrong on this
machine, and reading it correctly still gives the wrong answer.

    utilization.gpu, no load at all     42 %
    utilization.gpu, saturating matmuls 25 %

It is ANTI-CORRELATED here.  The cause is WDDM: on a laptop whose GPU also
drives the display, `nvidia-smi` counts the Windows compositor, Edge WebView and
the Claude app as compute clients, and the figure it reports tracks display
work rather than the kernels anyone cares about.  A speed phase was postponed on
the strength of that 42 %.

So this file exists to make the check an ASSERTION rather than a habit, the same
way `quantize.is_canonical_codebook` was exported once a silent fast-path miss
had spoiled four measurements.

WHICH SIGNAL WORKS WAS MEASURED, NOT ASSUMED, and two of the three obvious ones
do not.  Against a foreign process running a matmul loop, and a second one
holding 3 GiB and doing nothing:

    signal                          idle      foreign load    foreign 3 GiB
    utilization.gpu                 42 %      99 %            41 %
    torch.cuda.mem_get_info free    6944 MiB  6944 MiB        6944 MiB
    clocks.sm                       ~1350     2745            1312

`mem_get_info` is BLIND: three gigabytes held by another process moved it not
one byte, because WDDM reports the calling context's budget rather than the
device's free memory.  `nvidia-smi --query-compute-apps` is no better -- it
lists the Windows shell and returns `[N/A]` for every process's memory.  So on
this machine only `clocks.sm` separates a working card from an idle one, and it
separates it cleanly: 42% of maximum against 89%.

The memory check is kept because it costs nothing and is not blind everywhere,
but it must never be the thing relied on -- hence the measured row above, so the
next person does not rediscover it as a surprise.

And a third signal, which is the one that cannot be fooled: the SPREAD across
alternating repeats.  Contention shows up as variance whatever its source,
including sources none of the counters name.  `alternating` reports it next to
every timing so a reader can see whether to believe the median.  That is why
this module does not stop at a pre-flight check: a pre-flight check answers "was
the card quiet a moment ago", the spread answers "was this measurement stable",
and only the second one is the question.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Callable

import torch

__all__ = ["GPUState", "gpu_state", "require_quiet_gpu", "alternating"]

#: Foreign memory tolerated before `require_quiet_gpu` complains.  The Windows
#: desktop alone reads ~1.2 GiB here.  MEASURED BLIND ON THIS MACHINE (see the
#: module docstring) -- a 3 GiB foreign allocation did not move the number -- so
#: this catches nothing locally and is kept only for platforms where
#: `mem_get_info` is device-wide.
FOREIGN_MIB_LIMIT = 2048

#: Fraction of the maximum SM clock above which the card is taken to be working
#: for somebody.  THE ONLY SIGNAL THAT WORKS HERE.  Idle samples land at 42% of
#: the 3090 MHz ceiling and a foreign matmul loop pins it at 89%, so the gap is
#: wide and the threshold is not delicate -- but note the idle floor is 42%, not
#: something near zero, because the display runs on the same card.  A threshold
#: chosen for a headless box would fail every check here.
BUSY_CLOCK_FRACTION = 0.60


@dataclass(frozen=True)
class GPUState:
    foreign_mib: float
    sm_clock_mhz: float
    max_clock_mhz: float
    utilization_pct: float          # reported, NOT trusted -- see the module docstring

    @property
    def clock_fraction(self) -> float:
        return self.sm_clock_mhz / self.max_clock_mhz if self.max_clock_mhz else 0.0

    def __str__(self) -> str:
        return (f"{self.foreign_mib:.0f} MiB foreign, "
                f"SM {self.sm_clock_mhz:.0f}/{self.max_clock_mhz:.0f} MHz "
                f"({self.clock_fraction:.0%}), "
                f"utilization {self.utilization_pct:.0f}% [untrusted]")


#: Executables nvidia-smi lists as "compute apps" on this machine that are the
#: desktop, not anyone's workload.  WDDM puts the shell, the search box and every
#: Electron window in that list, so a bare "is anything else on the card" check
#: is useless without it.
_DESKTOP = ("shellhost.exe", "crossdeviceresume.exe", "startmenuexperiencehost.exe",
            "searchhost.exe", "textinputhost.exe", "msedgewebview2.exe",
            "claude.exe", "explorer.exe", "dwm.exe", "ms-teams.exe",
            "whatsapp.root.exe")


def foreign_compute_pids() -> list[tuple[int, str]]:
    """Processes on the card that are neither ours nor the desktop.

    THE SIGNAL THAT SURVIVES OUR OWN LOAD, which the clock does not.  Once a
    measurement is running the clock is high because WE are working the card --
    `require_quiet_gpu` says so in as many words -- so a during-the-run check
    keyed on the clock fires on itself.  It did, on 2026-08-25, on the first
    measurement after the watch was added.

    A foreign PID is different in kind: it is there or it is not, and our own
    kernels cannot produce one.

    Two entries here are unnameable -- WDDM reports `[Insufficient Permissions]`
    for processes of other sessions -- and they are always present, so an
    absolute list is the wrong test.  `alternating` takes a BASELINE before it
    starts and flags only what ARRIVES, which is the thing a pre-flight check
    cannot see anyway.
    """
    import os
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0:
        return []
    mine = os.getpid()
    found = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid, name = int(parts[0]), parts[1]
        if pid == mine or name.rsplit("\\", 1)[-1].lower() in _DESKTOP:
            continue
        found.append((pid, name))
    return found


def _smi(fields: str) -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    return [f.strip() for f in out.stdout.strip().splitlines()[0].split(",")]


def gpu_state() -> GPUState:
    """One sample of what the card is doing, foreign memory included."""
    clock, max_clock, util = _smi("clocks.sm,clocks.max.sm,utilization.gpu")
    free, total = torch.cuda.mem_get_info()
    # Our own allocator's blocks are ours, so they do not count as foreign.
    ours = torch.cuda.memory_reserved()
    return GPUState(
        foreign_mib=max(0.0, (total - free - ours) / 2 ** 20),
        sm_clock_mhz=float(clock),
        max_clock_mhz=float(max_clock),
        utilization_pct=float(util),
    )


def require_quiet_gpu(
    *,
    samples: int = 4,
    foreign_mib_limit: float = FOREIGN_MIB_LIMIT,
    busy_clock_fraction: float = BUSY_CLOCK_FRACTION,
    strict: bool = True,
) -> GPUState:
    """Assert nothing else is working the card, and say what was seen.

    Call this BEFORE the speed phase, never during: once our own kernels are in
    flight the clock is high because of us and the signal is gone.

    `strict=False` reports instead of raising, for a run that wants the numbers
    recorded with the caveat attached rather than not taken at all.
    """
    worst = None
    for _ in range(samples):
        s = gpu_state()
        if worst is None or s.sm_clock_mhz > worst.sm_clock_mhz:
            worst = s
    assert worst is not None

    problems = []
    if worst.clock_fraction > busy_clock_fraction:
        problems.append(
            f"SM clock at {worst.clock_fraction:.0%} of maximum "
            f"({worst.sm_clock_mhz:.0f} MHz) -- something is running"
        )
    if worst.foreign_mib > foreign_mib_limit:
        problems.append(
            f"{worst.foreign_mib:.0f} MiB held by another process "
            f"(limit {foreign_mib_limit})"
        )
    if problems and strict:
        raise RuntimeError(
            "refusing to time on a busy GPU: " + "; ".join(problems) +
            ".  Contention does not merely add noise -- it moves the bottleneck "
            "onto the card and hides the very latency a launch-bound change "
            "removes, so the result reads 1.00x (docs/STATUS.md section 14.2)."
        )
    if problems:
        print(f"  [WARNING] timing on a busy GPU: {'; '.join(problems)}")
    return worst


def _once(fn: Callable[[], None]) -> float:
    """One timed call, with the queue drained on both sides.

    Both synchronizes matter and for different reasons: the first keeps the
    previous arm's tail out of this arm's time, the second stops an arm from
    "finishing" while its kernels are still queued.  Without the second one an
    arm that only ENQUEUES faster looks faster, which is precisely the mistake a
    launch-overhead measurement is trying not to make.
    """
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t0 = time.perf_counter()
    fn()
    sync()
    return time.perf_counter() - t0


def alternating(
    arms: dict[str, Callable[[], None]],
    *,
    reps: int = 5,
    warmup: int = 2,
    watch: bool = True,
    strict: bool = True,
) -> dict[str, dict]:
    """Run every arm round-robin in ONE process and report the spread.

    Round-robin rather than arm-by-arm because the same measurement moves 14-37%
    between runs on this machine, so two arms timed in sequence are not
    comparable at all -- one of this project's speedups once read +37% for a
    change that was actually -0.8%.  Interleaving makes a drift hit both arms.

    The warmup is not a formality: the SM clock idles at ~23% of maximum here
    and ramps under load, so the first repetitions of anything are measuring the
    clock coming up.  They are run and discarded.

    Returns per arm: median, min, max, and `spread` = (max - min) / median.  A
    large spread is the signal to distrust the median, and it catches contention
    that no counter reports.

    `watch` SAMPLES THE CARD BETWEEN REPETITIONS, and that is not the same check
    `require_quiet_gpu` does.  That one is pre-flight -- it answers "was the card
    quiet a moment ago" -- and this file's own docstring told callers to run it
    before the speed phase and not during.  A measurement that takes minutes can
    have another job arrive halfway through, and nothing would notice: the arms
    interleave, so contention lands on both and the SPREAD stays small while
    every ratio drifts toward 1.00x.  That is the exact signature of the trap
    section 14.2 records, and on 2026-08-25 it produced a clean-looking 0.993x
    and 1.005x for a lever whose whole benefit is bandwidth -- with another
    project's Python on the card the entire time.
    """
    if not arms:
        raise ValueError("no arms to time")
    if reps < 1:
        raise ValueError(f"reps must be positive, got {reps}")

    for _ in range(warmup):
        for fn in arms.values():
            _once(fn)

    # Sampled between repetitions, never inside a timed region: the nvidia-smi
    # call costs tens of milliseconds and would land in whichever arm ran next.
    #
    # A FOREIGN PID, not the clock.  The clock is high during a measurement
    # because we are the ones working the card, so a clock-keyed watch fires on
    # itself -- which is exactly what happened the first time this was written.
    watching = watch and torch.cuda.is_available()
    baseline = dict(foreign_compute_pids()) if watching else {}
    intruders: dict[int, str] = {}

    def sweep():
        for pid, name in foreign_compute_pids():
            if pid not in baseline:
                intruders[pid] = name

    samples: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(reps):
        if watching:
            sweep()
        for name, fn in arms.items():
            samples[name].append(_once(fn))
    if watching:
        sweep()

    out = {}
    for name, xs in samples.items():
        xs_sorted = sorted(xs)
        median = xs_sorted[len(xs_sorted) // 2]
        out[name] = {
            "median": median,
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
            "spread": (xs_sorted[-1] - xs_sorted[0]) / median if median else 0.0,
            "samples": xs,
            "foreign_processes": [f"{p} {n}" for p, n in intruders.items()],
        }

    if intruders:
        who = ", ".join(f"{p} ({n.rsplit(chr(92), 1)[-1]})"
                        for p, n in intruders.items())
        msg = (f"another process ARRIVED on the card mid-measurement: {who}."
               f"  Interleaving hides that -- contention lands on both arms, the "
               f"spread stays small, and every ratio drifts toward 1.00x")
        if strict:
            raise RuntimeError("timing taken on a contended GPU: " + msg)
        print(f"  [WARNING] {msg}")
    return out
