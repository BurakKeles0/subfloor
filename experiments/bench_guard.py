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
    """
    if not arms:
        raise ValueError("no arms to time")
    if reps < 1:
        raise ValueError(f"reps must be positive, got {reps}")

    for _ in range(warmup):
        for fn in arms.values():
            _once(fn)

    samples: dict[str, list[float]] = {name: [] for name in arms}
    for _ in range(reps):
        for name, fn in arms.items():
            samples[name].append(_once(fn))

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
        }
    return out
