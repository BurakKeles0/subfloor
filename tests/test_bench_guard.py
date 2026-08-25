"""The guard that decides whether a timing is worth believing.

`docs/STATUS.md` section 14.2 asked for this check as a discipline -- look at
`nvidia-smi` before the speed phase -- and the discipline failed on 2026-08-25
because the number it names reads 42% on a completely idle card here and 25%
under load.  A rule you have to apply by hand, against a signal that lies, is
not a rule.

So these tests are about the DECISION, not about any particular card: the state
is injected and what is checked is what the guard concludes from it.  A test that
needed a busy GPU could only run when the thing it tests is already broken.
"""

from __future__ import annotations

import pytest
import torch

import bench_guard as BG


def _state(clock=1300.0, foreign=1207.0, util=42.0, maximum=3090.0):
    return BG.GPUState(foreign_mib=foreign, sm_clock_mhz=clock,
                       max_clock_mhz=maximum, utilization_pct=util)


# --------------------------------------------------------------------------- #
# What the guard concludes
# --------------------------------------------------------------------------- #

def test_an_idle_card_passes_even_though_utilization_reads_high(monkeypatch):
    """The measured idle state on this machine: 42% utilization, 42% clock.

    A guard keyed on utilization would refuse to ever measure here, which is
    exactly the failure that prompted the module -- a speed phase was postponed
    on the strength of that number.
    """
    monkeypatch.setattr(BG, "gpu_state", lambda: _state(clock=1300.0, util=42.0))
    got = BG.require_quiet_gpu(samples=2)
    assert got.utilization_pct == 42.0        # high, and correctly ignored
    assert got.clock_fraction < BG.BUSY_CLOCK_FRACTION


def test_a_working_card_is_refused(monkeypatch):
    """The measured foreign-load state: SM pinned at 89% of maximum."""
    monkeypatch.setattr(BG, "gpu_state", lambda: _state(clock=2745.0, util=99.0))
    with pytest.raises(RuntimeError, match="busy GPU"):
        BG.require_quiet_gpu(samples=2)


def test_the_refusal_says_why_it_matters():
    """Not decoration.  A bare "GPU busy" invites someone to rerun and hope;
    the reason it reads 1.00x is what stops them."""
    import inspect
    src = inspect.getsource(BG.require_quiet_gpu)
    assert "1.00x" in src and "hides" in src


def test_a_memory_hog_is_refused_where_the_signal_works(monkeypatch):
    """Kept for platforms where `mem_get_info` is device-wide.  It is BLIND on
    this one -- 3 GiB foreign did not move it -- which is why the clock is the
    primary signal and this is the secondary."""
    monkeypatch.setattr(BG, "gpu_state", lambda: _state(foreign=6000.0))
    with pytest.raises(RuntimeError, match="another process"):
        BG.require_quiet_gpu(samples=2)


def test_the_worst_sample_decides_not_the_last(monkeypatch):
    """Contention is bursty.  Taking the last sample would let a quiet instant
    clear a card that was working a moment earlier."""
    clocks = iter([1300.0, 2745.0, 1300.0, 1300.0])
    monkeypatch.setattr(BG, "gpu_state", lambda: _state(clock=next(clocks)))
    with pytest.raises(RuntimeError, match="busy GPU"):
        BG.require_quiet_gpu(samples=4)


def test_non_strict_reports_instead_of_raising(monkeypatch, capsys):
    monkeypatch.setattr(BG, "gpu_state", lambda: _state(clock=2745.0))
    got = BG.require_quiet_gpu(samples=2, strict=False)
    assert got.sm_clock_mhz == 2745.0
    assert "WARNING" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The spread -- the signal no counter can fool
# --------------------------------------------------------------------------- #

def test_alternating_runs_the_arms_round_robin():
    """Arm-by-arm timing is not comparable on this machine: the same measurement
    moves 14-37% between runs, and one change here once read +37% when it was
    actually -0.8%.  Interleaving makes a drift hit both arms."""
    order = []
    arms = {"a": lambda: order.append("a"), "b": lambda: order.append("b")}
    BG.alternating(arms, reps=3, warmup=1, watch=False)
    # warmup pass, then three interleaved pairs -- never "aaa" then "bbb".
    assert "".join(order) == "ab" * 4
    assert order[2::2] == ["a", "a", "a"]


def test_alternating_reports_the_spread():
    """The median alone cannot say whether to believe it."""
    times = iter([0.0] * 2 + [1.0, 3.0, 2.0])       # warmup discarded
    arms = {"a": lambda: None}

    import bench_guard
    real = bench_guard._once
    bench_guard._once = lambda fn: next(times)
    try:
        out = BG.alternating(arms, reps=3, warmup=2, watch=False)
    finally:
        bench_guard._once = real

    assert out["a"]["median"] == 2.0
    assert out["a"]["min"] == 1.0 and out["a"]["max"] == 3.0
    assert out["a"]["spread"] == pytest.approx(1.0)      # (3-1)/2
    assert out["a"]["samples"] == [1.0, 3.0, 2.0]


def test_alternating_discards_the_warmup():
    """The SM clock idles at 42% of maximum here and ramps under load, so the
    first repetitions measure the clock coming up rather than the work."""
    calls = []
    arms = {"a": lambda: calls.append(1)}
    out = BG.alternating(arms, reps=2, warmup=3, watch=False)
    assert len(calls) == 5                    # 3 warmup + 2 measured
    assert len(out["a"]["samples"]) == 2      # only the measured ones reported


def test_alternating_refuses_a_measurement_taken_under_contention(monkeypatch):
    """The gap `require_quiet_gpu` cannot cover, and it cost a real measurement.

    That check is pre-flight: it answers "was the card quiet a moment ago".  A
    timing that runs for minutes can have another job arrive halfway through and
    nothing notices -- the arms interleave, so contention lands on both, the
    SPREAD stays small, and every ratio drifts toward 1.00x.  On 2026-08-25 that
    produced a clean-looking 0.993x and 1.005x for a bandwidth lever while
    another project's Python held the card the whole time.

    So the window is watched, not just its start.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    seen = iter([[], [(4242, "other.exe")], [(4242, "other.exe")]])
    monkeypatch.setattr(BG, "foreign_compute_pids",
                        lambda: next(seen, [(4242, "other.exe")]))
    with pytest.raises(RuntimeError, match="contended GPU"):
        BG.alternating({"a": lambda: None}, reps=2, warmup=0)


def test_alternating_watches_for_a_PID_not_a_busy_card(monkeypatch):
    """The signal has to survive our OWN load, and the clock does not.

    A during-the-run check keyed on the clock fires on itself: once the
    measurement is going the card is hot because WE are working it, which
    `require_quiet_gpu` says in as many words.  The first version of this watch
    was written that way anyway and refused the very next measurement.

    A foreign PID is different in kind -- our kernels cannot produce one -- and
    what is flagged is arrival, since the unnameable entries WDDM always lists
    would make an absolute check fire forever.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    always_there = [(11060, "[Insufficient Permissions]")]
    monkeypatch.setattr(BG, "foreign_compute_pids", lambda: list(always_there))
    out = BG.alternating({"a": lambda: None}, reps=2, warmup=0)
    assert out["a"]["foreign_processes"] == [], (
        "a process present from the start is the baseline, not an intruder"
    )


def test_alternating_rejects_an_empty_or_degenerate_call():
    with pytest.raises(ValueError, match="no arms"):
        BG.alternating({}, reps=2, watch=False)
    with pytest.raises(ValueError, match="reps must be positive"):
        BG.alternating({"a": lambda: None}, reps=0, watch=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_alternating_waits_for_the_queue():
    """A CUDA arm that only ENQUEUES faster must not read faster.  Without the
    trailing synchronize this is exactly the mistake a launch-overhead
    measurement would make about itself."""
    x = torch.randn(1024, 1024, device="cuda")

    def heavy():
        y = x
        for _ in range(40):
            y = y @ x
    out = BG.alternating({"heavy": heavy, "light": lambda: None},
                         reps=3, warmup=2, watch=False)
    assert out["heavy"]["median"] > out["light"]["median"] * 10
