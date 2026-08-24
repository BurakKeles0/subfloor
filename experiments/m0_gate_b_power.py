"""M0 -- how many calibration draws does Gate B need?

The pre-registration (section 7) leaves one blank that has to be filled before
M1 runs: the minimum detectable difference.  `gate_b` already refuses to decide
below five draws, but "five is enough to attempt a decision" and "five is enough
to detect the effect we expect" are different claims, and only the first is in
the code.  If five draws cannot separate the interior from the edges at the
effect size M1 will actually see, M1 produces "undetermined" at every budget and
the GPU time buys nothing.

This answers it by simulation through `m1_gates.gate_b` itself -- the real rule,
with its Bonferroni correction and its `min_seeds` guard, not an idealized
z-test.  A closed-form power calculation would silently drop the selection
correction, which is the part that costs the most power.

Two halves:

  power_curve()   verdicts under a known truth, as a function of the effect size
                  in units of the paired noise.  Protocol-independent: it gives
                  the constant `c` in `MDD = c * sigma`.
  measure_noise() sigma itself, from the pipeline.  Synthetic, so it anchors the
                  order of magnitude and no more -- the real sigma has to come
                  off the first M1 budget.

It also settles a question the two halves raise together: `GateRun` varies
`run_config(seed=...)`, which is the ROTATION seed, while the pre-registration's
draws are CALIBRATION draws.  Those are different noise sources and the pairing
argument only holds for the second.  `measure_noise` reports both.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m1_gates as M                          # noqa: E402
import tiling as Tl                           # noqa: E402
from calibrate import synthetic_problem       # noqa: E402

TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)
DRAW_GRID = (3, 5, 8, 10, 15, 20, 30)
EFFECT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)

#: A plausible level for `rel_output_error`; only sets the scale, every quantity
#: below is a difference or a ratio.
BASELINE = 0.30


# --------------------------------------------------------------------------- #
# The truth we simulate under
# --------------------------------------------------------------------------- #

def u_curve(delta: float, t_opt: int = 8, spread: float = 1.5,
            tiles: tuple = TILES) -> dict:
    """A U in log2(T): both edges at the baseline, a minimum `delta` below it.

    `spread` is how flat the interior is.  It matters more than it looks: a flat
    interior makes the verdict EASIER (several tiles are genuinely better than
    both edges) and the choice of T* HARDER (they are hard to tell apart).  The
    two failure modes trade off along this knob, so both are reported.
    """
    mu = {}
    for t in tiles:
        if t == 1 or t == Tl.MAX_TILE:
            mu[t] = BASELINE
        else:
            z = (math.log2(t) - math.log2(t_opt)) / spread
            mu[t] = BASELINE - delta * math.exp(-z * z)
    return mu


def simulate_records(mu: dict, n_draws: int, sigma: float,
                     generator: torch.Generator, draw_effect: float = 0.0) -> list:
    """One M1 grid's worth of records, drawn from `mu`.

    `draw_effect` is noise shared by every tile within a draw -- a calibration
    sample that happens to be harder.  It should not change anything, because
    `gate_b` compares paired differences; leaving it switchable makes that
    testable rather than assumed.
    """
    tiles = list(mu)
    # float64 throughout: the shared term is meant to cancel EXACTLY in the
    # paired differences, and in float32 a large shared term eats six digits of
    # the difference on its way out.
    kw = dict(generator=generator, dtype=torch.float64)
    shared = torch.randn(n_draws, **kw) * draw_effect
    noise = torch.randn((len(tiles), n_draws), **kw) * sigma
    return [
        {"tile_size": t, "rel_output_error": mu[t] + float(shared[s] + noise[i, s])}
        for i, t in enumerate(tiles)
        for s in range(n_draws)
    ]


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #

def power_at(n_draws: int, effect: float, *, n_trials: int = 400,
             t_opt: int = 8, spread: float = 1.5, sigma: float = 1.0,
             draw_effect: float = 0.0, seed: int = 0) -> dict:
    """Run the real `gate_b` over `n_trials` synthetic M1 grids.

    `effect` is delta/sigma, so the answer transfers to any protocol: whatever
    sigma turns out to be on the real model, the detectable difference is this
    many multiples of it.
    """
    mu = u_curve(effect * sigma, t_opt=t_opt, spread=spread)
    g = torch.Generator().manual_seed(seed)
    counts = {"interior": 0, "edge": 0, "undetermined": 0}
    correct_t_star = 0
    for _ in range(n_trials):
        recs = simulate_records(mu, n_draws, sigma, g, draw_effect=draw_effect)
        v = M.gate_b(recs)
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
        if v.get("t_star") == t_opt:
            correct_t_star += 1
    return {
        "n_draws": n_draws,
        "effect": effect,
        "power": counts["interior"] / n_trials,
        "p_edge": counts["edge"] / n_trials,
        "p_undetermined": counts["undetermined"] / n_trials,
        "p_correct_t_star": correct_t_star / n_trials,
        "n_trials": n_trials,
    }


def power_curve(*, draws=DRAW_GRID, effects=EFFECT_GRID, n_trials: int = 400,
                t_opt: int = 8, spread: float = 1.5, seed: int = 0) -> list:
    return [power_at(n, e, n_trials=n_trials, t_opt=t_opt, spread=spread,
                     seed=seed + 1000 * i + j)
            for i, n in enumerate(draws) for j, e in enumerate(effects)]


def mdd(rows: list, target_power: float = 0.8) -> dict:
    """Smallest effect reaching `target_power`, per draw count.

    Linearly interpolated between grid points, which is honest to the resolution
    of the grid and no more.  `None` means the grid never got there.
    """
    out = {}
    for n in sorted({r["n_draws"] for r in rows}):
        pts = sorted((r["effect"], r["power"]) for r in rows if r["n_draws"] == n)
        found = None
        for (e0, p0), (e1, p1) in zip(pts, pts[1:]):
            if p0 < target_power <= p1:
                found = e0 + (target_power - p0) * (e1 - e0) / (p1 - p0)
                break
        out[n] = found
    return out


# --------------------------------------------------------------------------- #
# Telling two interior tiles apart
# --------------------------------------------------------------------------- #

def selection_power(gap: float, n_draws: int, *, n_candidates: int = 5,
                    alpha: float = 0.05) -> dict:
    """Can the grid distinguish T=4 from T=16?

    A separate question from the verdict, and the one the Gate A dry run left
    open: at anchor 1 those two cells came out too close to call.  Gate B can
    return "interior" with confidence while T* itself is a coin flip, and then
    the headline number -- WHICH granularity is optimal -- is not supported.

    Closed form, because here it is exact.  For a paired difference of two tiles
    with per-observation noise sigma, the difference has sd sigma*sqrt(2), so

        P(argmin picks the better one)  = Phi(gap * sqrt(n) / (sigma*sqrt(2)))
        P(a CI at alpha' excludes zero) = Phi(gap * sqrt(n)/(sigma*sqrt(2)) - z)

    with alpha' Bonferroni-corrected exactly as `gate_b` corrects it.  `gap` is
    in units of sigma.
    """
    from statistics import NormalDist

    nd = NormalDist()
    z = nd.inv_cdf(1 - alpha / n_candidates / 2)
    t = gap * math.sqrt(n_draws) / math.sqrt(2.0)
    return {
        "gap": gap, "n_draws": n_draws,
        "p_argmin_correct": nd.cdf(t),
        "p_ci_separates": nd.cdf(t - z),
        "z_bonferroni": z,
    }


def draws_for_selection(gap: float, target: float = 0.9,
                        n_candidates: int = 5, alpha: float = 0.05,
                        limit: int = 500) -> int | None:
    for n in range(2, limit + 1):
        if selection_power(gap, n, n_candidates=n_candidates,
                           alpha=alpha)["p_argmin_correct"] >= target:
            return n
    return None


# --------------------------------------------------------------------------- #
# sigma, from the pipeline
# --------------------------------------------------------------------------- #

def redraw_activations(n_out: int, n_in: int, n_samples: int, *,
                       weight_seed: int = 0, data_seed: int = 0):
    """One fixed layer, a fresh calibration sample.

    `synthetic_problem` draws weights and activations from one seed, which is
    right for a smoke test and wrong here: a calibration draw must hold the
    layer fixed.  Same generative shape -- correlated input channels with a few
    fat ones -- rebuilt with the two seeds separated.
    """
    from calibrate import LayerProblem

    gw = torch.Generator().manual_seed(weight_seed)
    W = torch.randn((n_out, n_in), generator=gw, dtype=torch.float64)
    W *= torch.exp(torch.randn((n_out, 1), generator=gw, dtype=torch.float64) * 0.5)
    mixing = torch.randn((n_in, n_in), generator=gw, dtype=torch.float64) / n_in ** 0.5

    gx = torch.Generator().manual_seed(10_000 + data_seed)
    X = torch.randn((n_samples, n_in), generator=gx, dtype=torch.float64) @ mixing
    X[:, ::37] *= 8.0
    return LayerProblem(W, X, name=f"synthetic-{n_out}x{n_in}-data{data_seed}")


def measure_noise(*, n_draws: int = 8, n_out: int = 128, n_in: int = 256,
                  n_samples: int = 512, budget: float = 1.5,
                  tiles: tuple = (1, 4, 8, 16, Tl.MAX_TILE),
                  axis: str = "calibration", progress=None) -> dict:
    """Per-observation noise on `rel_output_error`, and how much pairing removes.

    `axis="calibration"` redraws the layer problem -- new weights, new
    activations -- which is the pre-registration's notion of a draw.
    `axis="rotation"` holds the problem fixed and varies only the rotation seed,
    which is what `GateRun` currently varies.  They are not the same number and
    the pairing argument only covers the first.

    Synthetic, so this fixes an order of magnitude and a ratio, not a value.
    The real sigma has to come off M1's first budget.
    """
    if axis not in ("calibration", "rotation"):
        raise ValueError(f"unknown axis {axis!r}")

    by_tile = {t: [] for t in tiles}
    for s in range(n_draws):
        # A calibration draw redraws the DATA, not the weights.  The model is
        # fixed; only which tokens we calibrate on changes.  Redrawing W too
        # would fold layer-to-layer variation into sigma and overstate it.
        problem = redraw_activations(
            n_out, n_in, n_samples, data_seed=s if axis == "calibration" else 0)
        for t in tiles:
            # Pinned to the pre-2026-08-25 pipeline (the three levers off):
            # the sigma this script measures is what Gate B's power rests on.
            r = M.run_config(problem, budget_bits=budget, tile_size=t,
                             rotate_kron=False, search_dtype=None,
                             compensate_block=None,
                             seed=0 if axis == "calibration" else s)
            by_tile[t].append(None if "skipped" in r else r["rel_output_error"])
            if progress:
                progress(s, t)

    live = {t: v for t, v in by_tile.items() if all(x is not None for x in v)}
    means = {t: sum(v) / len(v) for t, v in live.items()}

    def sd(v):
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

    raw = {t: sd(v) for t, v in live.items()}
    # Pairing is the whole reason the gate works with a handful of draws: how
    # much of the raw spread survives a difference decides the effective sigma.
    paired = {}
    ts = sorted(live, key=str)
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            d = [x - y for x, y in zip(live[a], live[b])]
            paired[f"{a}-{b}"] = sd(d)

    sigma_eff = (sum(paired.values()) / len(paired) / math.sqrt(2.0)
                 if paired else float("nan"))
    mean_level = sum(means.values()) / len(means)
    return {
        "axis": axis, "n_draws": n_draws, "budget": budget,
        "shape": [n_out, n_in], "n_samples": n_samples,
        "means": {str(k): v for k, v in means.items()},
        "sd_raw": {str(k): v for k, v in raw.items()},
        "sd_paired_difference": paired,
        "sigma_effective": sigma_eff,
        "mean_level": mean_level,
        "sigma_over_mean": sigma_eff / mean_level,
        "pairing_gain": (sum(raw.values()) / len(raw)) / sigma_eff,
    }


# --------------------------------------------------------------------------- #

def load_or_measure_noise(cache: Path | None, n_draws: int) -> dict:
    """Cached, because these are the only minutes in the file that cost real
    compute and the simulation half gets re-run far more often."""
    if cache is not None and cache.exists():
        out = json.loads(cache.read_text(encoding="utf-8"))
        out["_from_cache"] = str(cache)
        return out
    out = {
        axis: measure_noise(n_draws=n_draws, axis=axis,
                            progress=lambda s, t: print(
                                f"    {axis}: draw {s} tile {t}   ",
                                end="\r", flush=True))
        for axis in ("calibration", "rotation")
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def translate(mdd_sigma: dict, noise: dict) -> dict:
    """Turn `MDD = c * sigma` into an absolute difference, and a relative one.

    The relative form is the transferable one: sigma and the mean error level
    both scale with the layer, their ratio much less so, so "the effect has to
    be N% of the error level" survives the move from this synthetic layer to a
    real one better than any absolute number does.
    """
    cal = noise.get("calibration")
    if not cal:
        return {}
    sigma, level = cal["sigma_effective"], cal["mean_level"]
    return {
        "sigma": sigma, "mean_level": level,
        "sigma_over_mean": sigma / level,
        "by_draws": {
            n: None if c is None else {
                "effect_sigma": c, "effect_absolute": c * sigma,
                "effect_pct_of_level": 100.0 * c * sigma / level,
            }
            for n, c in mdd_sigma.items()
        },
    }


def run(*, n_trials: int = 400, noise: bool = True, n_draws_noise: int = 8,
        noise_cache: Path | None = None, seed: int = 0) -> dict:
    rows = power_curve(n_trials=n_trials, seed=seed)
    out = {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": "how many calibration draws does gate_b need",
            "rule": "m1_gates.gate_b, called directly -- not a re-derivation",
            "t_opt": 8, "spread": 1.5, "n_trials": n_trials,
        },
        "power": rows,
        "mdd_80": {str(k): v for k, v in mdd(rows, 0.8).items()},
        "mdd_50": {str(k): v for k, v in mdd(rows, 0.5).items()},
        "type_i": [
            {"n_draws": r["n_draws"], "false_interior": r["power"]}
            for r in rows if r["effect"] == 0.0
        ],
        "flat_interior": [
            power_at(n, 1.0, n_trials=n_trials, spread=3.0, seed=seed + 77)
            for n in (5, 10, 20)
        ],
        "selection": [
            selection_power(gap, n)
            for gap in (0.25, 0.5, 1.0, 2.0) for n in (5, 10, 20)
        ],
        "draws_for_selection_90pct": {
            str(gap): draws_for_selection(gap) for gap in (0.25, 0.5, 1.0, 2.0)
        },
    }
    if noise:
        out["noise"] = load_or_measure_noise(noise_cache, n_draws_noise)
        out["absolute"] = translate(mdd(rows, 0.8), out["noise"])
    return out


def _verdict(out: dict) -> None:
    print("\n" + "=" * 72)
    print("  power of gate_b, by draws and by effect size (delta / sigma)")
    effects = sorted({r["effect"] for r in out["power"]})
    print("    n \\ d " + "".join(f"{e:>7.2f}" for e in effects))
    for n in sorted({r["n_draws"] for r in out["power"]}):
        row = {r["effect"]: r["power"] for r in out["power"] if r["n_draws"] == n}
        print(f"    {n:>5} " + "".join(f"{row[e]:>7.2f}" for e in effects))

    print("\n  minimum detectable difference, in units of sigma:")
    print(f"    {'draws':>7}  {'80% power':>10}  {'50% power':>10}")
    for n in sorted(int(k) for k in out["mdd_80"]):
        a, b = out["mdd_80"][str(n)], out["mdd_50"][str(n)]
        print(f"    {n:>7}  {('-' if a is None else f'{a:>10.2f}')}"
              f"  {('-' if b is None else f'{b:>10.2f}')}")

    print("\n  false 'interior' with no effect at all (delta = 0):")
    for r in out["type_i"]:
        print(f"    {r['n_draws']:>3} draws -> {r['false_interior']:.3f}")

    print("\n  a FLAT interior (spread 3.0), delta = 1 sigma:")
    print(f"    {'draws':>7}  {'verdict power':>14}  {'T* correct':>11}")
    for r in out["flat_interior"]:
        print(f"    {r['n_draws']:>7}  {r['power']:>14.2f}  "
              f"{r['p_correct_t_star']:>11.2f}")

    print("\n  telling two interior tiles apart (the T=4 vs T=16 question):")
    print(f"    {'gap/sigma':>10}  {'draws':>6}  {'argmin right':>13}"
          f"  {'CI separates':>13}")
    for r in out["selection"]:
        print(f"    {r['gap']:>10.2f}  {r['n_draws']:>6}  "
              f"{r['p_argmin_correct']:>13.3f}  {r['p_ci_separates']:>13.3f}")
    print("    draws needed for a 90% reliable T*:")
    for gap, n in out["draws_for_selection_90pct"].items():
        print(f"      gap {gap} sigma -> {n} draws")

    noise = out.get("noise")
    if noise:
        print("\n  sigma from the pipeline (synthetic -- an order of magnitude):")
        for axis, r in noise.items():
            if axis.startswith("_"):
                continue
            print(f"    {axis:<12} sigma_eff {r['sigma_effective']:.5f}"
                  f"  ({r['sigma_over_mean'] * 100:.2f}% of the mean level,"
                  f" pairing gain {r['pairing_gain']:.1f}x)")
    scaled = out.get("absolute")
    if scaled:
        print("\n  what that makes the minimum detectable difference (80% power):")
        print(f"    {'draws':>7}  {'x sigma':>8}  {'absolute':>10}"
              f"  {'% of error level':>17}")
        for n, v in sorted(scaled["by_draws"].items(), key=lambda kv: int(kv[0])):
            if v is None:
                print(f"    {n:>7}  {'-':>8}  {'-':>10}  {'-':>17}")
            else:
                print(f"    {n:>7}  {v['effect_sigma']:>8.2f}"
                      f"  {v['effect_absolute']:>10.5f}"
                      f"  {v['effect_pct_of_level']:>16.2f}%")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--no-noise", action="store_true",
                    help="skip the pipeline runs (minutes) and only simulate")
    ap.add_argument("--noise-draws", type=int, default=8)
    ap.add_argument("--noise-cache", type=Path,
                    default=Path("results/m0_gate_b_noise.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_gate_b_power.json"))
    args = ap.parse_args(argv)

    out = run(n_trials=args.trials, noise=not args.no_noise,
              n_draws_noise=args.noise_draws, noise_cache=args.noise_cache)
    _verdict(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
