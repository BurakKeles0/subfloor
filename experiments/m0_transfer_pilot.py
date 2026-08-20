"""M0 -- the transfer pilot: where does the pre-registration's tolerance come from?

The pre-registration predicts, for each tile size,

    Delta(T)_pred = Q(d(T)) + tau(T, d(T))

and then checks the prediction against what M1 measures.  Whether it "held" is
decided by a tolerance, and the v6 audit (section B3) found the obvious way of
setting that tolerance to be a trap: derive it from seed variance and it will
almost certainly be exceeded, because the error in the prediction is not noise.
It is BIAS -- `tau` is measured at equal density and without quantization, then
carried into a budget-matched, quantized setting.  A tolerance sized to noise
would lock the pre-registration into the "prediction failed" branch, and `T*`
would be uninterpretable no matter what the data said.

So the tolerance has to be sized to the transfer itself.  This measures it:
build the predictor exactly as the pre-registration defines it, run the real
pipeline next to it, and look at the gap.

Two numbers come out, and the point is the ratio between them:

    bias   |Delta_pred - Delta_measured|, averaged over draws -- the systematic
           part, which more draws will not reduce
    noise  the draw-to-draw spread, which they would

If bias dominates, a noise-sized tolerance is indeed the trap the audit
described, and the pilot's own number is what belongs in section 9.

The pilot is synthetic and layer-level, like every M0 rehearsal here.  It fixes
the ORDER OF MAGNITUDE and the ratio; the tolerance is frozen from the real
model's first budget, using the rule this file establishes.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import accounting as A                          # noqa: E402
import m1_gates as M                            # noqa: E402
import quantize as Qz                           # noqa: E402
import tiling as Tl                             # noqa: E402
from m0_gate_b_power import redraw_activations  # noqa: E402

TILES = (1, 2, 4, 8, 16, 32, Tl.MAX_TILE)
DEFAULT_BUDGET = 1.5

#: Both halves of every comparison are built with the same alignment.  LDLQ
#: forces 8 on the quantized side; if the unquantized side were left at 1 the
#: two would sit at different realized densities and the "equal density" in
#: tau's definition would be false.
ALIGN = Qz.E8P_DIM


def _scheme(tile_size: int | str) -> str:
    return {1: "unstructured", Tl.MAX_TILE: "structured"}.get(tile_size, "tile")


def point(problem, budget: float, tile_size: int | str, *, seed: int = 0,
          vq_bits: float = M.E8P_BITS) -> dict | None:
    """Everything the prediction needs at one tile size, plus the truth.

    Four runs of the pipeline, all on the same layer and the same draw:

        Delta_measured   T at d(T), quantized      -- what M1 will report
        Q                T=1 at d(T), quantized    -- the degradation curve,
                                                     read at THIS tile's density
        tau              [T at d(T)] - [T=1 at d(T)], both unquantized

    `tau` is the granularity tax as the pre-registration defines it: an
    equal-density difference, taken without the quantizer.  `Q` carries the
    quantizer because it is the whole degradation of the unstructured config,
    not a difference.  The asymmetry is the assumption under test, not a slip.
    """
    n_idx = problem.n_in
    d = A.density_for_budget(_scheme(tile_size), budget, None, n_idx,
                             tile_size=tile_size, vq_bits=vq_bits)
    if d is None or not 0.0 < d <= 1.0:
        return None

    def run(t, quantize):
        # T=1 is asked for at THIS tile's density, so it is a plain
        # unstructured run at d -- not the budget-matched T=1 cell.
        return M.run_config(problem, budget_bits=budget, tile_size=t,
                            quantize=quantize, ldlq=quantize, align=ALIGN,
                            seed=seed, vq_bits=vq_bits)

    # Both sides of every difference must sit at the same density.  Rather than
    # trust that, ask the accounting for the budget that puts T=1 at d, and
    # check the realized densities agree afterwards.
    budget_t1 = A.bits_per_position("unstructured", d, None, n_idx,
                                    vq_bits=vq_bits)

    def run_t1(quantize):
        return M.run_config(problem, budget_bits=budget_t1, tile_size=1,
                            quantize=quantize, ldlq=quantize, align=ALIGN,
                            seed=seed, vq_bits=vq_bits)

    tq, tn = run(tile_size, True), run(tile_size, False)
    oq, on = run_t1(True), run_t1(False)
    if any("skipped" in r for r in (tq, tn, oq, on)):
        return None

    densities = {r["density_realized"] for r in (tq, tn, oq, on)}
    if len(densities) != 1:
        raise AssertionError(
            f"equal-density premise broken at T={tile_size}: {sorted(densities)}")

    tau_noquant = tn["rel_output_error"] - on["rel_output_error"]
    tau_quant = tq["rel_output_error"] - oq["rel_output_error"]
    q_term = oq["rel_output_error"]
    measured = tq["rel_output_error"]
    predicted = q_term + tau_noquant

    return {
        "tile_size": tile_size,
        "budget": budget,
        "density": d,
        "density_realized": densities.pop(),
        "seed": seed,
        "Q": q_term,
        "tau_noquant": tau_noquant,
        "tau_quant": tau_quant,
        "delta_predicted": predicted,
        "delta_measured": measured,
        "prediction_error": predicted - measured,
        # The transfer itself: how much the granularity tax changes when the
        # quantizer is switched on.  `prediction_error` is exactly minus this,
        # by construction -- reported anyway, because seeing them cancel is the
        # cheapest possible check that the predictor was assembled correctly.
        "transfer_error": tau_quant - tau_noquant,
    }


def sweep(problem, budget: float, tiles=TILES, *, seed: int = 0) -> list:
    out = []
    for t in tiles:
        r = point(problem, budget, t, seed=seed)
        if r is not None:
            out.append(r)
    return out


def pilot(*, budget: float = DEFAULT_BUDGET, n_draws: int = 3,
          n_out: int = 128, n_in: int = 256, n_samples: int = 512,
          tiles=TILES, progress=None) -> dict:
    """The sweep over several calibration draws, so bias and noise separate.

    One draw cannot tell a systematic offset from a lucky sample.  Three can:
    the mean over draws is the bias, the spread is the noise, and the tolerance
    question is which of the two is larger.
    """
    draws = []
    for s in range(n_draws):
        prob = redraw_activations(n_out, n_in, n_samples, data_seed=s)
        if progress:
            progress(s)
        draws.append(sweep(prob, budget, tiles, seed=0))

    per_tile = {}
    for t in {r["tile_size"] for d in draws for r in d}:
        rows = [r for d in draws for r in d if r["tile_size"] == t]
        errs = [r["prediction_error"] for r in rows]
        per_tile[str(t)] = {
            "tile_size": t,
            "n_draws": len(rows),
            "density": rows[0]["density"],
            "delta_measured": statistics.fmean(r["delta_measured"] for r in rows),
            "delta_predicted": statistics.fmean(r["delta_predicted"] for r in rows),
            "tau_noquant": statistics.fmean(r["tau_noquant"] for r in rows),
            "tau_quant": statistics.fmean(r["tau_quant"] for r in rows),
            "bias": statistics.fmean(errs),
            "noise": statistics.stdev(errs) if len(errs) > 1 else 0.0,
        }
    return {"budget": budget, "draws": draws, "per_tile": per_tile}


# --------------------------------------------------------------------------- #
# What the tolerance should be
# --------------------------------------------------------------------------- #

def tolerance(pilot_out: dict, *, headroom: float = 1.5) -> dict:
    """Turn the pilot into the number section 9 needs.

    The rule, stated so it can be applied to the real model unchanged:

        tolerance = headroom * max_T |bias(T)|

    Sized to the largest transfer error the sweep found, with a margin, and
    NOT to the draw-to-draw spread.  `T=1` is excluded: there the prediction is
    an identity (tau is zero and Q is the measurement itself), so its error is
    structurally zero and would only drag the maximum down.

    The margin is a judgement call and is written down as one.  1.5x says the
    real model's transfer may be half again as bad as this synthetic layer's,
    which is a guess; what is not a guess is that sizing to noise instead would
    be too small by the factor this function reports as `bias_over_noise`.
    """
    rows = [v for v in pilot_out["per_tile"].values() if v["tile_size"] != 1]
    if not rows:
        raise ValueError("nothing to size a tolerance from")

    biases = [abs(v["bias"]) for v in rows]
    noises = [v["noise"] for v in rows if v["noise"] > 0]
    worst = max(rows, key=lambda v: abs(v["bias"]))
    level = statistics.fmean(v["delta_measured"] for v in rows)
    noise = statistics.fmean(noises) if noises else float("nan")

    return {
        "rule": "headroom * max_T |bias(T)|, T=1 excluded (identity)",
        "headroom": headroom,
        "max_abs_bias": max(biases),
        "worst_tile": worst["tile_size"],
        "mean_abs_bias": statistics.fmean(biases),
        "mean_noise": noise,
        "bias_over_noise": max(biases) / noise if noise == noise and noise else
        float("inf"),
        "tolerance": headroom * max(biases),
        "tolerance_pct_of_level": 100.0 * headroom * max(biases) / level,
        "error_level": level,
    }


def argmin_agreement(pilot_out: dict) -> dict:
    """Does the transfer error move `T*`?

    This is the question the tolerance is a proxy for, so it is worth asking
    directly.  A bias that were a constant offset would be harmless: it would
    shift every cell equally and leave the argmin alone.  The pilot's smoke run
    showed it is not constant -- it changes SIGN across the grid -- and a
    sign-changing bias can reorder the curve.

    If the predicted and measured optima disagree here, the pre-registration is
    on notice before M1 rather than after: `Delta = Q + tau` would be predicting
    the wrong granularity, which is the "separability invalid" branch of section
    5.1 and is a finding in its own right.
    """
    rows = [v for v in pilot_out["per_tile"].values() if v["tile_size"] != 1]
    if not rows:
        return {"checked": False}
    pred = min(rows, key=lambda v: v["delta_predicted"])["tile_size"]
    meas = min(rows, key=lambda v: v["delta_measured"])["tile_size"]
    signs = {1 if v["bias"] > 0 else -1 if v["bias"] < 0 else 0 for v in rows}
    return {
        "checked": True,
        "t_star_predicted": pred,
        "t_star_measured": meas,
        "agrees": pred == meas,
        "bias_changes_sign": len(signs - {0}) > 1,
    }


def identity_check(pilot_out: dict) -> dict:
    """T=1 must predict itself exactly.

    At T=1 the predictor reduces to `Q(d(1)) + 0`, and `Q(d(1))` is the measured
    value.  Any departure from zero means the predictor was wired to a different
    density or a different run than the truth it is compared against -- a bug
    that would otherwise show up as a plausible-looking small bias.
    """
    row = pilot_out["per_tile"].get("1")
    if row is None:
        return {"checked": False}
    return {
        "checked": True,
        "bias": row["bias"],
        "tau_noquant": row["tau_noquant"],
        "exact": abs(row["bias"]) < 1e-12 and abs(row["tau_noquant"]) < 1e-12,
    }


def run(*, budget: float = DEFAULT_BUDGET, n_draws: int = 3, n_out: int = 128,
        n_in: int = 256, headroom: float = 1.5, cache: Path | None = None,
        progress=None) -> dict:
    """The sweep, then everything derived from it.

    `cache` reuses a previous sweep's raw measurements.  Everything below the
    sweep is a pure function of them, so a new diagnostic or a different
    `headroom` does not deserve another eight minutes of pipeline.
    """
    if cache is not None and cache.exists():
        out = json.loads(cache.read_text(encoding="utf-8"))["pilot"]
        # JSON turns the MAX_TILE sentinel's dict keys into strings but leaves
        # the tile_size values alone; the reverse is what the code reads.
        for v in out["per_tile"].values():
            if v["tile_size"] == Tl.MAX_TILE:
                v["tile_size"] = Tl.MAX_TILE
    else:
        out = pilot(budget=budget, n_draws=n_draws, n_out=n_out, n_in=n_in,
                    progress=progress)
    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": "how large is the Delta = Q + tau transfer error",
            "why": ("the pre-registration's tolerance must be sized to this "
                    "bias, not to seed variance (v6 audit section B3)"),
            "shape": [n_out, n_in], "align": ALIGN,
            "n_draws": max(v["n_draws"] for v in out["per_tile"].values()),
            "from_cache": bool(cache is not None and cache.exists()),
        },
        "pilot": out,
        "identity_check": identity_check(out),
        "argmin_agreement": argmin_agreement(out),
        "tolerance": tolerance(out, headroom=headroom),
    }


def _verdict(out: dict) -> None:
    per = out["pilot"]["per_tile"]
    tol = out["tolerance"]
    ident = out["identity_check"]
    print("\n" + "=" * 78)
    print(f"  budget {out['pilot']['budget']}, {out['meta']['n_draws']} "
          f"calibration draws, layer {out['meta']['shape']}")
    print(f"    {'T':>5} {'d':>8} {'measured':>10} {'predicted':>10}"
          f" {'bias':>10} {'noise':>9} {'tau_nq':>9} {'tau_q':>9}")
    for k in sorted(per, key=lambda k: (per[k]["tile_size"] == Tl.MAX_TILE,
                                        per[k]["tile_size"])):
        v = per[k]
        print(f"    {str(v['tile_size']):>5} {v['density']:>8.4f}"
              f" {v['delta_measured']:>10.5f} {v['delta_predicted']:>10.5f}"
              f" {v['bias']:>+10.5f} {v['noise']:>9.5f}"
              f" {v['tau_noquant']:>9.5f} {v['tau_quant']:>9.5f}")

    print(f"\n  T=1 identity check: "
          f"{'exact' if ident.get('exact') else 'NOT EXACT -- predictor is miswired'}"
          f"  (bias {ident.get('bias', float('nan')):.2e})")

    am = out["argmin_agreement"]
    if am.get("checked"):
        print(f"\n  argmin: predicted T*={am['t_star_predicted']}, "
              f"measured T*={am['t_star_measured']}  -> "
              f"{'agree' if am['agrees'] else 'DISAGREE -- separability moves T*'}")
        if am["bias_changes_sign"]:
            print("    the bias changes sign across the grid, so it is not a "
                  "constant offset that would cancel in the argmin")

    print(f"\n  bias vs noise: worst |bias| = {tol['max_abs_bias']:.5f} at "
          f"T={tol['worst_tile']}, mean draw noise = {tol['mean_noise']:.5f}")
    print(f"    ratio {tol['bias_over_noise']:.1f}x  -- "
          f"{'bias dominates, the audit was right' if tol['bias_over_noise'] > 2 else 'noise is comparable'}")
    print(f"\n  proposed tolerance = {tol['headroom']}x worst bias = "
          f"{tol['tolerance']:.5f}"
          f"  ({tol['tolerance_pct_of_level']:.2f}% of the error level)")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--n-out", type=int, default=128)
    ap.add_argument("--n-in", type=int, default=256)
    ap.add_argument("--headroom", type=float, default=1.5)
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_transfer_pilot.json"))
    ap.add_argument("--reuse", action="store_true",
                    help="recompute the derived numbers from --out's sweep "
                         "instead of running the pipeline again")
    args = ap.parse_args(argv)

    out = run(budget=args.budget, n_draws=args.draws, n_out=args.n_out,
              n_in=args.n_in, headroom=args.headroom,
              cache=args.out if args.reuse else None,
              progress=lambda s: print(f"  draw {s} ...", flush=True))
    _verdict(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
