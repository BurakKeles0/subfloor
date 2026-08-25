"""Where, term by term, is the cost model wrong about a real block?

Section 6.16 measured the driver at 339 s per block against the model's 65 and
named a suspect: the lever factors handed to the model were DERIVED from
microbenchmarks rather than measured through a pass.  `m0_lever_audit.py` tested
that suspect; this file does the other half, putting the model's per-layer
prediction next to the same layer's measured time so the gap has a PLACE rather
than a size.

AND THE ANSWER WAS THAT THERE IS NO GAP HERE.  On a quiet card, one layer at a
time, the model reads 1.01x / 1.01x / 1.09x and 1.04x over a whole block.  The
arithmetic is priced correctly; the 339 s is context, and section 6.17 has
already measured 1.46x of it in one place (seven Hessians held for a block).

Which reframes the next question rather than answering it.  A term that matches
here and still misses in the driver is evidence FOR context and against
arithmetic -- and that is worth as much as a term that misses here, because it
says where not to look.

WHY PER LAYER AND PER TERM.  A single 5.2x on a block can be one term that is
five times too cheap or five terms that are each a little too cheap, and those
call for opposite work.  The model already carried the breakdown; what was
missing was a measured number to divide it by.

TWO DEFECTS FELL OUT OF WRITING THIS, both making the model PESSIMISTIC, which
is why nothing had ever complained: `rotation_seconds` had no Kronecker path and
billed a dense GEMM the pipeline stopped running on 2026-08-25, and
`model_cost` defaulted `compensate_block` to the exact sweep.  The first version
of this file worked around the first of them by substituting the audit's
measured time.  Both are fixed, so it asks the model directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import m0_cost_model as CM            # noqa: E402


#: (layer suffix) -> (n_out, n_in), the three shapes the audit covers.
SHAPES = {
    "q_proj": (4096, 4096),
    "gate_proj": (11008, 4096),
    "down_proj": (4096, 11008),
}


def model_terms(n_out: int, n_in: int, tile_size, budget: float, rates: dict,
                setup: str = "cuda_f32", *, hessian_block: int = 512,
                compensate_block: int | None = CM.DEFAULT_COMPENSATE_BLOCK,
                rotate_kron: bool = CM.DEFAULT_ROTATE_KRON) -> dict:
    """What the model charges ONE layer, split the way `model_cost` splits it.

    Both levers default to the pipeline's setting, which is the whole point:
    the first version of this file had to SUBSTITUTE the audit's measured
    Kronecker time for the model's dense charge, because `rotation_seconds` had
    no Kronecker path.  That was a workaround for a defect, and the defect is
    fixed -- so the substitution is gone and the model is asked directly.
    """
    c = CM.layer_cost(n_out, n_in, tile_size, budget, hessian_block=hessian_block)
    if c is None:
        return {}
    chol = c["n_tiles"] * CM.cholesky_seconds(c["k"], rates, setup,
                                              block=hessian_block)
    rot_dense = c["n_tiles"] * CM.rotation_seconds(c["k"], rates, setup)
    rot = c["n_tiles"] * CM.rotation_seconds(c["k"], rates, setup,
                                             kron=rotate_kron)
    per_vector = CM.codebook_seconds_per_vector(rates, setup, c["lines_per_tile"])
    cb = n_out * c["k"] * per_vector

    table = {(a, b): (e, bl) for a, b, e, bl
             in (rates["setups"][setup].get("compensate_timings")
                 or CM.COMPENSATE_TIMINGS[setup])}
    entry = table.get((n_out, n_in))
    comp = 0.0 if entry is None else (entry[0] if compensate_block is None
                                      else entry[1])
    return {
        "k": c["k"], "n_tiles": c["n_tiles"],
        "lines_per_tile": c["lines_per_tile"],
        "cholesky": chol, "rotation": rot, "rotation_dense": rot_dense,
        "codebook": cb, "compensate": comp,
        "rotate_kron": rotate_kron,
        "rotate_kron_priced": (not rotate_kron)
        or CM.prices_kron(rates, setup, c["k"]),
        "total": chol + rot + cb + comp,
        # What the same layer would cost with both levers off, for the record.
        "total_levers_off": chol + rot_dense + cb + (
            {(a, b): e for a, b, e, _bl
             in (rates["setups"][setup].get("compensate_timings")
                 or CM.COMPENSATE_TIMINGS[setup])}.get((n_out, n_in), 0.0)),
    }


def localize(audit: dict, rates: dict, *, setup: str = "cuda_f32") -> list[dict]:
    tile = audit["tile_size"]
    tile_size = int(tile) if str(tile).isdigit() else tile
    rows = []
    for layer in audit["layers"]:
        if "error" in layer:
            continue
        short = layer["layer"].rsplit(".", 1)[-1]
        if short not in SHAPES:
            continue
        n_out, n_in = SHAPES[short]
        mt = model_terms(n_out, n_in, tile_size, audit["budget_bits"], rates,
                         setup)
        measured = layer["timed"]["pipeline"]["median"]
        mic = layer["micro"]
        lv = layer["levers"]

        rot_kron = mic["rotation"]["kron_layer_s"]
        as_run = mt["total"]
        rows.append({
            "layer": short, "n_out": n_out, "n_in": n_in,
            "k": mt["k"], "n_tiles": mt["n_tiles"],
            "model_terms": mt,
            "measured_pipeline_s": measured,
            "model_as_modelled_s": mt["total_levers_off"],
            "model_as_run_s": as_run,
            "optimism_as_run": measured / as_run if as_run else float("nan"),
            # Term by term, where a measurement exists for the term alone.
            "rotation_measured_s": rot_kron,
            "rotation_modelled_s": mt["rotation"],
            "rotation_modelled_dense_s": mt["rotation_dense"],
            "compensate_measured_s": mic["compensate"]["blocked_s"],
            "compensate_modelled_s": mt["compensate"],
            # What is left once the two measured terms are removed from both
            # sides: the Cholesky and the codebook, which are what `TILE_TIMINGS`
            # was fitted on -- warm, per section 6.16.
            "rest_measured_s": measured - rot_kron - mic["compensate"]["blocked_s"],
            "rest_modelled_s": mt["cholesky"] + mt["codebook"],
            "levers": {t: {"observed": v["observed_saving_s"],
                           "micro": v["micro_saving_s"],
                           "kept": v["kept"]} for t, v in lv.items()},
        })
    return rows


def report(rows: list[dict]) -> None:
    print()
    print("=" * 78)
    print("WHERE THE MODEL IS OPTIMISTIC -- one layer, term by term")
    print("=" * 78)
    for r in rows:
        m = r["model_terms"]
        print(f"\n{r['layer']}  {r['n_out']}x{r['n_in']}  "
              f"k={r['k']} tiles={r['n_tiles']}")
        print(f"  measured pipeline           {r['measured_pipeline_s']:8.2f}s")
        print(f"  model, both levers off      {r['model_as_modelled_s']:8.2f}s")
        print(f"  model, as the code runs     {r['model_as_run_s']:8.2f}s"
              f"   -> optimistic {r['optimism_as_run']:5.2f}x")
        print(f"    rotation    measured {r['rotation_measured_s']:8.2f}s"
              f"   modelled        {r['rotation_modelled_s']:8.2f}s"
              f"   (dense would be {r['rotation_modelled_dense_s']:.2f}s)")
        print(f"    compensate  measured {r['compensate_measured_s']:8.2f}s"
              f"   modelled        {r['compensate_modelled_s']:8.2f}s")
        rest = (f"   -> {r['rest_measured_s'] / r['rest_modelled_s']:5.2f}x"
                if r["rest_modelled_s"] else "")
        print(f"    rest        measured {r['rest_measured_s']:8.2f}s"
              f"   modelled        {r['rest_modelled_s']:8.2f}s{rest}")
        print(f"      of which  cholesky {m['cholesky']:.2f}s  "
              f"codebook {m['codebook']:.2f}s")
    print()

    # Keyed on the shape, not on the row order: q/k/v/o are one shape and
    # gate/up another, and a positional weight would silently mis-scale a run
    # that asked for the layers in a different order.
    per_block = {(n_out, n_in): n for n_out, n_in, n in CM.LLAMA2_7B}
    tot_m = sum(r["measured_pipeline_s"] * per_block[(r["n_out"], r["n_in"])]
                for r in rows)
    tot_p = sum(r["model_as_run_s"] * per_block[(r["n_out"], r["n_in"])]
                for r in rows)
    if len(rows) == 3:
        print(f"one block (4 + 2 + 1):  measured {tot_m:.0f}s   "
              f"model as run {tot_p:.0f}s   optimistic {tot_m / tot_p:.2f}x")
        print("  -- and section 6.16 measured 339 s per block inside the "
              "driver, which is\n     the same seven layers plus the context "
              "they run in (section 6.17).")
    print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", type=Path,
                    default=Path("results/m0_lever_audit.json"))
    ap.add_argument("--rates", type=Path, default=Path("results/m0_rates.json"))
    ap.add_argument("--setup", default="cuda_f32")
    ap.add_argument("--out", type=Path, default=Path("results/m0_gap.json"))
    args = ap.parse_args(argv)

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    rates = json.loads(args.rates.read_text(encoding="utf-8"))
    rows = localize(audit, rates, setup=args.setup)
    report(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
