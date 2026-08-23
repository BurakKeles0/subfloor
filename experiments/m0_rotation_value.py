"""M0 -- is the rotation worth what it costs, on a REAL layer?

Everything we know about the rotation's benefit was measured on synthetic
fixtures: the pipeline numbers (-29.5% at T=4, -31.0% at T=16, plan section I3)
and the isolated block numbers behind them (-61.7% on a heavy-tailed block,
+17.5% on a Gaussian one).  Meanwhile the cost is now measured on the real
thing, and it is the largest single number in the project: rotating per tile
gives every tile its own basis, every basis its own sub-Hessian factorization,
and that is what puts M1 at sixty-one days.

Paying a structural cost for a synthetic benefit is not a position to run an
experiment from.  This closes the gap: one real Llama-2-7B layer, real
calibration activations, the same pipeline, rotation on and off.

What it can and cannot settle.  It measures layer output error, not perplexity,
so it speaks to the mechanism rather than to the headline.  That is enough for
the decision in front of us -- whether the rotation earns its place at all --
and not enough to quote anywhere else.

Cheap by construction: only block 0 is touched, only the named linear is
calibrated, and the sub-Hessians are streamed.  `self_attn.o_proj` at T=16 is
about three minutes per arm on this machine.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calibrate as Cal               # noqa: E402
import hf_llama as HF                 # noqa: E402
import m1_gates as M                  # noqa: E402
import tiling as Tl                   # noqa: E402

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"
DEFAULT_LAYER = "self_attn.o_proj"
DEFAULT_TILES = (4, 16, Tl.MAX_TILE)


def build_problem(model_name: str = DEFAULT_MODEL, *, layer: str = DEFAULT_LAYER,
                  n_seqs: int = 16, seqlen: int = 2048, batch: int = 4,
                  dataset: str = "wikitext2", rows: int | None = None,
                  device: str | None = None,
                  progress=print) -> M.LayerProblem:
    """One real layer, with the Hessian its own calibration data produces.

    Block 0 only.  Its inputs come from the model's own forward pass rather than
    a reconstruction -- the adapter catches them -- so the activations are the
    ones the layer really sees, outliers and all.  That matters here more than
    anywhere else in the project: the claim under test is precisely that
    survivors are heavy-tailed, and a synthetic fixture is where that claim
    would be easiest to accidentally assume.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    progress(f"loading {model_name} (cpu, fp16) ...")
    t0 = time.time()
    harness = HF.load_llama(model_name, dtype=torch.float16)
    progress(f"  {time.time() - t0:.0f}s")

    progress(f"tokenizing {n_seqs} x {seqlen} calibration tokens ...")
    tokens = Cal.load_calibration_tokens(harness.tokenizer, n_samples=n_seqs,
                                         seqlen=seqlen, seed=0, dataset=dataset)
    batches = [tokens[i:i + batch] for i in range(0, tokens.shape[0], batch)]

    progress("capturing block 0 inputs ...")
    hidden, block_kwargs = HF.capture_block_inputs(harness.model, batches)

    block = harness.blocks[0]
    progress(f"accumulating the Hessian for {layer} on {device} ...")
    t0 = time.time()
    block.to(device)
    hidden = [h.to(device) for h in hidden]
    # Nested: the rotary embeddings arrive as a tuple, so this has to recurse.
    block_kwargs = HF.to_device(block_kwargs, device)
    accs = Cal.collect_block_statistics(block, hidden, block_kwargs=block_kwargs,
                                        names=[layer])
    progress(f"  {time.time() - t0:.0f}s, {accs[layer].n_tokens:,} tokens")

    W = dict(Cal.find_linears(block))[layer].weight.detach().to(torch.float64).cpu()
    if rows is not None and rows < W.shape[0]:
        # A slice of the OUTPUT rows.  Every tile keeps its full width, its own
        # column set and the layer's real Hessian, so each tile is exactly the
        # problem it would be in the full layer -- there are simply fewer of
        # them, and cost is linear in the count.  What it does change is the
        # meaning of T=max, which becomes "one tile over `rows` rows" rather
        # than over all of them; the run records `rows` so that is never
        # implicit.
        W = W[:rows].contiguous()
    problem = M.LayerProblem.from_statistics(
        W, accs[layer].H, name=f"{model_name}:layers.0.{layer}",
        n_tokens=accs[layer].n_tokens)

    # The model is 13.5 GB and nothing below needs it.
    del harness, hidden, block, accs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return problem


def compare(problem, *, budget: float = 1.5, tiles=DEFAULT_TILES,
            seed: int = 0, progress=print) -> list:
    """Both arms at each tile size, on the same layer and the same Hessian.

    The only difference between the arms is `rotate_axis`; the mask, the
    density, the compensation and the LDLQ sweep are identical, so the
    difference is attributable.
    """
    rows = []
    for t in tiles:
        arms = {}
        for name, axis in (("rotated", "index"), ("plain", None)):
            t0 = time.time()
            r = M.run_config(problem, budget_bits=budget, tile_size=t,
                             rotate_axis=axis, seed=seed)
            if "skipped" in r:
                break
            arms[name] = r
            progress(f"  T={t} {name}: rel.err {r['rel_output_error']:.5f}  "
                     f"snr {r['snr_db']:.2f} dB  ({time.time() - t0:.0f}s)")
        if len(arms) != 2:
            continue
        rot, pln = arms["rotated"], arms["plain"]
        rows.append({
            "tile_size": t,
            "density": rot["density_realized"],
            "k": rot["survivors_per_tile"],
            "bits_realized": rot["bits_realized"],
            "rotated": rot["rel_output_error"],
            "plain": pln["rel_output_error"],
            "relative_change": (rot["rel_output_error"] - pln["rel_output_error"])
            / pln["rel_output_error"],
            "snr_rotated": rot["snr_db"],
            "snr_plain": pln["snr_db"],
        })
    return rows


#: What the synthetic pipeline measured, for the same budget and tile sizes
#: (plan section I3).  Recorded here so the comparison is against a number
#: written down before this run, not one recalled after it.
SYNTHETIC = {4: -0.295, 8: -0.232, 16: -0.310, 32: -0.270}


def run(model_name: str = DEFAULT_MODEL, *, layer: str = DEFAULT_LAYER,
        budget: float = 1.5, tiles=DEFAULT_TILES, n_seqs: int = 16,
        seqlen: int = 2048, dataset: str = "wikitext2",
        rows: int | None = None, progress=print) -> dict:
    problem = build_problem(model_name, layer=layer, n_seqs=n_seqs,
                            seqlen=seqlen, dataset=dataset, rows=rows,
                            progress=progress)
    rows = compare(problem, budget=budget, tiles=tiles, progress=progress)
    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": "does the rotation help on a real layer, not a fixture",
            "model": model_name, "layer": f"layers.0.{layer}",
            "calibration": dataset,
            "n_out": problem.n_out, "n_in": problem.n_in,
            "output_rows_used": rows,
            "n_tokens": problem.n_tokens,
            "budget": budget,
            "scope": ("layer output error, not perplexity -- speaks to the "
                      "mechanism, not to the headline"),
        },
        "rows": rows,
        "synthetic_reference": {str(k): v for k, v in SYNTHETIC.items()},
    }


def _verdict(out: dict) -> None:
    m, rows = out["meta"], out["rows"]
    print("\n" + "=" * 74)
    sliced = ("" if m.get("output_rows_used") is None
              else f" (first {m['output_rows_used']} output rows)")
    print(f"  {m['layer']}  {m['n_out']}x{m['n_in']}{sliced}  "
          f"{m['n_tokens']:,} calibration tokens  B={m['budget']}")
    print(f"    {'T':>5} {'d':>7} {'k':>6} {'plain':>9} {'rotated':>9}"
          f" {'change':>9} {'synthetic':>10}")
    for r in rows:
        ref = out["synthetic_reference"].get(str(r["tile_size"]))
        ref_s = "-" if ref is None else f"{ref * 100:+.1f}%"
        print(f"    {str(r['tile_size']):>5} {r['density']:7.4f} {r['k']:6d}"
              f" {r['plain']:9.5f} {r['rotated']:9.5f}"
              f" {r['relative_change'] * 100:+8.1f}% {ref_s:>10}")

    helped = [r for r in rows if r["relative_change"] < 0]
    print()
    if not rows:
        print("  no comparable rows -- nothing to conclude")
    elif not helped:
        print("  THE ROTATION DOES NOT HELP HERE.  On this layer it costs error")
        print("  as well as compute, and the synthetic result did not transfer.")
    elif len(helped) == len(rows):
        best = min(rows, key=lambda r: r["relative_change"])
        print(f"  The rotation helps at every tile size, best {best['relative_change'] * 100:+.1f}%"
              f" at T={best['tile_size']}.")
    else:
        print("  Mixed: the rotation helps at some tile sizes and not others,")
        print("  so its value is not a property of the layer alone.")

    if rows:
        real = sum(r["relative_change"] for r in rows) / len(rows)
        refs = [out["synthetic_reference"][str(r["tile_size"])] for r in rows
                if str(r["tile_size"]) in out["synthetic_reference"]]
        if refs:
            print(f"  mean change: real {real * 100:+.1f}% against synthetic "
                  f"{sum(refs) / len(refs) * 100:+.1f}% at the same tile sizes")
    print("=" * 74)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", default=DEFAULT_LAYER)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tiles", nargs="*", default=[str(t) for t in DEFAULT_TILES])
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    ap.add_argument("--rows", type=int, default=None,
                    help="use only this many output rows (cost is linear in it)")
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_rotation_value.json"))
    args = ap.parse_args(argv)

    tiles = [Tl.MAX_TILE if t == Tl.MAX_TILE else int(t) for t in args.tiles]
    out = run(args.model, layer=args.layer, budget=args.budget, tiles=tiles,
              n_seqs=args.seqs, seqlen=args.seqlen, dataset=args.dataset,
              rows=args.rows,
              progress=lambda s: print(s, flush=True))
    _verdict(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
