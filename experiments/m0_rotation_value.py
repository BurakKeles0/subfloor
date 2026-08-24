"""M0 -- is the rotation worth what it costs, on a REAL layer?

Round one asked whether the rotation earns its place at all, because everything
we knew about its benefit came from synthetic fixtures while its cost was
measured on the real thing.  It does, by more than the fixture said: -70.1% mean
layer output error on `o_proj`, against the fixture's -30.2%.

Round two asks what it takes to make the rotation AFFORDABLE, and it has to
start by correcting the premise `docs/STATUS.md` section 6.3 was written on.
That section says confining the rotation to blocks would rescue a shared Hessian
factorization.  It would not.  `rotation.rotate` already shares ONE rotation
across every tile (`share_across_tiles=True`), so the rotation is not why LDLQ
factorizes per tile -- the per-tile COLUMN SET is, and no rotation width touches
that.  Dropping the rotation entirely would leave the k^3 term exactly where it
is.

What does move it is `quantize.ldlq_quantize(hessian_block=b)`: keep only the
width-b diagonal blocks of the sub-Hessian and the factorization goes from k^3
to k*b^2.  Priced on this machine's measured rates, M1 falls from 120 days to
64, and to 11 with the scale fit sampled as well.  That is an approximation of
the OBJECTIVE, not of the arithmetic, so it has to be paid for in quality, and
this is where the bill is read.

The block width belongs on the rotation as well as on the feedback, because the
couplings the feedback drops are then the ones the rotation never created.  So
there are three families, kept separate on purpose:

    R{b}    rotation confined to width b, feedback unconstrained
            -- saves nothing offline; it is the diagnostic that attributes any
               quality loss to the rotation rather than to the feedback
    H{b}    rotation full, feedback confined
            -- the cheapest possible change to the pipeline as it stands
    RH{b}   both confined
            -- the coherent proposal

The width sweep is deliberately WIDE rather than eight.  The cost curve flattens
long before that: b=512 already collects 56 of the 57 days that b=8 would, so
section 6.3's suggestion would have spent the most quality for the least extra
saving.  A rotation cannot change the norm of the coordinates it spans, only
their direction, so at b=8 the spread of eight-group norms is left exactly as it
was found -- and that spread is what a single E8P scale has to cover.

What it can and cannot settle.  It measures layer output error, not perplexity,
so it speaks to the mechanism rather than to the headline.  That is enough for
the decision in front of us -- whether the grid can be run at all -- and not
enough to quote anywhere else.

Cheap by construction: only block 0 is touched, only the named linear is
calibrated, and the sub-Hessians are streamed.
"""

from __future__ import annotations

import argparse
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
                  device: str | None = None, solve_device: str = "cpu",
                  solve_dtype: torch.dtype = torch.float64,
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
    H = accs[layer].H
    if solve_device != "cpu" or solve_dtype is not torch.float64:
        # The pipeline follows its tensors, so moving these two moves everything.
        # float32 on the GPU is 16-45x faster than float64 on the CPU here and
        # the quantized weights agree to 5e-08 -- float32's own epsilon.  The
        # speedup grows with tile size because the codebook search is a big
        # batched product, which is where a CPU is worst and a GPU best.
        W = W.to(solve_device, solve_dtype)
        H = H.to(solve_device, solve_dtype)
    problem = M.LayerProblem.from_statistics(
        W, H, name=f"{model_name}:layers.0.{layer}",
        n_tokens=accs[layer].n_tokens)

    # The model is 13.5 GB and nothing below needs it.
    del harness, hidden, block, accs
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return problem


#: Widths to sweep.  Chosen off the cost curve rather than off round numbers:
#: the Cholesky saving is essentially complete by 512, so anything narrower buys
#: nothing and risks more.  8 is included only because section 6.3 proposed it,
#: and a suggestion is better refuted with a measurement than with an argument.
DEFAULT_BLOCKS = (2048, 1024, 512, 128, 8)

#: What the synthetic pipeline measured, for the same budget and tile sizes
#: (plan section I3).  Recorded here so the comparison is against a number
#: written down before this run, not one recalled after it.
SYNTHETIC = {4: -0.295, 8: -0.232, 16: -0.310, 32: -0.270}

#: Round one's measured `full` arm, at `--rows 512 --seqs 16` on cuda/float32.
#: The `full` arm below is the same computation -- `block=None` degenerates to
#: `structured_orthogonal` -- so reproducing these is a regression check that
#: comes free with the sweep, and `_verdict` flags it if it drifts.
ROUND_ONE = {"4": 0.09648948907852173, "16": 0.1953037828207016,
             "max": 0.18654577434062958}


@dataclass(frozen=True)
class Arm:
    """One pipeline configuration.  `None` means unconstrained."""
    name: str
    rotate_axis: str | None = "index"
    rotate_block: int | None = None
    hessian_block: int | None = None
    rotate_kron: bool = False


def arms_for(blocks=DEFAULT_BLOCKS, families=("R", "H", "RH")) -> list[Arm]:
    """The two reference arms, then one arm per family per width.

    `plain` and `full` bracket everything: `plain` is where we would be with no
    rotation at all, `full` is round one's -70%.  Every constrained arm is read
    against both, because "does it still beat no rotation" and "how much of the
    -70% survived" are different questions and one reference cannot answer both.
    """
    out = [Arm("plain", rotate_axis=None), Arm("full")]
    if "K" in families:
        # Same rotation, contracted against its Kronecker factors instead of
        # formed densely (`rotation.rotate_hessian`).  Not a width family, so it
        # takes no `b`: paired against `full` and against `H512`, the arm the
        # pipeline actually runs, because the question is whether an arithmetic
        # that is 19x cheaper and measurably MORE accurate in float32 changes
        # the answer on a real layer.
        out.append(Arm("fullK", rotate_kron=True))
        out.append(Arm("H512", hessian_block=512))
        out.append(Arm("H512K", hessian_block=512, rotate_kron=True))
    for b in blocks:
        if "R" in families:
            out.append(Arm(f"R{b}", rotate_block=b))
        if "H" in families:
            out.append(Arm(f"H{b}", hessian_block=b))
        if "RH" in families:
            out.append(Arm(f"RH{b}", rotate_block=b, hessian_block=b))
    return out


def cholesky_ratio(k: int, block: int | None) -> float:
    """Factorization work relative to the unconstrained arm, at this k.

    Reported beside the quality so the trade reads off one row instead of two
    documents.  It is arithmetic on the kernel LDLQ calls, not a wall-clock
    measurement of this script: this script forms the full rotated sub-Hessian
    in every arm, which a production path with a block-diagonal rotation would
    not need to do.
    """
    if block is None or block >= k:
        return 1.0
    return sum(min(block, k - o) ** 3 for o in range(0, k, block)) / k ** 3


def compare(problem, *, budget: float = 1.5, tiles=DEFAULT_TILES,
            arms=None, seed: int = 0, progress=print) -> list:
    """Every arm at each tile size, on the same layer and the same Hessian.

    Nothing varies across arms but `rotate_block` and `hessian_block`: the mask,
    the density, the compensation and the scale policy are identical, so any
    difference is attributable to the constraint.
    """
    arms = list(arms or arms_for())
    rows = []
    for t in tiles:
        measured = {}
        for arm in arms:
            t0 = time.time()
            r = M.run_config(problem, budget_bits=budget, tile_size=t,
                             rotate_axis=arm.rotate_axis,
                             rotate_block=arm.rotate_block,
                             rotate_kron=arm.rotate_kron,
                             hessian_block=arm.hessian_block, seed=seed)
            if "skipped" in r:
                measured = {}
                break
            measured[arm.name] = r
            progress(f"  T={t} {arm.name:>7}: rel.err {r['rel_output_error']:.5f}"
                     f"  snr {r['snr_db']:.2f} dB  ({time.time() - t0:.0f}s)")
        if len(measured) != len(arms):
            continue
        plain = measured["plain"]["rel_output_error"]
        full = measured["full"]["rel_output_error"]
        k = measured["full"]["survivors_per_tile"]
        for arm in arms:
            r = measured[arm.name]
            err = r["rel_output_error"]
            rows.append({
                "tile_size": t,
                "arm": arm.name,
                "rotate_block": arm.rotate_block,
                "rotate_kron": arm.rotate_kron,
                "hessian_block": arm.hessian_block,
                "density": r["density_realized"],
                "k": k,
                "bits_realized": r["bits_realized"],
                "rel_output_error": err,
                "snr_db": r["snr_db"],
                # Two references, because they answer different questions.
                "vs_plain": (err - plain) / plain,
                "vs_full": (err - full) / full,
                "cholesky_ratio": cholesky_ratio(k, arm.hessian_block),
            })
    return rows


def run(model_name: str = DEFAULT_MODEL, *, layer: str = DEFAULT_LAYER,
        budget: float = 1.5, tiles=DEFAULT_TILES, arms=None, n_seqs: int = 16,
        seqlen: int = 2048, dataset: str = "wikitext2",
        rows: int | None = None, solve_device: str = "cpu",
        solve_dtype: torch.dtype = torch.float64, progress=print) -> dict:
    arms = list(arms or arms_for())
    problem = build_problem(model_name, layer=layer, n_seqs=n_seqs,
                            seqlen=seqlen, dataset=dataset, rows=rows,
                            solve_device=solve_device, solve_dtype=solve_dtype,
                            progress=progress)
    measured = compare(problem, budget=budget, tiles=tiles, arms=arms,
                       progress=progress)
    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "question": ("how much of the rotation's -70% survives when the "
                         "rotation and the LDLQ feedback are confined to "
                         "width-b blocks -- the only lever that moves the k^3 "
                         "factorization"),
            "correction": ("STATUS 6.3 attributes the per-tile factorization "
                           "to the rotation; the rotation is already shared "
                           "across tiles, and the per-tile column set is what "
                           "forces it"),
            "model": model_name, "layer": f"layers.0.{layer}",
            "calibration": dataset,
            "n_out": problem.n_out, "n_in": problem.n_in,
            "output_rows_used": rows,
            "solve_device": solve_device, "solve_dtype": str(solve_dtype),
            "n_tokens": problem.n_tokens,
            "budget": budget,
            "arms": [a.name for a in arms],
            "scope": ("layer output error, not perplexity -- speaks to the "
                      "mechanism, not to the headline"),
        },
        "rows": measured,
        "synthetic_reference": {str(k): v for k, v in SYNTHETIC.items()},
        "round_one_reference": ROUND_ONE,
    }


def _verdict(out: dict) -> None:
    m, rows = out["meta"], out["rows"]
    print("\n" + "=" * 78)
    sliced = ("" if m.get("output_rows_used") is None
              else f" (first {m['output_rows_used']} output rows)")
    print(f"  {m['layer']}  {m['n_out']}x{m['n_in']}{sliced}  "
          f"{m['n_tokens']:,} calibration tokens  B={m['budget']}")
    print(f"  solved on {m.get('solve_device', 'cpu')} in "
          f"{m.get('solve_dtype', 'torch.float64').replace('torch.', '')}")

    tiles = list(dict.fromkeys(r["tile_size"] for r in rows))
    for t in tiles:
        here = [r for r in rows if r["tile_size"] == t]
        print(f"\n  T={t}   k={here[0]['k']}   d={here[0]['density']:.4f}")
        print(f"    {'arm':>8} {'rel.err':>9} {'vs plain':>9} {'vs full':>9}"
              f" {'chol':>9}")
        for r in here:
            ratio = r["cholesky_ratio"]
            chol = "1x" if ratio == 1.0 else f"1/{1 / ratio:.0f}"
            print(f"    {r['arm']:>8} {r['rel_output_error']:9.5f}"
                  f" {r['vs_plain'] * 100:+8.1f}% {r['vs_full'] * 100:+8.1f}%"
                  f" {chol:>9}")

    # Round one IS the `full` arm.  If it moved, something unrelated moved too
    # and nothing below is safe to read.
    drift = [(r["tile_size"], abs(r["rel_output_error"] - ref))
             for r in rows if r["arm"] == "full"
             for ref in [out["round_one_reference"].get(str(r["tile_size"]))]
             if ref is not None]
    if drift:
        worst = max(drift, key=lambda d: d[1])
        flag = "" if worst[1] < 1e-6 else "   <-- CHECK, this should be ~0"
        print(f"\n  round-one reproduction (full arm): worst drift "
              f"{worst[1]:.2e} at T={worst[0]}{flag}")

    print("\n" + "-" * 78)
    pairs = [("full", "fullK"), ("H512", "H512K")]
    have = {r["arm"] for r in rows}
    if any(a in have and b in have for a, b in pairs):
        print()
        print("-" * 78)
        print("  K   same rotation, contracted against its Kronecker factors")
        print(f"    {'pair':>14} {'T':>5} {'dense':>9} {'kron':>9} {'kron vs dense':>14}")
        worst = 0.0
        for a, b in pairs:
            for t in tiles:
                da = next((r for r in rows if r["arm"] == a and r["tile_size"] == t), None)
                kb = next((r for r in rows if r["arm"] == b and r["tile_size"] == t), None)
                if da is None or kb is None:
                    continue
                rel = kb["rel_output_error"] / da["rel_output_error"] - 1.0
                worst = max(worst, abs(rel))
                print(f"    {a + '/' + b:>14} {str(t):>5} {da['rel_output_error']:9.5f}"
                      f" {kb['rel_output_error']:9.5f} {rel * 100:+13.4f}%")
        # Gate B separates neighbouring tiles at 0.31 sigma, which is 3.2% of
        # the error level (docs/STATUS.md 5.6 and 5.8).  Far under that is below
        # what M1 can see at all; approaching it would have to be argued for.
        verdict = "below" if worst < 0.032 else "NOT below"
        print()
        print(f"    worst |change| {worst * 100:.4f}%  against the 3.2% Gate B "
              f"can resolve -> {verdict} it")

    for fam, label in (("H", "feedback confined, rotation full"),
                       ("RH", "both confined")):
        cand = [r for r in rows if r["arm"][:len(fam)] == fam
                and r["arm"][len(fam):].isdigit()]
        if not cand:
            continue
        print(f"  {fam:>3}  {label}")
        # Widest width that stays inside 10% of the full rotation at EVERY tile
        # size and still beats no rotation.  Widest, not best: the cost curve is
        # flat below ~512, so a narrower width scoring marginally better is
        # buying nothing.
        widths = sorted({r["hessian_block"] for r in cand}, reverse=True)
        held = None
        for w in widths:
            at_w = [r for r in cand if r["hessian_block"] == w]
            if all(r["vs_full"] <= 0.10 and r["vs_plain"] < 0.0 for r in at_w):
                held = (w, max(r["vs_full"] for r in at_w),
                        max(r["vs_plain"] for r in at_w))
                break
        if held is None:
            best = min(cand, key=lambda r: r["vs_full"])
            print(f"       nothing holds within 10% of the full rotation at "
                  f"every tile size;")
            print(f"       best single cell is {best['arm']} at T="
                  f"{best['tile_size']}, {best['vs_full'] * 100:+.1f}%")
        else:
            w, vf, vp = held
            print(f"       holds at b={w}: worst {vf * 100:+.1f}% against full, "
                  f"{vp * 100:+.1f}% against plain")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--layer", default=DEFAULT_LAYER)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--tiles", nargs="*", default=[str(t) for t in DEFAULT_TILES])
    ap.add_argument("--blocks", nargs="*", type=int, default=list(DEFAULT_BLOCKS),
                    help="rotation / feedback widths to sweep")
    ap.add_argument("--families", nargs="*", default=["R", "H", "RH"],
                    choices=["R", "H", "RH", "K"])
    ap.add_argument("--seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    ap.add_argument("--rows", type=int, default=None,
                    help="use only this many output rows (cost is linear in it)")
    ap.add_argument("--solve-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--solve-dtype", default="float64",
                    choices=["float64", "float32"])
    ap.add_argument("--out", type=Path,
                    default=Path("results/m0_rotation_blocks.json"))
    args = ap.parse_args(argv)

    tiles = [Tl.MAX_TILE if t == Tl.MAX_TILE else int(t) for t in args.tiles]
    out = run(args.model, layer=args.layer, budget=args.budget, tiles=tiles,
              arms=arms_for(tuple(args.blocks), tuple(args.families)),
              n_seqs=args.seqs, seqlen=args.seqlen, dataset=args.dataset,
              rows=args.rows, solve_device=args.solve_device,
              solve_dtype=getattr(torch, args.solve_dtype),
              progress=lambda s: print(s, flush=True))
    _verdict(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
