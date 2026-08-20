"""M0 -- dense perplexity, and which published protocol family we are in.

Spec v7 section 6 makes this the first real measurement: two incompatible
families circulate for Llama-2-7B (dense 5.12 -- Wanda, QTIP; dense 5.47 --
QuIP#, SliceGPT) and the same method differs by 0.47 ppl between them, which is
larger than the effect Gate B has to resolve.  Until we know which one our setup
reproduces, no published number may be quoted alongside ours.

It also tests a claim of ours.  `eval/perplexity.py` records the hypothesis that
the split is the evaluation window -- 5.12 at seqlen 4096 (Llama-2's
max_position_embeddings, which the pruning codebases take as `model.seqlen`) and
5.47 at 2048 (what QuIP# states).  Measuring both windows either confirms that
or shows it was wrong; both outcomes are worth having, and the hypothesis is
recorded in the module so it cannot be quietly retrofitted.

Memory: the model stays on CPU and one block at a time visits the GPU, so this
runs on 8 GiB of VRAM.  Start with --max-windows to check the wiring and get a
time estimate before committing to the full pass.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import hf_llama as HF                     # noqa: E402
import perplexity as PPL                  # noqa: E402
import streamed as ST                     # noqa: E402
from calibrate import load_calibration_tokens  # noqa: F401,E402

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"


def run(
    model_name: str = DEFAULT_MODEL,
    seqlens: tuple[int, ...] = (2048, 4096),
    *,
    device: str | None = None,
    dataset: str = "wikitext2",
    max_windows: int | None = None,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float16,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {model_name} (cpu, {dtype}) ...", flush=True)
    t0 = time.time()
    harness = HF.load_llama(model_name, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.0f}s, "
          f"{len(harness.blocks)} blocks, "
          f"{sum(p.numel() for p in harness.model.parameters()) / 1e9:.2f}B params",
          flush=True)

    print(f"tokenizing {dataset} test split ...", flush=True)
    tokens = PPL.load_eval_tokens(harness.tokenizer, dataset=dataset, split="test")
    print(f"  {tokens.numel():,} tokens", flush=True)

    results = {}
    for seqlen in seqlens:
        n = tokens.numel() // seqlen
        if max_windows:
            n = min(n, max_windows)
        print(f"\nseqlen {seqlen}: {n} windows on {device}", flush=True)
        t0 = time.time()
        r = ST.streamed_perplexity(
            harness.model, tokens, seqlen=seqlen, device=device,
            batch_size=batch_size, dataset=dataset, model_name=model_name,
            max_windows=max_windows,
            progress=lambda i: print(f"    block {i}", end="\r", flush=True),
        )
        elapsed = time.time() - t0
        # A truncated run measures a slice of the test set, not the test set.
        # Perplexity varies enough between sections that a partial number
        # cannot identify a protocol family -- so it is not allowed to.
        proto = None if max_windows is not None else PPL.identify_protocol(r)
        print(f"  ppl = {r.perplexity:.4f}   ({elapsed:.0f}s, {r.n_windows} windows)",
              flush=True)
        print(f"  protocol: "
              f"{'(not claimed: truncated run)' if max_windows is not None else (proto or 'NEITHER FAMILY')}",
              flush=True)
        results[str(seqlen)] = {
            "perplexity": r.perplexity,
            "nll": r.nll,
            "n_windows": r.n_windows,
            "n_tokens": r.n_tokens,
            "convention": r.convention,
            "protocol": proto,
            "seconds": elapsed,
        }

    return {
        "meta": {
            "utc": datetime.now(timezone.utc).isoformat(),
            "model": model_name,
            "dataset": dataset,
            "device": device,
            "dtype": str(dtype),
            "max_windows": max_windows,
            "torch": torch.__version__,
        },
        "results": results,
        "hypothesis": {
            "claim": "the 5.12 / 5.47 split is the evaluation window",
            "predicts": {"4096": 5.12, "2048": 5.47},
        },
    }


def _verdict(out: dict) -> None:
    """Say plainly what the numbers mean for the hypothesis and for quoting."""
    res = out["results"]
    pred = out["hypothesis"]["predicts"]
    print("\n" + "=" * 62)

    if out["meta"].get("max_windows") is not None:
        n = out["meta"]["max_windows"]
        print(f"  TRUNCATED RUN ({n} windows) -- no claim is made.")
        print("  The wiring and the timing are real; the perplexity is not.")
        print("  Measured on this machine: 4 windows gave 6.19 and 8 gave 5.03,")
        print("  which is how much a slice of the test set can move the number.")
        for seqlen, r in res.items():
            print(f"    seqlen {seqlen}: {r['perplexity']:.4f} "
                  f"({r['n_windows']} windows, {r['seconds']:.0f}s)")
        print("=" * 62)
        return

    for seqlen, want in pred.items():
        if seqlen not in res:
            continue
        got = res[seqlen]["perplexity"]
        mark = "matches" if abs(got - want) <= 0.05 else "DOES NOT match"
        print(f"  seqlen {seqlen}: measured {got:.4f}, hypothesis said {want}"
              f"  -> {mark}")

    protos = {s: r["protocol"] for s, r in res.items()}
    print()
    if all(p is None for p in protos.values()):
        print("  Our setup reproduces NEITHER published family.")
        print("  Consequence: quote no published number; report our own dense")
        print("  baseline and compare only against runs we produce ourselves.")
    else:
        for s, p in protos.items():
            if p:
                print(f"  seqlen {s} -> {p}: this family may be quoted alongside")
                print(f"    our numbers AT THAT SEQLEN ONLY.")
    print("=" * 62)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seqlens", type=int, nargs="*", default=[2048, 4096])
    ap.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4"])
    ap.add_argument("--max-windows", type=int, default=None,
                    help="smoke run: check the wiring and time a few windows first")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=Path("results/m0_dense_ppl.json"))
    args = ap.parse_args(argv)

    out = run(
        args.model, tuple(args.seqlens), device=args.device,
        dataset=args.dataset, max_windows=args.max_windows,
        batch_size=args.batch_size,
    )
    _verdict(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
