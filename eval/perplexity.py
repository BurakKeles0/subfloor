"""Perplexity, and the protocol discipline around it.

Computing perplexity is easy.  The trap is comparing it.  While building the
Gate A dry run (docs/gate_a_dry_run.md) two incompatible protocols turned up in
the literature for the SAME model:

    dense Llama-2-7B = 5.12    Wanda, QTIP
    dense Llama-2-7B = 5.47    QuIP#, SliceGPT

QuIP# 2-bit is 6.66 in its own paper and 6.19 in QTIP's table.  That 0.47 gap is
larger than the effect Gate B is trying to resolve, so a number quoted from the
wrong family does not just add noise -- it can invert a conclusion.

The likely cause is the evaluation window (Llama-2 has
max_position_embeddings=4096, and the pruning codebases take `model.seqlen` from
there, while QuIP# states ctx 2048).  That is a HYPOTHESIS.  This module
therefore keys protocols by their measured dense baseline, which is verified,
rather than by the window length, which is inferred: run the dense model, see
which baseline you land on, and only then quote published numbers from that
family.  `identify_protocol` does exactly that.

Windowing follows the GPTQ/SparseGPT/Wanda convention so the numbers are
comparable with the literature -- including its small quirk of dividing the
summed loss by `seqlen` when only `seqlen - 1` tokens were predicted.  Pass
`convention="exact"` for the textbook definition.  The two differ by exactly
`ppl_gptq = ppl_exact ** ((seqlen - 1) / seqlen)` -- under 0.1% at seqlen 2048
for a realistic perplexity, but growing as the window shrinks, so the
convention is part of the protocol key rather than a rounding detail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import Tensor

__all__ = [
    "PerplexityResult",
    "PUBLISHED",
    "perplexity",
    "identify_protocol",
    "compare",
    "load_eval_tokens",
]


# --------------------------------------------------------------------------- #
# Published reference numbers, grouped by the protocol they belong to
# --------------------------------------------------------------------------- #
# Keyed by the DENSE baseline, which every one of these papers reports and which
# we can reproduce.  Never mix families.  See docs/gate_a_dry_run.md.

PUBLISHED: dict[str, dict] = {
    "llama-2-7b/dense-5.12": {
        "dense": 5.12,
        "sources": ["Wanda arXiv:2306.11695 Table 3", "QTIP arXiv:2406.11235 Table 5"],
        "results": {
            "wanda-50%-unstructured": 6.42,
            "sparsegpt-50%-unstructured": 6.51,
            "magnitude-50%": 14.89,
            "wanda-4:8": 7.97,
            "sparsegpt-4:8": 8.12,
            "sparsegpt-2:4": 10.17,
            "wanda-2:4": 11.02,
            "qtip-4bit": 5.17,
            "quip#-4bit": 5.19,
            "qtip-3bit": 5.28,
            "quip#-3bit": 5.41,
            "qtip-2bit": 5.86,
            "quip#-2bit": 6.19,
        },
    },
    "llama-2-7b/dense-5.47": {
        "dense": 5.47,
        "sources": ["QuIP# arXiv:2402.04396 Table 2", "SliceGPT arXiv:2401.15024 Table 1"],
        "results": {
            "quip#-4bit": 5.56,
            "quip#-3bit": 5.79,
            "quip#-2bit": 6.66,
            "quarot-gptq-4bit": 5.60,
            "quarot-gptq-3bit": 6.09,
            "quarot-gptq-2bit": 22.07,
            "slicegpt-10%": 5.89,
            "slicegpt-20%": 6.64,
            "slicegpt-25%": 7.24,
            "slicegpt-30%": 8.12,
            "sparsegpt-2:4": 8.69,
        },
    },
    "llama-2-13b/dense-4.57": {
        "dense": 4.57,
        "sources": ["Wanda arXiv:2306.11695 Table 3"],
        "results": {
            "wanda-50%-unstructured": 5.56,
            "sparsegpt-50%-unstructured": 5.63,
            "wanda-4:8": 6.55,
            "wanda-2:4": 8.27,
        },
    },
}


@dataclass
class PerplexityResult:
    """A perplexity, plus everything needed to know what it may be compared to."""

    perplexity: float
    nll: float
    n_tokens: int
    n_windows: int
    seqlen: int
    dataset: str
    convention: str = "gptq"
    model: str = "unknown"
    protocol: str | None = None
    extra: dict = field(default_factory=dict)

    def key(self) -> tuple:
        """What must match before two results may be subtracted."""
        return (self.model, self.dataset, self.seqlen, self.convention)


# --------------------------------------------------------------------------- #
# Computation
# --------------------------------------------------------------------------- #

def _logits(model: Callable, batch: Tensor) -> Tensor:
    out = model(batch)
    return getattr(out, "logits", out)


def perplexity(
    model: Callable,
    tokens: Tensor,
    *,
    seqlen: int,
    dataset: str = "unknown",
    model_name: str = "unknown",
    device: torch.device | str | None = None,
    batch_size: int = 1,
    convention: str = "gptq",
    max_windows: int | None = None,
) -> PerplexityResult:
    """Sliding-window perplexity over a flat token stream.

    `tokens` is 1-D; it is cut into non-overlapping windows of `seqlen` and any
    tail shorter than a full window is dropped, which is what every reference
    implementation does.

    convention="gptq"  divide the summed loss by n_windows * seqlen
    convention="exact" divide by the number of tokens actually predicted,
                       n_windows * (seqlen - 1)
    """
    if convention not in ("gptq", "exact"):
        raise ValueError(f"unknown convention {convention!r}")
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be a 1-D stream, got {tuple(tokens.shape)}")
    if seqlen < 2:
        raise ValueError(f"seqlen must be at least 2, got {seqlen}")

    n_windows = tokens.numel() // seqlen
    if n_windows == 0:
        raise ValueError(
            f"stream of {tokens.numel()} tokens holds no window of {seqlen}"
        )
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)

    windows = tokens[: n_windows * seqlen].view(n_windows, seqlen)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    total = torch.zeros((), dtype=torch.float64)
    with torch.no_grad():
        for lo in range(0, n_windows, batch_size):
            batch = windows[lo: lo + batch_size]
            if device is not None:
                batch = batch.to(device)
            logits = _logits(model, batch).float()
            # Predict token t+1 from everything up to t.
            shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
            shift_labels = batch[:, 1:].reshape(-1)
            total += loss_fn(shift_logits, shift_labels).double().cpu()

    denom = n_windows * (seqlen if convention == "gptq" else seqlen - 1)
    nll = float(total) / denom
    return PerplexityResult(
        perplexity=math.exp(nll),
        nll=nll,
        n_tokens=n_windows * seqlen,
        n_windows=n_windows,
        seqlen=seqlen,
        dataset=dataset,
        convention=convention,
        model=model_name,
    )


# --------------------------------------------------------------------------- #
# Protocol discipline
# --------------------------------------------------------------------------- #

def identify_protocol(
    dense_result: PerplexityResult, tol: float = 0.05
) -> str | None:
    """Which published family does our dense measurement belong to?

    Measure the dense model first; whichever baseline it reproduces is the only
    family whose numbers may be quoted alongside ours.  Landing between them
    means neither -- report our own dense number and stop borrowing.
    """
    hits = [
        name for name, block in PUBLISHED.items()
        if abs(dense_result.perplexity - block["dense"]) <= tol
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def compare(a: PerplexityResult, b: PerplexityResult) -> float:
    """b.perplexity - a.perplexity, but only when the protocols match.

    Refusing here is the whole point: a difference between families is not a
    difference between methods.
    """
    if a.key() != b.key():
        raise ValueError(
            "refusing to compare across protocols:\n"
            f"  a: model={a.model} dataset={a.dataset} seqlen={a.seqlen} "
            f"convention={a.convention}\n"
            f"  b: model={b.model} dataset={b.dataset} seqlen={b.seqlen} "
            f"convention={b.convention}\n"
            "See docs/gate_a_dry_run.md -- the same method differs by 0.47 ppl "
            "between the two published families for Llama-2-7B."
        )
    return b.perplexity - a.perplexity


def published(protocol: str, name: str) -> float:
    """One published number, from an explicitly named protocol."""
    if protocol not in PUBLISHED:
        raise KeyError(f"unknown protocol {protocol!r}; have {sorted(PUBLISHED)}")
    block = PUBLISHED[protocol]
    if name == "dense":
        return block["dense"]
    if name not in block["results"]:
        raise KeyError(
            f"{name!r} is not recorded for {protocol!r}; "
            f"have {sorted(block['results'])}"
        )
    return block["results"][name]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_eval_tokens(tokenizer, dataset: str = "wikitext2", split: str = "test") -> Tensor:
    """The whole evaluation set as one flat token stream.

    WikiText-2 is joined with newlines and encoded in one pass, matching the
    reference implementations; C4 uses its validation shard.
    """
    from datasets import load_dataset

    if dataset == "wikitext2":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        text = "\n\n".join(raw["text"])
    elif dataset == "c4":
        raw = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        text = " ".join(raw[:1100]["text"])
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return tokenizer(text, return_tensors="pt").input_ids.reshape(-1)
