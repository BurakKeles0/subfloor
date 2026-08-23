"""Perplexity for a model that does not fit on the GPU.

Llama-2-7B is 13.5 GiB in fp16 and this card has 8.  But the blocks do not all
need to be resident at once: run every window through block 0, then block 1, and
so on, with one block on the device at a time.

    hidden states   n_windows x seqlen x hidden, fp16   ~2.35 GiB for WikiText-2
    one block       202M params, fp16                    ~0.40 GiB
                                                        -------
                                                        ~2.75 GiB

That is the same shape as `calibrate.sequential_calibrate`, which is the point:
the streaming machinery is needed for calibration anyway, and evaluation reuses
it rather than duplicating it.

The correctness condition is exact agreement with a full-model forward -- see
`test_streamed_matches_the_full_model`.  A streamed evaluator that quietly
drifts would corrupt the one number this project has to get right first: which
published protocol family our dense measurement belongs to (spec v7 section 6).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import torch
from torch import Tensor, nn

import hf_llama as HF
from perplexity import PerplexityResult

__all__ = ["streamed_perplexity", "stream_blocks"]


def _block_out(out: Any) -> Tensor:
    return out[0] if isinstance(out, (tuple, list)) else out


def stream_blocks(
    blocks: Sequence[nn.Module],
    hidden: list[Tensor],
    *,
    block_kwargs: dict | None = None,
    device: torch.device | str = "cpu",
    keep_on_device: bool = False,
    progress: Callable[[int], None] | None = None,
) -> list[Tensor]:
    """Push every chunk through every block, one block resident at a time.

    `hidden` is updated in place and returned.  With `keep_on_device` the
    activations stay on the GPU between blocks (faster, needs them to fit);
    otherwise they live on the host and each chunk is copied across per block.
    """
    kwargs_dev = HF.to_device(block_kwargs or {}, device)
    for i, block in enumerate(blocks):
        block.to(device)
        try:
            with torch.no_grad():
                for j, chunk in enumerate(hidden):
                    out = _block_out(block(chunk.to(device), **kwargs_dev))
                    hidden[j] = out if keep_on_device else out.to("cpu")
        finally:
            block.to("cpu")
        if progress:
            progress(i)
    return hidden


def streamed_perplexity(
    model: nn.Module,
    tokens: Tensor,
    *,
    seqlen: int,
    device: torch.device | str = "cpu",
    batch_size: int = 1,
    dataset: str = "unknown",
    model_name: str = "unknown",
    convention: str = "gptq",
    max_windows: int | None = None,
    keep_on_device: bool = False,
    progress: Callable[[int], None] | None = None,
) -> PerplexityResult:
    """Perplexity with one transformer block on the device at a time.

    Numerically identical to `perplexity.perplexity`; the difference is only
    where the weights live while the arithmetic happens.
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

    batches = [
        windows[lo: lo + batch_size] for lo in range(0, n_windows, batch_size)
    ]

    # The embedding, causal mask and rotary come from the model itself.
    hidden, block_kwargs = HF.capture_block_inputs(model, batches)
    blocks = HF.get_blocks(model)
    stream_blocks(
        blocks, hidden, block_kwargs=block_kwargs, device=device,
        keep_on_device=keep_on_device, progress=progress,
    )

    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total = torch.zeros((), dtype=torch.float64)
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None) or getattr(inner, "ln_f", None)
    head = model.lm_head

    if norm is not None:
        norm.to(device)
    head.to(device)
    try:
        with torch.no_grad():
            for chunk, ids in zip(hidden, batches):
                logits = HF.forward_head(model, chunk.to(device)).float()
                shift_logits = logits[:, :-1, :].reshape(-1, logits.shape[-1])
                shift_labels = ids[:, 1:].reshape(-1).to(device)
                total += loss_fn(shift_logits, shift_labels).double().cpu()
    finally:
        if norm is not None:
            norm.to("cpu")
        head.to("cpu")

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
        extra={"streamed": True, "device": str(device)},
    )
