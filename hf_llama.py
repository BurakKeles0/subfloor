"""HuggingFace adapter: a real Llama into `calibrate.sequential_calibrate`.

`sequential_calibrate` takes any `list[nn.Module]` and a list of input batches.
Everything here exists to produce those two things from a HF causal LM, and to
put the result back together for evaluation.

The inputs to block 0 are CAPTURED, not reconstructed.  Rebuilding the causal
mask and rotary embeddings by hand means duplicating whatever the installed
transformers does, and drifting silently when it changes.  Instead block 0 is
temporarily wrapped, the model's own forward runs, and the wrapper records the
exact `(hidden_states, kwargs)` the model would have passed -- then aborts
before any real work happens.  This is what the reference pruning codebases do
and it is version-agnostic by construction.

Testable without a download: `tiny_llama()` builds a randomly initialized
LlamaForCausalLM from a small config.  Same class, same forward path, same
rotary and GQA code -- just small.  A 7B checkpoint proves nothing here that a
2-layer model does not.

Memory: the model stays on CPU and one block at a time moves to the GPU.  For
Llama-2-7B a 128 x 2048 calibration set is ~2.1 GiB of fp16 activations, so an
8 GiB card can hold the activations plus one block, but not the whole model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "LlamaHarness",
    "load_llama",
    "tiny_llama",
    "get_blocks",
    "capture_block_inputs",
    "forward_head",
    "HeadOnlyLM",
]


class _StopForward(Exception):
    """Raised by the catcher once block 0's inputs are in hand."""


class _Catcher(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.hidden: list[Tensor] = []
        self.kwargs: dict[str, Any] | None = None

    def forward(self, hidden_states: Tensor, **kwargs: Any) -> Tensor:
        self.hidden.append(hidden_states.detach())
        if self.kwargs is None:
            # Keep everything except the cache: calibration is a pure forward,
            # and a live Cache would accumulate across batches.
            self.kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ("past_key_values", "past_key_value", "use_cache")
            }
        raise _StopForward


def get_blocks(model: nn.Module) -> list[nn.Module]:
    """The transformer blocks, in order.

    Covers the usual layouts: `model.model.layers` (Llama, Mistral, Qwen),
    `model.transformer.h` (GPT-2/NeoX style), and a bare `model.layers`.
    """
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        else:
            return list(obj)
    raise AttributeError(
        f"cannot find the block list on {type(model).__name__}; "
        "pass the blocks explicitly"
    )


def capture_block_inputs(
    model: nn.Module,
    token_batches: Sequence[Tensor],
    *,
    device: torch.device | str = "cpu",
) -> tuple[list[Tensor], dict[str, Any]]:
    """Run the model far enough to see what block 0 receives.

    Returns `(hidden_states_per_batch, block_kwargs)` -- exactly the two things
    `sequential_calibrate` needs.  `block_kwargs` holds the causal mask, the
    rotary embeddings and the position ids, produced by the model itself.
    """
    if not token_batches:
        raise ValueError("no calibration batches")

    blocks = get_blocks(model)
    if not blocks:
        raise ValueError("model has no transformer blocks")

    catcher = _Catcher(blocks[0])
    parent, index = _locate_blocks(model)
    parent[index] = catcher
    try:
        with torch.no_grad():
            for batch in token_batches:
                try:
                    model(batch.to(device))
                except _StopForward:
                    pass
    finally:
        parent[index] = catcher.inner

    if len(catcher.hidden) != len(token_batches):
        raise RuntimeError(
            f"captured {len(catcher.hidden)} of {len(token_batches)} batches; "
            "the model did not reach block 0 for every batch"
        )
    return catcher.hidden, catcher.kwargs or {}


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Move every tensor inside a nested structure; leave everything else alone.

    `capture_block_inputs` returns `block_kwargs` holding the causal mask and the
    rotary embeddings, and the rotary entry is a TUPLE of tensors.  A flat
    `{k: v.to(device) ...}` therefore leaves half of it behind and the block
    forward dies on a device mismatch several frames deep in transformers -- so
    this lives next to the function that produces the structure rather than
    being rewritten by each caller.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_device(o, device) for o in obj)
    if isinstance(obj, list):
        return [to_device(o, device) for o in obj]
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


def _locate_blocks(model: nn.Module) -> tuple[Any, int]:
    """The container holding the blocks, so block 0 can be swapped in place."""
    for path in (("model", "layers"), ("transformer", "h"), ("layers",)):
        obj = model
        for name in path:
            obj = getattr(obj, name, None)
            if obj is None:
                break
        else:
            return obj, 0
    raise AttributeError("cannot locate the block container")


def forward_head(model: nn.Module, hidden: Tensor) -> Tensor:
    """Final norm + lm_head, for turning calibrated hidden states into logits."""
    inner = getattr(model, "model", model)
    norm = getattr(inner, "norm", None) or getattr(inner, "ln_f", None)
    if norm is not None:
        hidden = norm(hidden)
    head = getattr(model, "lm_head", None)
    if head is None:
        raise AttributeError("model has no lm_head")
    return head(hidden)


class HeadOnlyLM(nn.Module):
    """Adapts a compressed model to `eval.perplexity`, which wants ids -> logits."""

    def __init__(self, model: nn.Module, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.model = model
        self.device_ = device

    def forward(self, ids: Tensor) -> Tensor:
        with torch.no_grad():
            return self.model(ids.to(self.device_)).logits


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass
class LlamaHarness:
    model: nn.Module
    tokenizer: Any = None
    blocks: list[nn.Module] = field(default_factory=list)

    @property
    def config(self) -> Any:
        return self.model.config

    def prepare(
        self, token_batches: Sequence[Tensor], device: torch.device | str = "cpu"
    ) -> tuple[list[Tensor], dict[str, Any]]:
        return capture_block_inputs(self.model, token_batches, device=device)


def load_llama(
    name: str = "meta-llama/Llama-2-7b-hf",
    *,
    dtype: torch.dtype = torch.float16,
    device_map: str | None = None,
) -> LlamaHarness:
    """Load a checkpoint, on CPU by default.

    `device_map=None` keeps the whole model on CPU on purpose: the calibration
    loop moves one block at a time, which is what makes a 7B model tractable on
    a small card.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=dtype, device_map=device_map, low_cpu_mem_usage=True
    )
    model.eval()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    return LlamaHarness(model=model, tokenizer=tokenizer, blocks=get_blocks(model))


def tiny_llama(
    *,
    vocab_size: int = 128,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 4,
    num_key_value_heads: int = 2,
    max_position_embeddings: int = 128,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> LlamaHarness:
    """A randomly initialized Llama, small enough to run anywhere.

    `num_key_value_heads < num_attention_heads` on purpose: that exercises GQA,
    which is what makes T=max coordination non-trivial (spec v7 section 4.2).
    """
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        use_cache=False,
    )
    model = LlamaForCausalLM(cfg).to(dtype)
    model.eval()
    return LlamaHarness(model=model, tokenizer=None, blocks=get_blocks(model))
