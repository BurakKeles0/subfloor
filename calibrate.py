"""Sequential layer-wise calibration.

Spec v6 section 4.6 makes sequential calibration mandatory, and section 7 trap 20
says why: statistics must come from the COMPRESSED model, not the dense one.
Once layer L is compressed, layer L+1 sees different activations, so a Hessian
gathered from the dense forward pass describes a model that no longer exists.
`sequential_calibrate` therefore compresses a block and only then runs it to
produce the inputs of the next one.

The other constraint is memory.  For Llama-2-7B a calibration set of 128x2048
tokens through an 11008-wide layer is ~5.8 GiB of activations in fp16, so X is
never materialized.  Only the sufficient statistics are kept:

    H        = sum_t x_t x_t^T      [n_in, n_in]
    act_norm = sqrt(diag(H))        [n_in]

which is all the pipeline needs -- Wanda reads `act_norm`, OBS and LDLQ read `H`,
and the objective itself is a function of H alone:

    ||X (W - W_hat)^T||_F^2 = tr(E H E^T)

`LayerProblem` lives here rather than in the experiment driver because it is the
seam: `from_statistics` for real layers, `synthetic_problem` for smoke tests, and
the same object feeds both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "LayerProblem",
    "synthetic_problem",
    "HessianAccumulator",
    "find_linears",
    "collect_block_statistics",
    "sequential_calibrate",
    "load_calibration_tokens",
]

#: WikiText moved under a namespace; a bare "wikitext" is rejected by current
#: huggingface_hub ("Repository id must be 'namespace/name'").
WIKITEXT_REPO = "Salesforce/wikitext"


# --------------------------------------------------------------------------- #
# The unit of work
# --------------------------------------------------------------------------- #

@dataclass
class LayerProblem:
    """One linear layer plus the calibration statistics it is scored against.

    Build it from raw activations (`LayerProblem(W, X)`) for tests, or from
    accumulated statistics (`LayerProblem.from_statistics`) for real layers,
    where X is far too large to keep.
    """

    W: Tensor                             # [n_out, n_in]
    X: Tensor | None = None               # [n_samples, n_in], optional
    name: str = "layer"
    _H: Tensor | None = None
    _act_norm: Tensor | None = None
    n_tokens: int = 0

    def __post_init__(self) -> None:
        if self.W.ndim != 2:
            raise ValueError(f"W must be 2-D, got {tuple(self.W.shape)}")
        if self.X is not None:
            if self.X.ndim != 2:
                raise ValueError(f"X must be 2-D, got {tuple(self.X.shape)}")
            if self.W.shape[1] != self.X.shape[1]:
                raise ValueError(
                    f"W has {self.W.shape[1]} input channels but X has "
                    f"{self.X.shape[1]}"
                )
        elif self._H is None:
            raise ValueError("give either X or accumulated statistics")

    @classmethod
    def from_statistics(
        cls, W: Tensor, H: Tensor, act_norm: Tensor | None = None,
        name: str = "layer", n_tokens: int = 0,
    ) -> "LayerProblem":
        """For real layers: H and the activation norms, no raw activations."""
        n_in = W.shape[1]
        if H.shape != (n_in, n_in):
            raise ValueError(
                f"H must be ({n_in}, {n_in}) to match W, got {tuple(H.shape)}"
            )
        if act_norm is None:
            act_norm = torch.diagonal(H).clamp_min(0).sqrt()
        return cls(W=W, X=None, name=name, _H=H, _act_norm=act_norm,
                   n_tokens=n_tokens)

    @property
    def n_out(self) -> int:
        return self.W.shape[0]

    @property
    def n_in(self) -> int:
        return self.W.shape[1]

    @property
    def act_norm(self) -> Tensor:
        if self._act_norm is None:
            self._act_norm = self.X.norm(dim=0)
        return self._act_norm

    @property
    def H(self) -> Tensor:
        if self._H is None:
            self._H = self.X.T @ self.X
        return self._H

    def output_error(self, W_hat: Tensor) -> float:
        """Relative layer-output error.

        Computed through H so it works without X:
        ||X E^T||_F^2 = tr(E H E^T), with E = W - W_hat.
        """
        E = self.W - W_hat
        H = self.H
        num = torch.einsum("ij,jk,ik->", E, H, E)
        den = torch.einsum("ij,jk,ik->", self.W, H, self.W)
        return float((num / den).clamp_min(0).sqrt())


def synthetic_problem(
    n_out: int = 128, n_in: int = 256, n_samples: int = 512, seed: int = 0
) -> LayerProblem:
    """Weights and activations shaped like a real layer: correlated input
    channels, a few fat ones, and a heavy-tailed weight distribution."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn((n_out, n_in), generator=g, dtype=torch.float64)
    W *= torch.exp(torch.randn((n_out, 1), generator=g, dtype=torch.float64) * 0.5)
    mixing = torch.randn((n_in, n_in), generator=g, dtype=torch.float64) / n_in ** 0.5
    X = torch.randn((n_samples, n_in), generator=g, dtype=torch.float64) @ mixing
    X[:, ::37] *= 8.0
    return LayerProblem(W, X, name=f"synthetic-{n_out}x{n_in}")


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

class HessianAccumulator:
    """Accumulates H = sum_t x_t x_t^T over a forward hook.

    Kept in float64 (or float32 on device) regardless of the model's dtype: H is
    a sum over hundreds of thousands of tokens and fp16 loses it.

    `compute_dtype` splits the two jobs the dtype was doing: the PRODUCT is
    `O(tokens * n_in^2)` and wants to be fast, the SUM is `O(n_in^2)` per batch
    and could be accurate.

    MEASURED, AND IT BUYS NOTHING HERE.  Adding float32 products into a float64
    H costs 9% more time and lands on 5.08e-06 against float64's answer, where
    a plain float32 accumulator lands on 5.06e-06 -- the same number.  The error
    is in the PRODUCT, not in the sum: each batch's `x^T x` already rounds
    `sqrt(tokens) * eps` worth of it away, and a wider accumulator cannot undo
    what the multiply discarded.  Splitting the batches finer does not help
    either, since the error over the whole pass goes as `sqrt(total tokens)`
    however it is grouped.

    Kept anyway, and documented, for two reasons: a card with real float64
    throughput would make the trade differently, and it is the first thing
    anyone will reach for on seeing 5e-06 -- better to find the measurement here
    than to repeat it.
    """

    def __init__(self, n_in: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float64,
                 compute_dtype: torch.dtype | None = None) -> None:
        self.n_in = n_in
        self.H = torch.zeros((n_in, n_in), dtype=dtype, device=device)
        self.compute_dtype = compute_dtype or dtype
        self.n_tokens = 0

    def update(self, x: Tensor) -> None:
        """`x` is [..., n_in]; every leading dimension counts as tokens.

        `x` is moved to H's device rather than H to x's: the accumulator is the
        thing that must not be reallocated per batch.
        """
        flat = x.reshape(-1, self.n_in).to(device=self.H.device,
                                           dtype=self.compute_dtype)
        if flat.shape[1] != self.n_in:
            raise ValueError(
                f"expected last dim {self.n_in}, got {flat.shape[1]}"
            )
        self.H += (flat.T @ flat).to(self.H.dtype)
        self.n_tokens += flat.shape[0]

    @property
    def act_norm(self) -> Tensor:
        """||X_j||_2 -- the quantity Wanda scores with."""
        return torch.diagonal(self.H).clamp_min(0).sqrt()


def find_linears(block: nn.Module, prefix: str = "") -> dict[str, nn.Linear]:
    """Every nn.Linear inside a block, by dotted name."""
    return {
        f"{prefix}{name}" if prefix else name: mod
        for name, mod in block.named_modules()
        if isinstance(mod, nn.Linear)
    }


def collect_block_statistics(
    block: nn.Module,
    inputs: Sequence[Tensor],
    *,
    block_kwargs: dict | None = None,
    dtype: torch.dtype = torch.float64,
    names: Iterable[str] | None = None,
    device: torch.device | str | None = None,
    compute_dtype: torch.dtype | None = None,
) -> dict[str, HessianAccumulator]:
    """Run `block` over `inputs` and accumulate each linear's input Hessian.

    `inputs` is a sequence of batches so the caller controls peak memory; each
    entry is fed through the block once.

    `device` is where the Hessians live and where `x^T x` therefore runs.  It
    used to be hard-wired to the CPU, which meant copying every activation off
    the card it was already on and doing the largest matmul in the calibration
    on the slower processor.  Measured on one Llama-2-7B block over 16,384
    tokens: 22.4 s that way, 0.89 s accumulating on the GPU in float32 -- 25x,
    and this is the single biggest term in a full-model run.  `None` means the
    block's own device.

    Two costs to weigh when overriding it.  The Hessians are resident for the
    whole pass: seven linears of a 7B block are 0.87 GiB at float32 and 1.73 GiB
    at float64, against activations that are already several GiB. And float64
    matmuls are roughly 1/64 rate on a consumer card, so `dtype=float64` alone
    would trade one slow path for another -- `compute_dtype=torch.float32` is
    what makes the accurate accumulator affordable.
    """
    linears = find_linears(block)
    if names is not None:
        keep = set(names)
        linears = {k: v for k, v in linears.items() if k in keep}
    if not linears:
        raise ValueError("block contains no nn.Linear modules")

    if device is None:
        device = next(block.parameters()).device
    accs = {
        name: HessianAccumulator(mod.in_features, device=device, dtype=dtype,
                                 compute_dtype=compute_dtype)
        for name, mod in linears.items()
    }

    handles = []
    for name, mod in linears.items():
        def hook(_mod, args, _out, _name=name):
            # No `.to(...)` here: the accumulator moves what it needs, so a
            # same-device activation is never copied.
            accs[_name].update(args[0].detach())
        handles.append(mod.register_forward_hook(hook, with_kwargs=False))

    block_device = next(block.parameters()).device
    try:
        with torch.no_grad():
            for batch in inputs:
                # ONE batch on the device at a time.  The whole calibration set
                # is 4.0 GiB at the preregistration's 128 x 4096 on a 4096-wide
                # model, against 31 MiB for a single window -- and it would sit
                # there beside the Hessians, the block and the compression's own
                # chunk, on a card with 6.8 GiB usable.  The transfer it costs
                # was measured at about 220 s per point, which is nothing
                # against hours (`docs/STATUS.md` section 7.2).
                block(batch.to(block_device), **(block_kwargs or {}))
    finally:
        for h in handles:
            h.remove()
    return accs


# --------------------------------------------------------------------------- #
# The sequential loop
# --------------------------------------------------------------------------- #

def _block_forward(block: nn.Module, batch: Tensor, block_kwargs: dict | None) -> Tensor:
    out = block(batch, **(block_kwargs or {}))
    return out[0] if isinstance(out, (tuple, list)) else out


def sequential_calibrate(
    blocks: Sequence[nn.Module],
    inputs: list[Tensor],
    compress_fn: Callable[[int, str, LayerProblem], Tensor],
    *,
    block_kwargs: dict | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
    compute_dtype: torch.dtype | None = None,
    progress: Callable[[int, str], None] | None = None,
    on_block_done: Callable[[int, list[Tensor], list[dict]], None] | None = None,
    block_offset: int = 0,
) -> list[dict]:
    """Walk the blocks in order, compressing each before moving on.

    For every block: gather statistics from the CURRENT inputs, compress each
    linear, then re-run the block so the next one sees what the compressed model
    actually produces.  Getting that order wrong is Spec v6 trap 20 -- the
    Hessians would describe a model that no longer exists.

    `compress_fn(block_index, name, problem) -> new_weight` is where the pipeline
    plugs in.  `inputs` is a list of batches and is updated IN PLACE, so peak
    memory is one block plus the activations -- and so a run that checkpoints
    can snapshot that list to resume from (`docs/STATUS.md` section 8.1).

    `device` moves the whole walk: the block, the batches, `block_kwargs`, and
    the `LayerProblem` handed to `compress_fn`.  All four have to agree, and
    until 2026-08-25 two of them did not.  `block_kwargs` was never moved at
    all, so the rotary embeddings stayed behind and the block died inside
    `apply_rotary_pos_emb`; and the problem's W was pinned to the CPU while its
    Hessian followed the block, which no argument could reconcile -- so this
    seam could not hand the pipeline a GPU problem at all, though `run_config`
    runs on one perfectly well.  Nothing caught either: every test ran on the
    CPU, where `.cpu()` and a CPU `block_kwargs` are both no-ops.  That is the
    blind spot `docs/STATUS.md` section 14.1 records, hit again by the commit
    that moved the accumulator onto the card.

    `dtype` is the problem's dtype AND the accumulator's, and on a GPU that
    choice is expensive: float64 runs at roughly 1/64 rate here, 29.9 s per
    block against 0.9 s in float32 -- slower even than the CPU float64 it
    replaced (19.7 s).  `experiments/m0_cost_model.py` prices the `cuda_f32`
    arm, so a full-model run wants `dtype=torch.float32`; leaving the default
    in place is 36 days of M1 (15.0 -> 51.0).  The default stays float64
    anyway, because it is the reference the float32 arm's 5.06e-06 is measured
    against and nothing has yet been measured THROUGH this driver -- picking
    the cheaper arm is a decision for whoever runs it, not a side effect.

    `on_block_done(index, inputs, records)` fires after a block is compressed
    AND re-run, which is the only moment a run can be resumed from: by then the
    block is final and `inputs` holds what the NEXT one will see.  That is the
    checkpoint unit `docs/STATUS.md` section 8.1 asks for, and the reason it is a
    callback rather than something the caller does around this function is that
    the caller has no way to observe that moment from outside.

    `block_offset` is added to the index in every record and callback, so a run
    resumed at block 17 reports block 17 rather than block 0.  Without it a
    resumed run's records silently renumber and nothing downstream can tell two
    halves of one point apart.

    Returns one record per compressed layer.
    """
    if not blocks:
        raise ValueError("no blocks to calibrate")
    if not inputs:
        raise ValueError("no calibration batches")

    if device is not None:
        # Imported here rather than at module scope, the way
        # `load_calibration_tokens` treats `datasets`: the walk is generic but
        # the structure is the adapter's.  `to_device` recurses because the
        # rotary entry is a TUPLE of tensors, so a flat comprehension over
        # `block_kwargs` moves half of it and the failure surfaces several
        # frames into transformers.
        from hf_llama import to_device
        block_kwargs = to_device(block_kwargs or {}, device)

    records: list[dict] = []
    for i, block in enumerate(blocks):
        # The block's index IN THE MODEL, which is not `i` when a resumed run
        # hands us `blocks[start:]`.  Everything a caller can see is keyed on
        # this: the progress line, the record, `on_block_done`, the layer name,
        # and `compress_fn`.  The last two used to be keyed on `i`, so a run
        # resuming at block 17 compressed `blocks.0.q_proj` -- one record, two
        # different block numbers in it -- and any `compress_fn` branching on
        # "is this block 0" (m1_run's E8P early warning) fired on the wrong
        # block.  Named once so the two cannot drift apart again.
        block_index = i + block_offset
        if device is not None:
            block.to(device)

        accs = collect_block_statistics(
            block, inputs, block_kwargs=block_kwargs, dtype=dtype,
            compute_dtype=compute_dtype,
        )
        linears = find_linears(block)

        # `list(...)`: each accumulator is dropped from `accs` as its layer
        # finishes, so the dict is mutated while it is walked.
        for name in list(accs):
            acc = accs[name]
            mod = linears[name]
            W = mod.weight.data
            # W follows H, wherever the accumulator put it.  `run_config` reads
            # the two inside single expressions -- `prune` scores W against H,
            # `output_error` contracts E with H -- so a problem split across
            # devices is not slow, it is unusable.
            W_ref = W.to(device=acc.H.device, dtype=dtype)
            problem = LayerProblem.from_statistics(
                W_ref, acc.H, acc.act_norm,
                name=f"blocks.{block_index}.{name}", n_tokens=acc.n_tokens,
            )
            if progress:
                progress(block_index, name)
            new_W = compress_fn(block_index, name, problem)
            if new_W.shape != W.shape:
                raise ValueError(
                    f"{problem.name}: compress_fn returned {tuple(new_W.shape)}, "
                    f"expected {tuple(W.shape)}"
                )
            mod.weight.data = new_W.to(W.dtype).to(W.device)
            records.append({
                "block": block_index,
                "name": name,
                "layer": problem.name,
                "n_in": problem.n_in,
                "n_out": problem.n_out,
                "n_tokens": acc.n_tokens,
                "rel_output_error": problem.output_error(
                    new_W.to(device=acc.H.device, dtype=dtype)),
            })

            # RELEASE THIS LAYER'S HESSIAN NOW, and the measurement is why.
            #
            # A block's seven accumulators are 846 MB at Llama-2-7B's widths,
            # and compressing one layer peaks at 5.4 GiB against 6.8 usable on
            # this card.  Holding all seven for the whole block leaves the
            # allocator evicting and re-requesting rather than reusing, and it
            # costs far more than the bytes suggest: measured on a real block,
            # 122.7 s holding against 84.2 s releasing -- 1.46x -- while the
            # peak moved only 5.40 -> 5.02 GiB.  The win is not the peak, it is
            # the room to reuse.
            #
            # Nothing downstream needs it: `problem` is finished with, the
            # record carries the error, and the next layer builds its own.
            del problem, acc, W_ref, new_W
            accs.pop(name)

        # Only now: the next block must see the COMPRESSED output.  Written
        # back through `inputs[j]`, not into a new list: the caller's list is
        # the checkpoint unit, and rebinding would leave it frozen at block 0.
        target = next(block.parameters()).device
        with torch.no_grad():
            for j, batch in enumerate(inputs):
                out = _block_forward(block, batch.to(target), block_kwargs)
                inputs[j] = out.to(batch.device)

        if device is not None:
            block.to("cpu")

        # Only here: the block is final and `inputs` is what block i+1 sees.
        if on_block_done:
            on_block_done(block_index, inputs, records)

    return records


# --------------------------------------------------------------------------- #
# Calibration data
# --------------------------------------------------------------------------- #

def load_calibration_tokens(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    dataset: str = "c4",
) -> Tensor:
    """`n_samples` random windows of `seqlen` tokens -> [n_samples, seqlen].

    Spec v6 section 6: the seed IS the calibration draw, and results are
    reported over at least three of them.  Gate B needs more than that
    (plan section I1).
    """
    from datasets import load_dataset

    g = torch.Generator().manual_seed(seed)

    if dataset == "wikitext2":
        # WikiText rows are single LINES, and a line almost never reaches 2048
        # tokens -- sampling per row finds nothing.  The reference
        # implementations join the split and cut windows out of the stream, the
        # same way `eval.perplexity.load_eval_tokens` does for the test split.
        raw = load_dataset(WIKITEXT_REPO, "wikitext-2-raw-v1", split="train")
        stream = tokenizer("\n\n".join(raw["text"]),
                           return_tensors="pt").input_ids.reshape(-1)
        if stream.numel() <= seqlen:
            raise RuntimeError(
                f"wikitext-2 train has {stream.numel()} tokens, need > {seqlen}"
            )
        starts = torch.randint(stream.numel() - seqlen, (n_samples,), generator=g)
        return torch.stack([stream[s:s + seqlen] for s in starts.tolist()])

    if dataset != "c4":
        raise ValueError(f"unknown dataset {dataset!r}")

    # C4 rows are whole documents, so per-document sampling is right here and is
    # what the reference implementations do.
    raw = load_dataset(
        "allenai/c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    out = []
    tries = 0
    while len(out) < n_samples:
        tries += 1
        if tries > 100 * n_samples:
            raise RuntimeError(
                f"only found {len(out)}/{n_samples} windows of {seqlen} tokens"
            )
        i = int(torch.randint(len(raw), (1,), generator=g))
        enc = tokenizer(raw[i]["text"], return_tensors="pt").input_ids
        if enc.shape[1] <= seqlen:
            continue
        start = int(torch.randint(enc.shape[1] - seqlen, (1,), generator=g))
        out.append(enc[:, start:start + seqlen])
    return torch.cat(out, dim=0)
