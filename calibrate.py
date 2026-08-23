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
    """

    def __init__(self, n_in: int, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float64) -> None:
        self.n_in = n_in
        self.H = torch.zeros((n_in, n_in), dtype=dtype, device=device)
        self.n_tokens = 0

    def update(self, x: Tensor) -> None:
        """`x` is [..., n_in]; every leading dimension counts as tokens."""
        flat = x.reshape(-1, self.n_in).to(self.H.dtype)
        if flat.shape[1] != self.n_in:
            raise ValueError(
                f"expected last dim {self.n_in}, got {flat.shape[1]}"
            )
        self.H += flat.T @ flat
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
) -> dict[str, HessianAccumulator]:
    """Run `block` over `inputs` and accumulate each linear's input Hessian.

    `inputs` is a sequence of batches so the caller controls peak memory; each
    entry is fed through the block once.
    """
    linears = find_linears(block)
    if names is not None:
        keep = set(names)
        linears = {k: v for k, v in linears.items() if k in keep}
    if not linears:
        raise ValueError("block contains no nn.Linear modules")

    accs = {
        name: HessianAccumulator(mod.in_features, device="cpu", dtype=dtype)
        for name, mod in linears.items()
    }

    handles = []
    for name, mod in linears.items():
        def hook(_mod, args, _out, _name=name):
            accs[_name].update(args[0].detach().to("cpu"))
        handles.append(mod.register_forward_hook(hook, with_kwargs=False))

    try:
        with torch.no_grad():
            for batch in inputs:
                block(batch, **(block_kwargs or {}))
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
    progress: Callable[[int, str], None] | None = None,
) -> list[dict]:
    """Walk the blocks in order, compressing each before moving on.

    For every block: gather statistics from the CURRENT inputs, compress each
    linear, then re-run the block so the next one sees what the compressed model
    actually produces.  Getting that order wrong is Spec v6 trap 20 -- the
    Hessians would describe a model that no longer exists.

    `compress_fn(block_index, name, problem) -> new_weight` is where the pipeline
    plugs in.  `inputs` is a list of batches and is updated in place, so peak
    memory is one block plus the activations.

    Returns one record per compressed layer.
    """
    if not blocks:
        raise ValueError("no blocks to calibrate")
    if not inputs:
        raise ValueError("no calibration batches")

    records: list[dict] = []
    for i, block in enumerate(blocks):
        if device is not None:
            block.to(device)
            inputs = [b.to(device) for b in inputs]

        accs = collect_block_statistics(
            block, inputs, block_kwargs=block_kwargs, dtype=dtype
        )
        linears = find_linears(block)

        for name, acc in accs.items():
            mod = linears[name]
            W = mod.weight.data
            problem = LayerProblem.from_statistics(
                W.to(dtype).cpu(), acc.H, acc.act_norm,
                name=f"blocks.{i}.{name}", n_tokens=acc.n_tokens,
            )
            if progress:
                progress(i, name)
            new_W = compress_fn(i, name, problem)
            if new_W.shape != W.shape:
                raise ValueError(
                    f"{problem.name}: compress_fn returned {tuple(new_W.shape)}, "
                    f"expected {tuple(W.shape)}"
                )
            mod.weight.data = new_W.to(W.dtype).to(W.device)
            records.append({
                "block": i,
                "name": name,
                "layer": problem.name,
                "n_in": problem.n_in,
                "n_out": problem.n_out,
                "n_tokens": acc.n_tokens,
                "rel_output_error": problem.output_error(new_W.to(dtype).cpu()),
            })

        # Only now: the next block must see the COMPRESSED output.
        with torch.no_grad():
            for j, batch in enumerate(inputs):
                inputs[j] = _block_forward(block, batch, block_kwargs)

        if device is not None:
            block.to("cpu")
            inputs = [b.to("cpu") for b in inputs]

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
