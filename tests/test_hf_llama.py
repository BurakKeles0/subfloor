"""The HuggingFace adapter, against a real (if tiny) Llama.

`test_captured_inputs_reproduce_the_model_exactly` is the one that matters.  The
adapter's whole job is to hand `sequential_calibrate` the same hidden states and
kwargs the model would have used; if driving the blocks by hand reproduces the
model's own logits bit for bit, it has done that job. If it drifts, everything
downstream is calibrated against a model that does not exist.

No checkpoint is downloaded: `tiny_llama` builds a randomly initialized
LlamaForCausalLM. Same class, same forward path, same GQA and rotary code.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

import calibrate as Cal          # noqa: E402
import hf_llama as HF            # noqa: E402

DT = torch.float32
VOCAB, SEQ = 128, 16


@pytest.fixture(scope="module")
def harness():
    return HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, dtype=DT, seed=0)


def _ids(n_batches: int = 2, batch: int = 2, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(VOCAB, (batch, SEQ), generator=g) for _ in range(n_batches)]


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

def test_tiny_llama_is_a_real_llama(harness):
    from transformers import LlamaForCausalLM
    assert isinstance(harness.model, LlamaForCausalLM)
    assert len(harness.blocks) == 2
    # GQA on purpose: 4 query heads over 2 KV heads.
    assert harness.config.num_attention_heads == 4
    assert harness.config.num_key_value_heads == 2


def test_get_blocks_finds_the_layers(harness):
    assert HF.get_blocks(harness.model) == list(harness.model.model.layers)
    linears = Cal.find_linears(harness.blocks[0])
    assert {"self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"} <= set(linears)


def test_get_blocks_rejects_an_unknown_layout():
    with pytest.raises(AttributeError, match="cannot find the block list"):
        HF.get_blocks(torch.nn.Linear(4, 4))


# --------------------------------------------------------------------------- #
# THE TEST
# --------------------------------------------------------------------------- #

def test_captured_inputs_reproduce_the_model_exactly(harness):
    """Drive the blocks by hand from the captured state and land on the model's
    own logits. This is the adapter's correctness condition."""
    batches = _ids(1)
    with torch.no_grad():
        reference = harness.model(batches[0]).logits

    inputs, kwargs = HF.capture_block_inputs(harness.model, batches)
    x = inputs[0]
    with torch.no_grad():
        for block in harness.blocks:
            out = block(x, **kwargs)
            x = out[0] if isinstance(out, (tuple, list)) else out
        logits = HF.forward_head(harness.model, x)

    assert torch.allclose(logits, reference, atol=1e-5), (
        "hand-driven blocks diverged from the model's own forward"
    )


def test_capture_returns_one_state_per_batch(harness):
    batches = _ids(3)
    inputs, _ = HF.capture_block_inputs(harness.model, batches)
    assert len(inputs) == 3
    assert all(t.shape == (2, SEQ, 64) for t in inputs)


def test_capture_keeps_the_mask_and_rotary_but_drops_the_cache(harness):
    """A live Cache would accumulate across calibration batches, so it is
    removed; the mask and rotary embeddings are what the blocks actually need."""
    _, kwargs = HF.capture_block_inputs(harness.model, _ids(1))
    assert "position_embeddings" in kwargs
    cos, sin = kwargs["position_embeddings"]
    assert cos.shape[-2] == SEQ
    assert "past_key_values" not in kwargs and "use_cache" not in kwargs


def test_capture_restores_block_zero(harness):
    """The catcher is temporary; a swapped-in wrapper left behind would poison
    every later forward."""
    original = harness.model.model.layers[0]
    HF.capture_block_inputs(harness.model, _ids(1))
    assert harness.model.model.layers[0] is original
    with torch.no_grad():
        harness.model(_ids(1)[0])          # still runnable


def test_capture_rejects_empty_input(harness):
    with pytest.raises(ValueError, match="no calibration batches"):
        HF.capture_block_inputs(harness.model, [])


# --------------------------------------------------------------------------- #
# Into the calibration loop
# --------------------------------------------------------------------------- #

def test_sequential_calibration_runs_on_a_real_model(harness):
    model = HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, dtype=DT, seed=3)
    inputs, kwargs = HF.capture_block_inputs(model.model, _ids(2))

    seen = []
    records = Cal.sequential_calibrate(
        model.blocks, list(inputs),
        lambda i, n, p: (seen.append((i, n)), p.W)[1],
        block_kwargs=kwargs, dtype=torch.float64,
    )
    # 7 linears per Llama block (q,k,v,o,gate,up,down), two blocks.
    assert len(records) == 14
    assert {n for _, n in seen} == set(Cal.find_linears(model.blocks[0]))
    for r in records:
        assert r["rel_output_error"] == pytest.approx(0.0, abs=1e-9)
        assert r["n_tokens"] == 2 * 2 * SEQ


def test_trap_20_holds_on_a_real_model():
    """Spec v7 section 7 trap 20, now on an actual Llama: block 1's statistics
    must depend on what block 0 was compressed to."""
    def run(scale: float) -> torch.Tensor:
        m = HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, dtype=DT, seed=5)
        inputs, kwargs = HF.capture_block_inputs(m.model, _ids(2))
        seen: dict[str, torch.Tensor] = {}

        def compress(i, name, problem):
            if i == 1 and name == "self_attn.q_proj":
                seen[name] = problem.H.clone()
            return problem.W * (scale if i == 0 else 1.0)

        Cal.sequential_calibrate(m.blocks, list(inputs), compress,
                                 block_kwargs=kwargs, dtype=torch.float64)
        return seen["self_attn.q_proj"]

    assert not torch.allclose(run(1.0), run(0.1), rtol=1e-3), (
        "block 1 saw the same activations regardless of block 0 -- "
        "calibration is reading the dense model"
    )


def test_compression_changes_the_model_output(harness):
    """End to end: a lossy compress_fn has to move the logits."""
    model = HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, dtype=DT, seed=7)
    batches = _ids(1)
    with torch.no_grad():
        before = model.model(batches[0]).logits

    inputs, kwargs = HF.capture_block_inputs(model.model, batches)
    Cal.sequential_calibrate(model.blocks, list(inputs),
                             lambda i, n, p: p.W * 0.5,
                             block_kwargs=kwargs, dtype=torch.float64)
    with torch.no_grad():
        after = model.model(batches[0]).logits

    assert not torch.allclose(before, after, atol=1e-3)


def test_head_only_lm_feeds_perplexity(harness):
    """The adapter that lets eval/perplexity score a compressed model."""
    import perplexity as PPL

    lm = HF.HeadOnlyLM(harness.model)
    tokens = torch.randint(VOCAB, (SEQ * 8,))
    r = PPL.perplexity(lm, tokens, seqlen=SEQ, model_name="tiny-llama",
                       dataset="random")
    assert r.n_windows == 8
    # An untrained model over a uniform stream should sit near the vocabulary.
    assert 1.0 < r.perplexity < 10 * VOCAB


# --------------------------------------------------------------------------- #
# Moving a captured context to a device
# --------------------------------------------------------------------------- #

def test_to_device_recurses_into_the_structures_block_kwargs_actually_use():
    """The bug this exists to prevent: `block_kwargs` holds the rotary
    embeddings as a TUPLE of tensors, so a flat comprehension over the dict
    moves the mask and leaves the rotary halves behind.  The failure surfaces
    several frames deep inside transformers as a device mismatch, which is a
    long way from the line that caused it.
    """
    kwargs = {
        "attention_mask": torch.zeros(2, 2),
        "position_embeddings": (torch.zeros(3), torch.ones(3)),
        "past_key_values": [torch.zeros(1), {"inner": torch.ones(1)}],
        "use_cache": False,
        "position_ids": None,
    }
    moved = HF.to_device(kwargs, "cpu")

    assert torch.is_tensor(moved["attention_mask"])
    assert isinstance(moved["position_embeddings"], tuple)
    assert all(torch.is_tensor(t) for t in moved["position_embeddings"])
    assert isinstance(moved["past_key_values"], list)
    assert torch.is_tensor(moved["past_key_values"][1]["inner"])
    assert moved["use_cache"] is False and moved["position_ids"] is None


def test_to_device_leaves_values_unchanged():
    kwargs = {"a": torch.arange(4.0), "b": (torch.ones(2),)}
    moved = HF.to_device(kwargs, "cpu")
    assert torch.equal(moved["a"], kwargs["a"])
    assert torch.equal(moved["b"][0], kwargs["b"][0])
