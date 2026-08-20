"""Streamed evaluation.

`test_streamed_matches_the_full_model` is the correctness condition.  Streaming
changes only where the weights sit while the arithmetic runs; if the number
moves, the first real measurement this project makes -- which published protocol
family our dense Llama-2-7B belongs to -- would be built on a quiet bug.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")

import hf_llama as HF        # noqa: E402
import perplexity as PPL     # noqa: E402
import streamed as ST        # noqa: E402

VOCAB, SEQ = 128, 16


@pytest.fixture(scope="module")
def harness():
    return HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=3, num_attention_heads=4,
                         num_key_value_heads=2, dtype=torch.float32, seed=0)


@pytest.fixture(scope="module")
def tokens():
    g = torch.Generator().manual_seed(2)
    return torch.randint(VOCAB, (SEQ * 6,), generator=g)


# --------------------------------------------------------------------------- #
# THE CONDITION
# --------------------------------------------------------------------------- #

def test_streamed_matches_the_full_model(harness, tokens):
    full = PPL.perplexity(HF.HeadOnlyLM(harness.model), tokens, seqlen=SEQ)
    streamed = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ)

    assert streamed.perplexity == pytest.approx(full.perplexity, rel=1e-6)
    assert streamed.nll == pytest.approx(full.nll, rel=1e-6)
    assert streamed.n_windows == full.n_windows == 6
    assert streamed.extra["streamed"] is True


@pytest.mark.parametrize("keep", [False, True])
def test_activation_placement_does_not_change_the_answer(harness, tokens, keep):
    a = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, keep_on_device=keep)
    b = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, keep_on_device=not keep)
    assert a.perplexity == pytest.approx(b.perplexity, rel=1e-9)


@pytest.mark.parametrize("bs", [1, 2, 3])
def test_batching_does_not_change_the_answer(harness, tokens, bs):
    one = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, batch_size=1)
    many = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, batch_size=bs)
    assert one.perplexity == pytest.approx(many.perplexity, rel=1e-5)


# --------------------------------------------------------------------------- #
# Streaming mechanics
# --------------------------------------------------------------------------- #

def test_stream_blocks_reproduces_a_manual_loop(harness, tokens):
    batches = [tokens[: SEQ].view(1, SEQ), tokens[SEQ: 2 * SEQ].view(1, SEQ)]
    hidden, kwargs = HF.capture_block_inputs(harness.model, batches)

    manual = []
    with torch.no_grad():
        for chunk in hidden:
            x = chunk
            for block in harness.blocks:
                out = block(x, **kwargs)
                x = out[0] if isinstance(out, (tuple, list)) else out
            manual.append(x)

    hidden2, _ = HF.capture_block_inputs(harness.model, batches)
    streamed = ST.stream_blocks(harness.blocks, hidden2, block_kwargs=kwargs)

    for a, b in zip(manual, streamed):
        assert torch.allclose(a, b, atol=1e-6)


def test_blocks_end_up_back_on_the_host(harness, tokens):
    """One block resident at a time is the entire point; leaving them on the
    device would defeat it on the first model that does not fit."""
    ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ)
    for block in harness.blocks:
        for p in block.parameters():
            assert p.device.type == "cpu"


def test_progress_is_reported_per_block(harness, tokens):
    seen = []
    ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, progress=seen.append)
    assert seen == list(range(len(harness.blocks)))


def test_streamed_validates(harness, tokens):
    with pytest.raises(ValueError, match="1-D stream"):
        ST.streamed_perplexity(harness.model, tokens.view(-1, SEQ), seqlen=SEQ)
    with pytest.raises(ValueError, match="at least 2"):
        ST.streamed_perplexity(harness.model, tokens, seqlen=1)
    with pytest.raises(ValueError, match="no window"):
        ST.streamed_perplexity(harness.model, tokens[:4], seqlen=SEQ)
    with pytest.raises(ValueError, match="unknown convention"):
        ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, convention="nope")


def test_max_windows_truncates(harness, tokens):
    r = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, max_windows=2)
    assert r.n_windows == 2


def test_convention_carries_into_the_protocol_key(harness, tokens):
    g = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, convention="gptq")
    e = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, convention="exact")
    assert g.perplexity == pytest.approx(
        e.perplexity ** ((SEQ - 1) / SEQ), rel=1e-9
    )
    with pytest.raises(ValueError, match="refusing to compare"):
        PPL.compare(g, e)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_streaming_to_the_gpu_agrees_with_the_host(harness, tokens):
    cpu = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, device="cpu")
    gpu = ST.streamed_perplexity(harness.model, tokens, seqlen=SEQ, device="cuda")
    assert cpu.perplexity == pytest.approx(gpu.perplexity, rel=1e-4)
