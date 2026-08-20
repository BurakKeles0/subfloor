"""Perplexity and the protocol guard.

`test_uniform_model_has_perplexity_equal_to_vocab_size` anchors the arithmetic:
a model that predicts nothing must score exactly the vocabulary size.  The rest
of the file is about the harder problem -- making sure a correct number is not
compared against the wrong published family.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

import perplexity as PPL

V = 64


class UniformLM(nn.Module):
    """Predicts nothing: flat logits everywhere."""

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*ids.shape, V), dtype=torch.float32)


class CopyLM(nn.Module):
    """Predicts that the next token repeats the current one."""

    def __init__(self, strength: float = 12.0) -> None:
        super().__init__()
        self.strength = strength

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((*ids.shape, V), dtype=torch.float32)
        return logits.scatter_(2, ids.unsqueeze(-1), self.strength)


class WrappedOutput(nn.Module):
    """Returns an object with `.logits`, the way HF models do."""

    def forward(self, ids: torch.Tensor):
        class Out:
            logits = torch.zeros((*ids.shape, V), dtype=torch.float32)
        return Out()


# --------------------------------------------------------------------------- #
# Arithmetic
# --------------------------------------------------------------------------- #

def test_uniform_model_has_perplexity_equal_to_vocab_size():
    """Flat logits give cross-entropy log(V) per token, so ppl == V exactly."""
    tokens = torch.randint(V, (256,))
    r = PPL.perplexity(UniformLM(), tokens, seqlen=32, convention="exact")
    assert r.perplexity == pytest.approx(V, rel=1e-5)
    assert r.nll == pytest.approx(math.log(V), rel=1e-6)


def test_a_model_that_knows_something_scores_better():
    tokens = torch.full((256,), 7, dtype=torch.long)      # constant stream
    flat = PPL.perplexity(UniformLM(), tokens, seqlen=32)
    good = PPL.perplexity(CopyLM(), tokens, seqlen=32)
    assert good.perplexity < flat.perplexity
    assert good.perplexity < 1.1                          # nearly perfect


def test_windowing_arithmetic():
    tokens = torch.randint(V, (100,))
    r = PPL.perplexity(UniformLM(), tokens, seqlen=32)
    assert r.n_windows == 3                               # 100 // 32, tail dropped
    assert r.n_tokens == 96
    assert r.seqlen == 32


def test_batching_does_not_change_the_answer():
    tokens = torch.randint(V, (512,))
    one = PPL.perplexity(UniformLM(), tokens, seqlen=32, batch_size=1)
    many = PPL.perplexity(UniformLM(), tokens, seqlen=32, batch_size=8)
    # float32 inside CrossEntropyLoss, so the summation order shows at ~1e-7.
    assert one.perplexity == pytest.approx(many.perplexity, rel=1e-6)


def test_max_windows_truncates():
    tokens = torch.randint(V, (512,))
    r = PPL.perplexity(UniformLM(), tokens, seqlen=32, max_windows=4)
    assert r.n_windows == 4


def test_accepts_huggingface_style_output():
    tokens = torch.randint(V, (128,))
    r = PPL.perplexity(WrappedOutput(), tokens, seqlen=32, convention="exact")
    assert r.perplexity == pytest.approx(V, rel=1e-5)


def test_gptq_convention_differs_slightly_and_deliberately():
    """The reference implementations divide by seqlen while predicting
    seqlen-1 tokens.  We match them so numbers stay comparable, and the gap is
    small -- but it is not zero, so the convention is part of the protocol key.
    """
    tokens = torch.randint(V, (2048,))
    seqlen = 128
    gptq = PPL.perplexity(UniformLM(), tokens, seqlen=seqlen, convention="gptq")
    exact = PPL.perplexity(UniformLM(), tokens, seqlen=seqlen, convention="exact")

    # Exactly: ppl_gptq = ppl_exact ** ((seqlen - 1) / seqlen).
    assert gptq.perplexity == pytest.approx(
        exact.perplexity ** ((seqlen - 1) / seqlen), rel=1e-9
    )
    assert gptq.perplexity < exact.perplexity
    assert gptq.key() != exact.key()

    # The gap shrinks with the window and with a realistic perplexity: at
    # seqlen 2048 and ppl ~5 it is under 0.1%, which is why matching the
    # reference convention costs nothing and buys comparability.
    assert math.exp(-math.log(5.0) / 2048) > 0.999


def test_perplexity_validates():
    with pytest.raises(ValueError, match="1-D stream"):
        PPL.perplexity(UniformLM(), torch.randint(V, (4, 8)), seqlen=4)
    with pytest.raises(ValueError, match="at least 2"):
        PPL.perplexity(UniformLM(), torch.randint(V, (16,)), seqlen=1)
    with pytest.raises(ValueError, match="no window"):
        PPL.perplexity(UniformLM(), torch.randint(V, (8,)), seqlen=32)
    with pytest.raises(ValueError, match="unknown convention"):
        PPL.perplexity(UniformLM(), torch.randint(V, (64,)), seqlen=8,
                       convention="nope")


# --------------------------------------------------------------------------- #
# Protocol discipline -- the reason this module exists
# --------------------------------------------------------------------------- #

def _result(ppl: float, **kw) -> PPL.PerplexityResult:
    base = dict(nll=math.log(ppl), n_tokens=1000, n_windows=10, seqlen=2048,
                dataset="wikitext2", model="llama-2-7b")
    base.update(kw)
    return PPL.PerplexityResult(perplexity=ppl, **base)


def test_compare_refuses_across_protocols():
    """The finding from the dry run, enforced: QuIP# 2-bit is 6.66 in its own
    paper and 6.19 in QTIP's.  Subtracting across that boundary measures the
    protocol, not the method."""
    a = _result(5.47, seqlen=2048)
    b = _result(5.12, seqlen=4096)
    with pytest.raises(ValueError, match="refusing to compare"):
        PPL.compare(a, b)

    with pytest.raises(ValueError, match="refusing to compare"):
        PPL.compare(_result(5.47), _result(5.47, dataset="c4"))
    with pytest.raises(ValueError, match="refusing to compare"):
        PPL.compare(_result(5.47), _result(5.47, convention="exact"))


def test_compare_works_within_a_protocol():
    assert PPL.compare(_result(5.47), _result(6.66)) == pytest.approx(1.19)


def test_identify_protocol_from_the_dense_measurement():
    """Measure dense first; whichever baseline it reproduces is the only family
    we may quote alongside our numbers."""
    assert PPL.identify_protocol(_result(5.12)) == "llama-2-7b/dense-5.12"
    assert PPL.identify_protocol(_result(5.47)) == "llama-2-7b/dense-5.47"
    assert PPL.identify_protocol(_result(5.13)) == "llama-2-7b/dense-5.12"


def test_identify_protocol_declines_when_we_match_neither():
    """Landing between the families means our setup reproduces neither, and the
    honest move is to stop borrowing numbers."""
    assert PPL.identify_protocol(_result(5.30)) is None
    assert PPL.identify_protocol(_result(9.99)) is None


def test_published_table_is_internally_consistent():
    """Both families must agree on which method is better, even though they
    disagree on the absolute numbers -- otherwise one of them is transcribed
    wrong."""
    for proto in ("llama-2-7b/dense-5.12", "llama-2-7b/dense-5.47"):
        dense = PPL.published(proto, "dense")
        for name, value in PPL.PUBLISHED[proto]["results"].items():
            assert value > dense, f"{proto}/{name} beats dense, which is suspect"

    a, b = "llama-2-7b/dense-5.12", "llama-2-7b/dense-5.47"
    for lo, hi in [("quip#-4bit", "quip#-3bit"), ("quip#-3bit", "quip#-2bit")]:
        assert PPL.published(a, lo) < PPL.published(a, hi)
        assert PPL.published(b, lo) < PPL.published(b, hi)


def test_published_rejects_unknown_keys():
    with pytest.raises(KeyError, match="unknown protocol"):
        PPL.published("nope", "dense")
    with pytest.raises(KeyError, match="not recorded"):
        PPL.published("llama-2-7b/dense-5.12", "quarot-gptq-2bit")


def test_the_two_families_disagree_by_more_than_the_effect_we_chase():
    """Why any of this matters: the protocol gap on one method is 0.47 ppl,
    larger than the differences Gate B is trying to resolve."""
    gap = (PPL.published("llama-2-7b/dense-5.47", "quip#-2bit")
           - PPL.published("llama-2-7b/dense-5.12", "quip#-2bit"))
    assert gap == pytest.approx(0.47, abs=0.01)

    within = (PPL.published("llama-2-7b/dense-5.12", "quip#-2bit")
              - PPL.published("llama-2-7b/dense-5.12", "qtip-2bit"))
    assert gap > within, "the protocol gap exceeds the method gap it would mask"
