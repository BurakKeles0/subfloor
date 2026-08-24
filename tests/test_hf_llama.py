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


# --------------------------------------------------------------------------- #
# The driver on a device -- the path `experiments/m1_run.py` has to take
#
# Five defects sat here at once and the 599-test suite saw none of them, because
# every one of them is a no-op on the CPU: `.cpu()` on a CPU tensor, a CPU
# `block_kwargs` beside a CPU block, a rebind that happens to be harmless when
# nothing moves.  `docs/STATUS.md` section 14.1 already recorded this blind spot
# from two earlier fixes; it caught the next two as well.  So these tests watch
# WHERE things are, not whether an answer came back.
# --------------------------------------------------------------------------- #

def _model(seed: int):
    return HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, dtype=DT, seed=seed)


def test_sequential_calibrate_advances_the_callers_input_list():
    """`inputs` is documented as updated IN PLACE, and section 8.1's checkpoint
    has to save exactly that list to resume from.  Passing `device` rebound the
    local name instead, so the caller kept block 0's activations for the whole
    run -- indistinguishable from a correct run until someone resumed from it.

    Runs on the CPU: `device="cpu"` moves nothing but still takes the branch.
    """
    def walk(device):
        m = _model(11)
        inputs, kwargs = HF.capture_block_inputs(m.model, _ids(2))
        held = list(inputs)
        start = held[0].clone()
        Cal.sequential_calibrate(m.blocks, held, lambda i, n, p: p.W,
                                 block_kwargs=kwargs, dtype=torch.float64,
                                 device=device)
        return start, held

    start, held = walk("cpu")
    assert not torch.equal(held[0], start), (
        "caller's list still holds block 0's inputs; a checkpoint taken from it "
        "would resume the run at the wrong depth"
    )
    _, reference = walk(None)
    assert torch.allclose(held[0], reference[0], atol=1e-5), (
        "the device branch advanced the list somewhere else than the plain walk"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_sequential_calibrate_moves_block_kwargs_onto_the_device():
    """`block_kwargs` carries the rotary embeddings, and they were never moved.
    The block died inside `apply_rotary_pos_emb` -- on the exact call a
    full-model driver has to make.

    Watches what the block RECEIVES rather than whether the walk finished: a
    version that crashes and a version that quietly ran on the wrong device are
    different bugs and this separates them.
    """
    m = _model(13)
    inputs, kwargs = HF.capture_block_inputs(m.model, _ids(2))

    seen: list[str] = []

    def spy(_mod, _args, kw):
        cos, sin = kw["position_embeddings"]
        seen.append(f"{cos.device.type}/{sin.device.type}")

    handle = m.blocks[0].register_forward_pre_hook(spy, with_kwargs=True)
    try:
        Cal.sequential_calibrate(m.blocks, list(inputs), lambda i, n, p: p.W,
                                 block_kwargs=kwargs, device="cuda",
                                 dtype=torch.float32)
    finally:
        handle.remove()

    assert seen, "block 0 never ran"
    assert set(seen) == {"cuda/cuda"}, f"rotary reached the block on {set(seen)}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_sequential_calibrate_hands_compress_fn_a_single_device_problem():
    """W was pinned to the CPU while H followed the block, and no argument could
    reconcile them -- so this seam could not hand the pipeline a GPU problem at
    all.  `run_config` reads W and H inside single expressions, so a split
    problem is not slow, it is unusable.
    """
    m = _model(17)
    inputs, kwargs = HF.capture_block_inputs(m.model, _ids(2))

    placements = set()

    def compress(i, name, p):
        placements.add((p.W.device.type, p.H.device.type, p.act_norm.device.type,
                        p.W.dtype, p.H.dtype))
        return p.W

    Cal.sequential_calibrate(m.blocks, list(inputs), compress,
                             block_kwargs=kwargs, device="cuda",
                             dtype=torch.float32)
    assert placements == {("cuda", "cuda", "cuda",
                           torch.float32, torch.float32)}, placements


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda")
def test_the_pipeline_runs_through_the_seam_on_cuda():
    """The skeleton of `experiments/m1_run.py`: real Llama blocks -> the driver
    -> `run_config` -> compressed weights back into the model, on the card.

    This is the test that would have caught all five defects at once, and the
    reason none of them was caught is that nothing ever ran this composition --
    the same gap `docs/STATUS.md` section 14.3 blames for two missing cost terms.
    """
    import m1_gates as G

    m = _model(19)
    inputs, kwargs = HF.capture_block_inputs(m.model, _ids(2))

    def compress(i, name, p):
        r = G.run_config(p, budget_bits=1.5, tile_size=4, return_weight=True)
        assert "W_hat" in r, f"{name}: {r.get('skipped')}"
        return r["W_hat"]

    records = Cal.sequential_calibrate(m.blocks, list(inputs), compress,
                                       block_kwargs=kwargs, device="cuda",
                                       dtype=torch.float32)
    assert len(records) == 14
    # Lossy at 1.5 bits, so every layer has to move -- a zero here would mean
    # the fallback weight went back in and nothing was actually compressed.
    assert all(0.0 < r["rel_output_error"] < 1.0 for r in records)


# --------------------------------------------------------------------------- #
# The full-model driver, and the one property that makes its checkpoint worth
# having (`docs/STATUS.md` section 8.1)
# --------------------------------------------------------------------------- #

def _tiny_for_m1(tmp_path, monkeypatch):
    """A tiny Llama the driver can compress end to end, plus its token stream.

    `m1_run` loads a checkpoint by name; here it is handed the tiny model
    instead, which is what lets the resume property be tested at all.  On the
    real 7B one run is hours, so a test that needed it would never be run.
    """
    import m1_run

    def fake_load(name, *, dtype=torch.float16, device_map=None):
        h = HF.tiny_llama(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, dtype=DT, seed=21)
        h.tokenizer = None
        return h

    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(VOCAB, (8, SEQ), generator=g)
    stream = torch.randint(VOCAB, (SEQ * 6,), generator=g)

    monkeypatch.setattr(m1_run.HF, "load_llama", fake_load)
    monkeypatch.setattr(m1_run.Cal, "load_calibration_tokens",
                        lambda tok, **k: tokens)
    monkeypatch.setattr(m1_run.PPL, "load_eval_tokens",
                        lambda tok, dataset="wikitext2", split="test": stream)
    return m1_run


@pytest.mark.parametrize("device", ["cpu"])
def test_a_resumed_run_lands_where_an_uninterrupted_one_does(
        tmp_path, monkeypatch, device):
    """THE checkpoint test, and the reason section 8.1 insists resume is part of
    the driver rather than something added afterwards.

    A resume that restored only the weights and re-ran the dense model would
    calibrate the remaining blocks against activations no version of the model
    ever produced.  It would not fail -- it would return a plausible perplexity
    that is simply not the one the run was computing.  So the property is not
    "resume works", it is "resume is INVISIBLE in the answer".
    """
    m1_run = _tiny_for_m1(tmp_path, monkeypatch)
    spec = m1_run.PointSpec(model="tiny", budget_bits=1.5, tile_size=4, draw=0)
    common = dict(device=device, calib_samples=8, calib_seqlen=SEQ,
                  eval_datasets=("wikitext2",), eval_seqlen=SEQ, progress=lambda *a: None)

    uninterrupted = m1_run.run_point(spec, resume_root=None, **common)

    root = tmp_path / "resume"
    stopped = m1_run.run_point(spec, resume_root=root, stop_after_block=1, **common)
    assert stopped["stopped_after_block"] == 1
    assert (root / spec.slug() / "state.json").exists(), "no checkpoint was left"

    resumed = m1_run.run_point(spec, resume_root=root, **common)

    assert resumed["perplexity"]["wikitext2"] == pytest.approx(
        uninterrupted["perplexity"]["wikitext2"], rel=1e-9), (
        "the resumed run reached a different model than the uninterrupted one"
    )
    assert len(resumed["records"]) == len(uninterrupted["records"])
    assert [r["block"] for r in resumed["records"]] == \
           [r["block"] for r in uninterrupted["records"]]


def test_the_checkpoint_is_cleared_when_the_point_finishes(tmp_path, monkeypatch):
    """A point that completed must not leave state a later run would resume
    from -- that would silently skip the whole compression."""
    m1_run = _tiny_for_m1(tmp_path, monkeypatch)
    spec = m1_run.PointSpec(model="tiny", budget_bits=1.5, tile_size=4, draw=0)
    root = tmp_path / "resume"
    m1_run.run_point(spec, device="cpu", resume_root=root, calib_samples=8,
                     calib_seqlen=SEQ, eval_datasets=("wikitext2",),
                     eval_seqlen=SEQ, progress=lambda *a: None)
    assert not (root / spec.slug()).exists()


def test_the_checkpoint_refuses_to_resume_a_different_configuration(
        tmp_path, monkeypatch):
    """The key is `(model, budget, tile, draw)`.  Resuming one point's blocks
    into another's would mix two configurations into one perplexity, and every
    number downstream would be of a model that was never specified."""
    m1_run = _tiny_for_m1(tmp_path, monkeypatch)
    root = tmp_path / "resume"
    a = m1_run.PointSpec(model="tiny", budget_bits=1.5, tile_size=4, draw=0)
    m1_run.run_point(a, device="cpu", resume_root=root, stop_after_block=0,
                     calib_samples=8, calib_seqlen=SEQ,
                     eval_datasets=("wikitext2",), eval_seqlen=SEQ,
                     progress=lambda *a: None)

    b = m1_run.PointSpec(model="tiny", budget_bits=1.5, tile_size=4, draw=1)
    ckpt = m1_run.Checkpoint(root, b)
    ckpt.dir = root / a.slug()             # point b at a's directory
    with pytest.raises(ValueError, match="refusing to resume"):
        ckpt.load()


def test_the_driver_reports_the_early_warning_rule(tmp_path, monkeypatch):
    """Section 3.2's assumption is this project's largest single risk and the
    driver can check it for free -- it already has both numbers."""
    m1_run = _tiny_for_m1(tmp_path, monkeypatch)
    spec = m1_run.PointSpec(model="tiny", budget_bits=1.5, tile_size=4, draw=0)
    out = m1_run.run_point(spec, device="cpu", resume_root=None, calib_samples=8,
                           calib_seqlen=SEQ, eval_datasets=("wikitext2",),
                           eval_seqlen=SEQ, progress=lambda *a: None)
    assert out["diagnostics"], "no early-warning diagnostics were produced"
    for d in out["diagnostics"]:
        assert d["block"] == 0
        assert d["ratio_to_dense"] > 0
        assert isinstance(d["assumption_broken"], bool)
    # And the levers that produced the numbers travel with them.
    assert out["levers"]["rotate_kron"] is True
