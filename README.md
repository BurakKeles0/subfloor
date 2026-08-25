# tilesparse

*English · [Türkçe](README.tr.md)*

**Sparsity Below the Quantization Floor** — jointly optimising
`(survivor quantizer, granularity, density)` once the bit budget drops below the
PTQ floor.

> **Status.** The pipeline runs end to end against a real Llama-2-7B: dense
> perplexity reproduces the published figure to within 0.006, the full-model
> driver compresses real weights, and an interrupted run lands where an
> uninterrupted one does. **The compressed model's perplexity has never been
> measured** — every quality number below is either a layer-level proxy or
> synthetic data. Details in [`docs/STATUS.md`](docs/STATUS.md).

---

## The question

Dense post-training quantization has a practical floor around 2 bits. Below it,
things fall apart: on Llama-2-7B, QuaRot-GPTQ at 2 bits gives **22.07** ppl and
QuIP# at 2 bits gives **6.66**, against a dense 5.47. Sparsity has a floor too,
but it is set by the **index format** — a bitmap cannot go below 1 bit per
position, while an index shared across a tile falls to `1/T`.

Why that matters: the bit budget *is* context length. Llama-2-70B on a 24 GiB
card, 22.5 GiB usable:

| bits/position | context that fits |
|---|---|
| 2.0 (the PTQ floor) | ~15.6k |
| 1.5 | ~28.4k |

Going below 2 bits means doubling the context.

## The core identity

With `W` bits per survivor, `d` density and `T` the tile size, in the bitmap
regime:

```
d(T) − d(1) = (1 − 1/T) / W               ← independent of the budget, CONSTANT
[d(T) − d(1)] / [d(∞) − d(1)] = 1 − 1/T   ← independent of W as well
```

The absolute advantage is constant. The ratio grows only because the denominator
shrinks — reading that as "the advantage grows with the budget" is a common and
wrong interpretation.

The lever grows as `W` falls: `0.2256` at GPTQ-4bit, **`0.4688`** at lattice VQ
(`W = 2.0`). Which is why the choice of survivor quantizer decides the budget
regime rather than merely improving it.

## The design invariant

```
score → select mask (in the UNROTATED basis) → freeze
      → compact → rotate → LDLQ → compensate
```

**The mask is always selected in the unrotated basis.** Rotation flattens the
magnitude distribution, and pruning feeds on concentrated energy; pruning in the
rotated basis destroys the model (QuaRot+Wanda at 50% sparsity → **5868 ppl**,
OBR Table 1).

But this is an *ordering* problem, not a prohibition. Once the mask is frozen,
rotation cannot spoil it — the selection has already happened. `prune()`
**raises** if called in the wrong order; it is not left to convention.

---

## Install and run

`transformers >= 5` is required: `hf_llama.load_llama` calls
`from_pretrained(dtype=...)`, which is a v5 keyword. On v4 it fails several
frames deep with an unrelated-looking message.

```bash
pip install "torch" "transformers>=5" datasets numpy pytest
python -m pytest -q                    # 674 tests
```

Synthetic smoke test — the whole pipeline including both gates, no GPU needed:

```bash
python experiments/m1_gates.py --synthetic --n-out 64 --n-in 128 --seeds 3 --budgets 1.5
```

Real model, one grid point (calibrate → compress → evaluate, checkpointed at
block granularity):

```bash
python -u experiments/m1_run.py --budget 1.5 --tile 16 --calib-seqlen 2048
```

Re-measure this machine's constants — **mandatory on a different card**, because
every timing constant here says "measured on this machine" and means it:

```bash
python experiments/m0_cost_model.py
python -u experiments/m0_lever_audit.py --build --rot-sweep
```

To run on a free cloud GPU: [`cloud/README.md`](cloud/README.md).

---

## Layout

| Module | Job |
|---|---|
| `accounting.py` | bit budgets, the `1−1/T` identity, the `B*` wall, the live-band filter |
| `scoring.py` | saliency — two per-weight metrics, two aggregation directions |
| `tiling.py` | tile partition, frozen mask (`T=1` unstructured, `T=max` structured) |
| `prune.py` | mask selection + forward compensation; asserts the design invariant |
| `compact.py` | gather survivors into dense per-tile blocks |
| `rotation.py` | mask-preserving rotation, `kron(RHT(2^a), orthogonal(m))` |
| `quantize.py` | QuIP# E8P codebook + LDLQ (Hessian-aware rounding) |
| `calibrate.py` | sequential calibration; statistics come from the **compressed** model |
| `eval/perplexity.py` | ppl + protocol guard |
| `eval/streamed.py` | layer-streamed ppl for a model that does not fit the GPU |
| `hf_llama.py` | HuggingFace adapter — captures what block 0 receives |
| `experiments/m1_gates.py` | M1's two gates; the pipeline itself (`run_config`) |
| `experiments/m1_run.py` | **full-model driver** — block-granular checkpoint and resume |
| `experiments/m0_cost_model.py` | what a real run costs, from measured curves |
| `experiments/bench_guard.py` | asserts, by **raising**, that the card is idle enough to time on |
| `cloud/` | run one point on a free cloud session; the pipeline gets no additions |

The remaining `experiments/m0_*.py` files are individual measurements: the value
of rotation, the scale fit, the precision levers, the tile timings, the lever
audit. Each docstring carries what it measured, why, and where it was wrong.

**Documents:**

- [`docs/STATUS.md`](docs/STATUS.md) — **start here.** What is verified, what is
  assumed, why each decision was taken, what is next, and which environment traps
  cost hours
- [`docs/spec_v7.md`](docs/spec_v7.md) — the specification. The maths, the
  accounting and the protocol are binding
- [`preregistration.md`](preregistration.md) — M1's preregistration. **Not frozen
  yet**; §9 lists what is missing
- [`docs/audit.md`](docs/audit.md) — the pre-M0 audit of v6. Kept as written: the
  record of what was known when each decision was made
- [`docs/gate_a_dry_run.md`](docs/gate_a_dry_run.md) — Gate A rehearsed against
  the literature before spending any GPU time

---

## What is verified, what is assumed

This distinction has to be made explicitly.

**Verified (tested):**

- All of the accounting. The golden constants are derived independently in exact
  rational arithmetic; `accounting.py` computes them through a general dispatch.
  Two routes, one answer
- The E8P codebook construction — 227+29 source patterns by enumeration, 2¹⁶
  distinct codewords, lattice membership, **exactly 2 bits per weight**
- **The cost side of `vq_bits = 2.0`, from real checkpoints** — in the QuIP# E8P
  and QTIP releases the codeword payload is exactly 2.000000, and 2.005204 /
  2.006740 with per-layer side information. The manifest arithmetic and the total
  file size agree exactly (`experiments/m0_vq_bits.py`)
- That rotation preserves the mask (on both axes, at every `T`)
- That compensation is forward-only, and that its gain comes from channel
  correlation
- That calibration reads the compressed model — on synthetic blocks **and on a
  real Llama**
- That the adapter reproduces the model's own computation exactly (hand-driven
  blocks → the model's logits, 1e-5)
- That Gate B does **not** say "interior" on noise
- **The `Δ = Q + τ` transfer bias** — the predictor was built and run alongside
  the real pipeline. The `T=1` identity check is exactly zero; the bias exceeds
  draw noise by 12.3×, so the preregistration's tolerance cannot be derived from
  seed variance
- **Gate B's statistical power** — 5 draws detect 2.29 σ and the measured effect
  is 6.7 σ. But neighbouring tiles (0.31 σ) do not separate, which is why `T*` is
  reported as a **set** rather than a point (`experiments/m0_gate_b_power.py`)
- **The value of rotation on a real layer** — synthetic data read 3%, the real
  `o_proj` reads **−70%**. The synthetic measurement was two orders of magnitude
  out, because rotation's job is to spread a heavy tail and synthetic data has no
  tail (`experiments/m0_rotation_value.py`)
- **The full-model driver and its resume** — real Llama-2-7B blocks were
  compressed, a checkpoint was written, and the test is not "resume works" but
  **"resume is invisible in the answer"**: an interrupted run lands on the
  uninterrupted one's perplexity. The alternative would be calibrating block 17
  against activations no version of the model ever produced — and that does not
  crash, it is quietly wrong
- **Two precision levers audited through a real block** — each measured twice
  (the term alone, and the term inside the pipeline), and in six comparisons out
  of six the in-situ saving is 95–107% of the isolated one. A third lever (fp16)
  **failed** the same audit and was withdrawn
  (`experiments/m0_lever_audit.py`)

**Assumed — not verified:**

> That E8P holds its 2-bit quality on the **compacted survivor submatrix**.
> Survivors are by definition the heavy tail of the distribution, and a lattice
> quantizer wants near-Gaussian input. Rotation should fix that, but it has not
> been shown.
>
> Measurement closed the **cost** side, not the quality side: it is certain that
> 2 bits are paid, and open whether 2 bits of *quality* are received for them.

The cheap experiment that would test this assumption was deliberately skipped.
The early-warning rule and the fallback are defined in `preregistration.md` §9.1.

**The first real measurement (2026-08-21).** Llama-2-7B dense perplexity,
WikiText-2, layer-streamed on an 8 GB card:

| seqlen | measured | published | difference |
|---|---|---|---|
| 2048 | **5.4675** | 5.47 | −0.0025 |
| 4096 | **5.1143** | 5.12 | −0.0057 |

This gives two things at once: the pipeline reproduces published numbers, **and**
the 5.12/5.47 split in the literature is confirmed to be a sequence-length
artefact — a hypothesis that was recorded before the measurement.

**But compression quality is still unmeasured.** Outside the dense baseline,
every number is either a layer-level `tr(E H Eᵀ)` proxy or synthetic data. The
synthetic smoke test does produce a U-shaped error curve and Gate A does pass —
**but we generated the data, so that is not evidence for the thesis.**

---

## Cost — and the cost model is itself a result

The M1 grid (3 budgets × 7 tile sizes × 5 draws) first priced out at **120 days**
on this machine. It is now **11.7 days**, and most of the difference is not
faster code but a more accurate model:

| | what happened |
|---|---|
| 120 → 12 days | the Cholesky rate was measured at k=2048 and applied at every width; at real widths it overcharged **9.4×** |
| 12 → ~40 days | two terms the model **did not know existed** were found: calibration and forward compensation. The number went **up** once |
| ~40 → 15 days | those two terms were fixed (Hessians on the block's own device: **25×**) |
| 15 → 11.7 days | the model began pricing the arithmetic the pipeline **actually runs** |

The last one was of this kind: `rotation_seconds` had no Kronecker path at all
and was billing the dense form the pipeline had stopped running. It was
**pessimistic**, which is why nothing ever complained.

The model has been wrong nine times, and **seven of those were missing terms
rather than wrong ratios.** So the question to ask this model is not "is the
ratio right" but **"what is not on the list"**.

**Measurement hygiene is this project's real output**, and it lives in
`docs/STATUS.md` §14. The traps that kept landing: a constant measured in one
regime and applied in all of them, a test that watches the answer instead of the
path, a composition nobody ever ran, and a measurement that does not traverse the
path it changed. As rules: **a test must watch the path, not the answer**, and
**a new test is not accepted until it has been shown red against the old code**.

---

## Gaps

- **The compressed model's perplexity.** This is the critical path; the driver
  exists, it has not been run
- **The E8P assumption** (above) — the project's single largest risk
- **Freezing the preregistration.** The `Δ(T)` prediction curve and `T*_predicted`
  are open; both depend on the `τ` sweep and the sweep script is not written
- **The driver's context cost.** The arithmetic is priced to within 1.03× on a
  real layer, but the driver runs the same seven layers far slower, and the
  difference is hitting the memory ceiling — compression peaks at 5.4 GiB on an
  8 GiB card
- **LDLQ for Axis A** — currently `NotImplementedError`

---

## Licence

[Apache License 2.0](LICENSE). Chosen over MIT for the explicit patent grant:
this repository's contribution is a *method*, and MIT is silent on patents, which
leaves anyone using the code unprotected if a third party later claims one
nearby.

Nothing here is vendored. QuIP#, QTIP, QuaRot, SparseGPT, Wanda, GPTQ, VENOM and
OBR are all reimplemented from their papers — the E8P codebook by enumeration,
the accounting from the identity — so no upstream licence obligation is
inherited.

---

## Notes

Test code is **1.4×** the size of production code (5824 against 4115 lines, 674
tests) and that is deliberate: most of the code carries a mathematical claim that
has to be verified, and the most expensive class of error here is silently
producing a wrong number. Most of the tests check a **claim** rather than a
behaviour — where the identity is valid, how the codebook is constructed, that
rotation preserves the mask, that the gate does not pass on noise.

`tests/golden.py` is a special file — it does **not** import `accounting.py`. A
test that calls the thing it is checking proves nothing, so the same numbers are
reached along two independent routes.

Parts of this work were developed with AI assistance (Claude).
