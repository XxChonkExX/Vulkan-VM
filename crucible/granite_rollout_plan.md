# Granite rollout plan

Status: **adapter ready and tested, model not yet run.** The blocker is
memory, not tooling. Everything below is decided in advance so the first
Granite battery is a scheduled run rather than an archaeology dig.

## What the adapter already handles

`crucible/family_adapters.py`, verified by `crucible/test_family_adapters.py`
(34 CPU-only checks, no weights loaded).

| hazard | status |
|---|---|
| arch dispatch, fail-closed on unknown | done |
| thinking block leaks answer into scored text | done — `enable_thinking=False` mandatory |
| shipped `do_sample=true, temp 1.0` | done — greedy forced, eos from generation_config |
| stray / truncated `<think>` in output | done — 3-case stripper, no-op when absent |
| residual-stream layer discovery | done — per-family, refuses to guess |
| RoPE | done — explicit *status string*: Gemma patched, Granite native |

Two of those were live corruption risks, not plumbing. Granite's default
generation prompt ends `<|im_start|>assistant\n<think>\n` — an **open**
reasoning block. The trace would contain the answer before the scored
response, so every scorer would read reasoning text instead of disposition.
And `generation_config` ships `do_sample: true, temperature: 1.0,
top_p: 0.95`, which would have made every number irreproducible.

## Measured shape

- `GraniteForCausalLM`, 64 layers, hidden 4096
- GQA: 32 heads / 8 KV heads
- **`sliding_window`: absent → full attention, all 64 layers**
- `rope_theta` 5e7, `rope_type: default` (nested `rope_parameters`)
- 58.55 GB bf16 ≈ 29B params, 11 shards
- max_position 131072
- eos == pad == `<|im_end|>` (100257); ChatML; vocab 100352

## The blocker

A full-attention eager load of this checkpoint previously froze the box.
Cause is arithmetic, not mysterious: 64 layers × full attention × no
sliding window, with 58.6 GB of weights already resident. There is no
headroom for a KV cache or activations at long sequence length.

## Order of attempts — stop at the first that works

1. **Eager, small batch, short sequences.** `--batch 1`, seq ~2048. If
   58.6 GB weights + 64 layers of activations fit, this is the reference
   path and everything downstream is calibrated against it.
2. **Eager + tiled recompute via the existing Chonk machinery**
   (`patch_eager_attention_recompute` already exists in
   `train_granite_chonk.py`). Per-layer, recompute in tiles.
3. **Partial CPU offload** (`device_map` with some layers on CPU). Slow but
   it will not freeze. Acceptable for *measurement* runs where the battery
   is 468 prompts and we need activations at 64 layers, not throughput.
4. **INT4 / quantized weights.** Changes the geometry we are measuring.
   **Not acceptable for the abstain-direction extraction** — quantization
   perturbs exactly the low-variance directions this project is about. Only
   ever use for smoke tests and throughput work.

Rule: if 1–3 all fail, the answer is *not* to quantize the measurement
path. It is to report that the checkpoint does not fit this hardware and
use a smaller Granite variant for the cross-family claim, with the size
difference stated as a limit.

## Confounds to carry into every Granite claim

These are **not** bugs and must appear in any writeup:

1. **Template confound.** Granite uses ChatML (`<|im_start|>role`); Gemma
   uses its own `<turn>` roles. Granite also *injects an empty system
   block* that Gemma does not. Cross-family refusal rates therefore
   conflate template with model. This is a ceiling on what the comparison
   can mean, not a detail to footnote.
2. **Thinking disabled.** Required for scorer validity (see above), which
   means a Granite number is not comparable to a thinking-enabled Granite
   run. Say which one produced every published figure.
3. **Single tokenizer per family.** Same probe text through two tokenizers
   confounds tokenizer with the embedding matrices. Separating this needs
   matched-data/different-tokenizer *training*, which is a study, not a
   patch.
4. **Size.** 29B vs Gemma 12B. Any refusal/abstention difference is
   confounded with scale until a same-size pair exists.

## What a successful first run buys

- Cross-family replication of the **L46 abstention locus**. It held across
  DPO, PhaseA, and two unrelated-tuning corpora on Gemma. If it also holds
  on Granite, the claim upgrades from "a Gemma architecture locus" to
  "something that recurs across families" — which is the whole thesis of
  the project.
- The third-template adapter, exercising the item-11 rule that a new model
  gets an adapter rather than a fork.

## Not scheduled

Granite stays parked until Gemma dose calibration closes (xc at 13.6k/14.8k
and the 24k extraction). This plan is written and ready; nothing is queued.
