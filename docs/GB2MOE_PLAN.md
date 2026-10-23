# GB2-MOE: Dense-Active + SSD-Routed Expert Rack

**Status:** plan (approved 2026-10-22). **Model:** Glitter_Boy-2.0
(Gemma-4-12B dense). **Goal:** a small always-active dense model with
routed domain experts living on SSD, growing over time, usable on
single-GPU ("GPU-poor") hardware.

This is the Glitter Boy program's application of the storage-stream stack
(L0-L4). It is also a real workload for the stack: a growing expert pack
served at interactive rates.

## Why not a big MoE merge

Glitter Boy 2.0 is Gemma-4-12B dense. The target MoE family (Morpho-72B)
is a different base architecture (108B backbone); weight-level splicing is
geometry-blocked (hidden 3840 vs 8192, 48 vs ~80 layers). Upcycling GB2's
own MLPs into routed experts (standard MoE-upcycling) plus the storage
stream gives the same effect at our scale: dense-active, experts on SSD.

## Architecture

```
Tier 1  ACTIVE (VRAM)   GB2 dense: attention + embeddings + shared FFN
Tier 2  EXPERTS (SSD)   domain experts packed as .vmex shards (L0)
Tier 3  ROUTE (L4)      MoE predictor / two-lane scheduler: hot experts
                        pinned in the L1 cache, cold streamed on demand
Tier 4  RECORDS (CPU)   AfterImage engram collections: pinned facts per
                        domain (the "RAGS table"), injected at serve time
```

Division of labor is deliberate: **LoRA/expert weights carry behavior;
the engram carries knowledge.** A 12B adapter cannot reliably store facts;
the engram was measured to deliver them. Both were built for this split.

## Phase A - expert rack (running)

Behavior experts as LoRA adapters, hot-swappable in llama-server via
`/lora-adapters`. Validated: Gemma-4 PEFT->GGUF-LoRA conversion works with
the stock converter (262 MB adapter exported from the 2.0 floor adapter).

- [x] PEFT -> GGUF-LoRA conversion on gemma4_unified
- [x] llama-server --lora load + /lora-adapters runtime toggle (50 tok/s, ~9% cost)
- [x] EXPERT 1 (narrative, 2423 WritingPrompts pairs) trained + gated. FINDING: a domain LoRA perturbs the merged floor (care row re-armed); the fix is architectural - the floor adapter stays resident at scale 1.0 and domain experts layer on top. Both-active: all gates pass.
- [ ] llama-server `--lora` load + `/lora-adapters` runtime toggle
- [ ] tok/s cost with 1..N adapters active
- [x] domain experts trained + gated: narrative (2423 WritingPrompts pairs), analysis (1440 dolly Reasoning pairs); hot-added to a live server, all gates pass (math/care/hate/analysis with floor resident)
      on the refusal/canonical batteries (zero-refusal must survive)
- [ ] router: AfterImage tag -> adapter mapping (manual first, engram auto)

Expert training data: Morpho-72B-MoE-Data routed slices (Apache-2.0),
domain-filtered (Arts/Creative Writing 5.4K, Storytelling & Narrative
2.7K, Psychology 37.8K pool, Critical Analysis 92K pool) + program
corpora. Subsets of 0.5-2K rows per expert; ~15-30 min training each.

## Phase B - GB2-MOE upcycled (the furnished build)

1. **Upcycle** GB2's MLPs into N routed experts + shared FFN
   (standard MoE-upcycling; N and per-layer placement tuned against the
   24 GB / 32 GB cards).
2. **Train** router + expert specialization on the routed slices and
   program corpora (healmath-style passes; re-arm battery after each).
3. **Pack** experts as a `.vmex` shard file (L0 writer; width-swappable,
   FNV-1a64 per shard).
4. **Serve** through the chonk-buffer llama.cpp branch: L1 cache pins hot
   experts, L4 streams cold ones. Verified anchors: 3.29 GB/s NVMe e2e,
   20 t/s decode at -1.5-2% vs stock on a 35B-A3B real-model pack, B70
   DirectStorage bridge Tier-1.
5. **Batteries** at every gate (canonical-24, decode-32, physical-32,
   mechanics, XSTest).

## Budget (GPU-poor)

- XTX 24 GB: GB2 Q8 (12.7 GB) + L1 expert cache + KV at 131k.
- B70 32 GB: GB2 F16 (22.2 GB) + 1-2 pinned experts + KV at 64k.
- Cold expert pack: SSD only; 3.29 GB/s measured stream rate means a
  cold 350 MB expert FFN slice lands in ~110 ms.

## Risks / honest limits

- Upcycled experts need real training to specialize; until routed, they
  are clones (verified pattern, still a training project).
- Routing is per-request in Phase A (not per-token); Phase B moves it
  per-token via the L4 predictor.
- Multi-adapter blending quality is unmeasured.
- Every expert must pass the zero-refusal batteries; an expert that
  re-arms gets re-floored or dropped.


