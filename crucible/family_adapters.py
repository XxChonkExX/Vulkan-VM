"""Per-family adapters for Crucible.

Doctrine (MODEL_ONBOARDING item 11 / METHOD.md): the UNIVERSAL layer is
questions, scorer families, label contracts and thresholds. Everything
model-shaped lives here. A new model gets an adapter, never a fork of the
measurement tools.

What an adapter owns, and why each one is a correctness hazard rather than
plumbing:

  load()            architecture dispatch. Fails CLOSED on unknown arch --
                    a silent wrong-architecture load degrades into
                    random-weight generation and produces numbers.
  find_layers()     the residual stream we hook. Gemma hides it behind
                    get_text_config(decoder=True); assuming a field name
                    is how you hook the wrong tensor.
  rope_status()     Gemma 4 needed a broadcast-multiply patch (verified
                    bit-exact). Granite uses rope_type "default" which
                    transformers handles natively. Applying the Gemma patch
                    to Granite would be wrong, and SKIPPING it silently on
                    a model that needs it is worse. So this returns a
                    status string, never a bare bool.
  chat_text()       the surface form. This is the single largest confound in
                    cross-family work and it is NOT a nuisance: refusal rates
                    are template-dependent. Granite primes a <think> block by
                    default; left on, the trace leaks the answer before the
                    scored response and every scorer reads the wrong text.
  gen_kwargs()      greedy, explicit. Granite ships do_sample=true,
                    temperature=1.0, top_p=0.95. A diagnostic that depends on
                    reproducibility cannot inherit a sampling default.
  strip_reasoning() defence in depth. Even with thinking disabled, a model
                    can emit <think> spontaneously; the scorer must never
                    see it.

CONFOUNDS THESE ADAPTERS DO NOT SOLVE (documented, not fixable here):
  Granite uses ChatML (<|im_start|>role ... <|im_end|>); Gemma 4 uses its
  own <turn> roles. Any cross-family refusal/abstention comparison therefore
  confounds template with model, exactly as the same probe text through two
  tokenizers confounds tokenizer with weights. Report it as a ceiling on
  what cross-family numbers can mean. Separating template from model needs
  matched-data/different-template training, which is a study, not a patch.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)
_OPEN_THINK = re.compile(r"<think>.*", re.S)
_ORPHAN_CLOSE = re.compile(r"</think>+")


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Remove <think> spans. Returns (clean_text, found_any).

    Three cases, in order:

    1. CLOSED blocks. Removed iteratively, not in one pass. A single
       non-greedy pass is wrong when blocks repeat: on
       '<think>a</think>x<think>b</think>y' it would match first-open to
       LAST close and delete the real text 'x' between them.
    2. UNCLOSED tail -- generation hit max_tokens mid-thought. Everything
       from the dangling <think> onward is dropped. The unfinished
       reasoning is not a scored response, and keeping it would let a
       model score as coherent by reasoning audibly before answering.
    3. ORPHAN closers with no opener. Cannot be attributed to any block,
       so we keep the text and delete just the markup. Discarding
       everything after a stray '</think>' would throw away real scored
       content on the strength of a mis-nested tag.

    Case 3 is the one that is a judgement call rather than a rule; it is
    listed here so the choice is visible instead of looking accidental.
    """
    if not text:
        return text, False
    cleaned, n = text, 0
    while True:
        # Replace with a space, not "": deleting a tag outright welds the
        # words on either side of it ("tags</think>kept" -> "tagskept").
        cleaned, k = _THINK_BLOCK.subn(" ", cleaned)
        n += k
        if k == 0:
            break
    if _OPEN_THINK.search(cleaned):
        cleaned = _OPEN_THINK.sub(" ", cleaned)
        n += 1
    cleaned, k = _ORPHAN_CLOSE.subn(" ", cleaned)
    n += k
    if n == 0:
        # Byte-identical passthrough. Important: this must not normalise
        # whitespace when there was no reasoning, or it stops being a
        # no-op for families that do not use think blocks.
        return text, False
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned, True


def greedy_gen_kwargs(gen_cfg: dict | None = None) -> dict:
    """Explicit greedy decode. Never inherit a sampling default."""
    kw = {
        "do_sample": False,
        "num_beams": 1,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }
    if gen_cfg:
        eos = gen_cfg.get("eos_token_id")
        if eos is not None:
            kw["eos_token_id"] = eos if isinstance(eos, list) else [eos]
    return kw


# --------------------------------------------------------------------------
# adapter contract
# --------------------------------------------------------------------------


@dataclass
class FamilyAdapter:
    name: str
    arch_patterns: tuple[str, ...]
    load_fn: Callable
    layers_fn: Callable
    rope_fn: Callable
    chat_fn: Callable
    gen_fn: Callable = greedy_gen_kwargs
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- convenience passthroughs ------------------------------------------
    def strip_reasoning(self, text: str) -> tuple[str, bool]:
        return strip_reasoning(text)

    def gen_kwargs(self, gen_cfg: dict | None = None) -> dict:
        return self.gen_fn(gen_cfg)

    def chat_text(self, tok, prompt: str) -> str:
        return self.chat_fn(tok, prompt)

    def chat_texts(self, tok, prompts: list[str]) -> list[str]:
        return [self.chat_fn(tok, p) for p in prompts]


# --------------------------------------------------------------------------
# Gemma 4  (current reference behaviour, extracted verbatim)
# --------------------------------------------------------------------------


def _gemma_load(model_dir: str, dev: str, ignore_mismatch: bool = False):
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              Gemma4ForConditionalGeneration)

    kw = {"ignore_mismatched_sizes": True} if ignore_mismatch else {}
    cfg = AutoConfig_shim(model_dir)
    archs = list(getattr(cfg, "architectures", None) or [])
    if any("Gemma4" in a for a in archs):
        model = Gemma4ForConditionalGeneration.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=dev, **kw).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=dev,
            trust_remote_code=True, **kw).eval()
    return model, AutoTokenizer.from_pretrained(model_dir)


def AutoConfig_shim(model_dir: str):
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model_dir, trust_remote_code=True)


def _gemma_layers(model):
    import torch
    want = (model.config.get_text_config(decoder=True).num_hidden_layers
            if hasattr(model.config, "get_text_config")
            else model.config.num_hidden_layers)
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == want:
            return mod
    raise SystemExit(
        f"[{type(model).__name__}] no ModuleList of {want} layers found; "
        "refusing to guess which tensor is the residual stream.")


def _gemma_rope(model) -> str:
    """Apply the verified broadcast-multiply patch.

    Verified max-abs-diff 0.0 against a CPU reference. Reports status
    rather than returning silently, so a skip is visible in the log.
    """
    import sys
    for p in (os.environ.get("CRUCIBLE_SCRIPTS"),
              "/home/chonke/Downloads/gemma412b",
              os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
            break
    try:
        from rope_patch import patch_rope_elementwise
        patch_rope_elementwise(model)
        return "patched: broadcast-multiply (verified 0.0 vs CPU reference)"
    except Exception as exc:                                    # noqa: BLE001
        return f"SKIPPED: {type(exc).__name__}: {exc}"


def _gemma_chat(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def _gemma_gen(gen_cfg=None):
    return greedy_gen_kwargs(gen_cfg)


GEMMA4 = FamilyAdapter(
    name="gemma4",
    arch_patterns=("Gemma4",),
    load_fn=_gemma_load,
    layers_fn=_gemma_layers,
    rope_fn=_gemma_rope,
    chat_fn=_gemma_chat,
    gen_fn=_gemma_gen,
    notes=(
        "Official arm must load WITHOUT ignore_mismatched_sizes; every "
        "other arm needs it (adapter geometry differs from base).",
        "grad_ckpt_patch.py required for training: HF's "
        "gradient_checkpointing_enable() is a silent no-op on Gemma4.",
    ),
)


# --------------------------------------------------------------------------
# Granite  (new -- IBM Granite, ChatML template, thinking block)
# --------------------------------------------------------------------------


def _granite_load(model_dir: str, dev: str, ignore_mismatch: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"ignore_mismatched_sizes": True} if ignore_mismatch else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map=dev,
        trust_remote_code=True, **kw).eval()
    return model, AutoTokenizer.from_pretrained(model_dir)


def _granite_layers(model):
    import torch
    want = model.config.num_hidden_layers
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == want:
            return mod
    raise SystemExit(
        f"[Granite] no ModuleList of {want} layers found; refusing to guess.")


def _granite_rope(model) -> str:
    """Granite needs no patch.

    config.rope_parameters = {"rope_theta": 50000000, "rope_type": "default"}
    (transformers >=4.57 nested format). rope_type "default" is handled
    natively. The Gemma patch keys off a different layout, so applying it
    here would be wrong -- hence an explicit status, not a silent skip.
    """
    rp = getattr(model.config, "rope_parameters", None) or {}
    rtype = rp.get("rope_type", getattr(model.config, "rope_scaling", None)
                   and "scaled") or "default"
    if rtype == "default":
        return (f"not-required: rope_type=default, "
                f"theta={rp.get('rope_theta', getattr(model.config, 'rope_theta', '?'))} "
                "(native); Gemma patch deliberately NOT applied")
    return f"REVIEW: rope_type={rtype!r} is not plain default -- verify before use"


def _granite_chat(tok, prompt: str) -> str:
    """enable_thinking=False is MANDATORY here.

    Default Granite emits '<|im_start|>assistant\\n<think>\\n' -- an OPEN
    reasoning block. The trace would contain the answer before the scored
    response, so every scorer would read reasoning text instead of the
    disposition being measured. Disabling yields a closed '<think></think>'
    so the answer starts immediately.

    Side effect worth knowing: this template also injects an EMPTY system
    block ('<|im_start|>system\\n<|im_end|>\\n') even with no system
    message. Harmless but it means Granite's prompt is not textually
    comparable to Gemma's, on top of the delimiter difference.
    """
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _granite_gen(gen_cfg=None):
    return greedy_gen_kwargs(gen_cfg)


GRANITE = FamilyAdapter(
    name="granite",
    arch_patterns=("GraniteForCausalLM", "GraniteMoeForCausalLM"),
    load_fn=_granite_load,
    layers_fn=_granite_layers,
    rope_fn=_granite_rope,
    chat_fn=_granite_chat,
    gen_fn=_granite_gen,
    notes=(
        "SHIPPED do_sample=true, temperature=1.0, top_p=0.95. MUST be "
        "overridden to greedy -- reproducibility is a precondition for the "
        "diagnostic, not a preference.",
        "eos == pad == <|im_end|> (100257). Left-padding with the eos id is "
        "safe only while attention_mask masks it.",
        "CONFOUND: ChatML + empty injected system block differ from Gemma's "
        "<turn> roles. Cross-family refusal rates conflate template with "
        "model. Treat as a ceiling on cross-family claims.",
        "CONFOUND: thinking disabled removes reasoning from the scored text. "
        "That is required for scorer validity but it means Granite's numbers "
        "are NOT comparable to a thinking-enabled Granite run.",
        "MEMORY: 29B bf16 = 58.6GB, 64 layers, NO sliding_window (full "
        "attention). Full-attention eager load previously froze the box. "
        "See granite_rollout_plan.md before scheduling.",
    ),
)


# --------------------------------------------------------------------------
# registry -- fail closed
# --------------------------------------------------------------------------

REGISTRY: dict[str, FamilyAdapter] = {a.name: a for a in (GEMMA4, GRANITE)}


def adapter_for_config(config) -> FamilyAdapter:
    archs = list(getattr(config, "architectures", None) or [])
    for ad in REGISTRY.values():
        if any(p in a for a in archs for p in ad.arch_patterns):
            return ad
    raise SystemExit(
        f"UNSUPPORTED ARCHITECTURE {archs!r}. Refusing to load: an "
        "architecture mismatch degrades SILENTLY into random-weight "
        "generation. Add a FamilyAdapter rather than a branch in a tool.")


def adapter_for_dir(model_dir: str) -> FamilyAdapter:
    return adapter_for_config(AutoConfig_shim(model_dir))
