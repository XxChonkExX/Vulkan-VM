"""CPU-only checks for family_adapters. No model weights loaded.

Run:  python crucible/test_family_adapters.py
Exit non-zero on any failure. These are the checks that would otherwise
only fail halfway through a 30-minute GPU run.
"""

from __future__ import annotations

import sys

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


print("== reasoning stripping ==")
from family_adapters import strip_reasoning  # noqa: E402

t, found = strip_reasoning("<think>secret</think>visible answer")
check("closed think removed", found and t == "visible answer", repr(t))
t, found = strip_reasoning("answer text<think>unfinished reasoning")
check("open think truncated", found and t == "answer text", repr(t))
t, found = strip_reasoning("<think>a</think>x<think>b</think>y")
check("multiple blocks removed", found and t == "x y", repr(t))
t, found = strip_reasoning("<think></think>")
check("empty think -> empty", found and t == "", repr(t))
t, found = strip_reasoning("<think>has </think> literal tags</think>kept")
check("orphan closer keeps text, drops markup",
      found and t == "literal tags kept", repr(t))
t, found = strip_reasoning("keep this</think>drop only the tag")
check("bare orphan closer is not a block boundary",
      found and t == "keep this drop only the tag", repr(t))
raw = "no reasoning here"
t, found = strip_reasoning(raw)
check("plain text untouched", not found and t == raw, repr(t))
check("no-op is byte-identical (no whitespace normalisation)",
      t == raw and strip_reasoning("  padded  ")[0] == "  padded  ",
      repr(strip_reasoning("  padded  ")))
check("removal does not weld words", "tagskept" not in strip_reasoning(
      "<think>x</think> tags</think>kept")[0])

print("\n== greedy generation kwargs ==")
from family_adapters import greedy_gen_kwargs  # noqa: E402

g = greedy_gen_kwargs({"eos_token_id": 100257})
check("do_sample forced off", g["do_sample"] is False)
check("num_beams 1", g["num_beams"] == 1)
check("temperature cleared", g["temperature"] is None)
check("top_p cleared", g["top_p"] is None)
check("eos wrapped in list", g["eos_token_id"] == [100257], repr(g.get("eos_token_id")))
g2 = greedy_gen_kwargs({"eos_token_id": [1, 2]})
check("eos list passthrough", g2["eos_token_id"] == [1, 2])
g3 = greedy_gen_kwargs(None)
check("no gen_cfg tolerated", "eos_token_id" not in g3)

print("\n== fail-closed registry ==")
from family_adapters import adapter_for_config, REGISTRY  # noqa: E402


class Cfg:
    def __init__(self, archs):
        self.architectures = archs


check("gemma4 dispatches", adapter_for_config(Cfg(["Gemma4ForConditionalGeneration"])).name == "gemma4")
check("granite dispatches", adapter_for_config(Cfg(["GraniteForCausalLM"])).name == "granite")
check("granite moe dispatches", adapter_for_config(Cfg(["GraniteMoeForCausalLM"])).name == "granite")
try:
    adapter_for_config(Cfg(["SomeUnknownForCausalLM"]))
    check("unknown arch fails closed", False, "no exception raised")
except SystemExit as e:
    check("unknown arch fails closed", "SILENTLY" in str(e))
try:
    adapter_for_config(Cfg([]))
    check("empty arch fails closed", False, "no exception raised")
except SystemExit:
    check("empty arch fails closed", True)
check("registry has 2 adapters", len(REGISTRY) == 2, str(sorted(REGISTRY)))

print("\n== granite chat template (tokenizer only, no weights) ==")
GRANITE_DIR = "/home/chonke/Downloads/granite"
try:
    from transformers import AutoTokenizer
    import family_adapters as fa

    gtok = AutoTokenizer.from_pretrained(GRANITE_DIR)
    on = gtok.apply_chat_template([{"role": "user", "content": "PROBE"}],
                                  tokenize=False, add_generation_prompt=True)
    off = fa.GRANITE.chat_text(gtok, "PROBE")
    check("thinking disabled by adapter", off.endswith("<think></think>"), repr(off[-24:]))
    check("raw default WOULD leak", on.endswith("<think>\n"), repr(on[-16:]))
    check("no open think after adapter", not off.rstrip().endswith("<think>"))
    check("user turn present", "<|im_start|>user\nPROBE<|im_end|>" in off)
    check("chatml delimiter", "<|im_start|>assistant" in off)
    check("empty system block injected (documented)",
          off.startswith("<|im_start|>system\n<|im_end|>"), repr(off[:34]))
    check("eos is im_end", gtok.eos_token == "<|im_end|>" and gtok.eos_token_id == 100257)
    check("thinking-tokens are single ids",
          gtok.convert_tokens_to_ids("<think>") == 100274
          and gtok.convert_tokens_to_ids("</think>") == 100275)
    check("gen kwargs greedy for granite",
          fa.GRANITE.gen_kwargs({"eos_token_id": 100257})["do_sample"] is False)
except Exception as exc:                                        # noqa: BLE001
    check(f"granite template checks (skipped: {type(exc).__name__}: {exc})", False)

print("\n== gemma template parity (must match the pre-adapter inline code) ==")
try:
    from transformers import AutoTokenizer
    import family_adapters as fa

    GEMMA_DIR = "/home/chonke/Downloads/gemma412b/gemma-4-12B-it-official"
    dtok = AutoTokenizer.from_pretrained(GEMMA_DIR)
    for probe in ["PROBE_SENTINEL", "What is 2+2?", ""]:
        inline = dtok.apply_chat_template(
            [{"role": "user", "content": probe}],
            tokenize=False, add_generation_prompt=True)
        viaad = fa.GEMMA4.chat_text(dtok, probe)
        check(f"byte-identical {probe[:14]!r}", inline == viaad)
except Exception as exc:                                        # noqa: BLE001
    check(f"gemma parity (skipped: {type(exc).__name__}: {exc})", False)

print()
if FAIL:
    print(f"{len(FAIL)} FAILURE(S): {FAIL}")
    sys.exit(1)
print("all adapter checks passed")
