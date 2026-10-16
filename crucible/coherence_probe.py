"""coherence_probe.py -- model onboarding coherency check (b70-box,
2026-10-15). Given ANY checkpoint dir, answers: what class loads it
cleanly, does it need a template to cohere, does it leak channel
tokens, does it need adapters? One load, ~5 min XPU.
Stages (fail-loud, each gated):
 1. LOAD: AutoModelForCausalLM + output_loading_info on the EXPLICIT
    correct class (unified vs non-unified from config model_type).
    Reports missing/unexpected/mismatched; VOID if nonzero missing.
 2. TEMPLATE: does chat_template exist? render 2 fixed prompts; parse
    for thought/channel block openers (containment-relevant).
 3. GENERATION: 2 prompts x 2 arms (bare vs templated), greedy 24,
    entropy + top-1 + text, decoded BOTH skip settings (leak check).
 4. ADAPTERS: adapter_config.json present? report composition need.
VERDICT: coherent-bare / needs-template / degenerate + the load
recipe (class + template + adapter instructions).
Usage: py -3 coherence_probe.py MODEL_DIR OUT_JSON
"""
import json
import os
import sys

import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "coherence_probe.json"
PROMPTS = ["The dragon woke at dawn and",
           "In the year 1847, the cartographer"]
DEV = sys.argv[3] if len(sys.argv) > 3 else "xpu"


def main():
    if MODEL is None:
        sys.exit("need MODEL_DIR")
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    rep = {"model": MODEL, "stages": {}}
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    mt = getattr(cfg, "model_type", "?")
    tcfg = getattr(cfg, "text_config", None)
    rep["stages"]["config"] = {
        "model_type": mt,
        "text_model_type": getattr(tcfg, "model_type", None)
        if tcfg else None,
        "transformers_version": getattr(cfg, "transformers_version",
                                        None)}
    # 1. LOAD (explicit class by model_type family)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model, info = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map=DEV,
        output_loading_info=True, trust_remote_code=True)
    model.eval()
    rep["stages"]["load"] = {
        "class": type(model).__name__,
        "missing": len(info["missing_keys"]),
        "unexpected": len(info["unexpected_keys"]),
        "mismatched": len(info.get("mismatched_keys", []))}
    print(f"load: {rep['stages']['load']}", flush=True)
    assert rep["stages"]["load"]["missing"] == 0, "VOID: load not clean"
    # 2. TEMPLATE
    has_tpl = bool(getattr(tok, "chat_template", None))
    tpl_info = {"exists": has_tpl, "thought_block": False,
                "render_ok": False}
    if has_tpl:
        try:
            r = tok.apply_chat_template(
                [{"role": "user", "content": PROMPTS[0]}],
                tokenize=False, add_generation_prompt=True)
            tpl_info["render_ok"] = True
            tpl_info["thought_block"] = "<|channel>thought" in r
            tpl_info["tail"] = r[-80:]
        except Exception as e:
            tpl_info["render_error"] = str(e)[:120]
    rep["stages"]["template"] = tpl_info
    print(f"template: exists={has_tpl} thought={tpl_info['thought_block']} "
          f"render={tpl_info['render_ok']}", flush=True)
    # 3. GENERATION both arms
    arms = {}
    for arm in ("bare", "templated"):
        if arm == "templated" and not tpl_info["render_ok"]:
            arms[arm] = {"skipped": "no template"}
            continue
        rows = []
        for p in PROMPTS:
            if arm == "bare":
                enc = tok(p, return_tensors="pt").to(DEV)
            else:
                t = tok.apply_chat_template(
                    [{"role": "user", "content": p}], tokenize=False,
                    add_generation_prompt=True)
                enc = tok(t, return_tensors="pt",
                          add_special_tokens=False).to(DEV)
            with torch.inference_mode():
                o = model.generate(**enc, max_new_tokens=24,
                                   do_sample=False)
                logits = model(**enc).logits[0, -1].float()
            probs = torch.softmax(logits, dim=-1)
            ent = float(-(probs * probs.clamp_min(1e-12).log()).sum())
            top = torch.topk(probs, 1)
            g_skip = tok.decode(o[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True)
            g_raw = tok.decode(o[0][enc["input_ids"].shape[1]:],
                               skip_special_tokens=False)
            rows.append({
                "prompt": p[:40], "entropy": round(ent, 4),
                "top": [tok.decode([top.indices[0].item()]),
                        round(float(top.values[0]), 3)],
                "gen": g_skip[:120],
                "channel_leak": ("<|channel>" in g_raw
                                 or "thought" in g_raw.split()[:30])})
        ents = [r["entropy"] for r in rows]
        arms[arm] = {"rows": rows,
                     "mean_entropy": round(sum(ents) / len(ents), 4)}
        print(f"{arm}: ent={arms[arm]['mean_entropy']} "
              f"leak={[r['channel_leak'] for r in rows]}", flush=True)
    rep["stages"]["generation"] = arms
    # 4. ADAPTERS
    rep["stages"]["adapters"] = {
        "adapter_present": os.path.exists(
            os.path.join(MODEL, "adapter_config.json")),
        "merged_safetensors": os.path.exists(
            os.path.join(MODEL, "model.safetensors"))}
    # VERDICT
    b = arms.get("bare", {})
    t = arms.get("templated", {})
    if isinstance(b, dict) and "mean_entropy" in b and \
            b["mean_entropy"] < 2.5:
        verdict = "DEGENERATE-BARE"
    elif isinstance(t, dict) and "mean_entropy" in t:
        verdict = "NEEDS-TEMPLATE" if not (
            isinstance(b, dict) and "mean_entropy" in b
            and b["mean_entropy"] > 2.5) else "COHERENT-BARE"
    else:
        verdict = "UNDETERMINED"
    rep["verdict"] = verdict
    recipe = {"class": rep["stages"]["load"]["class"]}
    if verdict == "NEEDS-TEMPLATE":
        recipe["probe_with"] = "chat template (bare is off-distribution)"
    if rep["stages"]["adapters"]["adapter_present"]:
        recipe["compose"] = "base + adapter (not merged)"
    rep["load_recipe"] = recipe
    json.dump(rep, open(OUT, "w"), indent=1)
    print(f"VERDICT: {verdict} -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
