"""Full GB2-MOE evaluation: decode-32, physical-32, mech-20 in one load.

Rebuilds the upcycled MoE from moe_state.pt and runs the program banks.
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes.nn.modules as bnb_nn

_o = bnb_nn.fix_4bit_weight_quant_state_from_module


def _f(m):
    try:
        return _o(m)
    except AssertionError:
        pass


bnb_nn.fix_4bit_weight_quant_state_from_module = _f

R = r"C:\llmwork\refusal_arm"
sys.path.insert(0, R)
from moe_gemma4 import upcycle  # noqa: E402

cfg = json.load(open(os.path.join(R, "ckpt", "gb2moe", "moe_config.json"),
                    encoding="utf-8"))
M = cfg["base"]
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16)
tok = AutoTokenizer.from_pretrained(M, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    M, quantization_config=bnb, device_map="xpu", dtype=torch.bfloat16,
    attn_implementation="sdpa", local_files_only=True)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
moes = upcycle(model, cfg["layer_ids"], n_experts=cfg["n_experts"],
               expert_intermediate=cfg["expert_intermediate"], top_k=1)
st = torch.load(os.path.join(R, "ckpt", "gb2moe", "moe_state.pt"),
                map_location="xpu", weights_only=False)
model.load_state_dict(st["model"], strict=False)
model.eval()
print("[moe] loaded", flush=True)


def gen(prompt, maxnew=320, rp=False):
    s = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                tokenize=False, add_generation_prompt=True)
    enc = tok(s, return_tensors="pt").to("xpu")
    kw = dict(max_new_tokens=maxnew, do_sample=False)
    if rp:
        kw.update(repetition_penalty=1.15, no_repeat_ngram_size=4)
    with torch.inference_mode():
        o = model.generate(**enc, **kw)
    return tok.decode(o[0][enc["input_ids"].shape[1]:],
                      skip_special_tokens=True)


def run_bank(bank_file, outdir, field, maxnew=320):
    rows = [json.loads(l) for l in open(bank_file, encoding="utf-8")
            if l.strip()]
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "transcripts.jsonl"), "w",
              encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            o = gen(r[field], maxnew)
            f.write(json.dumps({"pid": r.get("id", r.get("pid", i)),
                                "category": r.get("category", ""),
                                "prompt": r[field], "output": o},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{os.path.basename(outdir)} {i}/{len(rows)}] "
                  f"{o[:80].replace(chr(10), ' ')}", flush=True)


# decode-32
run_bank(os.path.join(R, "decode_bank_small.jsonl"),
         os.path.join(R, "scored", "decode_gb2moe"), "prompt")
# physical-32
run_bank(os.path.join(R, "physical_bank32.jsonl"),
         os.path.join(R, "scored", "phys32_gb2moe"), "prompt")

# mech-20: pull BANK from mech_probe.py without running its main
src = open(os.path.join(R, "mech_probe.py"), encoding="utf-8").read()
head = src.split("ap = argparse.ArgumentParser()")[0]
ns = {}
exec(head, ns)
BANK = ns["BANK"]
out = os.path.join(R, "scored", "mech_gb2moe")
os.makedirs(out, exist_ok=True)
summary = {}
for name, rp in (("greedy", False), ("greedy_rp115_nr4", True)):
    ok = 0
    rows = []
    for i, (q, exp) in enumerate(BANK):
        o = gen(q + " Answer with only the number or yes/no.", 48, rp)
        hit = any(x.lower() in o.lower() for x in exp)
        ok += int(hit)
        rows.append({"i": i, "q": q, "exp": exp, "ok": hit, "out": o})
        print(f"[mech {name} {i+1}/{len(BANK)}] "
              f"{'OK ' if hit else 'MISS'} exp={exp} | "
              f"{o[:60].replace(chr(10), ' ')}", flush=True)
    summary[name] = {"n": len(BANK), "ok": ok, "acc": round(ok / len(BANK), 3)}
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("MOE_EVAL_DONE", json.dumps(summary), flush=True)
