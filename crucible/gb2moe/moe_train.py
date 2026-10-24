"""GB2-MOE training: upcycled experts + router on routed domain data.

Checkpoint/resume built in (train_state.pt). At end saves
moe_state.pt (trainables) + moe_config.json (layer ids / dims) for the
serving loader.

Usage:
  python moe_train.py --model D:\\llmwork\\ckpt\\gb_hm30g \
      --pairs slices\\all_domains.jsonl --steps 5250 --resume
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes.nn.modules as bnb_nn

_orig = bnb_nn.fix_4bit_weight_quant_state_from_module


def _fix(module):
    try:
        return _orig(module)
    except AssertionError:
        pass


bnb_nn.fix_4bit_weight_quant_state_from_module = _fix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_gemma4 import upcycle, trainable_params  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--pairs", required=True)
ap.add_argument("--adapter-out", required=True)
ap.add_argument("--steps", type=int, required=True)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--n-experts", type=int, default=7)
ap.add_argument("--expert-intermediate", type=int, default=1280)
ap.add_argument("--router-ce-w", type=float, default=1.0)
ap.add_argument("--balance-w", type=float, default=0.01)
ap.add_argument("--layer-start", type=int, default=8)
ap.add_argument("--layer-step", type=int, default=3)
ap.add_argument("--layer-end", type=int, default=40)
ap.add_argument("--ckpt-every", type=int, default=100)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--reset-schedule", action="store_true", help="resume weights/opt but restart the step counter")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

import random
random.seed(a.seed)
torch.manual_seed(a.seed)
dev = "xpu" if torch.xpu.is_available() else "cpu"

LAYER_IDS = list(range(a.layer_start, a.layer_end + 1, a.layer_step))
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
rows = [json.loads(l) for l in open(a.pairs, encoding="utf-8") if l.strip()]
examples = []
for r in rows:
    prefix = tok.apply_chat_template(r["prompt"], tokenize=False,
                                     add_generation_prompt=True)
    text = prefix + r["chosen"][0]["content"]
    ids = tok(text, add_special_tokens=False).input_ids
    pids = tok(prefix, add_special_tokens=False).input_ids
    k = 0
    while k < min(len(pids), len(ids)) and pids[k] == ids[k]:
        k += 1
    if len(ids) > a.seq or k == 0:
        continue
    labels = list(ids)
    labels[:k] = [-100] * k
    examples.append((ids, labels, int(r.get("domain", 0))))
print(f"[data] {len(examples)} usable examples", flush=True)
if not examples:
    raise SystemExit("no usable rows")

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=False)
model = AutoModelForCausalLM.from_pretrained(
    a.model, quantization_config=bnb, device_map="xpu", dtype=torch.bfloat16,
    attn_implementation="eager", local_files_only=True)
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": True})
model.config.use_cache = False

print(f"[upcycle] layers={LAYER_IDS} n={a.n_experts} "
      f"ei={a.expert_intermediate}", flush=True)
moes = upcycle(model, LAYER_IDS, n_experts=a.n_experts,
               expert_intermediate=a.expert_intermediate, top_k=1)
trainable = trainable_params(model)
n_tr = sum(p.numel() for p in trainable)
print(f"[moe] trainable {n_tr/1e6:.1f}M params", flush=True)

opt = torch.optim.AdamW(trainable, lr=a.lr, betas=(0.9, 0.95))
os.makedirs(a.adapter_out, exist_ok=True)
ckpt_path = os.path.join(a.adapter_out, "moe_train_state.pt")
cursor, start_step = 0, 1
if a.resume and os.path.exists(ckpt_path):
    st = torch.load(ckpt_path, map_location=dev, weights_only=False)
    missing = model.load_state_dict(st["model"], strict=False)
    opt.load_state_dict(st["opt"])
    cursor, start_step = st["cursor"], st["step"] + 1
    if a.reset_schedule:
        cursor, start_step = 0, 1
    print(f"[resume] step {st['step']} cursor {cursor}", flush=True)

log = open(os.path.join(a.adapter_out, "moe_loss_log.jsonl"),
           "a" if start_step > 1 else "w", encoding="utf-8")
tstart = time.time()
model.train()
opt.zero_grad(set_to_none=True)


def save_ckpt(step, cur):
    torch.save({"model": {n: p for n, p in model.named_parameters()
                          if p.requires_grad},
                "opt": opt.state_dict(), "step": step, "cursor": cur},
               ckpt_path)


for step in range(start_step, a.steps + 1):
    loss_sum = 0.0
    acc_sum = 0.0
    acc_n = 0
    for j in range(a.accum):
        ids, labels, dom = examples[(cursor + j) % len(examples)]
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        y = torch.tensor([labels], dtype=torch.long, device=dev)
        d = torch.full((len(ids),), dom, dtype=torch.long, device=dev)
        out = model(input_ids=x, labels=y)
        rce = 0.0
        bal = 0.0
        for moe in moes:
            lg = moe._last_logits.float()
            rce = rce + torch.nn.functional.cross_entropy(lg, d)
            fs = torch.tensor([moe._last_counts.get(i, 0)
                               for i in range(a.n_experts)],
                              dtype=torch.float, device=dev)
            fs = fs / fs.sum().clamp(min=1.0)
            bal = bal + a.n_experts * torch.sum(fs * moe._last_probs_mean)
        loss = out.loss + a.router_ce_w * (rce / len(moes)) \
            + a.balance_w * (bal / len(moes))
        loss.backward()
        loss_sum += float(out.loss.detach())
        with torch.no_grad():
            for moe in moes:
                acc_sum += float((moe._last_logits.float().argmax(-1)
                                  == d).float().mean())
        acc_n += len(moes)
        del out, loss, x, y, d
    cursor = (cursor + a.accum) % len(examples)
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    avg = loss_sum / a.accum
    racc = acc_sum / max(1, acc_n)
    log.write(json.dumps({"step": step, "loss": avg,
                          "route_acc": round(racc, 3)}) + "\n")
    log.flush()
    if step % 25 == 0 or step == 1:
        u = moes[0]._last_counts
        print(f"[step {step}/{a.steps}] loss {avg:.4f} route {u} "
              f"peak {torch.xpu.max_memory_allocated()/1e9:.2f}GB "
              f"{time.time()-tstart:.0f}s", flush=True)
    if step % a.ckpt_every == 0 or step == a.steps:
        torch.save({"model": {n: p for n, p in model.named_parameters()
                              if p.requires_grad},
                    "opt": opt.state_dict(), "step": step, "cursor": cursor},
                   ckpt_path)
        print(f"[ckpt] step {step}", flush=True)

torch.save({"model": {n: p for n, p in model.named_parameters()
                      if p.requires_grad},
            "opt": opt.state_dict()}, 
           os.path.join(a.adapter_out, "moe_state.pt"))
json.dump({"layer_ids": LAYER_IDS, "n_experts": a.n_experts,
           "expert_intermediate": a.expert_intermediate, "top_k": 1,
           "base": a.model, "steps": a.steps},
          open(os.path.join(a.adapter_out, "moe_config.json"), "w"),
          indent=1)
print("[save] moe_state + config ->", a.adapter_out, flush=True)
print("[done]", flush=True)


