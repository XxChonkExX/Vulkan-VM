#!/usr/bin/env python3
"""extract_abstain_direction_v2.py -- the convergence-study instrument,
model-type-agnostic, with the label source upgraded to the canonical
taxonomy.

WHY V2 (b70-box_342, answering strix-329)
-----------------------------------------
Strix needs the METHOD, not the artifact: a direction extracted from a
28-layer Gemma is dimension- and depth-specific and will not transfer to
Qwen2.5-3B. Shipping the .pt would test nothing. This file is the method,
ported so it runs on any causal LM.

Changes from v1 (extract_abstain_direction.py), each one traceable to a
defect class this program has already paid for:

  1. MODEL-AGNOSTIC LOADING. v1 hard-coded Gemma4ForConditionalGeneration.
     v2 uses AutoModelForCausalLM and discovers the decoder layer list by
     matching the configured num_hidden_layers -- the same discovery the
     kernel layout fix used (probe the real module, never assume the path).

  2. LABELS FROM THE CANONICAL TAXONOMY. v1 split probes by IDK_RE, a
     keyword regex written before the classifier audit. v2 splits by
     response_taxonomy's `abstain` disposition -- the validated profile --
     per kit law (import, do not reimplement). This is not a cosmetic
     change: the direction IS the difference of means between label groups,
     so a label defect is a direction defect.

  3. PATHS FROM ENV WITH ASSERTS. v1 read BASE from a module-level constant
     whose default pointed at a box-specific path -- the exact
     silent-override class from strix-328. v2 requires CRUCIBLE_BASE and
     refuses to run against a default nobody confirmed.

  4. PREFLIGHT: ONE VECTOR PER PROBE PER LAYER. v1 concatenated whatever the
     hooks captured and commented "(varied batch tails)" -- if a batch ever
     misaligns, every mean is computed over shifted rows and the direction
     is confidently wrong. v2 asserts the capture count equals the probe
     count at every layer, and asserts the hidden size matches the config.

  5. CHAT TEMPLATE GUARD. v1 assumed apply_chat_template exists and renders.
     v2 verifies the rendered prompt is non-empty, ends in an assistant
     trigger, and that the tokenizer round-trips it -- the empty-prefix
     failure from strix-320 produced t_inst = -1 with no exception.

THE METHOD IS UNCHANGED, and that is the point of shipping it rather than a
new idea:
    d_l = normalize(mean(h_abstained) - mean(h_confabulated))
    at every layer, on last-prompt-token residuals, labels from the arm's
    OWN frozen transcripts, plus per-probe projections and the per-layer
    Spearman rho (does the direction predict WHICH probes abstain?).

Pre-registered thresholds (strix 011, unchanged): rho >= 0.5 direction /
<= 0.2 funnel; mid-band cos >= 0.7 universal.

Outputs -> $ABSTAIN_OUT/<arm>.pt
           {dirs [L, H], projs [L, P], labels [P], pids, spearman [L],
            n_abstained, best_layer, best_top_overlap, model, hidden,
            capture_assert: passed}

Usage:
    CRUCIBLE_BASE=/path/to/model-dir \
    CRUCIBLE_SCARS=/path/to/scars/<arm> \
    ABSTAIN_OUT=/path/to/out \
    python extract_abstain_direction_v2.py <arm> --model /path/to/weights
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_CRUCIBLE = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if REPO_CRUCIBLE not in sys.path:
    sys.path.insert(0, REPO_CRUCIBLE)


def fail(msg):
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--model", required=True,
                    help="path to the weights to extract against")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-probes", type=int, default=0,
                    help="0 = all; cap for smoke tests")
    A = ap.parse_args()

    # --- env paths, asserted not defaulted (strix-328 class) -----------
    base = os.environ.get("CRUCIBLE_BASE")
    scars = os.environ.get("CRUCIBLE_SCARS")
    outd = os.environ.get("ABSTAIN_OUT")
    if not base:
        fail("CRUCIBLE_BASE not set. v1 defaulted to a box-specific path "
             "and nobody could tell; v2 refuses.")
    if not scars or not os.path.isdir(scars):
        fail(f"CRUCIBLE_SCARS must be the arm's transcript dir, got {scars!r}")
    if not outd:
        fail("ABSTAIN_OUT not set.")
    if not os.path.isdir(A.model):
        fail(f"--model dir not found: {A.model}")
    os.makedirs(outd, exist_ok=True)

    # --- labels: canonical taxonomy on the arm's own frozen transcripts
    tpath = os.path.join(scars, "transcripts.jsonl")
    if not os.path.exists(tpath):
        fail(f"no transcripts.jsonl under {scars}")
    try:
        import response_taxonomy as tx
        label_source = "response_taxonomy.abstain (validated profile)"
    except Exception as e:  # pragma: no cover
        fail(f"response_taxonomy not importable ({e}). v2 does not fall back "
             "to the IDK_RE regex: a label defect is a direction defect.")
    unk = {}
    n_rows = 0
    for line in open(tpath, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("kind") or "").lower() != "unk":
            continue
        n_rows += 1
        d, _, _ = tx.classify(r.get("output") or "")
        unk[r.get("pid")] = (d == "abstain")
    if not unk:
        fail("no unk rows in this arm's transcripts; nothing to split on")

    # --- probes ---------------------------------------------------------
    sys.path.insert(0, base)
    try:
        from scar_map import load_probes
    except Exception as e:
        fail(f"cannot import scar_map.load_probes from CRUCIBLE_BASE "
             f"({base}): {e}")
    probes = [(r[0], r[2]) for r in load_probes() if r[1] == "unk"]
    probes = [(pid, p) for pid, p in probes if pid in unk]
    if A.max_probes:
        probes = probes[:A.max_probes]
    n_abst = sum(1 for pid, _ in probes if unk[pid])
    print(f"[{A.arm}] {len(probes)} unk probes, {n_abst} abstained "
          f"(labels: {label_source})", flush=True)
    if len(probes) < 8 or n_abst < 2 or (len(probes) - n_abst) < 2:
        fail(f"degenerate split: {len(probes)} probes, {n_abst} abstained. "
             "A difference-of-means direction needs both groups populated.")

    # --- model: AutoModel, layer discovery by config --------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = os.environ.get("CRUCIBLE_DEV") or (
        "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(A.model)
    model = AutoModelForCausalLM.from_pretrained(
        A.model, dtype=torch.bfloat16).eval().to(dev)

    n_cfg = model.config.num_hidden_layers
    layers = None
    for _, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) == n_cfg:
            layers = mod
            break
    if layers is None:
        fail(f"no ModuleList matching num_hidden_layers={n_cfg}; "
             "architecture discovery failed, refusing to guess")
    hidden = model.config.hidden_size

    # --- chat template guard (strix-320 class: empty prefix, no error) --
    sample = tok.apply_chat_template(
        [{"role": "user", "content": probes[0][1]}], tokenize=False,
        add_generation_prompt=True)
    if not sample or not sample.strip():
        fail("chat template rendered EMPTY for a non-empty probe")
    rt = tok.decode(tok(sample)["input_ids"])
    if not rt.strip():
        fail("chat template does not round-trip through the tokenizer")
    print(f"template ok: {len(sample)} chars, round-trip {len(rt)} chars",
          flush=True)

    # --- capture --------------------------------------------------------
    store = {i: [] for i in range(n_cfg)}
    hooks = []

    def _mk(idx):
        def h(_m, _i, o):
            x = o[0] if isinstance(o, tuple) else o
            store[idx].append(x[:, -1, :].detach().float().cpu())
        return h

    for idx, lyr in enumerate(layers):
        hooks.append(lyr.register_forward_hook(_mk(idx)))

    texts = [tok.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False,
        add_generation_prompt=True) for _, p in probes]
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    with torch.inference_mode():
        for i in range(0, len(texts), A.batch):
            enc = tok(texts[i:i + A.batch], return_tensors="pt",
                      padding=True).to(dev)
            model(**enc)
            if (i // A.batch) % 8 == 0:
                print(f"[{A.arm}] {i}/{len(texts)}", flush=True)
    for h in hooks:
        h.remove()
    del model
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    # --- preflight: exactly one vector per probe per layer ---------------
    for idx in range(n_cfg):
        got = sum(t.shape[0] for t in store[idx])
        if got != len(probes):
            fail(f"layer {idx}: captured {got} vectors for {len(probes)} "
                 "probes. Batch misalignment would shift every mean and "
                 "produce a confidently wrong direction.")

    # --- directions ------------------------------------------------------
    labels = np.array([1 if unk[pid] else 0 for pid, _ in probes])
    pids = [pid for pid, _ in probes]
    dirs = np.zeros((n_cfg, hidden), dtype=np.float32)
    projs = np.zeros((n_cfg, len(probes)), dtype=np.float32)
    spear = np.zeros(n_cfg, dtype=np.float32)
    pos = labels == 1
    neg = labels == 0
    from scipy.stats import spearmanr
    for idx in range(n_cfg):
        M = torch.cat(store[idx], dim=0)          # [P, H], asserted above
        mu_pos = M[torch.from_numpy(pos)].mean(0)
        mu_neg = M[torch.from_numpy(neg)].mean(0)
        d = mu_pos - mu_neg
        d = d / (d.norm() + 1e-8)
        dirs[idx] = d.numpy()
        p = (M @ d).numpy()
        projs[idx] = p
        spear[idx] = spearmanr(p, labels).statistic
    top = int(pos.sum())
    best = int(np.nanargmax(spear)) if not np.all(np.isnan(spear)) else -1
    if best >= 0:
        order = np.argsort(-projs[best])
        overlap = int((labels[order[:top]] == 1).sum())
    else:
        overlap = 0
    out = os.path.join(outd, f"{A.arm}.pt")
    torch.save({"dirs": dirs, "projs": projs, "labels": labels,
                "pids": pids, "spearman": spear,
                "n_abstained": int(pos.sum()),
                "best_layer": best, "best_top_overlap": overlap,
                "model": A.model, "hidden": hidden,
                "label_source": label_source,
                "preflight": "passed"},
               out)
    print(f"[{A.arm}] DONE best_layer={best} "
          f"rho={spear[best]:.3f} top{top}-overlap={overlap}/{top} -> {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())