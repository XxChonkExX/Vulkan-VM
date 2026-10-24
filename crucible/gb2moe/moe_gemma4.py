"""GB2-MOE: MoE-upcycling for Gemma-4-12B (dense-active + routed experts).

MoEMLP: wraps GB2's original MLP as a always-on SHARED ffnn, adds N small
routed experts (reduced intermediate, bf16, trainable) with a token-level
top-1 router. Base model stays 4-bit frozen; experts train at bf16.

upcycle(model, layer_ids, n_experts, expert_intermediate):
  replaces layers[i].mlp with MoEMLP in-place.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEMLP(nn.Module):
    def __init__(self, base_mlp, n_experts=4, expert_intermediate=1536,
                 top_k=1, noise=0.02):
        super().__init__()
        self.shared = base_mlp            # frozen reference (always on)
        self.shared.requires_grad_(False)
        hidden = base_mlp.gate_proj.in_features
        self.n_experts = n_experts
        self.top_k = top_k
        # small routed experts: reduced-intermediate FFNs, warm-started
        # from slices of the shared weights + noise, trained at bf16
        self.experts = nn.ModuleList()
        dtype = torch.bfloat16
        import bitsandbytes.functional as bnbF
        Wg = bnbF.dequantize_4bit(base_mlp.gate_proj.weight,
                                  base_mlp.gate_proj.quant_state).to(dtype)
        Wu = bnbF.dequantize_4bit(base_mlp.up_proj.weight,
                                  base_mlp.up_proj.quant_state).to(dtype)
        Wd = bnbF.dequantize_4bit(base_mlp.down_proj.weight,
                                  base_mlp.down_proj.quant_state).to(dtype)
        dtype = torch.bfloat16
        for _ in range(n_experts):
            gate = nn.Linear(hidden, expert_intermediate, bias=False).to(
                dtype=dtype, device=base_mlp.gate_proj.weight.device)
            up = nn.Linear(hidden, expert_intermediate, bias=False).to(
                dtype=dtype, device=base_mlp.gate_proj.weight.device)
            down = nn.Linear(expert_intermediate, hidden, bias=False).to(
                dtype=dtype, device=base_mlp.gate_proj.weight.device)
            with torch.no_grad():
                # warm start from slices of the shared (dequantized) weights
                gate.weight.copy_(Wg[:expert_intermediate, :])
                up.weight.copy_(Wu[:expert_intermediate, :])
                down.weight.copy_(Wd[:, :expert_intermediate])
                for lin in (gate, up, down):
                    lin.weight.add_(torch.randn_like(lin.weight) * noise)
            e = nn.Module()
            e.gate_proj, e.up_proj, e.down_proj = gate, up, down
            e.requires_grad_(True)
            e = e.to(dtype=dtype)
            self.experts.append(e)
        self.router = nn.Linear(hidden, n_experts, bias=False).to(
            dtype=torch.bfloat16, device=base_mlp.gate_proj.weight.device)
        self.router.requires_grad_(True)

    def forward(self, x):
        shared_out = self.shared(x)
        b, s, h = x.shape
        x2 = x.reshape(-1, h)
        logits = self.router(x2.to(self.router.weight.dtype))   # [b*s, n]
        probs = F.softmax(logits.float(), dim=-1)
        self._last_logits = logits                # live, for router CE
        self._last_probs_mean = probs.mean(dim=0)  # live, for balance loss
        top_p, top_i = torch.topk(probs, self.top_k, dim=-1)
        top_i = top_i[:, 0]
        gate = top_p[:, 0].to(x.dtype).unsqueeze(-1)
        expert_out = torch.zeros_like(x2, dtype=x.dtype)
        with torch.no_grad():
            self._last_counts = {i: int((top_i == i).sum().item())
                                 for i in range(self.n_experts)}
        for i, e in enumerate(self.experts):
            mask = top_i == i
            if not mask.any():
                continue
            xs = x2[mask].to(e.gate_proj.weight.dtype)
            eo = e.down_proj(F.silu(e.gate_proj(xs)) * e.up_proj(xs))
            expert_out[mask] = eo.to(x2.dtype)
        moe_out = (expert_out * gate).reshape(b, s, h)
        return shared_out + moe_out


def upcycle(model, layer_ids, n_experts=4, expert_intermediate=1536,
            top_k=1, noise=0.02):
    """Replace layers[i].mlp with MoEMLP for i in layer_ids. Returns
    the list of MoEMLP modules."""
    lm = getattr(model, "language_model", None)
    if lm is None:
        lm = model.model.language_model
    layers = lm.layers
    moes = []
    for i in layer_ids:
        mlp = layers[i].mlp
        moe = MoEMLP(mlp, n_experts=n_experts,
                     expert_intermediate=expert_intermediate,
                     top_k=top_k, noise=noise)
        layers[i].mlp = moe
        moes.append(moe)
    return moes


def trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]

