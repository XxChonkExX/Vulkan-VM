# vulkanvm_attn_gemma.py
# Gemma-4 training-time tiled causal attention for the Chonk Buffer.
#
# Sibling of vulkanvm_attn_granite.py, same proven skeleton:
#   * pure torch (HIP BLAS) -- NO Triton, NO aotriton flash, NO vendor SDPA.
#     Vendor fused paths through Chonk dma-buf memory wedge the gfx1151 ring
#     (this module's Granite header documents the same failure); this kernel
#     never touches them.
#   * Flash-style online softmax over K-tiles: exact full-row softmax,
#     workspace O(B*kvH*g*QCHUNK*KTILE) regardless of sequence length.
#   * no_grad forward + hand-written exact-recompute backward (the Granite
#     ratchet fixes apply verbatim: detached stashed out, no graph in forward).
#
# Deltas vs the Granite module:
#   * No KV cache / cur_len: training forward sees the full packed block, so
#     the Q axis is ALSO tiled (QCHUNK, default 4096) -- scores never exceed
#     one (QCHUNK x KTILE) fp32 cell per visited tile.
#   * Per-layer sliding window (Gemma4: 40x sliding-1024 + 8x full): each
#     q-chunk visits only k-tiles inside its window -- sliding layers touch
#     ~2 tiles per chunk instead of all of them.
#   * Softcap replication (Gemma attention logit softcapping).
#   * Analytic causal+window mask per cell from position offsets -- no 4D
#     attention-mask tensor (which would be 2GB bf16 at 32k). If HF hands us
#     a non-None attention_mask we slice-add it on top; when None (packed
#     batches, no padding) the analytic mask carries causality alone.
#
# Interface: patches transformers.models.gemma4.modeling_gemma4
# .eager_attention_forward, same signature/contract (returns (out, None)).
#
# Geometry (Gemma4-12B): 16 Q heads / 8 KV heads (g=2), head_dim 256.
# Tiled cell fp32 at defaults (QC=4096, KT=8192): 8*2*4096*8192*4B = 1.07 GB.

import os
import torch
import torch.nn.functional as F

_QCHUNK = int(os.environ.get("CHONK_Q_CHUNK", "4096"))
_KTILE = int(os.environ.get("CHONK_ATTN_TILE", "8192"))


# Active KV-cache registry for sequence-chunked training (truncated BPTT).
# Module level (NOT class level): the trainer imports set_active_kv_cache
# directly. The dispatcher stashes (cache, layer_idx) in ctx so backward can
# source cached-span k/v values with ZERO extra retention.
# None = legacy full-sequence path (bit-identical behavior).
_ACTIVE_CACHE = None


def set_active_kv_cache(cache):
    global _ACTIVE_CACHE
    _ACTIVE_CACHE = cache


# ---------------------------------------------------------------------------
# Span cache
# ---------------------------------------------------------------------------
# Why this is NOT HF's DynamicCache: transformers' GradientCheckpointingLayer
# hard-nulls past_key_values while training (modeling_layers.py:94-97), and for
# good reason -- DynamicLayer.update is append-only (`torch.cat`, cache_utils.py),
# so it is NOT idempotent. Letting the checkpointed forward populate it would
# append every span twice (once under no_grad, once during recompute) and
# silently corrupt the cache. Un-nulling HF's cache is therefore not a fix.
#
# So we own the cache and commit it exactly once per span, OUTSIDE the
# checkpointed region:
#   1. trainer: begin_span()          -- clear staging
#   2. forward:  stage(idx, k, v)     -- idempotent overwrite each pass
#   3. trainer: commit_span(cache)    -- append once per layer per span
#
# The prefix is consumed DETACHED, so no gradient crosses a span boundary:
# that is the truncation, and it is what makes this truncated BPTT rather than
# full-sequence attention with extra steps.
#
# Sliding-window layers are cropped to their window on commit. Gemma4 is 40
# sliding (1024) + 8 full out of 48; without the crop a 32k document would
# retain full-length k/v for all 40 sliding layers.

class _LayerKV:
    __slots__ = ("keys", "values")

    def __init__(self):
        self.keys = None
        self.values = None


class SpanCache:
    """Cross-span KV state, one entry per layer. Deliberately not a
    transformers Cache subclass: HF indexes layers by cache-slot, collapsing
    shared-KV layers onto one slot, which does not line up with layer_idx."""

    def __init__(self, n_layers, layer_windows=None):
        self.layers = [_LayerKV() for _ in range(n_layers)]
        # 0 = full attention (grow unbounded); >0 = sliding window size.
        self.layer_windows = list(layer_windows or [0] * n_layers)

    def prefix_len(self, idx):
        if not (0 <= idx < len(self.layers)):
            return 0
        k = self.layers[idx].keys
        return 0 if k is None else int(k.shape[2])

    def get_prefix(self, idx):
        if not (0 <= idx < len(self.layers)):
            return None, None
        e = self.layers[idx]
        return e.keys, e.values

    def append(self, idx, k, v):
        if not (0 <= idx < len(self.layers)):
            return
        e = self.layers[idx]
        if e.keys is None:
            e.keys, e.values = k, v
        else:
            e.keys = torch.cat([e.keys, k], dim=2)
            e.values = torch.cat([e.values, v], dim=2)
        w = self.layer_windows[idx] if idx < len(self.layer_windows) else 0
        if w and w > 0 and e.keys.shape[2] > w:
            e.keys = e.keys[:, :, -w:].contiguous()
            e.values = e.values[:, :, -w:].contiguous()

    def reset(self):
        for e in self.layers:
            e.keys = None
            e.values = None

    def nbytes(self):
        tot = 0
        for e in self.layers:
            if e.keys is not None:
                tot += e.keys.numel() * e.keys.element_size()
                tot += e.values.numel() * e.values.element_size()
        return tot


# Per-span staging: layer_idx -> (k, v) detached. The dispatcher writes here on
# every pass (including checkpoint recompute, where it rewrites identical
# values, so it stays idempotent); the trainer commits once per span.
_SPAN_STAGING = {}

# Per-layer attention window, filled in by patch_gemma_attention_tiled().
_LAYER_WINDOWS = []


def make_span_cache(n_layers):
    """Build a SpanCache whose sliding-layer crops match the live patch."""
    return SpanCache(n_layers, _LAYER_WINDOWS)


def begin_span():
    _SPAN_STAGING.clear()


def commit_span(cache):
    n = 0
    for idx, (k, v) in _SPAN_STAGING.items():
        cache.append(idx, k, v)
        n += 1
    _SPAN_STAGING.clear()
    return n


class GemmaAttnTiled(torch.autograd.Function):
    """Tiled exact causal attention (training, full block, no cache).

    forward inputs:
        q       : [B, H, T, D]   (full block, requires grad)
        k, v    : [B, kvH, T, D] (full block, requires grad)
        scaling : float
        softcap : float (0.0 = none)
        window  : int (0 = full attention, else sliding window width)
    Backward recomputes p_tile exactly from stashed (m, l, out) and
    accumulates dq per q-chunk, dk/dv over all visited k-tiles.
    """

    @staticmethod
    def forward(ctx, q, k, v, scaling, softcap, window, off=0, cache=None,
                lidx=-1):
        B, H, Tq, D = q.shape
        Tk = k.shape[2]
        if off is None:
            off = Tk - Tq
        # off = cached prefix length (0 = legacy full-sequence: Tk == Tq).
        # q rows live at absolute positions [off, off+Tq); k/v span [0, Tk).
        kvH = k.shape[1]
        g = H // kvH
        QC, KT = _QCHUNK, _KTILE
        dev = q.device
        DEBUG = os.environ.get("CHONK_ATTN_DEBUG", "0") == "1"
        ctx.scaling = float(scaling)
        ctx.softcap = float(softcap)
        ctx.window = int(window)
        ctx.n_groups = int(g)
        ctx.qchunk, ctx.ktile = QC, KT
        ctx.off = int(off)
        ctx.Tk = int(Tk)
        ctx.cache = cache
        ctx.lidx = int(lidx)
        if cache is None:
            # Legacy path: save full tensors (zero behavior change).
            ctx.save_for_backward(q, k, v)
            ctx.split = False
        else:
            # Split saves (truncated-BPTT port): clone ONLY the current
            # chunk's k/v rows (small); the cached span is sourced from the
            # live cache in backward (zero extra retention -- same shape as
            # the vulkanvm_autograd split-path fix, Qwen 131K line).
            kc = k[:, :, off:off + Tq].clone()
            vc = v[:, :, off:off + Tq].clone()
            ctx.save_for_backward(q, kc, vc)
            ctx.split = True

        with torch.no_grad():
            qg_full = q.view(B, kvH, g, Tq, D)
            m = torch.full((B, kvH, g, Tq), float("-inf"),
                           device=dev, dtype=torch.float32)
            l = torch.zeros(B, kvH, g, Tq, device=dev, dtype=torch.float32)
            acc = torch.zeros(B, kvH, g, Tq, D, device=dev,
                              dtype=torch.float32)
            out = torch.empty(B, H, Tq, D, device=dev, dtype=q.dtype)

            n_qchunks = (Tq + QC - 1) // QC
            for cq in range(n_qchunks):
                q0, q1 = cq * QC, min((cq + 1) * QC, Tq)
                qc = q1 - q0
                qg = qg_full[:, :, :, q0:q1, :]          # view
                # k-tile range visible to this q-chunk (ABSOLUTE positions).
                # Causal: no tile starts at/after absolute q1 = off + q1.
                if ctx.window > 0:
                    k_lo = max(0, (off + q0 - ctx.window) // KT * KT)
                else:
                    k_lo = 0
                k_hi = min(Tk, off + q1)
                mq = m[:, :, :, q0:q1]
                lq = l[:, :, :, q0:q1]
                accq = acc[:, :, :, q0:q1, :]
                for k0 in range(k_lo, k_hi, KT):
                    k1 = min(k0 + KT, k_hi)
                    kt = k[:, :, k0:k1]                 # view
                    vt = v[:, :, k0:k1]
                    s = torch.matmul(
                        qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                    if ctx.softcap > 0:
                        s = torch.tanh(
                            s.float() / ctx.softcap) * ctx.softcap
                    s_f = s.float()
                    if DEBUG and not torch.isfinite(s_f).all():
                        print(f"[attn-debug] S_NANINF at "
                              f"cq[{q0}:{q1}] kt[{k0}:{k1}] win={ctx.window} "
                              f"qg_fin={bool(torch.isfinite(qg).all())} "
                              f"kt_fin={bool(torch.isfinite(kt).all())} "
                              f"smax={float(s_f.abs().max())} "
                              f"qmax={float(qg.abs().max())} "
                              f"kmax={float(kt.abs().max())} "
                              f"qstride={tuple(qg.stride())} "
                              f"kshape={tuple(kt.shape)} scaling={ctx.scaling}",
                              flush=True)
                    # analytic causal + window mask for this cell.
                    # qp ABSOLUTE (off-shifted); kp absolute. off=0 falls
                    # out to the legacy relative behavior bit-identically.
                    qp = torch.arange(q0, q1, device=dev).view(-1, 1) + off
                    kp = torch.arange(k0, k1, device=dev).view(1, -1)
                    ok = kp <= qp
                    if ctx.window > 0:
                        ok = ok & ((qp - kp) < ctx.window)
                    s_f = s_f.masked_fill(~ok, float("-inf"))
                    m_new = torch.maximum(mq, s_f.amax(dim=-1))
                    # NaN guard: a row fully masked in this tile AND empty so
                    # far has mq == m_new == -inf, and exp(-inf - -inf) is NaN
                    # (fires on sliding layers once rows outrun the first tile
                    # by more than the window). An empty row must keep prior
                    # state (corr=1) and contribute zero mass (p_t=0).
                    valid = torch.isfinite(m_new)
                    corr = torch.exp(torch.where(
                        valid, mq - m_new, torch.zeros_like(mq)))
                    p_t = torch.exp(s_f - m_new.unsqueeze(-1))
                    p_t = torch.where(valid.unsqueeze(-1), p_t,
                                      torch.zeros_like(p_t))
                    if DEBUG:
                        tag = (f"cq[{q0}:{q1}] kt[{k0}:{k1}] win={ctx.window}")
                        for nm, t in (("m_new", m_new), ("corr", corr),
                                      ("p_t", p_t)):
                            if not torch.isfinite(t).all():
                                print(f"[attn-debug] non-finite {nm} at "
                                      f"{tag} nbad={int((~torch.isfinite(t)).sum())}",
                                      flush=True)
                    lq = lq * corr + p_t.sum(dim=-1)
                    accq = accq * corr.unsqueeze(-1) + \
                        torch.matmul(p_t.to(vt.dtype),
                                     vt.unsqueeze(2)).float()
                    mq = m_new
                out[:, :, q0:q1, :] = (
                    accq / lq.unsqueeze(-1)).to(q.dtype).reshape(B, H, qc, D)
                m[:, :, :, q0:q1] = mq
                l[:, :, :, q0:q1] = lq
                acc[:, :, :, q0:q1, :] = accq

        # Ratchet-proof stash (see Granite module header): out DETACHED.
        ctx.m_final = m
        ctx.l_final = l
        ctx.out = out.detach()
        return out

    @staticmethod
    def backward(ctx, dout):
        q, ks, vs = ctx.saved_tensors
        m = ctx.m_final
        l = ctx.l_final
        out = ctx.out
        B, H, Tq, D = q.shape
        kvH = ks.shape[1]
        g = ctx.n_groups
        QC, KT = ctx.qchunk, ctx.ktile
        dev, qdt = q.device, q.dtype
        if ctx.split:
            # Split saves: ks/vs are the current chunk's k/v clones;
            # cached-span values come from the live cache (zero extra
            # retention -- the cache outlives every chunk regardless).
            # Prefix rows are append-only: values read now equal chunk-time.
            off, Tk = ctx.off, ctx.Tk
            # Bind the current chunk's k/v FIRST and unconditionally. The
            # tile loop below indexes kc/vc for every tile at or past `off`,
            # so when the cache actually populated (len(layers) > 0) this
            # branch previously set only ck/cv and left kc/vc unbound --
            # a NameError on the first cached tile. It stayed hidden while
            # the cache was always empty and every call took the miss path.
            kc, vc = ks, vs
            ck = cv = None
            if ctx.cache is not None:
                _ll = (ctx.cache.layers[ctx.lidx]
                       if ctx.lidx < len(ctx.cache.layers) else None)
                if _ll is not None and _ll.keys is not None and off > 0:
                    ck = _ll.keys[:, :, :off].detach()
                    cv = _ll.values[:, :, :off].detach()
            if ck is None:
                # Shared-KV layer (no dedicated cache slot) or cache miss:
                # no usable prefix, so restrict to the current slice only.
                # off=0 makes every tile take the `k0 >= off` branch below.
                off = 0
                Tk = ks.shape[2]
        else:
            # Legacy path: saved full tensors (identical to old code).
            off, Tk = 0, q.shape[2]
            ck = cv = kc = vc = None
            k, v = ks, vs

        qg_full = q.view(B, kvH, g, Tq, D)
        dyg = dout.view(B, kvH, g, Tq, D)
        delta = (dyg.float() *
                 out.view(B, kvH, g, Tq, D).float()).sum(dim=-1)
        inv_l = (1.0 / l).unsqueeze(-1)

        dq = torch.zeros(B, kvH, g, Tq, D, device=dev, dtype=torch.float32)
        dk = torch.zeros(B, kvH, Tk, D, device=dev, dtype=torch.float32)
        dv = torch.zeros(B, kvH, Tk, D, device=dev, dtype=torch.float32)

        n_qchunks = (Tq + QC - 1) // QC
        for cq in range(n_qchunks):
            q0, q1 = cq * QC, min((cq + 1) * QC, Tq)
            qg = qg_full[:, :, :, q0:q1, :]
            dyg_c = dyg[:, :, :, q0:q1, :]
            delta_c = delta[:, :, :, q0:q1]
            mq = m[:, :, :, q0:q1]
            inv_c = inv_l[:, :, :, q0:q1, :]
            dq_c = torch.zeros(B, kvH, g, q1 - q0, D, device=dev,
                               dtype=torch.float32)
            if ctx.window > 0:
                k_lo = max(0, (off + q0 - ctx.window) // KT * KT)
            else:
                k_lo = 0
            k_hi = min(Tk, off + q1)
            for k0 in range(k_lo, k_hi, KT):
                k1 = min(k0 + KT, k_hi)
                if not ctx.split:
                    kt = k[:, :, k0:k1]
                    vt = v[:, :, k0:k1]
                elif k1 <= off:
                    kt = ck[:, :, k0:k1]
                    vt = cv[:, :, k0:k1]
                elif k0 >= off:
                    kt = kc[:, :, k0 - off:k1 - off]
                    vt = vc[:, :, k0 - off:k1 - off]
                else:
                    # Straddling tile (off not KT-aligned): one small
                    # transient cat per occurrence, freed per iteration.
                    kt = torch.cat([ck[:, :, k0:off],
                                    kc[:, :, 0:k1 - off]], dim=2)
                    vt = torch.cat([cv[:, :, k0:off],
                                    vc[:, :, 0:k1 - off]], dim=2)
                s = torch.matmul(
                    qg, kt.unsqueeze(2).transpose(-2, -1)) * ctx.scaling
                if ctx.softcap > 0:
                    s = torch.tanh(s.float() / ctx.softcap) * ctx.softcap
                s_f = s.float()
                qp = torch.arange(q0, q1, device=dev).view(-1, 1) + off
                kp = torch.arange(k0, k1, device=dev).view(1, -1)
                ok = kp <= qp
                if ctx.window > 0:
                    ok = ok & ((qp - kp) < ctx.window)
                # masked tiles recompute the same -inf cells; they
                # contribute exp(-inf - m) = 0 to p_t, so plain masking
                # of s_f is exact (matches forward cell math bit-wise).
                s_f = s_f.masked_fill(~ok, float("-inf"))
                p_t = torch.exp(s_f - mq.unsqueeze(-1)) * inv_c
                dp_t = torch.matmul(
                    dyg_c, vt.unsqueeze(2).transpose(-2, -1)).float()
                ds_t = p_t * (dp_t - delta_c.unsqueeze(-1))
                ds_bf = ds_t.to(qdt)
                dq_c += torch.matmul(
                    ds_bf, kt.unsqueeze(2)).float() * ctx.scaling
                dk[:, :, k0:k1, :] += torch.matmul(
                    ds_bf.transpose(-2, -1), qg).sum(dim=2).float() * \
                    ctx.scaling
                dv[:, :, k0:k1, :] += torch.matmul(
                    p_t.to(qdt).transpose(-2, -1), dyg_c).sum(dim=2).float()
            dq[:, :, :, q0:q1, :] = dq_c

        return (
            dq.to(qdt).reshape(B, H, Tq, D),
            dk.to(qdt),
            dv.to(vs.dtype),
            None, None, None, None, None, None,
        )


def patch_gemma_attention_tiled(model):
    """Swap Gemma4's eager attention for the Chonk tiled path.

    Same contract as the Granite patch: called where HF would call
    eager_attention_forward (module-first signature). Reads per-layer
    geometry from the module + enclosing config, applies the optional
    HF-built attention_mask slice on top of the analytic cell mask.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    txt = base.model if hasattr(base, "model") else base
    layers = list(txt.layers)
    id2idx = {id(l.self_attn): i for i, l in enumerate(layers)}
    cfg = txt.config if hasattr(txt, "config") else base.config
    layer_types = list(getattr(cfg, "layer_types", ["full_attention"] * 48))
    sw = int(getattr(cfg, "sliding_window", 0) or 0)
    # Publish the per-layer windows so the trainer can build a SpanCache with
    # matching crop behaviour instead of re-parsing the config.
    _LAYER_WINDOWS[:] = [sw if str(t).lower().startswith("sliding") else 0
                          for t in layer_types]

    try:
        import transformers.models.gemma4.modeling_gemma4 as gm
    except ImportError:
        import modeling_gemma4 as gm
    orig = gm.eager_attention_forward

    # Dispatch lever: modeling_gemma4 resolves the interface FRESH every
    # forward via ALL_ATTENTION_FUNCTIONS.get(impl, <module-global default>).
    # The default is looked up in the module namespace at CALL time, so
    # replacing the global takes effect -- but ONLY when impl == "eager"
    # (the model ships "sdpa"). NOTE: per-instance rebinding
    # (layer.self_attn.attention_interface = ...) is DEAD CODE here -- the
    # forward never reads self.attention_interface, unlike older HF models.
    base_cfg = getattr(txt, "config", getattr(base, "config", None))
    if base_cfg is not None:
        base_cfg._attn_implementation = "eager"
    assert getattr(base_cfg, "_attn_implementation", None) == "eager", \
        "could not force eager dispatch"

    def patched(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs):
        if dropout and dropout > 0:
            return orig(module, query, key, value, attention_mask, scaling,
                        dropout=dropout, **kwargs)
        idx = id2idx.get(id(module), -1)
        lt = layer_types[idx] if 0 <= idx < len(layer_types) \
            else "full_attention"
        window = sw if lt == "sliding_attention" else 0
        g = int(getattr(module, "num_key_value_groups",
                        query.shape[1] // key.shape[1]))
        softcap = float(kwargs.get("softcap", 0.0) or 0.0)
        # Cross-span path. HF hands us ONLY the current span's k/v (its cache
        # is nulled under gradient checkpointing), so we prepend the
        # accumulated prefix here. The prefix is DETACHED: gradient does not
        # cross the span boundary, which is precisely the truncation. The
        # backward pass re-reads the same prefix straight from the cache by
        # slice, so it stays correct even as the cache keeps growing.
        cache = _ACTIVE_CACHE
        off = 0
        fk, fv = key, value
        if cache is not None and 0 <= idx < len(cache.layers):
            pk, pv = cache.get_prefix(idx)
            if pk is not None and pk.shape[2] > 0:
                off = int(pk.shape[2])
                fk = torch.cat([pk.detach(), key], dim=2)
                fv = torch.cat([pv.detach(), value], dim=2)
            # Stage this span's own k/v for the trainer to commit AFTER the
            # span returns -- never inside the checkpointed region.
            _SPAN_STAGING[idx] = (key.detach(), value.detach())
        out = GemmaAttnTiled.apply(query, fk, fv, float(scaling),
                                   softcap, window, off, cache, idx)
        return out, None

    gm.eager_attention_forward = patched
    print(f"[+] Gemma4 eager attention -> Chonk tiled "
          f"(QC={_QCHUNK} KT={_KTILE}, dispatch=eager, "
          f"layers={len(layers)})",
          flush=True)
    return patched
