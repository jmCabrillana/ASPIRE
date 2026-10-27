"""Attention interventions on SmolVLA: add a bias to attention logits for chosen query/key ranges at chosen layers.

SmolVLA computes all attention in `vlm_with_expert.eager_attention_forward(mask, batch, head_dim, q, k, v)`; layers are called
through `forward_attn_layer` / `forward_cross_attn_layer(model_layers, inputs_embeds, layer_idx, ...)`. We wrap both to know the
layer index, and replace the attention function by one that adds `bias_fn(layer, q_len, k_len)` (a (q_len, k_len) or
(heads, q_len, k_len) tensor, or None) to the logits before the mask and softmax. Expert calls have q_len == chunk size (50):
their keys are [prefix (image 0:64, wrist 64:128, lang, state) | action tokens (self-attention layers only)].

    restore = patch(policy.policy.model.vlm_with_expert, bias_fn)      # offline (lab)
    ...; restore()
    In a program: patch_ctx(ctx, bias_fn) in setup(ctx) (restored after the episode by ctx.patch).
"""

from __future__ import annotations

import torch
from torch import nn


def _make(m, bias_fn, state):
    def attention(attention_mask, batch_size, head_dim, query_states, key_states, value_states):
        H, G = m.num_attention_heads, m.num_key_value_heads
        S = key_states.shape[1]
        k = key_states[:, :, :, None, :].expand(batch_size, S, G, H // G, head_dim).reshape(batch_size, S, H, head_dim)
        v = value_states[:, :, :, None, :].expand(batch_size, S, G, H // G, head_dim).reshape(batch_size, S, H, head_dim)
        q = query_states.to(torch.float32).transpose(1, 2)
        att = torch.matmul(q, k.to(torch.float32).transpose(1, 2).transpose(2, 3)) * head_dim ** -0.5
        b = bias_fn(state["layer"], q.shape[2], S)
        if b is not None:
            att = att + b.to(att.device, att.dtype)
        att = torch.where(attention_mask[:, None, :, :], att, torch.finfo(att.dtype).min)
        probs = nn.functional.softmax(att, dim=-1)
        if state.get("record") is not None:
            state["record"].append((state["layer"], probs[0].detach().float().cpu()))
        probs = probs.to(v.dtype)
        out = torch.matmul(probs, v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return out.reshape(batch_size, -1, H * head_dim)
    return attention


def _layer_wrappers(m, state):
    out = {}
    for name in ("forward_attn_layer", "forward_cross_attn_layer"):
        original = getattr(m, name)

        def wrapped(model_layers, inputs_embeds, layer_idx, *args, _o=original, **kwargs):
            state["layer"] = layer_idx
            return _o(model_layers, inputs_embeds, layer_idx, *args, **kwargs)
        out[name] = wrapped
    return out


def patch(m, bias_fn=lambda *a: None, record: list | None = None):
    """Offline patch on vlm_with_expert `m`; returns a restore() function. record: a list that receives (layer, probs (heads,
    q, k)) for every attention call (prefix pass first, then the expert's denoising passes)."""
    state = {"layer": -1, "record": record}
    saved = {n: m.__dict__.get(n) for n in ("forward_attn_layer", "forward_cross_attn_layer", "eager_attention_forward")}
    for n, f in _layer_wrappers(m, state).items():
        setattr(m, n, f)
    m.eager_attention_forward = _make(m, bias_fn, state)

    def restore():
        for n, f in saved.items():
            if f is None:
                m.__dict__.pop(n, None)
            else:
                setattr(m, n, f)
    return restore


def patch_ctx(ctx, bias_fn):
    """Same, inside a program (undone after the episode)."""
    m = ctx.vla.model.vlm_with_expert
    state = {"layer": -1}
    for n, f in _layer_wrappers(m, state).items():
        ctx.patch(m, n, f)
    ctx.patch(m, "eager_attention_forward", _make(m, bias_fn, state))


def segment_bias(segments: dict, amount: float, layers=None, queries: str = "expert", chunk: int = 50):
    """bias_fn adding `amount` to the logits of keys in `segments` ({name: slice}) for expert queries (q_len == chunk) or
    'prefix' queries (q_len != chunk), at `layers` (None = all)."""
    def fn(layer, q_len, k_len):
        if layers is not None and layer not in layers:
            return None
        if (queries == "expert") != (q_len == chunk):
            return None
        b = torch.zeros(q_len, k_len)
        for s in segments.values():
            b[:, s] += amount
        return b
    return fn
