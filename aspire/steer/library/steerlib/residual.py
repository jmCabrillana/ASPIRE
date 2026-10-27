"""Residual-stream recording and patching for the VLM layers of SmolVLA (prefix pass; the expert then reads the KV cache).

Each VLM layer computes  h_mid = h + attn(h);  h_out = h_mid + mlp(post_attention_layernorm(h_mid)).
record(): h_out of chosen layers, (177, 960) per layer, for one prefix pass.
patch():  set h_out[positions] := target at chosen layers (MLP output replaced by target - h_mid at those positions), so every
          later layer and the KV cache the action expert reads see the patched residual. Works offline (lab) and in programs
          (pass ctx to register through ctx.hook, removed after the episode).
add():    add vectors to h_out at positions (steering instead of replacement).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch


def _layers(vla):
    return vla.model.vlm_with_expert.get_vlm_model().text_model.layers


@contextmanager
def record(vla, which):
    out, mid, hs = {}, {}, []
    for L in which:
        layer = _layers(vla)[L]
        hs.append(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a, L=L: mid.__setitem__(L, a[0].detach())))
        hs.append(layer.mlp.register_forward_hook(
            lambda m, a, o, L=L: out.__setitem__(L, (mid[L] + o)[0].detach().float().cpu()) if o.shape[1] > 100 else None))
    try:
        yield out
    finally:
        for h in hs:
            h.remove()


def _install(vla, spec, mode, ctx=None):
    """spec: {layer: (positions (list/LongTensor), vectors (n, 960) tensor)}"""
    mid, hs = {}, []
    for L, (pos, vec) in spec.items():
        layer = _layers(vla)[L]
        pos_t = torch.as_tensor(pos, dtype=torch.long)

        def pre(m, a, L=L):
            mid[L] = a[0]

        def post(m, a, o, L=L, pos_t=pos_t, vec=vec):
            if o.shape[1] <= 100:                    # not the prefix pass
                return o
            o = o.clone()
            p = pos_t.to(o.device)
            v = vec.to(o.device, o.dtype)
            if mode == "replace":
                o[0, p] = v - mid[L][0, p]
            else:
                o[0, p] = o[0, p] + v
            return o

        if ctx is not None:
            hs.append(ctx.hook(layer.post_attention_layernorm, lambda m, a, f=pre: f(m, a), pre=True))
            hs.append(ctx.hook(layer.mlp, post))
        else:
            hs.append(layer.post_attention_layernorm.register_forward_pre_hook(pre))
            hs.append(layer.mlp.register_forward_hook(post))
    return hs


@contextmanager
def patch(vla, spec):
    hs = _install(vla, spec, "replace")
    try:
        yield
    finally:
        for h in hs:
            h.remove()


@contextmanager
def add(vla, spec):
    hs = _install(vla, spec, "add")
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def install(ctx, spec, mode="replace"):
    """Program version (hooks removed after the episode by ctx)."""
    return _install(ctx.vla, spec, mode, ctx=ctx)
