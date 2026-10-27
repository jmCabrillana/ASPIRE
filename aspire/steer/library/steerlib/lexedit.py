"""Lexical weight edits: teach the VLM that an unknown word means a known one, inside the network (no text rewrite at run time).

E15: the action expert reads a verb through the verb token's own KV entries at every VLM layer; the rest of the sentence does
not carry it. Patching the verb token's residual right after layer 0 with the rewrite's residual restores ~96% of the rewrite's
behaviour. So: rank-one / multi-pair edit (steer.edit) of layer L's MLP down projection such that, for the unknown token's key
(its neuron activations in context), the layer writes  v* = h_out(rewrite)[pos] - h_mid(paraphrase)[pos], i.e. the residual
leaving layer L becomes the known word's residual. Keys/values from a few observations (sample efficiency = n observations).
The mapping unknown -> known word comes from a general rule (steerlib.canon), never from the target's original wording.
"""
from __future__ import annotations

import difflib

import torch

from steer import edit as E
from steerlib import analysis as A


def _layer(vla, L):
    return vla.model.vlm_with_expert.get_vlm_model().text_model.layers[L]


def align(x: str, c: str):
    """Token positions of x whose token differs from c, paired with c's positions (equal-length replace blocks only)."""
    tx, tc = A.tokens(x), A.tokens(c)
    pairs = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=tx, b=tc, autojunk=False).get_opcodes():
        if op == "replace":
            n = min(i2 - i1, j2 - j1)
            pairs += [(128 + i1 + k, 128 + j1 + k) for k in range(n)]
            pairs += [(128 + i1 + n + k, 128 + j2 - 1) for k in range(i2 - i1 - n)]     # extra x tokens -> last c token
        elif op == "delete":
            pairs += [(128 + i1 + k, 128 + max(j1 - 1, 0) if j1 > 0 else 128) for k in range(i2 - i1)]
    return pairs


@torch.no_grad()
def collect(lab, obs_list, x, c, L):
    """Keys (n, 2560) of x's differing tokens at layer L and target values (n, 960) for the down projection."""
    vla = lab.policy.policy
    layer = _layer(vla, L)
    pairs = align(x, c)
    K, V = [], []
    for o in obs_list:
        got = {}
        h1 = layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: got.__setitem__("mid", a[0][0].detach().float()))
        h2 = layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: got.__setitem__("key", a[0][0].detach().float()))
        h3 = layer.mlp.register_forward_hook(lambda m, a, out: got.__setitem__("out", out[0].detach().float()))
        try:
            lab.chunk(o, c)
            out_c = (got["mid"] + got["out"]).clone()
            lab.chunk(o, x)
        finally:
            h1.remove(); h2.remove(); h3.remove()
        for px, pc in pairs:
            K.append(got["key"][px]); V.append(out_c[pc] - got["mid"][px])
    return torch.stack(K), torch.stack(V)


def fit(lab, obs_list, x, c, L=0, C=None, ridge=1e-3):
    """Delta for layer L's down_proj.weight (out, in) from len(obs_list) observations."""
    K, V = collect(lab, obs_list, x, c, L)
    W = _layer(lab.policy.policy, L).mlp.down_proj.weight
    return E.multi(W, K.T, V.T, C=C, ridge=ridge)


def apply_ctx(ctx, delta, L=0):
    """In a program: add the delta for the episode (restored by ctx)."""
    w = _layer(ctx.vla, L).mlp.down_proj.weight
    ctx.edit_weights(w, lambda W: W + delta.to(W.device, W.dtype))


class applied:
    """Offline context manager: add the delta, restore afterwards."""
    def __init__(self, vla, delta, L=0):
        self.w, self.delta = _layer(vla, L).mlp.down_proj.weight, delta

    def __enter__(self):
        self.saved = self.w.detach().clone()
        self.w.data += self.delta.to(self.w.device, self.w.dtype)

    def __exit__(self, *a):
        self.w.data.copy_(self.saved)
