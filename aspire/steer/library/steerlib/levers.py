"""Lever discovery by MINIMAL SUFFICIENT INTERVENTION (operator's discovery-method list): find the smallest set of MLP neurons
whose added activations turn the frozen VLA's predicted chunks into target chunks. The support of the sparse solution is the
discovered lever; its values are the dose. Gradient through the 10 flow steps (fixed noise), proximal L1 (soft-thresholding
after every Adam step) so neurons are exactly zero unless needed.

    from steerlib import levers
    sol = levers.sparse_neurons(lab, examples, where="expert", layers=[2, 4, 6], l1=1e-3, steps=150)
    sol["support"]  -> [(layer, neuron, value), ...] sorted by |value| * ||W_down[:, n]||
    with levers.apply(lab.policy, sol, "expert", scale=1.0): chunk = lab.chunk(obs, instruction)

examples = [(obs, instruction, target (50, 7) normalised chunk)] (steer.lab.normalised_chunk gives the policy's own).
Activations are added at all tokens of the site (action tokens for the expert, every prefix token for the VLM) unless `tokens`.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch

from steer.hooks import layers


def _down(policy, where, layer):
    return layers(policy, where)[layer].mlp.down_proj


@contextmanager
def _added(policy, where, acts: dict, tokens=None):
    hs = []
    for L, a in acts.items():
        def pre(mod, inp, a=a):
            x = inp[0]
            add = torch.zeros_like(x)
            sl = tokens if tokens is not None else slice(None)
            add[:, sl] = a.to(x.dtype)
            return (x + add,)
        hs.append(_down(policy, where, L).register_forward_pre_hook(pre))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def sparse_neurons(lab, examples, where: str, layers_: list[int], l1: float = 1e-3, steps: int = 150, lr: float = 0.05,
                   tokens=None, seed: int = 0, log_every: int = 0) -> dict:
    p = lab.policy
    dev = next(p.policy.parameters()).device
    for prm in p.policy.parameters():
        prm.requires_grad_(False)
    width = _down(p, where, layers_[0]).in_features
    acts = {L: torch.zeros(width, device=dev, requires_grad=True) for L in layers_}
    # per-neuron output norms: penalise the residual change, not the raw activation (neurons differ in ||W_down[:, n]||)
    wn = {L: _down(p, where, L).weight.float().norm(dim=0).detach() for L in layers_}
    opt = torch.optim.Adam(list(acts.values()), lr=lr)
    batches = [(p.policy._prepare_batch(p.pre(p.batch(o, ins))), torch.as_tensor(t, dtype=torch.float32, device=dev))
               for o, ins, t in examples]
    hist = []
    with _added(p, where, acts, tokens):
        for step in range(steps):
            opt.zero_grad()
            loss = 0.0
            for batch, target in batches:            # one backward per example (memory independent of the number of examples)
                torch.manual_seed(seed)
                with torch.enable_grad():
                    pred = p.policy._get_action_chunk(batch, None)[0]
                    li = ((pred[: len(target)] - target) ** 2).mean() / len(batches)
                    li.backward()
                loss += float(li)
            opt.step()
            with torch.no_grad():                                   # proximal L1 step on the residual-space size
                for L, a in acts.items():
                    thr = lr * l1 / (wn[L] + 1e-8) * wn[L].mean()
                    a.copy_(torch.sign(a) * torch.clamp(a.abs() - thr, min=0))
            nnz = sum(int((a != 0).sum()) for a in acts.values())
            hist.append((float(loss), nnz))
            if log_every and step % log_every == 0:
                print(f"  step {step} loss {float(loss):.4f} nonzero {nnz}", flush=True)
    support = []
    for L, a in acts.items():
        a = a.detach()
        for n in torch.nonzero(a).flatten().tolist():
            support.append((L, n, float(a[n]), float(a[n].abs() * wn[L][n])))
    support.sort(key=lambda r: -r[3])
    return {"where": where, "layers": layers_, "acts": {L: a.detach().cpu() for L, a in acts.items()},
            "support": [(L, n, v) for L, n, v, _ in support], "history": hist, "tokens": tokens}


@contextmanager
def apply(policy, sol: dict, where: str | None = None, scale: float = 1.0, keep: int | None = None):
    """Apply a solution (optionally only its `keep` strongest neurons) for the block."""
    where = where or sol["where"]
    if keep is None:
        acts = {L: a * scale for L, a in sol["acts"].items()}
    else:
        acts = {L: torch.zeros_like(a) for L, a in sol["acts"].items()}
        for L, n, v in sol["support"][:keep]:
            acts[L][n] = v * scale
    dev = next(policy.policy.parameters()).device
    with _added(policy, where, {L: a.to(dev) for L, a in acts.items()}, sol.get("tokens")):
        yield


def install(ctx, sol: dict, scale: float = 1.0, keep: int | None = None):
    """Program version: register the additions for the episode (ctx.hook, removed afterwards). Returns a dict whose 'scale'
    entry the program can change per chunk (state-dependent dose; 0 switches the lever off)."""
    state = {"scale": scale}
    acts = {L: a for L, a in sol["acts"].items()}
    if keep is not None:
        acts = {L: torch.zeros_like(a) for L, a in sol["acts"].items()}
        for L, n, v in sol["support"][:keep]:
            acts[L][n] = v
    stack = ctx.expert_layers if sol["where"] == "expert" else ctx.vlm_layers
    sl = sol.get("tokens") if sol.get("tokens") is not None else slice(None)
    for L, a in acts.items():
        def pre(mod, inp, a=a):
            x = inp[0]
            if state["scale"] == 0:
                return None
            add = torch.zeros_like(x)
            add[:, sl] = (a.to(x.device, x.dtype) * state["scale"])
            return (x + add,)
        ctx.hook(stack[L].mlp.down_proj, pre, pre=True)
    return state
