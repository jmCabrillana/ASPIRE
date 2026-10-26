"""Short LoRA adapters on the frozen SmolVLA, trained on saved episodes (scripts/run_program.py --save-obs), loaded as separate
modules and removed after use, so the base weights never change.

    lora = attach(policy, targets=("q_proj", "v_proj"), where="expert", rank=8)     # adds adapter layers (zero-initialised)
    train(policy, lora, episodes=[Lab.load(f) for f in files if success], steps=200)
    save(lora, "library/adapters/goal_lan_7.pt"); ...; detach(policy, lora)            # base model restored exactly
In a program: lora = attach(ctx.policy, ...); load(lora, path) in setup(ctx), detach(ctx.policy, lora) in on_end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from steer.hooks import layers


class LoRALinear(nn.Module):
    """y = W x + (alpha / r) B A x, with B initialised to zero (the adapter starts as the identity change)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    @property
    def weight(self):                       # SmolVLA reads q_proj.weight.dtype
        return self.base.weight

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + self.scale * ((x.to(self.A.dtype) @ self.A.T) @ self.B.T).to(x.dtype)


@dataclass
class Adapter:
    slots: list = field(default_factory=list)          # (parent module, attribute name, LoRALinear)

    def parameters(self):
        for _, _, m in self.slots:
            yield m.A
            yield m.B


def attach(policy, targets=("q_proj", "v_proj"), where: str = "expert", rank: int = 8, alpha: float = 16.0,
           which: list[int] | None = None) -> Adapter:
    """Wrap the named linear layers of the chosen site's attention/MLP modules with LoRA."""
    ad = Adapter()
    for i, layer in enumerate(layers(policy, where)):
        if which is not None and i not in which:
            continue
        for parent in (layer.self_attn, layer.mlp):
            for name in targets:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    wrapped = LoRALinear(lin, rank, alpha)
                    setattr(parent, name, wrapped)
                    ad.slots.append((parent, name, wrapped))
    return ad


def detach(policy, ad: Adapter) -> None:
    for parent, name, wrapped in ad.slots:
        setattr(parent, name, wrapped.base)
    ad.slots = []


def save(ad: Adapter, path) -> None:
    torch.save([(m.A.detach().cpu(), m.B.detach().cpu()) for _, _, m in ad.slots], path)


def load(ad: Adapter, path) -> None:
    for (_, _, m), (a, b) in zip(ad.slots, torch.load(path)):
        m.A.data.copy_(a.to(m.A.device))
        m.B.data.copy_(b.to(m.B.device))


def examples(policy, episodes, horizon: int = 50):
    """(raw batch with "action" and "action_is_pad") for every chunk start of the episodes."""
    for ep in episodes:
        for i in range(len(ep)):
            acts = torch.from_numpy(ep.actions(i, horizon)).float()
            pad = horizon - len(acts)
            if len(acts) == 0:
                continue
            batch = policy.batch(ep.obs(i), ep.instruction)
            batch["action"] = torch.cat([acts, acts[-1:].repeat(pad, 1)]).unsqueeze(0)
            batch["action_is_pad"] = torch.tensor([[False] * len(acts) + [True] * pad])
            yield batch


def train(policy, ad: Adapter, episodes, steps: int = 200, lr: float = 1e-4, log_every: int = 50) -> list[float]:
    """Flow-matching loss of SmolVLA on the episodes' (observation, executed action chunk) pairs; only the adapter learns."""
    data = list(examples(policy, episodes))
    if not data:
        raise ValueError("no training examples")
    vla = policy.policy
    for p in vla.parameters():
        p.requires_grad_(False)
    params = list(ad.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    vla.train()
    try:
        for step in range(steps):
            batch = policy.pre(dict(data[step % len(data)]))
            loss, _ = vla.forward(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if log_every and (step + 1) % log_every == 0:
                print(f"lora step {step + 1}: loss {sum(losses[-log_every:]) / log_every:.4f}", flush=True)
    finally:
        vla.eval()
        for p in params:
            p.requires_grad_(False)
    return losses
