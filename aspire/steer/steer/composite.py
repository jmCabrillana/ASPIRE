"""Composite ("Frankenstein") models: a saved bundle of edits that a program loads as a whole for an episode.

A composite is stored as library/composites/<name>.pt:
  {"deltas": {param_name: tensor},               weight deltas (rank-one edits, transplants from another model, merged adapters)
   "adapters": [{"path", "where", "targets", "rank", "which"}],   LoRA adapters to attach (steer/lora.py format)
   "steering": [(where, layer, vector, strength)],                 constant additive steering terms
   "notes": str}
Programs apply it in setup(ctx) and everything is undone after the episode (weights restored, adapters detached), so the frozen
model is unchanged between episodes. Any combination is allowed: edits found in different experiments, a base-model transplant,
several adapters and steering vectors, merged into one model for the episode.

    comp = Composite.capture(policy, frozen_state, notes="...")   # diff of the current (edited) model against the frozen weights
    comp.save("library/composites/x.pt")
    # in a program:
    def setup(ctx): ctx.memory["comp"] = Composite.load("library/composites/x.pt").apply(ctx)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from steer import lora


@dataclass
class Composite:
    deltas: dict[str, torch.Tensor] = field(default_factory=dict)
    adapters: list[dict] = field(default_factory=list)
    steering: list[tuple] = field(default_factory=list)
    notes: str = ""

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"deltas": {k: v.cpu() for k, v in self.deltas.items()}, "adapters": self.adapters,
                    "steering": [(w, l, v.cpu(), s) for w, l, v, s in self.steering], "notes": self.notes}, path)

    @classmethod
    def load(cls, path) -> "Composite":
        d = torch.load(path)
        return cls(d.get("deltas", {}), d.get("adapters", []), d.get("steering", []), d.get("notes", ""))

    @classmethod
    def capture(cls, policy, frozen: dict[str, torch.Tensor], notes: str = "", tol: float = 0.0) -> "Composite":
        """Weight deltas of the policy's current parameters against a saved frozen state_dict (only changed tensors)."""
        deltas = {}
        for name, p in policy.policy.named_parameters():
            if name in frozen:
                d = p.detach().float().cpu() - frozen[name].float()
                if d.abs().max() > tol:
                    deltas[name] = d
        return cls(deltas=deltas, notes=notes)

    def apply(self, ctx):
        """Apply everything through ctx (restored after the episode). Returns the attached adapters (to switch or inspect)."""
        params = dict(ctx.vla.named_parameters())
        for name, d in self.deltas.items():
            ctx.edit_weights(params[name], lambda w, d=d: w + d.to(w.device, w.dtype))
        attached = []
        for a in self.adapters:
            ad = lora.attach(ctx.policy, targets=tuple(a.get("targets", ("q_proj", "v_proj"))), where=a.get("where", "expert"),
                             rank=a.get("rank", 8), which=a.get("which"))
            lora.load(ad, a["path"])
            attached.append(ad)
        if attached:
            ctx.memory.setdefault("_composite_adapters", []).extend(attached)
            ctx._patches.append((_Detacher(ctx.policy, attached), "noop", None))
        if self.steering:
            ctx.steering.set(list(self.steering))
        return attached


class _Detacher:
    """Detaches adapters when Context.close() restores its patches (setattr on this object triggers the detach)."""

    def __init__(self, policy, adapters):
        self.policy, self.adapters = policy, adapters

    def __setattr__(self, name, value):
        if name == "noop":
            for ad in object.__getattribute__(self, "adapters"):
                lora.detach(object.__getattribute__(self, "policy"), ad)
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self.__setattr__(name, None)
