"""Activation steering and recording for SmolVLA.

SmolVLA runs its layers in a custom loop (smolvlm_with_expert.py), so whole-layer hooks never fire. Each layer's MLP is called as a
module, and its output is added to the residual stream, so a hook on `layer.mlp` can read or shift the residual stream of that layer:

    h_l  <-  h_l + strength * v          (v: unit direction in the layer's hidden space)

Two places, with different timing:
- "vlm": the VLM text layers. They run once per action chunk, on the prefix (images, instruction, state); later denoising steps
  read their cached keys and values. Steering here changes what the action expert attends to.
- "expert": the action-expert layers. They run at every flow-matching denoising step of every chunk.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def layers(policy, where: str):
    m = policy.policy.model.vlm_with_expert
    if where == "vlm":
        return m.get_vlm_model().text_model.layers
    if where == "expert":
        return m.lm_expert.layers
    raise ValueError(f"where must be 'vlm' or 'expert', not {where!r}")


def width(policy, where: str) -> int:
    return layers(policy, where)[0].mlp.down_proj.out_features


class Steering:
    """Additive steering vectors on MLP outputs. set() replaces the active set; clear() removes it."""

    def __init__(self, policy):
        self.policy = policy
        self.active: dict[tuple[str, int], torch.Tensor] = {}
        self.handles = []
        for where in ("vlm", "expert"):
            for i, layer in enumerate(layers(policy, where)):
                self.handles.append(layer.mlp.register_forward_hook(self._hook(where, i)))

    def _hook(self, where: str, i: int):
        def fn(module, inputs, output):
            v = self.active.get((where, i))
            return output if v is None else output + v.to(output.dtype)
        return fn

    def set(self, terms: list[tuple[str, int, torch.Tensor, float]]) -> None:
        """terms: (where, layer, unit direction, strength). Several terms on one layer add up."""
        active: dict[tuple[str, int], torch.Tensor] = {}
        device = next(self.policy.policy.parameters()).device
        for where, i, v, strength in terms:
            delta = strength * v.to(device).float()
            active[(where, i)] = active.get((where, i), 0) + delta
        self.active = active

    def clear(self) -> None:
        self.active = {}

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


@contextmanager
def record(policy, where: str, which: list[int]):
    """Collect the mean MLP output over tokens, per layer and per forward call: out[layer] = list of (batch, width) tensors."""
    out = {i: [] for i in which}
    handles = []
    for i in which:
        def fn(module, inputs, output, i=i):
            out[i].append(output.detach().float().mean(dim=1).cpu())
        handles.append(layers(policy, where)[i].mlp.register_forward_hook(fn))
    try:
        yield out
    finally:
        for h in handles:
            h.remove()
