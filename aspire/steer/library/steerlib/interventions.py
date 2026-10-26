"""Reusable inference-time interventions on the frozen SmolVLA. Each one is applied through ctx, so it is undone after the episode."""

from __future__ import annotations

import torch


def push(ctx, direction: str, layer: int, sigmas: float):
    """Additive term for program(): `sigmas` spreads of a stored direction at one layer (site from the direction file)."""
    return ctx.push(direction, layer, sigmas)


def ablate(ctx, module, unit: torch.Tensor, scale: float = 0.0):
    """Hook on an MLP (or any module with a (batch, tokens, width) output): keep `scale` of the component along `unit`
    (0 removes it, 2 doubles it). Call in setup(ctx)."""
    u = unit / unit.norm()

    def fn(m, inputs, out):
        uu = u.to(out.device, out.dtype)
        return out + (scale - 1.0) * (out @ uu)[..., None] * uu

    return ctx.hook(module, fn)


def scale_output(ctx, module, factor: float):
    """Multiply a module's output (e.g. one layer's MLP) by `factor`. Call in setup(ctx)."""
    return ctx.hook(module, lambda m, inputs, out: out * factor)
