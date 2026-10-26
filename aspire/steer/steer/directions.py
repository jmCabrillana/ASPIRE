"""Steering directions from recorded activations (scripts/record.py)."""

from __future__ import annotations

import torch


def episode_means(data: dict, where: str) -> torch.Tensor:
    """(episodes, layers, width): each episode's activation averaged over its chunks (vlm) or denoising calls (expert)."""
    return torch.stack([e[where].mean(dim=0) for e in data["episodes"]])


def contrast(pos: torch.Tensor, neg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit direction mean(pos) - mean(neg) per layer, and the spread of activations along it (for scaling strengths).

    pos, neg: (n, layers, width). Returns (layers, width) unit vectors and (layers,) standard deviations of the projections.
    """
    d = pos.mean(0) - neg.mean(0)
    unit = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    proj = torch.einsum("nlw,lw->nl", torch.cat([pos, neg]), unit)
    return unit, proj.std(0)


def success_direction(data: dict, where: str) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Successful minus failed episodes of one task. Returns (unit, spread, n_success, n_fail)."""
    x = episode_means(data, where)
    ok = torch.tensor([e["success"] for e in data["episodes"]])
    if ok.all() or not ok.any():
        raise ValueError("need both successful and failed episodes")
    unit, spread = contrast(x[ok], x[~ok])
    return unit, spread, int(ok.sum()), int((~ok).sum())


def shift_direction(target: dict, source: dict, where: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean activation of `target` episodes minus `source` episodes (e.g. the trained suite minus its perturbed version)."""
    return contrast(episode_means(target, where), episode_means(source, where))
