"""The steering library: named directions (library/directions/*.pt) and steering programs (library/programs/*.py).

A direction file holds, for one site ("vlm" or "expert"), a unit vector per layer and the spread of activations along it:
  {"where": str, "unit": (layers, width) tensor, "spread": (layers,) tensor, "source": str}
A program file defines   program(t, obs, internals, ctx) -> list of (where, layer, unit, strength)
and may define DIRECTIONS (names it uses, so they can be projected in `internals`) and DESCRIPTION.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1] / "library"


def load_direction(name: str) -> dict:
    return torch.load(ROOT / "directions" / f"{name}.pt")


def save_direction(name: str, where: str, unit: torch.Tensor, spread: torch.Tensor, source: str) -> Path:
    path = ROOT / "directions" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"where": where, "unit": unit.float(), "spread": spread.float(), "source": source}, path)
    return path


class Context:
    """What a program may use besides the observation: directions, the task, and per-episode memory (`ctx.memory`)."""

    def __init__(self, task):
        self.task, self.memory, self._dirs = task, {}, {}

    def direction(self, name: str, layer: int) -> tuple[str, int, torch.Tensor, float]:
        """(where, layer, unit vector, spread) of a stored direction at one layer; strength = k * spread is a k-sigma push."""
        d = self._dirs.setdefault(name, load_direction(name))
        return d["where"], layer, d["unit"][layer], float(d["spread"][layer])

    def push(self, name: str, layer: int, sigmas: float) -> tuple[str, int, torch.Tensor, float]:
        """A steering term: add `sigmas` spreads of direction `name` at `layer`."""
        where, layer, unit, spread = self.direction(name, layer)
        return where, layer, unit, sigmas * spread


def load_program(path: str | Path):
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"steer_program_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "program"):
        raise ValueError(f"{path} defines no program(t, obs, internals, ctx)")
    return module
