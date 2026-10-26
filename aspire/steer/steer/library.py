"""The steering library: named directions (library/directions/*.pt) and steering programs (library/programs/*.py).

A direction file holds, for one site ("vlm" or "expert"), a unit vector per layer and the spread of activations along it:
  {"where": str, "unit": (layers, width) tensor, "spread": (layers,) tensor, "source": str}
A program file may define any of (all optional):
  setup(ctx)                                once per episode, before the first step (register hooks, patch methods, ...)
  program(t, obs, internals, ctx)           before every new action chunk; returns additive steering terms (or [] / None)
  on_step(t, obs, action, ctx)              every step; returns a replacement action or None
  on_end(result, ctx)                       after the episode (undo in-place edits)
  DIRECTIONS (projected into `internals`), DESCRIPTION, N_ACTION_STEPS (replan every n steps instead of 50).
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
    """Everything a program may use. Programs are open: they can read and change anything in the VLA at inference time.

    ctx.vla          the SmolVLAPolicy (torch module): ctx.vla.model.vlm_with_expert, its config, ...
    ctx.vlm_layers   the 16 VLM text layers; ctx.expert_layers: the 16 action-expert layers
    ctx.hook(module, fn, pre=False, kwargs=False)   register a forward (or forward-pre) hook, removed after the episode
    ctx.patch(obj, name, value)                     replace an attribute (e.g. a method) for the episode, restored after it
    ctx.steering     additive steering (steer.hooks.Steering); ctx.push(name, layer, sigmas) builds a term from a direction
    ctx.introspect   the per-chunk internals collector (steer.introspect.Introspect)
    ctx.instruction  the instruction given to the VLA; a program may rewrite it
    ctx.log(key, value)   anything JSON-serialisable (tensors are converted); stored in the episode trace
    ctx.memory       a dict kept for the episode;  ctx.task: suite, task_id, name, instruction
    The weights must be the same after the episode as before (checked); temporary in-place edits must be undone in on_end.
    """

    def __init__(self, task, policy=None, steering=None, introspect=None):
        self.task, self.memory, self._dirs = task, {}, {}
        self.policy, self.steering, self.introspect = policy, steering, introspect
        self.vla = policy.policy if policy is not None else None
        self.instruction = task.instruction
        self.logs: list[dict] = []
        self._handles, self._patches = [], []
        self.t = 0

    @property
    def vlm_layers(self):
        return self.vla.model.vlm_with_expert.get_vlm_model().text_model.layers

    @property
    def expert_layers(self):
        return self.vla.model.vlm_with_expert.lm_expert.layers

    def hook(self, module, fn, pre: bool = False, kwargs: bool = False):
        h = (module.register_forward_pre_hook(fn, with_kwargs=kwargs) if pre
             else module.register_forward_hook(fn, with_kwargs=kwargs))
        self._handles.append(h)
        return h

    def patch(self, obj, name: str, value) -> None:
        self._patches.append((obj, name, obj.__dict__.get(name, _MISSING)))
        setattr(obj, name, value)

    def log(self, key: str, value) -> None:
        self.logs.append({"t": self.t, "key": key, "value": _jsonable(value)})

    def direction(self, name: str, layer: int) -> tuple[str, int, torch.Tensor, float]:
        """(where, layer, unit vector, spread) of a stored direction at one layer; strength = k * spread is a k-sigma push."""
        d = self._dirs.setdefault(name, load_direction(name))
        return d["where"], layer, d["unit"][layer], float(d["spread"][layer])

    def push(self, name: str, layer: int, sigmas: float) -> tuple[str, int, torch.Tensor, float]:
        """A steering term: add `sigmas` spreads of direction `name` at `layer`."""
        where, layer, unit, spread = self.direction(name, layer)
        return where, layer, unit, sigmas * spread

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        for obj, name, old in reversed(self._patches):
            if old is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, old)
        self._handles, self._patches = [], []
        if self.steering is not None:
            self.steering.clear()


_MISSING = object()


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().float().cpu()
        return round(float(v), 5) if v.numel() == 1 else [round(float(x), 5) for x in v.flatten()[:4096]]
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        return _jsonable(list(v.tolist()) if hasattr(v, "__len__") else v.tolist())
    return v


def weights_fingerprint(module) -> torch.Tensor:
    """Per-parameter sums: cheap, and changes when any weight is edited in place."""
    with torch.no_grad():
        return torch.stack([p.detach().float().sum() for p in module.parameters()]).cpu()


def load_program(path: str | Path):
    import sys
    if str(ROOT) not in sys.path:                  # programs import steerlib
        sys.path.insert(0, str(ROOT))
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"steer_program_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not any(hasattr(module, f) for f in ("setup", "program", "on_step")):
        raise ValueError(f"{path} defines none of setup, program, on_step")
    return module
