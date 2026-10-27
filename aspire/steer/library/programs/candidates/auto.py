"""Library entry point for a new perturbed task: dispatch by suite.
- *_lan  -> canon_v14 (English normalisation + canon v7 rewrite + frame-fill fallback + dialect completions; = v13 on all
           goal_lan / spatial_lan paraphrases, which are verified).
           Precompute the rewrites first: library/tools/precompute_canon.py <suite> (one process).
- *_swap -> carry_handback4 (carry_handback2 + nudge-and-retry) when library/references/<trained suite>__<task>.npz exists (built by tools/make_reference.py from
           the frozen policy's own successful trained-suite runs with --save-obs --trace); otherwise the frozen VLA.
- other  -> frozen VLA."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE.parents[1] / "references"
DESCRIPTION = "dispatcher: canon_v14 for *_lan, carry_handback4 for *_swap (if a reference exists)"
import sys as _sys
_suite = _sys.argv[_sys.argv.index("--suite") + 1] if "--suite" in _sys.argv else ""
N_ACTION_STEPS = 10 if _suite.endswith("_swap") else None     # read by run_program at import (the swap hybrid replans every 10)
_mods = {}


def _load(name):
    if name not in _mods:
        spec = importlib.util.spec_from_file_location(f"auto_{name}", HERE / f"{name}.py")
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); _mods[name] = m
    return _mods[name]


def _choice(ctx):
    s = ctx.task.suite
    if s.endswith("_lan"):
        return "canon_v14"
    if s.endswith("_swap") and (REF / f"{s[:-5]}__{ctx.task.task_id}.npz").exists():
        return "carry_handback4"
    return None


def setup(ctx):
    name = _choice(ctx); ctx.memory["auto"] = name; ctx.log("auto", name)
    if name and hasattr(_load(name), "setup"):
        _load(name).setup(ctx)


def program(t, obs, internals, ctx):
    m = ctx.memory.get("auto")
    return _load(m).program(t, obs, internals, ctx) if m and hasattr(_load(m), "program") else []


def on_step(t, obs, action, ctx):
    m = ctx.memory.get("auto")
    return _load(m).on_step(t, obs, action, ctx) if m and hasattr(_load(m), "on_step") else None


def on_end(result, ctx):
    m = ctx.memory.get("auto")
    if m and hasattr(_load(m), "on_end"):
        _load(m).on_end(result, ctx)
