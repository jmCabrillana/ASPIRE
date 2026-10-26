#!/usr/bin/env python3
"""Evaluate a steering program (or none) on a task: success rate, plus traces with the VLA's internals.

  .venv/bin/python scripts/run_program.py --program library/programs/my.py --suite libero_goal_swap --task 1 --episodes 0-19
  .venv/bin/python scripts/run_program.py --program none --suite libero_goal_swap --task 1 --episodes 0-9 --trace

Episodes 0-19 are for development; 20-49 are held out for verification (scripts/verify.py) and refused here unless --heldout.
Writes outputs/programs/<program>/<suite>__<task>/summary.json and, with --trace, episode_<n>.json per episode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from scripts.record import episodes as parse_episodes  # noqa: E402
from steer.hooks import Steering  # noqa: E402
from steer.introspect import Introspect  # noqa: E402
import torch  # noqa: E402

from steer.library import Context, load_direction, load_program, weights_fingerprint  # noqa: E402
from steer.rollout import Policy, load_task, make_env, run_episode  # noqa: E402

DEV_MAX = 19


def evaluate(policy, steering, intro, task, eps: list[int], module, trace: bool, out: Path | None) -> dict:
    env = make_env(task)
    runs, before = [], weights_fingerprint(policy.policy)
    for e in eps:
        ctx = Context(task, policy, steering, intro)
        call = (lambda name: getattr(module, name, None)) if module is not None else (lambda name: None)
        prog, step, setup, end = call("program"), call("on_step"), call("setup"), call("on_end")
        if setup:
            setup(ctx)
        fn = (lambda t, obs, internals: prog(t, obs, internals, ctx)) if prog else None
        on_step = (lambda t, obs, action: step(t, obs, action, ctx)) if step else None
        try:
            r = run_episode(env, policy, task, e, on_step=on_step, program=fn, steering=steering if fn else None,
                            introspect=intro, trace=trace, ctx=ctx)
            if end:
                end(r, ctx)
        finally:
            ctx.close()
        if not torch.allclose(weights_fingerprint(policy.policy), before):
            raise SystemExit("the program left the VLA weights changed; undo in-place edits in on_end(result, ctx)")
        if out is not None and trace:
            (out / f"episode_{e}.json").write_text(json.dumps(r, indent=1) + "\n")
        runs.append({k: v for k, v in r.items() if k != "trace"})
    env.close()
    return {"success_rate": sum(r["success"] for r in runs) / len(runs), "episodes": runs}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--program", required=True, help="path to a program file, or 'none' for the frozen VLA")
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--episodes", default="0-19")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--heldout", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--n-action-steps", type=int, help="replan every n steps instead of 50")
    a = p.parse_args()
    eps = parse_episodes(a.episodes)
    if max(eps) > DEV_MAX and not a.heldout:
        raise SystemExit(f"episodes above {DEV_MAX} are held out; use 0-{DEV_MAX}")
    module = None if a.program == "none" else load_program(a.program)
    name = "none" if module is None else Path(a.program).stem
    out = HERE / "outputs" / "programs" / name / f"{a.suite}__{a.task}"
    out.mkdir(parents=True, exist_ok=True)
    n_steps = a.n_action_steps or getattr(module, "N_ACTION_STEPS", None)
    policy = Policy(n_action_steps=n_steps)
    steering = Steering(policy)
    dirs = {}
    for d in getattr(module, "DIRECTIONS", []) if module else []:
        data = load_direction(d)
        for layer in range(data["unit"].shape[0]):
            dirs[f"{d}@{layer}"] = (data["where"], layer, data["unit"][layer])
    intro = Introspect(policy, directions=dirs)
    res = evaluate(policy, steering, intro, load_task(a.suite, a.task), eps, module, a.trace, out)
    res.update(program=name, suite=a.suite, task=a.task, n_action_steps=n_steps)
    (out / ("summary_heldout.json" if a.heldout else "summary.json")).write_text(json.dumps(res, indent=1) + "\n")
    print(f"{name} on {a.suite} {a.task}: success {res['success_rate']:.2f} over episodes {a.episodes}")


if __name__ == "__main__":
    main()
