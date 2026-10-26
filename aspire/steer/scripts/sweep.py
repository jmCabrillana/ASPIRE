#!/usr/bin/env python3
"""Fixed steering sweep: add strength x spread x direction at one layer during whole episodes, measure success.

  .venv/bin/python scripts/sweep.py --suite libero_goal_swap --task 3 --direction success --where vlm \
      --layers 4 8 12 --strengths -2 -1 1 2 --episodes 20-39

Directions come from outputs/record/<suite>__<task>.pt (episodes used there should not overlap --episodes):
  success: successful minus failed episodes of the task;  shift:<suite>: the recorded <suite> (same task id) minus this task.
Writes outputs/sweep/<suite>__<task>__<direction>__<where>.json (results appended per setting).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from steer.directions import shift_direction, success_direction  # noqa: E402
from steer.hooks import Steering  # noqa: E402
from steer.rollout import Policy, load_task, make_env, run_episode  # noqa: E402
from scripts.record import episodes  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--direction", default="success")
    p.add_argument("--where", default="vlm", choices=("vlm", "expert"))
    p.add_argument("--layers", type=int, nargs="+", required=True)
    p.add_argument("--strengths", type=float, nargs="+", required=True)
    p.add_argument("--episodes", default="20-39")
    a = p.parse_args()
    rec = HERE / "outputs" / "record"
    data = torch.load(rec / f"{a.suite}__{a.task}.pt")
    if a.direction == "success":
        unit, spread, *_ = success_direction(data, a.where)
    elif a.direction.startswith("shift:"):
        unit, spread = shift_direction(torch.load(rec / f"{a.direction[6:]}__{a.task}.pt"), data, a.where)
    else:
        raise SystemExit(f"unknown direction {a.direction}")
    out = HERE / "outputs" / "sweep" / f"{a.suite}__{a.task}__{a.direction.replace(':', '-')}__{a.where}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(out.read_text()) if out.exists() else []
    done = {(r["layer"], r["strength"]) for r in results}
    policy, task = Policy(), load_task(a.suite, a.task)
    steer, env = Steering(policy), make_env(task)
    eps = episodes(a.episodes)
    for layer in a.layers:
        for s in a.strengths:
            if (layer, s) in done:
                continue
            steer.set([(a.where, layer, unit[layer], s * float(spread[layer]))])
            runs = [run_episode(env, policy, task, e) for e in eps]
            rate = sum(r["success"] for r in runs) / len(runs)
            results.append({"layer": layer, "strength": s, "success_rate": rate, "episodes": eps,
                            "successes": [r["episode"] for r in runs if r["success"]]})
            out.write_text(json.dumps(results, indent=1) + "\n")
            print(f"{a.where} layer {layer:2d} strength {s:+.1f}: {rate:.2f}", flush=True)
    steer.clear()
    env.close()


if __name__ == "__main__":
    main()
