#!/usr/bin/env python3
"""Roll out the frozen SmolVLA and record per-layer activations (mean MLP output over tokens) with each episode's outcome.

  .venv/bin/python scripts/record.py --suite libero_goal_swap --task 3 --episodes 0-39

Writes outputs/record/<suite>__<task>.pt: {"episodes": [{"episode", "success", "steps", "vlm": (chunks, layers, 960),
"expert": (calls, layers, 720)}], "instruction", ...}. Rerunning appends missing episodes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from steer.hooks import layers, record  # noqa: E402
from steer.rollout import Policy, load_task, make_env, run_episode  # noqa: E402


def episodes(spec: str) -> list[int]:
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b) + 1)) if b else [int(a)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--episodes", default="0-19")
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "record")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    f = a.out / f"{a.suite}__{a.task}.pt"
    data = torch.load(f) if f.exists() else {"episodes": []}
    done = {e["episode"] for e in data["episodes"]}
    policy, task = Policy(), load_task(a.suite, a.task)
    data.update(suite=a.suite, task_id=a.task, name=task.name, instruction=task.instruction)
    env = make_env(task)
    nv, ne = len(layers(policy, "vlm")), len(layers(policy, "expert"))
    for e in episodes(a.episodes):
        if e in done:
            continue
        with record(policy, "vlm", list(range(nv))) as rv, record(policy, "expert", list(range(ne))) as re_:
            r = run_episode(env, policy, task, e)
        r["vlm"] = torch.stack([torch.cat(rv[i]) for i in range(nv)], dim=1)       # (chunks, layers, width)
        r["expert"] = torch.stack([torch.cat(re_[i]) for i in range(ne)], dim=1)   # (calls, layers, width)
        data["episodes"].append(r)
        torch.save(data, f)
        print(f"{a.suite} {a.task} episode {e}: {'success' if r['success'] else 'fail'} in {r['steps']} steps", flush=True)
    env.close()
    rate = sum(x["success"] for x in data["episodes"]) / len(data["episodes"])
    print(f"{a.suite} {a.task}: {rate:.2f} over {len(data['episodes'])} episodes  ({task.instruction})")


if __name__ == "__main__":
    main()
