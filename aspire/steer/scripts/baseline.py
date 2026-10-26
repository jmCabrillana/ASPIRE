#!/usr/bin/env python3
"""Success rate of the frozen SmolVLA per task.

  .venv/bin/python scripts/baseline.py --suites libero_goal libero_goal_swap --tasks 0 1 2 --episodes 10

Writes outputs/baseline/<suite>__<task_id>.json (one per task, skipped when it exists) and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from steer.rollout import Policy, load_task, make_env, run_episode  # noqa: E402

from libero import benchmark  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suites", nargs="+", required=True)
    p.add_argument("--tasks", type=int, nargs="*", help="task ids (default: all tasks of each suite)")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--out", type=Path, default=HERE / "outputs" / "baseline")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    policy = Policy()
    for suite in a.suites:
        ids = a.tasks if a.tasks else range(benchmark.get_benchmark_dict()[suite]().n_tasks)
        for i in ids:
            f = a.out / f"{suite}__{i}.json"
            if f.exists():
                continue
            task = load_task(suite, i)
            env = make_env(task)
            eps = [run_episode(env, policy, task, e) for e in range(a.episodes)]
            env.close()
            rate = sum(e["success"] for e in eps) / len(eps)
            f.write_text(json.dumps({"suite": suite, "task_id": i, "name": task.name, "instruction": task.instruction,
                                     "success_rate": rate, "episodes": eps}, indent=1) + "\n")
            print(f"{suite:22s} {i:2d} {rate:.2f}  {task.instruction}", flush=True)


if __name__ == "__main__":
    main()
