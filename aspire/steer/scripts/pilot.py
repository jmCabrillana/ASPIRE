#!/usr/bin/env python3
"""First steering pilot: does pulling a perturbed task's representation towards the trained one raise success?

  .venv/bin/python scripts/pilot.py --trained libero_goal --perturbed libero_goal_lan

Picks the task with the largest drop from the trained to the perturbed suite (outputs/baseline), records 20 episodes of both
(episodes 0-19), stores the shift direction (trained minus perturbed, VLM and expert sites), then sweeps layer x strength on
held-out episodes 20-39 of the perturbed task, with strength 0 as the control on the same episodes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = str(HERE / ".venv" / "bin" / "python")


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=HERE, check=True)


def rate(suite: str, task: int) -> float:
    return json.loads((HERE / "outputs" / "baseline" / f"{suite}__{task}.json").read_text())["success_rate"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trained", default="libero_goal")
    p.add_argument("--perturbed", default="libero_goal_lan")
    p.add_argument("--task", type=int, help="default: the largest drop with the trained rate >= 0.8")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 7, 11, 15])
    p.add_argument("--strengths", type=float, nargs="+", default=[0, 1, 2, 4])
    p.add_argument("--where", nargs="+", default=["vlm", "expert"])
    a = p.parse_args()
    task = a.task
    if task is None:
        drops = {i: rate(a.trained, i) - rate(a.perturbed, i) for i in range(10) if rate(a.trained, i) >= 0.8}
        task = max(drops, key=drops.get)
        print(f"task {task}: trained {rate(a.trained, task):.2f}, perturbed {rate(a.perturbed, task):.2f}")
    for suite in (a.trained, a.perturbed):
        run("scripts/record.py", "--suite", suite, "--task", str(task), "--episodes", "0-19")
    for where in a.where:
        name = f"{a.perturbed}_{task}_shift_{where}"
        run("scripts/make_direction.py", "--name", name, "--record", f"{a.perturbed}__{task}",
            "--kind", f"shift:{a.trained}__{task}", "--where", where)
        run("scripts/sweep.py", "--suite", a.perturbed, "--task", str(task), "--direction", f"shift:{a.trained}",
            "--where", where, "--layers", *map(str, a.layers), "--strengths", *map(str, a.strengths), "--episodes", "20-39")


if __name__ == "__main__":
    main()
