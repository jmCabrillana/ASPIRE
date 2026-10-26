#!/usr/bin/env python3
"""One agent round on a target task: a Claude session (tmux) searches for a steering program on development episodes, then
the best program is verified on held-out episodes (scripts/verify.py).

  .venv/bin/python scripts/agent_round.py --suite libero_goal_lan --task 4 --budget 12
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from steer.rollout import load_task  # noqa: E402

def trained_suite(suite: str) -> str | None:
    for marker in ("_lan", "_swap", "_object", "_with_", "_env", "_task"):
        if marker in suite:
            return suite.split(marker)[0]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--hours", type=float, default=4)
    a = p.parse_args()
    task = load_task(a.suite, a.task)
    trained = trained_suite(a.suite)
    hint = f"the same task id in {trained}" if trained else "none"
    text = (HERE / "prompts" / "agent.md").read_text()
    for k, v in {"SUITE": a.suite, "TASK": str(a.task), "INSTRUCTION": task.instruction, "BUDGET": str(a.budget),
                 "TRAINED_HINT": hint}.items():
        text = text.replace("{{" + k + "}}", v)
    cand = HERE / "library" / "programs" / "candidates"
    cand.mkdir(parents=True, exist_ok=True)
    best = cand / f"BEST_{a.suite}__{a.task}"
    best.unlink(missing_ok=True)
    prompt = HERE / "outputs" / f"agent_{a.suite}__{a.task}.md"
    prompt.parent.mkdir(parents=True, exist_ok=True)
    prompt.write_text(text)
    session = f"steer-{a.suite}-{a.task}"
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    cmd = (f"unset ANTHROPIC_API_KEY; export MUJOCO_GL=egl PATH=$HOME/.local/bin:$PATH; "
           f"claude --model claude-opus-5-5 \"$(cat {prompt})\"; bash")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(HERE), cmd], check=True)
    deadline = time.time() + a.hours * 3600
    while not best.exists():
        if time.time() > deadline:
            raise SystemExit(f"no {best.name} after {a.hours} h")
        time.sleep(30)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    program = best.read_text().strip()
    if program and program != "none":
        subprocess.run([str(HERE / ".venv" / "bin" / "python"), "scripts/verify.py", "--program", program,
                        "--suite", a.suite, "--task", str(a.task)], cwd=HERE, check=True)
    else:
        print("the agent found nothing better than the frozen VLA")


if __name__ == "__main__":
    os.environ.pop("ANTHROPIC_API_KEY", None)
    main()
