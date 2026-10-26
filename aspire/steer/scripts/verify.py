#!/usr/bin/env python3
"""Verify a steering program on held-out initial states (episodes 20-49) against the frozen VLA; record it in library/INDEX.md.

  .venv/bin/python scripts/verify.py --program library/programs/candidates/x.py --suite libero_goal_lan --task 4

Promotes the program to library/programs/ when it is better at the 5% level (one-sided Fisher exact test).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from scipy.stats import fisher_exact

HERE = Path(__file__).resolve().parents[1]
PY = str(HERE / ".venv" / "bin" / "python")


def heldout(program: str, suite: str, task: int, extra: list[str]) -> dict:
    name = "none" if program == "none" else Path(program).stem
    f = HERE / "outputs" / "programs" / name / f"{suite}__{task}" / "summary_heldout.json"
    if not f.exists():
        subprocess.run([PY, "scripts/run_program.py", "--program", program, "--suite", suite, "--task", str(task),
                        "--episodes", "20-49", "--heldout", *extra], cwd=HERE, check=True)
    return json.loads(f.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--program", required=True)
    p.add_argument("--suite", required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--n-action-steps", type=int)
    p.add_argument("--library-version", default="-", help="library version the program was written with")
    a = p.parse_args()
    extra = ["--n-action-steps", str(a.n_action_steps)] if a.n_action_steps else []
    base, prog = heldout("none", a.suite, a.task, []), heldout(a.program, a.suite, a.task, extra)
    n = len(base["episodes"])
    kb, kp = sum(e["success"] for e in base["episodes"]), sum(e["success"] for e in prog["episodes"])
    pval = fisher_exact([[kp, n - kp], [kb, n - kb]], alternative="greater").pvalue
    verdict = "PROMOTED" if pval < 0.05 and kp > kb else "not promoted"
    line = f"| {Path(a.program).stem} | {a.suite} {a.task} | {kb}/{n} | {kp}/{n} | {pval:.3f} | {verdict} | {a.library_version} |"
    index = HERE / "library" / "INDEX.md"
    if not index.exists():
        index.write_text("# Steering library\n\nHeld-out initial states 20-49; frozen VLA vs program; one-sided Fisher exact test.\n\n"
                         "| program | task | frozen | program | p | verdict | library |\n|---|---|---|---|---|---|---|\n")
    with index.open("a") as f:
        f.write(line + "\n")
    if verdict == "PROMOTED":
        shutil.copy(a.program, HERE / "library" / "programs" / Path(a.program).name)
    print(line)


if __name__ == "__main__":
    main()
