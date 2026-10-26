#!/usr/bin/env python3
"""LaTeX tables for doc/code_as_steering.tex from outputs/ (baseline per suite and task; steering sweeps).

  .venv/bin/python scripts/tables.py && cd doc && ~/miniconda3/envs/tex/bin/tectonic code_as_steering.tex
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT, DOC = HERE / "outputs", HERE / "doc"


def tex(s: str) -> str:
    return s.replace("_", r"\_")


def baseline() -> str:
    rows = defaultdict(dict)
    for f in sorted((OUT / "baseline").glob("*.json")):
        d = json.loads(f.read_text())
        rows[d["suite"]][d["task_id"]] = d["success_rate"]
    if not rows:
        return "No baseline yet.\n"
    suites = sorted(rows, key=lambda s: (s.split("_")[1], len(s), s))
    n_tasks = max(len(r) for r in rows.values())
    head = " & ".join(str(i) for i in range(n_tasks))
    lines = [r"\begin{table}[h]\centering\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l" + "c" * n_tasks + "c}", r"\toprule", f"suite & {head} & mean \\\\", r"\midrule"]
    for s in suites:
        vals = [rows[s].get(i) for i in range(n_tasks)]
        cells = " & ".join("" if v is None else f"{v:.1f}" for v in vals)
        done = [v for v in vals if v is not None]
        lines.append(f"\\texttt{{{tex(s)}}} & {cells} & {sum(done) / len(done):.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Frozen SmolVLA (\texttt{lerobot/smolvla\_libero}): success rate per task (10 episodes, initial states 0--9). "
              r"\texttt{\_swap}: object positions swapped; \texttt{\_lan}: paraphrased instruction.}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def sweeps() -> str:
    parts = []
    for f in sorted((OUT / "sweep").glob("*.json")):
        res = json.loads(f.read_text())
        suite, task, direction, where = f.stem.split("__")
        layers = sorted({r["layer"] for r in res})
        strengths = sorted({r["strength"] for r in res})
        cell = {(r["layer"], r["strength"]): r["success_rate"] for r in res}
        lines = [r"\begin{table}[h]\centering\small", r"\begin{tabular}{l" + "c" * len(strengths) + "}", r"\toprule",
                 "layer & " + " & ".join(f"{s:+g}$\\sigma$" for s in strengths) + r" \\", r"\midrule"]
        for l in layers:
            lines.append(f"{where} {l} & " + " & ".join("" if (l, s) not in cell else f"{cell[(l, s)]:.2f}" for s in strengths) + r" \\")
        n = len(res[0]["episodes"])
        lines += [r"\bottomrule", r"\end{tabular}",
                  f"\\caption{{Fixed steering on \\texttt{{{tex(suite)}}} task {task}: direction \\texttt{{{tex(direction)}}}, "
                  f"success over held-out initial states ({n} episodes); strength 0 is the frozen policy on the same states.}}",
                  r"\end{table}"]
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n" if parts else "No steering sweep yet.\n"


def main() -> None:
    (DOC / "results_baseline.tex").write_text(baseline() + "\n" + sweeps())
    print("wrote", DOC / "results_baseline.tex")


if __name__ == "__main__":
    main()
