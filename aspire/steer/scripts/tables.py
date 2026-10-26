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


KIND = {"canon": "text rewrite", "lexedit": "in-network edit (live)", "lexfacts": "in-network edit (stored fact)",
        "carry_handback": "hybrid hand-back", "approach_handback": "hybrid hand-back"}


def verified() -> str:
    """The held-out verifications of library/INDEX.md: best program per task."""
    index = HERE / "library" / "INDEX.md"
    if not index.exists():
        return ""
    best = {}
    for line in index.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6 or not cells[2][:1].isdigit():
            continue
        program, task, frozen, prog, pval, verdict = cells[:6]
        k = int(prog.split("/")[0])
        better = task not in best or k > best[task][2] or (k == best[task][2] and program.startswith("lex"))   # ties: in-network
        if better:
            best[task] = (program, frozen, k, prog, pval, verdict)
    rows = sorted(best.items(), key=lambda kv: (kv[0].split()[0], int(kv[0].split()[1])))
    lines = [r"\begin{table}[h]\centering\small", r"\begin{tabular}{llllll}", r"\toprule",
             r"task & program & kind & frozen & program & $p$ \\", r"\midrule"]
    for task, (program, frozen, _, prog, pval, verdict) in rows:
        kind = next((v for k, v in KIND.items() if program.startswith(k)), "")
        mark = "" if verdict == "PROMOTED" else " (n.s.)"
        lines.append(f"\\texttt{{{tex(task)}}} & \\texttt{{{tex(program)}}} & {kind} & {frozen} & {prog}{mark} & {pval} \\\\")
    n = sum(v[5] == "PROMOTED" for v in best.values())
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{Held-out verification (initial states 20--49, 30 episodes): best program per task against the frozen "
              f"policy on the same states, one-sided Fisher exact test. {n} of {len(best)} tasks promoted.}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    (DOC / "results_baseline.tex").write_text(baseline() + "\n" + sweeps())
    (DOC / "results_verified.tex").write_text(verified())
    print("wrote", DOC / "results_baseline.tex")


if __name__ == "__main__":
    main()
