#!/usr/bin/env python3
"""Per-round table for one or more task-family evosearch runs (read-only).

  summarize_family_run.py RUN_DIR [RUN_DIR ...] [--json out.json]

For each finished round: the task, the rule that chose it, best / mean dev pass rate, the committee's realised disagreement
(D_seed primary, D_prog, mixed-seed fraction), and the parents chosen from it. If final/best.json exists: the tournament
winner and macro-average. With several runs, their final macro-averages are listed side by side.

If rounds have a family score (iter_NN/score/best.json from select_task.py score-pick), the regret of each run is printed:
the sum over rounds of (target - family score), and the first round that reaches the target. The target is --target, or the
best family score seen in any of the given runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402


def summarize(run_dir: Path) -> dict:
    rounds = []
    for h in core.load_history(run_dir):
        d = Path(h["iter_dir"])
        sel = json.loads((d / "selection.json").read_text())["parents"] if (d / "selection.json").exists() else []
        tj = json.loads((d / "task.json").read_text()) if (d / "task.json").exists() else {}
        m = h["metrics"]
        rates = m["pass_rates"]
        score = d / "score" / "best.json"
        rounds.append({"round": h["round"], "task_id": h["task_id"], "rule": tj.get("rule", "?"),
                       "family_score": json.loads(score.read_text())["macro_pass"] if score.exists() else None,
                       "best": max(rates.values()), "mean": m["mean_pass"], "D_prog": m["D_prog"],
                       "D_seed": m["D_seed"], "mixed_seed_frac": m["mixed_seed_frac"], "parents": sel})
    out = {"run_dir": str(run_dir), "rounds": rounds}
    best = run_dir / "final" / "best.json"
    if best.exists():
        b = json.loads(best.read_text())
        out["final"] = {"best": b["best"], "macro_pass": b["macro_pass"], "family": b["family"]}
    return out


def render(s: dict) -> str:
    lines = [f"{s['run_dir']}",
             "  rnd  rule     best  mean  D_prog D_seed mixed  score  task / parents"]
    for r in s["rounds"]:
        score = "  -  " if r["family_score"] is None else f"{r['family_score']:.3f}"
        lines.append(f"  {r['round']:3d}  {r['rule']:8s} {r['best']:.2f}  {r['mean']:.2f}  {r['D_prog']:.3f}  "
                     f"{r['D_seed']:.3f}  {r['mixed_seed_frac']:.2f}  {score}  {r['task_id'].split('/')[-1][:50]}")
        if r["parents"]:
            lines.append("       parents: " + ", ".join(r["parents"]))
    if "final" in s:
        lines.append(f"  final tournament: {s['final']['best']}  macro pass {s['final']['macro_pass']:.3f}")
    return "\n".join(lines)


def regret(scores: list[float], target: float) -> dict:
    """Sum over rounds of the gap to the target, and the first round at or above it (None if never)."""
    reached = next((i for i, s in enumerate(scores) if s >= target), None)
    return {"regret": round(sum(max(0.0, target - s) for s in scores), 4), "rounds_to_target": reached}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+", type=Path)
    p.add_argument("--json", type=Path)
    p.add_argument("--target", type=float, help="family score to reach (default: best seen in the given runs)")
    args = p.parse_args(argv)
    summaries = [summarize(d.resolve()) for d in args.run_dirs]
    for s in summaries:
        print(render(s))
    finals = [(s["run_dir"], s["final"]["macro_pass"]) for s in summaries if "final" in s]
    if len(finals) > 1:
        print("\nfinal macro pass: " + "   ".join(f"{Path(d).name}={v:.3f}" for d, v in finals))
    curves = {s["run_dir"]: [r["family_score"] for r in s["rounds"]] for s in summaries}
    curves = {d: c for d, c in curves.items() if c and None not in c}
    if curves:
        target = args.target if args.target is not None else max(max(c) for c in curves.values())
        print(f"\nregret against a family score of {target:.3f}")
        for s in summaries:
            if s["run_dir"] in curves:
                s["regret"] = regret(curves[s["run_dir"]], target)
                hit = s["regret"]["rounds_to_target"]
                print(f"  {Path(s['run_dir']).parent.name}/{Path(s['run_dir']).name}: regret {s['regret']['regret']:.3f} over "
                      f"{len(curves[s['run_dir']])} rounds, target " + ("not reached" if hit is None else f"reached in round {hit}"))
    if args.json:
        args.json.write_text(json.dumps(summaries, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
