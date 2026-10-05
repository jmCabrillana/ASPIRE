#!/usr/bin/env python3
"""Offline comparison of task selectors on a known candidates x tasks x seeds outcome matrix.

Replays a selector for a budget of B rounds on a fixed committee and reports the cumulative realised D_prog of its picks
(round 0 is fixed and shared, so excluded) against the distribution of random orders and the oracle.

  backtest.py --matrix R/final/matrix.json --rules random,knn,oracle --budget 5
  backtest.py --synthetic --rules random,knn,oracle --budget 5 [--proxy proxy.py]

The matrix comes from `select_task.py final-pick` (real outcomes of the tournament) or from --synthetic (planted
structure: one cluster of tasks where the committee splits, one where all pass, one where all fail).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from core import fail  # noqa: E402

RULES = ("random", "knn", "oracle", "harness")


def make_synthetic(seed: int = 0, per_cluster: int = 4, k: int = 8, n_seeds: int = 15) -> dict:
    """Three clusters of tasks: committee split (high disagreement), all pass, all fail."""
    rng = random.Random(seed)
    names = [f"candidate_{chr(ord('A') + i)}" for i in range(k)]
    clusters = [
        ("drawer", "open the top drawer", [0.9] * (k // 2) + [0.1] * (k - k // 2)),
        ("basket", "pick up the milk and place it in the basket", [0.92] * k),
        ("stove", "turn on the stove knob", [0.04] * k),
    ]
    tasks, outcomes = [], {}
    for cname, text, probs in clusters:
        for j in range(per_cluster):
            tid = f"synthetic/{cname}_{j}"
            tasks.append({"id": tid, "suite": "synthetic", "task": f"{cname}_{j}",
                          "language_static": f"{text} variant{j}"})
            outcomes[tid] = {n: [int(rng.random() < min(1.0, max(0.0, p + rng.uniform(-0.04, 0.04))))
                                 for _ in range(n_seeds)] for n, p in zip(names, probs)}
    return {"family": "synthetic", "tasks": tasks, "candidates": names, "outcomes": outcomes}


def round0_of(matrix: dict) -> str:
    ids = [t["id"] for t in matrix["tasks"]]
    if matrix.get("round0") in ids:
        return matrix["round0"]
    name = matrix.get("family", "matrix")
    return ids[int(hashlib.sha256(name.encode()).hexdigest(), 16) % len(ids)]


def _history_entry(matrix: dict, n: int, tid: str) -> dict:
    out = matrix["outcomes"][tid]
    return {"round": n, "task_id": tid, "outcomes": out, "metrics": core.committee_metrics(out),
            "seeds": list(range(len(next(iter(out.values())))))}


def simulate(matrix: dict, rule: str, budget: int, round0: str, salt: str = "s0",
             proxy: Path | None = None) -> dict:
    ids = [t["id"] for t in matrix["tasks"]]
    family = {"tasks": matrix["tasks"]}
    if budget < 1 or budget > len(ids):
        fail(f"budget must be in 1..{len(ids)}")
    history = [_history_entry(matrix, 0, round0)]
    d_true = {tid: core.committee_metrics(matrix["outcomes"][tid])["D_prog"] for tid in ids}
    for n in range(1, budget):
        visited = [h["task_id"] for h in history]
        unvisited = [t for t in ids if t not in visited]
        if rule == "random":
            tid = core.pick_random(matrix.get("family", "matrix"), ids, visited, salt)
        elif rule == "knn":
            tid = core.argmax_in_order(core.knn_scores(family, history), ids)
        elif rule == "oracle":
            tid = core.argmax_in_order({t: d_true[t] for t in unvisited}, ids)
        elif rule == "harness":
            if proxy is None:
                fail("rule 'harness' needs --proxy")
            ctx = {"family": matrix.get("family"), "round": n,
                   "tasks": [{"id": t["id"], "suite": t["suite"], "task": t["task"],
                              "language": t["language_static"], "scene_image": None} for t in matrix["tasks"]],
                   "visited": visited, "unvisited": unvisited, "history": history, "committee": [], "parents": []}
            tid = core.argmax_in_order(core.run_proxy(proxy, ctx, unvisited), ids)
        else:
            fail(f"unknown rule {rule!r}")
        history.append(_history_entry(matrix, n, tid))
    picks = [h["task_id"] for h in history[1:]]
    d = [d_true[t] for t in picks]
    return {"rule": rule, "picks": picks, "D_prog": d, "cumulative_D_prog": round(sum(d), 6)}


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def backtest(matrix: dict, rules: list[str], budget: int, draws: int, proxy: Path | None = None,
             round0: str | None = None) -> dict:
    r0 = round0 or round0_of(matrix)
    ids = [t["id"] for t in matrix["tasks"]]
    if r0 not in ids:
        fail(f"round0 {r0!r} not in matrix tasks")
    random_runs = sorted(simulate(matrix, "random", budget, r0, salt=f"s{i}")["cumulative_D_prog"]
                         for i in range(draws))
    report = {"family": matrix.get("family"), "budget": budget, "round0": r0, "n_tasks": len(ids),
              "random": {"draws": draws, "mean": round(statistics.fmean(random_runs), 6),
                         "std": round(statistics.pstdev(random_runs), 6),
                         "ci95": [round(percentile(random_runs, 0.025), 6), round(percentile(random_runs, 0.975), 6)]},
              "rules": {}}
    for rule in rules:
        if rule == "random":
            continue
        res = simulate(matrix, rule, budget, r0, proxy=proxy)
        below = sum(x < res["cumulative_D_prog"] for x in random_runs)
        ties = sum(x == res["cumulative_D_prog"] for x in random_runs)
        res["fraction_of_random_beaten"] = round((below + 0.5 * ties) / len(random_runs), 4)
        report["rules"][rule] = res
    return report


def print_report(rep: dict) -> None:
    r = rep["random"]
    print(f"family={rep['family']}  tasks={rep['n_tasks']}  budget={rep['budget']}  round0={rep['round0']}")
    print(f"  random ({r['draws']} orders)   cumulative D_prog mean {r['mean']:.3f}  std {r['std']:.3f}  "
          f"95% range [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    for rule, res in rep["rules"].items():
        print(f"  {rule:8s} cumulative D_prog {res['cumulative_D_prog']:.3f}  beats {res['fraction_of_random_beaten']:.0%} "
              f"of random orders   picks: " + ", ".join(p.split('/')[-1][:24] for p in res["picks"]))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--matrix", type=Path)
    g.add_argument("--synthetic", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="synthetic generator seed")
    p.add_argument("--rules", default="random,knn,oracle")
    p.add_argument("--budget", type=int, default=5)
    p.add_argument("--random-draws", type=int, default=500)
    p.add_argument("--proxy", type=Path)
    p.add_argument("--round0")
    p.add_argument("--json", type=Path, help="write the report here")
    args = p.parse_args(argv)
    matrix = make_synthetic(args.seed) if args.synthetic else json.loads(args.matrix.read_text())
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    bad = [r for r in rules if r not in RULES]
    if bad:
        fail(f"unknown rules {bad}; known: {RULES}")
    rep = backtest(matrix, rules, args.budget, args.random_draws, args.proxy, args.round0)
    print_report(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
