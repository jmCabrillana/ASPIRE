#!/usr/bin/env python3
"""Offline comparison of task selectors on a known candidates x tasks x seeds outcome matrix.

Replays a selector for a budget of B rounds on a fixed committee and reports the cumulative realised D_seed of its picks
(round 0 is fixed and shared, so excluded) against the distribution of random orders and the oracle.

Probe rules (probe = per-seed split, probe_regret, probe_learnability) score every unvisited task from the first --probe-seeds seeds of the
matrix. When one is requested, the realised D_seed of EVERY rule is computed on the remaining seeds only.

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

RULES = ("random", "knn", "oracle", "harness", "probe", "probe_regret", "probe_learnability")


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


def _restrict(outcomes: dict[str, list[int]], idx: list[int]) -> dict[str, list[int]]:
    return {n: [row[i] for i in idx] for n, row in outcomes.items()}


def simulate(matrix: dict, rule: str, budget: int, round0: str, salt: str = "s0",
             proxy: Path | None = None, probe_seeds: int = 0) -> dict:
    """probe_seeds > 0 holds the first probe_seeds seeds out of the realised metric (for every rule)."""
    ids = [t["id"] for t in matrix["tasks"]]
    family = {"tasks": matrix["tasks"]}
    if budget < 1 or budget > len(ids):
        fail(f"budget must be in 1..{len(ids)}")
    history = [_history_entry(matrix, 0, round0)]
    n_seeds = len(next(iter(matrix["outcomes"][ids[0]].values())))
    if not 0 <= probe_seeds < n_seeds:
        fail(f"probe seeds must be in 0..{n_seeds - 1}")
    probe_idx, eval_idx = list(range(probe_seeds)), list(range(probe_seeds, n_seeds))
    d_true = {tid: core.committee_metrics(_restrict(matrix["outcomes"][tid], eval_idx))[core.PRIMARY] for tid in ids}
    probe_trials = 0
    for n in range(1, budget):
        visited = [h["task_id"] for h in history]
        unvisited = [t for t in ids if t not in visited]
        if rule == "random":
            tid = core.pick_random(matrix.get("family", "matrix"), ids, visited, salt)
        elif rule == "knn":
            tid = core.argmax_in_order(core.knn_scores(family, history), ids)
        elif rule == "oracle":
            tid = core.argmax_in_order({t: d_true[t] for t in unvisited}, ids)
        elif rule.startswith("probe"):
            if not probe_idx:
                fail("probe rules need probe_seeds >= 1")
            kind = rule.removeprefix("probe_") if "_" in rule else "split"
            scores = {t: core.probe_score(_restrict(matrix["outcomes"][t], probe_idx), kind) for t in unvisited}
            probe_trials += len(unvisited) * len(matrix["candidates"]) * len(probe_idx)
            tid = (core.argmax_in_order(scores, ids) if max(scores.values()) > 0
                   else core.pick_random(matrix.get("family", "matrix"), ids, visited, salt))
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
    return {"rule": rule, "picks": picks, "realised": d, "cumulative": round(sum(d), 6), "probe_trials": probe_trials}


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    pos = q * (len(sorted_values) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def backtest(matrix: dict, rules: list[str], budget: int, draws: int, proxy: Path | None = None,
             round0: str | None = None, probe_seeds: int = 1) -> dict:
    r0 = round0 or round0_of(matrix)
    ids = [t["id"] for t in matrix["tasks"]]
    if r0 not in ids:
        fail(f"round0 {r0!r} not in matrix tasks")
    held = probe_seeds if any(r.startswith("probe") for r in rules) else 0
    random_runs = sorted(simulate(matrix, "random", budget, r0, salt=f"s{i}", probe_seeds=held)["cumulative"]
                         for i in range(draws))
    report = {"family": matrix.get("family"), "budget": budget, "round0": r0, "n_tasks": len(ids),
              "metric": core.PRIMARY, "probe_seeds_held_out": held,
              "random": {"draws": draws, "mean": round(statistics.fmean(random_runs), 6),
                         "std": round(statistics.pstdev(random_runs), 6),
                         "ci95": [round(percentile(random_runs, 0.025), 6), round(percentile(random_runs, 0.975), 6)]},
              "rules": {}}
    for rule in rules:
        if rule == "random":
            continue
        res = simulate(matrix, rule, budget, r0, proxy=proxy, probe_seeds=held)
        below = sum(x < res["cumulative"] for x in random_runs)
        ties = sum(x == res["cumulative"] for x in random_runs)
        res["fraction_of_random_beaten"] = round((below + 0.5 * ties) / len(random_runs), 4)
        report["rules"][rule] = res
    return report


def print_report(rep: dict) -> None:
    r = rep["random"]
    held = f"  (first {rep['probe_seeds_held_out']} seed(s) held out for probing)" if rep["probe_seeds_held_out"] else ""
    print(f"family={rep['family']}  tasks={rep['n_tasks']}  budget={rep['budget']}  round0={rep['round0']}  "
          f"metric=cumulative {rep['metric']}{held}")
    print(f"  random ({r['draws']} orders)  mean {r['mean']:.3f}  std {r['std']:.3f}  "
          f"95% range [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]")
    for rule, res in rep["rules"].items():
        cost = f"  probe trials {res['probe_trials']}" if res["probe_trials"] else ""
        print(f"  {rule:12s} {res['cumulative']:.3f}  beats {res['fraction_of_random_beaten']:.0%} of random orders{cost}"
              f"   picks: " + ", ".join(p.split('/')[-1][:24] for p in res["picks"]))


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
    p.add_argument("--probe-seeds", type=int, default=1, help="seeds a probe rule measures per task")
    p.add_argument("--json", type=Path, help="write the report here")
    args = p.parse_args(argv)
    matrix = make_synthetic(args.seed) if args.synthetic else json.loads(args.matrix.read_text())
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    bad = [r for r in rules if r not in RULES]
    if bad:
        fail(f"unknown rules {bad}; known: {RULES}")
    rep = backtest(matrix, rules, args.budget, args.random_draws, args.proxy, args.round0, args.probe_seeds)
    print_report(rep)
    if args.json:
        args.json.write_text(json.dumps(rep, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
