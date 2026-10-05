#!/usr/bin/env python3
"""Score predicted pass probabilities against a measured outcome matrix.

  score_predictions.py --matrix M/matrix.json --predictions P.json [P2.json ...] [--exclude TASK_ID ...]

A predictions file is {"predictions": {task_id: {candidate: probability}}}. Reported per file, and for the mean of
all files: Brier score against constant baselines, AUC over (candidate, task) cells, and what a selector using the
predictions would get: the task with the highest predicted mixed-seed probability, and the rank correlation between
predicted and realised disagreement (D_seed) across tasks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from core import fail  # noqa: E402


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    va, vb = sum((x - ma) ** 2 for x in ra), sum((y - mb) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / math.sqrt(va * vb)


def auc(scores: list[float], labels: list[int]) -> float:
    """Probability that a passing trial got a higher prediction than a failing one (ties count half)."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def predicted_mixed(probs: list[float]) -> float:
    """Probability that a seed has both a pass and a fail, treating candidates as independent."""
    return 1.0 - math.prod(probs) - math.prod(1.0 - p for p in probs)


def score(matrix: dict, pred: dict[str, dict[str, float]], exclude: set[str]) -> dict:
    tasks = [t["id"] for t in matrix["tasks"] if t["id"] not in exclude]
    names = matrix["candidates"]
    for t in tasks:
        missing = [n for n in names if n not in pred.get(t, {})]
        if missing:
            fail(f"predictions lack {missing} for task {t}")
    p_flat, y_flat = [], []
    for t in tasks:
        for n in names:
            for y in matrix["outcomes"][t][n]:
                p_flat.append(min(1.0, max(0.0, float(pred[t][n]))))
                y_flat.append(int(y))
    base = sum(y_flat) / len(y_flat)

    def brier(ps):
        return sum((p - y) ** 2 for p, y in zip(ps, y_flat)) / len(y_flat)

    realised = [core.committee_metrics(matrix["outcomes"][t])["D_seed"] for t in tasks]
    expected = [predicted_mixed([float(pred[t][n]) for n in names]) for t in tasks]
    pick = max(range(len(tasks)), key=lambda i: (expected[i], -i))
    return {
        "cells": len(tasks) * len(names), "base_rate": round(base, 4),
        "brier": round(brier(p_flat), 4),
        "brier_always_0": round(brier([0.0] * len(y_flat)), 4),
        "brier_best_constant": round(brier([base] * len(y_flat)), 4),
        "auc": round(auc(p_flat, y_flat), 4),
        "spearman_disagreement": round(spearman(expected, realised), 4),
        "pick": tasks[pick], "pick_D_seed": realised[pick],
        "pick_rank": 1 + sum(r > realised[pick] for r in realised), "n_tasks": len(tasks),
        "mean_D_seed_random_pick": round(sum(realised) / len(realised), 4),
        "best_D_seed": max(realised),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, required=True)
    p.add_argument("--predictions", type=Path, nargs="+", required=True)
    p.add_argument("--exclude", nargs="*", default=[], help="task ids to leave out (e.g. the round-0 task)")
    args = p.parse_args(argv)
    matrix = json.loads(args.matrix.read_text())
    preds = [json.loads(f.read_text())["predictions"] for f in args.predictions]
    rows = [(f.parent.name + "/" + f.name, score(matrix, pr, set(args.exclude))) for f, pr in zip(args.predictions, preds)]
    if len(preds) > 1:
        tasks = [t["id"] for t in matrix["tasks"] if t["id"] not in args.exclude]
        mean = {t: {n: sum(float(pr[t][n]) for pr in preds) / len(preds) for n in matrix["candidates"]} for t in tasks}
        rows.append((f"mean of {len(preds)}", score(matrix, mean, set(args.exclude))))
    s0 = rows[0][1]
    print(f"{s0['cells']} cells, base rate {s0['base_rate']:.3f}; Brier baselines: always-0 {s0['brier_always_0']:.4f}, "
          f"best constant {s0['brier_best_constant']:.4f}; D_seed of a random pick {s0['mean_D_seed_random_pick']:.3f}, "
          f"best {s0['best_D_seed']:.3f}")
    for name, s in rows:
        print(f"  {name:40s} Brier {s['brier']:.4f}  AUC {s['auc']:.3f}  Spearman {s['spearman_disagreement']:+.2f}  "
              f"pick {s['pick'].split('/')[-1][:28]} (D_seed {s['pick_D_seed']:.3f}, rank {s['pick_rank']}/{s['n_tasks']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
