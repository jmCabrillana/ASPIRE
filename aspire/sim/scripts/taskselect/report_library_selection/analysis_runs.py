#!/usr/bin/env python3
"""Quantitative comparison of the library-build runs: all runs per round, the noise floor, and what difference it can detect.

  python3 scripts/taskselect/report_library_selection/analysis_runs.py

Metric: mean pass rate over the 27 programs (9 tasks x 3) of a probe. Older runs have 15 seeds (51-65); the 30-seed runs are
also reported on seeds 51-65 so that every run can be compared on the same seeds.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[3] / "outputs"
RUNS = {"lex r1": "library_lex_r1", "unsolved r1": "library_unsolved_r1", "lex r2": "library_lex_r2", "unsolved r2": "library_unsolved_r2",
        "lex s30 r1": "library_lex_s30_r1", "unsolved s30 r1": "library_unsolved_s30_r1"}
GENERATIONS = {"round 0, rep 1 (patched)": ("library_unsolved_r1", 0), "batch a": ("library_noise_a", 0), "batch b": ("library_noise_b", 0),
               "round 0, rep 2": ("library_unsolved_r2", 0), "round 0, s30 rep 1": ("library_unsolved_s30_r1", 0)}
RNG = np.random.default_rng(0)


def probe(run: str, r: int, first: int | None = None):
    """(tasks, programs, seeds) array of 0/1 outcomes and the stage labels, optionally the first `first` seeds only."""
    f = OUT / run / f"round_{r:02d}" / "probe.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    tasks = sorted(d["tasks"])
    a = np.array([[d["tasks"][t]["outcomes"][p] for p in sorted(d["tasks"][t]["outcomes"])] for t in tasks], dtype=float)
    s = [[d["tasks"][t]["stages"][p] for p in sorted(d["tasks"][t]["stages"])] for t in tasks]
    n = first or a.shape[2]
    return a[:, :, :n], np.array(s, dtype=object)[:, :, :n], tasks


def mean27(a: np.ndarray) -> float:
    return float(a.mean())


def crash(s: np.ndarray) -> float:
    return float((s == "crash").mean())


def boot_sd(a: np.ndarray, n: int = 2000) -> float:
    """SD of the mean when the 9 tasks and then the seeds of every program are resampled."""
    t, k, s = a.shape
    out = np.empty(n)
    for i in range(n):
        ts = RNG.integers(0, t, t)
        idx = RNG.integers(0, s, (t, k, s))
        out[i] = np.take_along_axis(a[ts], idx, axis=2).mean()
    return float(out.std(ddof=1))


def table_runs() -> None:
    print("## mean pass rate over the 27 programs, by round (seeds 51-65 | all seeds of the run)")
    for name, run in RUNS.items():
        row15, rowall, crashes = [], [], []
        for r in range(6):
            p15, pall = probe(run, r, 15), probe(run, r)
            if p15 is None:
                break
            row15.append(f"{mean27(p15[0]):.3f}"); rowall.append(f"{mean27(pall[0]):.3f}"); crashes.append(f"{100 * crash(pall[1]):.0f}%")
        print(f"{name:16s} 51-65: {' '.join(row15):38s} all: {' '.join(rowall):38s} crash: {' '.join(crashes)}")


def noise() -> tuple[float, float]:
    print("\n## noise floor: independent round-0 generations from the empty library (seeds 51-65)")
    vals = {}
    for name, (run, r) in GENERATIONS.items():
        p = probe(run, r, 15)
        if p is not None:
            vals[name] = mean27(p[0])
            print(f"  {name:28s} {vals[name]:.3f}   seed+task bootstrap SD {boot_sd(p[0]):.3f}")
    v = list(vals.values())
    unpatched = [x for k, x in vals.items() if "patched" not in k]
    sd, sd_u = st.stdev(v), st.stdev(unpatched)
    print(f"  mean {st.fmean(v):.3f}, SD over {len(v)} generations {sd:.3f}; without the hand-patched one: mean {st.fmean(unpatched):.3f},"
          f" SD {sd_u:.3f} (n={len(unpatched)})")
    return sd, sd_u


def detectability(sd: float) -> None:
    print("\n## what difference can be detected (two arms, generation noise only)")
    sd_diff = sd * 2 ** 0.5
    print(f"  SD of the difference of two single runs: {sd_diff:.3f}")
    for n in (1, 2, 3, 4, 6, 8):
        mde = 2.8 * sd_diff / n ** 0.5            # two-sided 5%, 80% power, n repetitions averaged per arm
        print(f"  {n} repetition(s) per arm: smallest reliably detected difference {mde:.3f}")


def gains() -> None:
    print("\n## change of the final round against round 0 (same run, all seeds of the run; programs paired by task)")
    for name, run in RUNS.items():
        rounds = [r for r in range(6) if probe(run, r) is not None]
        a0, a1 = probe(run, 0)[0], probe(run, rounds[-1])[0]
        d = a1.mean() - a0.mean()
        boots = []
        for _ in range(2000):
            ts = RNG.integers(0, 9, 9)
            boots.append(np.take_along_axis(a1[ts], RNG.integers(0, a1.shape[2], a1[ts].shape), 2).mean()
                         - np.take_along_axis(a0[ts], RNG.integers(0, a0.shape[2], a0[ts].shape), 2).mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {name:16s} round {rounds[-1]} - round 0 = {d:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]")


def per_task(run: str) -> None:
    print(f"\n## per task, passes of 90 trials (3 programs x seeds) by round: {run}")
    rounds = [r for r in range(6) if probe(run, r) is not None]
    names = probe(run, 0)[2]
    for i, t in enumerate(names):
        print(f"  {t.split('__')[-1][:48]:48s} " + " ".join(f"{int(probe(run, r)[0][i].sum()):3d}" for r in rounds))


def unseen_seeds() -> None:
    print("\n## 30-seed runs: mean pass rate on seeds 66-80, which no repair ever used")
    for run in ("library_lex_s30_r1", "library_unsolved_s30_r1"):
        rows = [probe(run, r) for r in range(6)]
        print(f"  {run:26s} " + " ".join(f"{p[0][:, :, 15:].mean():.3f}" for p in rows if p is not None))


def main() -> None:
    table_runs()
    sd, sd_u = noise()
    detectability(sd)
    gains()
    unseen_seeds()
    per_task("library_lex_s30_r1")


if __name__ == "__main__":
    main()
