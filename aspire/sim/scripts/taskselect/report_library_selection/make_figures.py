#!/usr/bin/env python3
"""Figures for REPORT.md: lex vs unsolved library builds, and the generation/execution noise floor.

  .venv-libero/bin/python3 scripts/taskselect/report_library_selection/make_figures.py

Reads outputs/library_{lex,unsolved}_r1 (4 probed rounds each) and, when present, outputs/library_noise_{a,b}
(two fresh generations of round 0 from the empty library). Writes figs/*.png and figs/numbers.json.
Colours: blue = lex (disagreement), orange = unsolved (random among unsolved), grey = noise-floor runs.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[2] / "outputs"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8c8b86"
ARMS = {"lex": ("library_lex_r1", BLUE, "lex (disagreement)"), "unsolved": ("library_unsolved_r1", ORANGE, "unsolved (random)")}
STAGES = ["pass", "released", "slip", "air", "perceive", "crash"]

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


RNG = np.random.default_rng(0)
T_975_2DOF = 4.303      # Student t, 97.5th percentile, 2 degrees of freedom (three generations)


def outcome_array(tasks: dict) -> np.ndarray:
    """(tasks, programs, seeds) array of 0/1 outcomes, tasks in probe order."""
    return np.array([[row for row in t["outcomes"].values()] for t in tasks.values()], dtype=float)


def metric(a: np.ndarray, kind: str) -> float:
    rates = a.mean(axis=2)
    return float((rates.max(axis=1) if kind == "best" else rates.mean(axis=1)).mean())


def boot_ci(a: np.ndarray, kind: str, n: int = 4000) -> tuple[float, float]:
    """95% percentile interval of the 9-task mean, resampling tasks and then each program's seeds."""
    t, k, s = a.shape
    out = np.empty(n)
    for i in range(n):
        ts = RNG.integers(0, t, t)
        idx = RNG.integers(0, s, (t, k, s))
        g = np.take_along_axis(a[ts], idx, axis=2).mean(axis=2)
        out[i] = (g.max(axis=1) if kind == "best" else g.mean(axis=1)).mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def boot_diff_ci(a: np.ndarray, b: np.ndarray, kind: str, n: int = 4000) -> tuple[float, float]:
    """95% interval of metric(a) - metric(b): the same tasks resampled for both, seeds resampled independently."""
    t, k, s = a.shape
    out = np.empty(n)
    for i in range(n):
        ts = RNG.integers(0, t, t)
        vals = []
        for x in (a, b):
            idx = RNG.integers(0, s, (t, k, s))
            g = np.take_along_axis(x[ts], idx, axis=2).mean(axis=2)
            vals.append((g.max(axis=1) if kind == "best" else g.mean(axis=1)).mean())
        out[i] = vals[0] - vals[1]
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)



def load_probe(run: str, r: int) -> dict:
    return json.loads((OUT / run / f"round_{r:02d}" / "probe.json").read_text())["tasks"]


def short(task_id: str) -> str:
    names = {"open_the_middle_drawer": "open middle drawer", "put_the_bowl_on_the_stove": "bowl on stove",
             "turn_on_the_stove": "turn on stove", "between_the_plate": "bowl between plate/ramekin",
             "next_to_the_ramekin": "bowl next to ramekin", "from_table_center": "bowl from table center",
             "alphabet_soup": "alphabet soup", "cream_cheese": "cream cheese", "salad_dressing": "salad dressing"}
    return next((v for k, v in names.items() if k in task_id), task_id.split("/")[-1][:24])


def best_rate(tasks: dict) -> float:
    return st.fmean(max(sum(r) for r in t["outcomes"].values()) / 15 for t in tasks.values())


def mean_rate(tasks: dict) -> float:
    return st.fmean(st.fmean(sum(r) / 15 for r in t["outcomes"].values()) for t in tasks.values())


def stage_share(tasks: dict, stage: str) -> float:
    n = sum(len(s) for t in tasks.values() for s in t["stages"].values())
    return sum(x == stage for t in tasks.values() for s in t["stages"].values() for x in s) / n


def noise_batches() -> dict[str, dict]:
    out = {}
    for b in "ab":
        p = OUT / f"library_noise_{b}" / "round_00" / "probe.json"
        if p.exists():
            out[b] = json.loads(p.read_text())["tasks"]
    return out


def style(ax, ylabel=None, xlabel=None):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


# ---- figure 3: noise floor ---------------------------------------------------------------------------------------
def fig_noise(numbers: dict, noise: dict) -> None:
    """Mean and SD over the 27 programs' pass rates: independent round-0 generations from the empty library, and the final round of each arm."""
    rows = [("fresh batch a", noise["a"], GREY), ("fresh batch b", noise["b"], GREY),
            ("round 0, pilot", load_probe("library_unsolved_r2", 0), GREY),
            ("round 0, campaign rep 1", load_probe("library_unsolved_s30_r1", 0), GREY),
            ("round 0, campaign rep 2", load_probe("library_unsolved_s30_r2", 0), GREY)]
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    out, ys = {}, []
    for i, (name, run, color) in enumerate(rows):
        r = outcome_array(run).mean(axis=2).ravel()          # 27 programs, pass rate of each over its seeds
        m, sd = float(r.mean()), float(r.std(ddof=1))
        y = i
        ys.append(y)
        ax.errorbar(m, y, xerr=sd, fmt="o", color=color, markersize=8, capsize=4, elinewidth=1.8, markeredgecolor=SURFACE)
        ax.text(0.62, y, f"{m:.3f} ± {sd:.3f}", va="center", fontsize=9, color=INK)
        out[name] = {"mean": round(m, 3), "sd": round(sd, 3), "n_programs": int(r.size)}
    numbers["noise_all"] = out
    ax.set_yticks(ys, [n for n, _, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("pass rate of a program: mean ± SD over the 27 programs")
    fig.suptitle("Noise floor: five generations from the same empty library",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_noise_floor.png", dpi=160)
    plt.close(fig)


def main() -> None:
    numbers: dict = {}
    noise = noise_batches()
    fig_noise(numbers, noise)
    (FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers, indent=1)[:900])


if __name__ == "__main__":
    main()
