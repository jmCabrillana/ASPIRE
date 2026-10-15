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


# ---- figure 1: the two arms, round by round -----------------------------------------------------------------------
def fig_arms(numbers: dict, noise: dict) -> None:
    probes = {a: [load_probe(d, r) for r in range(4)] for a, (d, _, _) in ARMS.items()}
    x = np.arange(4)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, kind, title, ylabel in [(axes[0], "best", "Best program per task", "mean pass rate over the 9 tasks"),
                                    (axes[1], "all", "Mean over all 27 programs", "mean pass rate over the 9 tasks")]:
        for j, (a_, (_, color, label)) in enumerate(ARMS.items()):
            arrs = [outcome_array(p) for p in probes[a_]]
            y = [metric(v, kind) for v in arrs]
            ci = [boot_ci(v, kind) for v in arrs]
            xs = x + (j - 0.5) * 0.12
            ax.errorbar(xs, y, yerr=[[v - lo for v, (lo, _) in zip(y, ci)], [hi - v for v, (_, hi) in zip(y, ci)]], color=color,
                        linewidth=1.8, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        elinewidth=1.1, capsize=3, label=label)
            numbers[f"{kind}_{a_}"] = [{"value": round(v, 3), "ci95": [round(lo, 3), round(hi, 3)]} for v, (lo, hi) in zip(y, ci)]
        ax.set_xticks(x, ["0\n(shared)", "1", "2", "3\n(final)"])
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(0, 0.55 if kind == "best" else 0.36)
        style(ax, ylabel, "round (library after n repairs)")
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[2]
    for a_, (_, color, label) in ARMS.items():
        y = [100 * stage_share(p, "crash") for p in probes[a_]]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        numbers[f"crash_pct_{a_}"] = [round(v, 1) for v in y]
    ax.set_xticks(x, ["0\n(shared)", "1", "2", "3\n(final)"])
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(0, 50)
    style(ax, "% of trials that crash", "round")
    ax.set_title("Crashed trials", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.annotate("point-prompt API bug:\n120 / 90 / 60 trials in lex,\n30 / 15 / 0 in unsolved", xy=(1, 43), xytext=(1.3, 33),
                fontsize=8, color=MUTED, arrowprops={"arrowstyle": "-", "color": GRID})

    diffs = {}
    for kind in ("best", "all"):
        for r in range(1, 4):
            lo, hi = boot_diff_ci(outcome_array(probes["lex"][r]), outcome_array(probes["unsolved"][r]), kind)
            d = metric(outcome_array(probes["lex"][r]), kind) - metric(outcome_array(probes["unsolved"][r]), kind)
            diffs[f"{kind}_round{r}"] = {"lex_minus_unsolved": round(d, 3), "ci95": [round(lo, 3), round(hi, 3)]}
    numbers["arm_difference"] = diffs
    fig.suptitle("Library build with task selection: lex vs random-among-unsolved, one run each (points with 95% intervals)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig1_arms_by_round.png", dpi=160)
    plt.close(fig)


# ---- figure 2: per task, per round -----------------------------------------------------------------------------
def fig_heatmap(numbers: dict) -> None:
    probes = {a: [load_probe(d, r) for r in range(4)] for a, (d, _, _) in ARMS.items()}
    tasks = list(probes["lex"][0])
    repaired = {a: {x["task"]: x["round"] for x in json.loads((OUT / d / "state.json").read_text())["rounds"]}
                for a, (d, _, _) in ARMS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    cmap = matplotlib.colormaps["Blues"]
    for ax, (a, (_, color, label)) in zip(axes, ARMS.items()):
        grid = np.array([[max(sum(r) for r in probes[a][k][t]["outcomes"].values()) for k in range(4)] for t in tasks])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=15, aspect="auto")
        for i, t in enumerate(tasks):
            for k in range(4):
                v = grid[i, k]
                ax.text(k, i, str(v), ha="center", va="center", fontsize=10, color="white" if v > 8 else INK)
                if repaired[a].get(t) == k:
                    ax.add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=color, linewidth=2.5))
        ax.set_xticks(range(4), ["0", "1", "2", "3"])
        ax.set_yticks(range(len(tasks)), [short(t) for t in tasks])
        ax.set_xlabel("round")
        ax.set_title(label, loc="left", fontsize=11, fontweight="bold", color=color)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Passes of 15 by the best of 3 programs, per task and round (outlined = repaired after this probe)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2_task_by_round.png", dpi=160)
    plt.close(fig)


# ---- figure 3: noise floor ---------------------------------------------------------------------------------------
def fig_noise(numbers: dict, noise: dict) -> None:
    """Each value with its 95% interval: three generations of round 0 from the same empty library, and the two arms."""
    gens = [("round 0 (shared, patched)", load_probe("library_unsolved_r1", 0)), ("fresh batch a", noise["a"]),
            ("fresh batch b", noise["b"])]
    arms = [(f"{a_} arm, round 3", load_probe(d, 3), c) for a_, (d, c, _) in ARMS.items()]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    y = 0.0
    ticks, labels = [], []
    for kind, group in [("best", "best of the 3 programs per task"), ("all", "mean of all 27 programs")]:
        ax.text(0.003, y - 0.7, group, fontsize=10, fontweight="bold", color=INK, va="center")
        y += 0.0
        vals = []
        for name, run in gens:
            a = outcome_array(run)
            v, (lo, hi) = metric(a, kind), boot_ci(a, kind)
            vals.append(v)
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=INK if "round 0" in name else GREY, markersize=7,
                        capsize=4, elinewidth=1.6, markeredgecolor=SURFACE)
            ticks.append(y); labels.append(name)
            numbers.setdefault(f"noise_{kind}", {})[name] = {"value": round(v, 3), "ci95": [round(lo, 3), round(hi, 3)]}
            y += 1
        m, sd = st.fmean(vals), st.stdev(vals)
        half = T_975_2DOF * sd / len(vals) ** 0.5
        ax.errorbar(m, y, xerr=[[m - max(m - half, 0)], [half]], fmt="D", color=INK, markersize=8, capsize=5, elinewidth=2.4,
                    markeredgecolor=SURFACE)
        ticks.append(y); labels.append("mean of the 3 generations (t interval)")
        numbers[f"noise_{kind}"]["mean of 3 generations"] = {"value": round(m, 3), "ci95_t": [round(max(m - half, 0), 3), round(m + half, 3)],
                                                             "sd": round(sd, 3)}
        y += 1.0
        for name, run, color in arms:
            a = outcome_array(run)
            v, (lo, hi) = metric(a, kind), boot_ci(a, kind)
            ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="^", color=color, markersize=8, capsize=4, elinewidth=1.6,
                        markeredgecolor=SURFACE)
            ticks.append(y); labels.append(name)
            numbers[f"noise_{kind}"][name] = {"value": round(v, 3), "ci95": [round(lo, 3), round(hi, 3)]}
            y += 1
        y += 1.4
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.55)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("mean pass rate over the 9 tasks, with 95% interval")
    fig.suptitle("Noise floor: three generations from the same empty library, and the two arms (95% intervals)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Intervals on a single generation: bootstrap over tasks and seeds (4000 resamples). Diamond: mean of the three\n"
             "generations with a Student t interval (2 degrees of freedom), clipped at 0.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGS / "fig3_noise_floor.png", dpi=160)
    plt.close(fig)


def main() -> None:
    numbers: dict = {}
    noise = noise_batches()
    fig_arms(numbers, noise)
    fig_heatmap(numbers)
    fig_noise(numbers, noise)
    (FIGS / "numbers.json").write_text(json.dumps(numbers, indent=2) + "\n")
    print(json.dumps(numbers, indent=1)[:900])


if __name__ == "__main__":
    main()
