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
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for a, (_, color, label) in ARMS.items():
        y = [best_rate(p) for p in probes[a]]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(3.08, y[-1], label, color=INK, va="center", fontsize=9)
        numbers[f"best_rate_{a}"] = [round(v, 3) for v in y]
    if len(noise) == 2:
        vals = [best_rate(noise["a"]), best_rate(noise["b"]), best_rate(load_probe("library_unsolved_r1", 0))]
        ax.axhspan(min(vals), max(vals), color=GREY, alpha=0.18, linewidth=0)
        ax.text(1.0, min(vals) + 0.006, "range of 3 independent round-0\ngenerations (same empty library)", color=MUTED, fontsize=8)
    ax.set_xticks(x, ["0\n(shared)", "1", "2", "3\n(final)"])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0, 0.36)
    style(ax, "mean best-program pass rate (9 tasks)", "round (library after n repairs)")
    ax.set_title("Mean best-program rate", loc="left", fontsize=11, fontweight="bold")

    ax = axes[1]
    for a, (_, color, label) in ARMS.items():
        y = [mean_rate(p) for p in probes[a]]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(3.08, y[-1], label, color=INK, va="center", fontsize=9)
        numbers[f"mean_rate_{a}"] = [round(v, 3) for v in y]
    ax.set_xticks(x, ["0\n(shared)", "1", "2", "3\n(final)"])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0, 0.36)
    style(ax, "mean pass rate over all programs", "round")
    ax.set_title("Mean over all 27 programs", loc="left", fontsize=11, fontweight="bold")

    ax = axes[2]
    for a, (_, color, label) in ARMS.items():
        y = [100 * stage_share(p, "crash") for p in probes[a]]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(3.08, y[-1], label, color=INK, va="center", fontsize=9)
        numbers[f"crash_pct_{a}"] = [round(v, 1) for v in y]
    ax.set_xticks(x, ["0\n(shared)", "1", "2", "3\n(final)"])
    ax.set_xlim(-0.2, 4.2)
    ax.set_ylim(0, 50)
    style(ax, "% of trials that crash", "round")
    ax.set_title("Crashed trials", loc="left", fontsize=11, fontweight="bold")
    ax.annotate("point-prompt API bug:\n120 / 90 / 60 trials in lex,\n30 / 15 / 0 in unsolved", xy=(1, 43), xytext=(1.25, 38),
                fontsize=8, color=MUTED, arrowprops={"arrowstyle": "-", "color": GRID})

    fig.suptitle("Library build with task selection: lex vs random-among-unsolved, one run each", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
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
    """One plot: three independent generations of round 0 from the same empty library, two ways of averaging."""
    gens = [("round 0 (shared, patched)", load_probe("library_unsolved_r1", 0), "o", INK),
            ("fresh batch a", noise["a"], "D", GREY), ("fresh batch b", noise["b"], "s", GREY)]
    tasks = list(gens[0][1])

    def value(run: dict, t: str, kind: str) -> float:
        rows = [sum(r) for r in run[t]["outcomes"].values()]
        return (max(rows) if kind == "best" else st.fmean(rows)) / 15

    kinds = [("best", "best of the 3 programs\nper task"), ("all", "mean of all\n27 programs")]
    arms_final = {a: load_probe(d, 3) for a, (d, _, _) in ARMS.items()}
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    stats: list[str] = []
    for x, (kind, label) in enumerate([(k, l) for k, l in kinds]):
        per_gen = [st.fmean(value(run, t, kind) for t in tasks) for _, run, _, _ in gens]
        mean, sd = st.fmean(per_gen), st.stdev(per_gen)
        pooled = (st.fmean(st.variance([value(run, t, kind) for _, run, _, _ in gens]) for t in tasks)) ** 0.5 / len(tasks) ** 0.5
        ax.add_patch(plt.Rectangle((x - 0.17, mean - sd), 0.34, 2 * sd, color=GRID, zorder=1))
        ax.plot([x - 0.17, x + 0.17], [mean, mean], color=INK, linewidth=2, zorder=2)
        for j, ((name, _, marker, color), v) in enumerate(zip(gens, per_gen)):
            ax.scatter(x + (j - 1) * 0.09, v, s=70, marker=marker, color=color, edgecolor=SURFACE, linewidth=1.3, zorder=3,
                       label=name if x == 0 else None)
        for a_, (_, color, _) in ARMS.items():
            v = best_rate(arms_final[a_]) if kind == "best" else mean_rate(arms_final[a_])
            ax.scatter(x + 0.30, v, s=60, marker="^", color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=f"{a_} arm, final round" if x == 0 else None)
        stats.append(f"{label}\n\nmean {mean:.3f}\nSD across generations {sd:.3f}\nper-task pooled estimate {pooled:.3f}")
        numbers[f"noise_{kind}"] = {"generations": [round(v, 3) for v in per_gen], "mean": round(mean, 3),
                                    "sd_across_generations": round(sd, 3), "sd_from_pooled_task_variance": round(pooled, 3)}
    ax.set_xticks([0, 1], stats)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 0.32)
    style(ax, "mean pass rate over the 9 tasks")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)
    ax.set_title("Noise floor: same empty library, three independent generations\n(grey bar = mean \u00b1 1 SD; triangles = the two arms after 3 repairs)",
                 loc="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
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
