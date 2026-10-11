"""Shared logic for task-family evosearch with disagreement-based task selection (design: DESIGN.md)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILIES_PATH = HERE / "families.json"
DESCRIPTORS_DIR = HERE / "descriptors"
KNN_EXPLORATION = 0.25
PRIMARY = "D_seed"      # per-seed pass/fail split: the contrast the next round learns from
PROBE_SCORES = ("split", "regret", "learnability")
TRACE_SCORES = ("trace", "progress")     # scores that read the trial traces, not only pass/fail
STAGES = ("crash", "perceive", "no_motion", "no_close", "air", "slip", "held_end", "released", "pass")
HELD_WIDTH = 0.03       # normalised gripper width after closing: below this the fingers closed on nothing
SEEN_SCORE = 0.15       # SAM3 best-mask score: below this the queried object was not found
PROXY_TIMEOUT_S = 120


def fail(message: str) -> "NoReturn":  # noqa: F821
    raise SystemExit(f"ERROR: {message}")


def load_families(path: Path | None = None) -> dict:
    return json.loads((path or FAMILIES_PATH).read_text())


def get_family(doc: dict, name: str) -> dict:
    if name not in doc["families"]:
        fail(f"unknown family {name!r}; known: {sorted(doc['families'])}")
    return doc["families"][name]


def task_ids(family: dict) -> list[str]:
    return [t["id"] for t in family["tasks"]]


def task_by_id(family: dict) -> dict[str, dict]:
    return {t["id"]: t for t in family["tasks"]}


def descriptor_dir(task: dict, root: Path | None = None) -> Path:
    return (root or DESCRIPTORS_DIR) / f"{task['suite']}__{task['task']}"


def describe(task: dict, root: Path | None = None) -> dict:
    """Perception-only descriptor: runtime instruction if describe_tasks.py ran, else the static one."""
    d = descriptor_dir(task, root)
    language_file, image = d / "task_language.txt", d / "scene.jpg"
    return {
        "id": task["id"], "suite": task["suite"], "task": task["task"],
        "language": language_file.read_text().strip() if language_file.exists() else task["language_static"],
        "language_is_runtime": language_file.exists(),
        "scene_image": str(image) if image.exists() else None,
    }


def round_number(iter_dir: Path) -> int:
    m = re.fullmatch(r"iter_(\d+)", iter_dir.name)
    if not m:
        fail(f"{iter_dir} is not an iter_NN directory")
    return int(m.group(1))


def load_round(iter_dir: Path) -> dict:
    """One finished round: task, seeds and the per-seed 0/1 outcome of every candidate."""
    summary = json.loads((iter_dir / "iter_summary.json").read_text())
    seeds = sorted(int(s) for s in summary["trial_seeds"])
    outcomes = {}
    for c in sorted(summary["candidates"], key=lambda c: c["candidate"]):
        by_seed = {int(t["trial"]): int(t["task_completed"]) for t in c["trial_results"]}
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            fail(f"{c['candidate']}: no result for seeds {missing} in {iter_dir}")
        outcomes[c["candidate"]] = [by_seed[s] for s in seeds]
    m = re.fullmatch(r"iter_(\d+)", iter_dir.name)     # tournament dirs (final/task_NN) have no round number
    return {
        "round": int(m.group(1)) if m else None, "iter_dir": str(iter_dir),
        "task_id": f"{summary['suite']}/{summary['task']}",
        "suite": summary["suite"], "task": summary["task"],
        "seeds": seeds, "outcomes": outcomes,
    }


def finished_rounds(run_dir: Path) -> list[Path]:
    return sorted((d for d in run_dir.glob("iter_[0-9][0-9]") if (d / "iter_summary.json").exists()),
                  key=round_number)


def load_history(run_dir: Path, before_round: int | None = None) -> list[dict]:
    """Finished rounds 0..before_round-1 (all of them if None), each with committee metrics.

    Rounds must be contiguous from 0: a gap means a round was skipped or is still being evaluated.
    """
    dirs = finished_rounds(run_dir)
    if before_round is not None:
        dirs = [d for d in dirs if round_number(d) < before_round]
        if [round_number(d) for d in dirs] != list(range(before_round)):
            fail(f"rounds 0..{before_round - 1} are not all finished in {run_dir}: "
                 f"found {[round_number(d) for d in dirs]}")
    elif [round_number(d) for d in dirs] != list(range(len(dirs))):
        fail(f"finished rounds are not contiguous from 0 in {run_dir}")
    history = []
    for d in dirs:
        h = load_round(d)
        h["metrics"] = committee_metrics(h["outcomes"])
        task_json = d / "task.json"
        h["rule"] = json.loads(task_json.read_text()).get("rule") if task_json.exists() else None
        history.append(h)
    return history


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def committee_metrics(outcomes: dict[str, list[int]]) -> dict:
    """Disagreement of K candidates on one task (see DESIGN.md, "Disagreement")."""
    names = sorted(outcomes)
    if not names:
        fail("committee_metrics: no candidates")
    n_seeds = len(outcomes[names[0]])
    if n_seeds == 0 or any(len(outcomes[n]) != n_seeds for n in names):
        fail("committee_metrics: candidates must have the same, non-zero number of seeds")
    if any(x not in (0, 1) for n in names for x in outcomes[n]):
        fail("committee_metrics: outcomes must be 0/1")
    k = len(names)
    rates = {n: sum(outcomes[n]) / n_seeds for n in names}
    pairs = list(combinations(names, 2))
    d_prog = sum(abs(rates[a] - rates[b]) for a, b in pairs) / len(pairs) if pairs else 0.0
    passing = [sum(outcomes[n][s] for n in names) for s in range(n_seeds)]
    hamming = (sum(sum(x != y for x, y in zip(outcomes[a], outcomes[b])) for a, b in pairs)
               / (len(pairs) * n_seeds)) if pairs else 0.0
    mean_pass = sum(rates.values()) / k
    return {
        "K": k, "n_seeds": n_seeds,
        "pass_rates": {n: round(r, 4) for n, r in rates.items()},
        "mean_pass": round(mean_pass, 4),
        "D_prog": round(d_prog, 6),
        "D_seed": round(sum(binary_entropy(c / k) for c in passing) / n_seeds, 6),
        "hamming_mean": round(hamming, 6),
        "mixed_seed_frac": round(sum(0 < c < k for c in passing) / n_seeds, 6),
        "all_fail": mean_pass == 0.0, "all_pass": mean_pass == 1.0,
    }


def rng_for(*parts) -> random.Random:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(digest, 16))


def random_order(family_name: str, ids: list[str], salt: str) -> list[str]:
    """Fixed random permutation of the family for (family, salt); the random arm walks along it."""
    order = list(ids)
    rng_for("taskselect-random", family_name, salt).shuffle(order)
    return order


def pick_random_any(family_name: str, ids: list[str], salt: str, round_idx: int) -> str:
    """Uniform pick with replacement (runs that may revisit a task)."""
    return rng_for("taskselect-random-any", family_name, salt, round_idx).choice(list(ids))


def pick_random(family_name: str, ids: list[str], visited: list[str], salt: str) -> str:
    seen = set(visited)
    for t in random_order(family_name, ids, salt):
        if t not in seen:
            return t
    fail("family exhausted: every task is already visited")


_STOP = {"the", "a", "an", "of", "to", "it", "and"}


def tokens(task: dict, descriptor: dict | None = None) -> set[str]:
    text = " ".join([(descriptor or {}).get("language") or task.get("language_static", ""),
                     task["task"].replace("_", " "), task["suite"].replace("_", " ")])
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def knn_scores(family: dict, history: list[dict], descriptors: dict[str, dict] | None = None) -> dict[str, float]:
    """Predicted disagreement of every unvisited task: similarity-weighted D_seed of visited tasks
    plus an exploration bonus for tasks unlike anything visited. No LLM, no simulator state."""
    by_id = task_by_id(family)
    tok = {i: tokens(t, (descriptors or {}).get(i)) for i, t in by_id.items()}
    visited = {h["task_id"]: h["metrics"][PRIMARY] for h in history}
    scores = {}
    for tid in by_id:
        if tid in visited:
            continue
        sims = {v: jaccard(tok[tid], tok[v]) for v in visited}
        total = sum(sims.values())
        pred = (sum(sims[v] * visited[v] for v in visited) / total if total > 0
                else sum(visited.values()) / len(visited))
        scores[tid] = round(pred + KNN_EXPLORATION * (1.0 - max(sims.values(), default=0.0)), 6)
    return scores


def probe_score(outcomes: dict[str, list[int]], kind: str = "split") -> float:
    """Score of a task from a few measured seeds of every candidate.

    split:  D_seed, the mean binary entropy of the fraction of candidates passing a seed.
    regret: mean over seeds of (best - mean) outcome; highest when exactly one candidate passes.
    learnability: p(1 - p) with p the pass rate pooled over candidates and seeds; highest at 50%.
    """
    if kind == "split":
        return committee_metrics(outcomes)["D_seed"]
    if kind not in PROBE_SCORES:
        fail(f"unknown probe score {kind!r}; known: {PROBE_SCORES}")
    rows = list(outcomes.values())
    if kind == "learnability":
        p = sum(map(sum, rows)) / (len(rows) * len(rows[0]))
        return round(p * (1.0 - p), 6)
    per_seed = [sum(r[s] for r in rows) / len(rows) for s in range(len(rows[0]))]
    return round(sum((1.0 - f) if f > 0 else 0.0 for f in per_seed) / len(per_seed), 6)


def trial_signature(trial_dir: Path) -> str:
    """What one trial did, from its API trace and printed output only: "<stage>|<tags>".

    Stage is how far the trial got (STAGES, grasp-centred): crashed, an object not found, no motion, moved without
    closing, closed on air, held then lost, still holding at the end, held and released, pass. Tags are the failure
    messages the program printed.
    """
    trace = json.loads((trial_dir / "trace.json").read_text())
    moved = any(e["function"] in ("goto_pose", "move_to_joints") for e in trace)
    closes, held, lost, released = 0, False, False, False
    for e in trace:
        fn, result = e["function"], e.get("result") or {}
        if fn == "close_gripper":
            closes += 1
            held = result.get("gripper_width", 0.0) >= HELD_WIDTH
        elif fn == "open_gripper":
            released, held = released or held, False
        elif fn == "get_observation" and held and (result.get("robot_cartesian_pos") or [1.0])[-1] < HELD_WIDTH:
            held, lost = False, True
    summary = trial_dir / "summary.txt"
    stdout = summary.read_text().split("Stdout:")[-1].split("Stderr:")[0].strip() if summary.exists() else ""
    tags = re.findall(r"^\[(\w+)\] (not found|[\w ]*failed|no [\w ]+|could not [\w ]+)", stdout, flags=re.M)
    unseen = any(e["function"].startswith("segment_sam3") and (e.get("result") or {}).get("mask_0_score", 1.0) < SEEN_SCORE
                 for e in trace) or any(b == "not found" for _, b in tags)
    if "taskcompleted_1" in trial_dir.name:
        stage = "pass"
    elif "sandboxrc_1" in trial_dir.name or "Traceback" in (summary.read_text() if summary.exists() else ""):
        stage = "crash"
    elif unseen:
        stage = "perceive"
    elif not moved:
        stage = "no_motion"
    elif closes == 0:
        stage = "no_close"
    elif released:
        stage = "released"
    elif held:
        stage = "held_end"
    else:
        stage = "slip" if lost else "air"
    return stage + "|" + ";".join(sorted({f"{a}:{b.strip()}" for a, b in tags}))


def task_mode(signatures: dict[str, list[str]]) -> str:
    """Most common failing stage of a task's trials; ties go to the earlier stage."""
    fails = Counter(s.split("|")[0] for row in signatures.values() for s in row if not s.startswith("pass"))
    return min(fails, key=lambda st: (-fails[st], STAGES.index(st))) if fails else "pass"


def render_trace(trial_dir: Path) -> str:
    """Compact text of a trial's API trace (one line per call, repeated motions collapsed), for a reader or an LLM."""
    def r3(v):
        return "(" + ", ".join(f"{x:.3f}" for x in v[:3]) + ")"
    lines: list[str] = []
    for e in json.loads((trial_dir / "trace.json").read_text()):
        fn, args, res = e["function"], e.get("args") or {}, e.get("result") or {}
        if fn == "get_observation":
            pose = res.get("robot_cartesian_pos")
            line = f"observe: hand at {r3(pose)}, gripper {pose[-1]:.3f}" if pose else "observe"
        elif fn == "segment_sam3_text_prompt":
            line = f"sam3 {args.get('text_prompt')!r}: {res.get('num_masks', 0)} masks, top score {res.get('mask_0_score', 0):.2f}"
        elif fn == "mask_to_world_points":
            line = "mask -> " + str(res.get("result", "")).replace("<ndarray shape=", "points ").split(" dtype")[0]
        elif fn == "get_oriented_bounding_box_from_3d_points":
            line = f"box: centre {r3(res['center'])}, extent {r3(res['extent'])}" if "center" in res else "box"
        elif fn == "plan_grasp":
            line = f"plan_grasp: {res.get('num_grasps', 0)} grasps, best score {res.get('best_score', 0):.2f}"
        elif fn in ("open_gripper", "close_gripper"):
            line = f"{fn}: width {res.get('gripper_width', float('nan')):.3f}"
        else:
            line = fn
        if lines and lines[-1].split(" x")[0] == line and fn in ("goto_pose", "move_to_joints"):
            count = int(lines[-1].split(" x")[1]) + 1 if " x" in lines[-1] else 2
            lines[-1] = f"{line} x{count}"
        else:
            lines.append(line)
    return "\n".join(lines)


def load_signatures(task_dir: Path, names: list[str], seeds: list[int]) -> dict[str, list[str]]:
    """Signature of every (candidate, seed) trial that evosearch_eval.py saved under task_dir."""
    out = {}
    for name in names:
        by_seed = {}
        for d in (task_dir / name / "eval").glob("trial_*"):
            m = re.match(r"trial_(\d+)_", d.name)
            if m:
                by_seed[int(m.group(1))] = trial_signature(d) if (d / "trace.json").exists() else "no_trace|"
        missing = [s for s in seeds if s not in by_seed]
        if missing:
            fail(f"no trace for {name} seeds {missing} under {task_dir}")
        out[name] = [by_seed[s] for s in seeds]
    return out


def trace_score(signatures: dict[str, list[str]], kind: str = "trace") -> float:
    """Disagreement of the candidates' traces on one task, mean over seeds.

    trace:    entropy (bits) of the signatures; zero when every candidate did the same thing.
    progress: mean pairwise gap in stage reached, as a fraction of the stage range.
    """
    if kind not in TRACE_SCORES:
        fail(f"unknown trace score {kind!r}; known: {TRACE_SCORES}")
    rows = list(signatures.values())
    total = 0.0
    for s in range(len(rows[0])):
        column = [r[s] for r in rows]
        if kind == "trace":
            total -= sum(c / len(column) * math.log2(c / len(column)) for c in Counter(column).values())
        else:
            stages = [STAGES.index(x.split("|")[0]) for x in column]
            pairs = list(combinations(stages, 2))
            total += sum(abs(a - b) for a, b in pairs) / (len(pairs) * (len(STAGES) - 1)) if pairs else 0.0
    return round(total / len(rows[0]), 6)


def argmax_in_order(scores: dict[str, float], order: list[str]) -> str:
    """Highest score; ties go to the task that comes first in family order."""
    candidates = [t for t in order if t in scores]
    if not candidates:
        fail("no task to choose from")
    return max(candidates, key=lambda t: (scores[t], -order.index(t)))


_RUNNER = (
    "import importlib.util, json, sys\n"
    "spec = importlib.util.spec_from_file_location('taskselect_proxy', sys.argv[1])\n"
    "mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
    "ctx = json.load(sys.stdin)\n"
    "print('\\n@@SCORES@@' + json.dumps(mod.score(ctx)))\n"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_proxy(proxy_path: Path, ctx: dict, unvisited: list[str], timeout: int = PROXY_TIMEOUT_S) -> dict[str, float]:
    """Run an LLM-written proxy in a subprocess: scrubbed environment, temp cwd, timeout.

    This limits accidents (credentials in the environment, stray writes into the repo); it is NOT a security
    sandbox: generated code still has full import access (see the repo security boundary).
    """
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8"}
    with tempfile.TemporaryDirectory(prefix="taskselect_proxy_") as tmp:
        env["HOME"] = tmp
        try:
            proc = subprocess.run([sys.executable, "-I", "-c", _RUNNER, str(Path(proxy_path).resolve())],
                                  input=json.dumps(ctx), capture_output=True, text=True,
                                  timeout=timeout, env=env, cwd=tmp)
        except subprocess.TimeoutExpired:
            fail(f"proxy {proxy_path} exceeded {timeout}s")
    if proc.returncode != 0:
        fail(f"proxy {proxy_path} crashed (exit {proc.returncode}):\n{proc.stderr[-1500:]}")
    marker = proc.stdout.rfind("@@SCORES@@")
    if marker < 0:
        fail(f"proxy {proxy_path} produced no scores")
    try:
        raw = json.loads(proc.stdout[marker + len("@@SCORES@@"):])
    except json.JSONDecodeError as e:
        fail(f"proxy {proxy_path} returned scores that are not JSON-serialisable: {e}")
    if not isinstance(raw, dict):
        fail("proxy score() must return a dict {task_id: float}")
    missing = [t for t in unvisited if t not in raw]
    if missing:
        fail(f"proxy gave no score for unvisited tasks {missing}")
    scores = {}
    for t in unvisited:
        v = raw[t]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            fail(f"proxy score for {t} is not a finite number: {v!r}")
        scores[t] = float(v)
    return scores


def lock_proxy(run_dir: Path, proxy_path: Path, round_idx: int, unlock: bool) -> str:
    """Freeze the proxy: first use records its hash; a changed file is refused unless --unlock (logged)."""
    sha = file_sha256(proxy_path)
    lock_path = run_dir / "proxy.lock.json"
    if not lock_path.exists():
        lock_path.write_text(json.dumps({"sha256": sha, "first_round": round_idx,
                                         "history": [{"round": round_idx, "sha256": sha, "unlocked": False}]},
                                        indent=2) + "\n")
        return sha
    lock = json.loads(lock_path.read_text())
    if lock["sha256"] != sha:
        if not unlock:
            fail(f"proxy changed since round {lock['first_round']} (locked {lock['sha256'][:12]}, "
                 f"now {sha[:12]}); pass --unlock to accept and log the change")
        lock["sha256"] = sha
        lock["history"].append({"round": round_idx, "sha256": sha, "unlocked": True})
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    return sha
