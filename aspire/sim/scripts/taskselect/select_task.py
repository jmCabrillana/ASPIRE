#!/usr/bin/env python3
"""Choose, record and verify the task of every round of a task-family evosearch run.

The task of round N is chosen after the candidates of round N exist, so the committee is the set of programs about to be
evaluated. The agent never writes task.json by hand.

  select       --family F --run-dir R --round N --rule random|knn|prompt|harness|probe [--salt r1] [--choice f] [--proxy f]
  probe-plan   --family F --run-dir R --round N [--probe-seeds 1]   probe rule: commands measuring the round's candidates on
                                                        the first dev seed(s) of every unvisited task
  context      --family F --run-dir R --round N        briefing for the prompt / harness rules (also saved to iter_NN/context.json)
  verify       --family F --run-dir R                   consistency of every round's task.json with the family and the eval summary
  eval-cmd     --run-dir R --round N [--gpu 0]          the exact evosearch_eval.py command for the chosen task
  init-from    --family F --run-dir R --from-run REF    copy iter_00 (same candidates, same results) from a reference run
  score-plan   --family F --run-dir R --round N        family score of round N: its 3 parents on every family task (score seeds)
  score-pick   --run-dir R --round N                   writes iter_NN/score/best.json (macro_pass = the family score)
  final-plan   --family F --run-dir R                   tournament: last round's 3 parents on every family task
  final-pick   --run-dir R                              macro-average winner of the tournament; writes final/best.json, final/matrix.json
  heldout-cmds --family F --run-dir R [--gpu 0]         held-out commands for the tournament winner on every family task
  heldout-summary --family F --run-dir R                per-task and macro-average held-out pass rate; writes R/heldout.json
  render-prompt --family F --arm A --rep N [--ref-run P] the coordinator prompt for one arm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from core import fail  # noqa: E402

RULES = ("random", "knn", "prompt", "harness", "probe")
REVISIT_RULES = ("random", "probe")
SCRIPT = Path(__file__).resolve()
PROMPTS = core.HERE / "prompts"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def committee_of(iter_dir: Path) -> list[dict]:
    return [{"candidate": d.name, "code_path": str(d / "code.py")}
            for d in sorted(iter_dir.glob("candidate_*")) if (d / "code.py").exists()]


def load_family(args: argparse.Namespace) -> tuple[dict, str, dict]:
    doc = core.load_families(Path(args.families) if getattr(args, "families", None) else None)
    return doc, args.family, core.get_family(doc, args.family)


def build_context(doc: dict, family_name: str, family: dict, history: list[dict], round_idx: int,
                  run_dir: Path, descriptors_dir: Path | None) -> dict:
    ids = core.task_ids(family)
    visited = [h["task_id"] for h in history]
    iter_dir = run_dir / f"iter_{round_idx:02d}"
    prev = run_dir / f"iter_{round_idx - 1:02d}"
    selection = prev / "selection.json"
    return {
        "family": family_name, "relation": family["relation"], "description": family["description"],
        "round": round_idx, "dev_seeds": doc["dev_seeds"],
        "tasks": [core.describe(t, descriptors_dir) for t in family["tasks"]],
        "visited": visited, "unvisited": [t for t in ids if t not in visited],
        "history": [{"round": h["round"], "task_id": h["task_id"], "seeds": h["seeds"],
                     "outcomes": h["outcomes"], "metrics": h["metrics"], "iter_dir": h["iter_dir"]}
                    for h in history],
        "committee": committee_of(iter_dir),
        "parents": json.loads(selection.read_text())["parents"] if selection.exists() else [],
    }


def validate_choice(choice_path: Path, unvisited: list[str], committee_names: list[str]) -> dict:
    if not choice_path.exists():
        fail(f"{choice_path} not found: write it first (see the arm instructions)")
    try:
        choice = json.loads(choice_path.read_text())
    except json.JSONDecodeError as e:
        fail(f"{choice_path} is not valid JSON: {e}")
    if not isinstance(choice, dict):
        fail(f"{choice_path} must be a JSON object")
    if choice.get("task") not in unvisited:
        fail(f"task_choice.task {choice.get('task')!r} is not an unvisited family task; options: {unvisited}")
    predicted = choice.get("predicted_pass")
    if not isinstance(predicted, dict) or sorted(predicted) != sorted(committee_names):
        fail(f"predicted_pass must have exactly the committee's candidates {sorted(committee_names)}")
    for name, p in predicted.items():
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0.0 <= p <= 1.0:
            fail(f"predicted_pass[{name}] must be a probability in [0, 1], got {p!r}")
    rationale = choice.get("rationale")
    if not isinstance(rationale, str) or len(rationale.strip()) < 20:
        fail("rationale must be a sentence or more explaining why the committee should disagree on this task")
    return {"task": choice["task"], "predicted_pass": {k: float(v) for k, v in sorted(predicted.items())},
            "rationale": rationale.strip()}


def _eval_command(iter_dir: Path, suite: str, task: str, seeds: list[int], gpu: str, parallel: str) -> str:
    return (f"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv-libero/bin/python3 scripts/libero/evosearch_eval.py "
            f"--iter-dir {iter_dir} --suite {suite} --task {task} --trials {len(seeds)} "
            f"--trial-seeds {' '.join(map(str, seeds))} --sim-gpus {gpu} --parallel-per-gpu {parallel} --no-highlights")


def cmd_probe_plan(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    n = args.round
    if n < 1:
        fail("round 0 has a fixed task; nothing to probe")
    iter_dir = run_dir / f"iter_{n:02d}"
    committee = committee_of(iter_dir)
    if not committee:
        fail(f"write {iter_dir}/candidate_*/code.py before probing")
    visited = [h["task_id"] for h in core.load_history(run_dir, before_round=n)]
    seeds = doc["dev_seeds"][:args.probe_seeds]
    code = {c["candidate"]: core.file_sha256(Path(c["code_path"])) for c in committee}
    plan_path = iter_dir / "probe" / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (plan["seeds"] != seeds or plan["code_sha256"] != code or plan["visited"] != visited
                or plan["allow_revisit"] != args.allow_revisit):
            fail(f"{plan_path} was made for other seeds, candidates or history; refusing to mix probes")
    else:
        tasks = []
        for k, t in enumerate(family["tasks"]):
            if t["id"] in visited and not args.allow_revisit:
                continue
            tdir = iter_dir / "probe" / f"task_{k:02d}"
            for c in committee:
                (tdir / c["candidate"]).mkdir(parents=True, exist_ok=True)
                shutil.copy2(c["code_path"], tdir / c["candidate"] / "code.py")
            tasks.append({"id": t["id"], "suite": t["suite"], "task": t["task"], "dir": str(tdir)})
        plan = {"round": n, "family": family_name, "seeds": seeds, "visited": visited, "code_sha256": code,
                "allow_revisit": args.allow_revisit,
                "tasks": tasks, "created_at": now()}
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    for t in plan["tasks"]:
        if not (Path(t["dir"]) / "iter_summary.json").exists():
            print(_eval_command(Path(t["dir"]), t["suite"], t["task"], plan["seeds"], args.gpu, args.parallel))
    return 0


def read_probe(iter_dir: Path, pool: list[str], kind: str) -> tuple[dict[str, float], int]:
    """Probe scores of every task in the pool, after checking the probe measured the current candidates."""
    plan_path = iter_dir / "probe" / "plan.json"
    if not plan_path.exists():
        fail(f"{plan_path} missing: run probe-plan and its commands first")
    plan = json.loads(plan_path.read_text())
    code = {c["candidate"]: core.file_sha256(Path(c["code_path"])) for c in committee_of(iter_dir)}
    if plan["code_sha256"] != code:
        fail("candidates changed after the probe was planned; the probe no longer describes them")
    if [t["id"] for t in plan["tasks"]] != pool:
        fail("probe plan does not cover exactly the tasks this round may select")
    scores, trials = {}, 0
    for t in plan["tasks"]:
        d = Path(t["dir"])
        if not (d / "iter_summary.json").exists():
            fail(f"probe not finished: no results for {t['id']}")
        r = core.load_round(d)
        if r["task_id"] != t["id"] or r["seeds"] != sorted(plan["seeds"]) or sorted(r["outcomes"]) != sorted(code):
            fail(f"{d}: probe results do not match the plan (task, seeds or candidates)")
        scores[t["id"]] = core.probe_score(r["outcomes"], kind)
        trials += len(r["outcomes"]) * len(r["seeds"])
    return scores, trials


def cmd_select(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    ids = core.task_ids(family)
    n = args.round
    iter_dir = run_dir / f"iter_{n:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    descriptors_dir = Path(args.descriptors_dir) if args.descriptors_dir else None
    record: dict = {"round": n, "family": family_name, "requested_rule": args.rule}

    if n == 0:
        history, visited = [], []
        chosen, rule = family["round0"], "fixed"
    else:
        history = core.load_history(run_dir, before_round=n)
        visited = [h["task_id"] for h in history]
        unknown = [t for t in visited if t not in ids]
        if unknown:
            fail(f"history contains tasks outside family {family_name}: {unknown}")
        rule, revisit = args.rule, args.allow_revisit
        if revisit and rule not in REVISIT_RULES:
            fail(f"--allow-revisit is supported for rules {REVISIT_RULES} only")
        if not revisit and len(set(visited)) != len(visited):
            fail(f"a task was evaluated twice: {visited}")
        unvisited = [t for t in ids if t not in visited]
        if not revisit and not unvisited:
            fail("family exhausted: every task is already visited")
        record["allow_revisit"] = revisit

        def random_pick() -> str:
            if revisit:
                return core.pick_random_any(family_name, ids, args.salt, n)
            return core.pick_random(family_name, ids, visited, args.salt)

        if rule == "random":
            chosen = random_pick()
            record["salt"] = args.salt
        elif rule == "probe":
            scores, n_trials = read_probe(iter_dir, ids if revisit else unvisited, args.probe_score)
            record.update({"scores": scores, "probe_score": args.probe_score, "probe_trials": n_trials, "salt": args.salt})
            if max(scores.values()) > 0:
                chosen = core.argmax_in_order(scores, ids)
            else:
                chosen = random_pick()
                record["fallback"] = "random: no pass/fail split on any probed task"
        elif rule == "knn":
            descriptors = {t["id"]: core.describe(t, descriptors_dir) for t in family["tasks"]}
            scores = core.knn_scores(family, history, descriptors)
            chosen = core.argmax_in_order(scores, ids)
            record["scores"] = scores
        else:
            committee = committee_of(iter_dir)
            if not committee:
                fail(f"rule {rule!r} judges the programs of round {n}: write {iter_dir}/candidate_*/code.py first")
            if rule == "prompt":
                choice_path = Path(args.choice) if args.choice else iter_dir / "task_choice.json"
                choice = validate_choice(choice_path, unvisited, [c["candidate"] for c in committee])
                chosen = choice["task"]
                record["choice"] = choice
                record["choice_sha256"] = core.file_sha256(choice_path)
            else:  # harness
                if not args.proxy:
                    fail("rule 'harness' needs --proxy path/to/proxy.py")
                proxy = Path(args.proxy)
                if not proxy.exists():
                    fail(f"proxy {proxy} not found")
                record["proxy_sha256"] = core.lock_proxy(run_dir, proxy, n, args.unlock)
                ctx = build_context(doc, family_name, family, history, n, run_dir, descriptors_dir)
                scores = core.run_proxy(proxy, ctx, unvisited)
                chosen = core.argmax_in_order(scores, ids)
                record["scores"] = scores

    task = core.task_by_id(family)[chosen]
    record.update({
        "rule": rule, "task_id": chosen, "suite": task["suite"], "task": task["task"],
        "seeds": doc["dev_seeds"], "visited_before": visited,
        "unvisited_before": [t for t in ids if t not in visited],
        "script_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(), "created_at": now(),
    })
    out = iter_dir / "task.json"
    if out.exists():
        previous = json.loads(out.read_text())
        if previous.get("task_id") != chosen or previous.get("rule") != rule:
            fail(f"{out} already records {previous.get('rule')} -> {previous.get('task_id')}; refusing to overwrite")
        print(f"unchanged: {out}")
    else:
        out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {out}")
    print(f"round {n}: rule={rule}  task={chosen}")
    if "scores" in record:
        for t, s in sorted(record["scores"].items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {s:8.4f}  {t}")
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    history = core.load_history(run_dir, before_round=args.round) if args.round else []
    ctx = build_context(doc, family_name, family, history, args.round, run_dir,
                        Path(args.descriptors_dir) if args.descriptors_dir else None)
    text = json.dumps(ctx, indent=2)
    out = run_dir / f"iter_{args.round:02d}" / "context.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    print(f"wrote {out}", file=sys.stderr)
    print(text)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    ids = core.task_ids(family)
    problems: list[str] = []
    seen: dict[str, int] = {}
    rules, salts, revisits = set(), set(), set()
    iters = sorted((d for d in run_dir.glob("iter_[0-9][0-9]")), key=core.round_number)
    if not iters:
        fail(f"no iter_NN directories in {run_dir}")
    expected_visited: list[str] = []
    for d in iters:
        n = core.round_number(d)
        tj = d / "task.json"
        summary = d / "iter_summary.json"
        if not tj.exists():
            if summary.exists():
                problems.append(f"{d.name}: evaluated but has no task.json (task was not chosen by select_task.py)")
            continue
        rec = json.loads(tj.read_text())
        if rec.get("round") != n or rec.get("family") != family_name:
            problems.append(f"{d.name}: task.json round/family mismatch ({rec.get('round')}, {rec.get('family')})")
        if rec.get("task_id") not in ids:
            problems.append(f"{d.name}: task {rec.get('task_id')} is not in family {family_name}")
        if rec.get("allow_revisit"):
            revisits.add(n)
        elif rec.get("task_id") in seen:
            problems.append(f"{d.name}: task {rec['task_id']} already used in iter_{seen[rec['task_id']]:02d}")
        seen[rec.get("task_id")] = n
        if n == 0 and rec.get("task_id") != family["round0"]:
            problems.append(f"iter_00: task {rec.get('task_id')} != family round0 {family['round0']}")
        if n > 0:
            rules.add(rec.get("requested_rule"))
            if rec.get("rule") != rec.get("requested_rule"):
                problems.append(f"{d.name}: rule {rec.get('rule')} != requested {rec.get('requested_rule')}")
            if rec.get("rule") in ("random", "probe"):
                salts.add(rec.get("salt"))
            if rec.get("visited_before") != expected_visited:
                problems.append(f"{d.name}: visited_before {rec.get('visited_before')} != earlier rounds {expected_visited}")
        if sorted(rec.get("seeds", [])) != sorted(doc["dev_seeds"]):
            problems.append(f"{d.name}: seeds {rec.get('seeds')} != dev seeds {doc['dev_seeds']}")
        if summary.exists():
            s = json.loads(summary.read_text())
            if f"{s['suite']}/{s['task']}" != rec.get("task_id"):
                problems.append(f"{d.name}: evaluated on {s['suite']}/{s['task']} but task.json says {rec.get('task_id')}")
            if sorted(s["trial_seeds"]) != sorted(doc["dev_seeds"]):
                problems.append(f"{d.name}: evaluated on seeds {sorted(s['trial_seeds'])}, expected {doc['dev_seeds']}")
        expected_visited.append(rec.get("task_id"))
    selected = {core.round_number(d) for d in iters if core.round_number(d) > 0 and (d / "task.json").exists()}
    if revisits and revisits != selected:
        problems.append(f"only rounds {sorted(revisits)} allowed revisits; a run must use one setting")
    if len(rules) > 1:
        problems.append(f"rounds used different rules: {sorted(map(str, rules))}")
    if len(salts) > 1:
        problems.append(f"random rounds used different salts: {sorted(map(str, salts))}")
    if problems:
        print("FAIL")
        for p in problems:
            print("  " + p)
        return 1
    listing = ", ".join("iter_%02d=%s" % (n, t.split("/")[-1][:32]) for t, n in seen.items())
    print(f"OK: {len(seen)} round(s), tasks {listing}")
    return 0


def cmd_eval_cmd(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    iter_dir = run_dir / f"iter_{args.round:02d}"
    tj = iter_dir / "task.json"
    if not tj.exists():
        fail(f"{tj} missing: run select_task.py select first")
    rec = json.loads(tj.read_text())
    print(_eval_command(iter_dir, rec["suite"], rec["task"], rec["seeds"], args.gpu, args.parallel))
    return 0


def cmd_init_from(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir, ref = Path(args.run_dir).resolve(), Path(args.from_run).resolve()
    src = ref / "iter_00"
    if not (src / "iter_summary.json").exists():
        fail(f"{src} has no iter_summary.json: the reference run's round 0 is not finished")
    summary = json.loads((src / "iter_summary.json").read_text())
    if f"{summary['suite']}/{summary['task']}" != family["round0"]:
        fail(f"reference iter_00 ran {summary['suite']}/{summary['task']}, family round0 is {family['round0']}")
    dst = run_dir / "iter_00"
    if dst.exists() and any(dst.iterdir()):
        fail(f"{dst} is not empty; refusing to overwrite")
    skip = shutil.ignore_patterns("*.mp4", "*.npz", "*.npy")
    shutil.copytree(src, dst, ignore=skip, dirs_exist_ok=True)
    (run_dir / "init_from.json").write_text(json.dumps({
        "from_run": str(ref), "copied": "iter_00 (candidates, traces, iter_summary.json; videos/arrays skipped)",
        "iter_summary_sha256": core.file_sha256(src / "iter_summary.json"), "created_at": now()}, indent=2) + "\n")
    print(f"copied {src} -> {dst}")
    return 0


def _letters(n: int) -> list[str]:
    return [f"candidate_{chr(ord('A') + i)}" for i in range(n)]


def _tournament_dir(run_dir: Path, round_idx: int | None) -> Path:
    """final/ for the end-of-run tournament, iter_NN/score/ for the family score of one round."""
    return run_dir / "final" if round_idx is None else run_dir / f"iter_{round_idx:02d}" / "score"


def cmd_final_plan(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    rounds = core.finished_rounds(run_dir)
    if not rounds:
        fail(f"no finished rounds in {run_dir}")
    last = rounds[-1] if args.round is None else run_dir / f"iter_{args.round:02d}"
    if last not in rounds:
        fail(f"{last} is not a finished round")
    selection = last / "selection.json"
    if not selection.exists():
        fail(f"{selection} missing: run scripts/tc1/select_parents.py select --rule top3 --iter-dir {last}")
    parents = json.loads(selection.read_text())["parents"]
    default = doc["dev_seeds"][:3] if args.round is None else doc.get("score_seeds", doc["dev_seeds"][:3])
    seeds = [int(s) for s in (args.seeds or default)]
    final = _tournament_dir(run_dir, args.round)
    plan_path = final / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old["parents"] != parents or old["seeds"] != seeds or old["family"] != family_name:
            fail(f"{plan_path} exists with different parents/seeds/family; refusing to change a started tournament")
        plan = old
    else:
        names = _letters(len(parents))
        mapping = dict(zip(names, parents))
        tasks = []
        for k, t in enumerate(family["tasks"]):
            tdir = final / f"task_{k:02d}"
            for name, parent in mapping.items():
                (tdir / name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(run_dir / parent / "code.py", tdir / name / "code.py")
            tasks.append({"k": k, "id": t["id"], "suite": t["suite"], "task": t["task"],
                          "language_static": t["language_static"], "dir": str(tdir)})
        plan = {"family": family_name, "parents": parents, "map": mapping, "seeds": seeds, "tasks": tasks,
                "created_at": now()}
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    for t in plan["tasks"]:
        print(_eval_command(Path(t["dir"]), t["suite"], t["task"], plan["seeds"], args.gpu, args.parallel))
    return 0


def cmd_final_pick(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).resolve()
    final = _tournament_dir(run_dir, args.round)
    plan_path = final / "plan.json"
    if not plan_path.exists():
        fail(f"{plan_path} missing: run final-plan (or score-plan) first")
    plan = json.loads(plan_path.read_text())
    outcomes: dict[str, dict[str, list[int]]] = {}
    missing = []
    for t in plan["tasks"]:
        d = Path(t["dir"])
        if not (d / "iter_summary.json").exists():
            missing.append(t["id"])
            continue
        r = core.load_round(d)
        if r["task_id"] != t["id"] or r["seeds"] != sorted(plan["seeds"]):
            fail(f"{d}: evaluated on {r['task_id']} seeds {r['seeds']}, plan says {t['id']} seeds {plan['seeds']}")
        outcomes[t["id"]] = {plan["map"][c]: v for c, v in r["outcomes"].items()}
    if missing:
        fail(f"tournament not finished: no results for {missing}")
    sources = sorted(plan["map"].values())
    per_task = {src: {tid: sum(o[src]) / len(o[src]) for tid, o in outcomes.items()} for src in sources}
    macro = {src: sum(v.values()) / len(v) for src, v in per_task.items()}
    total = {src: sum(sum(o[src]) for o in outcomes.values()) for src in sources}
    ranking = sorted(sources, key=lambda s: (-macro[s], -total[s], s))
    winner = ranking[0]
    best = {
        "rule": "highest macro-average pass rate over all family tasks; ties: total passes, then name",
        "family": plan["family"], "seeds": plan["seeds"], "best": winner,
        "code_path": str(run_dir / winner / "code.py"), "macro_pass": round(macro[winner], 4),
        "ranking": [{"candidate": s, "macro_pass": round(macro[s], 4), "total_passes": total[s],
                     "per_task": {k: round(v, 4) for k, v in per_task[s].items()}} for s in ranking],
        "D_prog_per_task": {tid: core.committee_metrics(o)["D_prog"] for tid, o in outcomes.items()},
        "created_at": now(),
    }
    (final / "best.json").write_text(json.dumps(best, indent=2) + "\n")
    (final / "matrix.json").write_text(json.dumps({
        "family": plan["family"],
        "tasks": [{k: t[k] for k in ("id", "suite", "task", "language_static")} for t in plan["tasks"]],
        "candidates": sources, "outcomes": outcomes}, indent=2) + "\n")
    print(f"best: {winner}  macro pass {macro[winner]:.3f}  -> {final / 'best.json'}")
    for s in ranking:
        print(f"  {s:28s} macro {macro[s]:.3f}  passes {total[s]}")
    return 0


def cmd_heldout_cmds(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    best_path = run_dir / "final" / "best.json"
    if not best_path.exists():
        fail(f"{best_path} missing: run final-pick first")
    best = json.loads(best_path.read_text())
    if best["family"] != family_name:
        fail(f"best.json is for family {best['family']}, not {family_name}")
    seeds = " ".join(str(s) for s in doc["heldout_seeds"])
    for t in family["tasks"]:
        print(f".venv-libero/bin/python3 scripts/libero/run_fix_loop_validation.py --suite {t['suite']} --task {t['task']} "
              f"--gpu {args.gpu} --fix-code {best['code_path']} --output-dir {run_dir / 'heldout'} --seeds {seeds} --resume")
    return 0


def cmd_heldout_summary(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    run_dir = Path(args.run_dir).resolve()
    want = sorted(doc["heldout_seeds"])
    per_task, missing = {}, []
    for t in family["tasks"]:
        manifests = sorted((run_dir / "heldout" / t["suite"] / t["task"] / "runs").glob("*/manifest.json"))
        manifests = [m for m in manifests if sorted(json.loads(m.read_text())["identity"]["seeds"]) == want]
        if len(manifests) != 1:
            missing.append(f"{t['id']} ({len(manifests)} matching manifests)")
            continue
        m = json.loads(manifests[0].read_text())
        if m.get("status") != "complete" or m["trials"] != len(want):
            missing.append(f"{t['id']} (status {m.get('status')}, {m['trials']}/{len(want)} trials)")
            continue
        per_task[t["id"]] = {"passes": m["passes"], "trials": m["trials"], "pass_rate": m["pass_rate"]}
    if missing:
        fail("held-out not finished: " + "; ".join(missing))
    macro = sum(v["pass_rate"] for v in per_task.values()) / len(per_task)
    out = {"family": family_name, "seeds": want, "macro_pass": round(macro, 4), "per_task": per_task, "created_at": now()}
    (run_dir / "heldout.json").write_text(json.dumps(out, indent=2) + "\n")
    for tid, v in per_task.items():
        print(f"  {v['passes']:2d}/{v['trials']}  {tid}")
    print(f"macro-average held-out pass rate: {macro:.3f}  -> {run_dir / 'heldout.json'}")
    return 0


def cmd_render_prompt(args: argparse.Namespace) -> int:
    doc, family_name, family = load_family(args)
    base = (PROMPTS / "family_evosearch_prompt.md").read_text()
    arm_file = PROMPTS / f"arm_{args.arm}.md"
    if not arm_file.exists():
        fail(f"no arm instructions {arm_file}; arms: {[p.stem[4:] for p in sorted(PROMPTS.glob('arm_*.md'))]}")
    arm_text = arm_file.read_text()
    if args.allow_revisit:
        if args.arm not in REVISIT_RULES:
            fail(f"--allow-revisit is supported for arms {REVISIT_RULES} only")
        arm_text = arm_text.replace("--round N", "--round N --allow-revisit")
    text = base.replace("{{ARM_SECTION}}", arm_text)
    values = {"FAMILY": family_name, "ARM": args.arm, "REP": str(args.rep), "RELATION": family["relation"],
              "N_TASKS": str(len(family["tasks"])), "ROUND0": family["round0"],
              "REF_RUN": args.ref_run or "none (this run writes iter_00 and is the reference run)",
              "DEV_SEEDS": " ".join(map(str, doc["dev_seeds"])),
              "HELDOUT_SEEDS": " ".join(map(str, doc["heldout_seeds"]))}
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", v)
    if "{{" in text:
        fail("unreplaced placeholder in rendered prompt: " + text[text.index("{{"):][:40])
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, family=True, rnd=False):
        if family:
            sp.add_argument("--family", required=True)
            sp.add_argument("--families", help="families.json override (tests)")
        sp.add_argument("--run-dir", required=True)
        if rnd:
            sp.add_argument("--round", type=int, required=True)
        sp.add_argument("--descriptors-dir")

    s = sub.add_parser("select"); common(s, rnd=True)
    s.add_argument("--rule", choices=RULES, required=True)
    s.add_argument("--salt", default="r0", help="random rule: repetition salt, e.g. r1")
    s.add_argument("--choice", help="prompt rule: task_choice.json (default iter_NN/task_choice.json)")
    s.add_argument("--proxy", help="harness rule: proxy.py")
    s.add_argument("--unlock", action="store_true", help="harness rule: accept a changed proxy (logged)")
    s.add_argument("--probe-score", choices=core.PROBE_SCORES, default="split", help="probe rule: how a probed task is scored")
    s.add_argument("--allow-revisit", action="store_true", help="random/probe: a task may be selected again")
    pp = sub.add_parser("probe-plan"); common(pp, rnd=True)
    pp.add_argument("--allow-revisit", action="store_true"); pp.add_argument("--probe-seeds", type=int, default=1); pp.add_argument("--gpu", default="0"); pp.add_argument("--parallel", default="2")
    c = sub.add_parser("context"); common(c, rnd=True)
    v = sub.add_parser("verify"); common(v)
    e = sub.add_parser("eval-cmd"); common(e, family=False, rnd=True)
    e.add_argument("--gpu", default="0"); e.add_argument("--parallel", default="2")
    i = sub.add_parser("init-from"); common(i)
    i.add_argument("--from-run", required=True)
    f = sub.add_parser("final-plan"); common(f)
    f.add_argument("--seeds", type=int, nargs="+"); f.add_argument("--gpu", default="0"); f.add_argument("--parallel", default="2")
    f.set_defaults(round=None)
    fp = sub.add_parser("final-pick"); common(fp, family=False); fp.set_defaults(round=None)
    sc = sub.add_parser("score-plan"); common(sc, rnd=True)
    sc.add_argument("--seeds", type=int, nargs="+"); sc.add_argument("--gpu", default="0"); sc.add_argument("--parallel", default="2")
    sk = sub.add_parser("score-pick"); common(sk, family=False, rnd=True)
    hc = sub.add_parser("heldout-cmds"); common(hc); hc.add_argument("--gpu", default="0")
    hs = sub.add_parser("heldout-summary"); common(hs)
    r = sub.add_parser("render-prompt")
    r.add_argument("--family", required=True); r.add_argument("--families")
    r.add_argument("--arm", choices=RULES, required=True); r.add_argument("--rep", type=int, required=True)
    r.add_argument("--ref-run"); r.add_argument("--allow-revisit", action="store_true")
    return p


COMMANDS = {"probe-plan": cmd_probe_plan, "select": cmd_select, "context": cmd_context, "verify": cmd_verify, "eval-cmd": cmd_eval_cmd,
            "init-from": cmd_init_from, "final-plan": cmd_final_plan, "final-pick": cmd_final_pick,
            "score-plan": cmd_final_plan, "score-pick": cmd_final_pick,
            "heldout-cmds": cmd_heldout_cmds, "heldout-summary": cmd_heldout_summary,
            "render-prompt": cmd_render_prompt}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
