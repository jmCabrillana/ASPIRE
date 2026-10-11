#!/usr/bin/env python3
r"""Skill-library build with disagreement-based task selection.

    family F ──split(eval-split)──► A (train pool)          B (held-out, same in every arm)
                                     │                        │
    L_0 = ∅ | copy(--library)          ▼                        ▼  n ∈ C
    round t:  plan ─► K programs/task from L_t ─► probe ─► select x_t ─► repair(x_t) + promote ─► L_{t+1}
                                                                   │
                                                 eval-plan ─► K programs/task from L_n ─► eval ─► SR_n

    o     : Π × A × S -> {0,1}                                      outcome of program π on task x, seed s
    D(x)  : A -> [0,1]      D(x) = mean_s H2( k_s / K ),  k_s = Σ_π o(π, x, s)        (split)
    T(x)  : A -> R_+        T(x) = mean_s H( sig(π, x, s) )_π                          (trace)
    x_t   = U( argmax_{x ∈ A \ solved} D(x) )                    U = salt-seeded uniform pick among the ties
    solved(x) ⇔ ∃π ∀s ∈ S_probe : o(π, x, s) = 1                  repair checked on S_check, S_check ∩ S_debug = ∅
    SR_n  = mean_{x ∈ B} max_π mean_{s ∈ S_eval} o(π, x, s)     sr_mean: mean_π instead of max_π
    cov_t = |solved| / |A|

    command    key              default    selects
    plan       --pool           -          family F (round 0)
               --eval-split     3          |B|; split hash-seeded by F only
               --library        live       source of L_0; "empty" gives L_0 = ∅; round 0 activates L_0
               --checkpoints    any        C, the values of n where eval is allowed
               --k              4          programs per task
               --repair         fixloop    fixloop (fix_code.py) | evosearch (evosearch_best_code.py, A = best probe program)
    probe      --seeds          51 52      S_probe
    select     --rule           -          random (U over A) | unsolved (U over A \ solved) | split (D) | trace (T)
               --salt           r1         seed of U
               -                -          repair dirs: R/round_tt/fixloop/{<suite>/<task>, debug}, promotion ledger
    repaired   --code           TASK_DIR   repaired program for x_t (default: the repair's output file); saves L_{t+1}
               --seeds          71 72      S_check
    eval-plan  -                -          lists R/eval/after_nn/<task> for B
    eval       --seeds          1 2 3      S_eval
    activate   -                -          live library <- L_n, n = rounds finished
    report     --target         max cov    regret = Σ_t max(0, target - cov_t); SR_n per checkpoint

The live library is .claude/libero/skills. Every command that reads it fails unless it equals the run's L_n.
Repair: fix-loop subagent + record_skill_promotion.py (prompts/library_round.md).
"""

from __future__ import annotations

import argparse
import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from core import fail  # noqa: E402

REPO = core.HERE.parents[1]
RULES = ("random", "unsolved", "split", "trace")
REPAIRS = {"fixloop": "fix_code.py", "evosearch": "evosearch_best_code.py"}
SKILLS = REPO / ".claude" / "libero" / "skills"
DEBUG_SEEDS = set(range(51, 66))


# ---- state --------------------------------------------------------------------------------------------------------

def state_path(run: Path) -> Path:
    return run / "state.json"


def load_state(run: Path) -> dict:
    return json.loads(state_path(run).read_text())


def save_state(run: Path, state: dict) -> None:
    state_path(run).write_text(json.dumps(state, indent=2) + "\n")


def round_dir(run: Path, t: int) -> Path:
    return run / f"round_{t:02d}"


def fix_build(run: Path, t: int) -> Path:
    return round_dir(run, t) / "fixloop"


def fix_task_dir(run: Path, t: int, task: str) -> Path:
    return fix_build(run, t).joinpath(*task.split("/", 1))


def unsolved(state: dict) -> list[str]:
    return [t for t in state["pool"] if t not in state["solved"]]


def task_dir_name(task: str) -> str:
    return task.replace("/", "__")


# ---- skill library ------------------------------------------------------------------------------------------------

def library_dir(run: Path, n: int) -> Path:
    return run / "library" / f"L{n:02d}"


def tree_digest(root: Path) -> str:
    """Hash of every file's relative path and bytes (editor checkpoint folders ignored)."""
    h = hashlib.sha256()
    for f in sorted(root.rglob("*")):
        rel = f.relative_to(root)
        if f.is_file() and ".ipynb_checkpoints" not in rel.parts:
            h.update(str(rel).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def snapshot_library(run: Path, n: int) -> None:
    dst = library_dir(run, n)
    if dst.exists():
        fail(f"{dst} already exists")
    shutil.copytree(SKILLS, dst, ignore=shutil.ignore_patterns(".ipynb_checkpoints"))


def require_active(run: Path, state: dict) -> None:
    n = len(state["rounds"])
    if tree_digest(SKILLS) != tree_digest(library_dir(run, n)):
        fail(f"{SKILLS} is not this run's library L{n:02d}; run: library_loop.py activate {run}")


# ---- simulator ----------------------------------------------------------------------------------------------------

def run_eval(task_dir: Path, suite: str, task: str, seeds: list[int], gpu: str) -> None:
    """evosearch_eval.py on task_dir/candidate_*/code.py; writes task_dir/iter_summary.json. Skipped when it exists."""
    if (task_dir / "iter_summary.json").exists():
        return
    cmd = [str(REPO / ".venv-libero/bin/python3"), "scripts/libero/evosearch_eval.py", "--iter-dir", str(task_dir),
           "--suite", suite, "--task", task, "--trials", str(len(seeds)), "--trial-seeds", *map(str, seeds),
           "--sim-gpus", gpu, "--parallel-per-gpu", "2", "--no-highlights"]
    env = dict(os.environ, MUJOCO_GL="egl", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    env.pop("ANTHROPIC_API_KEY", None)
    with (task_dir / "eval.log").open("w") as log:
        code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0 or not (task_dir / "iter_summary.json").exists():
        fail(f"eval failed for {suite}/{task} (exit {code}); see {task_dir / 'eval.log'}")


def passing_program(outcomes: dict[str, list[int]]) -> str | None:
    """First candidate (by name) that passed every seed."""
    return next((c for c, row in sorted(outcomes.items()) if all(row)), None)


# ---- commands -----------------------------------------------------------------------------------------------------

def cmd_plan(args) -> int:
    run = Path(args.run).resolve()
    if args.round == 0:
        if state_path(run).exists():
            fail(f"{run} already exists")
        ids = core.task_ids(core.get_family(core.load_families(), args.pool))
        if not 0 < args.eval_split < len(ids):
            fail(f"--eval-split must be between 1 and {len(ids) - 1}")
        held = set(core.random_order(args.pool, ids, "eval-split")[:args.eval_split])
        source = None if args.library == "empty" else Path(args.library).resolve() if args.library else SKILLS
        run.mkdir(parents=True)
        if source is None:
            library_dir(run, 0).mkdir(parents=True)
        else:
            shutil.copytree(source, library_dir(run, 0), ignore=shutil.ignore_patterns(".ipynb_checkpoints"))
        save_state(run, {"pool": [t for t in ids if t not in held], "eval_pool": [t for t in ids if t in held],
                         "family": args.pool, "k": args.k, "repair": args.repair, "checkpoints": args.checkpoints, "library_source": str(source or "empty"),
                         "solved": {}, "rounds": [], "evals": {}})
        cmd_activate(args)
    state = load_state(run)
    if len(state["rounds"]) != args.round:
        fail(f"round {args.round} cannot start: {len(state['rounds'])} round(s) finished")
    require_active(run, state)
    rd = round_dir(run, args.round)
    for task in unsolved(state):
        (rd / "programs" / task_dir_name(task)).mkdir(parents=True, exist_ok=True)
    print(f"round {args.round}: {len(unsolved(state))} unsolved of {len(state['pool'])}; write {state['k']} programs "
          f"(candidate_A..) into each of:")
    for task in unsolved(state):
        print(f"  {rd / 'programs' / task_dir_name(task)}")
    return 0


def cmd_probe(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    require_active(run, state)
    rd = round_dir(run, args.round)
    seeds = sorted(args.seeds)
    probe = {"seeds": seeds, "tasks": {}}
    for task in unsolved(state):
        tdir = rd / "programs" / task_dir_name(task)
        programs = sorted(d.name for d in tdir.glob("candidate_*") if (d / "code.py").exists())
        if len(programs) != state["k"]:
            fail(f"{tdir}: {len(programs)} programs, expected {state['k']}")
        suite, name = task.split("/", 1)
        run_eval(tdir, suite, name, seeds, args.gpu)
        outcomes = core.load_round(tdir)["outcomes"]
        signatures = core.load_signatures(tdir, programs, seeds)
        probe["tasks"][task] = {
            "outcomes": outcomes, "stages": {c: [s.split("|")[0] for s in sig] for c, sig in signatures.items()},
            "split": core.probe_score(outcomes, "split"), "trace": core.trace_score(signatures, "trace"),
            "passing": passing_program(outcomes),
        }
        if probe["tasks"][task]["passing"]:
            state["solved"][task] = {"program": str(tdir / probe["tasks"][task]["passing"] / "code.py"),
                                     "round": args.round, "how": "zero-shot"}
    (rd / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    save_state(run, state)
    print(f"round {args.round}: probed {len(probe['tasks'])} tasks on seeds {seeds}; "
          f"solved so far {len(state['solved'])}/{len(state['pool'])}")
    for task, p in probe["tasks"].items():
        passes = "".join(str(sum(p["outcomes"][c])) for c in sorted(p["outcomes"]))
        print(f"  {task.split('/')[-1][:40]:40s} passes {passes}  split {p['split']:.2f}  trace {p['trace']:.2f}"
              + ("  SOLVED" if p["passing"] else ""))
    return 0


def cmd_select(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    rd = round_dir(run, args.round)
    probe = json.loads((rd / "probe.json").read_text())
    todo = unsolved(state)
    if not todo:
        fail("every task is solved")
    scores = None
    if args.rule == "random":
        chosen = core.pick_random_any(state["family"], state["pool"], args.salt, args.round)
    elif args.rule == "unsolved":
        chosen = core.pick_random_any(state["family"], todo, args.salt, args.round)
    else:
        scores = {t: probe["tasks"][t][args.rule] for t in todo}
        best = [t for t in todo if scores[t] == max(scores.values())]
        chosen = core.pick_random_any(state["family"], best, args.salt, args.round)
    task_dir = fix_task_dir(run, args.round, chosen)
    outcomes = probe["tasks"].get(chosen, {}).get("outcomes")
    start = (str(rd / "programs" / task_dir_name(chosen) / min(outcomes, key=lambda c: (-sum(outcomes[c]), c)) / "code.py")
             if outcomes else None)
    fix = {"method": state["repair"], "build_dir": str(fix_build(run, args.round)), "task_dir": str(task_dir),
           "debug_dir": str(fix_build(run, args.round) / "debug"), "code": str(task_dir / REPAIRS[state["repair"]]),
           "start_code": start}
    (rd / "selection.json").write_text(json.dumps({"rule": args.rule, "task": chosen, "scores": scores,
                                                   "salt": args.salt, "repair": fix}, indent=2) + "\n")
    print(f"round {args.round}: {args.rule} -> {chosen}" + ("" if chosen in todo else "  (already solved)"))
    print("\n".join(f"  {k} = {v}" for k, v in fix.items()))
    return 0


def cmd_repaired(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    if len(state["rounds"]) != args.round:
        fail(f"round {args.round} cannot be recorded: {len(state['rounds'])} round(s) finished")
    seeds = sorted(args.seeds)
    if DEBUG_SEEDS & set(seeds):
        fail(f"check seeds {sorted(DEBUG_SEEDS & set(seeds))} are fix-loop debug seeds (51-65)")
    rd = round_dir(run, args.round)
    selection = json.loads((rd / "selection.json").read_text())
    task = selection["task"]
    tdir = rd / "repair" / task_dir_name(task)
    (tdir / "candidate_fix").mkdir(parents=True, exist_ok=True)
    code = Path(args.code or selection["repair"]["code"])
    (tdir / "candidate_fix" / "code.py").write_bytes(code.read_bytes())
    suite, name = task.split("/", 1)
    run_eval(tdir, suite, name, seeds, args.gpu)
    outcomes = core.load_round(tdir)["outcomes"]
    solved = passing_program(outcomes) is not None
    if solved:
        state["solved"][task] = {"program": str(tdir / "candidate_fix" / "code.py"), "round": args.round, "how": "repair"}
    snapshot_library(run, args.round + 1)
    state["rounds"].append({"round": args.round, "task": task, "repair_solved": solved, "check_seeds": seeds,
                            "library_changed": tree_digest(library_dir(run, args.round)) != tree_digest(SKILLS),
                            "coverage": len(state["solved"]) / len(state["pool"])})
    save_state(run, state)
    print(f"round {args.round}: repair of {task.split('/')[-1]} {'solved' if solved else 'did not solve'} the task; "
          f"coverage {state['rounds'][-1]['coverage']:.2f}")
    return 0


def eval_dir(run: Path, n: int) -> Path:
    return run / "eval" / f"after_{n:02d}"


def cmd_eval_plan(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    n = len(state["rounds"])
    if state["checkpoints"] is not None and n not in state["checkpoints"]:
        fail(f"{n} round(s) finished, not a checkpoint {state['checkpoints']}")
    require_active(run, state)
    for task in state["eval_pool"]:
        (eval_dir(run, n) / task_dir_name(task)).mkdir(parents=True, exist_ok=True)
    print(f"checkpoint {n}: write {state['k']} zero-shot programs (candidate_A..) from L{n:02d} into each of:")
    for task in state["eval_pool"]:
        print(f"  {eval_dir(run, n) / task_dir_name(task)}")
    return 0


def cmd_eval(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    n = len(state["rounds"])
    require_active(run, state)
    seeds = sorted(args.seeds)
    tasks = {}
    for task in state["eval_pool"]:
        tdir = eval_dir(run, n) / task_dir_name(task)
        programs = sorted(d.name for d in tdir.glob("candidate_*") if (d / "code.py").exists())
        if len(programs) != state["k"]:
            fail(f"{tdir}: {len(programs)} programs, expected {state['k']} (run eval-plan first)")
        suite, name = task.split("/", 1)
        run_eval(tdir, suite, name, seeds, args.gpu)
        rates = {c: sum(row) / len(row) for c, row in core.load_round(tdir)["outcomes"].items()}
        tasks[task] = {"rates": rates, "best": max(rates.values()), "mean": sum(rates.values()) / len(rates)}
    result = {"after_rounds": n, "seeds": seeds, "tasks": tasks,
              "sr": sum(t["best"] for t in tasks.values()) / len(tasks),
              "sr_mean": sum(t["mean"] for t in tasks.values()) / len(tasks)}
    (eval_dir(run, n) / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    state["evals"][str(n)] = {"sr": result["sr"], "sr_mean": result["sr_mean"], "seeds": seeds}
    save_state(run, state)
    print(f"checkpoint {n}: SR {result['sr']:.2f} (best program per task), {result['sr_mean']:.2f} (mean program) "
          f"over {len(tasks)} held-out tasks, seeds {seeds}")
    return 0


def cmd_activate(args) -> int:
    run = Path(args.run).resolve()
    n = len(load_state(run)["rounds"])
    for child in SKILLS.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    shutil.copytree(library_dir(run, n), SKILLS, dirs_exist_ok=True)
    print(f"{SKILLS} <- {library_dir(run, n)}")
    return 0


def cmd_report(args) -> int:
    curves = {}
    for r in args.runs:
        state = load_state(Path(r).resolve())
        curves[r] = [x["coverage"] for x in state["rounds"]]
        print(f"{r}: " + "  ".join(f"r{x['round']} {x['task'].split('/')[-1][:20]} {'+' if x['repair_solved'] else '-'} "
                                   f"{x['coverage']:.2f}" for x in state["rounds"]))
    target = args.target if args.target is not None else max((max(c) for c in curves.values() if c), default=0.0)
    for r, c in curves.items():
        regret = sum(max(0.0, target - x) for x in c)
        hit = next((i for i, x in enumerate(c) if x >= target), None)
        print(f"  {r}: regret {regret:.2f} over {len(c)} rounds vs coverage {target:.2f}; "
              + ("not reached" if hit is None else f"reached at round {hit}"))
    for r in args.runs:
        evals = load_state(Path(r).resolve()).get("evals", {})
        print(f"  {r}: SR on B " + ("  ".join(f"n={n} {e['sr']:.2f}/{e['sr_mean']:.2f}"
                                             for n, e in sorted(evals.items(), key=lambda x: int(x[0])))
                                    or "(no checkpoint evaluated)"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan"); plan.add_argument("run"); plan.add_argument("--round", type=int, required=True)
    plan.add_argument("--pool"); plan.add_argument("--k", type=int, default=4)
    plan.add_argument("--repair", choices=sorted(REPAIRS), default="fixloop")
    plan.add_argument("--eval-split", type=int, default=3, help="tasks held out of the family as B (round 0)")
    plan.add_argument("--library", help="L0 (round 0): a directory, or 'empty'; default: the live library")
    plan.add_argument("--checkpoints", type=int, nargs="+", help="rounds finished at which B is evaluated (round 0); "
                                                                 "default: any")
    probe = sub.add_parser("probe"); probe.add_argument("run"); probe.add_argument("--round", type=int, required=True)
    probe.add_argument("--seeds", type=int, nargs="+", default=[51, 52]); probe.add_argument("--gpu", default="0")
    sel = sub.add_parser("select"); sel.add_argument("run"); sel.add_argument("--round", type=int, required=True)
    sel.add_argument("--rule", choices=RULES, required=True); sel.add_argument("--salt", default="r1")
    rep = sub.add_parser("repaired"); rep.add_argument("run"); rep.add_argument("--round", type=int, required=True)
    rep.add_argument("--code"); rep.add_argument("--gpu", default="0")
    rep.add_argument("--seeds", type=int, nargs="+", default=[71, 72])
    ep = sub.add_parser("eval-plan"); ep.add_argument("run")
    ev = sub.add_parser("eval"); ev.add_argument("run")
    ev.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3]); ev.add_argument("--gpu", default="0")
    act = sub.add_parser("activate"); act.add_argument("run")
    report = sub.add_parser("report"); report.add_argument("runs", nargs="+"); report.add_argument("--target", type=float)
    args = p.parse_args(argv)
    if args.cmd == "plan" and args.round == 0 and not args.pool:
        p.error("plan --round 0 needs --pool")
    return {"plan": cmd_plan, "probe": cmd_probe, "select": cmd_select, "repaired": cmd_repaired,
            "eval-plan": cmd_eval_plan, "eval": cmd_eval, "activate": cmd_activate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
