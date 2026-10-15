#!/usr/bin/env python3
r"""Skill-library build with disagreement-based task selection.

    family F ──split──► A (train pool)                                 B (held-out, same in every arm)
                         │                                              │
    L_0 = ∅ | copy(D)    ▼                                              ▼  any saved L_n, n ≤ rounds
    round t:  plan ─► K programs/task of A from L_t ─► probe ─► select x_t ─► repair(x_t) + promote ─► L_{t+1}
                                                                                    │
                                                  eval-plan ─► K programs/task of B from L_n ─► eval ─► SR_n

    o      : Π × T × S -> {0,1}                                     outcome of program π on task x, seed s
    D(x)   : A -> [0,1]      D(x) = mean_s H2( k_s / K ),  k_s = Σ_π o(π, x, s)       (split)
    T(x)   : A -> R_+        T(x) = mean_s H( sig(π, x, s) )_π                         (trace)
    U_t    = { x ∈ A : ¬∃π ∀s ∈ S_probe  o(π, x, s) = 1 }           unsolved under L_t, re-measured every round
    x_t    = π_salt(A)[t]                                           order: ASPIRE's top-to-bottom walk, no revisit
    x_t    = U( argmax_{x ∈ U_t} D(x) )                             split / trace; U = salt-seeded pick among ties
    m(x)   = mode of the failing stages of x                        crash < perceive < no_motion < ... (core.STAGES)
    sh(x)  = |{ y ∈ U_t : m(y) = m(x) }|
    x_t    = U( argmax_{x ∈ U_t} (D(x), sh(x), T(x)) )              lex: repairable, then shared failure, then varied
    cov_t  = 1 - |U_t| / |A|
    SR_n   = mean_{x ∈ B} mean_π mean_{s ∈ S_eval} o(π, x, s)     sr_best: max_π instead of mean_π
    S_check ∩ S_debug = S_eval ∩ S_debug = ∅,   S_debug = {51..65}
    ledger: before(x_t) = L_t,  after(x_t) = live = L_{t+1}

    command    key              default    selects
    plan       --pool           -          family F (round 0)
               --library        -          L_0: a directory, or "empty" (round 0); round 0 activates L_0
               --eval-tasks     -          B by task name (round 0)
               --eval-split     0          |B| when --eval-tasks is absent; split hash-seeded by F only
               --checkpoints    none       C: values of n evaluated during the build; any L_n can be evaluated later
               --k              3          programs per task, rewritten from L_t every round
               --repair         fixloop    fixloop (fix_code.py) | evosearch (evosearch_best_code.py)
               --repair-iters   5          repair iterations on the selected task (fixloop: rewrite + replay 51-65;
                                           evosearch: rounds)
    probe      --seeds          51..65     S_probe, every task of A
    select     --rule           order      order (π_salt) | random | unsolved | split | trace | lex
               --salt           r1         seed of U
               --force          off        select again after the repair started
               --gpu            0          GPU named in round_tt/repair_prompt.md (fixloop: prompts/repair_fixloop.md)
    repaired   --code           repair     repaired program for x_t; fixloop default: argmax_i passes(iter_i), ties
                                           to the later iteration, iter_00 = start_code; at most --repair-iters
               --seeds          71 72      S_check
    eval-plan  --after          rounds     n: the saved library L_n to evaluate (without it: only n ∈ C)
    eval       --after          rounds     n
               --seeds          1..10      S_eval
    activate   --after          rounds     live library <- L_n
    report     --target         max cov    regret = Σ_t max(0, target - cov_t); SR_n per evaluated n
               --force          off        compare runs with different F, B, K, repair, L_0 or seeds

The live library is .claude/libero/skills. Commands that read or write it fail unless it equals the expected L_n.
Repair: prompts/library_round.md step 4; promotion: record_skill_promotion.py --build-dir R/round_tt/fixloop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
from core import fail  # noqa: E402

REPO = core.HERE.parents[1]
RULES = ("order", "random", "unsolved", "split", "trace", "lex")
REPAIRS = {"fixloop": "fix_code.py", "evosearch": "evosearch_best_code.py"}
SKILLS = REPO / ".claude" / "libero" / "skills"
DEBUG_SEEDS = set(range(51, 66))
COMPARED = ("family", "pool", "eval_pool", "k", "repair", "repair_iters", "library_source", "probe_seeds")


# ---- state --------------------------------------------------------------------------------------------------------

def state_path(run: Path) -> Path:
    return run / "state.json"


def load_state(run: Path) -> dict:
    return json.loads(state_path(run).read_text())


def save_state(run: Path, state: dict) -> None:
    state_path(run).write_text(json.dumps(state, indent=2) + "\n")


def round_dir(run: Path, t: int) -> Path:
    return run / f"round_{t:02d}"


def eval_dir(run: Path, n: int) -> Path:
    return run / "eval" / f"after_{n:02d}"


def fix_build(run: Path, t: int) -> Path:
    return round_dir(run, t) / "fixloop"


def fix_task_dir(run: Path, t: int, task: str) -> Path:
    return fix_build(run, t).joinpath(*task.split("/", 1))


def task_dir_name(task: str) -> str:
    return task.replace("/", "__")


def no_debug_seeds(seeds: list[int], what: str) -> list[int]:
    if DEBUG_SEEDS & set(seeds):
        fail(f"{what} seeds {sorted(DEBUG_SEEDS & set(seeds))} are fix-loop debug seeds (51-65)")
    return sorted(seeds)


# ---- skill library ------------------------------------------------------------------------------------------------

def library_dir(run: Path, n: int) -> Path:
    return run / "library" / f"L{n:02d}"


def file_hashes(root: Path) -> dict[str, str]:
    return {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob("*"))
            if f.is_file() and ".ipynb_checkpoints" not in f.relative_to(root).parts}


def copy_library(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".ipynb_checkpoints"), dirs_exist_ok=True)


def require_library(run: Path, n: int) -> None:
    if not library_dir(run, n).is_dir():
        fail(f"{library_dir(run, n)} does not exist")
    if file_hashes(SKILLS) != file_hashes(library_dir(run, n)):
        fail(f"{SKILLS} is not this run's library L{n:02d}; run: library_loop.py activate {run} --after {n}")


def after(args, state: dict) -> int:
    n = len(state["rounds"]) if args.after is None else args.after
    if not 0 <= n <= len(state["rounds"]):
        fail(f"--after {n}: the run has libraries L00..L{len(state['rounds']):02d}")
    return n


def ledger_record(build: Path, task: str) -> dict:
    """The finished promotion of task in build/<suite>/skill_promotions; record_skill_promotion.py writes it."""
    suite, name = task.split("/", 1)
    records = [json.loads(p.read_text()) for p in sorted((build / suite / "skill_promotions").glob("*/record.json"))]
    records = [r for r in records if r.get("task") == name]
    if not records:
        fail(f"no finished promotion for {task} under {build}: run record_skill_promotion.py --build-dir {build} "
             f"begin/finish (finish --reason for a no-op)")
    return records[-1]


def skill_relative(hashes: dict[str, str]) -> dict[str, str]:
    return {k.split("skills/", 1)[1]: v for k, v in hashes.items()}


# ---- simulator ----------------------------------------------------------------------------------------------------

def run_eval(task_dir: Path, suite: str, task: str, seeds: list[int], gpu: str) -> None:
    """evosearch_eval.py on task_dir/candidate_*/code.py; writes task_dir/iter_summary.json. Skipped when it exists."""
    if (task_dir / "iter_summary.json").exists():
        return
    for stale in task_dir.glob("candidate_*/eval"):
        shutil.rmtree(stale)
    for stale in task_dir.glob("candidate_*/eval_results.json"):
        stale.unlink()
    cmd = [str(REPO / ".venv-libero/bin/python3"), "scripts/libero/evosearch_eval.py", "--iter-dir", str(task_dir),
           "--suite", suite, "--task", task, "--trials", str(len(seeds)), "--trial-seeds", *map(str, seeds),
           "--sim-gpus", gpu, "--parallel-per-gpu", os.environ.get("EVAL_PARALLEL", "2"),
           "--no-highlights"]
    env = dict(os.environ, MUJOCO_GL="egl", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    env.pop("ANTHROPIC_API_KEY", None)
    with (task_dir / "eval.log").open("w") as log:
        code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0 or not (task_dir / "iter_summary.json").exists():
        fail(f"eval failed for {suite}/{task} (exit {code}); see {task_dir / 'eval.log'}")


def programs_in(task_dir: Path, k: int, hint: str) -> list[str]:
    programs = sorted(d.name for d in task_dir.glob("candidate_*") if (d / "code.py").exists())
    if len(programs) != k:
        fail(f"{task_dir}: {len(programs)} programs, expected {k} ({hint})")
    return programs


def passing_program(outcomes: dict[str, list[int]]) -> str | None:
    return next((c for c, row in sorted(outcomes.items()) if all(row)), None)


# ---- commands -----------------------------------------------------------------------------------------------------

def create_run(run: Path, args) -> None:
    if state_path(run).exists():
        fail(f"{run} already exists")
    if not args.library:
        fail("plan --round 0 needs --library <dir>|empty")
    ids = core.task_ids(core.get_family(core.load_families(), args.pool))
    if args.eval_tasks:
        by_name = {t.split("/", 1)[1]: t for t in ids}
        unknown = [t for t in args.eval_tasks if t not in by_name and t not in ids]
        if unknown:
            fail(f"--eval-tasks not in {args.pool}: {unknown}")
        held = {by_name.get(t, t) for t in args.eval_tasks}
    else:
        held = set(core.random_order(args.pool, ids, "eval-split")[:args.eval_split])
    if len(held) >= len(ids):
        fail(f"B must leave tasks in A ({len(held)} of {len(ids)} held out)")
    source = None if args.library == "empty" else Path(args.library).resolve()
    if source is not None and not source.is_dir():
        fail(f"--library {source} is not a directory")
    library_dir(run, 0).mkdir(parents=True)
    if source is not None:
        copy_library(source, library_dir(run, 0))
    save_state(run, {"family": args.pool, "pool": [t for t in ids if t not in held],
                     "eval_pool": [t for t in ids if t in held], "k": args.k, "repair": args.repair, "repair_iters": args.repair_iters,
                     "library_source": str(source or "empty"), "checkpoints": args.checkpoints, "probe_seeds": None,
                     "unsolved": [t for t in ids if t not in held], "rounds": [], "evals": {}})
    activate(run, 0)


def cmd_plan(args) -> int:
    run = Path(args.run).resolve()
    if args.round == 0:
        create_run(run, args)
    state = load_state(run)
    if len(state["rounds"]) != args.round:
        fail(f"round {args.round} cannot start: {len(state['rounds'])} round(s) finished")
    require_library(run, args.round)
    rd = round_dir(run, args.round)
    for task in state["pool"]:
        (rd / "programs" / task_dir_name(task)).mkdir(parents=True, exist_ok=True)
    print(f"round {args.round}: write {state['k']} zero-shot programs (candidate_A..) from L{args.round:02d} "
          f"into each of:")
    for task in state["pool"]:
        print(f"  {rd / 'programs' / task_dir_name(task)}")
    return 0


def cmd_probe(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    require_library(run, args.round)
    seeds = sorted(args.seeds)
    if state["probe_seeds"] not in (None, seeds):
        fail(f"probe seeds {seeds} differ from the run's {state['probe_seeds']}")
    rd = round_dir(run, args.round)
    probe = {"seeds": seeds, "tasks": {}}
    for task in state["pool"]:
        tdir = rd / "programs" / task_dir_name(task)
        programs = programs_in(tdir, state["k"], "write them first")
        run_eval(tdir, *task.split("/", 1), seeds, args.gpu)
        outcomes = core.load_round(tdir)["outcomes"]
        signatures = core.load_signatures(tdir, programs, seeds)
        probe["tasks"][task] = {
            "outcomes": outcomes, "stages": {c: [s.split("|")[0] for s in sig] for c, sig in signatures.items()},
            "split": core.probe_score(outcomes, "split"), "trace": core.trace_score(signatures, "trace"),
            "mode": core.task_mode(signatures),
            "passing": passing_program(outcomes),
        }
    state["probe_seeds"] = seeds
    state["unsolved"] = [t for t in state["pool"] if not probe["tasks"][t]["passing"]]
    probe["coverage"] = 1 - len(state["unsolved"]) / len(state["pool"])
    (rd / "probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    save_state(run, state)
    print(f"round {args.round}: L{args.round:02d} solves {len(state['pool']) - len(state['unsolved'])}"
          f"/{len(state['pool'])} tasks of A zero-shot on seeds {seeds}")
    for task, p in probe["tasks"].items():
        passes = "".join(str(sum(p["outcomes"][c])) for c in sorted(p["outcomes"]))
        print(f"  {task.split('/')[-1][:40]:40s} passes {passes}  split {p['split']:.2f}  trace {p['trace']:.2f}"
              + ("  SOLVED" if p["passing"] else ""))
    return 0


def cmd_select(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    if len(state["rounds"]) != args.round:
        fail(f"round {args.round} is not the open round ({len(state['rounds'])})")
    require_library(run, args.round)
    rd = round_dir(run, args.round)
    if (rd / "selection.json").exists() and not args.force:
        previous = json.loads((rd / "selection.json").read_text())
        if Path(previous["repair"]["task_dir"]).exists():
            fail(f"the repair of {previous['task']} already started; pass --force to select again")
    probe = json.loads((rd / "probe.json").read_text())
    todo = state["unsolved"]
    if args.rule in ("unsolved", "split", "trace") and not todo:
        fail("every task of A is solved")
    scores = None
    if args.rule == "order":
        candidates = [core.pick_random(state["family"], state["pool"], [r["task"] for r in state["rounds"]], args.salt)]
    elif args.rule == "random":
        candidates = state["pool"]
    elif args.rule == "unsolved":
        candidates = todo
    else:
        p = probe["tasks"]
        scores = {t: ([p[t]["split"], sum(p[u]["mode"] == p[t]["mode"] for u in todo), p[t]["trace"]]
                      if args.rule == "lex" else p[t][args.rule]) for t in todo}
        candidates = [t for t in todo if scores[t] == max(scores.values())]
    chosen = core.pick_random_any(state["family"], candidates, args.salt, args.round)
    outcomes = probe["tasks"][chosen]["outcomes"]
    start = min(outcomes, key=lambda c: (-sum(outcomes[c]), c))
    task_dir = fix_task_dir(run, args.round, chosen)
    repair = {"method": state["repair"], "build_dir": str(fix_build(run, args.round)), "task_dir": str(task_dir),
              "debug_dir": str(fix_build(run, args.round) / "debug"), "code": str(task_dir / REPAIRS[state["repair"]]),
              "iterations": state["repair_iters"],
              "start_code": str(rd / "programs" / task_dir_name(chosen) / start / "code.py"),
              "probe_dir": str(rd / "programs" / task_dir_name(chosen))}
    (rd / "selection.json").write_text(json.dumps({"rule": args.rule, "task": chosen, "scores": scores,
                                                   "salt": args.salt, "repair": repair}, indent=2) + "\n")
    print(f"round {args.round}: {args.rule} -> {chosen}" + ("" if chosen in todo else "  (already solved)"))
    if state["repair"] == "fixloop":
        fields = {"GPU": args.gpu, "SUITE": chosen.split("/", 1)[0], "TASK": chosen.split("/", 1)[1],
                  "ITERS": state["repair_iters"], "TASK_DIR": task_dir, "PROBE_DIR": repair["probe_dir"],
                  "MODES": "\n".join(f"  {t}: {probe['tasks'][t]['mode']}" for t in todo)}
        text = (core.HERE / "prompts" / "repair_fixloop.md").read_text()
        for key, value in fields.items():
            text = text.replace("{{" + key + "}}", str(value))
        (rd / "repair_prompt.md").write_text(text)
    print("\n".join(f"  {k} = {v}" for k, v in repair.items()))
    return 0


def kept_program(repair: dict, rd: Path, state: dict) -> Path:
    if repair["method"] != "fixloop":
        return Path(repair["code"])
    iters = sorted(Path(repair["task_dir"]).glob("iter_[0-9][0-9]"))
    if len(iters) > state["repair_iters"]:
        fail(f"{len(iters)} iterations in {repair['task_dir']}, budget {state['repair_iters']}")
    start = Path(repair["start_code"])
    probe = json.loads((rd / "probe.json").read_text())["tasks"][json.loads((rd / "selection.json").read_text())["task"]]
    scored = [(sum(probe["outcomes"][start.parent.name]), 0, start)]
    scored += [(sum(core.load_round(d)["outcomes"]["candidate_A"]), i + 1, d / "candidate_A" / "code.py")
               for i, d in enumerate(iters)]
    best = max(scored)[2]
    Path(repair["code"]).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, repair["code"])
    return best


def cmd_repaired(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    if len(state["rounds"]) != args.round:
        fail(f"round {args.round} cannot be recorded: {len(state['rounds'])} round(s) finished")
    seeds = no_debug_seeds(args.seeds, "check")
    rd = round_dir(run, args.round)
    selection = json.loads((rd / "selection.json").read_text())
    task = selection["task"]
    record = ledger_record(Path(selection["repair"]["build_dir"]), task)
    if skill_relative(record["skill_sha256_before"]) != file_hashes(library_dir(run, args.round)):
        fail(f"the promotion of {task} started from a library other than L{args.round:02d}")
    if skill_relative(record["skill_sha256_after"]) != file_hashes(SKILLS):
        fail(f"{SKILLS} changed after the promotion of {task} finished")
    tdir = rd / "repair" / task_dir_name(task)
    (tdir / "candidate_fix").mkdir(parents=True, exist_ok=True)
    code = Path(args.code) if args.code else kept_program(selection["repair"], rd, state)
    (tdir / "candidate_fix" / "code.py").write_bytes(code.read_bytes())
    run_eval(tdir, *task.split("/", 1), seeds, args.gpu)
    solved = passing_program(core.load_round(tdir)["outcomes"]) is not None
    shutil.rmtree(library_dir(run, args.round + 1), ignore_errors=True)
    copy_library(SKILLS, library_dir(run, args.round + 1))
    probe = json.loads((rd / "probe.json").read_text())
    state["rounds"].append({"round": args.round, "task": task, "rule": selection["rule"], "repair_solved": solved,
                            "check_seeds": seeds, "library_changed": bool(record["changed_skill_files"]),
                            "coverage": probe["coverage"]})
    save_state(run, state)
    print(f"round {args.round}: repair of {task.split('/')[-1]} {'passed' if solved else 'failed'} seeds {seeds}; "
          f"{len(record['changed_skill_files'])} skill files changed; saved L{args.round + 1:02d}")
    return 0


def cmd_eval_plan(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    n = after(args, state)
    if args.after is None and n not in state["checkpoints"]:
        fail(f"{n} round(s) finished, not a checkpoint {state['checkpoints']}; pass --after {n} to evaluate anyway")
    require_library(run, n)
    if not state["eval_pool"]:
        fail("this run has no held-out tasks (plan --eval-tasks)")
    for task in state["eval_pool"]:
        (eval_dir(run, n) / task_dir_name(task)).mkdir(parents=True, exist_ok=True)
    print(f"L{n:02d}: write {state['k']} zero-shot programs (candidate_A..) into each of:")
    for task in state["eval_pool"]:
        print(f"  {eval_dir(run, n) / task_dir_name(task)}")
    return 0


def cmd_eval(args) -> int:
    run = Path(args.run).resolve()
    state = load_state(run)
    n = after(args, state)
    require_library(run, n)
    seeds = no_debug_seeds(args.seeds, "eval")
    tasks = {}
    for task in state["eval_pool"]:
        tdir = eval_dir(run, n) / task_dir_name(task)
        programs_in(tdir, state["k"], "run eval-plan and write them first")
        run_eval(tdir, *task.split("/", 1), seeds, args.gpu)
        rates = {c: sum(row) / len(row) for c, row in core.load_round(tdir)["outcomes"].items()}
        tasks[task] = {"rates": rates, "best": max(rates.values()), "mean": sum(rates.values()) / len(rates)}
    result = {"library": f"L{n:02d}", "seeds": seeds, "tasks": tasks,
              "sr": sum(t["mean"] for t in tasks.values()) / len(tasks),
              "sr_best": sum(t["best"] for t in tasks.values()) / len(tasks)}
    (eval_dir(run, n) / "eval.json").write_text(json.dumps(result, indent=2) + "\n")
    state["evals"][str(n)] = {"sr": result["sr"], "sr_best": result["sr_best"], "seeds": seeds}
    save_state(run, state)
    print(f"L{n:02d}: SR {result['sr']:.3f} (mean program), {result['sr_best']:.3f} (best program) "
          f"over {len(tasks)} held-out tasks, seeds {seeds}")
    return 0


def activate(run: Path, n: int) -> None:
    for child in SKILLS.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    copy_library(library_dir(run, n), SKILLS)
    print(f"{SKILLS} <- {library_dir(run, n)}")


def cmd_activate(args) -> int:
    run = Path(args.run).resolve()
    n = after(args, load_state(run))
    if not library_dir(run, n).is_dir():
        fail(f"{library_dir(run, n)} does not exist")
    activate(run, n)
    return 0


def cmd_report(args) -> int:
    states = {r: load_state(Path(r).resolve()) for r in args.runs}
    first = next(iter(states.values()))
    differ = [k for k in COMPARED if any(s.get(k) != first.get(k) for s in states.values())]
    if differ and not args.force:
        fail(f"runs differ in {differ}; pass --force to compare anyway")
    curves = {r: [x["coverage"] for x in s["rounds"]] for r, s in states.items()}
    target = args.target if args.target is not None else max((max(c) for c in curves.values() if c), default=0.0)
    for r, s in states.items():
        c = curves[r]
        print(f"{r}: " + "  ".join(f"r{x['round']} {x['task'].split('/')[-1][:20]} {'+' if x['repair_solved'] else '-'}"
                                   f" {x['coverage']:.2f}" for x in s["rounds"]))
        hit = next((i for i, x in enumerate(c) if x >= target), None)
        print(f"  regret {sum(max(0.0, target - x) for x in c):.2f} over {len(c)} rounds vs coverage {target:.2f}; "
              + ("not reached" if hit is None else f"reached at round {hit}"))
        evals = sorted(s["evals"].items(), key=lambda e: int(e[0]))
        print("  SR on B " + ("  ".join(f"L{int(n):02d} {e['sr']:.3f}/{e['sr_best']:.3f}" for n, e in evals)
                               or "(no library evaluated)"))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    plan = sub.add_parser("plan"); plan.add_argument("run"); plan.add_argument("--round", type=int, required=True)
    plan.add_argument("--pool"); plan.add_argument("--k", type=int, default=3)
    plan.add_argument("--repair-iters", type=int, default=5)
    plan.add_argument("--repair", choices=sorted(REPAIRS), default="fixloop")
    plan.add_argument("--library"); plan.add_argument("--eval-tasks", nargs="+")
    plan.add_argument("--eval-split", type=int, default=0); plan.add_argument("--checkpoints", type=int, nargs="*", default=[])
    probe = sub.add_parser("probe"); probe.add_argument("run"); probe.add_argument("--round", type=int, required=True)
    probe.add_argument("--seeds", type=int, nargs="+", default=list(range(51, 66))); probe.add_argument("--gpu", default="0")
    sel = sub.add_parser("select"); sel.add_argument("run"); sel.add_argument("--round", type=int, required=True)
    sel.add_argument("--rule", choices=RULES, default="order"); sel.add_argument("--salt", default="r1")
    sel.add_argument("--force", action="store_true"); sel.add_argument("--gpu", default="0")
    rep = sub.add_parser("repaired"); rep.add_argument("run"); rep.add_argument("--round", type=int, required=True)
    rep.add_argument("--code"); rep.add_argument("--gpu", default="0")
    rep.add_argument("--seeds", type=int, nargs="+", default=[71, 72])
    for name in ("eval-plan", "eval", "activate"):
        q = sub.add_parser(name); q.add_argument("run"); q.add_argument("--after", type=int)
        if name == "eval":
            q.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11))); q.add_argument("--gpu", default="0")
    report = sub.add_parser("report"); report.add_argument("runs", nargs="+"); report.add_argument("--target", type=float)
    report.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "plan" and args.round == 0 and not args.pool:
        p.error("plan --round 0 needs --pool")
    return {"plan": cmd_plan, "probe": cmd_probe, "select": cmd_select, "repaired": cmd_repaired,
            "eval-plan": cmd_eval_plan, "eval": cmd_eval, "activate": cmd_activate, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
