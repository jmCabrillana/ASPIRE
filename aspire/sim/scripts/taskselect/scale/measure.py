#!/usr/bin/env python3
"""Measure one fixed skill library on the held-out tasks of a build: K fresh programs per task, run on S seeds.

  .venv/bin/python3 scripts/taskselect/scale/measure.py --tasks-run outputs/scale_unsolved_r1 \
      --library outputs/scale_unsolved_r1/library/L05 --out outputs/scale_unsolved_r1/measure/after_05 --k 12 --seeds 1 2 3 4 5

The tasks are state.json "eval_pool" of --tasks-run. Writer sessions read the library from its own directory (the live skills
directory is never touched). Output: OUT/measure.json with every program's per-seed 0/1 outcome and failure stage per task.
Rerunning skips finished steps; a restarted writer session only fills missing programs. Needs the GPU for the evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402
import library_loop as ll  # noqa: E402
import run_reps  # noqa: E402


def program_files(out: Path, tasks: list[str]) -> list[Path]:
    return [out / "programs" / ll.task_dir_name(t) for t in tasks]


def missing(out: Path, tasks: list[str], k: int) -> int:
    return sum(not (d / f"candidate_{n:02d}" / "code.py").exists() or (d / f"candidate_{n:02d}" / "code.py").stat().st_size == 0
               for d in program_files(out, tasks) for n in range(1, k + 1))


def write_programs(out: Path, lib: Path, tasks: list[str], k: int, tag: str) -> None:
    if missing(out, tasks, k) == 0:
        return
    for d in program_files(out, tasks):
        d.mkdir(parents=True, exist_ok=True)
    prompt = out / "writer.md"
    prompt.write_text((run_reps.SIM / "scripts/taskselect/prompts/measure_writer.md").read_text()
                      .replace("{{LIB}}", str(lib)).replace("{{DIR}}", str(out)).replace("{{K}}", str(k))
                      .replace("{{GROUPS}}", str(k // 3)))
    (out / "WRITER_DONE").unlink(missing_ok=True)
    session = f"meas-{tag}"
    run_reps.log(f"measure {tag}: writer session ({missing(out, tasks, k)} programs missing)")
    run_reps.start_claude(session, prompt)
    run_reps.wait_for(out / "WRITER_DONE", session, 12, f"measure writer {tag}")
    if missing(out, tasks, k):
        ll.fail(f"{tag}: {missing(out, tasks, k)} programs are still missing")


def evaluate(out: Path, tasks: list[str], k: int, seeds: list[int], tag: str) -> None:
    result = {"seeds": seeds, "k": k, "tasks": {}}
    for task in tasks:
        tdir = out / "programs" / ll.task_dir_name(task)
        names = sorted(d.name for d in tdir.glob("candidate_*"))
        ll.run_eval(tdir, *task.split("/", 1), seeds, "0")
        outcomes = core.load_round(tdir)["outcomes"]
        sigs = core.load_signatures(tdir, names, seeds)
        result["tasks"][task] = {"outcomes": outcomes, "stages": {c: [s.split("|")[0] for s in sig] for c, sig in sigs.items()}}
    (out / "measure.json").write_text(json.dumps(result, indent=2) + "\n")
    run_reps.log(f"measure {tag}: done")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks-run", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True, help="skill library directory to write from (may be an empty directory)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    a = p.parse_args()
    if a.k % 3:
        p.error("--k must be a multiple of 3")
    out, lib = a.out.resolve(), a.library.resolve()
    lib.mkdir(parents=True, exist_ok=True)
    if (out / "measure.json").exists():
        return
    tasks = ll.load_state(a.tasks_run.resolve())["eval_pool"]
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{out.parent.parent.name if out.parent.name == 'measure' else out.parent.name}-{out.name}"
    write_programs(out, lib, tasks, a.k, tag)
    run_reps.log(f"measure {tag}: evaluating {len(tasks)} tasks x {a.k} programs x {len(a.seeds)} seeds")
    evaluate(out, tasks, a.k, a.seeds, tag)


if __name__ == "__main__":
    main()
