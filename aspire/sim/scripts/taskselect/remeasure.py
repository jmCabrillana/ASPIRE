#!/usr/bin/env python3
"""Re-measure the final library of finished runs with a second, independent set of programs (3 new per task, 6 in total).

  .venv/bin/python3 scripts/taskselect/remeasure.py library_lex_s30_r1 library_unsolved_s30_r1 ...

Per run: a writer session writes 3 new programs per task from the run's final library (read from the run's own library
directory, so the live skills directory is never touched), then they are probed on the run's probe seeds. Output:
outputs/remeasure/<run>/probe.json (same format as a round's probe.json). The original final probe is not rerun.
Rerunning skips finished steps. Needs the GPU for the probe (EVAL_PARALLEL workers, default 2).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402
import library_loop as ll  # noqa: E402
import run_reps  # noqa: E402

OUT = run_reps.OUT
os.environ.setdefault("EVAL_PARALLEL", "2")


def final_round(run: Path) -> int:
    return max(int(p.parent.name.split("_")[1]) for p in run.glob("round_*/probe.json"))


def write_programs(run: Path, out: Path, lib: Path) -> None:
    if (out / "WRITER_DONE").exists():
        return
    state = json.loads((run / "state.json").read_text())
    for task in state["pool"]:
        (out / "programs" / ll.task_dir_name(task)).mkdir(parents=True, exist_ok=True)
    prompt = out / "writer.md"
    prompt.write_text((run_reps.SIM / "scripts/taskselect/prompts/remeasure_writer.md").read_text()
                      .replace("{{LIB}}", str(lib)).replace("{{DIR}}", str(out)))
    session = f"rem-{run.name}"
    run_reps.log(f"remeasure {run.name}: writer session")
    run_reps.start_claude(session, prompt)
    run_reps.wait_for(out / "WRITER_DONE", session, 1.5, f"remeasure writer {run.name}")


def probe(run: Path, out: Path, seeds: list[int]) -> None:
    if (out / "probe.json").exists():
        return
    state = json.loads((run / "state.json").read_text())
    result = {"seeds": seeds, "tasks": {}}
    for task in state["pool"]:
        tdir = out / "programs" / ll.task_dir_name(task)
        names = ll.programs_in(tdir, state["k"], "the writer did not finish")
        ll.run_eval(tdir, *task.split("/", 1), seeds, "0")
        outcomes = core.load_round(tdir)["outcomes"]
        sigs = core.load_signatures(tdir, names, seeds)
        result["tasks"][task] = {"outcomes": outcomes, "stages": {c: [s.split("|")[0] for s in sig] for c, sig in sigs.items()}}
    (out / "probe.json").write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    for name in sys.argv[1:]:
        run = OUT / name
        n = final_round(run)
        out = OUT / "remeasure" / name
        out.mkdir(parents=True, exist_ok=True)
        seeds = json.loads((run / f"round_{n:02d}" / "probe.json").read_text())["seeds"]
        write_programs(run, out, ll.library_dir(run, n))
        run_reps.log(f"remeasure {name}: probe L{n:02d} programs on {len(seeds)} seeds")
        probe(run, out, seeds)
        run_reps.log(f"remeasure {name}: done")


if __name__ == "__main__":
    main()
