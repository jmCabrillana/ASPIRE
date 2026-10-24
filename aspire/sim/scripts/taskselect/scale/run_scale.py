#!/usr/bin/env python3
"""One build of the scale-up design (POWER.md): LIBERO-90, 70 build tasks and 20 held out, R repair rounds, unattended.

  .venv/bin/python3 scripts/taskselect/scale/run_scale.py --arm unsolved --rep 1 --rounds 10

Build: a coordinator Claude session (tmux) runs the rounds of prompts/library_round_scale.md on outputs/scale_<arm>_r<rep>:
each round probes CANDIDATES random unsolved tasks (3 programs x 5 seeds 66-70), picks one at random, repairs it (fixloop), promotes.
Measurements (held-out tasks, K programs, seeds 1-5; see measure.py) run beside it from the saved libraries: the empty library once
(outputs/scale_L0, shared by every build), then after rounds R/2 and R. Logs: outputs/campaign.log; failure: outputs/campaign.FAILED.
Rerunning resumes from what is on disk.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import library_loop as ll  # noqa: E402
import run_reps  # noqa: E402

OUT, SIM, ENV = run_reps.OUT, run_reps.SIM, run_reps.ENV
MEASURE = [str(SIM / ".venv/bin/python3"), "scripts/taskselect/scale/measure.py"]


def measure(run: Path, library: Path, out: Path, k: int) -> None:
    r = subprocess.run(MEASURE + ["--tasks-run", str(run), "--library", str(library), "--out", str(out), "--k", str(k)],
                       cwd=SIM, env=ENV, capture_output=True, text=True)
    if r.returncode:
        run_reps.fail(f"measure {out.name}: {r.stdout[-300:]} {r.stderr[-300:]}")


def measure_worker(run: Path, args) -> None:
    """The empty library first, then each checkpoint as soon as its library is saved."""
    measure(run, OUT / "scale_L0_library", OUT / "scale_L0", args.k_l0)
    for n in sorted({args.rounds // 2, args.rounds}):
        while len(ll.load_state(run)["rounds"]) < n:
            time.sleep(60)
        measure(run, ll.library_dir(run, n), run / "measure" / f"after_{n:02d}", args.k_measure)


def build(run: Path, args) -> None:
    """Run the coordinator session until the last library is saved."""
    done = len(ll.load_state(run)["rounds"])
    if done >= args.rounds:
        return
    if done:
        run_reps.loop("activate", str(run), "--after", str(done))
    text = (SIM / "scripts/taskselect/prompts/library_round_scale.md").read_text()
    text = text.replace("{{ARM}}", args.arm).replace("{{REP}}", str(args.rep)).replace("{{ROUNDS}}", str(args.rounds)) \
               .replace("{{CANDIDATES}}", str(args.candidates))
    if done:
        text += (f"\nRESUME NOTICE: rounds 0 to {done - 1} are done (library/L{done:02d} saved and live). Continue at ROUND {done}; "
                 f"if its programs or probe already exist, do not redo them.\n")
    prompt = run / "coordinator.md"
    prompt.write_text(text)
    session = f"scale-{args.arm}-r{args.rep}"
    run_reps.log(f"scale {run.name}: start build coordinator at round {done}")
    run_reps.start_claude(session, prompt)
    deadline = time.time() + 4 * args.rounds * 3600
    while len(ll.load_state(run)["rounds"]) < args.rounds:
        if time.time() > deadline:
            run_reps.fail(f"scale {run.name}: {args.rounds} rounds not done after {4 * args.rounds} h")
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
            run_reps.fail(f"scale {run.name}: coordinator session ended after round {len(ll.load_state(run)['rounds'])}")
        time.sleep(60)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="unsolved", choices=("unsolved",))
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--candidates", type=int, default=10)
    p.add_argument("--k-measure", type=int, default=12)
    p.add_argument("--k-l0", type=int, default=30)
    args = p.parse_args()
    (OUT / "campaign.FAILED").unlink(missing_ok=True)
    run = OUT / f"scale_{args.arm}_r{args.rep}"
    if not (run / "state.json").exists():
        run_reps.log(f"scale {run.name}: plan round 0")
        run_reps.loop("plan", str(run), "--round", "0", "--pool", "libero_90", "--library", "empty", "--eval-split", "20",
                      "--candidates", str(args.candidates), "--salt", f"r{args.rep}")
    worker = threading.Thread(target=measure_worker, args=(run, args), daemon=True)
    worker.start()
    build(run, args)
    run_reps.log(f"scale {run.name}: build done; waiting for the measurements")
    worker.join()
    if (OUT / "campaign.FAILED").exists():
        sys.exit(1)
    (OUT / "campaign.DONE").write_text("scale\n")
    run_reps.log(f"scale {run.name}: all done")


if __name__ == "__main__":
    main()
