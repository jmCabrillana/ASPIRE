#!/usr/bin/env python3
"""Repetitions of the lex-vs-unsolved library build, unattended.

  .venv/bin/python3 scripts/taskselect/run_reps.py --reps 2 3 4

Per repetition k (same settings as library_lex_r1 / library_unsolved_r1, plus the API facts in the writers' prompts):
  1. outputs/library_unsolved_r<k>: plan round 0 from the empty library, a writer session writes 27 programs, probe.
  2. outputs/library_lex_r<k>: round 0 copied from it (probe.json identical), so the two arms are paired.
  3. each arm in its own Claude coordinator session (tmux), 3 repairs + a final probe; arm order alternates with k.
Everything is logged to outputs/campaign.log. A failure writes outputs/campaign.FAILED and stops. Rerunning skips finished steps.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SIM = Path(__file__).resolve().parents[2]
OUT = SIM / "outputs"
LOOP = [str(SIM / ".venv/bin/python3"), "scripts/taskselect/library_loop.py"]
LIBERO_PY = str(SIM / ".venv-libero/bin/python3")
POOL = "library_9_swap"
SEEDS, ROUNDS, TAG, TREATMENT = list(range(51, 66)), 3, "", "lex"      # set from the command line
ACCEPT_SEEDS = " ".join(map(str, range(81, 96)))
ENV = dict(os.environ, ASPIRE_ROOT=str(SIM), PYTHON_ROOT=str(SIM.parents[1]), MUJOCO_GL="egl",
           TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", EVAL_PARALLEL="4")
ENV.pop("ANTHROPIC_API_KEY", None)


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with (OUT / "campaign.log").open("a") as f:
        f.write(line + "\n")


def fail(msg: str) -> None:
    log("FAILED: " + msg)
    (OUT / "campaign.FAILED").write_text(msg + "\n")
    sys.exit(1)


def loop(*args: str) -> str:
    r = subprocess.run(LOOP + list(args), cwd=SIM, env=ENV, capture_output=True, text=True)
    if r.returncode:
        fail(f"library_loop {' '.join(args)}: {r.stdout[-400:]} {r.stderr[-400:]}")
    return r.stdout


def start_claude(session: str, prompt_file: Path) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    env = " ".join(f"{k}={v}" for k, v in ENV.items() if k in ("ASPIRE_ROOT", "PYTHON_ROOT", "MUJOCO_GL",
                                                               "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "EVAL_PARALLEL"))
    cmd = (f"unset ANTHROPIC_API_KEY; export {env} PATH=$HOME/.local/bin:$PATH; "
           f"claude --model claude-sonnet-5 \"$(cat {prompt_file})\"; bash")
    subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(SIM), cmd], check=True)


def wait_for(path: Path, session: str, hours: float, what: str) -> None:
    deadline = time.time() + hours * 3600
    while not path.exists():
        if time.time() > deadline:
            fail(f"{what}: {path.name} not there after {hours} h")
        if subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode:
            if path.exists():
                break
            fail(f"{what}: session {session} ended before {path.name}")
        time.sleep(30)
    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


HARNESS_STEP = """
HARNESS ARM (the repair is checked before its lesson enters the library). Changes to the steps above:
  4. The repair prompt is round_$ROUND/repair_prompt.md verbatim, followed (in the same prompt) by the full text of
     scripts/taskselect/prompts/repair_harness_addendum.md. Tell the subagent nothing about the acceptance test below.
  4a. When the subagent has returned and findings.md exists, run the acceptance test in the background and wait for it
      (TD = repair.task_dir, START = repair.start_code from selection.json):
        nohup .venv/bin/python3 scripts/taskselect/harness/accept.py $TD $START --seeds {accept} --gpu 0 > $TD/accept.log 2>&1 &
        until [ -f $TD/accept.json ] || grep -q Traceback $TD/accept.log; do sleep 10; done
      (Bash timeout 600000; repeat until it returns). If $TD/accept.json is still missing, read $TD/accept.log, re-run the same
      command once, and if it fails again stop and report.
  4b. Read $TD/accept.json. If "accepted" is true, promote as usual. If it is false, do NOT edit the skills: run `begin`, then
      `finish ... --reason "no-op: rejected by acceptance (start X, kept Y of 15 on seeds 81-95)"` with the real numbers.
      Either way continue with step 5.
"""

COORDINATOR_TAIL = """
THIS RUN: ROUNDS = {n} (ROUND 0 .. {last} = {n} repairs). After ROUND {last}'s `repaired`, do a final measurement only (no
select, no repair): `$L activate $RUN --after {n}`, `$L plan $RUN --round {n}`, write the K programs for the tasks it lists,
then `$L probe $RUN --round {n} --gpu 0 --seeds {seeds}`. Then print `$L report $RUN` and stop.

WAITING RULES for every subagent (add these lines after the verbatim repair prompt in your dispatch message): never end
your turn while an evaluation is running. Wait inside one tool call:
  until [ -f <iter_dir>/iter_summary.json ]; do sleep 10; done      (Bash timeout 600000; repeat the call until it returns)
A crashed evaluation may be re-run once as the repair prompt says. If a subagent returns without a kept program or without
findings.md, do NOT dispatch a second one: record it with `repaired` if the iterations allow, otherwise stop and report.

Never push. Do not change .claude/ other than skills in step 4, and never edit scripts/taskselect/.
"""


def run_dir(rule: str, rep: int) -> Path:
    return OUT / f"library_{rule}{TAG}_r{rep}"


def coordinator_prompt(rule: str, rep: int) -> str:
    harness = rule == "harness"
    seeds = " ".join(map(str, SEEDS))
    tpl = (SIM / "scripts/taskselect/prompts/library_round.md").read_text()
    a, b = tpl.index("Round 0 creates the run"), tpl.index("Then for ROUND")
    start = ("Round 0 is ALREADY done for this run: the run exists, its 27 programs are written and probed, and probe.json is\n"
             "identical to the other arm's of this repetition (round 0 is shared). Do NOT run plan --round 0 and do NOT write\n"
             f"round-0 programs. Pool: {POOL}. The live skill library was already set to L00 (empty). Start at step 3 (select)\n"
             "of ROUND 0.\n")
    tail = (HARNESS_STEP.replace("{accept}", ACCEPT_SEEDS) if harness else "") + COORDINATOR_TAIL
    tail = tail.replace("{n}", str(ROUNDS)).replace("{last}", str(ROUNDS - 1)).replace("{seeds}", seeds)
    text = tpl[:a] + start + tpl[b:] + tail
    head_end = text.index("Every subagent inherits")
    text = text[:text.index("Start:")] + text[head_end:]   # drop the launcher example line from the template header
    text = text.replace("library_{{RULE}}_r{{REP}}", run_dir(rule, rep).name)
    text = text.replace("--rule {{RULE}}", "--rule unsolved" if harness else "--rule {{RULE}}")
    text = text.replace("--round $ROUND --gpu 0     (every program on seeds 51-65)",
                        f"--round $ROUND --gpu 0 --seeds {seeds}     (every program on these seeds)")
    return text.replace("{{RULE}}", rule).replace("{{REP}}", str(rep)).replace("{{ROUNDS}}", str(ROUNDS))


def make_round0(rep: int) -> None:
    base, lex = run_dir("unsolved", rep), run_dir(TREATMENT, rep)
    if (lex / "round_00" / "probe.json").exists():
        return
    if not (base / "state.json").exists():
        log(f"rep {rep}: plan round 0 (empty library)")
        loop("plan", str(base), "--round", "0", "--pool", POOL, "--library", "empty")
    if not (base / "round_00" / "WRITER_DONE").exists():
        prompt = base / "round0_writer.md"
        prompt.write_text((SIM / "scripts/taskselect/prompts/round0_writer.md").read_text().replace("{{RUN}}", f"outputs/{base.name}"))
        log(f"rep {rep}: round-0 writer session")
        start_claude(f"rep{rep}-writer", prompt)
        wait_for(base / "round_00" / "WRITER_DONE", f"rep{rep}-writer", 1.5, f"rep {rep} writer")
    if not (base / "round_00" / "probe.json").exists():
        log(f"rep {rep}: round-0 probe")
        loop("probe", str(base), "--round", "0", "--gpu", "0", "--seeds", *map(str, SEEDS))
    log(f"rep {rep}: copy round 0 to the {TREATMENT} arm")
    loop("plan", str(lex), "--round", "0", "--pool", POOL, "--library", "empty")
    shutil.copytree(base / "round_00" / "programs", lex / "round_00" / "programs", dirs_exist_ok=True)
    loop("probe", str(lex), "--round", "0", "--gpu", "0", "--seeds", *map(str, SEEDS))
    if (base / "round_00" / "probe.json").read_bytes() != (lex / "round_00" / "probe.json").read_bytes():
        fail(f"rep {rep}: the two arms' round-0 probe.json differ")


def run_arm(rule: str, rep: int) -> None:
    run = run_dir(rule, rep)
    final = run / f"round_{ROUNDS:02d}" / "probe.json"
    if final.exists():
        return
    loop("activate", str(run), "--after", "0")
    prompt = run / "coordinator.md"
    prompt.write_text(coordinator_prompt(rule, rep))
    session = f"arm-{rule}-r{rep}"
    log(f"rep {rep}: start arm {rule}")
    start_claude(session, prompt)
    wait_for(final, session, 3 * ROUNDS + 6, f"arm {rule} r{rep}")
    log(f"rep {rep}: arm {rule} done")


def main() -> None:
    global SEEDS, ROUNDS, TAG, TREATMENT
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="probe seeds of every round")
    p.add_argument("--rounds", type=int, default=ROUNDS, help="repairs per arm (one more probe follows)")
    p.add_argument("--treatment", default=TREATMENT, choices=("lex", "harness"), help="the arm run against `unsolved`")
    p.add_argument("--tag", default=TAG, help="added to the experiment ids: library_<rule><tag>_r<k>")
    args = p.parse_args()
    SEEDS, ROUNDS, TAG, TREATMENT = args.seeds, args.rounds, args.tag, args.treatment
    (OUT / "campaign.FAILED").unlink(missing_ok=True)
    for i, rep in enumerate(args.reps):
        make_round0(rep)
        order = (TREATMENT, "unsolved") if i % 2 == 0 else ("unsolved", TREATMENT)
        for rule in order:
            run_arm(rule, rep)
    log("campaign done")
    (OUT / "campaign.DONE").write_text(time.strftime("%c") + "\n")


if __name__ == "__main__":
    main()
