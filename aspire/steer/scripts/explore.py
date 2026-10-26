#!/usr/bin/env python3
"""Run the research agent (prompts/explorer.md) unattended for some hours, and verify the programs it submits.

  .venv/bin/python scripts/explore.py --hours 8

The agent works in a tmux session (steer-explorer). When its session ends before the deadline it is restarted with a note to
continue from library/notes/JOURNAL.md. Every minute the driver looks for library/programs/candidates/VERIFY_<suite>__<task>__*
files and verifies each on held-out episodes (scripts/verify.py), which appends to library/INDEX.md. Log: outputs/explore.log.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steer.versions import snapshot  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
CAND = HERE / "library" / "programs" / "candidates"
SESSION = "steer-explorer"
LOG = HERE / "outputs" / "explore.log"
ENDED = HERE / "outputs" / "explorer_ended"         # touched by the tmux command when claude exits


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def start(model: str, resume: bool) -> None:
    prompt = HERE / "outputs" / "explorer_prompt.md"
    text = (HERE / "prompts" / "explorer.md").read_text()
    if resume:
        text += ("\nRESUME: an earlier session of yours ended. Read library/notes/JOURNAL.md, library/INDEX.md and "
                 "library/steerlib/README.md first and continue the most promising line of work.\n")
    prompt.write_text(text)
    subprocess.run(["tmux", "kill-session", "-t", SESSION], capture_output=True)
    ENDED.unlink(missing_ok=True)
    cmd = (f"unset ANTHROPIC_API_KEY; export MUJOCO_GL=egl PATH=$HOME/.local/bin:$PATH; "
           f"claude --model {model} \"$(cat {prompt})\"; touch {ENDED}; bash")
    subprocess.run(["tmux", "new-session", "-d", "-s", SESSION, "-c", str(HERE), cmd], check=True)
    log(f"explorer session started ({'resume' if resume else 'fresh'})")


def alive() -> bool:
    """True while claude runs inside the session (the session itself stays open after it exits)."""
    exists = subprocess.run(["tmux", "has-session", "-t", SESSION], capture_output=True).returncode == 0
    return exists and not ENDED.exists()


def verify_requests() -> None:
    for req in sorted(CAND.glob("VERIFY_*")):
        program = req.read_text().strip()
        try:
            suite, task, _ = req.name[len("VERIFY_"):].split("__", 2)
        except ValueError:
            log(f"bad request name {req.name}")
            req.rename(req.with_name("BAD_" + req.name))
            continue
        version = snapshot(f"verify {Path(program).stem} on {suite} {task}")
        log(f"verifying {program} on {suite} {task} (library {version})")
        r = subprocess.run([str(HERE / ".venv" / "bin" / "python"), "scripts/verify.py", "--program", program,
                            "--suite", suite, "--task", task, "--library-version", version], cwd=HERE, capture_output=True, text=True)
        log((r.stdout.strip().splitlines() or ["(no output)"])[-1] if r.returncode == 0 else f"verify failed: {r.stderr[-300:]}")
        req.rename(req.with_name("DONE_" + req.name))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--max-restarts", type=int, default=5)
    a = p.parse_args()
    os.environ.pop("ANTHROPIC_API_KEY", None)
    CAND.mkdir(parents=True, exist_ok=True)
    (HERE / "library" / "notes").mkdir(parents=True, exist_ok=True)
    deadline, restarts = time.time() + a.hours * 3600, 0
    if alive():
        log("explorer session already running; attaching")
    else:
        start(a.model, resume=(HERE / "library" / "notes" / "JOURNAL.md").exists())
    last_snapshot = 0.0
    while time.time() < deadline:
        verify_requests()
        if time.time() - last_snapshot > 900:                # a version every 15 min when the library changed
            v = snapshot("periodic")
            last_snapshot = time.time()
            log(f"library {v}")
        if not alive():
            if restarts >= a.max_restarts:
                log("explorer ended too often; stopping")
                break
            restarts += 1
            start(a.model, resume=True)
        time.sleep(60)
    subprocess.run(["tmux", "send-keys", "-t", SESSION, "Escape"], capture_output=True)
    log("deadline reached; explorer stopped (session kept for inspection)")
    verify_requests()
    log(f"final library {snapshot('end of exploration')}")


if __name__ == "__main__":
    main()
