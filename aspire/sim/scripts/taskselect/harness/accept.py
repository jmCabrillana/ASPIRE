#!/usr/bin/env python3
"""Acceptance test of a repair: the kept program against the program it started from, on seeds no repair or probe used.

  .venv/bin/python3 scripts/taskselect/harness/accept.py TASK_DIR START_CODE [--seeds 81..95] [--gpu 0] [--margin 3]

The kept program is the iteration with the most development passes (ties: the later one), as `library_loop.py repaired`
chooses it. Both programs run on the acceptance seeds in one evaluation; the repair is accepted when the kept program has at
least `margin` more passes than the start. Writes TASK_DIR/accept.json and prints one line. Needs the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SIM = Path(__file__).resolve().parents[3]
SEEDS = list(range(81, 96))


def kept_iteration(task_dir: Path) -> tuple[Path, int]:
    """(code path, development passes) of the iteration `repaired` would keep."""
    best = None
    for f in sorted(task_dir.glob("iter_*/iter_summary.json")):
        c = json.loads(f.read_text())["candidates"][0]
        if best is None or c["pass_count"] >= best[1]:
            best = (Path(c["code_path"]), c["pass_count"])
    if best is None:
        sys.exit(f"no iter_*/iter_summary.json under {task_dir}")
    return best


def decide(start: int, kept: int, margin: int) -> bool:
    return kept - start >= margin


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_dir", type=Path)
    p.add_argument("start_code", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--gpu", default="0")
    p.add_argument("--margin", type=int, default=3)
    args = p.parse_args()
    td = args.task_dir
    kept_code, dev = kept_iteration(td)
    acc = td / "accept"
    summary = acc / "iter_summary.json"
    if not summary.exists():
        for name, code in (("candidate_A", args.start_code), ("candidate_B", kept_code)):
            (acc / name).mkdir(parents=True, exist_ok=True)
            shutil.copy(code, acc / name / "code.py")
        first = json.loads(sorted(td.glob("iter_*/iter_summary.json"))[0].read_text())
        env = dict(os.environ, MUJOCO_GL="egl", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
        cmd = [str(SIM / ".venv-libero/bin/python3"), "scripts/libero/evosearch_eval.py", "--iter-dir", str(acc), "--suite", first["suite"],
               "--task", first["task"], "--trials", str(len(args.seeds)), "--trial-seeds", *map(str, args.seeds), "--sim-gpus", args.gpu,
               "--parallel-per-gpu", os.environ.get("EVAL_PARALLEL", "2"), "--no-highlights"]
        with (acc / "eval.log").open("w") as log:
            subprocess.run(cmd, cwd=SIM, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    passes = {c["candidate"]: c["pass_count"] for c in json.loads(summary.read_text())["candidates"]}
    start, kept = passes["candidate_A"], passes["candidate_B"]
    result = {"seeds": args.seeds, "start_passes": start, "kept_passes": kept, "kept_dev_passes": dev, "margin": args.margin,
              "accepted": decide(start, kept, args.margin)}
    (td / "accept.json").write_text(json.dumps(result, indent=1) + "\n")
    print(f"accept: start {start}/{len(args.seeds)}, kept {kept}/{len(args.seeds)} on seeds {args.seeds[0]}-{args.seeds[-1]} -> "
          + ("ACCEPTED" if result["accepted"] else f"REJECTED (needs +{args.margin})"))


if __name__ == "__main__":
    main()
