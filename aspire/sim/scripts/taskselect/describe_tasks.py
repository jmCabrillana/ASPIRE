#!/usr/bin/env python3
"""Perception-only descriptors for the tasks of a family: runtime instruction and reset-frame images.

Replays scripts/libero/scene_snapshot.py once per task (first dev seed, no policy code) and stores
descriptors/<suite>__<task>/{task_language.txt, scene.jpg, wrist.jpg}. Needs .venv-libero and the simulator.

  .venv-libero/bin/python3 scripts/taskselect/describe_tasks.py --family goal_swap [--gpu 0] [--only SUITE/TASK ...]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

REPO = core.HERE.parents[1]


def describe_one(task: dict, gpu: str, seed: int, root: Path, timeout: int) -> str | None:
    """Returns an error string, or None on success."""
    out = core.descriptor_dir(task, root)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="describe_") as tmp:
        snap = Path(tmp) / "snap"
        env = dict(os.environ, SNAPSHOT_DIR=str(snap), MUJOCO_GL="egl", CUDA_VISIBLE_DEVICES=gpu,
                   TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1", ASPIRE_SAM3_PROMPTS="")
        env.pop("ANTHROPIC_API_KEY", None)
        cmd = [str(REPO / ".venv-libero/bin/python3"), "scripts/libero/replay_trial.py",
               "--args.suite", task["suite"], "--args.task", task["task"], "--args.trial", str(seed),
               "--args.replay-code", str(REPO / "scripts/libero/scene_snapshot.py"),
               "--args.config", "env_configs/libero/franka_libero_traced.yaml",
               "--args.output-dir", str(Path(tmp) / "replay")]
        try:
            proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"timeout after {timeout}s"
        m = re.search(r"^TASK_LANGUAGE: (.+)$", proc.stdout, flags=re.M)
        if proc.returncode != 0 or not m or not (snap / "scene_snapshot.jpg").exists():
            return f"exit {proc.returncode}; no snapshot/instruction. tail: {proc.stdout[-400:]!r} {proc.stderr[-400:]!r}"
        (out / "task_language.txt").write_text(m.group(1).strip() + "\n")
        shutil.copy2(snap / "scene_snapshot.jpg", out / "scene.jpg")
        wrist = snap / "scene_snapshot_wrist.jpg"
        if wrist.exists():
            shutil.copy2(wrist, out / "wrist.jpg")
    return None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--only", nargs="*", help="restrict to these task ids")
    p.add_argument("--descriptors-dir", type=Path, default=core.DESCRIPTORS_DIR)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--force", action="store_true", help="redo tasks that already have a descriptor")
    args = p.parse_args(argv)
    doc = core.load_families()
    family = core.get_family(doc, args.family)
    seed = doc["dev_seeds"][0]
    failures = 0
    for t in family["tasks"]:
        if args.only and t["id"] not in args.only:
            continue
        d = core.descriptor_dir(t, args.descriptors_dir)
        if not args.force and (d / "task_language.txt").exists() and (d / "scene.jpg").exists():
            print(f"skip   {t['id']} (already described)")
            continue
        err = describe_one(t, args.gpu, seed, args.descriptors_dir, args.timeout)
        if err:
            failures += 1
            print(f"FAILED {t['id']}: {err}")
        else:
            print(f"ok     {t['id']}: {(d / 'task_language.txt').read_text().strip()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
