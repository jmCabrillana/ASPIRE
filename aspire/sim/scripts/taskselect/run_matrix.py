#!/usr/bin/env python3
"""Evaluate one set of candidates on every task of a family: the candidates x tasks x seeds outcome matrix.

  .venv-libero/bin/python3 scripts/taskselect/run_matrix.py --family goal_swap \
      --candidates outputs/family_matrix/goal_swap/iter_00 --seeds 51 52 53 54 55 [--gpu 0] [--parallel 2]

Runs scripts/libero/evosearch_eval.py once per task (sequential, resumable: finished tasks are skipped) under
<candidates>/../matrix/task_KK/, then writes matrix/matrix.json (the format backtest.py reads) and prints the table.
"""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", required=True)
    p.add_argument("--candidates", type=Path, required=True, help="directory holding candidate_*/code.py")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--parallel", default="2")
    args = p.parse_args(argv)
    family = core.get_family(core.load_families(), args.family)
    src = args.candidates.resolve()
    names = sorted(d.name for d in src.glob("candidate_*") if (d / "code.py").exists())
    if not names:
        fail(f"no candidate_*/code.py in {src}")
    seeds = sorted(args.seeds)
    root = src.parent / "matrix"
    env = dict(os.environ, MUJOCO_GL="egl", TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1")
    env.pop("ANTHROPIC_API_KEY", None)

    outcomes = {}
    for k, t in enumerate(family["tasks"]):
        tdir = root / f"task_{k:02d}"
        if not (tdir / "iter_summary.json").exists():
            for name in names:
                (tdir / name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / name / "code.py", tdir / name / "code.py")
            cmd = [str(REPO / ".venv-libero/bin/python3"), "scripts/libero/evosearch_eval.py", "--iter-dir", str(tdir),
                   "--suite", t["suite"], "--task", t["task"], "--trials", str(len(seeds)),
                   "--trial-seeds", *map(str, seeds), "--sim-gpus", args.gpu, "--parallel-per-gpu", args.parallel,
                   "--no-highlights"]
            print(f"RUN  {t['id']}", flush=True)
            with (tdir / "eval.log").open("w") as log:
                code = subprocess.run(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code != 0 or not (tdir / "iter_summary.json").exists():
                fail(f"eval failed for {t['id']} (exit {code}); see {tdir / 'eval.log'}")
        r = core.load_round(tdir)
        if r["task_id"] != t["id"] or r["seeds"] != seeds or sorted(r["outcomes"]) != names:
            fail(f"{tdir} holds results for another task, seed set or candidate set; remove it to re-run")
        outcomes[t["id"]] = r["outcomes"]

    (root / "matrix.json").write_text(json.dumps({
        "family": args.family, "round0": family["round0"], "seeds": seeds, "candidates": names,
        "tasks": [{k: t[k] for k in ("id", "suite", "task", "language_static")} for t in family["tasks"]],
        "outcomes": outcomes}, indent=2) + "\n")

    short = [n.removeprefix("candidate_") for n in names]
    print(f"\npasses out of {len(seeds)} seeds   " + "  ".join(short) + "   D_seed D_prog mixed")
    for t in family["tasks"]:
        o, m = outcomes[t["id"]], core.committee_metrics(outcomes[t["id"]])
        print(f"{t['task'][:30]:30s}  " + "  ".join(str(sum(o[n])) for n in names)
              + f"   {m['D_seed']:.3f}  {m['D_prog']:.3f}  {m['mixed_seed_frac']:.2f}")
    print(f"wrote {root / 'matrix.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
