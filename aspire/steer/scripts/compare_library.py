#!/usr/bin/env python3
"""Library-transfer comparison: the same research agent, the same budget and the same unseen target tasks, once with the seed
library (before any agent work) and once with a grown library version. Arms run one after the other in isolated copies of the
project (outputs/compare/<name>/<arm>/: own library, outputs and verification index; the .venv is shared).

  .venv/bin/python scripts/compare_library.py --name object_lan --version v003 --hours 2 \
      --targets libero_object_lan:3 libero_object_lan:7

Result: each arm's library/INDEX.md (held-out verification), JOURNAL.md, and outputs/compare/<name>/summary.md.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import io
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SEED_COMMIT = "faf67ec"        # library as it was when the first research agent started (seed steerlib, example, literature)


def make_arm(dest: Path, library_source: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("steer", "scripts", "prompts", "README.md", "constraints.txt"):
        src = HERE / name
        (shutil.copytree(src, dest / name, ignore=shutil.ignore_patterns("__pycache__")) if src.is_dir()
         else shutil.copy(src, dest / name))
    (dest / ".venv").symlink_to(HERE / ".venv")
    (dest / "outputs").mkdir(exist_ok=True)
    for f in (HERE / "outputs" / "baseline").glob("*.json"):           # frozen baselines are shared knowledge
        (dest / "outputs" / "baseline").mkdir(exist_ok=True)
        shutil.copy(f, dest / "outputs" / "baseline" / f.name)
    if library_source == "seed":
        repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        rel = HERE.relative_to(repo)
        data = subprocess.run(["git", "archive", SEED_COMMIT, f"{rel}/library"], cwd=repo, capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            tar.extractall(dest / "_seed")
        shutil.move(str(dest / "_seed" / rel / "library"), dest / "library")
        shutil.rmtree(dest / "_seed")
    else:
        shutil.copytree(HERE / "outputs" / "library_versions" / library_source, dest / "library")
    for f in ("INDEX.md",):
        (dest / "library" / f).unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--version", required=True, help="grown library version (outputs/library_versions/<v>)")
    p.add_argument("--hours", type=float, default=2)
    p.add_argument("--targets", nargs="+", required=True, help="suite:task")
    a = p.parse_args()
    root = HERE / "outputs" / "compare" / a.name
    targets = ", ".join(t.replace(":", " task ") for t in a.targets)
    extra = (f"COMPARISON RUN. Your only targets are: {targets}. Ignore the TARGETS and RESERVED paragraphs above. Your library is "
             f"library/ in this working directory (read it first); you have {a.hours} hours. Make these targets succeed as often as "
             f"possible on held-out verification.\n")
    for arm, source in (("seed", "seed"), ("library", a.version)):
        dest = root / arm
        if not dest.exists():
            make_arm(dest, source)
        (dest / "prompts" / "explorer_extra.md").write_text(extra)
        print(f"arm {arm} ({source}) in {dest}", flush=True)
        subprocess.run([str(dest / ".venv" / "bin" / "python"), "scripts/explore.py", "--hours", str(a.hours),
                        "--session", f"compare-{a.name}-{arm}", "--max-restarts", "3"], cwd=dest, check=False)
    lines = [f"# Library comparison {a.name}", "", f"targets: {targets}; budget {a.hours} h per arm; grown library {a.version}", ""]
    for arm in ("seed", "library"):
        idx = root / arm / "library" / "INDEX.md"
        lines += [f"## {arm}", "", idx.read_text() if idx.exists() else "(no verified program)", ""]
    (root / "summary.md").write_text("\n".join(lines))
    print((root / "summary.md").read_text())


if __name__ == "__main__":
    main()
