"""Versions of the library: content-addressed snapshots under outputs/library_versions/ with a manifest.

    snapshot("verify canon_rewrite")  -> "v007" (a new version only when the content changed)
    current()                          -> {"version": "v007", "hash": "...", ...}
    restore("v003", dest)              -> copy of that library (e.g. to evaluate an old library on new tasks)
Every evaluation records the library hash it ran with (scripts/run_program.py), so results stay tied to a version.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "library"
STORE = ROOT / "outputs" / "library_versions"
MANIFEST = STORE / "versions.json"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")


def files() -> list[Path]:
    return sorted(p for p in LIBRARY.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")


def library_hash() -> str:
    h = hashlib.sha256()
    for p in files():
        h.update(str(p.relative_to(LIBRARY)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []


def current() -> dict | None:
    m = manifest()
    return m[-1] if m else None


def snapshot(trigger: str) -> str:
    """Copy the library if it changed since the last version; return the version id."""
    h, m = library_hash(), manifest()
    if m and m[-1]["hash"] == h:
        return m[-1]["version"]
    version = f"v{len(m):03d}"
    STORE.mkdir(parents=True, exist_ok=True)
    shutil.copytree(LIBRARY, STORE / version, ignore=IGNORE)
    previous = {e for e in (m[-1]["files"] if m else [])}
    now = [str(p.relative_to(LIBRARY)) for p in files()]
    m.append({"version": version, "hash": h, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "trigger": trigger,
              "files": now, "added": sorted(set(now) - previous), "removed": sorted(previous - set(now))})
    MANIFEST.write_text(json.dumps(m, indent=1) + "\n")
    return version


def version_of(h: str) -> str | None:
    return next((e["version"] for e in manifest() if e["hash"] == h), None)


def restore(version: str, dest: Path) -> Path:
    shutil.copytree(STORE / version, dest, dirs_exist_ok=False)
    return dest
