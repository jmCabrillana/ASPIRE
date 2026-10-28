"""Fixed development panel for the curve 'library version -> success' (operator paper target 4).

    .venv/bin/python library/tools/panel.py v033            # a snapshot in outputs/library_versions/
    .venv/bin/python library/tools/panel.py current         # the library as it is now
    .venv/bin/python library/tools/panel.py frozen          # frozen VLA, read from outputs/baseline (episodes 0-4)
    .venv/bin/python library/tools/panel.py table           # print/write library/notes/panel.md

Each version runs ITS OWN programs/candidates/auto.py (versions without auto.py = frozen VLA) in an isolated copy of the project
(outputs/panel/<version>/: copied steer/, scripts/, library snapshot; shared .venv), one simulator job at a time, development
episodes 0-4 of every panel task. The panel is fixed once and for all (do not edit PANEL; add a PANEL2 if needed).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
PANEL = [("libero_goal_lan", t) for t in (0, 3, 5, 6, 7, 8)] + [("libero_spatial_lan", t) for t in (0, 1, 4, 7)] + \
        [("libero_spatial_swap", t) for t in (0, 1, 2, 7)] + [("libero_goal_swap", t) for t in (1, 9)]
EPS = "0-4"
OUT = HERE / "outputs" / "panel"


def make_arm(version: str) -> Path:
    dest = OUT / version
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    for name in ("steer", "scripts"):
        shutil.copytree(HERE / name, dest / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("README.md", "constraints.txt"):
        if (HERE / name).exists():
            shutil.copy(HERE / name, dest / name)
    (dest / ".venv").symlink_to(HERE / ".venv")
    (dest / "outputs" / "baseline").mkdir(parents=True)
    for f in (HERE / "outputs" / "baseline").glob("*.json"):
        shutil.copy(f, dest / "outputs" / "baseline" / f.name)
    src = HERE / "library" if version == "current" else HERE / "outputs" / "library_versions" / version
    shutil.copytree(src, dest / "library", ignore=shutil.ignore_patterns("__pycache__", "VERIFY_*"))
    return dest


def frozen() -> dict:
    res = {}
    for s, t in PANEL:
        eps = json.loads((HERE / "outputs" / "baseline" / f"{s}__{t}.json").read_text())["episodes"]
        res[f"{s}:{t}"] = sum(e["success"] for e in eps if e["episode"] <= 4)
    return res


def run(version: str) -> dict:
    dest = make_arm(version)
    auto = dest / "library" / "programs" / "candidates" / "auto.py"
    path = dest / "panel.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    if not auto.exists():
        res = frozen()
    for s, t in PANEL:
        key = f"{s}:{t}"
        if key in res:
            continue
        out = subprocess.run([str(dest / ".venv" / "bin" / "python"), "scripts/run_program.py", "--program",
                              "library/programs/candidates/auto.py", "--suite", s, "--task", str(t), "--episodes", EPS],
                             cwd=dest, capture_output=True, text=True)
        summ = dest / "outputs" / "programs" / "auto" / f"{s}__{t}" / "summary.json"
        res[key] = sum(e["success"] for e in json.loads(summ.read_text())["episodes"]) if summ.exists() else None
        print(version, key, res[key], (out.stderr or "")[-300:] if res[key] is None else "", flush=True)
        path.write_text(json.dumps(res, indent=1))
    path.write_text(json.dumps(res, indent=1))
    return res


def table() -> str:
    versions = json.loads((HERE / "outputs" / "library_versions" / "versions.json").read_text())
    when = {v["version"]: v["time"] for v in versions}
    rows = [("frozen", "", frozen())]
    for d in sorted(OUT.glob("*/panel.json")):
        rows.append((d.parent.name, when.get(d.parent.name, ""), json.loads(d.read_text())))
    keys = [f"{s}:{t}" for s, t in PANEL]
    short = [k.replace("libero_", "").replace(":", " ") for k in keys]
    lines = ["# Development panel: library version -> success (auto.py, episodes 0-4 per task, /5)", "",
             "| version | time | " + " | ".join(short) + " | total /%d |" % (5 * len(keys)),
             "|---|---|" + "---|" * (len(keys) + 1)]
    for name, t, r in rows:
        vals = [r.get(k) for k in keys]
        tot = sum(v for v in vals if v is not None)
        done = sum(v is not None for v in vals)
        lines.append(f"| {name} | {t} | " + " | ".join("" if v is None else str(v) for v in vals) +
                     f" | {tot}{'' if done == len(keys) else f' ({done} tasks)'} |")
    text = "\n".join(lines) + "\n"
    (HERE / "library" / "notes" / "panel.md").write_text(text)
    return text


if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "table":
        print(table())
    elif arg == "frozen":
        print(frozen())
    else:
        print(run(arg))
        print(table())
