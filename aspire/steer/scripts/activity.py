#!/usr/bin/env python3
"""Build a zoomable timeline of everything the research agent did (reports/activity/activity_<library version>.html, plus
reports/activity/latest.html).

  .venv/bin/python scripts/activity.py

Sources: the agent's Claude Code transcripts (~/.claude/projects/<this folder>/*.jsonl: every shell command, file write and its
output), the explorer log (held-out verifications), the library version manifest, the lab scripts and the journal. Each event
is classified (simulator run, offline lab, code written, adapter training, verification, inspection) and placed in a research
thread lane (language, swapped layouts, edits and neurons, adapters, library and tools).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
REPORTS = HERE / "reports" / "activity"
TRANSCRIPTS = Path.home() / ".claude" / "projects" / ("-" + str(HERE).strip("/").replace("/", "-"))
LOCAL = timezone(timedelta(hours=8))
sys.path.insert(0, str(HERE))
from steer.versions import current  # noqa: E402
THREADS = [
    ("swap", "Swapped layouts", r"swap|handback|hand-back|virtual|track|siglip|scene\.|calib|homograph|footprint|reference|carry"),
    ("edits", "Edits and neurons", r"lexedit|lexfact|edit\.py|rank.one|neuron|attribut|levers|optimize_vector|patch|probe|bank|interp|jacobian|cluster|dose"),
    ("adapters", "Adapters", r"lora|adapter|selftrain|composite"),
    ("language", "Language", r"_lan|canon|paraphrase|rewrite|dialect|lm_probe|instruction|verb|token"),
    ("library", "Library and tools", r"steerlib|knowledge|readme|journal|summary|panel|tools/|recipes|discover"),
]


def parse_time(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def classify(cmd: str, desc: str) -> str:
    c = cmd
    if "run_program.py" in c:
        return "sim"
    if "VERIFY_" in c:
        return "verify"
    if re.search(r"lora\.train|selftrain|train_adapter|lora_train", c):
        return "train"
    if re.search(r"notes/lab/e\d+|steer\.lab|from steer\.(lab|interp|edit)|levers|optimize_vector", c):
        return "lab"
    if re.search(r"cat\s*>\s*\S+|tee\s+\S+\.py|<<\s*'?EOF'?\s*>\s*\S+", c) and not c.lstrip().startswith(("python", ".venv")):
        return "write"
    if re.search(r"\.venv/bin/python|python3? ", c):
        return "script"
    return "inspect"


def thread(text: str) -> str:
    low = text.lower()
    scores = [(len(re.findall(p, low)), key) for key, _, p in THREADS]
    best = max(scores)
    return best[1] if best[0] else "library"


def sim_fields(cmd: str, out: str) -> dict:
    f = {}
    for k, pat in (("program", r"--program\s+(\S+)"), ("suite", r"--suite\s+(\S+)"), ("task", r"--task\s+(\S+)"),
                   ("episodes", r"--episodes\s+(\S+)")):
        m = re.search(pat, cmd)
        if m:
            f[k] = Path(m.group(1)).stem if k == "program" else m.group(1)
    m = re.findall(r"success (\d\.\d+) over episodes", out)
    if m:
        f["success"] = float(m[-1])
    return f


def transcript_events() -> list[dict]:
    events, sessions = [], []
    for path in sorted(TRANSCRIPTS.glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        uses, results, first, last = {}, {}, None, None
        for line in path.open():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = d.get("timestamp")
            if not ts:
                continue
            t = parse_time(ts)
            first, last = first or t, t
            msg = d.get("message") or {}
            for c in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
                if c.get("type") == "tool_use":
                    uses[c["id"]] = (t, c["name"], c.get("input", {}))
                elif c.get("type") == "tool_result":
                    content = c.get("content")
                    text = content if isinstance(content, str) else " ".join(
                        x.get("text", "") for x in (content or []) if isinstance(x, dict))
                    results[c.get("tool_use_id")] = (t, text)
        if first is None:
            continue
        sessions.append({"id": path.stem[:8], "start": first, "end": last})
        for uid, (t, name, inp) in uses.items():
            end, out = results.get(uid, (t, ""))
            if name == "Bash":
                cmd, desc = inp.get("command", ""), inp.get("description", "")
                kind = classify(cmd, desc)
            elif name in ("Write", "Edit"):
                cmd, desc, kind = inp.get("file_path", ""), f"{name} {Path(inp.get('file_path', '')).name}", "write"
            elif name == "Read":
                cmd, desc, kind = inp.get("file_path", ""), f"Read {Path(inp.get('file_path', '')).name}", "inspect"
            else:
                cmd, desc, kind = json.dumps(inp)[:400], name, "script"
            e = {"t": t, "end": max(end, t), "kind": kind, "lane": thread(cmd + " " + desc), "title": desc or cmd[:80],
                 "cmd": cmd[:1600], "out": out[:1400], "session": path.stem[:8]}
            if kind == "sim":                     # the run itself comes from its summary (simulator_runs); keep the command
                kind = e["kind"] = "script"
            events.append(e)
    return events, sessions


def simulator_runs() -> list[dict]:
    """Development evaluations from their summaries (outputs/programs/<program>/<suite>__<task>/summary.json; the latest run of
    each program and task, since a rerun overwrites the summary). Many runs were queued in the background, so the transcript
    alone misses them."""
    out = []
    for f in (HERE / "outputs" / "programs").glob("*/*/summary.json"):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        eps = [e["episode"] for e in d.get("episodes", [])]
        secs = sum(e.get("seconds", 0) for e in d.get("episodes", []))
        t1 = f.stat().st_mtime
        rate = d.get("success_rate", 0.0)
        k = sum(e.get("success", False) for e in d.get("episodes", []))
        prog, suite, task = d.get("program", f.parent.parent.name), d.get("suite", ""), d.get("task", "")
        out.append({"t": t1 - secs, "end": t1, "kind": "sim", "lane": thread(f"{prog} {suite}"),
                    "title": f"{prog} on {suite} {task}: {k}/{len(eps)}", "cmd": f"scripts/run_program.py --program {prog} "
                    f"--suite {suite} --task {task} --episodes {min(eps, default=0)}-{max(eps, default=0)}",
                    "out": "", "program": prog, "suite": suite, "task": str(task),
                    "episodes": f"{min(eps, default=0)}-{max(eps, default=0)}", "success": rate})
    return out


def verifications() -> list[dict]:
    out, log = [], HERE / "outputs" / "explore.log"
    if not log.exists():
        return out
    for line in log.read_text().splitlines():
        m = re.match(r"(\d\d)-(\d\d) (\d\d:\d\d:\d\d) \| (\S+) \| (\S+) (\d+) \| (\d+)/(\d+) \| (\d+)/(\d+) \| ([\d.]+) \| ([^|]+)\|", line)
        if not m:
            continue
        mo, dy, hms, prog, suite, task, kb, n, kp, _, p, verdict = m.groups()
        t = datetime.strptime(f"2026-{mo}-{dy} {hms}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL).timestamp()
        promoted = verdict.strip() == "PROMOTED"
        out.append({"t": t, "end": t, "kind": "verified" if promoted else "rejected", "lane": thread(prog + " " + suite),
                    "title": f"{prog} on {suite} {task}: {kb}/{n} → {kp}/{n}" + ("" if promoted else " (not significant)"),
                    "cmd": f"held-out verification, initial states 20-49, one-sided Fisher p = {p}", "out": "", "program": prog,
                    "suite": suite, "task": task, "frozen": int(kb), "steered": int(kp), "n": int(n)})
    return out


def library_versions() -> list[dict]:
    f = HERE / "outputs" / "library_versions" / "versions.json"
    if not f.exists():
        return []
    out = []
    for v in json.loads(f.read_text()):
        t = datetime.strptime(v["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL).timestamp()
        added = ", ".join(v.get("added", [])[:12])
        out.append({"t": t, "end": t, "kind": "version", "lane": "library", "title": f"library {v['version']}",
                    "cmd": f"{v['trigger']}", "out": f"added: {added}" if added else "", "version": v["version"]})
    return out


def experiments() -> list[dict]:
    """Journal sections of numbered experiments (E01, E02, ...), timed by their lab script."""
    journal = HERE / "library" / "notes" / "JOURNAL.md"
    if not journal.exists():
        return []
    text = journal.read_text()
    scripts = {m.group(1): p for p in (HERE / "library" / "notes" / "lab").glob("e*.py")
               if (m := re.match(r"e(\d+)", p.name))}
    out = []
    for m in re.finditer(r"^## (E(\d+)[^\n]*)\n(.*?)(?=^## |\Z)", text, flags=re.M | re.S):
        head, num, body = m.group(1), m.group(2), m.group(3).strip()
        script = scripts.get(num) or scripts.get(num.zfill(2))
        if script is None:
            continue
        t = script.stat().st_mtime
        out.append({"t": t, "end": t, "kind": "finding", "lane": thread(head + " " + body), "title": head[:140],
                    "cmd": f"{script.relative_to(HERE)}", "out": body[:1800]})
    return out


def program_uses(stem: str) -> list[str]:
    """Library entries a program uses: steerlib modules it imports and library files it loads."""
    for d in ("programs/candidates", "programs", "programs/probes"):
        f = HERE / "library" / d / f"{stem}.py"
        if f.exists():
            src = f.read_text()
            break
    else:
        return []
    mods = set()
    for m in re.finditer(r"from steerlib import ([A-Za-z_, ]+(?: as \w+)?(?:, [A-Za-z_]+(?: as \w+)?)*)", src):
        mods |= {x.strip().split(" as ")[0] for x in m.group(1).split(",") if x.strip()}
    mods |= set(re.findall(r"steerlib\.(\w+)", src))
    uses = sorted(f"steerlib/{m}.py" for m in mods if (HERE / "library" / "steerlib" / f"{m}.py").exists())
    uses += sorted(set(re.findall(r"((?:edits|adapters|directions|composites|knowledge)/[A-Za-z0-9_.\-]+)", src)))
    return uses


def library_items() -> dict[str, dict]:
    """When each library file first appeared (version and time), from the version manifest."""
    f = HERE / "outputs" / "library_versions" / "versions.json"
    items = {}
    if f.exists():
        for v in json.loads(f.read_text()):
            t = datetime.strptime(v["time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL).timestamp()
            for path in v.get("added", []):
                items.setdefault(path, {"version": v["version"], "t": t})
    return items


def main() -> None:
    events, sessions = transcript_events()
    events += simulator_runs() + verifications() + library_versions() + experiments()
    events.sort(key=lambda e: e["t"])
    items = library_items()
    for e in events:
        if e.get("program"):
            used = program_uses(e["program"])
            e["uses"] = [u for u in used if any(k == u or k.startswith(u) for k in items)] or used
    data = {"generated": datetime.now(LOCAL).strftime("%Y-%m-%d %H:%M"), "sessions": sessions, "events": events,
            "lanes": [{"key": k, "name": n} for k, n, _ in THREADS], "items": items}
    version = (current() or {}).get("version", "v000")
    data["library_version"] = version
    template = (HERE / "scripts" / "activity_template.html").read_text()
    page = template.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"activity_{version}.html"         # one page per library version; latest.html is the newest
    out.write_text(page)
    (REPORTS / "latest.html").write_text(page)
    kinds = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    print(f"wrote {out} ({out.stat().st_size // 1024} KB): {len(events)} events {kinds}")


if __name__ == "__main__":
    main()
