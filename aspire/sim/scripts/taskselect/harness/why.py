#!/usr/bin/env python3
"""Why did these trials fail? One compact table per eval directory, from the API traces only (no simulator state).

  .venv/bin/python3 scripts/taskselect/harness/why.py <dir with trial_* or candidate_*/eval> [...]

Per trial: failure stage and tags, every SAM3 text query (best score, runner-up, 2D centre of the best mask), the centre of
each oriented box fitted to a mask, every gripper close (width, last IK target before it) and open. Then, per query, how often
the best score is below the "seen" threshold, and where passing and failing trials grasped (mean x, y).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import core  # noqa: E402


def trial_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("trial_*")) or sorted(root.glob("candidate_*/eval/trial_*"))


def events(trial: Path) -> dict:
    """Queries, boxes, grasps and releases of one trial, in call order."""
    queries, boxes, grasps, opens, target = [], [], [], 0, None
    for e in json.loads((trial / "trace.json").read_text()):
        fn, a, r = e["function"], e.get("args", {}), e.get("result") or {}
        if fn == "segment_sam3_text_prompt":
            b = r.get("mask_0_bbox") or [0, 0, 0, 0]
            queries.append({"prompt": a.get("text_prompt"), "n": r.get("num_masks", 0), "score": r.get("mask_0_score", 0.0),
                            "second": r.get("mask_1_score"), "uv": [round((b[0] + b[2]) / 2), round((b[1] + b[3]) / 2)]})
        elif fn == "get_oriented_bounding_box_from_3d_points":
            boxes.append([round(x, 3) for x in r.get("center", [])])
        elif fn == "solve_ik":
            target = [round(x, 3) for x in a.get("position", [])]
        elif fn == "close_gripper":
            grasps.append({"width": round(r.get("gripper_width", 0.0), 3), "at": target})
        elif fn == "open_gripper":
            opens += 1
    return {"queries": queries, "boxes": boxes, "grasps": grasps, "opens": opens}


def main() -> None:
    rows = []
    for root in map(Path, sys.argv[1:]):
        for t in trial_dirs(root):
            sig = core.trial_signature(t)
            rows.append((t, sig, events(t)))
            ev = rows[-1][2]
            print(f"{t.parent.parent.name + '/' if t.parent.name == 'eval' else ''}{t.name.split('_sandbox')[0]}  {sig}")
            for q in ev["queries"]:
                second = "" if q["second"] is None else f" second {q['second']:.2f}"
                print(f"    sam3 '{q['prompt']}': {q['n']} masks, best {q['score']:.2f}{second} at pixel {q['uv']}")
            if ev["boxes"]:
                print(f"    box centres {ev['boxes'][:4]}")
            for g in ev["grasps"]:
                print(f"    close width {g['width']} after IK target {g['at']}")
    print("\n== per query: trials where the best score is under the seen threshold "
          f"({core.SEEN_SCORE}), and the lowest best score")
    by: dict[str, list[float]] = {}
    for _, _, ev in rows:
        for q in ev["queries"]:
            by.setdefault(q["prompt"], []).append(q["score"])
    for p, s in sorted(by.items()):
        print(f"  '{p}': {sum(x < core.SEEN_SCORE for x in s)}/{len(s)} unseen, min {min(s):.2f}, mean {fmean(s):.2f}")
    print("\n== first grasp target (x, y): mean over passing trials against failing trials")
    for label, keep in (("pass", True), ("fail", False)):
        pts = [ev["grasps"][0]["at"] for _, sig, ev in rows if sig.startswith("pass") == keep and ev["grasps"] and ev["grasps"][0]["at"]]
        print(f"  {label}: n={len(pts)}" + (f", x {fmean(p[0] for p in pts):.3f}, y {fmean(p[1] for p in pts):.3f}" if pts else ""))


if __name__ == "__main__":
    main()
