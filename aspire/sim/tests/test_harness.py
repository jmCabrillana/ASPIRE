"""Harness tools: acceptance decision and kept-iteration choice, the diagnosis table, and the harness coordinator prompt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "taskselect"))
sys.path.insert(0, str(ROOT / "scripts" / "taskselect" / "harness"))

import accept  # noqa: E402
import run_reps  # noqa: E402
import why  # noqa: E402


def write_iter(task_dir: Path, name: str, passes: int) -> None:
    d = task_dir / name
    d.mkdir(parents=True)
    (d / "iter_summary.json").write_text(json.dumps({"candidates": [{"candidate": "candidate_A", "pass_count": passes,
                                                                       "code_path": str(d / "candidate_A" / "code.py")}]}))


def test_decide_needs_the_margin():
    assert accept.decide(4, 7, 3) and not accept.decide(4, 6, 3) and not accept.decide(9, 4, 3)


def test_kept_iteration_is_the_most_passes_and_the_later_on_a_tie(tmp_path):
    for name, n in (("iter_01", 3), ("iter_02", 7), ("iter_03", 2), ("iter_04", 7)):
        write_iter(tmp_path, name, n)
    code, passes = accept.kept_iteration(tmp_path)
    assert code.parts[-3] == "iter_04" and passes == 7


def make_trial(root: Path, name: str, score: float, close_width: float) -> None:
    t = root / name
    t.mkdir(parents=True)
    trace = [{"function": "segment_sam3_text_prompt", "args": {"text_prompt": "ramekin"},
              "result": {"num_masks": 3, "mask_0_score": score, "mask_0_bbox": [0, 0, 10, 10], "mask_1_score": 0.1}},
             {"function": "solve_ik", "args": {"position": [0.5, 0.1, 0.2]}, "result": {}},
             {"function": "move_to_joints", "args": {}, "result": {}},
             {"function": "close_gripper", "args": {}, "result": {"gripper_width": close_width}}]
    (t / "trace.json").write_text(json.dumps(trace))
    (t / "summary.txt").write_text("Stdout:\n\nStderr:\n")


def test_why_reports_unseen_queries_and_grasps(tmp_path, capsys, monkeypatch):
    make_trial(tmp_path, "trial_51_sandboxrc_0_reward_0.000_taskcompleted_0", 0.002, 0.0)
    make_trial(tmp_path, "trial_52_sandboxrc_0_reward_1.000_taskcompleted_1", 0.9, 0.05)
    monkeypatch.setattr(sys, "argv", ["why.py", str(tmp_path)])
    why.main()
    out = capsys.readouterr().out
    assert "perceive|" in out and "pass|" in out
    assert "'ramekin': 1/2 unseen" in out and "close width 0.0 after IK target [0.5, 0.1, 0.2]" in out


def test_harness_prompt_selects_unsolved_and_checks_the_repair():
    run_reps.SEEDS, run_reps.ROUNDS, run_reps.TAG = list(range(51, 81)), 5, "_s30"
    text = run_reps.coordinator_prompt("harness", 2)
    assert "RUN = outputs/library_harness_s30_r2" in text and "--rule unsolved --salt r2" in text
    assert "harness/accept.py" in text and "--seeds 81 82" in text and "repair_harness_addendum.md" in text
    assert "HARNESS ARM" not in run_reps.coordinator_prompt("lex", 2)
