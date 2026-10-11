"""library_loop.py on a fake simulator: the eval step is replaced by a table of outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "taskselect"))

import core  # noqa: E402
import library_loop as ll  # noqa: E402

POOL = "goal_swap"
K = 2


def fake_eval(table: dict):
    """table[task][candidate] = pass (bool, or one bool per seed). Writes what evosearch_eval.py would."""
    def run(task_dir, suite, task, seeds, gpu):
        if (task_dir / "iter_summary.json").exists():
            return
        cands = []
        for d in sorted(task_dir.glob("candidate_*")):
            spec = table.get(f"{suite}/{task}", {}).get(d.name, False)
            per_seed = dict(zip(seeds, spec if isinstance(spec, list) else [spec] * len(seeds)))
            cands.append({"candidate": d.name, "errors": 0,
                          "trial_results": [{"trial": s, "task_completed": int(per_seed[s])} for s in seeds]})
            for s in seeds:
                passed = per_seed[s]
                t = d / "eval" / f"trial_{s}_sandboxrc_0_reward_{float(passed):.3f}_taskcompleted_{int(passed)}"
                t.mkdir(parents=True)
                calls = [{"function": "goto_pose", "result": {}},
                         {"function": "close_gripper", "result": {"gripper_width": 0.3 if passed else 0.015}}]
                (t / "trace.json").write_text(json.dumps(calls))
                (t / "summary.txt").write_text("Stdout: \nStderr: \n")
        (task_dir / "iter_summary.json").write_text(json.dumps(
            {"suite": suite, "task": task, "trial_seeds": seeds, "candidates": cands}))
    return run


def write_programs(run: Path, t: int, pool: list[str], solved: set[str]):
    for task in pool:
        if task in solved:
            continue
        for name in ("candidate_A", "candidate_B"):
            d = run / f"round_{t:02d}" / "programs" / task.replace("/", "__") / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "code.py").write_text(f"# {task} {name} round {t}\n")


@pytest.fixture(autouse=True)
def skills(tmp_path, monkeypatch):
    live = tmp_path / "skills"
    live.mkdir()
    (live / "grasp.md").write_text("v0\n")
    monkeypatch.setattr(ll, "SKILLS", live)
    return live


@pytest.fixture
def family():
    return core.task_ids(core.get_family(core.load_families(), POOL))


def train_pool(run: Path) -> list[str]:
    return ll.load_state(run)["pool"]


def test_one_round_end_to_end(tmp_path, family, skills, monkeypatch, capsys):
    run = tmp_path / "run"
    assert ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--k", str(K), "--eval-split", "3"]) == 0
    pool = train_pool(run)
    assert len(pool) == 7 and not set(pool) & set(ll.load_state(run)["eval_pool"])
    drawer, bottle = pool[0], pool[2]
    table = {bottle: {"candidate_A": True, "candidate_B": True},            # solved zero-shot
             drawer: {"candidate_A": [True, False], "candidate_B": False}}  # fragile: the only split, not solved
    monkeypatch.setattr(ll, "run_eval", fake_eval(table))
    assert sum(line.startswith("  ") for line in capsys.readouterr().out.splitlines()) == len(pool)
    with pytest.raises(SystemExit):                                          # programs missing
        ll.main(["probe", str(run), "--round", "0"])
    write_programs(run, 0, pool, set())
    assert ll.main(["probe", str(run), "--round", "0"]) == 0
    state = ll.load_state(run)
    assert set(state["solved"]) == {bottle} and state["solved"][bottle]["how"] == "zero-shot"
    probe = json.loads((run / "round_00" / "probe.json").read_text())
    assert probe["tasks"][drawer]["split"] == 0.5 and probe["tasks"][drawer]["trace"] == 0.5
    assert probe["seeds"] == [51, 52]
    assert probe["tasks"][pool[1]]["split"] == 0.0 and probe["tasks"][bottle]["passing"] == "candidate_A"
    for rule in ("split", "trace"):
        assert ll.main(["select", str(run), "--round", "0", "--rule", rule]) == 0
        assert json.loads((run / "round_00" / "selection.json").read_text())["task"] == drawer
    assert ll.main(["select", str(run), "--round", "0", "--rule", "unsolved", "--salt", "r1"]) == 0
    assert json.loads((run / "round_00" / "selection.json").read_text())["task"] != bottle
    assert ll.main(["select", str(run), "--round", "0", "--rule", "split"]) == 0
    fix = tmp_path / "fix_code.py"
    fix.write_text("# repaired\n")
    table[drawer] = {"candidate_fix": True}
    with pytest.raises(SystemExit):                                          # debug seeds are refused
        ll.main(["repaired", str(run), "--round", "0", "--code", str(fix), "--seeds", "51", "52"])
    (skills / "grasp.md").write_text("v1\n")                                 # promotion
    assert ll.main(["repaired", str(run), "--round", "0", "--code", str(fix)]) == 0
    state = ll.load_state(run)
    assert state["rounds"] == [{"round": 0, "task": drawer, "repair_solved": True, "check_seeds": [71, 72],
                                "library_changed": True, "coverage": 2 / 7}]
    assert (run / "library" / "L01" / "grasp.md").read_text() == "v1\n"
    with pytest.raises(SystemExit):                                          # a round is recorded once
        ll.main(["repaired", str(run), "--round", "0", "--code", str(fix)])
    sel = json.loads((run / "round_00" / "selection.json").read_text())
    assert sel["repair"]["task_dir"] == str(run / "round_00" / "fixloop" / drawer)
    assert sel["repair"]["code"].endswith("fix_code.py")
    assert sel["repair"]["start_code"].endswith(drawer.replace("/", "__") + "/candidate_A/code.py")
    assert state["solved"][drawer]["how"] == "repair"
    capsys.readouterr()
    assert ll.main(["plan", str(run), "--round", "1"]) == 0                 # next round lists the 8 unsolved tasks
    assert len(capsys.readouterr().out.strip().splitlines()) == 1 + len(pool) - 2
    with pytest.raises(SystemExit):
        ll.main(["plan", str(run), "--round", "3"])


def test_library_guard_and_activate(tmp_path, skills, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    a, b = tmp_path / "a", tmp_path / "b"
    ll.main(["plan", str(a), "--round", "0", "--pool", POOL, "--k", str(K)])
    ll.main(["plan", str(b), "--round", "0", "--pool", POOL, "--k", str(K)])
    assert ll.load_state(a)["eval_pool"] == ll.load_state(b)["eval_pool"]
    write_programs(a, 0, train_pool(a), set())
    ll.main(["probe", str(a), "--round", "0"])
    ll.main(["select", str(a), "--round", "0", "--rule", "unsolved"])
    (skills / "grasp.md").write_text("a1\n")
    fix = tmp_path / "fix.py"
    fix.write_text("x\n")
    ll.main(["repaired", str(a), "--round", "0", "--code", str(fix)])
    write_programs(b, 0, train_pool(b), set())
    with pytest.raises(SystemExit):                                          # b still expects v0
        ll.main(["probe", str(b), "--round", "0"])
    assert ll.main(["activate", str(b)]) == 0 and (skills / "grasp.md").read_text() == "v0\n"
    assert ll.main(["probe", str(b), "--round", "0"]) == 0
    with pytest.raises(SystemExit):
        ll.main(["plan", str(a), "--round", "1"])
    ll.main(["activate", str(a)])
    assert (skills / "grasp.md").read_text() == "a1\n"


def test_eval_on_held_out(tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--k", str(K), "--checkpoints", "0", "2"])
    b0, b1, b2 = ll.load_state(run)["eval_pool"]
    monkeypatch.setattr(ll, "run_eval", fake_eval({b0: {"candidate_A": True, "candidate_B": [True, False, False]},
                                                   b1: {"candidate_A": [True, False, False]}}))
    assert ll.main(["eval-plan", str(run)]) == 0
    with pytest.raises(SystemExit):                                          # programs missing
        ll.main(["eval", str(run)])
    for task in (b0, b1, b2):
        for name in ("candidate_A", "candidate_B"):
            d = run / "eval" / "after_00" / task.replace("/", "__") / name
            d.mkdir(parents=True)
            (d / "code.py").write_text("x\n")
    assert ll.main(["eval", str(run)]) == 0
    e = ll.load_state(run)["evals"]["0"]
    assert e["sr"] == pytest.approx((1 + 1 / 3 + 0) / 3) and e["sr_mean"] == pytest.approx((2 / 3 + 1 / 6) / 3)
    state = ll.load_state(run)
    state["rounds"].append({"round": 0, "task": "x", "repair_solved": False, "coverage": 0.0})
    ll.save_state(run, state)
    (run / "library" / "L00").rename(run / "library" / "L01")
    with pytest.raises(SystemExit):                                          # n = 1 is not a checkpoint
        ll.main(["eval-plan", str(run)])
    capsys.readouterr()
    assert ll.main(["report", str(run)]) == 0
    assert "SR on B n=0 0.44/0.28" in capsys.readouterr().out


def test_failed_repair_and_report(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    runs = []
    for name, coverages in (("a", [False, True]), ("b", [False, False])):
        run = tmp_path / name
        ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--k", str(K)])
        pool = train_pool(run)
        fix = tmp_path / "fix.py"
        fix.write_text("x\n")
        for t, ok in enumerate(coverages):
            write_programs(run, t, pool, set(ll.load_state(run)["solved"]))
            ll.main(["probe", str(run), "--round", str(t)])
            ll.main(["select", str(run), "--round", str(t), "--rule", "random", "--salt", "r2"])
            task = json.loads((run / f"round_{t:02d}" / "selection.json").read_text())["task"]
            monkeypatch.setattr(ll, "run_eval", fake_eval({task: {"candidate_fix": ok}}))
            ll.main(["repaired", str(run), "--round", str(t), "--code", str(fix)])
        runs.append(str(run))
    assert [x["coverage"] for x in ll.load_state(Path(runs[0]))["rounds"]] == [0.0, 1 / 7]
    capsys.readouterr()
    assert ll.main(["report", *runs]) == 0
    out = capsys.readouterr().out
    assert "a: regret 0.14 over 2 rounds vs coverage 0.14; reached at round 1" in out
    assert "b: regret 0.29 over 2 rounds vs coverage 0.14; not reached" in out


def test_empty_library(tmp_path, skills, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = tmp_path / "run"
    ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--k", str(K), "--library", "empty"])
    assert not any((run / "library" / "L00").iterdir()) and not any(skills.iterdir())
    assert ll.main(["eval-plan", str(run)]) == 0
    write_programs(run, 0, train_pool(run), set())
    ll.main(["probe", str(run), "--round", "0"])
    ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"])
    (skills / "grasp.md").write_text("first skill\n")
    fix = tmp_path / "fix.py"
    fix.write_text("x\n")
    ll.main(["repaired", str(run), "--round", "0", "--code", str(fix)])
    assert ll.load_state(run)["rounds"][0]["library_changed"]
    assert (run / "library" / "L01" / "grasp.md").read_text() == "first skill\n"


def test_repaired_reads_fix_code_from_round_dir(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--k", str(K)])
    write_programs(run, 0, train_pool(run), set())
    ll.main(["probe", str(run), "--round", "0"])
    ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"])
    task_dir = Path(json.loads((run / "round_00" / "selection.json").read_text())["repair"]["task_dir"])
    task_dir.mkdir(parents=True)
    (task_dir / "fix_code.py").write_text("# fixed\n")
    assert ll.main(["repaired", str(run), "--round", "0"]) == 0
    task = ll.load_state(run)["rounds"][0]["task"]
    code = run / "round_00" / "repair" / task.replace("/", "__") / "candidate_fix" / "code.py"
    assert code.read_text() == "# fixed\n"


def test_evosearch_repair_reads_best_code(tmp_path, monkeypatch):
    run = tmp_path / "run"
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--k", str(K), "--repair", "evosearch"])
    write_programs(run, 0, train_pool(run), set())
    ll.main(["probe", str(run), "--round", "0"])
    ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"])
    repair = json.loads((run / "round_00" / "selection.json").read_text())["repair"]
    assert repair["method"] == "evosearch" and repair["code"].endswith("evosearch_best_code.py")
    Path(repair["task_dir"]).mkdir(parents=True)
    Path(repair["code"]).write_text("# best\n")
    assert ll.main(["repaired", str(run), "--round", "0"]) == 0
