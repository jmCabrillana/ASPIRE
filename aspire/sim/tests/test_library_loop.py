"""library_loop.py on a fake simulator: the eval step is replaced by a table of outcomes."""

from __future__ import annotations

import importlib.util
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
PROBE = ["--seeds", "81", "82", "83"]
B = ["put_the_bowl_on_the_plate", "turn_on_the_stove", "put_the_wine_bottle_on_the_rack"]


def load_promotion():
    spec = importlib.util.spec_from_file_location("record_skill_promotion",
                                                  ROOT / "scripts" / "libero" / "record_skill_promotion.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


promotion = load_promotion()


def fake_eval(table: dict, calls: list | None = None):
    """table[task][candidate] = pass (bool, or one bool per seed). Writes what evosearch_eval.py would."""
    def run(task_dir, suite, task, seeds, gpu):
        if (task_dir / "iter_summary.json").exists():
            return
        if calls is not None:
            calls.append(task_dir)
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
                calls_ = [{"function": "goto_pose", "result": {}},
                          {"function": "close_gripper", "result": {"gripper_width": 0.3 if passed else 0.015}}]
                (t / "trace.json").write_text(json.dumps(calls_))
                (t / "summary.txt").write_text("Stdout: \nStderr: \n")
        (task_dir / "iter_summary.json").write_text(json.dumps(
            {"suite": suite, "task": task, "trial_seeds": seeds, "candidates": cands}))
    return run


@pytest.fixture
def sim_root(tmp_path, monkeypatch):
    root = tmp_path / "sim"
    live = root / ".claude" / "libero" / "skills"
    live.mkdir(parents=True)
    (live / "grasp.md").write_text("live\n")
    monkeypatch.setattr(ll, "SKILLS", live)
    return root


@pytest.fixture
def skills(sim_root):
    return ll.SKILLS


def new_run(path: Path, *extra: str) -> Path:
    assert ll.main(["plan", str(path), "--round", "0", "--pool", POOL, "--k", str(K), "--library", "empty",
                    "--eval-tasks", *B, *extra]) == 0
    return path


def write_programs(run: Path, t: int):
    for task in ll.load_state(run)["pool"]:
        for name in ("candidate_A", "candidate_B"):
            d = ll.round_dir(run, t) / "programs" / task.replace("/", "__") / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "code.py").write_text(f"# {task} {name} round {t}\n")


def promote(root: Path, run: Path, t: int, edits: dict | None):
    selection = json.loads((ll.round_dir(run, t) / "selection.json").read_text())
    suite, task = selection["task"].split("/", 1)
    build = Path(selection["repair"]["build_dir"])
    promotion.begin_promotion(root, suite=suite, task=task, build=build)
    for name, text in (edits or {}).items():
        (ll.SKILLS / name).write_text(text)
    promotion.finish_promotion(root, suite=suite, task=task, build=build, reason=None if edits else "no-op")


def write_fix(run: Path, t: int):
    code = Path(json.loads((ll.round_dir(run, t) / "selection.json").read_text())["repair"]["code"])
    code.parent.mkdir(parents=True, exist_ok=True)
    code.write_text("# fixed\n")


def one_round(root, run, t, rule, edits=None, salt="r1"):
    write_programs(run, t)
    assert ll.main(["probe", str(run), "--round", str(t), *PROBE]) == 0
    assert ll.main(["select", str(run), "--round", str(t), "--rule", rule, "--salt", salt]) == 0
    write_fix(run, t)
    promote(root, run, t, edits)
    assert ll.main(["repaired", str(run), "--round", str(t)]) == 0


def test_create_run(tmp_path, sim_root, skills):
    run = new_run(tmp_path / "run", "--checkpoints", "0", "2")
    state = ll.load_state(run)
    family = core.task_ids(core.get_family(core.load_families(), POOL))
    assert [t.split("/")[1] for t in state["eval_pool"]] == [t.split("/")[1] for t in family if t.split("/")[1] in B]
    assert len(state["pool"]) == 7 and not set(state["pool"]) & set(state["eval_pool"])
    assert not any(skills.iterdir()) and not any(ll.library_dir(run, 0).iterdir())
    with pytest.raises(SystemExit):
        ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--library", "empty"])
    with pytest.raises(SystemExit):
        ll.main(["plan", str(tmp_path / "x"), "--round", "0", "--pool", POOL])                 # --library is required
    with pytest.raises(SystemExit):
        ll.main(["plan", str(tmp_path / "y"), "--round", "0", "--pool", POOL, "--library", str(tmp_path / "nope")])
    with pytest.raises(SystemExit):
        ll.main(["plan", str(tmp_path / "z"), "--round", "0", "--pool", POOL, "--library", "empty",
                 "--eval-tasks", "not_a_task"])


def test_round_end_to_end(tmp_path, sim_root, skills, monkeypatch, capsys):
    run = new_run(tmp_path / "run")
    pool = ll.load_state(run)["pool"]
    drawer, bottle = pool[0], pool[2]
    table = {bottle: {"candidate_A": True, "candidate_B": True},
             drawer: {"candidate_A": [True, False, True], "candidate_B": False}}
    monkeypatch.setattr(ll, "run_eval", fake_eval(table))
    with pytest.raises(SystemExit):                                                           # programs missing
        ll.main(["probe", str(run), "--round", "0", *PROBE])
    write_programs(run, 0)
    assert ll.main(["probe", str(run), "--round", "0", *PROBE]) == 0
    probe = json.loads((ll.round_dir(run, 0) / "probe.json").read_text())
    assert probe["seeds"] == [81, 82, 83] and probe["coverage"] == pytest.approx(1 / 7)
    assert ll.load_state(run)["unsolved"] == [t for t in pool if t != bottle]
    assert probe["tasks"][drawer]["split"] == pytest.approx(2 / 3)
    for rule in ("split", "trace"):
        assert ll.main(["select", str(run), "--round", "0", "--rule", rule]) == 0
        assert json.loads((ll.round_dir(run, 0) / "selection.json").read_text())["task"] == drawer
    repair = json.loads((ll.round_dir(run, 0) / "selection.json").read_text())["repair"]
    assert repair["task_dir"] == str(ll.round_dir(run, 0) / "fixloop" / drawer)
    assert repair["code"].endswith("fix_code.py") and repair["start_code"].endswith("candidate_A/code.py")
    write_fix(run, 0)
    with pytest.raises(SystemExit):                                                           # repair started
        ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"])
    assert ll.main(["select", str(run), "--round", "0", "--rule", "split", "--force"]) == 0
    with pytest.raises(SystemExit):                                                           # no promotion yet
        ll.main(["repaired", str(run), "--round", "0"])
    promote(sim_root, run, 0, {"grasp.md": "drawer pattern\n"})
    with pytest.raises(SystemExit):
        ll.main(["repaired", str(run), "--round", "0", "--seeds", "51", "71"])
    table[drawer] = {"candidate_fix": True}
    assert ll.main(["repaired", str(run), "--round", "0"]) == 0
    assert ll.load_state(run)["rounds"] == [{"round": 0, "task": drawer, "rule": "split", "repair_solved": True,
                                             "check_seeds": [71, 72], "library_changed": True,
                                             "coverage": pytest.approx(1 / 7)}]
    assert (ll.library_dir(run, 1) / "grasp.md").read_text() == "drawer pattern\n"
    with pytest.raises(SystemExit):
        ll.main(["repaired", str(run), "--round", "0"])
    table[bottle] = {}                                                                        # L01 loses the bottle
    write_programs(run, 1)
    with pytest.raises(SystemExit):                                                           # S_probe is fixed
        ll.main(["probe", str(run), "--round", "1", "--seeds", "81"])
    assert ll.main(["probe", str(run), "--round", "1", *PROBE]) == 0
    assert bottle in ll.load_state(run)["unsolved"]
    with pytest.raises(SystemExit):
        ll.main(["plan", str(run), "--round", "3"])


def test_repaired_checks_the_ledger(tmp_path, sim_root, skills, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = new_run(tmp_path / "run")
    write_programs(run, 0)
    ll.main(["probe", str(run), "--round", "0", *PROBE])
    ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"])
    write_fix(run, 0)
    promote(sim_root, run, 0, {"grasp.md": "a\n"})
    (skills / "grasp.md").write_text("edited after finish\n")
    with pytest.raises(SystemExit):
        ll.main(["repaired", str(run), "--round", "0"])
    (skills / "grasp.md").write_text("a\n")
    assert ll.main(["repaired", str(run), "--round", "0"]) == 0
    assert ll.load_state(run)["rounds"][0]["library_changed"]


def test_promotion_from_wrong_library_is_refused(tmp_path, sim_root, skills, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = new_run(tmp_path / "run")
    write_programs(run, 0)
    ll.main(["probe", str(run), "--round", "0", *PROBE])
    ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"])
    write_fix(run, 0)
    (skills / "other_arm.md").write_text("x\n")
    promote(sim_root, run, 0, {"grasp.md": "a\n"})
    with pytest.raises(SystemExit):
        ll.main(["repaired", str(run), "--round", "0"])


def test_failed_repair_no_op_promotion(tmp_path, sim_root, monkeypatch, capsys):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = new_run(tmp_path / "run")
    one_round(sim_root, run, 0, "unsolved")
    r = ll.load_state(run)["rounds"][0]
    assert not r["repair_solved"] and not r["library_changed"]
    assert ll.load_state(run)["unsolved"] == ll.load_state(run)["pool"]


def test_library_guard_activate_and_saved_libraries(tmp_path, sim_root, skills, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    a = new_run(tmp_path / "a")
    one_round(sim_root, a, 0, "unsolved", {"grasp.md": "a1\n"})
    b = new_run(tmp_path / "b")
    assert ll.load_state(a)["eval_pool"] == ll.load_state(b)["eval_pool"]
    with pytest.raises(SystemExit):                                                           # live is b's L00
        ll.main(["plan", str(a), "--round", "1"])
    write_programs(b, 0)
    ll.main(["probe", str(b), "--round", "0", *PROBE])
    ll.main(["activate", str(a)])
    with pytest.raises(SystemExit):
        ll.main(["select", str(b), "--round", "0", "--rule", "unsolved"])
    assert (skills / "grasp.md").read_text() == "a1\n"
    ll.main(["activate", str(a), "--after", "0"])
    assert not any(skills.iterdir())
    with pytest.raises(SystemExit):
        ll.main(["activate", str(a), "--after", "2"])
    assert sorted(p.name for p in (a / "library").iterdir()) == ["L00", "L01"]


def test_eval_any_saved_library(tmp_path, sim_root, skills, monkeypatch, capsys):
    run = new_run(tmp_path / "run", "--checkpoints", "0")
    b0, b1, b2 = ll.load_state(run)["eval_pool"]
    table = {b0: {"candidate_A": True, "candidate_B": [True] * 5 + [False] * 5},
             b1: {"candidate_A": [True] + [False] * 9}}
    monkeypatch.setattr(ll, "run_eval", fake_eval(table))
    one_round(sim_root, run, 0, "unsolved", {"grasp.md": "v1\n"})
    with pytest.raises(SystemExit):                                                           # n = 1 not in C
        ll.main(["eval-plan", str(run)])
    with pytest.raises(SystemExit):                                                           # live is L01
        ll.main(["eval-plan", str(run), "--after", "0"])
    ll.main(["activate", str(run), "--after", "0"])
    assert ll.main(["eval-plan", str(run), "--after", "0"]) == 0
    with pytest.raises(SystemExit):
        ll.main(["eval", str(run), "--after", "0"])
    for task in (b0, b1, b2):
        for name in ("candidate_A", "candidate_B"):
            d = ll.eval_dir(run, 0) / task.replace("/", "__") / name
            d.mkdir(parents=True)
            (d / "code.py").write_text("x\n")
    with pytest.raises(SystemExit):
        ll.main(["eval", str(run), "--after", "0", "--seeds", "51"])
    assert ll.main(["eval", str(run), "--after", "0"]) == 0
    e = ll.load_state(run)["evals"]["0"]
    assert e["seeds"] == list(range(1, 11))
    assert e["sr"] == pytest.approx((0.75 + 0.05 + 0) / 3) and e["sr_best"] == pytest.approx((1 + 0.1 + 0) / 3)
    ll.main(["activate", str(run)])
    capsys.readouterr()
    assert ll.main(["report", str(run)]) == 0
    assert "SR on B L00 0.267/0.367" in capsys.readouterr().out


def test_report_refuses_different_setups(tmp_path, sim_root, monkeypatch, capsys):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    a = new_run(tmp_path / "a")
    one_round(sim_root, a, 0, "unsolved")
    b = new_run(tmp_path / "b", "--repair", "evosearch")
    one_round(sim_root, b, 0, "random", salt="r2")
    with pytest.raises(SystemExit):
        ll.main(["report", str(a), str(b)])
    capsys.readouterr()
    assert ll.main(["report", str(a), str(b), "--force", "--target", "0.5"]) == 0
    assert capsys.readouterr().out.count("regret 0.50 over 1 rounds vs coverage 0.50; not reached") == 2


def test_evosearch_repair_reads_best_code(tmp_path, sim_root, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = new_run(tmp_path / "run", "--repair", "evosearch")
    one_round(sim_root, run, 0, "random")
    repair = json.loads((ll.round_dir(run, 0) / "selection.json").read_text())["repair"]
    assert repair["method"] == "evosearch" and repair["code"].endswith("evosearch_best_code.py")
    assert Path(repair["start_code"]).exists() and Path(repair["probe_dir"]).is_dir()


def test_run_eval_clears_a_partial_eval(tmp_path, monkeypatch):
    task_dir = tmp_path / "t"
    stale = task_dir / "candidate_A" / "eval" / "trial_81_sandboxrc_0_reward_1.000_taskcompleted_1"
    stale.mkdir(parents=True)
    (task_dir / "candidate_A" / "eval_results.json").write_text("{}")
    monkeypatch.setattr(ll.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("launched")))
    with pytest.raises(RuntimeError, match="launched"):
        ll.run_eval(task_dir, "libero_goal_swap", "x", [81], "0")
    assert not stale.exists() and not (task_dir / "candidate_A" / "eval_results.json").exists()


def test_missing_trace_is_a_signature(tmp_path):
    d = tmp_path / "candidate_A" / "eval" / "trial_81_sandboxrc_1_reward_0.000_taskcompleted_0"
    d.mkdir(parents=True)
    assert core.load_signatures(tmp_path, ["candidate_A"], [81]) == {"candidate_A": ["no_trace|"]}


def test_build_skips_eval_by_default_and_every_library_stays(tmp_path, sim_root, skills, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = new_run(tmp_path / "run")
    for t in range(3):
        one_round(sim_root, run, t, "unsolved", {"grasp.md": f"v{t + 1}\n"})
    with pytest.raises(SystemExit):
        ll.main(["eval-plan", str(run)])
    for n in range(4):
        ll.main(["activate", str(run), "--after", str(n)])
        assert ll.main(["eval-plan", str(run), "--after", str(n)]) == 0
        assert [p.name for p in skills.iterdir()] == ([] if n == 0 else ["grasp.md"])
    assert (ll.library_dir(run, 2) / "grasp.md").read_text() == "v2\n"


def test_order_walks_one_permutation_without_revisits(tmp_path, sim_root, monkeypatch):
    run = new_run(tmp_path / "run")
    pool = ll.load_state(run)["pool"]
    bottle = pool[2]
    monkeypatch.setattr(ll, "run_eval", fake_eval({bottle: {"candidate_A": True}}))
    for t in range(len(pool)):
        write_programs(run, t)
        ll.main(["probe", str(run), "--round", str(t), *PROBE])
        assert ll.main(["select", str(run), "--round", str(t), "--salt", "r3"]) == 0
        write_fix(run, t)
        promote(sim_root, run, t, None)
        ll.main(["repaired", str(run), "--round", str(t)])
    assert [r["task"] for r in ll.load_state(run)["rounds"]] == core.random_order(POOL, pool, "r3")
    write_programs(run, len(pool))
    ll.main(["probe", str(run), "--round", str(len(pool)), *PROBE])
    with pytest.raises(SystemExit):
        ll.main(["select", str(run), "--round", str(len(pool))])


def test_protocol_defaults(tmp_path, sim_root, monkeypatch):
    calls = []
    monkeypatch.setattr(ll, "run_eval", fake_eval({}, calls))
    run = new_run(tmp_path / "run")
    state = ll.load_state(run)
    assert state["k"] == 2 and state["repair_iters"] == 5
    for task in state["pool"]:
        for name in ("candidate_A", "candidate_B"):
            d = ll.round_dir(run, 0) / "programs" / task.replace("/", "__") / name
            d.mkdir(parents=True)
            (d / "code.py").write_text("x\n")
    assert ll.main(["probe", str(run), "--round", "0"]) == 0
    assert ll.load_state(run)["probe_seeds"] == list(range(51, 66)) and len(calls) == len(state["pool"])
    ll.main(["select", str(run), "--round", "0"])
    assert json.loads((ll.round_dir(run, 0) / "selection.json").read_text())["repair"]["iterations"] == 5
    other = tmp_path / "default_k"
    ll.main(["plan", str(other), "--round", "0", "--pool", POOL, "--library", "empty", "--eval-tasks", *B])
    assert ll.load_state(other)["k"] == 3


def test_no_held_out_by_default(tmp_path, sim_root):
    run = tmp_path / "run"
    ll.main(["plan", str(run), "--round", "0", "--pool", POOL, "--library", "empty"])
    state = ll.load_state(run)
    assert state["eval_pool"] == [] and len(state["pool"]) == 10
    with pytest.raises(SystemExit):
        ll.main(["eval-plan", str(run), "--after", "0"])


def test_capped_fixloop_keeps_the_best_iteration(tmp_path, sim_root, monkeypatch):
    run = new_run(tmp_path / "run", "--repair-iters", "2")
    task = ll.load_state(run)["pool"][0]
    table = {task: {"candidate_A": [True, False, False], "candidate_B": False}}
    monkeypatch.setattr(ll, "run_eval", fake_eval(table))
    write_programs(run, 0)
    ll.main(["probe", str(run), "--round", "0", *PROBE])
    ll.main(["select", str(run), "--round", "0", "--rule", "split", "--gpu", "3"])
    rd = ll.round_dir(run, 0)
    repair = json.loads((rd / "selection.json").read_text())["repair"]
    prompt = (rd / "repair_prompt.md").read_text()
    assert "{{" not in prompt and repair["task_dir"] in prompt and "GPU: 3" in prompt and "budget = 2 iterations" in prompt
    fake = fake_eval({task: {"candidate_A": [True, True, False]}})
    for i, text in ((1, "# iter 1\n"), (2, "# iter 2\n")):
        d = Path(repair["task_dir"]) / f"iter_{i:02d}"
        (d / "candidate_A").mkdir(parents=True)
        (d / "candidate_A" / "code.py").write_text(text)
        fake(d, *task.split("/", 1), [81, 82, 83], "0")
    promote(sim_root, run, 0, None)
    assert ll.main(["repaired", str(run), "--round", "0"]) == 0
    assert Path(repair["code"]).read_text() == "# iter 2\n"
    assert (rd / "repair" / task.replace("/", "__") / "candidate_fix" / "code.py").read_text() == "# iter 2\n"


def test_capped_fixloop_refuses_extra_iterations(tmp_path, sim_root, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = new_run(tmp_path / "run", "--repair-iters", "1")
    write_programs(run, 0)
    ll.main(["probe", str(run), "--round", "0", *PROBE])
    ll.main(["select", str(run), "--round", "0"])
    repair = json.loads((ll.round_dir(run, 0) / "selection.json").read_text())["repair"]
    for i in (1, 2):
        (Path(repair["task_dir"]) / f"iter_{i:02d}").mkdir(parents=True)
    promote(sim_root, run, 0, None)
    with pytest.raises(SystemExit):
        ll.main(["repaired", str(run), "--round", "0"])


def test_lex_breaks_split_ties_by_shared_failure(tmp_path, sim_root, monkeypatch):
    run = new_run(tmp_path / "run")
    pool = ll.load_state(run)["pool"]
    split_task = pool[4]
    monkeypatch.setattr(ll, "run_eval", fake_eval({split_task: {"candidate_A": [True, False, True]}}))
    write_programs(run, 0)
    ll.main(["probe", str(run), "--round", "0", *PROBE])
    rd = ll.round_dir(run, 0)
    ll.main(["select", str(run), "--round", "0", "--rule", "lex"])
    assert json.loads((rd / "selection.json").read_text())["task"] == split_task
    probe = json.loads((rd / "probe.json").read_text())
    for t, p in probe["tasks"].items():
        p["split"], p["trace"], p["mode"] = 0.0, 0.0, "perceive" if t in pool[5:8] else "air"
    probe["tasks"][pool[1]]["trace"] = 0.5
    (rd / "probe.json").write_text(json.dumps(probe))
    ll.main(["select", str(run), "--round", "0", "--rule", "lex", "--force"])
    assert json.loads((rd / "selection.json").read_text())["task"] == pool[1]
    for t in pool[2:7]:
        probe["tasks"][t]["mode"] = "perceive"                                                # 5 perceive vs 2 air
    (rd / "probe.json").write_text(json.dumps(probe))
    ll.main(["select", str(run), "--round", "0", "--rule", "lex", "--force"])
    assert json.loads((rd / "selection.json").read_text())["task"] in pool[2:7]
    assert "perceive" in (rd / "repair_prompt.md").read_text()


def test_candidate_rounds_probe_a_sample_and_never_repeat_a_repaired_task(sim_root, monkeypatch):
    monkeypatch.setattr(ll, "run_eval", fake_eval({}))
    run = sim_root / "run"
    new_run(run, "--candidates", "3")
    pool = ll.load_state(run)["pool"]
    assert len(pool) >= 4
    cands = json.loads((ll.round_dir(run, 0) / "candidates.json").read_text())
    assert len(cands) == 3 and set(cands) <= set(pool)
    assert {d.name for d in (ll.round_dir(run, 0) / "programs").iterdir()} == {c.replace("/", "__") for c in cands}
    for task in cands:
        for name in ("candidate_A", "candidate_B"):
            d = ll.round_dir(run, 0) / "programs" / task.replace("/", "__") / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "code.py").write_text("# p\n")
    assert ll.main(["probe", str(run), "--round", "0", *PROBE]) == 0
    probe = json.loads((ll.round_dir(run, 0) / "probe.json").read_text())
    assert set(probe["tasks"]) == set(cands)
    assert ll.main(["select", str(run), "--round", "0", "--rule", "unsolved"]) == 0
    chosen = json.loads((ll.round_dir(run, 0) / "selection.json").read_text())["task"]
    assert chosen in cands
    state = ll.load_state(run)
    assert set(state["unsolved"]) == set(pool)
