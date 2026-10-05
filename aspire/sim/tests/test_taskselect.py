"""Tests for scripts/taskselect (task-family evosearch with disagreement-based task selection).

Synthetic run directories only: no simulator, no GPU, no model calls.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "taskselect"))

import backtest  # noqa: E402
import core  # noqa: E402
import select_task  # noqa: E402

DEV = list(range(51, 66))
FAMILY = "goal_swap"
GOAL = "libero_goal_swap"


def write_summary(d: Path, task_id: str, outcomes: dict[str, list[int]], seeds: list[int], code: bool = True) -> Path:
    """iter_summary.json (as evosearch_eval.py writes it) straight into directory d."""
    suite, task = task_id.split("/", 1)
    d.mkdir(parents=True, exist_ok=True)
    cands = []
    for name, row in outcomes.items():
        assert len(row) == len(seeds)
        cands.append({"candidate": name, "errors": 0,
                      "trial_results": [{"trial": s, "task_completed": int(x)} for s, x in zip(seeds, row)]})
        if code:
            (d / name).mkdir(exist_ok=True)
            (d / name / "code.py").write_text(f"# {name}\nprint('{name}')\n")
    (d / "iter_summary.json").write_text(json.dumps({"suite": suite, "task": task, "trial_seeds": seeds,
                                                      "candidates": cands}))
    return d


def write_round(run_dir: Path, n: int, task_id: str, outcomes: dict[str, list[int]],
                seeds: list[int] | None = None, code: bool = True) -> Path:
    """An evosearch-style iter_NN directory with iter_summary.json (and candidate code)."""
    return write_summary(run_dir / f"iter_{n:02d}", task_id, outcomes, seeds or DEV, code)


def committee(rows: dict[str, int]) -> dict[str, list[int]]:
    """candidate -> number of passing dev seeds (first k seeds pass)."""
    return {name: [1] * k + [0] * (len(DEV) - k) for name, k in rows.items()}


def run_select(run_dir: Path, n: int, rule: str, *extra: str, family: str = FAMILY,
               families: Path | None = None) -> int:
    argv = ["select", "--family", family, "--run-dir", str(run_dir), "--round", str(n), "--rule", rule, *extra]
    if families:
        argv += ["--families", str(families)]
    return select_task.main(argv)


@pytest.fixture
def families_doc():
    return core.load_families()


@pytest.fixture
def goal(families_doc):
    return core.get_family(families_doc, FAMILY)


@pytest.fixture
def run_dir(tmp_path, goal):
    """A run with iter_00 evaluated on the family's round-0 task and the three-parent selection written."""
    r = tmp_path / "run"
    write_round(r, 0, goal["round0"], committee({"candidate_A": 3, "candidate_B": 9, "candidate_C": 0}))
    assert run_select(r, 0, "random") == 0
    return r


def test_families_file_is_consistent(families_doc):
    assert families_doc["dev_seeds"] == DEV
    assert len(families_doc["families"]) >= 6
    for name, fam in families_doc["families"].items():
        ids = core.task_ids(fam)
        assert len(ids) == len(set(ids)) >= 6, name
        assert fam["round0"] in ids, name
        assert fam["relation"] in {"tight", "strong", "medium", "weak"}
        for t in fam["tasks"]:
            assert t["id"] == f"{t['suite']}/{t['task']}" and t["language_static"], (name, t)
        assert not any(t["suite"].endswith("_task") for t in fam["tasks"]), "_task suites have remapped instructions"
    assert {f["relation"] for f in families_doc["families"].values()} == {"tight", "strong", "medium", "weak"}


def test_design_doc_table_matches_families_file(families_doc):
    text = (ROOT / "scripts" / "taskselect" / "DESIGN.md").read_text()
    rows = re.findall(r"^\| `(\w+)` \| (\w+) \| (\d+) \|", text, flags=re.M)
    assert rows, "family table not found in DESIGN.md"
    documented = {name: (relation, int(n)) for name, relation, n in rows}
    actual = {name: (f["relation"], len(f["tasks"])) for name, f in families_doc["families"].items()}
    assert documented == actual


def test_round0_is_stable_hash(families_doc):
    import hashlib
    for name, fam in families_doc["families"].items():
        ids = core.task_ids(fam)
        assert fam["round0"] == ids[int(hashlib.sha256(name.encode()).hexdigest(), 16) % len(ids)]


def test_committee_metrics_hand_computed():
    m = core.committee_metrics({"A": [1, 1, 1, 1], "B": [1, 1, 0, 0], "C": [0, 0, 0, 0], "D": [1, 0, 1, 0]})
    assert m["K"] == 4 and m["n_seeds"] == 4
    assert m["D_prog"] == pytest.approx(0.5)
    assert m["D_seed"] == pytest.approx((2 * 0.8112781244591328 + 2.0) / 4, abs=1e-5)
    assert m["hamming_mean"] == pytest.approx(14 / 24, abs=1e-5)
    assert m["mixed_seed_frac"] == pytest.approx(1.0)
    assert m["pass_rates"] == {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.5}
    assert not m["all_pass"] and not m["all_fail"]


def test_committee_metrics_degenerate_cases():
    allp = core.committee_metrics({"A": [1, 1], "B": [1, 1]})
    allf = core.committee_metrics({"A": [0, 0], "B": [0, 0]})
    for m in (allp, allf):
        assert m["D_prog"] == m["D_seed"] == m["hamming_mean"] == m["mixed_seed_frac"] == 0.0
    assert allp["all_pass"] and allf["all_fail"]
    # same pass RATE, different seeds: no program-level disagreement but seed-level disagreement
    noise = core.committee_metrics({"A": [1, 0], "B": [0, 1]})
    assert noise["D_prog"] == 0.0 and noise["D_seed"] == pytest.approx(2.0 / 2) and noise["mixed_seed_frac"] == 1.0
    # a split committee on every seed: D_seed is exactly 1 bit
    assert core.committee_metrics({"A": [1, 1], "B": [0, 0]})["D_seed"] == pytest.approx(1.0)


def test_committee_metrics_rejects_bad_input():
    with pytest.raises(SystemExit):
        core.committee_metrics({"A": [1, 0], "B": [1]})
    with pytest.raises(SystemExit):
        core.committee_metrics({"A": [1, 2]})
    with pytest.raises(SystemExit):
        core.committee_metrics({})


def test_load_round_agrees_with_select_parents(tmp_path):
    spec = importlib.util.spec_from_file_location("select_parents", ROOT / "scripts" / "tc1" / "select_parents.py")
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    d = write_round(tmp_path, 0, f"{GOAL}/put_the_bowl_on_the_stove",
                    {"candidate_B": [1, 0] * 7 + [1], "candidate_A": [0] * 15, "candidate_C": [1] * 15})
    cands, seeds = sp.load_round(d)
    mine = core.load_round(d)
    assert seeds == mine["seeds"]
    assert {c["candidate"]: c["passed"] for c in cands} == mine["outcomes"]


def test_random_walk_is_deterministic_and_without_repeats(families_doc, goal):
    ids = core.task_ids(goal)
    visited = [goal["round0"]]
    walk = []
    for _ in range(len(ids) - 1):
        t = core.pick_random(FAMILY, ids, visited, "r1")
        assert t not in visited
        visited.append(t)
        walk.append(t)
    assert sorted(visited) == sorted(ids)
    visited2 = [goal["round0"]]
    for expected in walk:
        t = core.pick_random(FAMILY, ids, visited2, "r1")
        assert t == expected
        visited2.append(t)
    other = core.random_order(FAMILY, ids, "r2")
    assert other != core.random_order(FAMILY, ids, "r1")
    with pytest.raises(SystemExit):
        core.pick_random(FAMILY, ids, ids, "r1")


def test_round0_is_fixed_whatever_the_rule(tmp_path, goal):
    for rule in ("random", "knn", "prompt", "harness"):
        r = tmp_path / rule
        assert run_select(r, 0, rule) == 0
        rec = json.loads((r / "iter_00" / "task.json").read_text())
        assert rec["rule"] == "fixed" and rec["requested_rule"] == rule and rec["task_id"] == goal["round0"]
        assert rec["seeds"] == DEV


def test_select_flow_and_idempotence(run_dir, goal):
    assert run_select(run_dir, 1, "random", "--salt", "r1") == 0
    rec = json.loads((run_dir / "iter_01" / "task.json").read_text())
    assert rec["rule"] == "random" and rec["task_id"] != goal["round0"]
    assert rec["visited_before"] == [goal["round0"]] and goal["round0"] not in rec["unvisited_before"]
    before = (run_dir / "iter_01" / "task.json").read_text()
    assert run_select(run_dir, 1, "random", "--salt", "r1") == 0
    assert (run_dir / "iter_01" / "task.json").read_text() == before
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "knn")
    assert (run_dir / "iter_01" / "task.json").read_text() == before


def test_select_needs_finished_previous_rounds(tmp_path, goal):
    r = tmp_path / "run"
    run_select(r, 0, "random")                       # task.json only, iter_00 not evaluated
    with pytest.raises(SystemExit):
        run_select(r, 1, "random")


def test_full_random_run_verifies_and_never_repeats(run_dir, goal, capsys):
    for n in range(1, 5):
        assert run_select(run_dir, n, "random", "--salt", "r3") == 0
        tid = json.loads((run_dir / f"iter_{n:02d}" / "task.json").read_text())["task_id"]
        write_round(run_dir, n, tid, committee({"candidate_A": n, "candidate_B": 2 * n}))
    capsys.readouterr()
    assert select_task.main(["verify", "--family", FAMILY, "--run-dir", str(run_dir)]) == 0
    assert "OK: 5 round(s)" in capsys.readouterr().out
    tasks = [json.loads((run_dir / f"iter_{n:02d}" / "task.json").read_text())["task_id"] for n in range(5)]
    assert len(set(tasks)) == 5


def verify(run_dir, capsys):
    capsys.readouterr()
    code = select_task.main(["verify", "--family", FAMILY, "--run-dir", str(run_dir)])
    return code, capsys.readouterr().out


def test_verify_catches_evaluating_the_wrong_task(run_dir, capsys):
    run_select(run_dir, 1, "random", "--salt", "r1")
    write_round(run_dir, 1, f"{GOAL}/turn_on_the_stove" if json.loads((run_dir / "iter_01/task.json").read_text())["task"] != "turn_on_the_stove"
                else f"{GOAL}/put_the_bowl_on_the_plate", committee({"candidate_A": 1, "candidate_B": 2}))
    code, out = verify(run_dir, capsys)
    assert code == 1 and "evaluated on" in out


def test_verify_catches_wrong_seeds_missing_task_json_and_duplicates(run_dir, capsys, goal):
    run_select(run_dir, 1, "random", "--salt", "r1")
    tid = json.loads((run_dir / "iter_01/task.json").read_text())["task_id"]
    write_round(run_dir, 1, tid, {"candidate_A": [1] * 10, "candidate_B": [0] * 10}, seeds=list(range(1, 11)))
    code, out = verify(run_dir, capsys)
    assert code == 1 and "seeds" in out
    # an evaluated round the script never chose
    write_round(run_dir, 2, f"{GOAL}/turn_on_the_stove", committee({"candidate_A": 1}))
    code, out = verify(run_dir, capsys)
    assert code == 1 and "no task.json" in out
    # a task evaluated twice
    rec = json.loads((run_dir / "iter_01/task.json").read_text())
    rec["round"], rec["task_id"] = 2, goal["round0"]
    (run_dir / "iter_02").mkdir(exist_ok=True)
    (run_dir / "iter_02" / "task.json").write_text(json.dumps(rec))
    code, out = verify(run_dir, capsys)
    assert code == 1 and "already used" in out


def test_verify_catches_mixed_rules_and_salts(run_dir, capsys):
    run_select(run_dir, 1, "random", "--salt", "r1")
    tid1 = json.loads((run_dir / "iter_01/task.json").read_text())["task_id"]
    write_round(run_dir, 1, tid1, committee({"candidate_A": 1, "candidate_B": 2}))
    run_select(run_dir, 2, "knn")
    code, out = verify(run_dir, capsys)
    assert code == 1 and "different rules" in out


def test_eval_cmd_matches_chosen_task(run_dir, capsys):
    run_select(run_dir, 1, "random", "--salt", "r1")
    rec = json.loads((run_dir / "iter_01/task.json").read_text())
    capsys.readouterr()
    assert select_task.main(["eval-cmd", "--run-dir", str(run_dir), "--round", "1", "--gpu", "0"]) == 0
    cmd = capsys.readouterr().out.strip()
    assert f"--suite {rec['suite']} --task {rec['task']}" in cmd
    assert "--trials 15" in cmd and "--trial-seeds 51 52" in cmd and cmd.endswith("--no-highlights")
    assert str(run_dir / "iter_01") in cmd


def test_knn_prefers_the_neighbour_of_a_disagreeing_task():
    def task(name):
        return {"id": f"s/{name}", "suite": "s", "task": name, "language_static": name.replace("_", " ")}
    family = {"tasks": [task("open_drawer_one"), task("turn_on_stove"), task("open_drawer_two"), task("turn_on_stove_again")]}
    history = [{"task_id": "s/open_drawer_one", "metrics": {"D_prog": 0.9}},
               {"task_id": "s/turn_on_stove", "metrics": {"D_prog": 0.0}}]
    scores = core.knn_scores(family, history)
    assert set(scores) == {"s/open_drawer_two", "s/turn_on_stove_again"}
    assert scores["s/open_drawer_two"] > scores["s/turn_on_stove_again"]


def test_argmax_in_order_breaks_ties_by_family_order():
    assert core.argmax_in_order({"c": 1.0, "a": 1.0, "b": 0.5}, ["a", "b", "c"]) == "a"
    assert core.argmax_in_order({"c": 2.0, "a": 1.0}, ["a", "b", "c"]) == "c"
    with pytest.raises(SystemExit):
        core.argmax_in_order({}, ["a"])


def test_knn_rule_records_scores(run_dir):
    assert run_select(run_dir, 1, "knn") == 0
    rec = json.loads((run_dir / "iter_01/task.json").read_text())
    assert rec["rule"] == "knn" and rec["scores"] and rec["task_id"] in rec["scores"]
    assert rec["scores"][rec["task_id"]] == max(rec["scores"].values())


def write_choice(path: Path, task: str, names, p=0.5, rationale="the programs handle the drawer handle differently"):
    path.write_text(json.dumps({"task": task, "predicted_pass": {n: p for n in names}, "rationale": rationale}))


def test_prompt_rule_accepts_valid_choice_and_rejects_invalid(run_dir, goal):
    names = ["candidate_A", "candidate_B", "candidate_C"]
    for n in names:                                              # committee of round 1 = programs not yet evaluated
        (run_dir / "iter_01" / n).mkdir(parents=True, exist_ok=True)
        (run_dir / "iter_01" / n / "code.py").write_text("pass\n")
    choice = run_dir / "iter_01" / "task_choice.json"
    target = f"{GOAL}/turn_on_the_stove"
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    write_choice(choice, goal["round0"], names)
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    write_choice(choice, "libero_goal_swap/not_a_task", names)
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    write_choice(choice, target, names[:2])                      # missing a candidate's prediction
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    write_choice(choice, target, names, p=1.5)                   # not a probability
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    write_choice(choice, target, names, rationale="short")       # no real rationale
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    choice.write_text("{not json")
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    assert not (run_dir / "iter_01" / "task.json").exists()      # nothing recorded on any failure
    write_choice(choice, target, names)
    assert run_select(run_dir, 1, "prompt") == 0
    rec = json.loads((run_dir / "iter_01/task.json").read_text())
    assert rec["task_id"] == target and rec["rule"] == "prompt"
    assert rec["choice"]["predicted_pass"] == {n: 0.5 for n in names} and rec["choice_sha256"]


def test_prompt_and_harness_need_the_candidates_to_exist(run_dir):
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "prompt")
    with pytest.raises(SystemExit):
        run_select(run_dir, 1, "harness", "--proxy", str(run_dir / "proxy.py"))


def test_context_is_perception_only_and_complete(run_dir, goal, capsys):
    (run_dir / "iter_01" / "candidate_A").mkdir(parents=True)
    (run_dir / "iter_01" / "candidate_A" / "code.py").write_text("pass\n")
    capsys.readouterr()
    assert select_task.main(["context", "--family", FAMILY, "--run-dir", str(run_dir), "--round", "1"]) == 0
    ctx = json.loads(capsys.readouterr().out)
    assert ctx["visited"] == [goal["round0"]] and goal["round0"] not in ctx["unvisited"]
    assert len(ctx["tasks"]) == len(goal["tasks"]) and ctx["committee"][0]["candidate"] == "candidate_A"
    assert ctx["history"][0]["metrics"]["D_prog"] > 0
    assert set(ctx["tasks"][0]) == {"id", "suite", "task", "language", "scene_image", "language_is_runtime"}
    assert (run_dir / "iter_01" / "context.json").exists()


PROXY_FAVOURS_STOVE = (
    "def score(ctx):\n"
    "    return {t: (1.0 if t.endswith('turn_on_the_stove') else 0.0) for t in ctx['unvisited']}\n"
)


@pytest.fixture
def harness_run(run_dir):
    for n in ("candidate_A", "candidate_B"):
        (run_dir / "iter_01" / n).mkdir(parents=True, exist_ok=True)
        (run_dir / "iter_01" / n / "code.py").write_text("pass\n")
    return run_dir


def test_harness_rule_runs_locks_and_records(harness_run):
    proxy = harness_run / "proxy.py"
    proxy.write_text(PROXY_FAVOURS_STOVE)
    assert run_select(harness_run, 1, "harness", "--proxy", str(proxy)) == 0
    rec = json.loads((harness_run / "iter_01/task.json").read_text())
    assert rec["task_id"] == f"{GOAL}/turn_on_the_stove" and rec["rule"] == "harness"
    assert rec["scores"][rec["task_id"]] == 1.0 and rec["proxy_sha256"] == core.file_sha256(proxy)
    lock = json.loads((harness_run / "proxy.lock.json").read_text())
    assert lock["sha256"] == rec["proxy_sha256"] and lock["first_round"] == 1


def test_harness_proxy_is_frozen_unless_explicitly_unlocked(harness_run):
    proxy = harness_run / "proxy.py"
    proxy.write_text(PROXY_FAVOURS_STOVE)
    run_select(harness_run, 1, "harness", "--proxy", str(proxy))
    tid = json.loads((harness_run / "iter_01/task.json").read_text())["task_id"]
    write_round(harness_run, 1, tid, committee({"candidate_A": 1, "candidate_B": 4}))
    (harness_run / "iter_02" / "candidate_A").mkdir(parents=True)
    (harness_run / "iter_02" / "candidate_A" / "code.py").write_text("pass\n")
    proxy.write_text(PROXY_FAVOURS_STOVE + "# edited\n")
    with pytest.raises(SystemExit):
        run_select(harness_run, 2, "harness", "--proxy", str(proxy))
    assert not (harness_run / "iter_02" / "task.json").exists()
    assert run_select(harness_run, 2, "harness", "--proxy", str(proxy), "--unlock") == 0
    lock = json.loads((harness_run / "proxy.lock.json").read_text())
    assert [h["unlocked"] for h in lock["history"]] == [False, True]


@pytest.mark.parametrize("body, why", [
    ("def score(ctx):\n    raise RuntimeError('boom')\n", "crash"),
    ("def score(ctx):\n    return {}\n", "missing scores"),
    ("def score(ctx):\n    return {t: float('nan') for t in ctx['unvisited']}\n", "nan"),
    ("def score(ctx):\n    return {t: 'high' for t in ctx['unvisited']}\n", "non-numeric"),
    ("def score(ctx):\n    return [1, 2, 3]\n", "not a dict"),
    ("x = 1\n", "no score function"),
])
def test_bad_proxies_fail_cleanly_and_record_nothing(harness_run, body, why):
    proxy = harness_run / "proxy.py"
    proxy.write_text(body)
    with pytest.raises(SystemExit):
        run_select(harness_run, 1, "harness", "--proxy", str(proxy))
    assert not (harness_run / "iter_01" / "task.json").exists(), why


def test_proxy_environment_is_scrubbed_and_times_out(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak")
    leak = tmp_path / "leak.py"
    leak.write_text("import os\n"
                    "def score(ctx):\n"
                    "    bad = [k for k in ('ANTHROPIC_API_KEY', 'HF_TOKEN') if k in os.environ]\n"
                    "    if bad:\n        raise RuntimeError('leaked ' + ','.join(bad))\n"
                    "    return {t: 0.0 for t in ctx['unvisited']}\n")
    assert core.run_proxy(leak, {"unvisited": ["a", "b"]}, ["a", "b"]) == {"a": 0.0, "b": 0.0}
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ndef score(ctx):\n    time.sleep(30)\n    return {}\n")
    with pytest.raises(SystemExit):
        core.run_proxy(slow, {"unvisited": ["a"]}, ["a"], timeout=2)


def test_init_from_copies_round0_and_checks_provenance(tmp_path, goal):
    ref, new = tmp_path / "ref", tmp_path / "new"
    d = write_round(ref, 0, goal["round0"], committee({"candidate_A": 2, "candidate_B": 5}))
    (d / "candidate_A" / "video.mp4").write_text("x")
    (d / "candidate_A" / "trace.json").write_text("{}")
    assert select_task.main(["init-from", "--family", FAMILY, "--run-dir", str(new), "--from-run", str(ref)]) == 0
    assert (new / "iter_00" / "candidate_A" / "code.py").exists() and (new / "iter_00" / "candidate_A" / "trace.json").exists()
    assert not (new / "iter_00" / "candidate_A" / "video.mp4").exists()
    prov = json.loads((new / "init_from.json").read_text())
    assert prov["iter_summary_sha256"] == core.file_sha256(d / "iter_summary.json")
    with pytest.raises(SystemExit):
        select_task.main(["init-from", "--family", FAMILY, "--run-dir", str(new), "--from-run", str(ref)])
    wrong = tmp_path / "wrong"
    write_round(wrong, 0, f"{GOAL}/turn_on_the_stove", committee({"candidate_A": 2}))
    with pytest.raises(SystemExit):                              # reference ran another task than the family's round 0
        select_task.main(["init-from", "--family", FAMILY, "--run-dir", str(tmp_path / "x"), "--from-run", str(wrong)])
    with pytest.raises(SystemExit):                              # reference round 0 not finished
        select_task.main(["init-from", "--family", FAMILY, "--run-dir", str(tmp_path / "y"), "--from-run", str(tmp_path / "none")])


@pytest.fixture
def toy_families(tmp_path):
    tasks = [{"id": f"s/t{i}", "suite": "s", "task": f"t{i}", "language_static": f"pick item {i}"} for i in range(3)]
    path = tmp_path / "toy_families.json"
    path.write_text(json.dumps({"schema_version": 1, "dev_seeds": DEV, "heldout_seeds": [1, 2],
                                "families": {"toy": {"relation": "tight", "description": "toy", "round0": "s/t0",
                                                     "tasks": tasks}}}))
    return path


def toy_run(tmp_path, toy_families):
    r = tmp_path / "toyrun"
    write_round(r, 0, "s/t0", committee({"candidate_A": 4, "candidate_B": 8, "candidate_C": 0, "candidate_D": 2}))
    (r / "iter_00" / "selection.json").write_text(json.dumps({
        "rule": "top3", "parents": ["iter_00/candidate_B", "iter_00/candidate_A", "iter_00/candidate_D"]}))
    return r


def test_final_plan_and_pick(tmp_path, toy_families, capsys):
    r = toy_run(tmp_path, toy_families)
    args = ["--family", "toy", "--run-dir", str(r), "--families", str(toy_families)]
    capsys.readouterr()
    assert select_task.main(["final-plan", *args, "--seeds", "51", "52", "53"]) == 0
    cmds = capsys.readouterr().out.strip().splitlines()
    assert len(cmds) == 3 and all("--trial-seeds 51 52 53" in c and "--trials 3" in c for c in cmds)
    plan = json.loads((r / "final" / "plan.json").read_text())
    assert sorted(plan["map"].values()) == ["iter_00/candidate_A", "iter_00/candidate_B", "iter_00/candidate_D"]
    for k in range(3):
        for name in plan["map"]:
            assert (r / "final" / f"task_{k:02d}" / name / "code.py").exists()
    with pytest.raises(SystemExit):
        select_task.main(["final-plan", *args, "--seeds", "51", "52"])
    with pytest.raises(SystemExit):
        select_task.main(["final-pick", "--run-dir", str(r)])
    # A passes t0 and t1 only, B passes t2 only, D passes one seed of t0
    inv = {src: name for name, src in plan["map"].items()}
    seeds3 = [51, 52, 53]
    table = {"s/t0": {"iter_00/candidate_A": [1, 1, 1], "iter_00/candidate_B": [0, 0, 0], "iter_00/candidate_D": [1, 0, 0]},
             "s/t1": {"iter_00/candidate_A": [1, 1, 1], "iter_00/candidate_B": [0, 0, 0], "iter_00/candidate_D": [0, 0, 0]},
             "s/t2": {"iter_00/candidate_A": [0, 0, 0], "iter_00/candidate_B": [1, 1, 1], "iter_00/candidate_D": [0, 0, 0]}}
    for t in plan["tasks"]:
        write_summary(Path(t["dir"]), t["id"], {inv[src]: row for src, row in table[t["id"]].items()},
                      seeds=seeds3, code=False)
    capsys.readouterr()
    assert select_task.main(["final-pick", "--run-dir", str(r)]) == 0
    best = json.loads((r / "final" / "best.json").read_text())
    assert best["best"] == "iter_00/candidate_A" and best["macro_pass"] == pytest.approx(2 / 3, abs=1e-3)
    assert [x["candidate"] for x in best["ranking"]] == ["iter_00/candidate_A", "iter_00/candidate_B", "iter_00/candidate_D"]
    assert best["code_path"] == str(r / "iter_00" / "candidate_A" / "code.py")
    matrix = json.loads((r / "final" / "matrix.json").read_text())
    assert matrix["family"] == "toy" and len(matrix["tasks"]) == 3 and set(matrix["outcomes"]) == {"s/t0", "s/t1", "s/t2"}
    rep = backtest.backtest(matrix, ["knn", "oracle"], budget=3, draws=20)
    assert rep["rules"]["oracle"]["cumulative_D_prog"] >= rep["rules"]["knn"]["cumulative_D_prog"]


def test_final_plan_needs_parent_selection(tmp_path, toy_families):
    r = tmp_path / "r"
    write_round(r, 0, "s/t0", committee({"candidate_A": 4}))
    with pytest.raises(SystemExit):
        select_task.main(["final-plan", "--family", "toy", "--run-dir", str(r), "--families", str(toy_families)])


def test_heldout_cmds_and_summary(tmp_path, toy_families, capsys):
    r = tmp_path / "r"
    (r / "final").mkdir(parents=True)
    (r / "final" / "best.json").write_text(json.dumps({"family": "toy", "code_path": str(r / "iter_00/candidate_A/code.py")}))
    args = ["--family", "toy", "--run-dir", str(r), "--families", str(toy_families)]
    capsys.readouterr()
    assert select_task.main(["heldout-cmds", *args, "--gpu", "0"]) == 0
    cmds = capsys.readouterr().out.strip().splitlines()
    assert len(cmds) == 3 and all("--seeds 1 2" in c and "--resume" in c and str(r / "heldout") in c for c in cmds)
    with pytest.raises(SystemExit):
        select_task.main(["heldout-summary", *args])
    for i, passes in enumerate([2, 1, 0]):
        m = r / "heldout" / "s" / f"t{i}" / "runs" / "abc" / "manifest.json"
        m.parent.mkdir(parents=True)
        m.write_text(json.dumps({"identity": {"seeds": [1, 2]}, "status": "complete", "passes": passes, "trials": 2,
                                 "pass_rate": passes / 2}))
    capsys.readouterr()
    assert select_task.main(["heldout-summary", *args]) == 0
    assert json.loads((r / "heldout.json").read_text())["macro_pass"] == pytest.approx(0.5)


def test_backtest_synthetic_selection_beats_random_and_oracle_bounds_it():
    matrix = backtest.make_synthetic(seed=0)
    rep = backtest.backtest(matrix, ["knn", "oracle"], budget=5, draws=300)
    knn, oracle = rep["rules"]["knn"], rep["rules"]["oracle"]
    assert knn["cumulative_D_prog"] > rep["random"]["mean"]
    assert oracle["cumulative_D_prog"] >= knn["cumulative_D_prog"]
    assert oracle["fraction_of_random_beaten"] >= 0.95
    assert len(knn["picks"]) == 4 and rep["round0"] not in knn["picks"]
    assert backtest.backtest(matrix, ["knn", "oracle"], budget=5, draws=300) == rep


def test_backtest_harness_proxy_and_oracle_proxy(tmp_path):
    matrix = backtest.make_synthetic(seed=0)
    good = tmp_path / "good.py"                      # a proxy that reads the language: drawer tasks split the committee
    good.write_text("def score(ctx):\n    return {t['id']: float('drawer' in t['language']) for t in ctx['tasks'] if t['id'] in ctx['unvisited']}\n")
    rep = backtest.backtest(matrix, ["harness", "oracle"], budget=4, draws=200, proxy=good)
    assert rep["rules"]["harness"]["cumulative_D_prog"] > rep["random"]["mean"]
    assert all("drawer" in p for p in rep["rules"]["harness"]["picks"])


def test_backtest_rejects_bad_budget_and_unknown_round0():
    matrix = backtest.make_synthetic(seed=0)
    with pytest.raises(SystemExit):
        backtest.backtest(matrix, ["knn"], budget=99, draws=5)
    with pytest.raises(SystemExit):
        backtest.backtest(matrix, ["knn"], budget=3, draws=5, round0="nope/none")


@pytest.mark.parametrize("arm, needle", [("random", "--rule random --salt r2"), ("knn", "--rule knn"),
                                         ("prompt", "task_choice.json"), ("harness", "proxy.py")])
def test_render_prompt_fills_every_placeholder(arm, needle, capsys):
    capsys.readouterr()
    assert select_task.main(["render-prompt", "--family", "drawer_ladder", "--arm", arm, "--rep", "2"]) == 0
    text = capsys.readouterr().out
    assert "{{" not in text and needle in text
    assert "outputs/family_%s_r2/drawer_ladder" % arm in text
    fam = core.get_family(core.load_families(), "drawer_ladder")
    assert fam["round0"] in text and str(len(fam["tasks"])) in text and "51 52 53" in text
    assert "none (this run writes iter_00" in text
    capsys.readouterr()
    select_task.main(["render-prompt", "--family", "drawer_ladder", "--arm", arm, "--rep", "2", "--ref-run", "outputs/ref/x"])
    assert "outputs/ref/x" in capsys.readouterr().out


@pytest.mark.parametrize("arm", ["random", "knn", "prompt", "harness"])
def test_every_select_task_command_in_the_prompt_parses(arm, capsys):
    """Each `select_task.py <cmd> ...` line in a rendered prompt is accepted by the real argument parser."""
    import shlex
    capsys.readouterr()
    select_task.main(["render-prompt", "--family", "goal_swap", "--arm", arm, "--rep", "1"])
    text = capsys.readouterr().out
    parser, checked = select_task.build_parser(), 0
    for line in text.splitlines():
        if "select_task.py" not in line or "..." in line:
            continue
        cmd = line.split("select_task.py", 1)[1]
        for cut in (" > ", " (", "`", " then", " #"):
            cmd = cmd.split(cut)[0]
        cmd = cmd.replace("$RUN_DIR", "/tmp/x").replace("<reference>", "/tmp/ref").replace("--round N", "--round 1")
        tokens = shlex.split(cmd)
        if not tokens or tokens[0] not in select_task.COMMANDS:
            continue
        parser.parse_args(tokens)
        checked += 1
    assert checked >= 8, f"only {checked} commands checked"
    assert set(select_task.COMMANDS) == set(select_task.build_parser()._subparsers._group_actions[0].choices)


def test_summarize_family_run(run_dir, capsys):
    run_select(run_dir, 1, "random", "--salt", "r1")
    tid = json.loads((run_dir / "iter_01/task.json").read_text())["task_id"]
    write_round(run_dir, 1, tid, committee({"candidate_A": 6, "candidate_B": 6}))
    spec = importlib.util.spec_from_file_location("summarize_family_run", ROOT / "scripts/taskselect/summarize_family_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    capsys.readouterr()
    assert mod.main([str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "random" in out and "fixed" in out and tid.split("/")[-1][:20] in out


def test_integration_five_rounds_with_real_select_parents(tmp_path, goal, capsys):
    """Drive a whole run through the real scripts/tc1/select_parents.py: its selection.json feeds final-plan,
    its verify accepts our parents.json, and every round's task is chosen by select_task.py."""
    import subprocess
    r = tmp_path / "run"
    sp = [sys.executable, str(ROOT / "scripts/tc1/select_parents.py")]
    pool = ["candidate_%s" % c for c in "ABCDEFGH"]
    write_round(r, 0, goal["round0"], {n: [1] * (3 * i % 9) + [0] * (15 - 3 * i % 9) for i, n in enumerate(pool)})
    assert run_select(r, 0, "random") == 0
    for n in range(1, 5):
        done = subprocess.run(sp + ["select", "--rule", "top3", "--iter-dir", str(r / f"iter_{n - 1:02d}")],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        parents = json.loads((r / f"iter_{n - 1:02d}" / "selection.json").read_text())["parents"]
        assert len(parents) == 3
        d = r / f"iter_{n:02d}"
        for name in pool:
            (d / name).mkdir(parents=True, exist_ok=True)
            (d / name / "code.py").write_text("pass\n")
        (d / "parents.json").write_text(json.dumps({"selection": "top3", "candidates": {
            name: {"parents": parents[: 1 + i % 3], "parent_pass_rates": []} for i, name in enumerate(pool)}}))
        done = subprocess.run(sp + ["verify", "--iter-dir", str(d)], capture_output=True, text=True)
        assert done.returncode == 0 and "OK" in done.stdout, done.stdout + done.stderr
        assert run_select(r, n, "knn") == 0
        tid = json.loads((d / "task.json").read_text())["task_id"]
        write_round(r, n, tid, {name: [1] * ((i * n) % 8) + [0] * (15 - (i * n) % 8) for i, name in enumerate(pool)})
    capsys.readouterr()
    assert select_task.main(["verify", "--family", FAMILY, "--run-dir", str(r)]) == 0
    assert subprocess.run(sp + ["select", "--rule", "top3", "--iter-dir", str(r / "iter_04")]).returncode == 0
    capsys.readouterr()
    assert select_task.main(["final-plan", "--family", FAMILY, "--run-dir", str(r)]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == len(goal["tasks"])
    plan = json.loads((r / "final" / "plan.json").read_text())
    assert all(p.startswith("iter_04/") for p in plan["parents"]) and plan["seeds"] == DEV[:3]
    assert len({json.loads((r / f"iter_{n:02d}" / "task.json").read_text())["task_id"] for n in range(5)}) == 5
