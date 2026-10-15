#!/usr/bin/env python3
"""Generate families.json from the LIBERO task lists (CPU only; run with .venv-libero).

The round-0 task of a family is fixed by hashing its name, so every arm starts on the same task.
`language_static` is the task-list text; describe_tasks.py records the runtime instruction.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DEV_SEEDS = list(range(51, 66))
HELDOUT_SEEDS = list(range(1, 11))
SCORE_SEEDS = [71, 72, 73]      # family score after each round; never shown to the candidate writer

BOWL = "put_the_bowl_on_the_stove"
# The with_* suites rename the task with their scene prefix (and some move it to another scene).
PERTURBED = [
    ("libero_goal", BOWL), ("libero_goal_swap", BOWL), ("libero_goal_lan", BOWL), ("libero_goal_object", BOWL),
    ("libero_goal_with_mug", "MAIN_TABLE_SCENE1_" + BOWL),
    ("libero_goal_with_yellow_book", "MAIN_TABLE_SCENE1_" + BOWL),
    ("libero_goal_with_milk", "FLOOR_SCENE1_" + BOWL),
    ("libero_goal_with_red_box", "FLOOR_SCENE1_" + BOWL),
    ("libero_goal_with_green_mug", "FLOOR_SCENE1_" + BOWL),
    ("libero_goal_with_alphabet_soup", "FLOOR_SCENE1_" + BOWL),
]

LIBRARY_9 = [
    ("libero_goal", "open_the_middle_drawer_of_the_cabinet"),
    ("libero_goal", "put_the_bowl_on_the_stove"),
    ("libero_goal", "turn_on_the_stove"),
    ("libero_spatial", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"),
    ("libero_spatial", "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"),
    ("libero_spatial", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"),
    ("libero_object", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
    ("libero_object", "pick_up_the_cream_cheese_and_place_it_in_the_basket"),
    ("libero_object", "pick_up_the_salad_dressing_and_place_it_in_the_basket"),
]

FAMILIES: dict[str, dict] = {
    "goal_swap": {
        "relation": "medium",
        "description": "One kitchen scene (bowl, plate, stove, cabinet, wine bottle, rack, cream cheese); "
                       "ten different goals needing different skills (pick-place, push, drawer, stove knob).",
        "suites": {"libero_goal_swap": "ALL"},
    },
    "object_swap": {
        "relation": "strong",
        "description": "Identical skill (pick an item, place it in the basket); only the object changes, "
                       "so disagreement should come from grasp geometry.",
        "suites": {"libero_object_swap": "ALL"},
    },
    "spatial_swap": {
        "relation": "strong",
        "description": "Same object and goal (black bowl onto the plate); only the spatial referring "
                       "expression changes, so disagreement should come from relational perception.",
        "suites": {"libero_spatial_swap": "ALL"},
    },
    "bowl_stove_perturbed": {
        "relation": "tight",
        "description": "One instruction (put the bowl on the stove) under ten LIBERO / LIBERO-PRO variants "
                       "(object positions, language, object, extra objects, other scenes); disagreement is nuisance-driven.",
        "pairs": PERTURBED,
    },
    "drawer_ladder": {
        "relation": "medium",
        "description": "Drawer skills of increasing composition length across two kitchen scenes "
                       "(open / close / put-in / put-in-and-close / open-and-put-in).",
        "pairs": [("libero_90", t) for t in [
            "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
            "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
            "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
            "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
            "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
        ]],
    },
    "library_9": {
        "relation": "weak",
        "description": "Library-build experiment: three short tasks from each of the goal, spatial and object suites.",
        "pairs": LIBRARY_9,
    },
    "library_9_swap": {
        "relation": "weak",
        "description": "library_9 on the swap suites (object positions randomised per seed), as in the tc1 task_selection runs.",
        "pairs": [(suite + "_swap", task) for suite, task in LIBRARY_9],
    },
    "three_suites": {
        "relation": "weak",
        "description": "Four short tasks from each of the goal, object and spatial swap suites "
                       "(three different scenes, shared pick-place / push / drawer / knob skills).",
        "pairs": [
            ("libero_goal_swap", "put_the_bowl_on_the_stove"),
            ("libero_goal_swap", "open_the_middle_drawer_of_the_cabinet"),
            ("libero_goal_swap", "turn_on_the_stove"),
            ("libero_goal_swap", "push_the_plate_to_the_front_of_the_stove"),
            ("libero_object_swap", "pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
            ("libero_object_swap", "pick_up_the_milk_and_place_it_in_the_basket"),
            ("libero_object_swap", "pick_up_the_ketchup_and_place_it_in_the_basket"),
            ("libero_object_swap", "pick_up_the_butter_and_place_it_in_the_basket"),
            ("libero_spatial_swap", "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"),
            ("libero_spatial_swap", "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate"),
            ("libero_spatial_swap", "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate"),
            ("libero_spatial_swap", "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"),
        ],
    },
    "mixed_short_skills": {
        "relation": "weak",
        "description": "Twelve short LIBERO-90 tasks from different scenes and skill types "
                       "(drawer, microwave door, stove knob, pick-place, stacking); the weakest relation.",
        "pairs": [("libero_90", t) for t in [
            "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
            "KITCHEN_SCENE3_turn_on_the_stove",
            "KITCHEN_SCENE6_close_the_microwave",
            "KITCHEN_SCENE7_open_the_microwave",
            "KITCHEN_SCENE8_turn_off_the_stove",
            "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
            "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
            "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
            "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
            "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
            "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
            "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
        ]],
    },
    "libero10_long": {
        "relation": "weak",
        "description": "The ten long-horizon LIBERO-10 tasks (different scenes, two-step goals). Expected to "
                       "be very hard for round-0 programs, so run the viability probe before using it.",
        "suites": {"libero_10": "ALL"},
    },
}


def main() -> int:
    from libero import benchmark as lb
    bench = lb.get_benchmark_dict()
    suites: dict[str, dict[str, str]] = {}

    def names(suite: str) -> dict[str, str]:
        if suite not in suites:
            if suite not in bench:
                raise SystemExit(f"unknown suite {suite!r}")
            ts = bench[suite]()
            suites[suite] = {ts.get_task(i).name: ts.get_task(i).language for i in range(ts.n_tasks)}
        return suites[suite]

    out = {"schema_version": 1, "dev_seeds": DEV_SEEDS, "heldout_seeds": HELDOUT_SEEDS,
           "score_seeds": SCORE_SEEDS, "families": {}}
    for name, spec in FAMILIES.items():
        pairs = list(spec.get("pairs", []))
        for suite, sel in spec.get("suites", {}).items():
            pairs += [(suite, t) for t in names(suite)] if sel == "ALL" else [(suite, t) for t in sel]
        tasks = []
        for suite, task in pairs:
            if task not in names(suite):
                raise SystemExit(f"{name}: task {task!r} not in suite {suite!r}")
            tasks.append({"id": f"{suite}/{task}", "suite": suite, "task": task,
                          "language_static": names(suite)[task]})
        ids = [t["id"] for t in tasks]
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{name}: duplicate members")
        round0 = ids[int(hashlib.sha256(name.encode()).hexdigest(), 16) % len(ids)]
        out["families"][name] = {"relation": spec["relation"], "description": spec["description"],
                                 "round0": round0, "tasks": tasks}
        print(f"{name:22s} {spec['relation']:7s} {len(tasks):2d} tasks  round0={round0}")
    path = Path(__file__).with_name("families.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
