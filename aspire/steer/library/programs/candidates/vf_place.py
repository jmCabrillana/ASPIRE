"""Place-only virtual frame (spatial_swap: the bowl stays, the plate moves). Before the grasp nothing changes; while holding,
the policy sees eef - delta_place, delta_place = image-tracked displacement of the trained destination (as carry_to_target:
SigLIP feature-template tracking, camera model). virtual_frame.py failed on goal_swap with 38 cm shifts (out-of-distribution
states); here shifts are 17-27 cm and only during transport."""
from pathlib import Path

import numpy as np

from steerlib import scene

DESCRIPTION = "virtual frame during transport only, destination displacement tracked from the image"
N_ACTION_STEPS = 10
REF = Path(__file__).resolve().parents[2] / "references"


def _twin(suite, task_id):
    for suffix in ("_swap", "_lan", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    base = _twin(ctx.task.suite, ctx.task.task_id)
    r = np.load(REF / f"{base[0]}__{base[1]}.npz", allow_pickle=False)
    ctx.memory["ref"] = {k: r[k] for k in ("image", "pick", "place")}


def program(t, obs, internals, ctx):
    mem = ctx.memory
    if "delta" not in mem:
        ref = mem["ref"]
        F_ref, F_live = scene.patch_features(ctx.vla, ref["image"]), scene.patch_features(ctx.vla, obs["agentview_image"])
        u = scene.project(ref["place"]); m = scene.region(F_ref, u, tau=0.3, radius=4)
        duv, s, s0 = scene.track(F_ref, m, F_live)
        d = scene.backproject(u + duv, ref["place"][2]) - scene.backproject(u, ref["place"][2]); d[2] = 0
        mem["delta"] = d
        ctx.log("delta_place_cm", np.round(d[:2] * 100, 1).tolist())
    if float(obs["robot0_gripper_qpos"][0]) < 0.03 and obs["robot0_eef_pos"][2] > mem["ref"]["pick"][2] + 0.02:
        mem["holding"] = True
    if mem.get("holding"):
        obs["robot0_eef_pos"] = np.asarray(obs["robot0_eef_pos"]) - mem["delta"]
    return []
