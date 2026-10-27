"""Hybrid for moved destinations: the frozen VLA picks; once the object is held and lifted, the program carries it to the
destination found in the image (where the trained-layout destination content moved), lowers it and releases.

Why: in swapped layouts the policy's targets are memorised (E05, E11: the target is decoded from the instruction tokens,
barely from the image) and steering the state or the probe direction does not move them (E10, E12). In spatial_swap the
picked bowl stays in place and only the plate moves, so the VLA grasp is right and only the destination must be supplied.
- reference: library/references/<trained suite>__<task>.npz (frozen policy's own successful trained-suite episodes);
- perception: the policy's SigLIP patch features, feature-template tracking of the region around the projected reference
  place point (steerlib.scene.region/track), back-projected with the agentview camera model at the reference place height;
- control: proportional steps toward (x, y) at the current height, then descend to the reference place height, open.
"""
from pathlib import Path

import numpy as np

from steerlib import gates, scene

DESCRIPTION = "VLA picks; program carries the object to the image-detected destination and releases"
REF = Path(__file__).resolve().parents[2] / "references"
STEP = 0.0095            # m per action unit per step (measured)


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
    if "goal" not in ctx.memory:
        ref = ctx.memory["ref"]
        F_ref = scene.patch_features(ctx.vla, ref["image"])
        F_live = scene.patch_features(ctx.vla, obs["agentview_image"])
        u_ref = scene.project(ref["place"])
        m = scene.region(F_ref, u_ref, tau=0.3, radius=4)          # E14: best on analysis ground truth
        duv, s, s0 = scene.track(F_ref, m, F_live)
        d = scene.backproject(u_ref + duv, ref["place"][2]) - scene.backproject(u_ref, ref["place"][2])
        goal = ref["place"].copy(); goal[:2] += d[:2]
        ctx.memory.update(goal=goal, mode="vla")
        ctx.log("place", {"delta_cm": np.round(d[:2] * 100, 1).tolist(), "score": round(s, 3), "score0": round(float(s0 or 0), 3),
                          "region": int(m.sum()), "goal": np.round(goal, 3).tolist()})
    return []


def on_step(t, obs, action, ctx):
    mem = ctx.memory
    eef = np.asarray(obs["robot0_eef_pos"])
    holding = gates.gripper_closed(obs, 0.03) and float(obs["robot0_gripper_qpos"][0]) > 0.0005   # closed on an object
    if mem.get("mode") == "vla":
        if holding:
            mem["hold_steps"] = mem.get("hold_steps", 0) + 1
        else:
            mem["hold_steps"] = 0
        if mem["hold_steps"] >= 15 and eef[2] > mem["ref"]["pick"][2] + 0.04:      # grasped and lifted: take over
            mem["mode"] = "carry"; mem["z_carry"] = max(eef[2], mem["goal"][2] + 0.06)
            ctx.log("takeover", {"t": t, "eef": np.round(eef, 3).tolist()})
        return None
    goal = mem["goal"]
    a = np.zeros(7, dtype=np.float32)
    for k in ("carry", "lower", "release"):
        mem.setdefault("n_" + k, 0)
    mem["n_" + mem["mode"]] += 1
    if mem["mode"] == "carry":
        tgt = np.array([goal[0], goal[1], mem["z_carry"]])
        if np.linalg.norm((tgt - eef)[:2]) < 0.008 or mem["n_carry"] > 120:
            mem["mode"] = "lower"
    if mem["mode"] == "lower":
        tgt = np.array([goal[0], goal[1], goal[2] - 0.015])            # press slightly below the reference height
        if (eef[2] < goal[2] + 0.004 and np.linalg.norm((goal - eef)[:2]) < 0.008) or mem["n_lower"] > 60:
            mem["mode"] = "release"
    if mem["mode"] == "release":
        a[6] = -1.0
        if mem["n_release"] > 15:                                     # opened: lift away so the fingers do not drag the object
            a[2] = 0.5
        return a
    a[:3] = np.clip((tgt - eef) / (STEP * 3), -1, 1)
    a[6] = 1.0
    return a
