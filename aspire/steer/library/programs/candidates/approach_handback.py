"""Approach-and-hand-back for swapped layouts where the PICK object may have moved too (goal_swap).
Both phases: the program moves the gripper above where the image says the trained-layout content moved (pick object, then
destination), and hands control back to the VLA with its proprio state shifted by that displacement, so the VLA believes it is
at its memorised point while its cameras show the object / destination below (E27: phase and release are read from the
images; E29: images shape the local motion). Displacements < 3 cm are treated as 'not moved' (the VLA runs unchanged).
Perception: SigLIP feature-template tracking of the reference regions (steerlib.scene), agentview camera model."""
from pathlib import Path

import numpy as np

from steerlib import gates, scene

DESCRIPTION = "approach + hand-back for pick and place with image-tracked displacements (virtual-frame state)"
N_ACTION_STEPS = 10
REF = Path(__file__).resolve().parents[2] / "references"
STEP = 0.0095


def _twin(suite, task_id):
    for suffix in ("_swap", "_lan", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def _track(F_ref, F_live, point):
    u = scene.project(point); m = scene.region(F_ref, u, tau=0.3, radius=4)
    duv, s, s0 = scene.track(F_ref, m, F_live)
    d = scene.backproject(u + duv, point[2]) - scene.backproject(u, point[2]); d[2] = 0.0
    return d if np.linalg.norm(d[:2]) > 0.03 else np.zeros(3)


def setup(ctx):
    base = _twin(ctx.task.suite, ctx.task.task_id)
    r = np.load(REF / f"{base[0]}__{base[1]}.npz", allow_pickle=False)
    ctx.memory["ref"] = {k: r[k] for k in ("image", "pick", "place")}


def program(t, obs, internals, ctx):
    mem = ctx.memory
    if "d_pick" not in mem:
        ref = mem["ref"]
        F_ref, F_live = scene.patch_features(ctx.vla, ref["image"]), scene.patch_features(ctx.vla, obs["agentview_image"])
        mem["d_pick"], mem["d_place"] = _track(F_ref, F_live, ref["pick"]), _track(F_ref, F_live, ref["place"])
        mem["mode"] = "approach" if np.any(mem["d_pick"]) else "vla_pick"
        ctx.log("deltas_cm", {"pick": np.round(mem["d_pick"][:2] * 100, 1).tolist(), "place": np.round(mem["d_place"][:2] * 100, 1).tolist()})
    if mem["mode"] in ("vla_pick", "hb_pick"):
        obs["robot0_eef_pos"] = np.asarray(obs["robot0_eef_pos"]) - mem["d_pick"]
    elif mem["mode"] in ("vla_place", "hb_place"):
        obs["robot0_eef_pos"] = np.asarray(obs["robot0_eef_pos"]) - mem["d_place"]
    return []


def _goto(eef, tgt, grip):
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip((tgt - eef) / (STEP * 3), -1, 1); a[6] = grip
    return a


def on_step(t, obs, action, ctx):
    mem, ref = ctx.memory, ctx.memory["ref"]
    eef = np.asarray(obs["robot0_eef_pos"])
    if "mode" not in mem:
        return None
    holding = gates.gripper_closed(obs, 0.03) and float(obs["robot0_gripper_qpos"][0]) > 0.0005
    if mem["mode"] == "approach":                         # gripper open, above the moved pick object
        tgt = ref["pick"] + mem["d_pick"] + np.array([0, 0, 0.08])
        if np.linalg.norm((tgt - eef)) < 0.015:
            mem["mode"] = "hb_pick"; ctx.log("hb_pick", t); ctx.policy.reset(); return None
        return _goto(eef, tgt, -1.0)
    if mem["mode"] in ("vla_pick", "hb_pick"):
        mem["hold"] = mem.get("hold", 0) + 1 if holding else 0
        if mem["hold"] >= 15 and eef[2] > ref["pick"][2] + 0.04:
            mem["mode"] = "carry" if np.any(mem["d_place"]) else "vla_place"
            mem["z_carry"] = max(eef[2], ref["place"][2] + 0.06); ctx.log("grasped", t); ctx.policy.reset()
        return None
    if mem["mode"] == "carry":
        g = ref["place"] + mem["d_place"]
        if np.linalg.norm((np.r_[g[:2], mem["z_carry"]] - eef)[:2]) < 0.012:
            mem["mode"] = "lower"
        return _goto(eef, np.r_[g[:2], mem["z_carry"]], 1.0)
    if mem["mode"] == "lower":
        g = ref["place"] + mem["d_place"]; mem["n_lower"] = mem.get("n_lower", 0) + 1
        if eef[2] < g[2] + 0.03 or mem["n_lower"] > 40:
            mem["mode"] = "hb_place"; ctx.log("hb_place", t); ctx.policy.reset(); return None
        return _goto(eef, np.r_[g[:2], g[2] + 0.02], 1.0)
    return None
