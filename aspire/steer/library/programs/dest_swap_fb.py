"""INTERNAL hand-back for swapped layouts: the VLA does everything itself (pick, carry, align, release); the program only
measures where the destination moved (image tracking, as carry_handback2) and, while the robot holds the object, adds a
destination vector inside the action expert (E37: learned by gradient through the frozen VLA) with a dose proportional to
the measured displacement:  v = (dx / Dx) * v_x + (dy / Dy) * v_y   (each v learned for a shift D of 8 cm, gain ~0.8).
State-dependent control: WHEN = holding phase (gripper closed on an object, lifted); HOW STRONGLY = the tracked offset;
FEEDBACK (FB > 0): near placing height, every chunk the lever input is corrected by FB x (goal - hand position): an integral
controller acting on the VLA's internal destination (the program never commands a motion itself).
RELEASE (v3): when the hand has stayed within REL_RADIUS of the goal at placing height for two chunks, the gripper-switch neuron
(E28: expert L4 #1855) is pushed with REL_DOSE (-2 flips the VLA's own gripper command from close to open on holding obs).
Env: GAIN (assumed vector gain, default 0.8), GATE (hold | always) for the gating ablation, IFACE (dest_pinv_L0: closed-form
interface, E40), FB."""
import os
from pathlib import Path

import numpy as np
import torch

from steerlib import scene

DESCRIPTION = "VLA alone + closed-form destination interface (E40) at expert L0, dosed by the tracked offset, gated on holding, integral feedback on the lever input"
N_ACTION_STEPS = 10
LIB = Path(__file__).resolve().parents[2]
GAIN = float(os.environ.get("GAIN", "1.0"))
GATE = os.environ.get("GATE", "hold")
VX = os.environ.get("VX", "dest_expert0_x8")
VY = os.environ.get("VY", "dest_expert0_y8")
FB = float(os.environ.get("FB", "0.5"))       # integral feedback gain on the lever input (0 = open loop)
REL_DOSE = float(os.environ.get("REL_DOSE", "-2"))     # gripper-switch neuron (E28, expert L4 #1855) push for the release
REL_RADIUS = float(os.environ.get("REL_RADIUS", "0.015"))
ACT_PER_DOSE = 81.04                                    # activation per dose unit (Neurons.dose_to_activation, holding obs)
FB_RADIUS = float(os.environ.get("FB_RADIUS", "0.10"))
IFACE = os.environ.get("IFACE", "dest_pinv_L0")          # e.g. dest_iface_L0 (E39): v = M @ (dx, dy), replaces the per-axis vectors


def _twin(suite, task_id):
    for suffix in ("_swap", "_lan"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    base = _twin(ctx.task.suite, ctx.task.task_id)
    r = np.load(LIB / "references" / f"{base[0]}__{base[1]}.npz", allow_pickle=False)
    ctx.memory["ref"] = {k: r[k] for k in ("image", "pick", "place")}
    vy = torch.load(LIB / "directions" / f"{VY}.pt") if not IFACE else None
    fx = LIB / "directions" / f"{VX}.pt"
    if not IFACE:
        vx = torch.load(fx) if fx.exists() else {"v": torch.zeros_like(vy["v"]), "D": [0.08, 0, 0], "layer": vy["layer"]}
        assert vx["layer"] == vy["layer"]
    st = ctx.memory
    if IFACE:
        f = torch.load(LIB / "directions" / f"{IFACE}.pt")
        st["M"] = (f["M"] if "M" in f else f["P"]).float()
        vx = vy = {"v": torch.zeros(st["M"].shape[0]), "D": [0.08, 0.08, 0], "layer": f["layer"]}
    st.update(vec=None, hold=0, z0=None, released=None,
              ux=(vx["v"].float(), float(vx["D"][0])), uy=(vy["v"].float(), float(vy["D"][1])))

    def hook(m, inp, out):
        if st["vec"] is None:
            return None
        return out + st["vec"].to(out.device, out.dtype)
    ctx.hook(ctx.expert_layers[vx["layer"]].mlp, hook)
    st.update(release=False, settled=0)

    def release(mod, inp):                       # state-gated internal release: the VLA's own gripper switch neuron
        if not st["release"] or REL_DOSE == 0:
            return None
        x = inp[0].clone()
        x[..., 1855] += REL_DOSE * ACT_PER_DOSE
        return (x,)
    ctx.hook(ctx.expert_layers[4].mlp.down_proj, release, pre=True)


def _vec(st, d):
    if "M" in st:
        return st["M"] @ torch.tensor([d[0] / GAIN, d[1] / GAIN], dtype=torch.float32)
    (vx, Dx), (vy, Dy) = st["ux"], st["uy"]
    return (d[0] / (Dx * GAIN)) * vx + (d[1] / (Dy * GAIN)) * vy


def program(t, obs, internals, ctx):
    st = ctx.memory
    if "delta" not in st:
        ref = st["ref"]
        F_ref = scene.patch_features(ctx.vla, ref["image"])
        F_live = scene.patch_features(ctx.vla, obs["agentview_image"])
        u_ref = scene.project(ref["place"])
        m = scene.region(F_ref, u_ref, tau=0.3, radius=4)
        duv, s, s0 = scene.track(F_ref, m, F_live)
        d = scene.backproject(u_ref + duv, ref["place"][2]) - scene.backproject(u_ref, ref["place"][2])
        st["delta"] = np.r_[d[:2], 0.0]
        st["dose_delta"] = st["delta"].copy()
        ctx.log("place", {"delta_cm": np.round(d[:2] * 100, 1).tolist(), "score": round(s, 3)})
        if GATE == "always":
            st["vec"] = _vec(st, st["dose_delta"])
    if GATE == "hold":
        on = st["hold"] >= 5 and st["z0"] is not None and obs["robot0_eef_pos"][2] > st["z0"] + 0.03 and st["released"] is None
        if on and FB > 0:                    # closed-loop INTERNAL controller: correct the VLA's destination, never its motion
            eef = np.asarray(obs["robot0_eef_pos"])
            goal = st["ref"]["place"][:2] + st["delta"][:2]
            near = np.linalg.norm(goal - eef[:2]) < FB_RADIUS
            if eef[2] < st["ref"]["place"][2] + 0.06 and near:   # local correction only (v1 also fired at the grasp: wrong way)
                err = goal - eef[:2]
                st["dose_delta"][:2] = st["dose_delta"][:2] + FB * err
                st["dose_delta"][:2] = np.clip(st["dose_delta"][:2], st["delta"][:2] - 0.2, st["delta"][:2] + 0.2)
                ctx.log("fb", {"t": t, "err_cm": np.round(err * 100, 1).tolist(),
                               "cmd_cm": np.round(st["dose_delta"][:2] * 100, 1).tolist()})
        if on:
            eef = np.asarray(obs["robot0_eef_pos"]); goal = st["ref"]["place"][:2] + st["delta"][:2]
            ok = np.linalg.norm(goal - eef[:2]) < REL_RADIUS and eef[2] < st["ref"]["place"][2] + 0.04
            st["settled"] = st["settled"] + 1 if ok else 0
            if st["settled"] >= 2 and not st["release"]:
                st["release"] = True
                ctx.log("release_lever", {"t": t, "eef": np.round(eef, 3).tolist()})
        st["vec"] = _vec(st, st["dose_delta"]) if on else None
        if on and not st.get("logged_on"):
            st["logged_on"] = True
            ctx.log("lever_on", {"t": t, "dose_delta_cm": np.round(st["dose_delta"][:2] * 100, 1).tolist()})
    return []


def on_step(t, obs, action, ctx):
    st = ctx.memory
    q = float(obs["robot0_gripper_qpos"][0])
    eef = np.asarray(obs["robot0_eef_pos"])
    if 0.0005 < q < 0.03:
        st["hold"] += 1
        if st["hold"] == 1:
            st["z0"] = eef[2]
    elif st["hold"] >= 10 and st["released"] is None:
        st["released"] = eef.round(3).tolist()
        ctx.log("release", {"t": t, "eef": st["released"],
                            "goal": np.round(st["ref"]["place"][:2] + st["delta"][:2], 3).tolist()})
    if q >= 0.03 and st["released"] is None:
        st["hold"] = 0
    st.setdefault("steps", []).append(eef)
    return None
