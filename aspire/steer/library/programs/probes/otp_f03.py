"""Skill composition for 'open the X and put the Y in it' (goal_lan 3 type): sequence two sub-skills of the frozen VLA and
supply the one thing neither sub-skill knows, the location of the container that the first sub-skill opened, from the
robot's OWN motion during that sub-skill (proprioception only, no simulator state).

phase 1 'open' : the VLA runs sub-instruction 1 (canonical first clause, e.g. 'open the top drawer of the cabinet').
                 The program records the eef path: the turnaround point (where the reach toward the handle reverses into
                 the pull) is the closed drawer's handle; the pull vector says where the drawer came out.
phase 2 'pick' : once the pull has stopped, sub-instruction 2 (a trained instruction with the same object, so the VLA
                 grasps it); session 1 (lan3 decomposition probe) showed the VLA picks but never places into the drawer.
phase 3 'carry': once holding and lifted, the program carries the object above the opened container (handle turnaround
                 + FRAC * pull vector, + DZ) and releases it.
Parameters (env, for the dev-episode grid): SUB1, SUB2, FRAC, DZ, ZC.
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

DESCRIPTION = "sequence open-skill and pick-skill; place into the container located from the robot's own pull motion"
N_ACTION_STEPS = 10
STEP = 0.0095
SUB1 = os.environ.get("SUB1", "open the top drawer of the cabinet")
SUB2 = os.environ.get("SUB2", "put the bowl in the top drawer of the cabinet")
FRAC = float(os.environ.get("FRAC", "0.5"))     # fraction of the pull vector from the closed-handle point
DZ = float(os.environ.get("DZ", "0.06"))        # release height above the handle
ZC = float(os.environ.get("ZC", "0.11"))
ZCLEAR = float(os.environ.get("ZCLEAR", "0.07"))  # gripper height above the handle at which the object clears the drawer front        # carry height above the handle
OUT = float(os.environ.get("OUT", "0.0"))       # extra offset along the pull direction (m)
GAIN = float(os.environ.get("GAIN", "3"))
ROTGAIN = float(os.environ.get("ROTGAIN", "0.4"))
HOME = int(os.environ.get("HOME_RESET", "1"))    # return to the episode's start pose before the next sub-skill


def setup(ctx):
    ctx.instruction = SUB1
    ctx.memory.update(mode="open", path=[], t_switch=None)
    ctx.log("subs", [SUB1, SUB2])


def program(t, obs, internals, ctx):
    return []


def _pull(path):
    """Turnaround point and pull vector from the eef path of the opening sub-skill (horizontal plane)."""
    P = np.array(path)
    start = P[0]
    d = np.linalg.norm((P - start)[:, :2], axis=1)
    i = int(np.argmax(d))                       # farthest point from the start = the handle (reach ends there)
    return i, P[i], P[-1] - P[i]


def on_step(t, obs, action, ctx):
    mem = ctx.memory
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    q = float(obs["robot0_gripper_qpos"][0])
    if mem["mode"] == "open":
        mem["path"].append(eef.copy())
        mem.setdefault("quat0", np.asarray(obs["robot0_eef_quat"], dtype=float).copy())
        P = np.array(mem["path"])
        i, handle, pull = _pull(mem["path"])
        moved = np.linalg.norm(pull[:2])
        # the pull is over when the hand has come back >= 6 cm from the handle and has stopped moving along the pull
        recent = np.linalg.norm((P[-1] - P[max(0, len(P) - 8)])[:2]) if len(P) > 8 else 1.0
        if (moved > 0.06 and recent < 0.008 and len(P) - i > 10) or t >= 150:
            mem.update(mode="pick", handle=handle, pull=pull, t_switch=t)
            u = pull[:2] / (np.linalg.norm(pull[:2]) + 1e-9)
            tgt = handle.copy()
            tgt[:2] += FRAC * pull[:2] + OUT * u
            mem["target"] = tgt
            mem["mode"] = "home" if HOME else "pick"
            if not HOME:
                ctx.instruction = SUB2
                ctx.policy.reset()
            ctx.log("switch", {"t": t, "handle": np.round(handle, 3).tolist(), "pull_cm": np.round(pull * 100, 1).tolist(),
                               "target": np.round(tgt, 3).tolist()})
        return None
    if mem["mode"] == "home":                 # each sub-skill was trained from the start pose: go back there first
        home = mem["path"][0]
        mem["n_home"] = mem.get("n_home", 0) + 1
        rot = (R.from_quat(mem["quat0"]) * R.from_quat(np.asarray(obs["robot0_eef_quat"], dtype=float)).inv()).as_rotvec()
        if (np.linalg.norm(home - eef) < 0.02 and np.linalg.norm(rot) < 0.1) or mem["n_home"] > 60:
            mem["mode"] = "pick"
            ctx.instruction = SUB2
            ctx.policy.reset()
            ctx.log("home", {"t": t, "eef": np.round(eef, 3).tolist(), "rot_deg": round(float(np.degrees(np.linalg.norm(rot))), 1)})
            return None
        a = np.zeros(7, dtype=np.float32)
        a[:3] = np.clip((home - eef) / (STEP * 3), -1, 1)
        a[3:6] = np.clip(rot / 0.5 * ROTGAIN, -1, 1)              # OSC: world-frame axis-angle delta, 1 <-> 0.5 rad
        a[6] = -1.0
        return a
    if mem["mode"] == "pick":
        holding = 0.0005 < q < 0.03
        mem["hold"] = mem.get("hold", 0) + 1 if holding else 0
        if mem["hold"] == 1:
            mem["z_grasp"] = eef[2]
        if mem["hold"] >= 12 and eef[2] > mem["z_grasp"] + 0.03:
            mem["mode"] = "carry"
            mem["z_carry"] = max(eef[2], mem["handle"][2] + ZC)
            ctx.log("takeover", {"t": t, "eef": np.round(eef, 3).tolist()})
        return None
    tgt3 = mem["target"]
    a = np.zeros(7, dtype=np.float32)
    a[6] = 1.0
    if mem["mode"] == "carry":
        tgt = np.array([tgt3[0], tgt3[1], mem["z_carry"]])
        if eef[2] < mem["handle"][2] + ZCLEAR:                   # lift until clear of the container's front, moving across meanwhile
            tgt = np.array([eef[0], eef[1], mem["z_carry"]])    # (moving across earlier pushed the drawer shut)
        elif np.linalg.norm((tgt - eef)[:2]) < 0.015:
            mem["mode"] = "lower"
    if mem["mode"] == "lower":
        tgt = np.array([tgt3[0], tgt3[1], tgt3[2] + DZ])
        mem["n_lower"] = mem.get("n_lower", 0) + 1
        zs = mem.setdefault("z_lower", [])
        zs.append(eef[2])
        stalled = len(zs) > 6 and zs[-6] - zs[-1] < 0.002          # the object rests on something: let go
        if abs(eef[2] - tgt[2]) < 0.01 or stalled or mem["n_lower"] > 40:
            mem["mode"] = "release"
            ctx.log("release", {"t": t, "eef": np.round(eef, 3).tolist()})
    if mem["mode"] == "release":
        mem["n_rel"] = mem.get("n_rel", 0) + 1
        a[6] = -1.0
        if mem["n_rel"] > 15:                                     # move up and away
            a[2] = 0.5
        return a
    a[:3] = np.clip((tgt - eef) / (STEP * GAIN), -1, 1)     # GAIN 2 was fast enough to shake the bowl loose
    return a
