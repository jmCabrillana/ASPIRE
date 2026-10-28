"""Closed-loop test of a gradient-learned destination vector (E37): add dose * v at the action expert's layer-L MLP output
while the robot holds an object (gated: gripper closed on something and lifted), so the VLA itself carries, aligns and releases
at a shifted place point. Env: DEST_VEC (library/directions/*.pt from E37), DOSE, GATE (hold | always)."""
import os
from pathlib import Path

import numpy as np
import torch

DESCRIPTION = "E37 destination vector in the expert, gated on the holding phase"
N_ACTION_STEPS = 10
VEC = os.environ.get("DEST_VEC", "dest_expert0_y8")
DOSE = float(os.environ.get("DOSE", "1"))
GATE = os.environ.get("GATE", "hold")
DIRS = Path(__file__).resolve().parents[2] / "directions"


def setup(ctx):
    d = torch.load(DIRS / f"{VEC}.pt")
    v = d["v"].float()
    st = ctx.memory
    st.update(scale=DOSE if GATE == "always" else 0.0, hold=0, released=None, closed_at=None)

    def hook(m, inp, out):
        if st["scale"] == 0:
            return None
        return out + (st["scale"] * v).to(out.device, out.dtype)
    ctx.hook(ctx.expert_layers[d["layer"]].mlp, hook)
    ctx.log("vector", {"name": VEC, "layer": d["layer"], "D_cm": np.round(np.asarray(d["D"]) * 100, 1).tolist(), "dose": DOSE,
                       "gate": GATE})


def program(t, obs, internals, ctx):
    st = ctx.memory
    if GATE == "hold":
        on = st["hold"] >= 5 and st.get("z0") is not None and obs["robot0_eef_pos"][2] > st["z0"] + 0.03
        st["scale"] = DOSE if on and st["released"] is None else 0.0
    return []


def on_step(t, obs, action, ctx):
    st = ctx.memory
    q = float(obs["robot0_gripper_qpos"][0])
    eef = np.asarray(obs["robot0_eef_pos"])
    if 0.0005 < q < 0.03:
        st["hold"] += 1
        if st["hold"] == 1:
            st["z0"] = eef[2]; st["closed_at"] = eef.round(3).tolist()
    elif st["hold"] >= 10 and st["released"] is None:
        st["released"] = eef.round(3).tolist()
        ctx.log("release", {"t": t, "eef": st["released"], "grasp": st["closed_at"]})
    if q >= 0.03 and st["released"] is None:
        st["hold"] = 0
    return None
