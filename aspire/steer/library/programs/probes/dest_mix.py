"""PROBE (mechanism evidence, not a candidate): closed-loop internal destination steering. Instruction A (the task's own) is
kept; at every chunk the residual of A's destination token (all VLM layers) is replaced by (1-a) r_A + a r_B, with r_B from
instruction B on the current observation (live). Env: DEST_B (instruction B, same token count), DEST_A_WORD, DEST_ALPHA.
Read the release point from the trace (eef when the gripper opens after holding)."""
import os
import torch
from steer.lab import Lab
from steerlib import analysis as A, residual as R

B, WORD, ALPHA = os.environ["DEST_B"], os.environ["DEST_A_WORD"], float(os.environ["DEST_ALPHA"])
B_WORD = os.environ.get("DEST_B_WORD", WORD)
UNTIL_GRASP = os.environ.get("DEST_UNTIL_GRASP", "0") == "1"
N_ACTION_STEPS = 25


def setup(ctx):
    toks = A.tokens(ctx.instruction)
    ctx.memory.update(pos=[128 + toks.index(WORD)], posb=[128 + A.tokens(B).index(B_WORD)], spec={}, lab=Lab(ctx.policy))
    layers = ctx.vlm_layers
    mid = {}
    for L in range(16):
        def pre(m, a, L=L): mid[L] = a[0]
        def post(m, a, o, L=L):
            sp = ctx.memory["spec"].get(L)
            if sp is None or o.shape[1] <= 100: return o
            o = o.clone(); p = torch.tensor(ctx.memory["pos"], device=o.device)
            o[0, p] = sp.to(o.device, o.dtype) - mid[L][0, p]
            return o
        ctx.hook(layers[L].post_attention_layernorm, pre, pre=True); ctx.hook(layers[L].mlp, post)
    ctx.log("dest_mix", {"B": B, "alpha": ALPHA})


def program(t, obs, internals, ctx):
    if ALPHA == 0: return []
    if UNTIL_GRASP and (ctx.memory.get("grasped") or obs["robot0_gripper_qpos"][0] < 0.03):
        ctx.memory["grasped"] = True; ctx.memory["spec"] = {}; return []
    pos, posb = ctx.memory["pos"], ctx.memory["posb"]
    ctx.memory["spec"] = {}
    with R.record(ctx.vla, range(16)) as ra: ctx.memory["lab"].chunk(obs, ctx.instruction)
    with R.record(ctx.vla, range(16)) as rb: ctx.memory["lab"].chunk(obs, B)
    ctx.memory["spec"] = {L: (1 - ALPHA) * ra[L][pos] + ALPHA * rb[L][posb] for L in range(16)}
    ctx.policy.reset()
    return []
