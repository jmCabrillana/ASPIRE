"""Example of an open program: custom logging hooks, an instruction rewrite, an activation edit and an action edit."""

import torch

DESCRIPTION = "example: log layer-8 token norms, slow the arm down after the gripper closes"


def setup(ctx):
    ctx.memory["calls"] = 0

    def log_tokens(module, inputs, output):          # VLM layer 8 MLP output on the prefix: per-token norms (first 8 tokens)
        ctx.memory["calls"] += 1
        ctx.log("vlm8_token_norms", output[0, :8].norm(dim=-1))

    ctx.hook(ctx.vlm_layers[8].mlp, log_tokens)


def program(t, obs, internals, ctx):
    if internals:
        ctx.log("wrist_attention_layer0", internals["attention"].get(0, {}).get("image2"))
    return []


def on_step(t, obs, action, ctx):
    if obs["robot0_gripper_qpos"][0] < 0.03:          # gripper closed on something: halve the translation
        action = action.copy()
        action[:3] *= 0.5
        return action
    return None


def on_end(result, ctx):
    ctx.log("prefix_passes", ctx.memory["calls"])
