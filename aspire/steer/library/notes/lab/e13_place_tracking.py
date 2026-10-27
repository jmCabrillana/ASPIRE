"""E13: does feature-template tracking find the moved plate in spatial_swap (and goal_swap) first frames? Draws the reference
place point (red, projected into the reference frame) and the tracked destination (green) on reference | live frames."""
import sys; sys.path[:0] = ['library', '.', 'library/programs/candidates']
import numpy as np
from types import SimpleNamespace
from PIL import Image, ImageDraw
from steerlib import analysis as A, scene
import carry_to_target as CT
policy, lab, _ = A.load()
rows = []
for suite, task in [("libero_spatial_swap", 0), ("libero_spatial_swap", 1), ("libero_spatial_swap", 5), ("libero_goal_swap", 1)]:
    logs = []
    ctx = SimpleNamespace(task=SimpleNamespace(suite=suite, task_id=task), memory={}, vla=policy.policy, log=lambda k, v: logs.append(v))
    CT.setup(ctx)
    for e in range(2):
        ctx.memory.pop("goal", None)
        o = lab.load(f"outputs/programs/none/{suite}__{task}/episode_{e}.npz").obs(0)
        CT.program(1, o, None, ctx)
        g = ctx.memory["goal"]; ref = ctx.memory["ref"]
        print(suite, task, e, logs[-1])
        I = Image.fromarray(np.concatenate([ref["image"], o["agentview_image"]], 1)); dr = ImageDraw.Draw(I)
        u0 = scene.project(ref["place"]); u1 = scene.project(g)
        dr.ellipse([u0[0] - 5, u0[1] - 5, u0[0] + 5, u0[1] + 5], outline=(255, 0, 0), width=2)
        dr.ellipse([256 + u1[0] - 5, u1[1] - 5, 256 + u1[0] + 5, u1[1] + 5], outline=(0, 255, 0), width=2)
        rows.append(np.array(I))
Image.fromarray(np.concatenate(rows, 0)).save(sys.argv[1])
