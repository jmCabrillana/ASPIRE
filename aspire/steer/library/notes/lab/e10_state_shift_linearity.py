"""E10: how does the predicted reach respond to large state shifts? Reach end (first 25 / 50 steps, cm, from the TRUE eef)
vs a shift s of the state's eef along x and y (s = -20..+20 cm), on trained-scene and swap first frames."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
for f, ins in [("outputs/programs/none/libero_goal__1/episode_0.npz", "put the bowl on the stove"),
               ("outputs/programs/none/libero_goal_swap__1/episode_0.npz", "put the bowl on the stove")]:
    o = lab.load(f).obs(0)
    for ax in (0, 1):
        row = []
        for s in (-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2):
            o2 = dict(o); p = o["robot0_eef_pos"].copy(); p[ax] += s; o2["robot0_eef_pos"] = p
            ch = lab.chunk(o2, ins)
            e25, e50 = A.path(ch[:25])[-1] * 100, A.path(ch)[-1] * 100
            row.append(f"{s*100:+.0f}:{e25[ax]:+.0f}/{e50[ax]:+.0f}")
        print(f.split('none/')[1][:20], "axis", "xy"[ax], "  shift:end25/end50 on that axis  ", " ".join(row))
