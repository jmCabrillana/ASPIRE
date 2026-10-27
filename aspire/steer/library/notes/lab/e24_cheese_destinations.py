"""E24: holding the cream cheese (goal scene), where does 'put the cream cheese in/on the X' send the arm? Destination nouns
X (single tokens) as anchors for the destination knob. Predicted chunk end (eef + path, 0.0095 m/unit)."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
for f, i in [("outputs/programs/none/libero_goal__6/episode_0.npz", 1), ("outputs/programs/none/libero_goal__6/episode_1.npz", 1),
             ("outputs/programs/none/libero_goal__6/episode_3.npz", 1)]:
    o = lab.load(f).obs(i)
    print(f[-14:], "eef", np.round(o["robot0_eef_pos"] * 100, 1), "grip", round(float(o["robot0_gripper_qpos"][0]), 4))
    for ins in ["put the cream cheese in the bowl", "put the cream cheese on the plate", "put the cream cheese on the stove",
                "put the cream cheese on the rack", "put the cream cheese on the cabinet", "put the cream cheese in the basket"]:
        ch = lab.chunk(o, ins); e = (o["robot0_eef_pos"] + A.path(ch)[-1]) * 100
        print(f"   {ins:36} end {np.round(e, 1)}  opens@{(np.where(ch[:, 6] < 0)[0][:1].tolist() or [None])[0]}")
