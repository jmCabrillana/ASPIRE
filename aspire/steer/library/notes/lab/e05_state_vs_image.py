"""E05: does the reach target come from the proprio state or from the image?
Shift the state's eef position by d (only the policy's input; images unchanged) and measure the change of the predicted path
end (first 25 steps, cm). Slope -1 = target computed from the state (absolute memorised coordinate); 0 = from the images.
Then shift the agentview image content by k pixels (np.roll, both axes) and measure the path change."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
INS = ["put the bowl on the plate", "put the wine bottle on the rack", "put the cream cheese in the bowl"]
def end(ch, n=25): return A.path(ch[:n])[-1] * 100
for e in (0, 1):
    ep = lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz")
    for ci in (0, 1):
        o = ep.obs(ci)
        for ins in INS:
            base = end(lab.chunk(o, ins))
            row = []
            for ax in range(3):
                for d in (-0.05, 0.05):
                    o2 = dict(o); p = o["robot0_eef_pos"].copy(); p[ax] += d; o2["robot0_eef_pos"] = p
                    row.append((end(lab.chunk(o2, ins))[ax] - base[ax]) / (d * 100))
            img = []
            for k in (-16, 16):
                for axis in (0, 1):
                    o3 = dict(o); o3["agentview_image"] = np.roll(o["agentview_image"], k, axis=axis)
                    img.append(np.round(end(lab.chunk(o3, ins)) - base, 1))
            print(f"ep{e} c{ci} {ins[:28]:28} state-slope x {row[0]:+.2f}/{row[1]:+.2f} y {row[2]:+.2f}/{row[3]:+.2f} z {row[4]:+.2f}/{row[5]:+.2f}"
                  f" | img roll16 d_end(cm) {img[0].tolist()} {img[1].tolist()} {img[2].tolist()} {img[3].tolist()}")
