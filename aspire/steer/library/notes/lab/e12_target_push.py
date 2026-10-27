"""E12: can the reach target be moved by pushing the E11 probe direction (expert neuron space, down_proj inputs)?
Add alpha * w (w = unit ridge-probe direction for grasp x or y at expert layer L, alpha in units of the probe's projection
spread) to every action token's neuron activations, and read the chunk's end displacement (50 steps, cm)."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steer.interp import Bank
from steer.lab import down_proj
from steerlib import analysis as A
policy, lab, _ = A.load()
d = np.load("outputs/bank_target_expert.npz", allow_pickle=True)
acts = {k[3:]: d[k] for k in d.files if k.startswith("a::")}; labels = {k[3:]: d[k] for k in d.files if k.startswith("l::")}
bank = Bank(acts, labels, [])
o = lab.load("outputs/programs/none/libero_goal__1/episode_0.npz").obs(0)
INS = "put the bowl on the stove"
for L in (4, 8, 12):
    for lbl, ax in (("gx", 0), ("gy", 1)):
        pr = bank.probe(f"expert{L}:all", lbl)
        w = pr["direction"]; proj = acts[f"expert{L}:all"] @ w; sd = proj.std()
        # slope: how much the label changes per unit projection (least squares)
        slope = np.polyfit(proj, labels[lbl], 1)[0]
        row = []
        for a in (-4, -2, 0, 2, 4):
            vec = torch.tensor(a * sd * w, dtype=torch.float32)
            h = down_proj(policy, "expert", L).register_forward_pre_hook(lambda m, i, v=vec: (i[0] + v.to(i[0].device, i[0].dtype),))
            try: ch = lab.chunk(o, INS)
            finally: h.remove()
            row.append(f"{a:+d}sd:{A.path(ch)[-1][ax]*100:+.0f}")
        print(f"expert{L} {lbl} R2 {pr['score']:.2f} (1 sd = {slope*sd*100:+.1f} cm of label)  end50 on axis: {' '.join(row)}")
