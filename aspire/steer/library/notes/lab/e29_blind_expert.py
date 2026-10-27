"""E29: how much does the reach target depend on the images, inside the expert? Block the action expert's attention to the
agentview / wrist / both image segments (all expert layers, all flow steps) on reaching observations (trained goal/spatial
episodes) and compare the predicted grasp (eef + path at the first closing command) and chunk path with the normal one."""
import sys, glob; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, attention as AT
policy, lab, _ = A.load(); m = policy.policy.model.vlm_with_expert
def pg(o, ch):
    c = np.where(ch[:, 6] > 0)[0]; i = int(c[0]) if len(c) else len(ch) - 1
    return o["robot0_eef_pos"][:2] + A.path(ch)[i][:2]
res = {k: [] for k in ("agentview", "wrist", "both")}; dpath = {k: [] for k in res}; noise = []
for f in sorted(glob.glob("outputs/programs/none/libero_spatial__*/episode_0.npz")) + sorted(glob.glob("outputs/programs/none/libero_goal__[1289]/episode_0.npz")):
    ep = lab.load(f); o, ins = ep.obs(0), ep.instruction
    a0 = lab.chunk(o, ins); g0 = pg(o, a0); noise.append(A.dist(a0, lab.chunk(o, ins, seed=1)))
    for k, cols in (("agentview", range(0, 64)), ("wrist", range(64, 128)), ("both", range(0, 128))):
        cols = torch.tensor(list(cols))
        def bias(layer, q_len, k_len, cols=cols):
            if q_len != 50: return None
            b = torch.zeros(q_len, k_len); b[:, cols] = -1e9; return b
        r = AT.patch(m, bias)
        try: a1 = lab.chunk(o, ins)
        finally: r()
        res[k].append(np.linalg.norm(pg(o, a1) - g0) * 100); dpath[k].append(A.dist(a1, a0))
print(f"{len(noise)} obs; noise floor (seed) path distance {np.mean(noise):.2f} cm")
for k in res: print(f"  expert blind to {k:9}: predicted grasp moves {np.mean(res[k]):5.1f} cm (median {np.median(res[k]):.1f}); path change {np.mean(dpath[k]):.2f} cm = {np.mean(dpath[k])/np.mean(noise):.1f}x noise")
