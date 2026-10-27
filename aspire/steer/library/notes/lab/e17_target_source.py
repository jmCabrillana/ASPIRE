"""E17: what determines the reach target: the instruction or the image? Spatial suite (layouts differ per task).
predicted grasp = eef0 + chunk path at the first closing command (0.0095 m/unit). (0) validate against actual grasps.
(a) scene A + instruction B: distance of the predicted grasp to B's memorised grasp vs to A's.
(b) scene A + instruction A, with the agentview tokens (prefix, all VLM layers) patched from scene B: does it move to B's grasp?"""
import sys, glob; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A, residual as R
policy, lab, _ = A.load(); vla = policy.policy
import json
INS = json.load(open('library/notes/lab/trained_instructions.json'))["libero_spatial"]
def pgrasp(o, ins):
    ch = lab.chunk(o, ins); c = np.where(ch[:, 6] > 0)[0]
    i = int(c[0]) if len(c) else len(ch) - 1
    return o["robot0_eef_pos"] + A.path(ch)[i], (int(c[0]) if len(c) else None)
ref = {t: np.load(f"library/references/libero_spatial__{t}.npz") for t in range(10)}
obs = {t: lab.load(sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz"))[0]).obs(0) for t in range(10)}
print("(0) predicted vs actual grasp (cm):")
for t in range(10):
    g, i = pgrasp(obs[t], INS[t]); print(f"  task {t}: close@{i} err {np.linalg.norm(g[:2]-ref[t]['pick'][:2])*100:.1f}")
print("(a) scene A + instruction B: d(grasp, B memorised) vs d(grasp, A's own)")
da, db = [], []
for a in range(10):
    for b in range(10):
        if a == b: continue
        g, _ = pgrasp(obs[a], INS[b])
        db.append(np.linalg.norm(g[:2] - ref[b]["pick"][:2])); da.append(np.linalg.norm(g[:2] - ref[a]["pick"][:2]))
print(f"  mean to B's memorised {np.mean(db)*100:.1f} cm, to A's {np.mean(da)*100:.1f} cm, closer to B in {np.mean(np.array(db)<np.array(da))*100:.0f}% of pairs")
print("(b) scene A, instruction A, agentview tokens from scene B (all layers) / the image itself swapped")
L = list(range(16)); img = list(range(0, 64))
dA, dB, dA2, dB2 = [], [], [], []
for a in range(10):
    for b in range(10):
        if a == b: continue
        with R.record(vla, L) as rb:
            lab.chunk(obs[b], INS[a])
        with R.patch(vla, {l: (img, rb[l][img]) for l in L}):
            g, _ = pgrasp(obs[a], INS[a])
        dA.append(np.linalg.norm(g[:2] - ref[a]["pick"][:2])); dB.append(np.linalg.norm(g[:2] - ref[b]["pick"][:2]))
        o2 = dict(obs[a]); o2["agentview_image"] = obs[b]["agentview_image"]
        g2, _ = pgrasp(o2, INS[a])
        dA2.append(np.linalg.norm(g2[:2] - ref[a]["pick"][:2])); dB2.append(np.linalg.norm(g2[:2] - ref[b]["pick"][:2]))
print(f"  patched agentview tokens: to A {np.mean(dA)*100:.1f} cm, to B {np.mean(dB)*100:.1f} cm, closer to B {np.mean(np.array(dB)<np.array(dA))*100:.0f}%")
print(f"  agentview image of B (wrist of A): to A {np.mean(dA2)*100:.1f} cm, to B {np.mean(dB2)*100:.1f} cm, closer to B {np.mean(np.array(dB2)<np.array(dA2))*100:.0f}%")
