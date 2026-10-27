"""E27: where does the policy get its task phase (reach vs transport) from? Reaching obs R (gripper open, before grasp) and
holding obs H (same episode). Swap ONE input of R with H's: gripper qpos (state), wrist image, agentview image, eef position.
Score = 1 - d(patched, chunk(H)) / d(chunk(R), chunk(H)) (1 = behaves fully like holding). Trained goal/spatial episodes."""
import sys, glob; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load()
import json
tr = json.load(open('library/notes/lab/trained_instructions.json'))
res = {k: [] for k in ("gripper", "wrist", "agentview", "eef_pos", "gripper+wrist", "all_images")}
for f in sorted(glob.glob("outputs/programs/none/libero_spatial__*/episode_*.npz")) + sorted(glob.glob("outputs/programs/none/libero_goal__[1289]/episode_*.npz")):
    ep = lab.load(f)
    g = [float(ep.obs(i)["robot0_gripper_qpos"][0]) for i in range(len(ep))]
    R = [i for i in range(len(ep)) if g[i] > 0.035]; H = [i for i in range(len(ep)) if g[i] < 0.02]
    if not R or not H: continue
    r, h = ep.obs(R[0]), ep.obs(H[0])
    ins = ep.instruction
    cr, ch = lab.chunk(r, ins), lab.chunk(h, ins); d0 = A.dist(cr, ch)
    if d0 < 0.5: continue
    def score(o): return 1 - A.dist(lab.chunk(o, ins), ch) / d0
    for k, keys in [("gripper", ["robot0_gripper_qpos"]), ("wrist", ["robot0_eye_in_hand_image"]), ("agentview", ["agentview_image"]),
                    ("eef_pos", ["robot0_eef_pos"]), ("gripper+wrist", ["robot0_gripper_qpos", "robot0_eye_in_hand_image"]),
                    ("all_images", ["agentview_image", "robot0_eye_in_hand_image"])]:
        o = dict(r); o.update({kk: h[kk] for kk in keys}); res[k].append(score(o))
print("pairs", len(res["gripper"]))
for k, v in res.items(): print(f"  {k:14} score {np.mean(v):+.2f} (median {np.median(v):+.2f})")
