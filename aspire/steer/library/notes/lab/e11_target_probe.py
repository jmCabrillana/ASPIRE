"""E11: where is the policy's reach target encoded? Bank over the frozen policy's trained-suite episodes (goal 1,2,4,8,9;
spatial 0-9), chunks before the first grasp, label = the episode's grasp point x / y (eef when the gripper first closes).
Linear probes (ridge, folds grouped by episode) per VLM segment (image, image2, lang, state) and expert layer."""
import sys, glob, json; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A, bank as B
policy, lab, _ = A.load()
files, grasp = [], {}
for f in sorted(glob.glob("outputs/programs/none/libero_*/episode_*.npz")):
    if any(s in f for s in ("_lan", "_swap", "object")): continue
    d = np.load(f)
    try: js = json.load(open(f.replace(".npz", ".json")))
    except FileNotFoundError: continue
    c = np.where(d["actions"][:, 6] > 0)[0]
    if not len(c): continue
    t = int(c[0]) + 1
    grasp[f] = np.array(min(js["trace"]["steps"], key=lambda x: abs(x["t"] - t))["eef"]); files.append(f)
print(len(files), "episodes")
lab_x = lambda ep, i, o: 0.0
bank = B.build(lab, files, [4, 8, 12, 15], "vlm", extra_labels={})
bank_e = B.build(lab, files, [4, 8, 12, 15], "expert", extra_labels={})
for bk in (bank, bank_e):
    ep = bk.labels["episode"].astype(int)
    pre = np.array([m["t"] < 1 + np.where(np.load(m["file"])["actions"][:, 6] > 0)[0][0] for m in bk.meta])
    gx = np.array([grasp[m["file"]][0] for m in bk.meta]); gy = np.array([grasp[m["file"]][1] for m in bk.meta])
    keep = pre
    for k in list(bk.acts): bk.acts[k] = bk.acts[k][keep]
    for k in list(bk.labels): bk.labels[k] = bk.labels[k][keep]
    bk.labels["gx"], bk.labels["gy"] = gx[keep], gy[keep]
    bk.meta = [m for m, kk in zip(bk.meta, keep) if kk]
    print("chunks", int(keep.sum()), "episodes", len(np.unique(bk.labels["episode"])))
    for key in sorted(bk.acts):
        r = [bk.probe(key, lbl) for lbl in ("gx", "gy")]
        print(f"  {key:16} R2 gx {r[0]['score']:+.2f}  gy {r[1]['score']:+.2f}")
bank.save("outputs/bank_target_vlm.npz"); bank_e.save("outputs/bank_target_expert.npz")
