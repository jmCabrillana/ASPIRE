"""E21: with the bowl held (spatial scenes), where does 'pick up the black bowl ... and place it on the X' send the arm?
Predicted chunk end (eef + path, 0.0095 m/unit) for destination nouns X, compared with ANALYSIS-ONLY object positions (BDDL)."""
import sys, glob, json; sys.path[:0] = ['library', '.', 'library/notes/lab']
import numpy as np
from steerlib import analysis as A
from gt_regions import regions
policy, lab, _ = A.load()
INS = json.load(open('library/notes/lab/trained_instructions.json'))["libero_spatial"]
NOUNS = ["plate", "cookie box", "ramekin", "stove", "wooden cabinet", "table center"]
for t in (0, 1, 2, 5):
    r, init = regions("libero_spatial", t)
    pos = {k: r.get(v) for k, v in init.items()}
    gt = {"plate": pos.get("plate_1"), "cookie box": pos.get("cookies_1"), "ramekin": pos.get("glazed_rim_porcelain_ramekin_1"),
          "stove": pos.get("flat_stove_1"), "wooden cabinet": pos.get("wooden_cabinet_1")}
    for f in sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz")):
        ep = lab.load(f)
        hold = [i for i in range(len(ep)) if ep.obs(i)["robot0_gripper_qpos"][0] < 0.03]
        if not hold: continue
        o = ep.obs(hold[0]); base = INS[t].rsplit(" and place it on ", 1)[0]
        print(f"spatial {t} {f[-13:]} holding at {np.round(o['robot0_eef_pos'][:2]*100,1)}")
        for n in NOUNS:
            ch = lab.chunk(o, f"{base} and place it on the {n}")
            e = (o["robot0_eef_pos"][:2] + A.path(ch)[-1][:2]) * 100
            g = gt.get(n)
            print(f"   on the {n:15} end {np.round(e, 1)}   object at {None if g is None else np.round(np.array(g)*100, 1)}")
        break
