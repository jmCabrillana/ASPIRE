"""E36: voting over reference frames for GOAL-scene tracking (pick and place), goal_swap 1 and 8 (frozen first frames eps 0-2).
References: first frames of the frozen trained goal episodes of the same task (eps 0-2). ANALYSIS-ONLY ground truth (BDDL)."""
import sys, glob; sys.path[:0] = ['library', '.', 'library/notes/lab']
import numpy as np
from steerlib import analysis as A, scene
from gt_regions import regions
policy, lab, _ = A.load(); vla = policy.policy
OBJ = {1: ("akita_black_bowl_1", "flat_stove_1"), 8: ("akita_black_bowl_1", "plate_1")}
def gt(t, obj):
    rb, ib = regions("libero_goal", t); rs, is_ = regions("libero_goal_swap", t)
    return np.array(rs[is_[obj]]) - np.array(rb[ib[obj]])
def vote(ds, ss, tol=0.04):
    sup = [int(np.sum(np.linalg.norm(ds - d, axis=1) < tol)) for d in ds]
    b = max(range(len(ds)), key=lambda i: (sup[i], ss[i])); return ds[np.linalg.norm(ds - ds[b], axis=1) < tol].mean(0), sup[b]
for t in (1, 8):
    ref = np.load(f"library/references/libero_goal__{t}.npz")
    Frs = [scene.patch_features(vla, np.load(f)["agentview_image"][0]) for f in sorted(glob.glob(f"outputs/programs/none/libero_goal__{t}/episode_*.npz"))[:3]]
    for nm, pt, obj in (("pick", ref["pick"], OBJ[t][0]), ("place", ref["place"], OBJ[t][1])):
        g = gt(t, obj); u = scene.project(pt)
        for e in range(3):
            Fl = scene.patch_features(vla, lab.load(f"outputs/programs/none/libero_goal_swap__{t}/episode_{e}.npz").obs(0)["agentview_image"])
            ds, ss = [], []
            for Fr in Frs:
                m = scene.region(Fr, u, tau=0.3, radius=4); duv, s, s0 = scene.track(Fr, m, Fl)
                ds.append((scene.backproject(u + duv, pt[2]) - scene.backproject(u, pt[2]))[:2]); ss.append(s)
            v, n = vote(np.array(ds), ss)
            print(f"goal_swap {t} {nm:5} ep{e}: votes {np.round(np.array(ds)*100,0).tolist()} -> {np.round(v*100,1)} (n={n}) gt {np.round(g*100,0)} err {np.linalg.norm(v-g)*100:.1f}")
