"""E30: robust destination tracking: median over several (tau, radius) trackers vs the single E14 setting. ANALYSIS-ONLY ground
truth (BDDL; spatial_swap 5: the plate sits on the cookie box -> its position)."""
import sys; sys.path[:0] = ['library', '.', 'library/notes/lab']
import numpy as np
from steerlib import analysis as A, scene
from gt_regions import regions
policy, lab, _ = A.load()
def gt(t):
    rb, ib = regions("libero_spatial", t); rs, is_ = regions("libero_spatial_swap", t)
    a = rb[ib["plate_1"]]; b = rs.get(is_.get("plate_1")) or rs[is_["cookies_1"]]
    return np.array(b) - np.array(a)
SET = [(0.3, 4), (0.3, 3), (0.4, 4), (0.25, 5), (0.35, 6), (0.5, 3)]
for t in (0, 1, 5):
    g = gt(t); ref = np.load(f"library/references/libero_spatial__{t}.npz"); Fr = scene.patch_features(policy.policy, ref["image"])
    u = scene.project(ref["place"])
    for e in range(3):
        o = lab.load(f"outputs/programs/none/libero_spatial_swap__{t}/episode_{e}.npz").obs(0)
        Fl = scene.patch_features(policy.policy, o["agentview_image"])
        ds = []
        for tau, rad in SET:
            m = scene.region(Fr, u, tau=tau, radius=rad); duv, s, s0 = scene.track(Fr, m, Fl)
            d = scene.backproject(u + duv, ref["place"][2]) - scene.backproject(u, ref["place"][2]); ds.append(d[:2])
        ds = np.array(ds); med = np.median(ds, 0)
        print(f"swap {t} ep{e} gt {np.round(g*100,1)}  single err {np.linalg.norm(ds[0]-g)*100:5.1f}  median err {np.linalg.norm(med-g)*100:5.1f}  all {np.round(ds*100,0).tolist()}")
