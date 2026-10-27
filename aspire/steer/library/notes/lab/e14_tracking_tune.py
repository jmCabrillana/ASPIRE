"""E14: tune destination tracking against ANALYSIS-ONLY ground truth (BDDL region centres; never used by programs).
Error (cm) of the estimated destination displacement for spatial_swap 0/1/5 and goal_swap 1 (eps 0-2), per (tau, radius)."""
import sys; sys.path[:0] = ['library', '.', 'library/notes/lab']
import numpy as np
from steerlib import analysis as A, scene
from gt_regions import regions
policy, lab, _ = A.load()
CASES = [("libero_spatial", "libero_spatial_swap", 0, "plate_1"), ("libero_spatial", "libero_spatial_swap", 1, "plate_1"),
         ("libero_spatial", "libero_spatial_swap", 5, "plate_1"), ("libero_goal", "libero_goal_swap", 1, "flat_stove_1")]
def gt(base, swap, t, obj):
    rb, ib = regions(base, t); rs, is_ = regions(swap, t)
    if obj not in is_ or ib.get(obj) is None: return None
    a, b = rb.get(ib[obj]), rs.get(is_[obj])
    return None if a is None or b is None else np.array(b) - np.array(a)
data = []
for base, swap, t, obj in CASES:
    g = gt(base, swap, t, obj)
    ref = np.load(f"library/references/{base}__{t}.npz")
    Fr = scene.patch_features(policy.policy, ref["image"])
    for e in range(3):
        o = lab.load(f"outputs/programs/none/{swap}__{t}/episode_{e}.npz").obs(0)
        data.append((swap, t, e, g, ref["place"], Fr, scene.patch_features(policy.policy, o["agentview_image"])))
print("gt (cm):", {f"{s}{t}": None if g is None else np.round(g * 100, 1).tolist() for s, t, e, g, *_ in data if e == 0})
for tau in (0.3, 0.4, 0.5, 0.6):
    for rad in (2, 4, 6):
        errs = []
        for s, t, e, g, place, Fr, Fl in data:
            u = scene.project(place); m = scene.region(Fr, u, tau=tau, radius=rad)
            duv, sc, s0 = scene.track(Fr, m, Fl)
            d = scene.backproject(u + duv, place[2]) - scene.backproject(u, place[2])
            if g is not None: errs.append(np.linalg.norm(d[:2] - g) * 100)
        print(f"tau {tau} radius {rad}: errors {np.round(errs, 0).astype(int).tolist()}  median {np.median(errs):.1f}")
