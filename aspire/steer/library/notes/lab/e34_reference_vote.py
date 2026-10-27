"""E34: vote across references: track the destination with each trained episode's first frame (eps 0-2) as reference and take
the largest cluster of agreeing deltas (within 4 cm). ANALYSIS-ONLY ground truth (BDDL) for spatial_swap 0/1/5 eps 0-2."""
import sys, glob; sys.path[:0] = ['library', '.', 'library/notes/lab']
import numpy as np
from steerlib import analysis as A, scene
from gt_regions import regions
policy, lab, _ = A.load(); vla = policy.policy
def gt(t):
    rb, ib = regions("libero_spatial", t); rs, is_ = regions("libero_spatial_swap", t)
    a = rb[ib["plate_1"]]; b = rs.get(is_.get("plate_1")) or rs.get(is_.get("cookies_1"))
    return np.array(b) - np.array(a)
def vote(ds, tol=0.04):
    best = max(range(len(ds)), key=lambda i: sum(np.linalg.norm(ds - ds[i], axis=1) < tol))
    cl = ds[np.linalg.norm(ds - ds[best], axis=1) < tol]
    return cl.mean(0), len(cl)
e1, e3 = [], []
for t in (0, 1, 5):
    g = gt(t); place = np.load(f"library/references/libero_spatial__{t}.npz")["place"]; u = scene.project(place)
    Frs = [scene.patch_features(vla, np.load(f)["agentview_image"][0]) for f in sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz"))[:3]]
    for e in range(3):
        Fl = scene.patch_features(vla, lab.load(f"outputs/programs/none/libero_spatial_swap__{t}/episode_{e}.npz").obs(0)["agentview_image"])
        ds = []
        for Fr in Frs:
            m = scene.region(Fr, u, tau=0.3, radius=4); duv, s, s0 = scene.track(Fr, m, Fl)
            ds.append((scene.backproject(u + duv, place[2]) - scene.backproject(u, place[2]))[:2])
        ds = np.array(ds); v, n = vote(ds)
        e1.append(np.linalg.norm(ds[0] - g) * 100); e3.append(np.linalg.norm(v - g) * 100)
        print(f"swap {t} ep{e}: refs {np.round(ds*100,0).tolist()} vote {np.round(v*100,1)} (n={n}) gt {np.round(g*100,0)} | err single {e1[-1]:.1f} vote {e3[-1]:.1f}")
print(f"single-reference errors: median {np.median(e1):.1f}, >10 cm in {sum(x > 10 for x in e1)}/9;  vote: median {np.median(e3):.1f}, >10 cm in {sum(x > 10 for x in e3)}/9")
