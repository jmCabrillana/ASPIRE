"""E33: object-appearance memory for tracking: one plate prototype = mean SigLIP feature of the destination regions of all 10
spatial references (context averages out). Locate the plate in the live frame and in the reference (centroid of the connected
component of similarity > 0.85 max around the argmax); delta = live - ref, back-projected at the reference place height.
Scored against ANALYSIS-ONLY ground truth (BDDL) on spatial_swap 0/1/5 (eps 0-2) and trained spatial frames (delta = 0)."""
import sys, glob; sys.path[:0] = ['library', '.', 'library/notes/lab']
import numpy as np
from scipy import ndimage
from steerlib import analysis as A, scene
from gt_regions import regions
policy, lab, _ = A.load(); vla = policy.policy
refs = {t: np.load(f"library/references/libero_spatial__{t}.npz") for t in range(10)}
Fr = {t: scene.patch_features(vla, refs[t]["image"]) for t in refs}
protos = []
for t in refs:
    m = scene.region(Fr[t], scene.project(refs[t]["place"]), tau=0.3, radius=4)
    protos.append(Fr[t][m].mean(0))
P = np.mean([p / np.linalg.norm(p) for p in protos], 0); P /= np.linalg.norm(P)
def locate(F):
    s = F @ P; r, c = np.unravel_index(int(np.argmax(s)), s.shape)
    lab_, _ = ndimage.label(s > 0.85 * s.max()); comp = lab_ == lab_[r, c]
    rs, cs = np.nonzero(comp); return np.array([cs.mean() * 8 + 4, rs.mean() * 8 + 4]), float(s.max())
def gt(t):
    rb, ib = regions("libero_spatial", t); rs, is_ = regions("libero_spatial_swap", t)
    a = rb[ib["plate_1"]]; b = rs.get(is_.get("plate_1")) or rs.get(is_.get("cookies_1"))
    return np.array(b) - np.array(a)
errs = []
for t in (0, 1, 5):
    g = gt(t); u_ref, _ = locate(Fr[t]); z = refs[t]["place"][2]
    for e in range(3):
        o = lab.load(f"outputs/programs/none/libero_spatial_swap__{t}/episode_{e}.npz").obs(0)
        u, s = locate(scene.patch_features(vla, o["agentview_image"]))
        d = scene.backproject(u, z) - scene.backproject(u_ref, z); errs.append(np.linalg.norm(d[:2] - g) * 100)
        print(f"swap {t} ep{e}: delta {np.round(d[:2]*100,1)} gt {np.round(g*100,1)} err {errs[-1]:.1f} cm (sim {s:.2f})")
for t in range(0, 10, 3):
    f = sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz"))[-1]
    u, _ = locate(scene.patch_features(vla, lab.load(f).obs(0)["agentview_image"])); u_ref, _ = locate(Fr[t]); z = refs[t]["place"][2]
    print(f"trained {t}: delta {np.round((scene.backproject(u, z) - scene.backproject(u_ref, z))[:2]*100,1)} (should be ~0)")
print("median swap error", round(float(np.median(errs)), 1), "cm; max", round(float(np.max(errs)), 1))
np.save("library/knowledge/plate_prototype_spatial.npy", P)
