"""Table-plane calibration of the agentview camera from the policy's own successful trained-suite episodes (no sim internals).
For each successful episode with --save-obs and --trace:
  grasp world point = eef (trace, every 5 steps) at the first step whose action closes the gripper (action[6] > 0);
  object pixel      = centroid of the footprint: pixels that are not table-coloured in frame 0 but table-coloured in the last
                      saved frame (the object has been carried away), above the robot base region.
Fits a homography pixel (u, v) of the RAW agentview image -> world (x, y) with RANSAC. Saves library/knowledge/agentview_H.npy.
Usage: calib2.py [--show out.png]"""
import sys, glob, json
import numpy as np
from scipy import ndimage

def table(img):
    im = img.astype(int); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > g) & (g > b) & (r - b > 25) & (im.mean(-1) > 110)

def footprint(f0, f1, vmax=165):
    m = ~table(f0) & table(f1)
    m[vmax:] = False
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    if n == 0: return None, None
    sizes = ndimage.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes))
    if sizes[k] < 40: return None, None
    vs, us = np.nonzero(lab == k + 1)
    return np.array([us.mean(), vs.mean()]), int(sizes[k])

def grasp_point(npz, js):
    a = npz["actions"]; close = np.where(a[:, 6] > 0)[0]
    if not len(close): return None
    t = int(close[0]) + 1
    steps = js["trace"]["steps"]
    s = min(steps, key=lambda x: abs(x["t"] - t))
    return np.array(s["eef"])

def fit_h(U, X):
    A = []
    for (u, v), (x, y) in zip(U, X):
        A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x]); A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
    return np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)

def apply_h(H, U):
    h = np.c_[U, np.ones(len(U))] @ H.T
    return h[:, :2] / h[:, 2:]

def pairs(pattern="outputs/programs/none/libero_*/episode_*.npz"):
    U, X, src = [], [], []
    for f in sorted(glob.glob(pattern)):
        if any(s in f for s in ("_lan", "_swap", "object")): continue
        d = np.load(f)
        if not bool(d["success"]): continue
        try:
            js = json.load(open(f.replace(".npz", ".json")))
        except FileNotFoundError:
            continue
        g = grasp_point(d, js); u, size = footprint(d["agentview_image"][0], d["agentview_image"][-1])
        if g is None or u is None: continue
        U.append(u); X.append(g); src.append((f, size))
    return np.array(U), np.array(X), src

if __name__ == "__main__":
    U, X, src = pairs()
    print(len(U), "pairs")
    for (f, s), u, x in zip(src, U, X): print(f"  {f.split('none/')[1]:40} px {np.round(u,1)} size {s} grasp {np.round(x,3)}")
    rng = np.random.default_rng(0); best, bi = None, 0
    for _ in range(2000):
        s = rng.choice(len(U), 4, replace=False)
        H = fit_h(U[s], X[s, :2]); r = np.linalg.norm(apply_h(H, U) - X[:, :2], axis=1)
        inl = r < 0.03
        if inl.sum() > bi: bi, best = inl.sum(), inl
    H = fit_h(U[best], X[best, :2]); r = np.linalg.norm(apply_h(H, U) - X[:, :2], axis=1)
    print(f"inliers {best.sum()}/{len(U)}  median err inliers {np.median(r[best])*100:.1f} cm  all {np.median(r)*100:.1f} cm")
    np.save("library/knowledge/agentview_H.npy", H)
