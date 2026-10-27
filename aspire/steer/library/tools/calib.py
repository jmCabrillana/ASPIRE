"""Self-calibration of the agentview camera from saved episodes (no simulator internals).
Gripper pixel per frame: arm mask = low-saturation pixels that changed vs frame 0 of the same episode (static grey objects
cancel); the gripper is the mask pixel farthest from the robot base (bottom centre of the raw image). Then fit a 3x4
projection P (DLT + RANSAC) from (eef world position from proprio, gripper pixel). Saves library/knowledge/agentview_P.npy."""
import sys, glob
import numpy as np

def tip(img, ref, base=(128, 250)):
    im, rf = img.astype(float), ref.astype(float)
    sat = im.max(-1) - im.min(-1)
    changed = np.abs(im - rf).sum(-1) > 60
    mask = changed & (sat < 25) & (im.mean(-1) > 40)
    mask[200:] = False                                       # floor area around the base
    from scipy import ndimage
    lab, n = ndimage.label(ndimage.binary_opening(mask, iterations=1))
    if n == 0: return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    comp = lab == (1 + int(np.argmax(sizes)))
    vs, us = np.nonzero(comp)
    if len(vs) < 200: return None
    top = vs <= np.percentile(vs, 1.5)                       # the arm rises from the bottom: its top edge is the gripper
    return np.array([us[top].mean(), vs[top].mean()])

def dlt(X, u):
    A = []
    for (x, y, z), (uu, vv) in zip(X, u):
        A.append([x, y, z, 1, 0, 0, 0, 0, -uu * x, -uu * y, -uu * z, -uu])
        A.append([0, 0, 0, 0, x, y, z, 1, -vv * x, -vv * y, -vv * z, -vv])
    return np.linalg.svd(np.array(A))[2][-1].reshape(3, 4)

def project(P, X):
    h = np.c_[X, np.ones(len(X))] @ P.T
    return h[:, :2] / h[:, 2:]

if __name__ == "__main__":
    X, U = [], []
    for f in sorted(glob.glob("outputs/programs/*/*/episode_*.npz")):
        d = np.load(f)
        if "swap" in f: continue
        for i in range(1, len(d["t"])):
            p = tip(d["agentview_image"][i], d["agentview_image"][0])
            if p is not None:
                X.append(d["robot0_eef_pos"][i]); U.append(p)
    X, U = np.array(X), np.array(U)
    print(len(X), "pairs")
    rng = np.random.default_rng(0); best, bi = None, 0
    for _ in range(3000):
        s = rng.choice(len(X), 6, replace=False)
        P = dlt(X[s], U[s]); r = np.linalg.norm(project(P, X) - U, axis=1)
        inl = r < 6
        if inl.sum() > bi: bi, best = inl.sum(), inl
    P = dlt(X[best], U[best]); r = np.linalg.norm(project(P, X) - U, axis=1)
    print(f"inliers {best.sum()}/{len(X)}, median residual inliers {np.median(r[best]):.2f}px, all {np.median(r):.2f}px")
    np.save("library/knowledge/agentview_P.npy", P)
