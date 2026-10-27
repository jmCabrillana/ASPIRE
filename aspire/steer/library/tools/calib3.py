"""Camera self-calibration v3: gripper detector from the policy's own SigLIP patch features, seeded by 4 hand-annotated gripper
pixels (one-time sensor calibration, no task/layout knowledge), refined iteratively: detect the gripper in every saved frame
(argmax of prototype similarity among patches that changed vs frame 0), fit a 3x4 projection P with RANSAC on (eef, pixel),
re-seed the prototype from the inlier detections, repeat. Saves library/knowledge/agentview_P.npy."""
import sys, glob
sys.path[:0] = ['library', '.', 'library/tools']
import numpy as np
from steerlib import analysis as A, scene
from calib import dlt, project
policy, lab, _ = A.load()
SEED = [("libero_goal_lan__7", 0, 1, (178, 150)), ("libero_goal_lan__7", 0, 3, (178, 140)),
        ("libero_goal_lan__3", 1, 2, (90, 180)), ("libero_goal_swap__1", 0, 2, (210, 185))]
frames = []
for f in sorted(glob.glob("outputs/programs/*/*/episode_*.npz")):
    if "object" in f: continue
    d = np.load(f)
    for i in range(1, len(d["t"])):
        frames.append((f, i, d["agentview_image"][i], d["agentview_image"][0], d["robot0_eef_pos"][i]))
print(len(frames), "frames")
feat = {}
def F(k, img):
    if k not in feat: feat[k] = scene.patch_features(policy.policy, img)
    return feat[k]
protos = []
for t, e, i, px in SEED:
    d = np.load(f"outputs/programs/none/{t}/episode_{e}.npz")
    Fi = scene.patch_features(policy.policy, d["agentview_image"][i])
    protos.append(Fi[px[1] // 8, px[0] // 8])
proto = np.mean(protos, 0); proto /= np.linalg.norm(proto)
SX = np.array([np.load(f"outputs/programs/none/{t}/episode_{e}.npz")["robot0_eef_pos"][i] for t, e, i, _ in SEED])
SU = np.array([px for *_, px in SEED], float)
def valid(P, X):
    pr = project(P, X)
    return pr.std(0).min() > 15 and np.median(np.linalg.norm(project(P, SX) - SU, axis=1)) < 25
for it in range(3):
    X, U, keys = [], [], []
    for n, (f, i, img, ref, eef) in enumerate(frames):
        Fi = F(n, img)
        changed = (np.abs(img.astype(int) - ref.astype(int)).sum(-1) > 40).reshape(32, 8, 32, 8).mean((1, 3)) > 0.2
        s = np.where(changed, Fi @ proto, -1)
        r, c = np.unravel_index(int(np.argmax(s)), s.shape)
        X.append(eef); U.append([c * 8 + 4, r * 8 + 4]); keys.append((n, r, c))
    X, U = np.array(X), np.array(U, float)
    rng = np.random.default_rng(it); bi, best = 0, None
    for _ in range(20000):
        sidx = rng.choice(len(X), 6, replace=False)
        P = dlt(X[sidx], U[sidx])
        if not valid(P, X): continue
        res = np.linalg.norm(project(P, X) - U, axis=1)
        inl = res < 10
        if inl.sum() > bi: bi, best = inl.sum(), inl
    P = dlt(X[best], U[best]); res = np.linalg.norm(project(P, X) - U, axis=1)
    print(f"iter {it}: inliers {best.sum()}/{len(X)} median res inliers {np.median(res[best]):.1f}px")
    proto = np.mean([F(n, None)[r, c] for (n, r, c), b in zip(keys, best) if b], 0); proto /= np.linalg.norm(proto)
np.save("library/knowledge/agentview_P.npy", P)
for t, e, i, px in SEED:
    d = np.load(f"outputs/programs/none/{t}/episode_{e}.npz")
    print("seed", px, "->P", np.round(project(P, d["robot0_eef_pos"][i][None])[0], 1))
