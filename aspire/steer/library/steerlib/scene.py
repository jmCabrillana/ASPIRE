"""Image-only scene perception for the agentview camera (no simulator internals): table masks, object footprints from the
policy's own episodes, change blobs between a reference image and the live image, colour-histogram matching, and the
self-calibrated pixel -> table (x, y) homography (library/knowledge/agentview_H.npy, fitted by tools/calib2.py).
All images are RAW agentview frames (as saved by run_program / given to programs in obs), 256x256, robot base at the bottom."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

H_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "agentview_H.npy"


def table(img):
    """Table-coloured pixels (beige wood): R > G > B, saturated, bright."""
    im = img.astype(int); r, g, b = im[..., 0], im[..., 1], im[..., 2]
    return (r > g) & (g > b) & (r - b > 25) & (im.mean(-1) > 110)


def largest(mask, min_size=30):
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1)); k = int(np.argmax(sizes))
    return (lab == k + 1) if sizes[k] >= min_size else None


def footprint(first, last, vmax=165):
    """Pixels of an object that was carried away: not table in `first`, table in `last` (largest component)."""
    m = ~table(first) & table(last)
    m[vmax:] = False
    return largest(ndimage.binary_opening(m, iterations=2), 40)


def blobs(ref, live, vmax=200, thr=60, min_size=40):
    """Connected regions where the live image differs from the reference and is not table (objects that appeared there)."""
    m = (np.abs(live.astype(int) - ref.astype(int)).sum(-1) > thr) & ~table(live)
    m[vmax:] = False
    m = ndimage.binary_opening(m, iterations=2)
    lab, n = ndimage.label(m)
    return [lab == k for k in range(1, n + 1) if (lab == k).sum() >= min_size]


def hist(img, mask, bins=8):
    px = img[mask].astype(int) // (256 // bins)
    h = np.bincount(px[:, 0] * bins * bins + px[:, 1] * bins + px[:, 2], minlength=bins ** 3).astype(float)
    return h / max(h.sum(), 1)


def similarity(h1, h2):
    return float(np.sqrt(h1 * h2).sum())          # Bhattacharyya coefficient (1 = identical colour distributions)


def centroid(mask):
    vs, us = np.nonzero(mask)
    return np.array([us.mean(), vs.mean()])


def find_moved(ref, live, obj_mask):
    """Where did the object at `obj_mask` (in ref) go in `live`? Returns (pixel, score, n_blobs)."""
    h = hist(ref, obj_mask & ~table(ref))
    best, bs, bl = None, -1.0, blobs(ref, live)
    for b in bl:
        s = similarity(h, hist(live, b))
        if s > bs:
            best, bs = b, s
    return (None if best is None else centroid(best)), bs, len(bl)


def to_table(px, H=None):
    """Pixel(s) (u, v) of the raw agentview image -> world (x, y) on the table plane (self-calibrated homography)."""
    H = np.load(H_PATH) if H is None else H
    p = np.atleast_2d(px)
    h = np.c_[p, np.ones(len(p))] @ H.T
    out = h[:, :2] / h[:, 2:]
    return out[0] if np.ndim(px) == 1 else out


def patch_features(vla, img) -> np.ndarray:
    """The policy's own SigLIP patch features of a RAW agentview frame: (32, 32, 768), unit-normalised, in RAW image orientation
    (the policy sees the image rotated 180 degrees and resized to 512; patch (r, c) covers raw pixels v 8r..8r+7, u 8c..8c+7).
    E08: matching the mean feature of an object's patches finds it after it moved (goal_swap bowl, 5/5 frames)."""
    import torch
    import torch.nn.functional as F
    vm = vla.model.vlm_with_expert.get_vlm_model().vision_model
    x = torch.from_numpy(np.ascontiguousarray(img[::-1, ::-1])).permute(2, 0, 1).float().div(255)[None]
    x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False) * 2 - 1
    with torch.no_grad():
        f = vm(pixel_values=x.to(next(vm.parameters()).device, dtype=vm.dtype)).last_hidden_state[0].float().cpu().numpy()
    f = f.reshape(32, 32, -1)[::-1, ::-1]
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


def patch_mask(mask) -> np.ndarray:
    """(256, 256) pixel mask -> (32, 32) patch mask (patches at least 30% covered)."""
    return mask.reshape(32, 8, 32, 8).mean((1, 3)) > 0.3


def disc_mask(px, radius_patches=1):
    """(32, 32) patch mask of the patches within `radius_patches` of pixel px = (u, v)."""
    c, r = int(px[0]) // 8, int(px[1]) // 8
    m = np.zeros((32, 32), bool)
    m[max(0, r - radius_patches):r + radius_patches + 1, max(0, c - radius_patches):c + radius_patches + 1] = True
    return m


def locate(F_ref, ref_patches, F_live, exclude=None):
    """Pixel (u, v) in the live frame whose patch best matches the mean reference feature over `ref_patches` (32x32 bool).
    Returns (pixel, similarity)."""
    proto = F_ref[ref_patches].mean(0)
    proto = proto / np.linalg.norm(proto)
    s = F_live @ proto
    if exclude is not None:
        s = np.where(exclude, -1, s)
    r, c = np.unravel_index(int(np.argmax(s)), s.shape)
    return np.array([c * 8 + 4.0, r * 8 + 4.0]), float(s[r, c])


def to_pixel(xy, H=None):
    """World (x, y) on the table -> raw agentview pixel (inverse homography)."""
    H = np.load(H_PATH) if H is None else H
    h = np.linalg.inv(H) @ np.array([xy[0], xy[1], 1.0])
    return h[:2] / h[2]


# ---- agentview camera model ---------------------------------------------------------------------------------------------
# Sensor calibration of the fixed agentview camera of the LIBERO tabletop scenes (pose as declared for the camera in the
# LIBERO scene setup, fovy 45 = MuJoCo default, 256x256; raw obs images are vertically flipped). Not task or layout knowledge:
# the same for every task. Checked against image evidence: hand-annotated gripper pixels (u within 4 px) and footprints of
# objects the frozen policy grasped in trained episodes (5-15 px, the offset being object base vs top). JOURNAL 18:20.
_CAM_POS = np.array([0.6586131746834771, 0.0, 1.6103500240372423])
_CAM_QUAT_WXYZ = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)


def _camera():
    from scipy.spatial.transform import Rotation
    w, x, y, z = _CAM_QUAT_WXYZ
    R = Rotation.from_quat([x, y, z, w]).as_matrix() @ np.diag([1, -1, -1])     # MuJoCo camera frame -> OpenCV
    f = 0.5 * 256 / np.tan(np.radians(45) / 2)
    K = np.array([[f, 0, 128], [0, f, 128], [0, 0, 1.0]])
    return K, R


def project(X):
    """World point(s) (x, y, z) -> raw agentview pixel(s) (u, v)."""
    K, R = _camera()
    X = np.atleast_2d(X)
    c = (X - _CAM_POS) @ R                       # camera coordinates (R columns are the camera axes in world)
    uv = (c @ K.T)[:, :2] / c[:, 2:3]
    uv[:, 1] = 255 - uv[:, 1]
    return uv[0] if len(uv) == 1 else uv


def backproject(px, z):
    """Raw agentview pixel (u, v) -> world point on the horizontal plane at height z."""
    K, R = _camera()
    u, v = float(px[0]), 255 - float(px[1])
    d = R @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))          # ray direction in world
    t = (z - _CAM_POS[2]) / d[2]
    return _CAM_POS + t * d


def region(F, px, tau=0.55, radius=5):
    """Object region around pixel px in a feature map: patches connected to px's patch with cosine similarity > tau to it,
    within `radius` patches. Returns a (32, 32) bool mask."""
    r0, c0 = int(px[1]) // 8, int(px[0]) // 8
    r0, c0 = min(max(r0, 0), 31), min(max(c0, 0), 31)
    sim = F @ F[r0, c0]
    m = sim > tau
    rr, cc = np.ogrid[:32, :32]
    m &= (np.abs(rr - r0) <= radius) & (np.abs(cc - c0) <= radius)
    lab, _ = ndimage.label(m)
    return lab == lab[r0, c0]


def track(F_ref, mask, F_live, max_shift=14):
    """Feature-template matching: the patch shift (dr, dc) maximising the mean cosine similarity between the reference features
    under `mask` and the live features at the shifted positions. Returns ((du, dv) pixel shift, score, score at zero shift)."""
    rs, cs = np.nonzero(mask)
    best, arg, zero = -9.0, (0, 0), None
    for dr in range(-max_shift, max_shift + 1):
        for dc in range(-max_shift, max_shift + 1):
            r2, c2 = rs + dr, cs + dc
            ok = (r2 >= 0) & (r2 < 32) & (c2 >= 0) & (c2 < 32)
            if ok.mean() < 0.7:
                continue
            s = float((F_ref[rs[ok], cs[ok]] * F_live[r2[ok], c2[ok]]).sum(-1).mean())
            if dr == 0 and dc == 0:
                zero = s
            if s > best:
                best, arg = s, (dr, dc)
    return np.array([arg[1] * 8.0, arg[0] * 8.0]), best, zero
