"""Virtual frame for swapped layouts: make the frozen policy act in the frame where objects sit at their trained places.

Mechanism (JOURNAL E05, E08): the policy reads its current position from the proprio state and goes to memorised target
positions (swap: it reaches the trained spots). We find where the pick object and the place target moved, from the agentview
image only, and feed the policy eef - delta, so that the real object is where the policy expects it. Images are unchanged.
- reference: library/references/<trained suite>__<task>.npz, built from the frozen policy's own successful trained-suite
  episodes of the same instruction (first frame, eef at grasp = pick point, final eef = place point, object footprint).
- perception: the policy's own SigLIP patch features (steerlib.scene.patch_features); the pick object's reference patches
  (footprint, else the projected pick point) and the place target's (projected place point) are matched in the live frame.
- pixel -> world: the fixed agentview camera model (steerlib.scene.project/backproject; sensor calibration, checked against
  image evidence), back-projected on the horizontal plane at the reference grasp / place height; deltas are differences.
- phase: before the gripper first closes, delta_pick; while holding, delta_place (steerlib.gates.phase).
No simulator state, no layout knowledge: everything comes from images, proprio and the policy's own trained-suite behaviour.
"""
from pathlib import Path

import numpy as np

from steerlib import gates, scene

DESCRIPTION = "virtual frame: shift the proprio state by image-estimated object displacements (swap layouts)"
N_ACTION_STEPS = 10
REF = Path(__file__).resolve().parents[2] / "references"


def _twin(suite, task_id):
    for suffix in ("_swap", "_lan", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    base = _twin(ctx.task.suite, ctx.task.task_id)
    r = np.load(REF / f"{base[0]}__{base[1]}.npz", allow_pickle=False)
    ctx.memory["ref"] = {k: r[k] for k in ("image", "footprint", "pick", "place")}


def _shift(F_ref, F_live, point, exclude=None, radius=1):
    """Where did the reference content at world `point` move? Object region grown from the projected point (feature
    similarity), tracked by feature-template matching. Returns (world delta (3,), score, (ref px, live px), region size)."""
    u_ref = scene.project(point)
    m = scene.region(F_ref, u_ref)
    duv, s, s0 = scene.track(F_ref, m, F_live)
    u_live = u_ref + duv
    d = scene.backproject(u_live, point[2]) - scene.backproject(u_ref, point[2])
    d[2] = 0.0
    return d, s, (u_ref, u_live), int(m.sum())


def _estimate(obs, ctx):
    ref = ctx.memory["ref"]
    F_ref = scene.patch_features(ctx.vla, ref["image"])
    F_live = scene.patch_features(ctx.vla, obs["agentview_image"])
    d_pick, s_pick, u_pick, n_pick = _shift(F_ref, F_live, ref["pick"])
    out = {"pick": d_pick, "s_pick": s_pick, "u_pick": u_pick, "n_pick": n_pick}
    if np.all(np.isfinite(ref["place"])):
        d_place, s_place, u_place, n_place = _shift(F_ref, F_live, ref["place"])
        out.update(place=d_place, s_place=s_place, u_place=u_place, n_place=n_place)
    else:
        out["place"] = d_pick
    return out


def program(t, obs, internals, ctx):
    if "delta" not in ctx.memory:
        ctx.memory["delta"] = d = _estimate(obs, ctx)
        ctx.log("delta", {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items() if k in ("pick", "place", "s_pick", "s_place")})
        ctx.log("pixels", {k: [np.round(x, 1).tolist() for x in v] for k, v in d.items() if k.startswith("u_")})
    d = ctx.memory["delta"]
    ph = gates.phase(obs, ctx.memory)
    delta = d["pick"] if ph == "reach" else d["place"]
    obs["robot0_eef_pos"] = np.asarray(obs["robot0_eef_pos"]) - delta      # the policy sees the virtual frame
    return []
