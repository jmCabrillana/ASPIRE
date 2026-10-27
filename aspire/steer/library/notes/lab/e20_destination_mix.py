"""E20: is the destination a continuous function of the destination-noun representation? Goal scene, bowl already held
(chunk 1 of trained goal 1 / goal 8 episodes). Patch the destination token's residual (all VLM layers) with
(1-a) r(plate) + a r(stove) and read the predicted end of the chunk (eef + path end, 0.0095 m/unit)."""
import sys, glob; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A, residual as R
policy, lab, _ = A.load(); vla = policy.policy
XA, XB = "put the bowl on the plate", "put the bowl on the stove"
pos = [128 + A.tokens(XA).index(" plate")]
assert A.tokens(XA)[:-2] == A.tokens(XB)[:-2]
for f in sorted(glob.glob("outputs/programs/none/libero_goal__[18]/episode_*.npz"))[:6]:
    ep = lab.load(f)
    if len(ep) < 2: continue
    o = ep.obs(1)
    if o["robot0_gripper_qpos"][0] > 0.03: continue          # not holding
    with R.record(vla, range(16)) as ra: lab.chunk(o, XA)
    with R.record(vla, range(16)) as rb: lab.chunk(o, XB)
    ends = []
    for a in (0, 0.25, 0.5, 0.75, 1.0):
        spec = {l: (pos, (1 - a) * ra[l][pos] + a * rb[l][pos]) for l in range(16)}
        with R.patch(vla, spec):
            ch = lab.chunk(o, XA)
        ends.append(o["robot0_eef_pos"][:2] + A.path(ch)[-1][:2])
    e = np.array(ends) * 100
    pa, pb = e[0], e[-1]
    t = [float(np.dot(x - pa, pb - pa) / np.dot(pb - pa, pb - pa)) for x in e]
    off = [float(np.linalg.norm((x - pa) - tt * (pb - pa))) for x, tt in zip(e, t)]
    print(f.split('none/')[1][:28], "ends (cm)", np.round(e, 1).tolist(), "| position along plate->stove", np.round(t, 2).tolist(), "| off-line", np.round(off, 1).tolist())
