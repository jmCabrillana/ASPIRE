"""E16: lexical weight edits (steerlib.lexedit) fitted on 2 observations, tested on 4 others (goal scene).
R = 1 - d(edited x, c) / d(x, c); upper bound = patching the verb token at layer L with c's residual (E15)."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, lexedit as LX, residual as R
policy, lab, _ = A.load(); vla = policy.policy
PAIRS = [("place the bowl on the stove", "put the bowl on the stove"), ("switch on the stove", "turn on the stove"),
         ("place the wine bottle on the rack", "put the wine bottle on the rack"),
         ("place the cream cheese inside the bowl", "put the cream cheese in the bowl")]
eps = [lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz") for e in range(4)]
FIT = [eps[0].obs(0), eps[1].obs(1)]
TEST = [eps[2].obs(0), eps[2].obs(2), eps[3].obs(0), eps[3].obs(3)]
for x, c in PAIRS:
    for L in (0, 1, 2):
        delta = LX.fit(lab, FIT, x, c, L=L)
        r_edit, r_patch, other = [], [], []
        for o in TEST:
            ac, ax = lab.chunk(o, c), lab.chunk(o, x); d0 = A.dist(ax, ac)
            with LX.applied(vla, delta, L):
                ae = lab.chunk(o, x)
                # side effect on an unrelated trained instruction (should stay unchanged)
                ao_e = lab.chunk(o, "put the bowl on the plate")
            ao = lab.chunk(o, "put the bowl on the plate")
            r_edit.append(1 - A.dist(ae, ac) / d0); other.append(A.dist(ao_e, ao))
            with R.record(vla, [L]) as rc:
                lab.chunk(o, c)
            pairs = LX.align(x, c)
            with R.patch(vla, {L: ([p for p, _ in pairs], rc[L][[q for _, q in pairs]])}):
                r_patch.append(1 - A.dist(lab.chunk(o, x), ac) / d0)
        print(f"{x[:34]:34} L{L}: edit R {np.mean(r_edit):.2f} (per obs {np.round(r_edit, 2)})  patch R {np.mean(r_patch):.2f}"
              f"  side effect on 'put the bowl on the plate' {np.mean(other):.2f} cm  |delta| {delta.norm():.2f}")
