"""E31: do two-layer lexical edits close the gap to activation patching? Sequential fit: layer A edit applied, then layer B
edit fitted on top (keys/values collected with A applied). Same pairs/obs as E16 (fit 2 obs, test 4)."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, lexedit as LX
policy, lab, _ = A.load(); vla = policy.policy
PAIRS = [("place the bowl on the stove", "put the bowl on the stove"), ("switch on the stove", "turn on the stove"),
         ("place the wine bottle on the rack", "put the wine bottle on the rack"),
         ("place the cream cheese inside the bowl", "put the cream cheese in the bowl")]
eps = [lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz") for e in range(4)]
FIT = [eps[0].obs(0), eps[1].obs(1)]; TEST = [eps[2].obs(0), eps[2].obs(2), eps[3].obs(0), eps[3].obs(3)]
def evalR(edits, x, c):
    r, side = [], []
    for o in TEST:
        ac, ax, ao = lab.chunk(o, c), lab.chunk(o, x), lab.chunk(o, "put the bowl on the plate")
        ctxs = [LX.applied(vla, d, L) for L, d in edits]
        for cm in ctxs: cm.__enter__()
        try: ae, aoe = lab.chunk(o, x), lab.chunk(o, "put the bowl on the plate")
        finally:
            for cm in reversed(ctxs): cm.__exit__()
        r.append(1 - A.dist(ae, ac) / A.dist(ax, ac)); side.append(A.dist(aoe, ao))
    return np.mean(r), np.mean(side)
for x, c in PAIRS:
    row = []
    for layers in ([1], [0, 1], [1, 2], [1, 3], [0, 1, 2]):
        edits = []
        for L in layers:
            ctxs = [LX.applied(vla, d, l) for l, d in edits]
            for cm in ctxs: cm.__enter__()
            try: d = LX.fit(lab, FIT, x, c, L=L)
            finally:
                for cm in reversed(ctxs): cm.__exit__()
            edits.append((L, d))
        R_, s_ = evalR(edits, x, c); row.append(f"L{'+'.join(map(str, layers))}: R {R_:.2f} side {s_:.2f}")
    print(f"{x[:30]:30} " + " | ".join(row))
