"""E15: where does the verb reach the action expert? Activation patching between a paraphrase x and its general-rule rewrite c
(canon; equal token counts; goal scene). R = 1 - d(patched, c) / d(x, c) (fraction of the rewrite's behaviour restored).
Variants per layer L: verb@L (differing tokens, layer L only), verb<=L (differing tokens, layers 0..L),
others@L (all other instruction tokens at L), lang@L (all instruction tokens at L), state@L (state token at L)."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, residual as R
policy, lab, _ = A.load(); vla = policy.policy
PAIRS = [("place the bowl on the stove", "put the bowl on the stove"), ("switch on the stove", "turn on the stove"),
         ("place the wine bottle on the rack", "put the wine bottle on the rack"),
         ("place the cream cheese inside the bowl", "put the cream cheese in the bowl"),
         ("place the bowl on top of the cabinet", "put the bowl on top of the cabinet")]
OBS = [lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz").obs(i) for e in (0, 1) for i in (0, 1)]
LAYERS = list(range(16))
res = {}
for x, c in PAIRS:
    tx, tc = A.tokens(x), A.tokens(c)
    assert len(tx) == len(tc), (tx, tc)
    diff = [128 + i for i, (a, b) in enumerate(zip(tx, tc)) if a != b]
    same = [128 + i for i, (a, b) in enumerate(zip(tx, tc)) if a == b]
    lang = list(range(128, 128 + len(tx)))
    for o in OBS:
        with R.record(vla, LAYERS) as rc:
            ac = lab.chunk(o, c)
        ax = lab.chunk(o, x); dxc = A.dist(ax, ac)
        def R_(spec):
            with R.patch(vla, spec):
                ap = lab.chunk(o, x)
            return 1 - A.dist(ap, ac) / dxc
        for L in LAYERS:
            for name, spec in [("verb@L", {L: (diff, rc[L][diff])}),
                               ("verb<=L", {l: (diff, rc[l][diff]) for l in range(L + 1)}),
                               ("others@L", {L: (same, rc[L][same])}),
                               ("lang@L", {L: (lang, rc[L][lang])}),
                               ("state@L", {L: ([176], rc[L][[176]])})]:
                res.setdefault((name, L), []).append(R_(spec))
        res.setdefault(("d(x,c) cm", 0), []).append(dxc)
print("d(x,c) mean cm", np.round(np.mean(res[("d(x,c) cm", 0)]), 1), "per pair/obs", np.round(res[("d(x,c) cm", 0)], 1))
print("layer  " + "  ".join(f"{n:>8}" for n in ("verb@L", "verb<=L", "others@L", "lang@L", "state@L")))
for L in LAYERS:
    print(f"L{L:<5} " + "  ".join(f"{np.mean(res[(n, L)]):8.2f}" for n in ("verb@L", "verb<=L", "others@L", "lang@L", "state@L")))
