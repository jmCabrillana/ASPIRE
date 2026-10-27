"""E22: does deleting tokens by attention masking (all queries, all layers) behave like removing the words from the text?
Pair (analysis for lan8 uses another task): x = 'put the wine bottle on top of the rack' masked at ' top',' of' vs
c = 'put the wine bottle on the rack' (goal 9). R = 1 - d(masked x, c)/d(x, c). Also mask only in the VLM or only in the expert."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, attention as AT
policy, lab, _ = A.load(); m = policy.policy.model.vlm_with_expert
x, c = "put the wine bottle on top of the rack", "put the wine bottle on the rack"
toks = A.tokens(x); cols = torch.tensor([128 + toks.index(" top"), 128 + toks.index(" of")])
def bias_for(where):
    def f(layer, q_len, k_len):
        is_expert = q_len == 50
        if (where == "vlm" and is_expert) or (where == "expert" and not is_expert): return None
        b = torch.zeros(q_len, k_len); b[:, cols] = -1e9; return b
    return f
obs = [lab.load(f"outputs/programs/none/libero_goal__9/episode_{e}.npz").obs(i) for e in (0, 1) for i in (0, 1)]
for where in ("all", "vlm", "expert"):
    r = []
    for o in obs:
        ac, ax = lab.chunk(o, c), lab.chunk(o, x)
        rs = AT.patch(m, bias_for(where))
        try: am = lab.chunk(o, x)
        finally: rs()
        r.append(1 - A.dist(am, ac) / A.dist(ax, ac))
    print(where, "R", np.round(r, 2), "mean", round(float(np.mean(r)), 2))
