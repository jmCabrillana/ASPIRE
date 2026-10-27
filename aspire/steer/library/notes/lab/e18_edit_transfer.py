"""E18: does a lexical edit learned on one sentence transfer to other sentences with the same unknown word?
Fit 'place'->'put' at VLM layer 1 on ONE source pair (2 obs), test on other target pairs (4 new obs), leave-one-out."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, lexedit as LX
policy, lab, _ = A.load(); vla = policy.policy
P = {1: ("place the bowl on the stove", "put the bowl on the stove"), 2: ("place the wine bottle on top of the cabinet", "put the wine bottle on top of the cabinet"),
     4: ("place the bowl on top of the cabinet", "put the bowl on top of the cabinet"), 6: ("place the cream cheese inside the bowl", "put the cream cheese in the bowl"),
     9: ("place the wine bottle on the rack", "put the wine bottle on the rack")}
eps = [lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz") for e in range(4)]
FIT = [eps[0].obs(0), eps[1].obs(1)]
TEST = [eps[2].obs(0), eps[2].obs(2), eps[3].obs(0), eps[3].obs(3)]
def R_of(delta, x, c, L=1):
    r = []
    for o in TEST:
        ac, ax = lab.chunk(o, c), lab.chunk(o, x)
        with LX.applied(vla, delta, L):
            ae = lab.chunk(o, x)
        r.append(1 - A.dist(ae, ac) / A.dist(ax, ac))
    return np.mean(r)
def first_token_only(x, c):        # fit only the 'place' position (the shared fact), not 'inside'
    return x, c.split()[0] + " " + " ".join(x.split()[1:])
for src in P:
    xs, cs = first_token_only(*P[src])
    delta = LX.fit(lab, FIT, xs, cs, L=1)
    row = {t: R_of(delta, *first_token_only(*P[t])) for t in P if t != src}
    print(f"fit on lan{src}: " + "  ".join(f"lan{t} {r:.2f}" for t, r in row.items()))
# pooled: one edit from all other pairs (leave target out)
for t in P:
    K, V = [], []
    for s in P:
        if s == t: continue
        k, v = LX.collect(lab, FIT, *first_token_only(*P[s]), 1); K.append(k); V.append(v)
    from steer import edit as E
    W = vla.model.vlm_with_expert.get_vlm_model().text_model.layers[1].mlp.down_proj.weight
    delta = E.multi(W, torch.cat(K).T, torch.cat(V).T, ridge=1e-3)
    print(f"pooled (others) -> lan{t}: R {R_of(delta, *first_token_only(*P[t])):.2f}")
