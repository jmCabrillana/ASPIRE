"""E19: how many MLP neurons does it take to turn an unknown verb into a known one? At VLM layer 1, the needed correction
at the verb token is dv = h_out(rewrite) - h_out(paraphrase) (mean over 2 fit observations). Greedy orthogonal matching
pursuit over the columns of W_down (one column = one neuron's output direction) chooses k neurons and their amounts;
push them at the verb position only and measure R on 4 new observations. Also: are these neurons verb-selective?"""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, residual as R, lexedit as LX
policy, lab, _ = A.load(); vla = policy.policy
W = vla.model.vlm_with_expert.get_vlm_model().text_model.layers[1].mlp.down_proj.weight.detach().float().cpu()   # (960, 2560)
eps = [lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz") for e in range(4)]
FIT = [eps[0].obs(0), eps[1].obs(1)]; TEST = [eps[2].obs(0), eps[2].obs(2), eps[3].obs(0), eps[3].obs(3)]
def omp(dv, k):
    S, res = [], dv.clone(); Wn = W / W.norm(dim=0, keepdim=True)
    for _ in range(k):
        S.append(int((Wn.T @ res).abs().argmax()))
        a = torch.linalg.lstsq(W[:, S], dv[:, None]).solution[:, 0]
        res = dv - W[:, S] @ a
    return S, a, float(res.norm() / dv.norm())
for x, c in [("switch on the stove", "turn on the stove"), ("place the bowl on the stove", "put the bowl on the stove")]:
    pos = [p for p, _ in LX.align(x, c)]
    dvs = []
    for o in FIT:
        with R.record(vla, [1]) as rc: lab.chunk(o, c)
        with R.record(vla, [1]) as rx: lab.chunk(o, x)
        dvs.append(rc[1][pos[0]] - rx[1][pos[0]])
    dv = torch.stack(dvs).mean(0)
    print(f"\n{x!r}: |dv| {dv.norm():.1f}")
    for k in (1, 4, 16, 64, 256):
        S, a, rel = omp(dv, k)
        vec = (W[:, S] @ a)[None]
        r = []
        for o in TEST:
            ac, ax = lab.chunk(o, c), lab.chunk(o, x)
            with R.add(vla, {1: (pos[:1], vec)}):
                r.append(1 - A.dist(lab.chunk(o, x), ac) / A.dist(ax, ac))
        print(f"  k={k:3d} residual {rel:.2f}  R {np.mean(r):.2f}  top neurons {S[:6]} amounts {np.round(a[:6].numpy(), 2)}")
