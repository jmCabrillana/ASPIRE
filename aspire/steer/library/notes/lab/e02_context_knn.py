"""E02: contextual nearest neighbours of paraphrase tokens among trained-instruction tokens, per VLM layer (residual stream).
Bank: all trained instructions of spatial/object/goal/10 except the target's twin (libero_goal task = target id), run on the
same observation."""
import sys, json; sys.path[:0] = ['library', '.']
import torch, numpy as np
from steerlib import analysis as A
from steer.hooks import layers
policy, lab, _ = A.load()
LAN = {0: "open the cabinet’s middle drawer", 1: "place the bowl on the stove", 3: "slide open the top drawer and place the bowl in it", 5: "move the plate to the stove’s front", 6: "place the cream cheese inside the bowl", 7: "switch on the stove", 8: "place the bowl on top of the plate", 9: "place the wine bottle on the rack"}
tr = json.load(open('library/notes/lab/trained_instructions.json'))
L = layers(policy, "vlm")

def residuals(obs, ins):
    """(17, n_lang, 960): residual stream entering each layer 0..15 and leaving layer 15, at the instruction tokens."""
    got = []
    hs = [l.input_layernorm.register_forward_pre_hook(lambda m, a: got.append(a[0][0].detach().float().cpu())) for l in L]
    norm = policy.policy.model.vlm_with_expert.get_vlm_model().text_model.norm
    hs.append(norm.register_forward_pre_hook(lambda m, a: got.append(a[0][0].detach().float().cpu())))
    try:
        lab.chunk(obs, ins)
    finally:
        for h in hs: h.remove()
    sp = A.spans(ins)["lang"]
    return torch.stack([g[sp] for g in got[:17]])     # prefix pass only (first 17 calls)

def run(task, ep, layers_show=(0, 2, 4, 6, 8, 10, 12, 14, 16)):
    e = lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{ep}.npz"); e.instruction = LAN[task]
    obs = e.obs(0)
    bank_txt, bank_rep = [], []
    for s, l in tr.items():
        for i, ins in enumerate(l):
            if s == "libero_goal" and i == task: continue
            r = residuals(obs, ins); toks = A.tokens(ins)
            for j, t in enumerate(toks):
                bank_txt.append((t, ins)); bank_rep.append(r[:, j])
    B = torch.stack(bank_rep, 1)                          # (17, N, 960)
    Bn = B / B.norm(dim=-1, keepdim=True)
    q = residuals(obs, e.instruction); qn = q / q.norm(dim=-1, keepdim=True)
    toks = A.tokens(e.instruction)
    print(f"\n== goal_lan {task}: '{e.instruction}'")
    for j, t in enumerate(toks):
        row = []
        for l in layers_show:
            sim = Bn[l] @ qn[l, j]; k = int(sim.argmax())
            row.append(f"L{l}:{bank_txt[k][0].strip() or repr(bank_txt[k][0])}({sim[k]:.2f})")
        print(f"  {t!r:12} " + " ".join(row))

for task in [int(x) for x in sys.argv[1].split(',')]:
    run(task, 0)
