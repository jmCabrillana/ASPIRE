"""E32: which HEADS of expert layer 11 (the main reader of the destination 'stove', E23) read it? Knock out one head's
attention to the 'stove' key at expert layer 11 (all flow steps); effect = d(knocked, normal) / d(stove deleted exactly, normal).
Also all heads of L11 together, and the top-2 heads at L7+L11."""
import sys, glob; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, attention as AT
policy, lab, _ = A.load(); model = policy.policy.model; m = model.vlm_with_expert
ins = "put the bowl on the stove"; toks = A.tokens(ins); col = 128 + toks.index(" stove")
obs = [lab.load(f).obs(1) for f in sorted(glob.glob("outputs/programs/none/libero_goal__1/episode_*.npz"))[:4]]
def deleted(o):
    orig = model.embed_prefix
    def w(images, img_masks, lang_tokens, lang_masks, state=None):
        lm = lang_masks.clone(); lm[:, toks.index(" stove")] = False
        return orig(images, img_masks, lang_tokens, lang_masks=lm, state=state)
    model.embed_prefix = w
    try: return lab.chunk(o, ins)
    finally: del model.embed_prefix
def knock(o, spec):     # spec: {layer: [heads]}
    def bias(layer, q_len, k_len):
        if q_len != 50 or layer not in spec: return None
        b = torch.zeros(15, q_len, k_len); b[spec[layer], :, col] = -1e9; return b
    r = AT.patch(m, bias)
    try: return lab.chunk(o, ins)
    finally: r()
eff = np.zeros(15); allL11 = []; refs = []
for o in obs:
    a0 = lab.chunk(o, ins); dref = A.dist(deleted(o), a0); refs.append(dref)
    for h in range(15): eff[h] += A.dist(knock(o, {11: [h]}), a0) / dref / len(obs)
    allL11.append(A.dist(knock(o, {11: list(range(15))}), a0) / dref)
order = np.argsort(-eff)
print("per-head effect at expert L11:", " ".join(f"h{h}:{eff[h]:.2f}" for h in order))
print("all 15 heads of L11:", round(float(np.mean(allL11)), 2), "| sum of single heads:", round(float(eff.sum()), 2))
