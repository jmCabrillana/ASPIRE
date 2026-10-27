"""E23: which action-expert layers read the verb / object token? Block expert queries (q_len 50) from attending to the word's
key positions at ONE expert layer j (all denoising steps); effect = d(blocked, normal) / d(word deleted exactly, normal).
Word deleted exactly = pad-deletion in embed_prefix (E22b). Goal-scene trained instructions (general analysis, not a target)."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, attention as AT
policy, lab, _ = A.load(); model = policy.policy.model; m = model.vlm_with_expert
CASES = [("put the bowl on the stove", "put"), ("put the bowl on the stove", " bowl"), ("put the bowl on the stove", " stove"),
         ("open the middle drawer of the cabinet", "open"), ("put the wine bottle on the rack", " wine")]
obs = [lab.load(f"outputs/programs/none/libero_goal__1/episode_{e}.npz").obs(0) for e in (0, 1, 2)]
def deleted(o, ins, idx):
    orig = model.embed_prefix
    def w(images, img_masks, lang_tokens, lang_masks, state=None):
        lm = lang_masks.clone(); lm[:, idx] = False
        return orig(images, img_masks, lang_tokens, lang_masks=lm, state=state)
    model.embed_prefix = w
    try: return lab.chunk(o, ins)
    finally: del model.embed_prefix
print("expert layer:      " + " ".join(f"{j:>5}" for j in range(16)))
for ins, word in CASES:
    toks = A.tokens(ins); idx = [toks.index(word)]; cols = torch.tensor([128 + i for i in idx])
    eff = np.zeros(16)
    for o in obs:
        a0 = lab.chunk(o, ins); dref = A.dist(deleted(o, ins, idx), a0)
        for j in range(16):
            def bias(layer, q_len, k_len, j=j):
                if q_len != 50 or layer != j: return None
                b = torch.zeros(q_len, k_len); b[:, cols] = -1e9; return b
            r = AT.patch(m, bias)
            try: eff[j] += A.dist(lab.chunk(o, ins), a0) / dref / len(obs)
            finally: r()
    print(f"{ins[:22]:22} {word!r:9}" + " ".join(f"{e:5.2f}" for e in eff))
