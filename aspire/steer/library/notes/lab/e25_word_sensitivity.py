"""E25: which words does the policy rely on? For trained goal + spatial instructions (all tasks; general analysis), delete each
token exactly (pad-deletion, E22b) and measure the chunk change d(deleted, normal) in units of the flow-noise floor
(mean d between two noise seeds). Aggregate by word class."""
import sys, json, glob; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A
policy, lab, _ = A.load(); model = policy.policy.model
tr = json.load(open('library/notes/lab/trained_instructions.json'))
CLS = {"put": "verb", "pick": "verb", "open": "verb", "turn": "verb", "push": "verb", "place": "verb", "up": "particle",
       "the": "article", "it": "pronoun", "and": "conj", "on": "prep", "in": "prep", "of": "prep", "to": "prep", "from": "prep",
       "next": "prep", "between": "prep", "inside": "prep", "top": "prep", "front": "noun", "black": "adj", "middle": "adj",
       "wooden": "adj", "\n": "newline"}
def deleted_chunk(o, ins, idx):
    orig = model.embed_prefix
    def w(images, img_masks, lang_tokens, lang_masks, state=None):
        lm = lang_masks.clone(); lm[:, idx] = False
        return orig(images, img_masks, lang_tokens, lang_masks=lm, state=state)
    model.embed_prefix = w
    try: return lab.chunk(o, ins)
    finally: del model.embed_prefix
res = {}
for suite in ("libero_goal", "libero_spatial"):
    for t, ins in enumerate(tr[suite]):
        fs = sorted(glob.glob(f"outputs/programs/none/{suite}__{t}/episode_*.npz"))
        if not fs: continue
        o = lab.load(fs[0]).obs(0)
        a0, a1 = lab.chunk(o, ins), lab.chunk(o, ins, seed=1); noise = max(A.dist(a0, a1), 1e-3)
        toks = A.tokens(ins)
        for i, tk in enumerate(toks):
            w = tk.strip() or "\n"
            c = CLS.get(w, "noun")
            res.setdefault(c, []).append(A.dist(deleted_chunk(o, ins, [i]), a0) / noise)
            res.setdefault(f"  word:{w}", []).append(res[c][-1])
for k in sorted(res, key=lambda k: (k.startswith(" "), -np.mean(res[k]))):
    if not k.startswith(" ") or len(res[k]) >= 3:
        print(f"{k:22} n={len(res[k]):3d}  effect {np.mean(res[k]):6.2f} x noise  (median {np.median(res[k]):.2f})")
