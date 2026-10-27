"""E22b: exact in-network deletion: mark the deleted tokens as padding in embed_prefix (attention mask AND rotary positions
follow, since position ids = cumsum(pad mask) - 1). Same pair as E22."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A
policy, lab, _ = A.load(); model = policy.policy.model
x, c = "put the wine bottle on top of the rack", "put the wine bottle on the rack"
toks = A.tokens(x); drop = [toks.index(" top"), toks.index(" of")]
orig = model.embed_prefix
def wrapped(images, img_masks, lang_tokens, lang_masks, state=None):
    lang_masks = lang_masks.clone(); lang_masks[:, drop] = False
    return orig(images, img_masks, lang_tokens, lang_masks, state=state)
obs = [lab.load(f"outputs/programs/none/libero_goal__9/episode_{e}.npz").obs(i) for e in (0, 1) for i in (0, 1)]
r = []
for o in obs:
    ac, ax = lab.chunk(o, c), lab.chunk(o, x)
    model.embed_prefix = wrapped
    try: am = lab.chunk(o, x)
    finally: del model.embed_prefix
    r.append(1 - A.dist(am, ac) / A.dist(ax, ac))
print("pad-deletion R", np.round(r, 3), "mean", round(float(np.mean(r)), 3))
