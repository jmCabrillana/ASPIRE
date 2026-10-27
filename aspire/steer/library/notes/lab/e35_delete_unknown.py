"""E35: is an unknown word harmful (interference) or just uninformative? Exact deletion (pad-mask) of the unknown word(s) vs the
rewrite: R_delete = 1 - d(deleted x, c) / d(x, c) (1 = deleting is as good as the right word). Goal scene, 4 obs."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A, lexedit as LX
policy, lab, _ = A.load(); model = policy.policy.model
PAIRS = [("switch on the stove", "turn on the stove"), ("place the bowl on the stove", "put the bowl on the stove"),
         ("place the wine bottle on the rack", "put the wine bottle on the rack"), ("set the bowl down on the stove", "put the bowl down on the stove"),
         ("move the bowl onto the plate", "put the bowl on the plate")]
obs = [lab.load(f"outputs/programs/none/libero_goal_lan__7/episode_{e}.npz").obs(i) for e in (2, 3) for i in (0, 2)]
def deleted(o, ins, idx):
    orig = model.embed_prefix
    def w(images, img_masks, lang_tokens, lang_masks, state=None):
        lm = lang_masks.clone(); lm[:, idx] = False
        return orig(images, img_masks, lang_tokens, lang_masks=lm, state=state)
    model.embed_prefix = w
    try: return lab.chunk(o, ins)
    finally: del model.embed_prefix
for x, c in PAIRS:
    idx = [p - 128 for p, _ in LX.align(x, c)]
    r = [1 - A.dist(deleted(o, x, idx), lab.chunk(o, c)) / A.dist(lab.chunk(o, x), lab.chunk(o, c)) for o in obs]
    print(f"{x:34} delete {[A.tokens(x)[i] for i in idx]}: R {np.mean(r):+.2f} {np.round(r, 2)}")
