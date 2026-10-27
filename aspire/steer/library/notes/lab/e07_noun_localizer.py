"""E07: does attention from an instruction noun token to the agentview patches localise that object (and follow it when it
moves)? Prefix pass only; per VLM layer, mean over heads of attention from the noun's tokens to the 64 agentview tokens (8x8).
Saves heatmaps over the image for trained-scene and swap obs."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from PIL import Image
from steerlib import analysis as A, attention as AT
policy, lab, _ = A.load()
m = policy.policy.model.vlm_with_expert
def maps(o, ins, word):
    rec = []
    r = AT.patch(m, record=rec)
    try: lab.chunk(o, ins)
    finally: r()
    toks = A.tokens(ins); idx = [128 + i for i, t in enumerate(toks) if word in t]
    P = 129 + len(toks)
    out = {}
    for layer, pr in rec:
        if pr.shape[1] != 177: continue                     # prefix pass only
        out[layer] = pr[:, idx, 0:64].mean((0, 1)).reshape(8, 8).numpy()[::-1, ::-1]   # model sees the image rotated 180
    return out
OUT = sys.argv[1]
rows = []
for name, f, ins, word in [("trained", "outputs/programs/none/libero_goal__1/episode_0.npz", "put the bowl on the stove", "bowl"),
                           ("swap", "outputs/programs/none/libero_goal_swap__1/episode_0.npz", "put the bowl on the stove", "bowl"),
                           ("trained", "outputs/programs/none/libero_goal__1/episode_0.npz", "put the bowl on the stove", "stove"),
                           ("swap", "outputs/programs/none/libero_goal_swap__1/episode_0.npz", "put the bowl on the stove", "stove")]:
    o = lab.load(f).obs(0); mp = maps(o, ins, word)
    tiles = [o["agentview_image"]]
    for layer in (2, 4, 6, 8, 10, 12, 14):
        h = mp[layer]; h = (h - h.min()) / (h.max() - h.min() + 1e-8)
        hm = np.array(Image.fromarray((h * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST))
        tiles.append((0.5 * o["agentview_image"] + 0.5 * np.stack([hm, 0 * hm, 0 * hm], -1)).astype(np.uint8))
        am = np.unravel_index(h.argmax(), h.shape)
        print(name, word, f"L{layer} argmax patch (row,col) {am} max {mp[layer].max():.3f} sum {mp[layer].sum():.3f}")
    rows.append(np.concatenate(tiles, 1))
Image.fromarray(np.concatenate(rows, 0)).resize((8 * 160, 4 * 160)).save(OUT)
