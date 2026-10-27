"""Build a trained-layout reference for the virtual-frame program from the frozen policy's OWN successful episodes of a trained
suite task (run_program --program none --save-obs --trace): first agentview frame, pick point (eef when the gripper first
closes), place point (eef when it first reopens after holding >= 10 steps), and the pick object's footprint mask.
Usage: make_reference.py <suite> <task>  ->  library/references/<suite>__<task>.npz   (no simulator internals involved)"""
import sys, glob, json
import numpy as np
sys.path[:0] = ['library']
from steerlib import scene

def events(actions, steps):
    g = actions[:, 6] > 0
    close = np.where(g)[0]
    if not len(close): return None, None
    c = int(close[0]); o = None
    for t in range(c + 10, len(g)):
        if not g[t]: o = t; break
    near = lambda t: np.array(min(steps, key=lambda x: abs(x["t"] - (t + 1)))["eef"])
    # successful episodes end when the goal predicate holds, usually before the gripper reopens: then the last eef is the place
    return near(c), (near(o) if o is not None else np.array(steps[-1]["eef"]))

suite, task = sys.argv[1], int(sys.argv[2])
out = []
for f in sorted(glob.glob(f"outputs/programs/none/{suite}__{task}/episode_*.npz")):
    d = np.load(f)
    if not bool(d["success"]): continue
    try: js = json.load(open(f.replace(".npz", ".json")))
    except FileNotFoundError: continue
    pick, place = events(d["actions"], js["trace"]["steps"])
    fp = scene.footprint(d["agentview_image"][0], d["agentview_image"][-1])
    out.append((f, d["agentview_image"][0], pick, place, fp))
    print(f.split('/')[-1], "pick", None if pick is None else np.round(pick, 3), "place", None if place is None else np.round(place, 3),
          "footprint", None if fp is None else (np.round(scene.centroid(fp), 1), int(fp.sum())))
good = [o for o in out if o[2] is not None]
if not good: sys.exit("no usable episode")
f, img, pick, place, fp = good[0]
if fp is None: fp = np.zeros(img.shape[:2], bool)
picks = np.array([o[2] for o in good]); places = np.array([o[3] for o in good if o[3] is not None])
np.savez_compressed(f"library/references/{suite}__{task}.npz", image=img, footprint=fp, pick=picks.mean(0),
                    place=places.mean(0) if len(places) else np.full(3, np.nan), source=f, n=len(good))
print("saved", len(good), "episodes; pick", np.round(picks.mean(0), 3), "place", np.round(places.mean(0), 3) if len(places) else None)
