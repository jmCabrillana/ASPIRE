"""E06 (goal_swap 1): does boosting the expert's attention to image tokens move the reach toward the bowl's real location?
Analysis references (not program inputs): in goal_swap the bowl sits at the trained wine-bottle spot (BDDL, notes only).
ref_true = reach of 'put the wine bottle on the rack' on a trained-scene obs; ref_mem = 'put the bowl on the stove' trained obs.
Metric: end of the first 25 steps of the predicted path (cm); report distance to ref_true and ref_mem."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np
from steerlib import analysis as A, attention as AT
policy, lab, _ = A.load()
m = policy.policy.model.vlm_with_expert
def end(ch, n=25): return A.path(ch[:n])[-1] * 100
tr = lab.load("outputs/programs/none/libero_goal__1/episode_0.npz").obs(0)
ref_true, ref_mem = end(lab.chunk(tr, "put the wine bottle on the rack")), end(lab.chunk(tr, "put the bowl on the stove"))
print("ref_true", np.round(ref_true, 1), "ref_mem", np.round(ref_mem, 1), "sep", round(float(np.linalg.norm(ref_true - ref_mem)), 1))
INS = "put the bowl on the stove"
for e in range(3):
    o = lab.load(f"outputs/programs/none/libero_goal_swap__1/episode_{e}.npz").obs(0)
    for seg in ({"image": slice(0, 64)}, {"image2": slice(64, 128)}, {"image": slice(0, 64), "image2": slice(64, 128)}):
        for amt in (0, 1, 2, 4):
            for layers in (None, range(0, 8), range(8, 16)):
                if amt == 0 and (layers is not None or len(seg) > 1 or "image2" in seg): continue
                r = AT.patch(m, AT.segment_bias(seg, amt, layers=None if layers is None else set(layers)))
                try:
                    p = end(lab.chunk(o, INS))
                finally:
                    r()
                print(f"ep{e} {'+'.join(seg)} {amt} L{'all' if layers is None else f'{layers.start}-{layers.stop-1}'}: "
                      f"to_true {np.linalg.norm(p-ref_true):5.1f} to_mem {np.linalg.norm(p-ref_mem):5.1f}  end {np.round(p,1)}")
