"""E09: offline test of the virtual frame on goal_swap 1 (and 8). Estimate delta_pick / delta_place from images (program code),
shift the state, and compare the first-25-step reach end with the analysis references of E06 (ref_true = reach of 'put the
wine bottle on the rack' in a trained frame = the bowl's new spot in goal_swap; ref_mem = trained 'put the bowl on the stove')."""
import sys; sys.path[:0] = ['library', '.', 'library/programs/candidates']
import numpy as np
from types import SimpleNamespace
from steerlib import analysis as A, scene
import virtual_frame as VF
policy, lab, _ = A.load()
def end(ch, n=25): return A.path(ch[:n])[-1] * 100
tr = lab.load("outputs/programs/none/libero_goal__1/episode_0.npz").obs(0)
ref_true, ref_mem = end(lab.chunk(tr, "put the wine bottle on the rack")), end(lab.chunk(tr, "put the bowl on the stove"))
H = None

for task, ins in [(1, "put the bowl on the stove"), (8, "put the bowl on the plate")]:
    ctx = SimpleNamespace(task=SimpleNamespace(suite="libero_goal_swap", task_id=task), memory={}, vla=policy.policy, log=lambda *a: None)
    VF.setup(ctx)
    for e in range(3):
        o = lab.load(f"outputs/programs/none/libero_goal_swap__{task}/episode_{e}.npz").obs(0)
        d = VF._estimate(o, ctx)
        o2 = dict(o); o2["robot0_eef_pos"] = o["robot0_eef_pos"] - d["pick"]
        p0, p1 = end(lab.chunk(o, ins)), end(lab.chunk(o2, ins))
        msg = f"task {task} ep{e} n {d.get('n_pick')},{d.get('n_place')} d_pick {np.round(d['pick'][:2]*100,1)}cm (sim {d['s_pick']:.2f}) d_place {np.round(d['place'][:2]*100,1)}cm"
        if task == 1:
            msg += f" | frozen to_true {np.linalg.norm(p0-ref_true):.1f} to_mem {np.linalg.norm(p0-ref_mem):.1f} | VF to_true {np.linalg.norm(p1-ref_true):.1f} to_mem {np.linalg.norm(p1-ref_mem):.1f}"
        print(msg, "px", {k: [np.round(x, 0).tolist() for x in v] for k, v in d.items() if k.startswith("u_")})
