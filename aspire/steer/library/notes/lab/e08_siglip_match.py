"""E08: locate a moved object by SigLIP patch features (the policy's own vision tower, 32x32 grid before pixel shuffle).
Reference object = footprint patches of the bowl in a trained goal episode; live = goal_swap 1 first frames."""
import sys; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, scene
policy, lab, _ = A.load()
vm = policy.policy.model.vlm_with_expert.get_vlm_model().vision_model
def feats(img):
    """(32, 32, 768) agentview patch features in RAW image orientation."""
    got = []
    h = vm.register_forward_hook(lambda m, i, o: got.append(o.last_hidden_state[0].float().cpu()))
    o = {"agentview_image": img, "robot0_eye_in_hand_image": img, "robot0_eef_pos": np.zeros(3),
         "robot0_eef_quat": np.array([0, 0, 0, 1.0]), "robot0_gripper_qpos": np.zeros(2)}
    try: lab.chunk(o, "x")
    finally: h.remove()
    f = got[0].reshape(32, 32, -1).numpy()[::-1, ::-1]          # undo the 180-degree rotation
    return f / np.linalg.norm(f, axis=-1, keepdims=True)
d = np.load('outputs/programs/none/libero_goal__1/episode_1.npz')
ref = d['agentview_image'][0]; fp = scene.footprint(ref, d['agentview_image'][-1])
F = feats(ref)
pm = fp.reshape(32, 8, 32, 8).mean((1, 3)) > 0.3                 # footprint patches
proto = F[pm].mean(0); proto /= np.linalg.norm(proto)
print("ref footprint patches", pm.sum(), "ref best", np.unravel_index((F @ proto).argmax(), (32, 32)))
for e in range(5):
    live = np.load(f'outputs/programs/none/libero_goal_swap__1/episode_{e}.npz')['agentview_image'][0]
    s = feats(live) @ proto
    r, c = np.unravel_index(s.argmax(), s.shape)
    top = np.argsort(-s.ravel())[:4]
    print(e, "best patch px", (c * 8 + 4, r * 8 + 4), round(float(s.max()), 3), "top4 px", [((t % 32) * 8 + 4, (t // 32) * 8 + 4) for t in top])
