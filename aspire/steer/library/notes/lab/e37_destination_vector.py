"""E37: a gradient-learned DESTINATION KNOB. Can one additive vector (steer.lab.optimize_vector) shift where the frozen VLA carries
the held object by a chosen offset D, across observations, episodes and tasks it was not fitted on? Which layer/site works best?

Data: holding-phase observations (gripper closed on the object) of the frozen policy's own trained-suite episodes.
Target: the policy's own normalised chunk with the translation of the first N steps biased so the chunk's endpoint moves by D.
Fit on tasks TRAIN, evaluate the achieved endpoint shift on tasks TEST (other scenes/instructions): along D (gain = achieved/wanted)
and orthogonal (cm), plus gripper-command change.
Usage: e37_destination_vector.py [dx dy dz] (metres)"""
import sys, glob, time
sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A
from steer.lab import optimize_vector, normalised_chunk
from steer.hooks import layers

D = np.array([float(x) for x in sys.argv[1:4]]) if len(sys.argv) > 3 else np.array([0.0, 0.08, 0.0])
N = 30
policy, lab, _ = A.load()
p = policy


def stats():
    z = torch.zeros(1, 7, device=p.device)
    mean = p.post(z).squeeze(0).float().cpu().numpy()
    std = p.post(z + 1).squeeze(0).float().cpu().numpy() - mean
    return mean, std


MEAN, STD = stats()
print("action std", np.round(STD, 3))


def holding(files, max_per_ep=6):
    out = []
    for f in files:
        ep = lab.load(f)
        q = ep.data["robot0_gripper_qpos"][:, 0]
        idx = [i for i in range(len(ep)) if 0.0005 < q[i] < 0.03][:max_per_ep]
        out += [(ep.obs(i), ep.instruction, f) for i in idx]
    return out


def files(tasks):
    fs = []
    for t in tasks:
        fs += sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz"))
        fs += sorted(glob.glob(f"outputs/programs/rec10/libero_spatial__{t}/episode_*.npz"))
    return fs


TRAIN = holding(files([0, 1, 2, 3, 4]), 4)
TEST = holding(files([5, 6, 7, 8, 9]), 4)
print(len(TRAIN), "train obs,", len(TEST), "test obs")


def target(o, ins):
    a = normalised_chunk(lab, o, ins)
    t = a.copy()
    t[:N, :3] += (D / (N * 0.0095)) / STD[:3]
    return t


def achieved(v, where, layer, tokens, data):
    sl = tokens if tokens is not None else slice(None)

    def hook(m, inp, out):
        add = torch.zeros_like(out); add[:, sl] = v.to(out.dtype); return out + add
    sh, orth, grip = [], [], []
    for o, ins, _ in data:
        a0 = lab.chunk(o, ins)
        h = layers(p, where)[layer].mlp.register_forward_hook(hook)
        try:
            a1 = lab.chunk(o, ins)
        finally:
            h.remove()
        d = A.path(a1)[-1] - A.path(a0)[-1]
        u = D / np.linalg.norm(D)
        sh.append(d @ u); orth.append(np.linalg.norm(d - (d @ u) * u)); grip.append(np.abs(a1[:, 6] - a0[:, 6]).mean())
    return np.mean(sh) / np.linalg.norm(D), np.mean(orth) * 100, np.mean(grip)


noise = np.mean([np.linalg.norm(A.path(lab.chunk(o, i))[-1] - A.path(lab.chunk(o, i, seed=1))[-1]) * 100 for o, i, _ in TEST[:8]])
print(f"D = {np.round(D * 100, 1)} cm; seed noise of the endpoint {noise:.2f} cm")
ex = [(o, i, target(o, i)) for o, i, _ in TRAIN]
SITES = [("expert", 0, None)] if __import__("os").environ.get("E37_FAST") else [("expert", 0, None), ("expert", 4, None), ("expert", 8, None), ("expert", 12, None),
         ("vlm", 2, None), ("vlm", 8, None), ("vlm", 14, None), ("vlm", 8, slice(-1, None)), ("vlm", 14, slice(-1, None))]
for where, layer, tok in SITES:
    t0 = time.time()
    v = optimize_vector(lab, ex, where, layer, steps=60, lr=0.05, tokens=tok)
    gtr = achieved(v, where, layer, tok, TRAIN[:12])
    gte = achieved(v, where, layer, tok, TEST)
    print(f"{where:6} L{layer:2} {'state' if tok else 'all  '} |v| {float(v.norm()):7.2f}  train gain {gtr[0]:5.2f} orth {gtr[1]:4.1f} cm"
          f" | TEST gain {gte[0]:5.2f} orth {gte[1]:4.1f} cm grip {gte[2]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
    torch.save({"v": v.cpu(), "D": D, "where": where, "layer": layer, "tokens": str(tok)},
               f"outputs/lab_e37_{where}{layer}{'s' if tok else ''}_{'_'.join(str(int(x * 100)) for x in D)}.pt")
