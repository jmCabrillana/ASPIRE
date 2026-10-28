"""E38: MINIMAL SUFFICIENT INTERVENTION for the destination (discovery method: sparse gradient search, steerlib.levers).
Same data and targets as E37 (holding-phase observations of trained spatial episodes; target = the policy's own chunk with its
endpoint moved by D). Proximal-L1 search over the MLP neurons of several expert layers at once; the support is the lever.
Report: support size, layers, train/test gain (achieved shift / wanted, along D) and orthogonal error, and how much of the
effect the top-k neurons keep (k = 1, 4, 16, all).
Usage: e38_sparse_destination.py dx dy dz l1"""
import sys, glob, time
sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A, levers
from steer.lab import normalised_chunk

D = np.array([float(x) for x in sys.argv[1:4]]) if len(sys.argv) > 3 else np.array([0.0, 0.08, 0.0])
L1 = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
LAYERS = [0, 2, 4, 6, 8, 10, 12, 14]
N = 30
policy, lab, _ = A.load()
p = policy
z = torch.zeros(1, 7, device=p.device)
STD = p.post(z + 1).squeeze(0).float().cpu().numpy() - p.post(z).squeeze(0).float().cpu().numpy()


def holding(tasks, per_ep):
    out = []
    for t in tasks:
        for f in sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz")) + \
                 sorted(glob.glob(f"outputs/programs/rec10/libero_spatial__{t}/episode_*.npz")):
            ep = lab.load(f); q = ep.data["robot0_gripper_qpos"][:, 0]
            idx = [i for i in range(len(ep)) if 0.0005 < q[i] < 0.03]
            idx = idx[:: max(1, len(idx) // per_ep)][:per_ep]
            out += [(ep.obs(i), ep.instruction) for i in idx]
    return out


TRAIN = holding([0, 1, 2, 3, 4], 2)
TEST = holding([5, 6, 7, 8, 9], 2)
print(len(TRAIN), "train obs,", len(TEST), "test obs; D", np.round(D * 100, 1), "cm; l1", L1, flush=True)


def target(o, ins):
    t = normalised_chunk(lab, o, ins).copy()
    t[:N, :3] += (D / (N * 0.0095)) / STD[:3]
    return t


def gain(sol, data, keep=None):
    u = D / np.linalg.norm(D); sh, orth = [], []
    for o, ins in data:
        a0 = lab.chunk(o, ins)
        with levers.apply(p, sol, keep=keep):
            a1 = lab.chunk(o, ins)
        d = A.path(a1)[-1] - A.path(a0)[-1]
        sh.append(d @ u); orth.append(np.linalg.norm(d - (d @ u) * u))
    return np.mean(sh) / np.linalg.norm(D), np.mean(orth) * 100


t0 = time.time()
ex = [(o, i, target(o, i)) for o, i in TRAIN]
sol = levers.sparse_neurons(lab, ex, "expert", LAYERS, l1=L1, steps=120, lr=0.05, log_every=20)
print(f"search {time.time() - t0:.0f} s; support {len(sol['support'])} neurons", flush=True)
by_layer = {}
for L, n, v in sol["support"]:
    by_layer[L] = by_layer.get(L, 0) + 1
print("per layer:", by_layer)
print("top 12:", [(L, n, round(v, 2)) for L, n, v in sol["support"][:12]])
for k in (1, 4, 16, None):
    g_tr, o_tr = gain(sol, TRAIN, k); g_te, o_te = gain(sol, TEST, k)
    print(f"keep {k}: train gain {g_tr:.2f} orth {o_tr:.1f} cm | TEST gain {g_te:.2f} orth {o_te:.1f} cm", flush=True)
torch.save(sol, f"outputs/lab_e38_{'_'.join(str(int(x * 100)) for x in D)}_{L1}.pt")
