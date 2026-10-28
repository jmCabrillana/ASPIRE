"""E41: E40 on REACHING observations (gripper open, first half of the episode): is the pick target controlled by the same linear
endpoint (cumulative displacement, metres, from the normalised chunk) with respect to an additive vector at a site (expert
layer L MLP output, all action tokens), at v = 0. Questions:
 1. Is J consistent across observations and tasks (cosine of rows, singular values of the stacked Jacobian)?
 2. Does the closed-form interface v = pinv(J_mean)[:, :2] @ (dx, dy) (no training, one backward per observation) move the
    endpoint as wanted on other tasks (gain, off-axis), for 8 cm and 20 cm offsets? Compare with the trained M (E39).
 3. Per layer: where is the destination most linearly controllable (smallest ||v|| per cm)?
Batched (16 obs per pass) because the GPU is time-sliced. Usage: e40_jacobian_basis.py [layers comma-separated]"""
import sys, glob, time
sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A
from steer.hooks import layers

LAYERS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 4, 8]
rng = np.random.default_rng(1)
policy, lab, _ = A.load()
p = policy
dev = next(p.policy.parameters()).device
z = torch.zeros(1, 7, device=dev)
MEAN = p.post(z).squeeze(0).float().to(dev)
STD = (p.post(z + 1).squeeze(0).float().to(dev) - MEAN)
for prm in p.policy.parameters():
    prm.requires_grad_(False)


def holding(tasks, per_ep):
    out = []
    for t in tasks:
        for f in sorted(glob.glob(f"outputs/programs/none/libero_spatial__{t}/episode_*.npz")) + \
                 sorted(glob.glob(f"outputs/programs/rec10/libero_spatial__{t}/episode_*.npz")):
            ep = lab.load(f); q = ep.data["robot0_gripper_qpos"][:, 0]
            idx = [i for i in range(len(ep)) if q[i] > 0.035 and i < len(ep) // 2]
            idx = idx[:: max(1, len(idx) // per_ep)][:per_ep]
            out += [(ep.obs(i), ep.instruction, t) for i in idx]
    return out


def bb(items):
    bs = [p.batch(o, i) for o, i, _ in items]
    out = {k: torch.cat([b[k] for b in bs]) for k in bs[0] if k != "task"}; out["task"] = [b["task"][0] for b in bs]
    return p.policy._prepare_batch(p.pre(out))


def endpoint(norm_chunk):
    """(B, 50, 7) normalised -> (B, 3) cumulative displacement in metres (clipped actions, 0.0095 m per unit per step)."""
    a = norm_chunk[..., :3] * STD[:3] + MEAN[:3]
    return a.clamp(-1, 1).sum(1) * 0.0095


TRAIN = holding([0, 1, 2, 3, 4, 6, 8], 1)
TEST = holding([5, 7, 9], 1)
print(len(TRAIN), "train obs,", len(TEST), "test obs", flush=True)
cur = {}


def run(batch, L, v, grad=False):
    def hook(m, inp, out):
        return out + v[:, None, :].to(out.dtype) if v.dim() == 2 else out + v.to(out.dtype)
    h = layers(p, "expert")[L].mlp.register_forward_hook(hook)
    try:
        torch.manual_seed(0)
        with torch.set_grad_enabled(grad):
            return p.policy._get_action_chunk(batch, None)
    finally:
        h.remove()


for L in LAYERS:
    t0 = time.time()
    width = layers(p, "expert")[L].mlp.down_proj.out_features
    Js = []
    for i in range(0, len(TRAIN), 16):
        items = TRAIN[i:i + 16]; batch = bb(items)
        v = torch.zeros(len(items), width, device=dev, requires_grad=True)
        e = endpoint(run(batch, L, v, grad=True))            # (B, 3); rows are independent across the batch
        rows = []
        for k in range(3):
            g, = torch.autograd.grad(e[:, k].sum(), v, retain_graph=k < 2)
            rows.append(g.detach())
        Js.append(torch.stack(rows, 1))                        # (B, 3, width)
    J = torch.cat(Js).float()                                  # (n, 3, width)
    Jm = J.mean(0)
    cos = torch.nn.functional.cosine_similarity(J[:, :2].reshape(len(J), -1), Jm[:2].reshape(1, -1)).cpu().numpy()
    S = torch.linalg.svdvals(J[:, :2].reshape(-1, width)).cpu().numpy()
    P = torch.linalg.pinv(Jm[:2])                              # (width, 2): minimum-norm v for a wanted (dx, dy)
    print(f"L{L}: J consistency cos(obs, mean) mean {cos.mean():.2f} min {cos.min():.2f}; top singular values "
          f"{np.round(S[:4], 3)}; ||pinv col|| per cm {float(P.norm(dim=0).mean()) / 100:.3f}  ({time.time() - t0:.0f} s)",
          flush=True)
    for R in (0.08, 0.20):
        res = {}
        for name, data in (("train", TRAIN[:16]), ("TEST", TEST)):
            gains, orth = [], []
            for i in range(0, len(data), 16):
                items = data[i:i + 16]; batch = bb(items)
                ang = rng.uniform(0, 2 * np.pi, len(items)); d = R * np.stack([np.cos(ang), np.sin(ang)], 1)
                dv = torch.as_tensor(d, dtype=torch.float32, device=dev)
                e0 = endpoint(run(batch, L, torch.zeros(width, device=dev)))[:, :2].cpu().numpy()
                e1 = endpoint(run(batch, L, dv @ P.T))[:, :2].cpu().numpy()
                s = e1 - e0
                for j in range(len(items)):
                    u = d[j] / R; gains.append(s[j] @ u / R); orth.append(abs(s[j][0] * u[1] - s[j][1] * u[0]) * 100)
            res[name] = (np.mean(gains), np.mean(orth))
        print(f"   pinv interface, |D| {R * 100:.0f} cm: train gain {res['train'][0]:.2f} off-axis {res['train'][1]:.1f} cm | "
              f"TEST gain {res['TEST'][0]:.2f} off-axis {res['TEST'][1]:.1f} cm", flush=True)
    torch.save({"P": P.cpu(), "Jm": Jm.cpu(), "layer": L, "where": "expert", "source": "E41 pinv of mean endpoint Jacobian, REACHING obs"},
               f"library/directions/reach_pinv_L{L}.pt")
hold = torch.load("library/directions/dest_pinv_L0.pt")["Jm"][:2].float()
reach = torch.load("library/directions/reach_pinv_L0.pt")["Jm"][:2].float()
print("cos(J_hold, J_reach) rows x, y:", [round(float(torch.nn.functional.cosine_similarity(hold[k], reach[k], dim=0)), 3) for k in range(2)])
