"""E39: a learned DESTINATION INTERFACE for the frozen VLA. Instead of one vector per offset (E37: nonlinear at dose -2 in closed
loop: -8..-10 cm instead of -15.6 and +7 cm off-axis drift), learn a linear map M (width x 2) at the action expert's layer-L MLP
output so that adding M @ (dx, dy) moves the predicted place point by (dx, dy) for any offset in a disc of radius R.
Training data: holding-phase observations of frozen trained spatial tasks (TRAIN tasks); for each observation K random offsets;
target = the policy's own normalised chunk with its translation biased by D / (N * 0.0095) per step over the first N steps.
Memory-safe (one backward per example). Test: other tasks, new random offsets: gain along D, off-axis error, gripper change.
Saves library/directions/dest_iface_L<L>.pt = {"M": (width, 2), "layer": L, "R": R, ...}.
Usage: e39_destination_interface.py [layer] [steps]"""
import sys, glob, time
sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A
from steer.lab import normalised_chunk
from steer.hooks import layers

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 0
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
R, N, K = 0.25, 50, 3
rng = np.random.default_rng(0)
policy, lab, _ = A.load()
p = policy
dev = next(p.policy.parameters()).device
z = torch.zeros(1, 7, device=dev)
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


def disc(n):
    a = rng.uniform(0, 2 * np.pi, n); r = R * np.sqrt(rng.uniform(0.05, 1, n))
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


TRAIN = holding([0, 1, 2, 3, 4, 6, 8], int(__import__("os").environ.get("PER_EP", "2")))
TEST = holding([5, 7, 9], 2)
print(len(TRAIN), "train obs,", len(TEST), "test obs", flush=True)
def bb(items):
    bs = [p.batch(o, i) for o, i in items]
    out = {k: torch.cat([b[k] for b in bs]) for k in bs[0] if k != "task"}; out["task"] = [b["task"][0] for b in bs]
    return p.policy._prepare_batch(p.pre(out))


width = layers(p, "expert")[LAYER].mlp.down_proj.out_features
M = torch.zeros(width, 2, device=dev, requires_grad=True)
cur = {"d": None}                      # (B, 2) offsets of the current batch


def hook(m, inp, out):
    if cur["d"] is None:
        return None
    return out + (cur["d"] @ M.T)[:, None, :].to(out.dtype)


def chunks(batch, d=None, grad=False):
    cur["d"] = d
    try:
        torch.manual_seed(0)
        with torch.set_grad_enabled(grad):
            return p.policy._get_action_chunk(batch, None)
    finally:
        cur["d"] = None


for prm in p.policy.parameters():
    prm.requires_grad_(False)
h = layers(p, "expert")[LAYER].mlp.register_forward_hook(hook)
B = 16
tb = [TRAIN[i:i + B] for i in range(0, len(TRAIN), B)]
batches = [(bb(items), chunks(bb(items)).detach()) for items in tb]
stdt = torch.as_tensor(STD[:2], dtype=torch.float32, device=dev)
opt = torch.optim.Adam([M], lr=0.05)
t0 = time.time()
for step in range(STEPS):
    opt.zero_grad(); tot = 0.0
    for batch, base in batches:
        d = torch.as_tensor(disc(len(base)), dtype=torch.float32, device=dev)
        target = base.clone(); target[:, :N, :2] += (d / (N * 0.0095) / stdt)[:, None, :]
        pred = chunks(batch, d, grad=True)
        li = ((pred - target) ** 2).mean() / len(batches)
        li.backward(); tot += float(li)
    opt.step()
    if step % 10 == 0:
        print(f"step {step} loss {tot:.4f} |M| {float(M.norm()):.2f} ({time.time() - t0:.0f} s)", flush=True)


def evaluate(data):
    gains, orth, grip = [], [], []
    for i in range(0, len(data), B):
        items = data[i:i + B]; batch = bb(items)
        a0 = chunks(batch)
        d = disc(len(items)); a1 = chunks(batch, torch.as_tensor(d, dtype=torch.float32, device=dev))
        for j in range(len(items)):
            u0 = np.stack([p.post(a0[j:j + 1, k]).squeeze(0).float().cpu().numpy() for k in range(50)])
            u1 = np.stack([p.post(a1[j:j + 1, k]).squeeze(0).float().cpu().numpy() for k in range(50)])
            s_ = (A.path(u1)[-1] - A.path(u0)[-1])[:2]; u = d[j] / np.linalg.norm(d[j])
            gains.append(s_ @ u / np.linalg.norm(d[j])); orth.append(abs(s_[0] * u[1] - s_[1] * u[0]) * 100)
            grip.append(np.abs(u1[:, 6] - u0[:, 6]).mean())
    return np.mean(gains), np.median(gains), np.mean(orth), np.mean(grip)


with torch.no_grad():
    print("TRAIN gain mean %.2f median %.2f, off-axis %.1f cm, grip %.3f" % evaluate(TRAIN[:16]), flush=True)
    print("TEST  gain mean %.2f median %.2f, off-axis %.1f cm, grip %.3f" % evaluate(TEST + TEST), flush=True)
h.remove()
torch.save({"M": M.detach().cpu(), "layer": LAYER, "R": R, "N": N, "where": "expert",
            "source": "E39: trained spatial tasks 0-4,6,8 holding obs, synthetic endpoint shifts"},
           f"library/directions/dest_iface_L{LAYER}.pt")
print("saved", f"library/directions/dest_iface_L{LAYER}.pt", f"({time.time() - t0:.0f} s)")
