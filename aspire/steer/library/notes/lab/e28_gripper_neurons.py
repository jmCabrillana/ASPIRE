"""E28: are there expert neurons that control GRIPPER TIMING? Attribution patching (activation x gradient) of
target = mean gripper command over the first 40 steps of the chunk (normalised), expert layers 4/8/12/15, on reaching
observations; then dosed pushes of the top-k neurons (all tokens, all denoising steps) and the step at which the gripper
command first closes."""
import sys, glob; sys.path[:0] = ['library', '.']
import numpy as np, torch
from steerlib import analysis as A
policy, lab, nr = A.load()
files = sorted(glob.glob("outputs/programs/none/libero_goal__[189]/episode_*.npz"))[:6]
obs = [(lab.load(f).obs(0), lab.load(f).instruction) for f in files]
LAY = [4, 8, 12, 15]
tot = {L: 0 for L in LAY}
for o, ins in obs:
    a = nr.attribute(o, ins, lambda act: act[:40, 6].mean(), "expert", LAY)
    for L in LAY: tot[L] = tot[L] + a[L]
cands = sorted([(float(v), L, int(n)) for L in LAY for n, v in enumerate(tot[L])], key=lambda z: -abs(z[0]))[:8]
print("top attributed (sum over obs):", [(L, n, round(v, 3)) for v, L, n in cands])
def close_step(ch):
    c = np.where(ch[:, 6] > 0)[0]; return int(c[0]) if len(c) else 50
for k in (1, 3, 8):
    for dose in (-2, -1, 1, 2):
        shifts = []
        for o, ins in obs:
            base = close_step(lab.chunk(o, ins))
            pushes = [("expert", L, n, np.sign(v) * nr.dose_to_activation(o, ins, "expert", L, n, dose)) for v, L, n in cands[:k]]
            with nr.push(pushes):
                shifts.append(close_step(lab.chunk(o, ins)) - base)
        print(f"top-{k} dose {dose:+d}: close-step shift mean {np.mean(shifts):+.1f} (per obs {shifts})")
