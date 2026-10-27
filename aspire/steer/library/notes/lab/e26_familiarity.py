"""E26: does the policy-internal FAMILIARITY of a rewrite predict its simulator success? Familiarity at VLM layer L = mean (and
min) over the rewrite's tokens of the max cosine of its residual to the residuals of all tokens of the trained instructions
(4 suites, twin task excluded), on an observation of the scene. Correlated with sim success rates measured this session."""
import sys, json, glob; sys.path[:0] = ['library', '.']
import numpy as np, torch
from scipy.stats import spearmanr
from steerlib import analysis as A, residual as R
policy, lab, _ = A.load(); vla = policy.policy
tr = json.load(open('library/notes/lab/trained_instructions.json'))
CASES = {  # (suite, task): (rewrite used, dev success rate)
 ("libero_goal_lan", 0): ("open the middle drawer of the cabinet", 0.90), ("libero_goal_lan", 1): ("put the bowl on the stove", 0.90),
 ("libero_goal_lan", 2): ("put the wine bottle on top of the cabinet", 1.0), ("libero_goal_lan", 4): ("put the bowl on top of the cabinet", 1.0),
 ("libero_goal_lan", 6): ("put the cream cheese in the bowl", 0.55), ("libero_goal_lan", 7): ("turn on the stove", 1.0),
 ("libero_goal_lan", 8): ("put the bowl on top of the plate", 0.0), ("libero_goal_lan", 9): ("put the wine bottle on the rack", 0.65),
 ("libero_goal_lan", 3): ("open the top drawer and put the bowl", 0.0), ("libero_goal_lan", 5): ("put the plate on the front of the stove", 0.0),
 ("libero_spatial_lan", 0): ("pick up the black bowl between the plate and ramekin and place it on the plate", 0.45),
 ("libero_spatial_lan", 1): ("pick up the black bowl next to the ramekin and place it on the plate", 0.95),
 ("libero_spatial_lan", 2): ("pick up the black bowl on the table center and place it on the plate", 0.30),
 ("libero_spatial_lan", 3): ("pick up the black bowl in the cookie box and place it on the plate", 0.75),
 ("libero_spatial_lan", 4): ("pick up the black bowl in the top drawer and place it on the plate", 0.0),
 ("libero_spatial_lan", 5): ("pick up the black bowl on the ramekin and place it on the plate", 0.80),
 ("libero_spatial_lan", 6): ("pick up the black bowl next to the cookie box and place it on the plate", 0.80),
 ("libero_spatial_lan", 7): ("lift bowl from stove and put plate", 0.0), ("libero_spatial_lan", 8): ("pick up the bowl next to the plate and place it on the plate", 0.40),
 ("libero_spatial_lan", 9): ("lift bowl from cabinet and put on the plate", 0.0)}
LAYERS = [2, 6, 10, 14]
def resid(o, ins):
    with R.record(vla, LAYERS) as r: lab.chunk(o, ins)
    sp = A.spans(ins)["lang"]; return {L: r[L][sp] for L in LAYERS}
obs_by_suite = {"libero_goal_lan": lab.load("outputs/programs/none/libero_goal_lan__7/episode_0.npz").obs(0),
                "libero_spatial_lan": lab.load(sorted(glob.glob("outputs/programs/none/libero_spatial__0/episode_*.npz"))[0]).obs(0)}
banks = {}
rows = []
for (suite, task), (rw, sr) in CASES.items():
    o = obs_by_suite[suite]; base = suite[:-4]
    key = (suite, base, task)
    bank = {L: [] for L in LAYERS}
    for s, l in tr.items():
        for i, ins in enumerate(l):
            if (s, i) == (base, task): continue
            ck = (suite, s, i)
            if ck not in banks: banks[ck] = resid(o, ins)
            for L in LAYERS: bank[L].append(banks[ck][L])
    q = resid(o, rw); fam = {}
    for L in LAYERS:
        B = torch.cat(bank[L]); B = B / B.norm(dim=1, keepdim=True); Q = q[L] / q[L].norm(dim=1, keepdim=True)
        m = (Q @ B.T).max(1).values
        fam[L] = (float(m.mean()), float(m.min()))
    rows.append((sr, fam)); print(f"{suite[7:]} {task} sr {sr:.2f} " + " ".join(f"L{L} {fam[L][0]:.3f}/{fam[L][1]:.2f}" for L in LAYERS))
for L in LAYERS:
    for j, nm in ((0, "mean"), (1, "min")):
        rho, p = spearmanr([r[0] for r in rows], [r[1][L][j] for r in rows])
        print(f"L{L} {nm}: Spearman rho {rho:+.2f} (p {p:.3f})")
