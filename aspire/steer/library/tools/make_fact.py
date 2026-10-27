"""Fit and store a lexical fact (layer-L edit of the VLM MLP down projection): the unknown word in sentence x reads as the known
word of the general-rule rewrite c. Stored in library/edits/<name>.pt with metadata; indexed in library/edits/facts.json.
Usage: make_fact.py <name> <source suite> <source task> "<x>" "<c>" [layer] [obs episodes file pattern]
Sample efficiency is recorded: 1 sentence pair, 2 observations (first frames of two saved frozen episodes of that scene)."""
import sys, json, glob
sys.path[:0] = ['library', '.']
import torch
from steerlib import analysis as A, lexedit as LX
name, suite, task, x, c = sys.argv[1:6]
L = int(sys.argv[6]) if len(sys.argv) > 6 else 1
pattern = sys.argv[7] if len(sys.argv) > 7 else f"outputs/programs/none/{suite}__{task}/episode_*.npz"
policy, lab, _ = A.load()
files = sorted(glob.glob(pattern))[:2]
obs = [lab.load(f).obs(0) for f in files]
delta = LX.fit(lab, obs, x, c, L=L)
torch.save({"delta": delta.cpu(), "layer": L}, f"library/edits/{name}.pt")
idx_path = "library/edits/facts.json"
try: idx = json.load(open(idx_path))
except FileNotFoundError: idx = {}
tx, tc = A.tokens(x), A.tokens(c)
idx[name] = {"layer": L, "source": [suite, int(task)], "x": x, "c": c, "trigger_tokens": [tx[p - 128] for p, _ in LX.align(x, c)],
             "n_sentences": 1, "n_observations": len(obs), "obs_files": files, "norm": float(delta.norm())}
json.dump(idx, open(idx_path, "w"), indent=1)
print(name, idx[name])
