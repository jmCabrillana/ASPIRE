"""Activation banks over saved episodes (wraps steer.interp) with real-token language segments.

The prefix is always 177 tokens (language padded to 48); steer.interp averages 'lang' over all 48 positions (padding
included). build() here averages over the instruction's real tokens only (steerlib.analysis.spans), plus per-token keys:
  "vlm<L>:tok<k>"   activation at language token k (k-th token of the instruction), when all instructions have >= k+1 tokens
  "vlm<L>:last"     the last instruction token (the newline), "vlm<L>:first" the first instruction token
Labels: those of interp.Bank.build, plus any extra labels given as {name: fn(episode, chunk_index, obs) -> float}.
"""
from __future__ import annotations

import numpy as np

from steer.interp import Bank, neuron_acts
from steerlib.analysis import spans


def build(lab, files, layers, where="vlm", instruction=None, extra_labels=None, max_chunks=None) -> Bank:
    acts, meta = {}, []
    labels = {k: [] for k in ("success", "episode", "phase", "t", "x", "y", "z", "dx", "dy", "dz", "grip")}
    for k in (extra_labels or {}):
        labels[k] = []
    for e_id, f in enumerate(files):
        ep = lab.load(f)
        text = instruction(ep) if callable(instruction) else (instruction or ep.instruction)
        sp = spans(text)
        n = len(ep) if max_chunks is None else min(len(ep), max_chunks)
        for i in range(n):
            o = ep.obs(i)
            a = neuron_acts(lab, o, text, where, layers)
            for l in layers:
                if where == "vlm":
                    x = a[l]
                    for seg, s in sp.items():
                        if seg == "pad":
                            continue
                        acts.setdefault(f"vlm{l}:{seg}", []).append(x[s].mean(0).numpy())
                    acts.setdefault(f"vlm{l}:first", []).append(x[sp["lang"].start].numpy())
                    acts.setdefault(f"vlm{l}:last", []).append(x[sp["lang"].stop - 1].numpy())
                else:
                    acts.setdefault(f"expert{l}:all", []).append(a[l].mean(dim=(0, 1)).numpy())
            act = ep.actions(i)
            labels["success"].append(float(ep.success)); labels["episode"].append(e_id)
            labels["phase"].append(float(o["robot0_gripper_qpos"][0] < 0.03)); labels["t"].append(float(ep.data["t"][i]))
            for k, v in zip("xyz", o["robot0_eef_pos"]):
                labels[k].append(float(v))
            for k, j in (("dx", 0), ("dy", 1), ("dz", 2), ("grip", 6)):
                labels[k].append(float(act[:, j].mean()) if len(act) else 0.0)
            for k, fn in (extra_labels or {}).items():
                labels[k].append(float(fn(ep, i, o)))
            meta.append({"file": str(f), "chunk": i, "t": int(ep.data["t"][i]), "instruction": text})
    return Bank({k: np.stack(v) for k, v in acts.items()}, {k: np.array(v) for k, v in labels.items()}, meta)
