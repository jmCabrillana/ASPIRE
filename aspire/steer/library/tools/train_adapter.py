"""Distil self-generated successes into a LoRA adapter on the frozen SmolVLA (base weights untouched; adapter stored).
Each training episode is relabelled with the ORIGINAL paraphrase of its task (library/notes/lab/lan_instructions.json), so the
adapter learns paraphrase -> the behaviour the steering program produced. Usage:
  train_adapter.py <name> <where vlm|expert|both> <steps> <npz> [<npz> ...]
Writes library/adapters/<name>.pt (+ .json with sources and sample efficiency)."""
import os, sys, json, re
sys.path[:0] = ['library', '.']
import torch
from steer import lora
from steerlib import analysis as A
name, where, steps, files = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
lan = json.load(open('library/notes/lab/lan_instructions.json'))
policy, lab, _ = A.load()
eps = []
for f in files:
    ep = lab.load(f)
    if not ep.success:
        continue
    m = re.search(r"/(libero_\w+?)__(\d+)/", f)
    suite, task = m.group(1), int(m.group(2))
    if os.environ.get("KEEP_LABEL") != "1":          # default: relabel with the original paraphrase
        ep.instruction = lan[suite][task] if suite in lan else ep.instruction
    eps.append(ep)
print(len(eps), "successful episodes,", sum(len(e) for e in eps), "chunks")
sites = ["vlm", "expert"] if where == "both" else [where]
ads = [lora.attach(policy, targets=("q_proj", "v_proj"), where=w, rank=8) for w in sites]
merged = lora.Adapter(slots=[s for a in ads for s in a.slots])
torch.manual_seed(0)
losses = lora.train(policy, merged, eps, steps=steps, lr=1e-4, log_every=max(steps // 4, 1))
lora.save(merged, f"library/adapters/{name}.pt")
json.dump({"where": where, "rank": 8, "targets": ["q_proj", "v_proj"], "steps": steps, "episodes": files,
           "n_success_episodes": len(eps), "n_chunks": sum(len(e) for e in eps), "final_loss": sum(losses[-20:]) / 20},
          open(f"library/adapters/{name}.json", "w"), indent=1)
for a in ads:
    lora.detach(policy, a)
print("saved", name)
