"""Distil successful hand-back hybrid episodes (carry_handback2 / approach_handback, --save-obs --trace) into a LoRA adapter so
the frozen VLA alone (no program) learns the swapped layout from its own self-generated successes.
The saved proprio state was shifted (virtual frame) after the hand-back: undone here with the logged delta (place) so the
adapter sees the real state. Label = the task instruction (unchanged in swap tasks).
Usage: train_adapter_swap.py <name> <where vlm|expert|both> <steps> <npz> [...]"""
import sys, json
sys.path[:0] = ['library', '.']
import numpy as np, torch
from steer import lora
from steerlib import analysis as A
name, where, steps, files = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
policy, lab, _ = A.load()
eps = []
for f in files:
    ep = lab.load(f)
    if not ep.success:
        continue
    log = {x["key"]: x["value"] for x in json.load(open(f.replace(".npz", ".json")))["trace"]["log"]}
    hb = (log.get("handback") or {}).get("t")
    delta = np.r_[np.array(log["place"]["delta_cm"]) / 100.0, 0.0]
    if hb is not None:
        after = ep.data["t"] > hb
        ep.data["robot0_eef_pos"] = ep.data["robot0_eef_pos"].copy()
        ep.data["robot0_eef_pos"][after] = ep.data["robot0_eef_pos"][after] + delta     # real state
    eps.append(ep)
print(len(eps), "successful episodes,", sum(len(e) for e in eps), "chunks")
sites = ["vlm", "expert"] if where == "both" else [where]
ads = [lora.attach(policy, targets=("q_proj", "v_proj"), where=w, rank=8) for w in sites]
merged = lora.Adapter(slots=[s for a in ads for s in a.slots])
torch.manual_seed(0)
losses = lora.train(policy, merged, eps, steps=steps, lr=1e-4, log_every=max(steps // 4, 1))
lora.save(merged, f"library/adapters/{name}.pt")
json.dump({"where": where, "rank": 8, "targets": ["q_proj", "v_proj"], "steps": steps, "episodes": files,
           "n_success_episodes": len(eps), "n_chunks": sum(len(e) for e in eps), "final_loss": sum(losses[-20:]) / 20,
           "source": "self-generated carry_handback2 successes (state un-shifted)"}, open(f"library/adapters/{name}.json", "w"), indent=1)
for a in ads:
    lora.detach(policy, a)
print("saved", name)
