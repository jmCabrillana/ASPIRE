"""Attach a stored LoRA adapter (library/adapters/<ADAPTER>.pt, trained by tools/train_adapter.py from self-generated
successes) for the episode; the instruction is NOT rewritten. ADAPTER is set by the environment variable STEER_ADAPTER
(recorded in the trace). Base weights are restored after the episode (adapters are separate modules)."""
import json
import os
from pathlib import Path

from steer import lora

DESCRIPTION = "stored LoRA adapter distilled from steering-program successes; original instruction"
AD = Path(__file__).resolve().parents[2] / "adapters"


def setup(ctx):
    name = os.environ["STEER_ADAPTER"]
    meta = json.loads((AD / f"{name}.json").read_text())
    sites = ["vlm", "expert"] if meta["where"] == "both" else [meta["where"]]
    ads = [lora.attach(ctx.policy, targets=tuple(meta["targets"]), where=w, rank=meta["rank"]) for w in sites]
    merged = lora.Adapter(slots=[s for a in ads for s in a.slots])
    lora.load(merged, AD / f"{name}.pt")
    ctx.memory["adapters"] = ads
    ctx.log("adapter", {"name": name, **{k: meta[k] for k in ("where", "steps", "n_success_episodes", "n_chunks")}})


def on_end(result, ctx):
    for a in ctx.memory.get("adapters", []):
        lora.detach(ctx.policy, a)
