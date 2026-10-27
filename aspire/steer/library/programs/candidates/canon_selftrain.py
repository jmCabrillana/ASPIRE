"""Canon v7 rewrite + a self-trained LoRA adapter (expert iteration): the adapter library/adapters/selftrain_<suite>__<task>
was trained on this program's own earlier successes (rewrite + frozen VLA) on development episodes 0-9, labelled with the
rewrite; it sharpens the policy on the rewritten instruction for this layout. Base weights restored after the episode."""
import json
from pathlib import Path

from steer import lora
from steerlib import canon

DESCRIPTION = "canon v7 rewrite + self-trained adapter from the program's own successes"
AD = Path(__file__).resolve().parents[2] / "adapters"


def _twin(suite, task_id):
    for suffix in ("_lan", "_swap", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    canon.SCORER["name"] = "NousResearch/Llama-2-7b-hf"
    canon.PRIOR["kind"] = "unseen234"
    canon.MULTI.update(on=True, compound=True)
    ctx.instruction = canon.canonical(ctx.task.instruction, exclude=_twin(ctx.task.suite, ctx.task.task_id), w=4.0, version="v3")
    name = f"selftrain_{ctx.task.suite}__{ctx.task.task_id}"
    if (AD / f"{name}.pt").exists():
        meta = json.loads((AD / f"{name}.json").read_text())
        sites = ["vlm", "expert"] if meta["where"] == "both" else [meta["where"]]
        ads = [lora.attach(ctx.policy, targets=tuple(meta["targets"]), where=w, rank=meta["rank"]) for w in sites]
        lora.load(lora.Adapter(slots=[s for a in ads for s in a.slots]), AD / f"{name}.pt")
        ctx.memory["adapters"] = ads
        ctx.log("adapter", {"name": name, "n_success_episodes": meta["n_success_episodes"]})
    ctx.log("instruction", ctx.instruction)


def on_end(result, ctx):
    for a in ctx.memory.get("adapters", []):
        lora.detach(ctx.policy, a)
