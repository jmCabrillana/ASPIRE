"""Apply stored lexical facts (library/edits/facts.json): weight edits of a VLM layer's MLP down projection, each teaching the
policy that an unknown word reads as a known one (E16, E18: a fact learned from ONE sentence and 2 observations transfers to
other sentences with the same word, R 0.69-0.82 offline). The instruction text is unchanged. A fact is used when its trigger
token occurs in the instruction, never when it was learned from this same task (transfer only). One fact per trigger.
"""
import json
from pathlib import Path

import torch

from steerlib import analysis as A

DESCRIPTION = "stored lexical weight edits (learned facts) whose trigger tokens occur in the instruction"
EDITS = Path(__file__).resolve().parents[2] / "edits"


def setup(ctx):
    facts = json.loads((EDITS / "facts.json").read_text())
    toks = A.tokens(ctx.instruction)
    used, seen = [], set()
    for name, f in facts.items():
        trig = tuple(f["trigger_tokens"])
        if f["source"] == [ctx.task.suite, ctx.task.task_id] or trig in seen:
            continue
        if all(t in toks for t in trig):
            d = torch.load(EDITS / f"{name}.pt")
            layer = ctx.vlm_layers[d["layer"]].mlp.down_proj.weight
            ctx.edit_weights(layer, lambda W, dd=d["delta"]: W + dd.to(W.device, W.dtype))
            used.append(name); seen.add(trig)
    ctx.log("facts", used)
