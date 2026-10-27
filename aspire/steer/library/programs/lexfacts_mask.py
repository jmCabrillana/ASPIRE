"""Stored lexical facts (weight edits, lexfacts.py) + in-network token DELETION by attention masking.

E15: the expert reads the instruction token by token (each token's own KV). A word is removed inside the network by marking its
tokens as padding in embed_prefix: attention ignores them AND the rotary positions close the gap (position ids = cumsum of the
pad mask). E22: plain attention masking restores only 61% (the position gap costs ~40%); pad-deletion is exact (R 0.995).
Deletion rule (general, dialect-based): a multi-word preposition 'on top of the X' is reduced to 'on the X' when the 4-gram
'top of the X' never occurs in the trained instructions (target's twin excluded) while 'on the X' does.
"""
import json
from pathlib import Path

import torch

from steerlib import analysis as A, canon

DESCRIPTION = "stored lexical weight edits + attention-mask deletion of out-of-dialect multi-word preposition parts"
EDITS = Path(__file__).resolve().parents[2] / "edits"
MULTI = [((" on", " top", " of"), 2)]          # pattern, number of trailing tokens to delete before ' the X'


def _twin(suite, task_id):
    for suffix in ("_lan", "_swap", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    toks = A.tokens(ctx.instruction)
    facts = json.loads((EDITS / "facts.json").read_text())
    used, seen = [], set()
    for name, f in facts.items():
        trig = tuple(f["trigger_tokens"])
        if f["source"] == [ctx.task.suite, ctx.task.task_id] or trig in seen or not all(t in toks for t in trig):
            continue
        d = torch.load(EDITS / f"{name}.pt")
        ctx.edit_weights(ctx.vlm_layers[d["layer"]].mlp.down_proj.weight, lambda W, dd=d["delta"]: W + dd.to(W.device, W.dtype))
        used.append(name); seen.add(trig)
    d = canon.Dialect(_twin(ctx.task.suite, ctx.task.task_id))
    drop = []
    for i in range(len(toks) - 4):
        for pat, n in MULTI:
            if tuple(toks[i:i + len(pat)]) == pat and toks[i + len(pat)] == " the":
                noun = toks[i + len(pat) + 1].strip()
                if not d.seen(["top", "of", "the", noun]) and d.seen(["on", "the", noun]):
                    drop += [128 + i + 1 + k for k in range(n)]
    if drop:
        model = ctx.vla.model
        orig = model.embed_prefix
        idx = [p - 128 for p in drop]

        def embed_prefix(images, img_masks, lang_tokens, lang_masks, state=None):
            lang_masks = lang_masks.clone()
            lang_masks[:, idx] = False
            return orig(images, img_masks, lang_tokens, lang_masks, state=state)
        ctx.patch(model, "embed_prefix", embed_prefix)
    ctx.log("facts", used); ctx.log("masked_positions", drop)
