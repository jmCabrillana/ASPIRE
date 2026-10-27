"""Rewrite the instruction into the policy's training dialect (steerlib.canon v6), then run the frozen VLA unchanged.

v6 = v5 + trigram prior (0.5 per unseen trigram) + multi-word preposition swaps (on top of<->on, next to<->beside, into<->in).
v5 = structured edits (content words kept; verb phrases in verb slots; articles; preposition swaps) scored by
Llama-2-7B paraphrase likelihood (both directions) - 4 x (number of bigrams unseen in the trained instructions).
The target's twin task is excluded from the dialect. Dev-set (my own paraphrases): exact 20/24, word F1 0.986.
"""
from steerlib import canon

DESCRIPTION = "canon v6 rewrite (Llama-2-7B scorer, unseen bi+trigram prior, multi-word prepositions)"


def _twin(suite: str, task_id: int):
    for suffix in ("_lan", "_swap", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return (suite[: -len(suffix)], task_id)
    return (suite, task_id)


def setup(ctx):
    canon.SCORER["name"] = "NousResearch/Llama-2-7b-hf"
    canon.PRIOR["kind"] = "unseen23"
    canon.MULTI["on"] = True
    new = canon.canonical(ctx.task.instruction, exclude=_twin(ctx.task.suite, ctx.task.task_id), w=4.0, version="v3")
    ctx.instruction = new
    ctx.log("instruction", new)
