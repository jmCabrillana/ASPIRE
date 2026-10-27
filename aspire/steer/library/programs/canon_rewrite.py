"""Rewrite the instruction into the policy's training dialect (steerlib.canon), then run the frozen VLA unchanged.

Why: E02 showed that paraphrase words the fine-tuned VLM never saw (switch, slide, move, sentence-initial place, possessives)
are represented like unrelated object nouns, so the action expert gets no usable verb. canon uses the base SmolVLM2 language
model (general English) plus a bigram model of the trained instructions; the target's twin task is excluded from the dialect.
"""
from steerlib import canon

DESCRIPTION = "noisy-channel rewrite of the instruction into the trained dialect (content words kept)"


def _twin(suite: str, task_id: int):
    base = suite
    for suffix in ("_lan", "_swap", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            base = suite[: -len(suffix)]
    return (base, task_id)


def setup(ctx):
    new = canon.canonical(ctx.task.instruction, exclude=_twin(ctx.task.suite, ctx.task.task_id), w=2.0, version="v2")
    ctx.instruction = new
    ctx.log("instruction", new)
