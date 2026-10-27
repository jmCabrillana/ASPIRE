"""canon v7 rewrite, with a frame-retrieval fallback (v12): if the rewrite still has a word in a verb slot that the trained
dialect never uses as a verb (canon failed to translate the verb, e.g. 'lift'), use the best dialect FRAME filled in order with
the paraphrase's own content words (steerlib.canon.frame_fill: completions from the same suite, ranked by Llama equivalence +
4 x unseen-n-gram prior, as canon v7). Twin excluded. Dev: spatial_lan 7 7/10 (frozen 0/10) with the first fill version."""
from steerlib import canon

DESCRIPTION = "canon v7 + dialect frame-fill fallback when a verb stays untranslated"


def _twin(suite, task_id):
    for suffix in ("_lan", "_swap", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    canon.SCORER["name"] = "NousResearch/Llama-2-7b-hf"
    canon.PRIOR["kind"] = "unseen234"
    canon.MULTI.update(on=True, compound=True)
    ex = _twin(ctx.task.suite, ctx.task.task_id)
    r = canon.canonical(ctx.task.instruction, exclude=ex, w=4.0, version="v3")
    d = canon.Dialect(ex)
    verbs = {p[0] for p in canon.dialect_verbs(d)}
    ws = r.split()
    if any(ws[i] not in verbs for i in canon.verb_slots(ws)):
        fills = [c for s, c in canon.frame_fill(ctx.task.instruction, exclude=ex, extra=[], top=1, suite_completions=True,
                                                prior_w=4.0)]
        if fills:
            ctx.log("fallback", {"canon": r, "frame_fill": fills[0]})
            r = fills[0]
    ctx.instruction = r
    ctx.log("instruction", r)
