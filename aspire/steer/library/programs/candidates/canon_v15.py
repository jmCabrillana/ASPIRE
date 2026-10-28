"""canon v15 = canon v14 with two fixes, both general rules:
1. the v12 frame-fill fallback fired when a NOUN followed 'and' ('between the plate and ramekin': canon.verb_slots treats every
   word after 'and' as a verb slot), replacing a correct rewrite by a wrong frame ('... next to the plate ...', spatial_lan 0
   0/10). A word the trained instructions use after 'the' (a noun) no longer counts as an untranslated verb.
2. coordination-article normalisation (canon.coord_article: 'the A and B' -> 'the A and the B' for a bare noun B; E25).
Otherwise identical to v14 (normalize -> v7 rewrite -> frame-fill fallback -> dialect completions)."""
import importlib.util
from pathlib import Path

from steerlib import canon

DESCRIPTION = "canon v15: v14 with the noun-after-'and' fallback fix + determiner repetition in coordinated noun phrases"
_spec = importlib.util.spec_from_file_location("canon_v14_base", Path(__file__).with_name("canon_v14.py"))
_v14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v14)
_twin = _v14._v12._twin


def rewrite(instruction, ex, log=lambda k, v: None):
    canon.SCORER["name"] = "NousResearch/Llama-2-7b-hf"
    canon.PRIOR["kind"] = "unseen234"
    canon.MULTI.update(on=True, compound=True)
    x = canon.normalize(instruction)
    r = canon.canonical(x, exclude=ex, w=4.0, version="v3")
    d = canon.Dialect(ex)
    verbs = {p[0] for p in canon.dialect_verbs(d)}
    nouns = {s[i + 1] for s in d.sents for i in range(len(s) - 1) if s[i] == "the"}
    ws = r.split()
    if any(ws[i] not in verbs and ws[i] not in nouns for i in canon.verb_slots(ws)):
        fills = [c for s, c in canon.frame_fill(x, exclude=ex, extra=[], top=1, suite_completions=True, prior_w=4.0)]
        if fills:
            log("fallback", {"canon": r, "frame_fill": fills[0]})
            r = fills[0]
    r = _v14.complete(r, ex)
    return canon.coord_article(r, ex)


def setup(ctx):
    ex = _twin(ctx.task.suite, ctx.task.task_id)
    ctx.instruction = rewrite(ctx.task.instruction, ex, ctx.log)
    ctx.log("instruction", ctx.instruction)
