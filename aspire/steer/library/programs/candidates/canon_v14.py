"""canon v14 = canon.normalize (general English normalisation) applied first, then v13: canon v12 + deterministic dialect noun-phrase completion (v13): after the rewrite, insert the dialect's completions:
a left modifier that the trained instructions of the same suite always put before a noun ('black' before 'bowl'), and a
right phrase that the trained instructions put after a noun more than half the time ('of the cabinet' after 'drawer').
Twin excluded. Motivation: E04 ('open the top drawer of the cabinet' reaches the drawer, 'open the top drawer' does not), E25
(every function word matters)."""
import importlib.util
from pathlib import Path

from steerlib import canon

DESCRIPTION = "canon v14: general English normalisation (politeness, wishes, declaratives, separable particles) + v13"
_spec = importlib.util.spec_from_file_location("canon_v12_base", Path(__file__).with_name("canon_v12.py"))
_v12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v12)


def complete(sentence, exclude):
    d = canon.Dialect(exclude)
    ds = canon.Dialect(exclude)
    ds.sents = [w for (s, i), w in zip(canon._suite_index(exclude), ds.sents) if s == exclude[0]]
    left, _ = canon.completions(ds, min_frac=0.9, min_count=1)
    _, right = canon.completions(d)
    multi = {w for a, b in canon.MULTIWORD for w in a + b}
    right = {w: ph for w, ph in right.items() if w not in multi}
    ws = sentence.split()
    out = []
    for i, w in enumerate(ws):
        if w in left and (not out or out[-1] in ("the", "a", "both")):          # only bare nouns (no modifier yet)
            out.append(left[w])
        out.append(w)
        if w in right and tuple(ws[i + 1:i + 1 + len(right[w])]) != right[w] and (i + 1 >= len(ws) or ws[i + 1] != "of"):
            out += list(right[w])
    ws, out = out, []                    # second pass: left completions on nouns inserted by right completions
    for w in ws:
        if w in left and (not out or out[-1] in ("the", "a", "both")):
            out.append(left[w])
        out.append(w)
    return " ".join(out)


def setup(ctx):
    ctx.task.instruction = canon.normalize(ctx.task.instruction)
    ctx.log("normalized", ctx.task.instruction)
    _setup13(ctx)


def _setup13(ctx):
    _v12.setup(ctx)
    ex = _v12._twin(ctx.task.suite, ctx.task.task_id)
    new = complete(ctx.instruction, ex)
    ctx.log("completed", {"from": ctx.instruction, "to": new})
    ctx.instruction = new
