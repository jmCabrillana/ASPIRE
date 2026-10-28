"""canon v16 = canon v15 + the dialect deletion rule of lexfacts_mask (composition of two library entries): in the rewrite,
'on top of the X' -> 'on the X' when the 4-gram 'top of the X' never occurs in the trained instructions (twin excluded) while
'on the X' does (text-level twin of the in-network pad-deletion, which E22 showed is exact). Found from the development panel:
v033's auto.py (canon only) scored 0/5 on goal_lan 8 while lexfacts_mask had 25/30 held-out."""
import importlib.util
from pathlib import Path

from steerlib import canon

DESCRIPTION = "canon v16: v15 + dialect deletion of out-of-dialect 'top of' (lexfacts_mask rule)"
_spec = importlib.util.spec_from_file_location("canon_v15_base", Path(__file__).with_name("canon_v15.py"))
_v15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v15)
_twin = _v15._twin


def delete_rule(sentence, ex):
    d = canon.Dialect(ex)
    ws = sentence.split()
    out, i = [], 0
    while i < len(ws):
        if ws[i:i + 3] == ["on", "top", "of"] and i + 4 < len(ws) + 1 and i + 3 < len(ws) and ws[i + 3] == "the" \
                and i + 4 < len(ws):
            noun = ws[i + 4]
            if not d.seen(["top", "of", "the", noun]) and d.seen(["on", "the", noun]):
                out += ["on"]; i += 3; continue
        out.append(ws[i]); i += 1
    return " ".join(out)


def rewrite(instruction, ex, log=lambda k, v: None):
    return delete_rule(_v15.rewrite(instruction, ex, log), ex)


def setup(ctx):
    ex = _twin(ctx.task.suite, ctx.task.task_id)
    ctx.instruction = rewrite(ctx.task.instruction, ex, ctx.log)
    ctx.log("instruction", ctx.instruction)
