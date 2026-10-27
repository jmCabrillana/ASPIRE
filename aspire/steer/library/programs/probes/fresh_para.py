"""PROBE: evaluate a FRESH paraphrase (library/tools/fresh_paraphrases.json, index FRESH_I) on its trained task, either frozen
(FRESH_MODE=frozen) or through canon_v13's rewrite with the task excluded from the dialect (FRESH_MODE=canon)."""
import importlib.util, json, os
from pathlib import Path

from steerlib import canon

I, MODE = int(os.environ["FRESH_I"]), os.environ["FRESH_MODE"]
suite, task, para = json.loads((Path(__file__).resolve().parents[2] / "tools" / os.environ.get("FRESH_FILE", "fresh_paraphrases.json")).read_text())[I]
_spec = importlib.util.spec_from_file_location("v13", Path(__file__).resolve().parents[1] / "candidates" / "canon_v13.py")
V13 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(V13)
if MODE == "canon14":
    _spec14 = importlib.util.spec_from_file_location("v14", Path(__file__).resolve().parents[1] / "candidates" / "canon_v14.py")
    V13 = importlib.util.module_from_spec(_spec14); _spec14.loader.exec_module(V13)


def setup(ctx):
    assert ctx.task.suite == suite and ctx.task.task_id == task
    if MODE == "frozen":
        ctx.instruction = para
    elif MODE == "canon14":
        ctx.task.instruction = para; ctx.task.suite = suite + "_lan"
        V13.setup(ctx); ctx.task.suite = suite
    elif MODE == "facts":                    # in-network: keep the fresh text, apply STORED facts (never one from this task)
        import json as _j, torch
        from steerlib import analysis as A
        ed = Path(__file__).resolve().parents[2] / "edits"
        facts = _j.loads((ed / "facts.json").read_text()); toks = A.tokens(para); used = []
        for name, f in facts.items():
            if f["source"][1] == task and f["source"][0].startswith(suite): continue
            if all(t_ in toks for t_ in f["trigger_tokens"]) and name not in used:
                d = torch.load(ed / f"{name}.pt")
                ctx.edit_weights(ctx.vlm_layers[d["layer"]].mlp.down_proj.weight, lambda W, dd=d["delta"]: W + dd.to(W.device, W.dtype))
                used.append(name)
        ctx.instruction = para; ctx.log("facts", used)
    elif MODE == "lexedit":                  # in-network: keep the fresh text, live layer-1 edit toward the rewrite
        ctx.task.instruction = para; ctx.task.suite = suite + "_lan"
        V13.setup(ctx); ctx.memory["c"] = ctx.instruction
        ctx.task.suite = suite; ctx.instruction = para
    else:
        ctx.task.instruction = para          # the rewrite pipeline sees only the fresh paraphrase
        ctx.task.suite = suite + "_lan"      # so _twin() excludes the task itself from the dialect
        V13.setup(ctx)
        ctx.task.suite = suite
    ctx.log("fresh", {"para": para, "mode": MODE, "given": ctx.instruction})


def program(t, obs, internals, ctx):
    if MODE != "lexedit" or ctx.memory.get("edited"):
        return []
    ctx.memory["edited"] = True
    from steer.lab import Lab
    from steerlib import lexedit
    x, c = ctx.instruction, ctx.memory["c"]
    if lexedit.align(x, c):
        lab = Lab(ctx.policy)
        delta = lexedit.fit(lab, [obs], x, c, L=1)
        lexedit.apply_ctx(ctx, delta, L=1)
        ctx.log("edit", {"pairs": lexedit.align(x, c), "c": c})
    ctx.policy.reset()
    return []
