"""In-network language fix: the VLA keeps receiving the ORIGINAL paraphrase; a weight edit of VLM layer 1's MLP down projection,
fitted live on the episode's first observation, makes the layer write the known word's residual for the unknown word's key.

Mechanism (E15): the action expert reads a verb from the verb token's own KV entries at every VLM layer; nothing else in the
sentence carries it. E16: a multi-pair edit at layer 1 fitted on 2 observations restores 0.71-0.95 of the rewrite's behaviour on
4 new observations (equal to activation patching at L1), side effect on an unrelated instruction 0.12-0.20 cm.
Unknown -> known word mapping: steerlib.canon v7 (general rule, twin task excluded). Sample efficiency: 1 observation, 0 episodes.
"""
import os

from steerlib import canon, lexedit

DESCRIPTION = "live lexical weight edit at VLM layer 1 (unknown word -> canon word), original text kept"
LAYER = int(os.environ.get("LEX_LAYER", "1"))


class _Lab:                         # minimal Lab for lexedit.collect (uses the program's policy)
    def __init__(self, policy):
        from steer.lab import Lab
        self._lab = Lab(policy)
        self.policy = policy

    def chunk(self, o, ins):
        return self._lab.chunk(o, ins)


def _twin(suite, task_id):
    for suffix in ("_lan", "_swap", "_object", "_task", "_env"):
        if suite.endswith(suffix):
            return suite[: -len(suffix)], task_id
    return suite, task_id


def setup(ctx):
    canon.SCORER["name"] = "NousResearch/Llama-2-7b-hf"
    canon.PRIOR["kind"] = "unseen234"                        # canon v7 (as programs/candidates/canon_v7.py)
    canon.MULTI.update(on=True, compound=True)
    ctx.memory["c"] = canon.canonical(ctx.task.instruction, exclude=_twin(ctx.task.suite, ctx.task.task_id), w=4.0, version="v3")
    ctx.log("rewrite_used_for_edit", ctx.memory["c"])


def program(t, obs, internals, ctx):
    if "edited" not in ctx.memory:
        ctx.memory["edited"] = True
        x, c = ctx.instruction, ctx.memory["c"]
        if lexedit.align(x, c):
            lab = _Lab(ctx.policy)
            delta = lexedit.fit(lab, [obs], x, c, L=LAYER)
            lexedit.apply_ctx(ctx, delta, L=LAYER)
            ctx.log("edit", {"layer": LAYER, "pairs": lexedit.align(x, c), "norm": float(delta.norm())})
        ctx.policy.reset()          # the fit ran the policy: clear its action queue so the episode starts fresh
    return []
