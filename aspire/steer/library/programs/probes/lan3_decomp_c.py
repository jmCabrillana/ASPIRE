"""FEASIBILITY PROBE ONLY (hand-written sub-instructions for goal_lan 3; not a candidate): does sequencing two sub-instructions
the policy can ground ('open the top drawer of the cabinet' then 'put the bowl in the top drawer of the cabinet') work at all?
Switch at step 130 (drawer opening takes 120-133 steps in successful goal_lan 0 runs)."""
import os
SUB2 = "open the top drawer and put the bowl"   # canon v2 rewrite of the paraphrase
if os.environ.get("SUB2_FILE"):
    SUB2 = open(os.environ["SUB2_FILE"]).read().strip()
SWITCH = int(os.environ.get("SWITCH", "130"))


def setup(ctx):
    ctx.instruction = "open the top drawer of the cabinet"


def program(t, obs, internals, ctx):
    if t >= SWITCH and ctx.memory.get("phase") != 2:
        ctx.memory["phase"] = 2
        ctx.instruction = SUB2
        ctx.log("switch", t)
    return []
