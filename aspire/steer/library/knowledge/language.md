# How smolvla_libero reads the instruction (mechanistic findings)

Model facts (checked in code): prefix = 64 agentview tokens, 64 wrist tokens, the instruction tokens (padded to the longest in
the batch: for a single instruction exactly its token count, newline appended by the processor), 1 state token. VLM = first
16 of SmolVLM2-500M's 32 layers, fine-tuned (train_expert_only = false). Language token positions: 128 .. 128+n-1.
`steer.lab.Lab.span("lang")` assumes 48 tokens; use `steerlib.analysis.spans(instruction)`.

## Paraphrase failures are lexical (JOURNAL E02)
Contextual nearest neighbours (residual stream, every VLM layer, goal scene) of paraphrase tokens among all tokens of the
trained instructions (twin task excluded):
- verbs the policy never saw ("switch", "move", "slide") are represented like OBJECT NOUNS (compartment, drawer, pots;
  cos 0.4-0.6); the model has no synonym structure left (the 16 fine-tuned layers lost general English).
- sentence-initial "place" (token "place", no leading space) is a different token from the trained " place" (second verb
  of pick-and-place); cos 0.6-0.76.
- possessive "’" maps to the newline token (end of instruction) at L4-L10; "s" to and/of.
- known words keep cos 0.93-1.00 at every layer: the rest of the sentence is read normally.
Consequence: the paraphrase does not pick a wrong known skill (E01: its chunk is 41-80 cm from every other skill's chunk),
it produces an unfamiliar mixture (lan7: a grasp loop at a fixed spot).

## What fixes it
Translate the instruction into the training dialect before it reaches the VLM (steerlib/canon.py). The general-English
knowledge must come from outside the policy: base SmolVLM2 (500M) is enough for single verbs, Llama-2-7B (local HF cache)
scores paraphrase equivalence much better (dev F1 0.944 -> 0.986).
Simulator (dev eps 0-9): goal_lan 7 0/10 -> 10/10 ('turn on the stove'), goal_lan 0 2/10 -> 7/10.
Limits: a verb that only the target's twin task uses (push for goal_lan 5) cannot be produced by the dialect rule.

## In-network fixes (text unchanged) and their limits (E15-E23)
- The action expert reads the instruction as (token KV, rotary position) pairs through its cross-attention (odd) layers
  (E15, E22, E23). VLM-side mixing across tokens is irrelevant for the verb (patching all non-verb tokens: ~0%).
- SUBSTITUTIONS can be fixed by a weight edit of VLM layer 1's down projection (key = unknown word in context, value = the known
  word's residual): fitted on 2 observations it matches activation patching (E16); learned once, it transfers to other
  sentences (E18) and tasks: 'place'->'put' learned on one sentence fixes goal_lan 1/2/4/9 (sim 19/20, 19/20, 17/20, 14/20).
- DELETIONS are exact by marking tokens as padding (attention + positions), E22 (R 0.995); goal_lan 8 18/20 with the
  place->put fact + deleting out-of-dialect 'top of'.
- INSERTIONS / REORDERINGS cannot be done by weight edits (they need new tokens or positions): spatial_lan 1 (lift -> pick UP
  THE ...) and goal_lan 0 (possessive reordering) fail in-network (0/5, 0/10) but work with the text rewrite (19/20, 18/20).
- Neurons: the correction for one word at L1 is dense (>256 neurons), E19.
- Destination (goal scene) is a continuous function of the destination noun's representation, also in closed loop
  (dest_mix probe: alpha 0.5 releases the bowl between plate and stove). The pick target is not steerable through the object
  noun (goal_swap 8 transplant probe).
