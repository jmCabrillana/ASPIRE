# SESSION 2 (2026-10-27 16:40-00:40) - read this block first

New verified: **spatial_lan 0 canon_v15 22/30 vs frozen 10/30 (PROMOTED)** -> all 10 spatial_lan tasks promoted.
auto.py now = canon_v16 for *_lan (v15: fallback fix + coordination article; v16: + dialect deletion of 'top of', goal_lan 8 9/10).
Development panel (tools/panel.py, notes/panel.md; 16 fixed tasks x eps 0-4): frozen 5/80, v033 (end of session 1) 35/80.
The current version was NOT scored (GPU starved, see below): run `tools/panel.py <latest version>` first next session.
Discovery ledger: knowledge/discoveries.md (method, data, cost for every intervention; operator paper target 1).

## Main finding: the action expert's first layer carries a LINEAR, SHARED planar motion code (destination and reach)
- E37 (gradient search, 9 sites): one additive vector at expert L0 moves the predicted place point by 79% of 8 cm on 5 unseen
  tasks (0.2 cm off-axis); the same search in any VLM layer <= 6%. E37b closed loop (gated on holding): the VLA carries and
  RELEASES BY ITSELF 7.1-7.9 cm from its memorised place point (3/3).
- E40 (control view): the Jacobian of the chunk endpoint w.r.t. an added vector at expert L0/L4/L8 is the SAME for every
  observation (cos 0.93) and rank ~2. Its pseudo-inverse is a closed-form destination interface, no training, ~1 GPU min:
  gain 1.00 / 0.3 cm off-axis at 8 cm, 0.97 at 20 cm on unseen tasks (gradient-trained interface E39: 0.60; single vector 0.79).
- E41: reaching observations share the code (Jacobian cos 0.86 with the holding one; pinv gain 0.92): the same interface can
  retarget the GRASP (untested in closed loop; for goal_swap where the pick object moved).
- dest_swap_fb (programs/candidates): the VLA performs every motion and the release itself on spatial_swap 2; the program only
  (a) tracks the plate in the image, (b) feeds the tracked offset into the E40 interface while holding, (c) corrects the
  interface input by integral feedback near the goal. Open loop: x drifts, no release (0/2). With feedback: 3/5 dev (eps 1-3;
  frozen 0/10). VERIFY requested 22:50 (driver running at the end of the session). A gated release through the gripper-switch
  neuron (E28) is in v3 but never fired (the VLA released by itself in every success).
- dest_swap_fb2.py (v4b): feedback engages after the transport (not by distance) and the lever is LATCHED until the release:
  spatial_swap 4 ep 0 success (v3 failed: its 'lifted above grasp height' gate switched the lever off during the descent because
  the bowl is picked from the drawer above the plate - a clean example of why the WHEN of an intervention matters).

## Other results
- goal_lan 3 skill composition (open_then_place.py): sequencing works; the opened drawer is located from the robot's own pull
  motion; a sub-skill must restart from the start pose INCLUDING orientation (else the VLA drifts or grasps the wine bottle);
  but the rim-grasped bowl slips during program carries, 0/5 in every variant. Idea for next time: retarget the VLA's OWN carry
  ('put the bowl on top of the cabinet') with the E40 interface toward the self-measured drawer, instead of a program carry.
- Operator's paper targets: (1) ledger; (2) gating: open-loop vs feedback on the internal lever (0/2 vs 3/5 on the same task);
  always-on vs gated not yet run; (3) general entries: E40 interface + tracking is task-independent; (4) panel tool.
- GPU: from ~18:05 an outside graphics workload (no CUDA process visible; Firefox/GNOME) time-sliced the GPU: matmul 4 TFLOPS,
  simulator episodes 5-10 min instead of 5-20 s. Batch lab work (16 obs per forward: 9x throughput). Operator notified.

## Next steps (session 3)
1. Score the panel for the latest version; check VERIFY dest_swap_fb spatial_swap 2.
2. dest_swap_fb2 (latched, v4b) on spatial_swap 2 (all eps), 0, 4, 5 (y shifts), 9, 7 (x shifts); verify the best; then the
   reach interface (directions/reach_pinv_L0.pt) for goal_swap (moved bowl). Compare with carry_handback4 on the same eps.
3. Gating ablations for paper target 2: GATE=always; FB=0 vs 0.5 on more episodes; release lever with a radius matched to the
   tracking error (~2-3 cm).
4. goal_lan 3: VLA carry retargeted by the interface into the self-measured drawer.
5. Sparse lever search (steerlib/levers.py, E38 script ready) on the destination code: how many neurons carry it?

# Code-as-Steering: state of the research (session 1, 2026-10-26 16:40 - 10-27 00:10)

Read this first; details and numbers in `JOURNAL.md`, mechanisms in `../knowledge/`, functions in `../steerlib/README.md`.

## Verified on held-out initial states 20-49 (driver, library/INDEX.md)
| task | program | frozen | program |
|---|---|---|---|
| goal_lan 7 | canon_rewrite (text rewrite, 'turn on the stove') | 1/30 | 29/30 |
| goal_lan 0 | canon_v5 | 11/30 | 26/30 |
| goal_lan 6 | canon_v5 | 6/30 | 18/30 |
| spatial_lan 1 | canon_v6 | 0/30 | 22/30 |
| spatial_lan 5 | canon_v6 | 0/30 | 24/30 |
| spatial_lan 6 | canon_v6 | 0/30 | 27/30 |
| goal_lan 7 | lexedit_live (IN-NETWORK, text unchanged, live layer-1 edit) | 1/30 | 29/30 |
| goal_lan 8 | lexfacts_mask (IN-NETWORK: stored fact from another sentence + pad-deletion) | 0/30 | 25/30 |
| goal_lan 9 | lexedit_live (IN-NETWORK) | 15/30 | 26/30 |
| spatial_lan 3 | canon_v7 | 1/30 | 24/30 |
| spatial_lan 2 | canon_v7 | 3/30 | 10/30 (p=0.03) |
| spatial_lan 4 | canon_v13 | 0/30 | 21/30 |
| spatial_lan 7 | canon_v12 | 0/30 | 24/30 |
| spatial_lan 8 | canon_v13 | 0/30 | 19/30 |
| spatial_lan 9 | canon_v12 | 0/30 | 20/30 |
| **spatial_swap 2** | carry_handback2 (VLA picks; program carries above the image-tracked plate; hands back with a virtual-frame state) | 0/30 | 20/30 |
| **spatial_swap 4** | carry_handback2 | 0/30 | 11/30 |
| **spatial_swap 5** | carry_handback2 | 0/30 | 8/30 |
| **spatial_swap 0** | carry_handback4 (+ nudge-and-retry) | 0/30 | 8/30 (v2: 3/30) |
| **spatial_swap 9** | carry_handback4 | 0/30 | 17/30 |
| **spatial_swap 7** | carry_handback4 | 0/30 | 18/30 |
| goal_lan 1 / 2 | lexfacts (in-network) | 27/30, 22/30 | 30/30, 26/30 (not significant: strong baselines) |
| spatial_lan 0 | canon_v7 | 10/30 | 15/30 (not significant) |
Not significant: approach_handback goal_swap 9 4/30 (p=0.056). All requests processed (library/INDEX.md).

## Development results not verified (eps 0-19)
goal_lan 4 lexfacts 17/20 dev (held-out 24/30 vs 23/30, n.s.); fresh paraphrases (8 new sentences, rewrite 0.25 -> 0.73; in-network 0.35 -> 0.75);
new styles with canon_v14 ('could you ...' 0/5 -> 4/5, 'I want ...' 3/5 -> 4/5, '... goes on ...' 3/5 -> 5/5).
Swap (hand-back hybrids: the VLA picks, the program carries above the image-tracked destination and descends, then hands
back to the VLA with a virtual-frame state so its own final alignment/release runs while the cameras show the real plate;
v4 adds nudge-and-retry when it hovers): dev spatial_swap 0 9/15 (v4), 2 5/5 (v4), 9 4/5 (v4), 4 8/15 (v2), 5 5/15 (v2),
7 2/5 (v4); 1, 3, 6, 8 ~0 (tracking). Frozen 0/10 everywhere. Earlier swap attempts (virtual frame,
open-loop carry, destination knob, place-only virtual frame) all 0.
Unsolved: goal_lan 3 (needs a 2-step skill), goal_lan 5 ('push' exists only in the twin), goal_swap (approach_handback under
test), spatial_swap tasks where tracking fails (E30).

## Generalisation to FRESH paraphrases (written before any result; trained goal/spatial tasks, task excluded from the dialect)
8/8 improved, mean success 0.25 frozen -> 0.73 with the canon_v13 rewrite (JOURNAL 20:55). In-network (live layer-1 edit,
text unchanged) on the 4 substitution paraphrases: 0.35 -> 0.75, equal to the text rewrite.

## What we learned (mechanisms)
1. Paraphrase failures are lexical: unknown verbs are represented like object nouns; possessives like end-of-sentence (E02).
2. The action expert reads the instruction as (token KV, rotary position) pairs through its cross-attention (odd) layers;
   the verb lives in the verb token's own KV at every VLM layer, no summary token (E15, E22, E23).
3. Hence in-network fixes: a closed-form layer-1 weight edit teaches 'switch'='turn' (fit on 1-2 observations; a fact
   learned on one sentence transfers to others, E16, E18); exact word deletion by padding (E22). They cannot insert or
   reorder words (structural paraphrases need the text rewrite).
4. Word identity is dense: >256 neurons needed (E19). Function words matter as much as verbs (E25): wordings are memorised.
5. Targets are memorised per instruction; the image barely moves them (E11, E17). The destination is a continuous function of
   the destination noun's representation (E20, closed loop too), but only among destinations trained with that object (E24),
   and spatial destinations are hard-wired (E21). Probe directions, state shifts and image-attention boosts are not levers.
6. Distilling one self-generated success into a LoRA adapter teaches the raw paraphrase (spatial_lan 1: 3/5 from 1 episode,
   4/5 from 3), but adapters do not transfer across tasks, and self-training on 3 successes did not beat the program.
   For swap, an adapter trained on 7 self-generated hybrid successes gives the VLA alone only 1/10 (program 8/15): few-demo
   fine-tuning is far below the program here.

## Sample efficiency of each learned component
| component | what it learns | data needed | result |
|---|---|---|---|
| canon rewrite (v5-v14) | translation into the training dialect | none (general LM + the trained instructions, twin excluded) | 9 verified PROMOTED tasks |
| lexical fact (edits/*.pt) | 'unknown word = known word' at VLM layer 1 | 1 sentence pair + 2 observations | transfers to other sentences/tasks (goal_lan 1/2/4/9; fresh 'set') |
| live lexical edit | same, fitted at run time | 1 observation (the episode's first) | goal_lan 7 29/30, goal_lan 9 26/30 held-out |
| pad-deletion rule | drop out-of-dialect words | none | goal_lan 8 25/30 held-out (with a fact) |
| LoRA adapter (lan) | paraphrase -> behaviour | 1 self-generated success (3 chunks), ~15 s training | spatial_lan 1: 3/5 (frozen 0/10); 3 eps 4/5 |
| LoRA adapter (swap) | new layout | 7 self-generated hybrid successes | 1/10 (program 8/15): not enough |
| swap references | memorised pick/place points + layout image | 3 frozen trained-suite episodes per task | carry_handback2: spatial_swap 0/2/4/5 5-8/15 dev |

## Entry point
`programs/candidates/auto.py` (dispatch by suite) + `tools/precompute_canon.py`; how-to in `knowledge/recipes.md`.

## Transparency note
Several general-rule rewrites coincide exactly with the original trained wording of the target (goal_lan 7 'turn on the stove';
spatial_lan 3, 4, 7, 9 via canon v7 / frame fill / completions). Every word comes from the paraphrase itself or from OTHER
trained sentences (twin excluded from all statistics); the operator accepted canon on this basis, flagged here for review.

## Rules learned the hard way
- Reserved: libero_object_lan, libero_object_swap, libero_10_lan (never touch). Programs never read sim state or BDDL.
- Canon must hit its cache in parallel simulator runs (two Llama-2-7B loads = CUDA OOM); precompute in one process.
- Queue chaining: never pkill/pgrep with a pattern that matches the shell running it (use bracket patterns, scripts).

## Next steps (for the next session)
1. goal_swap: perception is the bottleneck (several large moves per layout; E36). Try object-specific templates from
   footprints of EVERY object (not only the picked one), more reference frames, re-tracking during the episode, then
   approach_handback (pick + place hand-backs) - it already works when tracking is right (goal_swap 9 3/15).
2. spatial_swap: carry_handback2 is at 33-53% on 4 tasks; failures = tracking (E30/E34) and, with correct tracking, a
   few-cm offset after the hand-back: the VLA hovers and never releases because its wrist view does not show 'bowl over plate'
   (JOURNAL 22:55) -> add wrist-camera alignment before the hand-back. Voting (carry_handback3) did not add overall; combine voting with a consistency check against the VLA's own
   predicted place point.
3. In-network structural fixes: the policy reads (token, rotary position) pairs (E22); inserting tokens into padding slots with
   assigned position ids would make insertions possible inside the network (only deletions/substitutions are done so far).
4. goal_lan 3 (two-step skill whose second step only the twin teaches) and goal_lan 5 ('push' only in the twin): need a new
   mechanism (e.g. skill composition from other suites) - none found.
5. Swap self-improvement: rank-8 q/v LoRA on 7 hybrid successes gave 1/10; try MLP targets, higher rank, more episodes.
