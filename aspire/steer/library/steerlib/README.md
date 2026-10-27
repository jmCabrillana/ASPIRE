# steerlib index

Every reusable function of the steering library, with the evidence behind it. "Seed" rows were written by hand to start the
library and have no evidence yet. Evidence numbers: development episodes unless marked "held-out" (driver, episodes 20-49).

## canon.py: instruction canonicalisation (translate into the policy's training dialect)

| function | what | evidence |
|---|---|---|
| `canon.canonical(instr, exclude=(trained suite, task id), w, version)` | noisy-channel rewrite: argmax w*prior(c) + log P_LM(x\|c) + log P_LM(c\|x); content words kept, edits only in verb slots / function words; the twin task is excluded from the dialect; results cached in `notes/canon_cache.json` | goal_lan 7: v2 20/20 dev, **held-out 29/30 vs frozen 1/30 (PROMOTED)**; goal_lan 0: v5 18/20, **held-out 26/30 vs 11/30 (PROMOTED)**; goal_lan 6 v5 11/20 vs 2/10; goal_lan 1/2/4: 9/10, 10/10, 10/10 vs 7, 7, 8 of 10; spatial_lan (v6) 1: 19/20, 5: 16/20, 6: 16/20 (frozen 0/10 each); spatial_lan 3 (v7) 4/5 vs 0/10 |
| settings | `SCORER["name"]` (SmolVLM2-500M default, `NousResearch/Llama-2-7b-hf` = v4+), `PRIOR["kind"]` (`bigram`, `unseen` = v5, `unseen23` = v6, `unseen234` = v7), `MULTI` (`on`: multi-word prepositions v6; `compound`: verb phrase + article v7; `complete`: dialect noun-phrase completion v8; `beam`: beam search v9, worse) | dev set (`tools/canon_devset.json`, 24 own paraphrases): v3 16/24 exact, v4 19/24, v5 20/24 (F1 0.986), v6 20/24, v7 19/24, v9 18/24 |
| `canon.Dialect(exclude)` | bigram / n-gram statistics of the trained instructions minus the twin; `unseen`, `unseen_n`, `logp` | |
| `canon.possessive(words)` | "the A's B" -> "the B of the A" (general English) | goal_lan 0 (cabinet's middle drawer) |
| `canon.frame_fill(original, exclude, extra, top)` (v12) | dialect sentence frames (content words -> slots) filled in order with the paraphrase's content words (+ dialect completions such as 'black' before 'bowl'), ranked by Llama equivalence; used as fallback when a verb stays untranslated (programs/candidates/canon_v12.py) | spatial_lan 7/9: fills 'pick up the black bowl on the stove/cabinet and place it on the plate' (sim pending) |
| `canon.snap_tail(rewrite, original, exclude)` | clause retrieval for the tail after 'and' | no change on spatial/goal (Llama prefers copies) |
| `canon.Dialect.seen(ngram)` | n-gram occurrence in the trained instructions | used by lexfacts_mask deletion rule |
| `canon.completions(dialect)` | noun-phrase completions learned from n-gram statistics ('black' before 'bowl', 'of the cabinet' after 'drawer') | changes only goal_lan 3 rewrite; untested in sim |

Limits: verbs that only the twin uses cannot be produced (push for goal_lan 5: 0/10); 'on top of the plate' (goal_lan 8) is kept
by the LM (0/10); goal_lan 3 needs a two-step skill (drawer opens with 'open the top drawer of the cabinet', placing fails).

## lexedit.py + residual.py: in-network language fixes (text unchanged)

| function | what | evidence |
|---|---|---|
| `residual.record(vla, layers)` / `patch(vla, spec)` / `add(vla, spec)` / `install(ctx, spec)` | record / replace / add the VLM residual stream after layer L at chosen token positions (prefix pass; the expert reads the patched KV) | E15 map (verb token at L0: 96% of a rewrite; no other token carries it) |
| `lexedit.align(x, c)` | token positions of x that differ from the rewrite c (replace/delete blocks) | |
| `lexedit.collect / fit(lab, obs, x, c, L)` | keys (neuron activations of x's differing tokens) and values (the residual the rewrite has there) -> multi-pair closed-form edit of layer L's down projection (steer/edit.py) | E16: L1 edit from 2 obs reaches the patching bound (R 0.71-0.95), side effect 0.12-0.20 cm |
| `lexedit.apply_ctx(ctx, delta, L)` / `lexedit.applied(vla, delta, L)` | apply an edit for the episode (ctx.edit_weights, restored) / offline | |
| stored facts: `library/edits/*.pt` + `facts.json` (tools/make_fact.py) | 'place'->'put' (from goal_lan 9 or 1), 'switch'->'turn' (goal_lan 7); 1 sentence + 2 observations each | E18 transfer R 0.69-0.82; SIM lexfacts (transfer only): goal_lan 1 19/20 (frozen 7/10), 2 19/20 (7/10), 4 17/20 (8/10), 9 14/20 (5/10) |
| pad-deletion (in `programs/candidates/lexfacts_mask.py`) | delete words inside the network by marking them as padding in embed_prefix (attention + rotary positions close up) | E22: exact (R 0.995) vs plain attention masking R 0.61; SIM goal_lan 8 18/20 (frozen 1/10) with the place->put fact |
| programs | `lexedit_live.py` (fit live on the first observation from the canon rewrite): goal_lan 7 19/20, 9 10/10; fails on structural paraphrases (spatial_lan 1/5, goal_lan 0: insertions/reorderings) | sim |

## analysis.py: offline lab helpers

| function | what | evidence |
|---|---|---|
| `analysis.load()` | (policy, Lab, Neurons) on the GPU | |
| `analysis.tokens(instr)`, `analysis.spans(instr)` | instruction tokens as the policy sees them; prefix slices (177 tokens: image 0:64, wrist 64:128, lang 128:128+n, pad, state 176) | verified from attention shapes |
| `analysis.path(chunk)`, `summary`, `dist` | cumulative eef displacement of a chunk (0.0095 m per action unit per step, measured), summaries, path distance | |

## bank.py: activation banks

| function | what | evidence |
|---|---|---|
| `bank.build(lab, files, layers, where, extra_labels)` | steer.interp Bank with real-token language segments plus first/last-token keys and custom labels | E11 probes |

## attention.py: attention interventions

| function | what | evidence |
|---|---|---|
| `attention.patch(m, bias_fn, record)` / `patch_ctx(ctx, bias_fn)` | add a logit bias per (layer, query range, key range); optional recording of attention probabilities | E06: image-attention boosts do not retarget swap reaches (negative); E07: noun-token attention does not localise objects (negative) |
| `attention.segment_bias(segments, amount, layers, queries)` | bias on key segments for expert or prefix queries | E06 |

## scene.py: image-only perception (agentview)

| function | what | evidence |
|---|---|---|
| `scene.patch_features(vla, img)` | the policy's own SigLIP patch features (32x32x768) in raw image orientation | E08: finds the moved bowl in goal_swap 1 (5/5 frames) |
| `scene.region(F, px, tau, radius)` + `scene.track(F_ref, mask, F_live)` | grow an object region in the reference by feature similarity; feature-template matching for its displacement | E14 (tau 0.3, radius 4): plate displacement error 2 cm (3/3) on spatial_swap 0, 1-5 cm on 2/3 of spatial_swap 1; fails for very large moves (goal_swap 1 stove, 63 cm) |
| swap programs (`programs/candidates/carry_handback2.py`, `approach_handback.py`) | VLA picks (or program approaches the tracked pick object and hands back); program carries above the tracked destination and descends; hands back to the VLA with its state shifted by the tracked displacement so its own final alignment/release runs with the cameras seeing the target | HELD-OUT (frozen 0/30 each): spatial_swap 2 20/30, 4 11/30, 5 8/30 (carry_handback2); 0 8/30, 9 17/30 (carry_handback4 = + nudge-and-retry); dev goal_swap 9 3/15 (approach_handback); fails when tracking fails (E30, goal-scene destinations) |
| `scene.project(X)`, `scene.backproject(px, z)` | fixed agentview camera model (sensor calibration; accepted by the operator for now) | gripper annotations within 4 px in u; grasp points within 5-15 px of grasped objects' footprints |
| `scene.table`, `footprint`, `blobs`, `hist`, `find_moved`, `locate`, `disc_mask`, `patch_mask` | colour/footprint utilities | footprints useful for references; colour matching failed (bowl vs stove) |

## gates.py / interventions.py (seed)

| function | what | evidence |
|---|---|---|
| `gates.gripper_closed(obs)` | gripper closed | used by carry_to_target |
| `gates.phase(obs, memory)` | reach / hold / after | used by virtual_frame |
| `gates.attention_share(internals, layer, segment)` | attention share of the action tokens on a segment, previous chunk | seed |
| `interventions.push(ctx, direction, layer, sigmas)` | additive steering term from a stored direction | seed (the pilot's mean-difference directions: no effect) |
| `interventions.ablate(ctx, module, unit, scale)` | remove or amplify one direction in a module's output | seed |
| `interventions.scale_output(ctx, module, factor)` | rescale a module's output | seed |

## Neuron facts
| neuron | effect | evidence |
|---|---|---|
| expert L4 #1855 | gripper switch: steer.lab.Neurons dose -1 -> no closing, +2 -> immediate closing (6/6 obs) | E28 |

## Negative results worth remembering (details in notes/JOURNAL.md)
- E25: deleting 'the' moves the first chunk as much as deleting a verb (2.6x noise): wordings are memorised patterns.
- E26: token-level representation familiarity predicts rewrite success only weakly (Spearman 0.34-0.46).
- E19: single-neuron / sparse pushes cannot write a word: the switch->turn correction needs >256 neurons at L1.
- E21: in spatial scenes the destination word is ignored (hard-wired plate spot): noun mixing cannot redirect spatial_swap.
- E03: seed-consistency of action chunks does not detect instructions the policy does not understand.
- E10: shifting the proprio state does not move the first-chunk reach backwards (saturation); virtual_frame on goal_swap 0/5.
- E12: pushing along linear-probe directions for the reach target (R2 0.9) moves the reach by <= 2 cm: read-outs, not levers.
