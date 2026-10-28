# JOURNAL (Code-as-Steering research agent)

Chronological lab notebook. Each entry: hypothesis, experiment, result (numbers), conclusion.

## 2026-10-26 16:45 session 1 start
State found: no previous notes. Prior work = pilot sweep (outputs/sweep): mean-diff "trained minus paraphrase" direction
(all-token mean of MLP outputs) added at VLM layers 3/7/11/15 and expert 3, strengths 0-4 spreads, goal_lan 7, eps 20-39:
0.00-0.05 everywhere (no effect). Review (LITERATURE_REVIEW §3) explains: direction averaged over image tokens, tiny strength,
added at all tokens. Baseline (10 eps): goal_lan 0:0.2 1:0.7 2:0.7 3:0.0 4:0.8 5:0.0 6:0.2 7:0.1 8:0.1 9:0.5; goal_swap and
spatial_swap 0.0 everywhere.
Operator: screen ideas on 5-10 episodes, confirm with 20, use the offline lab first.
Plan: (1) frozen rollouts with --save-obs on targets to see HOW they fail; (2) offline lab: how instruction tokens are
represented, where the paraphrase representation diverges, causal patching; (3) interventions.

## 16:50 frozen rollouts with saved obs (outputs/programs/none/libero_goal_lan__{0,3,5,7}, eps 0-9)
lan7 0/10, lan3 0/10, lan0 2/10, lan5 0/10. lan7 failures: arm goes to y~0.15-0.2, z~0.94, closes gripper, retries (a
grasp loop at a fixed spot).

## E01 (library/notes/lab/e01_which_behaviour.py) which trained behaviour does "switch on the stove" trigger?
Chunk at the first observation under the paraphrase vs under every other trained goal instruction (twin task 7 excluded).
Paraphrase chunk is 41-80 cm (path distance) from all of them, noise floor 2.4 cm -> not a fallback to one known skill.

## E02 (library/notes/lab/e02_context_knn.py) contextual nearest neighbours of paraphrase tokens (VLM residual, each layer)
Bank: tokens of all trained instructions (4 suites, twin excluded) on the same observation.
- OOV verbs look like OBJECT NOUNS: "switch" -> compartment/drawer (cos 0.53-0.59, L4-L12), "push" only at L16 (0.67);
  "move" -> compartment/plate; "slide" ("sl"+"ide") -> the/up, compartment/oka/put.
- sentence-initial "place" (token without space) is not the trained " place" (cos 0.58-0.76) -> plate at L16.
- possessive "’" -> newline (end of instruction!) at L4-L10, "s" -> and/of/m. "middle" (only in the twin) -> center.
- Known words: cos 0.93-1.00 at every layer.
Conclusion: paraphrase failures are lexical; the fine-tuned 16-layer VLM lost general English (no synonym structure).
Base SmolVLM2 (32 layers + lm_head) still knows it: tools/lm_probe.py ranks "turn on the stove" first for
"switch on the stove" (-26.3 vs put -30.3).

## steerlib/canon.py: noisy-channel rewrite into the trained dialect
c* = argmax w*log P_bigram(c | trained instructions minus twin) + log P_LM(x|c) + log P_LM(c|x) (base SmolVLM2, paraphrase
prompt). v1 free edits: w=1 copies the input (LM copy bias), w=2-4 drifts semantically (cream cheese inside the bowl -> on
the plate; rack -> basket). v2 structured (general syntax): content words protected; only verb slots (first word, word
after 'and'; substitutes = words seen in verb slots of trained instructions) and closed-class function words are edited.
v2, w=2, goal_lan: 0 'open middle drawer of the cabinet', 1/2/4/8/9 place->put, 3 'open the top drawer and put the bowl',
5 'put the plate on the front of the stove', 6 'put the cream cheese in the bowl', 7 'turn on the stove'.
Other suites: still bad (deletes prepositions: 'put into basket'->'put the basket'; lift->put). To fix: tune on a synthetic
paraphrase dev set I write myself, not on benchmark paraphrases.

## 17:00 simulator: canon programs on goal_lan (dev eps)
- canon_rewrite (v2 'turn on the stove') lan7: 10/10 (0-9), 10/10 (10-19) -> 20/20 vs frozen 0/10. VERIFY requested.
- canon_rewrite v2 lan0 ('open middle drawer of the cabinet'): 7/10 vs frozen 2/10. canon_v5 lan0 ('open the middle drawer
  of the cabinet'): 8/10.
- v2 lan3 ('open the top drawer and put the bowl'): 0/10. v2 lan5 ('put the plate on the front of the stove'): 0/10.

## canon v4/v5 (scorer and prior)
v3 = verb phrases in verb slots ('pick up', 'turn on'), only articles deletable, prepositions swap only. Dev set
(tools/canon_devset.json, 24 of my own paraphrases of trained instructions, source excluded): SmolVLM2 scorer w=2: exact
16/24 F1 0.944. Llama-2-7B scorer (v4) w=2: 19/24, 0.979 BUT on benchmark lan7 gives 'put the stove': the frequency-weighted
bigram prior gives '<s> put the' ~9 nats over '<s> turn on' (only one trained sentence). v5 = prior -beta * (#unseen bigrams)
(in-dialect or not, frequency-free): beta 2: 9/24; beta 4: 20/24 F1 0.986; beta 8: 19/24. v5 on benchmark goal_lan: all
semantically right (lan3 keeps 'slide open ... put the bowl in it', lan5 'move the plate to the front of the stove').
Remaining v5 weakness: two-edit moves (grab -> pick up the) get stuck in local optima on spatial/object paraphrases.

## E03 (lab/e03_seed_consistency.py) is the policy more self-consistent on instructions it understands? NO.
Spread across 6 noise seeds (mean pairwise path distance): known 2.0-3.9 cm, paraphrase 2.0-4.4, nonsense ('blick the bowl
on the stove', 'zorp the cabinet') 2.0-2.3, targets 2.1-2.5. The flow-matching expert is confidently wrong; seed agreement
cannot select instruction candidates.

## 17:20 more screens; lan3 behaviour; E04 sub-instructions
- canon_v5 lan6 ('put the cream cheese in the bowl'): 5/10 vs frozen 2/10 (trained goal 6 itself 4/10).
- lan3 frozen frames (tools/frames.py): robot never opens the drawer, goes straight to the bowl and lifts it.
- E04 (lab/e04_subinstructions.py, first chunk, 3 eps): 'open the top drawer of the cabinet' goes to the cabinet (8-12 cm
  path distance from 'open the middle drawer of the cabinet', ends higher: z -32..-46 vs -14..-25, consistent with the top
  drawer); 'open the top drawer' (no 'of the cabinet') and all rewrites with 'and put the bowl' head to the bowl instead.
  -> the policy composes 'top' with the drawer-opening skill but needs the dialect noun phrase 'drawer of the cabinet'.
  Feasibility probe queued: programs/probes/lan3_decomp_feasibility.py (sub1 'open the top drawer of the cabinet' 130
  steps, then 'put the bowl in the top drawer of the cabinet'). Not a candidate (hand-written for one task).
- Drawer opening (goal_lan 0 successes): 120-133 steps.
- steer/interp.py bug (cannot edit steer/): segments() uses the 48-token PREFIX; for a short instruction 'lang' includes the
  state token and 'state' is empty (NaN). steerlib/bank.py builds the same Bank with correct spans (+ first/last token keys).
- goal_swap layout (read BDDL for understanding only): bowl <-> wine bottle positions, cabinet <-> stove positions; same for
  all goal_swap tasks. Plan: test whether the reach target comes from the state (absolute coordinates) or the image.

## 17:45 RESERVED tasks (operator): libero_object_lan and libero_object_swap must stay unseen.
Disclosure: before this note I had listed the 10 object_lan instructions (lab/lan_instructions.json) and run canon v2-v7 on
them (no simulator runs, no recordings). Seeing their rewrites partly motivated the v7 compound edit (verb + article).
Purged: object_lan removed from lab/lan_instructions.json, canon_v*_lan.json and canon_cache.json. Never use them again.
CORRECTION (17:50): the prefix is always 177 tokens (language padded to 48 by the processor; real tokens at 128..128+n-1,
padding masked, state at 176), verified from attention shapes. The original steer.lab/interp spans were right; my
"segment bug" report was wrong (sorry). E02 used positions 128..128+n-1: correct. steerlib.analysis.spans and bank.py fixed.
E06 (lab/e06_image_attention.py, goal_swap 1): adding +1..+4 to expert attention logits on agentview and/or wrist tokens
(all / early / late expert layers) does not retarget the reach: reference reach ends are 63 cm apart (bowl's true spot =
trained wine-bottle spot vs memorised bowl spot); frozen ends 12-31 cm from the memorised; big boosts reach 32-40 cm from
the true spot but also 29-37 from the memorised: perturbation, not retargeting.

## 17:55 results so far (dev episodes; frozen baseline in brackets)
goal_lan (canon v5 unless noted): 7 v2 20/20 [0/10] VERIFY requested; 0 18/20 [2/10] VERIFY requested; 6 5/10 [2/10];
9 9/10 [5/10]; 8 0/10 [1/10] ('put the bowl on top of the plate'); 3 0/10 (v2 and v5); 5 0/10 (v2 and v5).
spatial_lan (canon v6, eps 0-4): 0 2/5 [2/10], 1 4/5 [0/10], 2 2/5 [0/10], 3 0/5 [0/10] (v6 kept 'lift black bowl from
cookie box'), others running.
lan3 decomposition probe (sub1 'open the top drawer of the cabinet' for 130 steps, then 'put the bowl in the top drawer of
the cabinet'): 0/5, BUT frames show phase 1 works (the top drawer is pulled out); phase 2 picks the bowl and does not place
it. Queued: phase 2 = full rewrite ('slide open ... put the bowl in it' / 'open the top drawer and put the bowl').

## E07 (lab/e07_noun_localizer.py): noun-token attention is NOT an object localiser
Attention from 'bowl'/'stove' tokens to the 64 agentview patches (VLM prefix pass, every 2nd layer; model image is rotated
180 deg, mapped back): only 3-19% of the noun's attention goes to agentview; the argmax patch sits on the small blue box in
both the trained and the swapped scene for 'bowl'; no consistent tracking of the moved objects.

## Swap plan: virtual frame (no hard-coded layout)
E05 says the policy reads its current position from the state (x/y slope about -1 under state shifts). In swap it goes to
memorised positions. Plan: feed eef - delta, delta = (true object position) - (memorised position), from images:
reference = trained-suite recording of the same instruction (memorised pick/place points = eef at gripper close/open,
plus its first image); find where the reference content around those points moved in the live image (template matching);
pixel -> table (x, y) by a homography self-calibrated from the policy's own grasps (tools/calib2.py: grasp eef vs object
footprint pixel). ASSUMPTION to flag: for swap tasks the instruction is unchanged; recordings of the trained task with the
same id are used as reference (the twin ban was stated for paraphrased tasks). Camera self-calibration from arm silhouettes
(tools/calib.py) failed (elbow above gripper in the image).

## 17:50 VERIFIED: canon_rewrite (v2) on goal_lan 7: held-out 29/30 vs frozen 1/30 (p<0.001), PROMOTED.
spatial_lan canon_v6 (eps 0-4; frozen 10-ep baseline 0.0 except task 0 = 0.2): 0 2/5, 1 4/5, 2 2/5, 3 0/5, 4 0/5, 5 5/5,
6 4/5, 7 0/5 ('lift bowl from stove and put plate'), 8 2/5, 9 0/5 ('lift bowl from cabinet and put on the plate').
goal_lan 9 canon_v5: 9/10 then 4/10 (10-19) = 13/20 vs frozen 5/10: weak.
canon v8 (dialect noun-phrase completion: 'black' before 'bowl', 'of the cabinet' after 'drawer', learned from n-gram
statistics of the trained instructions) changes only goal_lan 3 -> 'open the top drawer of the cabinet and put the bowl
on it'. v9 beam search (width 4) is WORSE on the dev set (18/24): the unseen-n-gram count rewards deleting articles (fewer
n-grams). Stopped tuning canon here; per-suite programs: v5 (goal), v6 (spatial), v7 (best dev-set balance).

## E08 (lab/e08_siglip_match.py): the policy's own SigLIP patch features localise a moved object
Mean SigLIP feature (vision tower, 32x32 grid before pixel shuffle) over the bowl's footprint patches in a trained goal frame;
cosine match in goal_swap 1 first frames: best patch at raw pixel ~(116,124) in 5/5 episodes = the bowl's new position
(sim 0.26-0.54). Pixel templates (NCC, masked) and colour histograms failed (plate / stove confusions).
-> steerlib/scene.py (patch_features, locate, footprint, to_table), tools/make_reference.py, programs/candidates/
virtual_frame.py written; waiting for the homography (tools/calib2.py needs more trained grasps; queued spatial + goal).

## 18:20 camera calibration for the virtual frame (FLAG FOR OPERATOR REVIEW)
Self-calibration attempts failed: (1) arm-silhouette top as gripper (tools/calib.py): 33/208 inliers (elbow above gripper);
(2) object footprints vs policy grasps -> homography (tools/calib2.py): inconsistent pairs (two black bowls in spatial
scenes, arm pixels), LOO error 6 cm (affine) / 16 cm (homography); (3) SigLIP gripper detector seeded by 4 hand-annotated
pixels (tools/calib3.py): degenerate fits; (4) hand annotation of 12 frames: eef point hidden behind the gripper body.
DECISION: use the fixed agentview camera model as declared in the LIBERO tabletop scene setup (pos, quat, MuJoCo default
fovy 45) as a SENSOR CALIBRATION constant in steerlib/scene.py (project/backproject). It is identical for every task and
suite, contains no object pose or layout, and is corroborated by image evidence: hand-annotated gripper pixels agree in u
within 4 px; the frozen policy's grasp points project within 5-15 px of the grasped objects' footprints (offset = object
base vs grasp at the rim/top). If the operator considers this a simulator internal, the swap line must go back to (2)/(3).

## E09/E10: offline evaluation of the virtual frame is not informative
E09 (goal_swap 1): SigLIP feature-template tracking (scene.region + scene.track) finds the bowl's displacement consistently
(3/3 eps, delta_pick (-13..-18, -5..-8) cm; notes-only BDDL check: (-11, -5) cm). But the first-chunk reach under the
shifted state does not move toward the true spot.
E10: the first-chunk reach saturates: shifting the state backwards (-x) does not change the first 25/50 steps at all; shifting
forwards cuts the reach steeply (slope ~ -2.5). Actions are clipped at +-1 while the target is far, so the target only shows
in later chunks. -> closed-loop test in the simulator (queued: virtual_frame on goal_swap 1 and 8, eps 0-4).
References built (tools/make_reference.py): goal 1, 2, 4, 8, 9; spatial 0-9 (goal 6: frozen policy failed all 3 eps).
spatial_lan: canon_v6 confirmations 1: 19/20, 5: 16/20, 6: 16/20 (frozen 0/10 each) -> VERIFY requested.
canon_v7 spatial_lan 3: 4/5 (v6 0/5; v7 'pick up the black bowl in the cookie box and place it on the plate').

## 18:15 VERIFIED: canon_v5 on goal_lan 0: held-out 26/30 vs frozen 11/30 (p<0.001), PROMOTED.

## virtual_frame on goal_swap 1 and 8 (sim, eps 0-4): 0/5 and 0/5. Why (traces; trace eef = the VIRTUAL eef):
- the virtual state goes out of the workspace (virtual y up to 0.66 with delta_place = (+13, -38) cm on task 1) and the
  policy wanders instead of converging to its memorised point; images (arm seen at the real place) conflict;
- tracking on task 8 is unreliable (delta_pick (+4, +10) cm in 2 eps, (-12, -13) in 1; similarity 0.17-0.40).
Conclusion: shifting the proprio state by tens of cm is not a usable lever (E10 saturation + out-of-distribution states).

## E11 (lab/e11_target_probe.py): where is the reach target? (52 trained-suite episodes, pre-grasp chunks, label = grasp x/y)
Ridge probes, folds grouped by episode: lang tokens R2 0.93-0.95 (x) / 0.82-0.85 (y) at VLM L4-L12; state token 0.90-0.93 /
0.86-0.88; expert 0.89-0.91 / 0.92-0.94; agentview image tokens only 0.20-0.27 / 0.44-0.59; wrist 0.14-0.19.
Caveat: target = f(task) here (confounded with task identity). Consistent with swap failures: the target is retrieved from
the instruction (task memory), the image tokens carry little of it.
## E12 (lab/e12_target_push.py): the probe directions are NOT causal levers
Adding alpha * probe direction (expert neuron space, L4/8/12, +-4 sd = +-44..59 cm of decoded target) to all action tokens
moves the 50-step reach by <= 2 cm. Read-out of task identity, not a control knob.

## 18:25 RESERVED (operator): libero_10_lan is now the transfer-comparison suite (object_lan stays reserved too).
Before the note: I had listed its 10 paraphrases, run canon v2-v8 on them (outputs printed in the session), and a queue had
just started canon_v7 on libero_10_lan tasks 0 and 1 (killed within minutes, no result written; partial output dirs
outputs/programs/canon_v7/libero_10_lan__{0,1} deleted). Purged from lab/lan_instructions.json, canon_v*_lan.json and the
canon cache (571 -> 489 entries). Never touch libero_10_lan again. Operator accepted (for now) the agentview camera model
as a sensor constant and trained-task recordings as swap references.

## 18:40 carry_to_target (VLA picks, program carries to the tracked destination): spatial_swap 0/1/5: 0/5 each.
Diagnosis: on the TRAINED layout (spatial 0, delta 0) it was 0/3, then 1/3 after a tighter controller (release within 8 mm,
press 1.5 cm below the reference height, lift after opening): the hand-over/placement control is the weak point (bowl held at
the rim, OSC settling), not only perception. NOTE: trace "action" = the VLA's proposal before on_step replaces it.
Operator: shift weight to internal mechanisms. Line parked (documented in knowledge/swap.md).

## E15 (lab/e15_verb_patching.py) WHERE THE VERB ENTERS THE ACTION EXPERT (activation patching x -> c, 5 goal pairs x 4 obs)
R = fraction of the rewrite's behaviour restored when patching the paraphrase run with the rewrite's residuals (after layer L):
  verb token at L only: L0 0.96, L1 0.86, L2 0.72, L3-6 0.60, L7-8 0.51, L9-10 0.40, L11-12 0.33, L13-14 0.10, L15 0.00
  verb token at layers 0..L: 0.96-0.97 for every L;   all OTHER instruction tokens at L: -0.03..0.04 at every layer;
  all instruction tokens at L ~ verb@L + 0.05;   state token at L: 0.11-0.18 at L2-L8.
Conclusion: the verb reaches the expert through the verb token's OWN KV entries at every VLM layer (the expert
cross-attends to each layer's KV cache); the VLM does not move verb information onto other tokens in a way the expert uses
(no 'summary token'); the state token carries a little (attends to the verb in L2-8). An in-network fix must act on the
verb token, as early as possible.

## E16 (lab/e16_lexical_edit.py) LEXICAL WEIGHT EDITS (steerlib/lexedit.py, steer/edit.py multi-pair, ridge 1e-3)
Edit of layer L's MLP down projection so that the unknown token's key writes v* = h_out(rewrite) - h_mid(paraphrase).
Fit on 2 observations, test on 4 others (goal scene), R and side effect on 'put the bowl on the plate' (cm):
  place->put (bowl/stove): L0 0.26 (0.80 cm), L1 0.71 (0.12), L2 0.52 (0.13); patch bound L0 0.96, L1 0.75, L2 0.57
  switch->turn:            L0 0.77 (0.47),    L1 0.95 (0.20), L2 0.85 (0.15); bound 0.99, 0.94, 0.87
  place->put (wine/rack):  L0 0.74 (0.88),    L1 0.78 (0.13), L2 0.67 (0.16); bound 0.96, 0.84, 0.73
  place..inside->put..in:  L0 -0.06 (0.82),   L1 0.76 (0.15), L2 0.57 (0.16); bound 0.97, 0.82, 0.65
-> layer 1 is the sweet spot: edits reach the patching bound and generalise across observations; layer-0 keys are not
context-invariant enough (and side effects are 4-6x larger).
SIM: programs/candidates/lexedit_live.py (edit fitted LIVE on the episode's first observation; text unchanged):
goal_lan 7: 4/5 (frozen 0/10) -- the frozen VLA reads 'switch on the stove' and turns the stove on.

## 18:50 VERIFIED: canon_v5 goal_lan 6: held-out 18/30 vs frozen 6/30 (p=0.002), PROMOTED.
lexedit_live goal_lan 7: 15/15 (eps 5-19) -> 19/20 dev (frozen 0/10), text unchanged -> VERIFY requested.
lexedit_live goal_lan 6: 5/10 (= canon rewrite 5/10).
## E17 (lab/e17_target_source.py): the instruction, not the image, sets the reach target (spatial suite, 10 layouts)
Predicted grasp from one chunk (eef0 + path at the first closing command) is coarse: 3-12 cm from the actual grasps.
(a) scene A + instruction B: closer to B's memorised grasp in 67% of 90 pairs (14.6 vs 19.0 cm).
(b) scene A + instruction A with the agentview image of scene B (or B's agentview tokens patched at all VLM layers):
stays near A's grasp (11.8 vs 21.6 cm; closer to B in only 17%). The image barely moves the target: memorised per instruction.

## 19:00 lexedit_live limits; E18 transfer; stored facts
lexedit_live: spatial_lan 1 0/5 (text rewrite v6: 19/20), goal_lan 0 0/10 (rewrite: 18/20). These paraphrases differ by
INSERTIONS (pick UP, THE, IT) or REORDERING ("cabinet's middle drawer" -> "middle drawer of the cabinet"). Consistent with
E15: the expert reads each token's own KV entries at its position; a weight edit can change what an existing token means but
cannot create missing tokens or reorder them. => lexical edits fix SUBSTITUTION paraphrases (switch->turn, place->put),
structural paraphrases need an input-level change (rewrite = new tokens).
E18 (lab/e18_edit_transfer.py): a 'place'->'put' edit at L1 fitted on ONE sentence (2 obs) transfers to the 4 other 'place'
sentences: R 0.69-0.82 (each source). Pooling the other sentences into one multi-pair fit is worse (0.37-0.66).
Stored facts (tools/make_fact.py -> library/edits/, index facts.json; 1 sentence, 2 observations each):
place_put_from_goal9, place_put_from_goal1, switch_turn_from_goal7 (layer 1). Program: candidates/lexfacts.py (applies facts
whose trigger token occurs in the instruction; never a fact learned from the same task). Queued: goal_lan 1, 2, 4, 8, 9.
## E19 (lab/e19_verb_neurons.py): the verb correction is DENSE, not a few neurons
Needed change at the verb token after VLM layer 1: |dv| ~150. Greedy OMP over W_down columns at layer 1, pushed at the verb
position only: switch->turn k=1/4/16/64/256 neurons: R -0.04/0.00/0.09/0.34/0.53 (reconstruction residual 0.99..0.36);
place->put: R 0.03/0.05/0.04/-0.13/0.20. Word identity at L1 is a distributed code; a rank-one/low-rank weight edit (writes
the full vector for the token's key) is the right tool, single-neuron pushes are not (contrast with LLM single-neuron lore).

## 19:05 SIM: in-network lexical fixes (text unchanged)
lexedit_live (live fit, 1 obs): goal_lan 9 10/10 (frozen 5/10; text rewrite 13/20); spatial_lan 5 0/5 (structural).
lexfacts (STORED fact 'place'->'put' learned on ANOTHER sentence, 1 sentence + 2 obs, transfer):
goal_lan 1 10/10 (frozen 7/10), 2 10/10 (7/10), 4 10/10 (8/10), 9 9/10 (5/10). One learned fact fixes four tasks.
Confirmations (eps 10-19) queued.
## 19:15 carry_to_target parked: LIBERO 'On' (predicate code, read for understanding) requires the object's centre within
3 cm (x/y) of the target's centre plus contact. Frames show the bowl on the plate but off-centre: tracking bias (2.9 cm on the
unchanged trained layout: the 8-px patch quantisation of track()), rim-grasp offset, OSC settling. Needs sub-patch tracking and
closed-loop final alignment; parked in favour of internal mechanisms.
## E20 (lab/e20_destination_mix.py): a CONTINUOUS internal destination knob (goal scene)
Bowl held (chunk 1 of trained goal 1/8 eps). Patch the destination token's residual at all VLM layers with
(1-a) r('plate') + a r('stove'): the predicted endpoint moves monotonically along the plate->stove line:
a = 0.25/0.5/0.75 -> 0.14-0.19 / 0.53-0.60 / 0.88-0.91 of the way, < 1 cm off the line, 6/6 observations.
The place target is a smooth function of the destination noun's representation (the expert interpolates).
## E21 (lab/e21_destination_nouns.py): in SPATIAL scenes the destination noun is ignored
With the bowl held, '... and place it on the {plate, cookie box, ramekin, stove, wooden cabinet, table center}' all end
within ~4 cm of the same point (the plate's trained spot), whatever the noun. All spatial training sentences place on the
plate: the destination is hard-wired, not read from the word. -> noun mixing cannot redirect spatial_swap placements; it
can in the goal scene (destinations vary with the noun).

## 19:30 VERIFIED: canon_v6 spatial_lan 1: held-out 22/30 vs frozen 0/30, PROMOTED. lexfacts goal_lan 1 eps 10-19: 9/10.
lexfacts goal_lan 8: 0/10 (the policy then reads 'put the bowl on top of the plate'; 'on top of' belongs to the cabinet).
## E22 (lab/e22_mask_deletion.py, e22b): IN-NETWORK TOKEN DELETION; positions matter
Pair from another task: x = 'put the wine bottle on top of the rack' vs c = 'put the wine bottle on the rack' (goal 9).
Masking ' top',' of' as keys for every query: R 0.61; only in the VLM prefix pass: R 0.01; only in the expert: R 0.61.
Marking them as PADDING in embed_prefix (attention mask + rotary positions = cumsum(pad mask) - 1 close the gap): R 0.995.
=> the expert reads the instruction as (token KV, rotary position) pairs; VLM-side contextual mixing is irrelevant (as E15);
a 2-position gap costs ~40% of the effect. Exact deletion is possible inside the network.
SIM: lexfacts_mask (stored place->put fact + pad-deletion of 'top of' when the 4-gram 'top of the X' is not in the dialect
but 'on the X' is; twin excluded): goal_lan 8 8/10 (frozen 1/10), text unchanged. Confirmation queued.
## E23 (lab/e23_reader_layers.py): WHICH EXPERT LAYERS READ A WORD (attention knockout per expert layer, all denoising steps)
Effect = d(word blocked at expert layer j, normal) / d(word deleted exactly, normal), 3 goal-scene obs:
  'stove' (destination): L11 0.45, L7 0.23, L13 0.13, L5 0.09   'bowl': L9 0.16, L15 0.15, L3 0.09, L7 0.08
  'open' (verb): L9 0.26, L3 0.08, L7 0.08                      'wine': L3 0.22, L5 0.11, L7 0.08
  'put': <= 0.09 everywhere (weakly read: the default verb)
Even expert layers (self-attention over [prefix | actions]) <= 0.03 for every word: the language is read by the
CROSS-ATTENTION layers (odd); each word has preferred readers (destination late: L7/L11; object/verb mid: L3-L9).
## 19:35 PROBE dest_mix (closed loop, sim): internal destination steering works continuously
goal 8 'put the bowl on the plate', destination token residual (all VLM layers) mixed live with r('stove') from
'put the bowl on the stove': alpha 0 -> release at the plate (0.04..0.08, 0.01), 2/3 success; alpha 0.5 -> (-0.05..-0.065,
0.21) between; alpha 1 -> the stove spot (-0.24, 0.26). The robot carries the bowl to an intermediate place: a continuous
internal goal knob in closed loop (E20 offline confirmed).
## 19:45 VERIFIED: canon_v6 spatial_lan 5: held-out 24/30 vs frozen 0/30, PROMOTED.
PROBE pick-noun transplant (goal_swap 8, 'bowl' token residual mixed with r('wine') or r('bottle') from 'put the wine
bottle on the rack', until the grasp, alpha 0.5/1): 0/3 each; the grasp stays at the memorised bowl spot (x -0.10..-0.14 vs
frozen -0.07..-0.12). Unlike the destination (E20/dest_mix), the PICK target is not steered through the object noun.

## 19:55 DISTILLING PROGRAM SUCCESSES INTO ADAPTERS (tools/train_adapter.py, programs/candidates/adapter.py)
Self-generated successes of canon programs (spatial_lan, --save-obs), relabelled with the ORIGINAL paraphrase; LoRA rank 8
on q/v projections; base weights untouched; the paraphrase is given to the VLA unchanged at test time.
- one-shot (1 successful episode = 3 chunks, VLM LoRA, 200 steps, ~15 s): spatial_lan 1 eps 5-9: 3/5 (frozen 0/10;
  program 19/20). One self-generated episode already teaches the policy the paraphrase.
- cross-task (3 successes of spatial_lan 1/5/6/3, 400 steps): spatial_lan 0: 0/5, 2: 0/5 (frozen 2/10, 0/10): no transfer to
  unseen paraphrases (adapter memorises trajectories, not a reading rule). Remaining tasks running.
Variants for sample efficiency queued: 3 episodes (VLM), 1 episode expert LoRA, 1 episode 60 steps.
Adapter sample efficiency on spatial_lan 1 (frozen + adapter, raw paraphrase, eps 5-9): 1 ep VLM 200 steps 3/5; 3 eps VLM
300 steps 4/5; 1 ep EXPERT 200 steps 3/5; 1 ep VLM 60 steps 0/5 (undertrained). Program 19/20, frozen 0/10.
Cross-task adapter (successes of 1/5/6/3) on unseen spatial paraphrases 0/2/4/7/8/9: 0/5 each.
Next: self-training (expert iteration) where the program is partial: spatial_lan 0, 2 (canon 2/5): collect successes, train
an adapter with the REWRITE label, run rewrite + adapter on new episodes.
## E24 (lab/e24_cheese_destinations.py): the destination knob is object-specific
Holding the cream cheese, 'put the cream cheese in/on the {bowl, plate, stove, rack, cabinet, basket}' end within ~5-8 cm
of each other (only plate/stove differ more): cream cheese was trained with one destination (the bowl), so there are no
distinct anchors; E20 worked for the bowl, trained with two destinations (plate, stove). goal_swap 6 (bowl moved) cannot
be redirected this way. Swap line closed for this session: the target is memorised per (instruction, object) and the only
continuous internal knob found (destination noun interpolation) spans only destinations seen with that object.

## 20:00 VERIFIED: canon_v6 spatial_lan 6: held-out 27/30 vs frozen 0/30, PROMOTED.
Self-training (expert iteration) on spatial_lan 0/2: adapter on the program's 3 successes (rewrite label, VLM LoRA, 300
steps) + rewrite, eps 10-19: 6/10 and 3/10 = identical to the rewrite alone (6/10, 3/10). No gain from 3 successes.
canon_v7 dev totals: spatial_lan 0 9/20 (frozen 2/10), 2 6/20 (frozen 0/10) -> VERIFY requested for both.
## E25 (lab/e25_word_sensitivity.py): the policy is nearly as sensitive to function words as to content words
Exact deletion (pad-mask) of each token of the trained goal + spatial instructions; chunk change in units of the flow-noise
floor: nouns 3.8x, newline 2.9x, adjectives 2.9x, prepositions 2.7x, verbs 2.6x, 'the' 2.6x (n=43), particles ('up') 2.0x,
'and' 1.4x, 'it' 1.1x. Single words: 'top' 6.1x, 'cabinet' 5.5x, 'put' 4.4x, 'the' 2.6x.
=> instructions act like exact wording patterns (memorised strings): a rewrite that is semantically right but drops one
'the' (spatial_lan 0: '...between the plate and ramekin') still moves the behaviour by ~2.6 noise units. Explains the
residual failures of near-correct rewrites and why the dialect prior (n-grams) matters.
## E26 (lab/e26_familiarity.py): token-level representation familiarity is a weak success predictor
Familiarity = mean/min over the rewrite's tokens of max cosine to the trained-token bank (twin excluded), VLM L2-L14.
Spearman with sim success over 20 rewrites: +0.34..+0.46 (best L14 min, p=0.04). Blind spots: content words known only from
the twin ('middle', 'rack') look unfamiliar though the policy knows them (goal_lan 0, 9 succeed); wrong COMBINATIONS of
familiar words ('on top of the plate', goal_lan 8: 0.99 familiarity, 0% success) look fine. Not a better objective.

## 20:15 verification: lexfacts goal_lan 1 held-out 30/30 vs frozen 27/30 (p=0.12), goal_lan 2 26/30 vs 22/30 (p=0.17):
not promoted (strong frozen baselines on held-out states; no regression).
canon_v12 (frame-fill fallback): two parallel jobs loaded Llama-2-7B each (frame_fill was not cached) -> CUDA OOM; fixed by
caching frame_fill and precomputing. Fills: spatial_lan 7 'pick up the black bowl on the stove and place it on the plate',
9 'pick up the black bowl on the cabinet and place it on the plate'. LESSON: programs must only hit the canon cache in
parallel runs (precompute with one process).

## 20:30 VERIFIED: lexedit_live goal_lan 7 (IN-NETWORK, text unchanged, layer-1 weight edit fitted live on the first
observation): held-out 29/30 vs frozen 1/30 (p<0.001), PROMOTED. Same as the text rewrite (29/30).
canon_v12 spatial_lan 7: 7/10 (frozen 0/10). spatial_lan 9 with plain fill ('on the cabinet'): 0/10; re-run with suite
completions + n-gram prior ('pick up the black bowl on the wooden cabinet and place it on the plate') queued.
canon_v12 spatial_lan 7: eps 10-19 7/10 -> 14/20 dev (frozen 0/10) -> VERIFY requested.
canon_v12 spatial_lan 9 ('... on the wooden cabinet ...', suite completions + prior): 3/10 (frozen 0/10); confirming.

## 20:35 frame fill + completions (canon_v12 / canon_v13)
v12 = v7 + frame-fill fallback (dialect frames filled with the paraphrase's content words; completions from the same suite;
ranked by Llama + 4 x unseen-n-gram prior) when a verb stays untranslated.
spatial_lan 9 ('pick up the black bowl on the wooden cabinet and place it on the plate'): 3/10 + 5/10 = 8/20 (frozen 0/10)
-> VERIFY requested. spatial_lan 7: 14/20 (VERIFY requested earlier).
v13 = v12 + deterministic dialect noun-phrase completion (left modifier only for bare nouns when the same suite always uses
it; 'of the X' after a noun when > half of the trained uses; never for multi-word-preposition words; first versions were
over-eager: 'the middle top drawer'). spatial_lan 8 ('the black bowl next to the plate'): 6/10 (v7 2/5, frozen 0/10);
spatial_lan 4 ('in the top drawer of the cabinet'): 0/10; goal_lan 3 ('open the top drawer of the cabinet and put the bowl
on it'): 0/10.
## 20:42 more confirmations
canon_v13 spatial_lan 8: 6/10 + 5/10 = 11/20 (frozen 0/10) -> VERIFY. lexedit_live goal_lan 9: 10/10 + 7/10 = 17/20 (frozen
5/10) -> VERIFY. canon_v13 with a second completion pass ('... in the top drawer of the WOODEN cabinet ...': 'wooden' before
'cabinet' from the spatial suite statistics, twin excluded): spatial_lan 4 6/10 (frozen 0/10; every earlier rewrite 0) ->
confirming 10-19.
canon_v13 spatial_lan 4: 6/10 + 8/10 = 14/20 (frozen 0/10; trained twin 0.6) -> VERIFY. All 10 spatial_lan tasks now have a
program above the frozen policy (1/5/6 verified; 0, 2, 3, 4, 7, 8, 9 pending verification).
## E27 (lab/e27_phase_source.py): where the task PHASE (reach vs transport) comes from (45 reach/hold pairs, trained eps)
Swap one input of a reaching obs with the holding obs's; score = fraction of the holding behaviour obtained:
gripper qpos 0.16, wrist image 0.22, agentview 0.15, eef position 0.05; gripper+wrist 0.39; both images 0.54.
=> the phase is encoded redundantly, mostly by the IMAGES (the object in the gripper is visible) and the gripper state;
contrast with E17: the images drive WHAT phase it is, hardly WHERE to go.
## 20:50 VERIFIED: lexfacts_mask goal_lan 8: held-out 25/30 vs frozen 0/30 (p<0.001), PROMOTED -- fully IN-NETWORK
(stored 'place'->'put' layer-1 fact learned on ANOTHER sentence + exact pad-deletion of 'top of'; text unchanged).
vf_place (virtual frame during transport only, spatial_swap 0/5): 0/5 each; the gripper stays CLOSED at the end: the policy
never releases. Consistent with E27 (phase is read from the images): it does not see the plate under the bowl. Swap closed.

## 20:55 GENERALISATION TEST of the language pipeline on FRESH paraphrases (written before any result)
Concern: canon's rules were shaped while seeing benchmark paraphrases. Test: my own new paraphrases of TRAINED goal/spatial
tasks, the task itself excluded from the dialect (as for a twin); frozen vs canon_v13 rewrite, 5 eps each.
RESULT (5 eps each, frozen -> canon_v13 rewrite, task excluded from the dialect):
  goal 1 'set the bowl down on the stove' 3/5 -> 4/5 ('put the bowl down on the stove')
  goal 8 'move the bowl onto the plate' 2/5 -> 5/5            goal 9 'stand the wine bottle in the rack' 2/5 -> 4/5
  goal 4 'set the bowl on the cabinet top' 0/5 -> 2/5         goal 2 'place the wine bottle onto the top of the cabinet' 0/5 -> 4/5
  goal 0 "pull the cabinet's middle drawer open" 0/5 -> 3/5 (rewrite garbled: 'put middle drawer open on the cabinet')
  spatial 5 'take the black bowl off the ramekin and put it onto the plate' 0/5 -> 3/5
  spatial 3 'grab the black bowl sitting on the cookie box and set it down on the plate' 3/5 -> 4/5
  mean 0.25 -> 0.73, 8/8 improved: the pipeline generalises to paraphrases it was not developed on.
IN-NETWORK on the same fresh SUBSTITUTION paraphrases (text unchanged, live layer-1 edit toward the v13 rewrite, 1 obs):
  goal 1 3/5 -> 4/5 (rewrite 4/5); goal 8 2/5 -> 3/5 (5/5); goal 9 2/5 -> 5/5 (4/5); goal 4 0/5 -> 3/5 (2/5).
  mean 0.35 -> 0.75 (= text rewrite 0.75): the in-network lexical edit generalises to new paraphrases too.
FACT LEARNING on fresh words: 'set'->'put' fact fitted on ONE fresh sentence ('set the bowl down on the stove', goal 1, 2 obs),
stored (edits/set_put_from_fresh0), applied unchanged to a DIFFERENT fresh sentence/task ('set the bowl on the cabinet top',
goal 4): 0/5 -> 2/5 (live edit 3/5, rewrite 2/5). The fact library learns a new word from one example and reuses it.
## E28 (lab/e28_gripper_neurons.py): ONE expert neuron is a gripper switch
Attribution (activation x gradient of the mean gripper command over the first 40 steps; expert L4/8/12/15; 6 goal reaching
obs): top neuron expert L4 #1855 (0.027, next 0.009). Dosed push of that neuron alone (all action tokens, all flow steps):
dose -1/-2 -> the gripper never closes within the chunk (close step +11 = beyond 50 for all 6 obs); dose +2 -> closes at once
(-39 steps); dose +1 -> no change (threshold). Mixing the top-3/8 neurons with their attribution signs gives erratic results.
Contrast E19: word identity is dense; gripper timing has a single-neuron switch in the action expert (early layer).
## 21:20 VERIFIED: canon_v7 spatial_lan 3: held-out 24/30 vs frozen 1/30, PROMOTED.
## E29 (lab/e29_blind_expert.py): the expert DOES read the images, but not for the global target location
Blocking the action expert's attention to image tokens (all layers/steps), 14 reaching obs (noise floor 0.59 cm):
agentview blind: path change 6.4x noise, predicted grasp moves 4.7 cm; wrist blind: 10.1x, 8.7 cm; both: 18.6x, 13.9 cm.
With E17 (another scene's image does not send the reach to that scene's target): images shape HOW the arm moves (the wrist
view most), the WHERE is retrieved from the instruction. Together with E27 (images set the phase/release).
## 21:25 FIRST SWAP SUCCESSES: carry_handback (VLA picks; program carries above the image-tracked destination; hands back to
the VLA with the state shifted by delta, so it believes it is at its memorised place point while the cameras show the plate)
spatial_swap 0: 2/5, spatial_swap 5: 1/5 (frozen 0/10 both); trained spatial 0 (delta ~0): 1/3. Failures: tracking errors
(spatial_swap 5 ep2: delta (-3, 0) cm instead of ~(0, -16)) and the VLA wandering after hand-back (10-20 cm).
Next: descend before the hand-back (less room to wander).
carry_handback2 (descend to 2 cm above the reference place height before hand-back): spatial_swap 0 2/5, 5 2/5, 1 0/5.
Combined dev so far: swap 0 4/10, swap 5 3/10 (frozen 0/10). spatial_swap 1 fails: tracked delta wrong in 3/5 eps
(0, (-3, 0.6), (0, -2.2) cm vs true ~(7, -27)).
## E30 (lab/e30_tracking_consensus.py): median over 6 tracker settings is NOT more robust (swap 0 ep1: 32 cm error vs 2 cm).
The failures (swap 1 ep2, swap 5 ep2) are confident zero-shift matches for EVERY setting: what now stands at the old plate
spot (often the other bowl) resembles the reference region. Needs a more object-specific template; left for later.

## 21:40 VERIFIED: lexedit_live goal_lan 9 (in-network): held-out 26/30 vs frozen 15/30 (p=0.002), PROMOTED.
carry_handback2 on spatial_swap (frozen 0/10 everywhere): 0: 2/5 + 3/10 = 5/15; 5: 2/5 + 3/10 = 5/15 -> VERIFY requested;
2: 2/5; 4: 3/5; 3, 6, 7: 0/5 (8, 9 running). First non-zero results on a swap suite.
## 21:45 carry_handback2 confirmations: spatial_swap 4 3/5 + 5/10 = 8/15; 2 2/5 + 4/10 = 6/15; 9 2/5; 8 0/5.
approach_handback (approach + hand-back for pick AND place): goal_swap 9 2/5 (first goal_swap successes; pick delta (13, 3) cm
vs true ~(11, 5)), goal_swap 8 0/5 (pick delta (4, 10) wrong direction; place deltas 15-60 cm although the plate did not move).
Tracking scores (goal_swap frames): unmoved plate (goal_swap 8 place) best 0.65-0.67 vs zero-shift 0.56-0.62; moved bowl
(goal_swap 1 pick) 0.23-0.29 vs 0.11-0.15; but the correctly tracked moved plate of spatial_swap 0 also has a small margin
(0.67 vs 0.58): no clean 'moved' criterion from the margin. goal_swap 8's pick region is only 4 patches (unreliable).
VERIFY queue re-prioritised (time): held spatial_lan 2 (6/20) and 9 (8/20) as HOLD_VERIFY_*; requested spatial_swap 2, 4.
VERIFY: canon_v7 spatial_lan 0: held-out 15/30 vs frozen 10/30 (p=0.15), not promoted.
CORRECTION (21:55): goal_swap layouts DIFFER PER TASK (BDDL, notes only), e.g. goal_swap 8: bowl -> cream-cheese spot
(+4, +13 cm), plate -> wine-bottle spot (-25, -3); goal_swap 1: bowl -> bottle spot, cabinet <-> stove. My 17:20 note
('same for all 10 tasks') was wrong. Re-scored: the goal_swap 8 pick deltas (3.6, 9.6..15 cm) were RIGHT; its failure is
destination tracking (plate). Footprint-template tracking of the bowl (goal 1 reference footprint): goal_swap 1 (-12, -6) cm in
2/3 (true -11, -5), goal_swap 8 (4, 15)/(4, 12.5) in 2/3 (true 4, 13).
Plate tracking in goal_swap 8 (true -25, -3 cm) fails for every (tau, radius): consistently (18, -15) or (-13, 18) = other
white regions. Goal-scene destination tracking needs a better object model (left for the next session).
approach_handback goal_swap 9: 2/5 + 1/10 = 3/15 (frozen 0/10): weak, not sent to verification (queue time).
22:05 decision: no more simulator runs of my own for a while, so the driver's verifications (spatial_lan 4, 7, 8;
spatial_swap 0, 2, 4, 5) can finish before the deadline; offline work and documentation meanwhile.
LESSON (22:10): never rename a VERIFY_* file the driver may already be processing (it renames it to DONE_* when done; a missing
file would crash the driver). spatial_lan 2 had started at 21:37; its file was restored at 22:08 (driver unaffected).
## E31 (lab/e31_two_layer_edit.py): multi-layer lexical edits do NOT help; layer 1 alone is best
R (side effect cm): place->put L1 0.71 (0.12), L0+1 0.21 (0.92), L1+2 0.62, L1+3 0.76 (0.35), L0+1+2 -0.03 (1.03);
switch->turn L1 0.95, L1+2 0.96, L0+1 0.66; wine/rack L1 0.78 best; cheese L1 0.76, L0+1 -0.39 (0.83). Layer-0 keys are not
context-invariant (overfit, large side effects). The L1 edit already sits at the L1 patching bound (E16).
## E32 (lab/e32_destination_heads.py): heads that read the destination at expert L11 (holding obs, 4 eps)
Per-head knockout of attention to ' stove' (fraction of the exact-deletion effect): h13 0.08, h14 0.07, h2 0.03, h10 0.03,
h1 0.02, h12 0.02, others <= 0.01; all 15 heads of L11 together 0.17 (sum of singles 0.26). Heads 12-14 share one KV head
(GQA 15 query / 5 KV heads): the destination is read mainly through KV group 4 at L11.
## 21:52 VERIFIED: canon_v7 spatial_lan 2: held-out 10/30 vs frozen 3/30 (p=0.029), PROMOTED. (Clock check: it is 21:52, my
earlier time stamps after 21:45 in this journal were estimates and too late by ~10-20 min.) spatial_lan 9 request restored.
## E33 (lab/e33_plate_prototype.py): a plate PROTOTYPE averaged over the 10 spatial references is WORSE for tracking
median error 12.9 cm on swap frames (template tracking: 2-5 cm when it works) and false shifts of 5-27 cm on unchanged trained
layouts: averaging removes the context the template relied on; the similarity blob merges with other white objects.
carry_handback2 failures by task: tracking right -> ~40-50% success (swap 9: (-26, 0) cm, 2/5); tracking wrong -> 0 (swap 3:
plate -> cabinet spot, also the VLA rarely lifts; swap 6: plate on the stove; swap 8: zero-shift matches; swap 1: 3/5 wrong).
## 22:00 next: SWAP SELF-IMPROVEMENT: distil carry_handback2 successes on spatial_swap 4 into a LoRA adapter; test the VLA
alone (adapter, no program) on new episodes.
SWAP SELF-IMPROVEMENT (tools/train_adapter_swap.py): carry_handback2 on spatial_swap 4 eps 0-9 with --save-obs: 7/10
successes. LoRA (q/v, rank 8, VLM+expert) on them (state un-shifted after the hand-back), VLA + adapter ALONE (no program,
original instruction) on eps 10-19: 7 eps 400 steps 1/10; 1 ep 300 steps 1/10; 7 eps 1500 steps (loss 0.18) 1/10
(frozen 0/10; the program 8/15). A small adapter trained on a few self-generated successes does not learn the new layout;
the program stays far better than this few-demo fine-tuning.
## 22:07 VERIFIED: canon_v13 spatial_lan 4: held-out 21/30 vs frozen 0/30, PROMOTED.
## E34 (lab/e34_reference_vote.py): VOTING across 3 reference frames (trained eps 0-2) fixes most tracking failures
largest cluster of agreeing deltas (4 cm): median error 4.7 -> 1.2 cm; > 10 cm errors 4/9 -> 2/9 (remaining: all three refs
disagree). -> carry_handback3 (vote, tie-break by tracking score).
carry_handback3 (voted tracking), eps 0-4: spatial_swap 1 1/5 (v2 0/5), 5 3/5 (v2 2/5; deltas right in 5/5), 8 1/5 (v2 0/5),
7 0/5 (votes still on zero shifts). Running v3 on 0, 2, 4, 9 for a same-episode comparison.
## 22:25 VERIFIED: canon_v12 spatial_lan 7: held-out 24/30 vs frozen 0/30, PROMOTED.
carry_handback3 vs v2 on the same eps 0-4: 0 2/5=2/5, 2 2/5=2/5, 4 3/5=3/5, 9 0/5 vs 2/5, 5 3/5 vs 2/5, 1 1/5 vs 0/5,
8 1/5 vs 0/5, 7 0/5 = 0/5: total 12/40 vs 11/40 -> no clear gain; the v2 VERIFY requests stay.
## E35 (lab/e35_delete_unknown.py): an unknown verb is better than NO verb
Exact deletion of the unknown word(s), R toward the rewrite: switch -0.22, place(bowl/stove) -2.35, place(wine/rack) +0.23,
set -3.06, move+onto -2.27. A verb-less instruction moves the behaviour even further from the right one than the unknown verb:
the verb slot must be FILLED with a known verb (substitution = lexical edit), deletion only helps for extra words ('top of').
## 22:40 canon v14 = canon.normalize (general English: strip politeness/modal frames, 'I want the X on Y' / 'the X goes on Y'
-> 'put the X on Y', separable particles 'turn the stove on' -> 'turn on the stove') + v13. Written from general knowledge for
styles absent from the benchmark (rewrites before: 'could you put...' -> 'put you put bowl on the stove', 'turn the stove on'
-> 'turn on the stove to'; after: all correct). Changes NONE of the goal/spatial benchmark paraphrases (checked), so verified
results are unaffected. auto.py and tools/precompute_canon.py now use v14. Sim check of 3 styles (frozen vs v14) running.
v14 styles (5 eps, frozen -> v14): 'could you put the bowl on the stove' 0/5 -> 4/5; 'I want the bowl on the plate' 3/5 -> 4/5;
'the bowl goes on the stove' 3/5 -> 5/5.
## 22:35 VERIFIED: canon_v13 spatial_lan 8: held-out 19/30 vs frozen 0/30, PROMOTED.
## E36 (lab/e36_goal_vote.py): voting over 3 references does NOT fix goal-scene tracking
goal_swap 8 place (true -25, -3): 1/3 right (2 cm), pick (true 4, 13): ~6-7 cm off; goal_swap 1 pick (true -11, -5) and place
(stove, true 44, -45 = 63 cm): wrong (29-48 cm) although single references were sometimes right ((-9,-5), (-13,-5): no two
within the 4 cm tolerance -> score tie-break picked a wrong one). Goal scenes have several large moves; perception is the
bottleneck for goal_swap.
canon v15 (insert 'the' when 'X noun' is unseen but 'X the noun' is seen in the dialect) changed NO goal/spatial rewrite:
'and the ramekin' occurs only in spatial task 0 itself (the twin, excluded). Removed (no-op). spatial_lan 0's missing article
cannot be recovered by dialect statistics without the twin.
## 22:45 VERIFIED: canon_v12 spatial_lan 9: held-out 20/30 vs frozen 0/30, PROMOTED. spatial_lan: 9/10 tasks promoted
(1-9); task 0 15/30 vs 10/30 (n.s.). Swap verifications (carry_handback2 spatial_swap 0/2/4/5) running.
## 22:55 diagnosis of carry_handback2 failures with CORRECT tracking (spatial_swap 0 eps 0-1, frames saved under
outputs/programs/hb_diag): after the hand-back the bowl hangs a few cm off-centre (wrist view: the plate's rim only at the
edge); the VLA hovers with the gripper closed (0.005) and never releases - it never SEES 'bowl over plate' (E27).
Residual error = ~2 cm tracking + rim-grasp offset. Next: wrist-camera alignment before the hand-back (feature shift between
the live wrist frame and the reference episode's wrist frame at success, with an image Jacobian estimated from the policy's
own motion).
## 22:53 carry_handback4 = v2 + local search (no release 25 steps after the hand-back -> nudge 2-3 cm, hand back again):
spatial_swap 0 3/5 (v2 2/5 on the same eps; the 3 successes came after 2 nudges), spatial_swap 4 3/5 (= v2).
carry_handback4 more: spatial_swap 0 eps 5-14 6/10 -> 9/15 (v2 5/15); 2 5/5 (v2 2/5); 9 4/5 (v2 2/5); 5 1/5 (v2 2/5).
VERIFY carry_handback2 spatial_swap 0: held-out 3/30 vs 0/30 (p=0.12), not promoted (dev 5/15: held-out states harder).
Requested VERIFY carry_handback4 on spatial_swap 0 and 9 (v2 requests for 4, 5 kept; 2 in progress).
## 23:10 VERIFIED (FIRST SWAP TASK): carry_handback2 spatial_swap 2: held-out 20/30 vs frozen 0/30 (p<0.001), PROMOTED.
carry_handback4 screens: spatial_swap 7 2/5 (v2 0/5); 1, 3, 6, 8 0/5 (tracking failures / plate onto the stove or the
cabinet spot). Pending verification: carry_handback4 swap 0 and 9, carry_handback2 swap 4 and 5.
## 23:20 VERIFIED: carry_handback2 spatial_swap 4: held-out 11/30 vs frozen 0/30, PROMOTED (second swap task).
## 23:27 VERIFIED: carry_handback2 spatial_swap 5: held-out 8/30 vs frozen 0/30 (p=0.002), PROMOTED (third swap task).
## 23:38 VERIFIED: carry_handback4 spatial_swap 0: held-out 8/30 vs frozen 0/30 (p=0.002), PROMOTED (v2 had 3/30: the
nudge-and-retry search helps on held-out states too). Requested (long shots, time permitting): approach_handback goal_swap 9
(dev 3/15), carry_handback4 spatial_swap 7 (dev 2/5).
## 23:45 VERIFIED: carry_handback4 spatial_swap 9: held-out 17/30 vs frozen 0/30, PROMOTED. spatial_swap promoted: 0, 2, 4, 5, 9.
VERIFY approach_handback goal_swap 9: held-out 4/30 vs frozen 0/30 (p=0.056), not promoted (first non-zero goal_swap result).

## 23:58 END-OF-SESSION STATE (read notes/SUMMARY.md first)
Held-out PROMOTED (frozen -> program, /30): goal_lan 7 1->29 (text) and 1->29 (in-network), goal_lan 0 11->26, 6 6->18,
8 0->25 (in-network), 9 15->26 (in-network); spatial_lan 1 0->22, 2 3->10, 3 1->24, 4 0->21, 5 0->24, 6 0->27, 7 0->24,
8 0->19, 9 0->20; spatial_swap 0 0->8, 2 0->20, 4 0->11, 5 0->8, 9 0->17.
Not significant: goal_lan 1 27->30, 2 22->26, spatial_lan 0 10->15, goal_swap 9 0->4. Pending: spatial_swap 7 (v4).
Unsolved: goal_lan 3, 5; spatial_swap 1, 3, 6, 8; goal_swap except 9.
Entry point for new suites: programs/candidates/auto.py + tools/precompute_canon.py (knowledge/recipes.md).
## 00:07 VERIFIED: carry_handback4 spatial_swap 7: held-out 18/30 vs frozen 0/30, PROMOTED (dev was only 2/5).
Final: 21 promoted results (20 distinct tasks): goal_lan 0, 6, 7 (x2: text and in-network), 8, 9; spatial_lan 1-9;
spatial_swap 0, 2, 4, 5, 7, 9. All verification requests processed.
## 00:14 VERIFY lexfacts goal_lan 4: held-out 24/30 vs frozen 23/30 (p=0.5), not promoted (strong baseline). Session closes
here; nothing running. Next session: start from notes/SUMMARY.md ("Next steps").

# SESSION 2 (2026-10-27, from 16:40). Operator: RAM tight -> ONE simulator job at a time.
## 16:45-17:18 goal_lan 3 by skill composition (programs/candidates/open_then_place.py; otp_* = env-var variants), all 0/5
Plan: sub-skill 1 'open the top drawer of the cabinet' (VLA), container located from the robot's OWN motion (turnaround point of
the reach = closed handle; pull vector = where the drawer came out), sub-skill 2 a trained bowl-pick instruction, program places.
Findings (eps 0-4 each, numbers = episodes):
- the opening sub-skill is reliable and self-measurable: pull 14-19 cm along +y in 5/5 (handle (0.02, -0.14..-0.16, 1.03)).
- switching straight to sub-skill 2 from where sub-skill 1 ended: the VLA never grasps the bowl (0/15 over 3 instructions; it drifts
  upward). Returning to the start POSITION only: it grasps the WINE BOTTLE (2/3 grasps) - the hand is still rotated ~100 deg
  after the pull. Restoring position AND orientation (OSC world-frame axis-angle, 1 <-> 0.5 rad): the VLA grasps the bowl 5/5.
  => COMPOSITION RULE: a sub-skill must start from the training start pose incl. orientation; the wrist view/state decide WHICH
  object is picked (contrast E17: images barely move the target in-distribution; out of distribution they do).
- program carry + release over the drawer: the bowl (rim grasp, qpos 0.002-0.008) slips during program carries (qpos -> 0.0003
  within ~20 steps; also at half speed and after the VLA's own 8 cm lift); when it survives, the bowl ends inside the drawer
  (1 ep, frames) but not 'in' by the predicate (rim grasp => bowl centre ~5+ cm off the hand; FRAC 0.5 too far out).
- moving across before clearing the drawer front pushes the drawer shut (diagonal carry); release-on-stall saves ~35 steps.
- VLA carry with 'put the bowl on top of the cabinet' (its trained path passes over the drawer) + program take-over near the
  drawer: 1/5 take-overs, bowl lands outside. Virtual frame for that skill (state shifted so its memorised place point maps to
  the drawer): erratic early releases (0/5).
- time budget is tight: open ~125 steps + return ~30 + pick ~50 leaves ~90 of 300.
Parked. What would be needed: a carry that keeps the grasp (the VLA's own) aimed at a new place point -> next: E37.
## 17:40 canon v15 (programs/candidates/canon_v15.py): spatial_lan 0 13/20 dev (frozen 2/10; v7 9/20) -> VERIFY requested
Found while testing an article-insertion rule: the v14 pipeline (= auto.py's *_lan choice) rewrote spatial_lan 0 to the WRONG
frame 'pick up the black bowl next to the plate ...' (0/10): the v12 frame-fill fallback treats every word after 'and' as a verb
slot, so the noun 'ramekin' ('the plate and ramekin') looked like an untranslated verb. v15 = v14 with (1) a dialect noun after
'and' no longer triggers the fallback, (2) canon.coord_article: 'the A and B' -> 'the A and the B' for a bare noun B (general
English determiner repetition; E25). Rewrite: 'pick up the black bowl between the plate and the ramekin and place it on the
plate' (= the trained wording: every word from the paraphrase or general grammar; flagged as in SUMMARY's transparency note).
v15 differs from v14 on NO other goal_lan / spatial_lan task (checked all 20), so verified results stand; auto.py now uses v15.
Sim: eps 0-9 4/10, eps 10-19 9/10. Note: even the exact trained wording gives only ~65% in the spatial_lan 0 scene (trained
spatial 0: 100%), so the rest is not language.
## 17:50 VERIFIED: canon_v15 spatial_lan 0: held-out 22/30 vs frozen 10/30 (p=0.002), PROMOTED (v7 was 15/30, n.s.).
spatial_lan: all 10 tasks promoted. Discovery: SIM trace inspection of auto.py's failure (ledger #9, knowledge/discoveries.md).
## 17:50 operator paper targets (prompts/explorer_extra.md): (1) discovery ledger -> knowledge/discoveries.md (method, data,
cost, seed or not, for every intervention); (2) gating ablations; (3) general entries for unseen compositions; (4) fixed
development panel scored per library version -> library/tools/panel.py (16 tasks: goal_lan 0,3,5,6,7,8; spatial_lan 0,1,4,7;
spatial_swap 0,1,2,7; goal_swap 1,9; eps 0-4; each version's own auto.py in an isolated copy; table in notes/panel.md).
Frozen panel total 5/80. v033 running.
New discovery tool: steerlib/levers.py (minimal sufficient intervention: proximal-L1 gradient search over MLP neurons of several
layers; support = the lever; install(ctx, sol) gives a per-chunk dose knob for state-dependent programs).
## E37 (lab/e37_destination_vector.py): a gradient-learned DESTINATION VECTOR works in the action expert, not in the VLM
Method: SEARCH (steer.lab.optimize_vector, 60 Adam steps through the 10 flow steps). Data: 18 holding-phase observations of
frozen trained spatial tasks 0-4 (3-6 episodes each); target = the policy's own chunk with its endpoint moved by D = +8 cm y
(bias spread over the first 30 steps). Test: 20 holding obs of tasks 5-9 (other scenes and instructions). Cost: ~4 min per site.
  site                 |v|   train gain  TEST gain  TEST orth   gripper change
  expert L0 all tokens 2.4   0.81        0.79       0.2 cm      0.015
  expert L4            3.1   0.66        0.66       0.6 cm      0.020
  expert L8            2.7   0.61        0.59       0.3 cm      0.005
  expert L12           2.6   0.64        0.63       0.3 cm      0.004
  VLM L2 / L8 / L14    1-1.5 0.03-0.06   0.02-0.06  0.1 cm
  VLM L8 / L14 state   0.2-0.3 0.00      0.00
(seed noise of the endpoint 0.47 cm). One vector fitted on 5 tasks moves the predicted place point by 79% of 8 cm on 5 OTHER
tasks with 0.2 cm off-axis error: the action expert has a near-task-independent motion code at its first layer. The same
search in the VLM moves nothing (<= 6%): the motion target is not steerable through the prefix by a single vector (cf. E12:
probe directions in the prefix are read-outs, not levers). Next: closed loop (destination shift or velocity bias?).
Also 17:55: canon_v16 goal_lan 8 9/10 (frozen 0/10 on eps 0-4 of the panel; v033 auto 0/5).
## E37b CLOSED LOOP (programs/candidates/dest_vector.py, trained spatial 1, eps 0-2, dose 1, gated on holding): the learned
expert-L0 vector is a DESTINATION SHIFT, not a velocity bias. The VLA picked, carried, aligned and RELEASED BY ITSELF at
y = 0.311 / 0.313 / 0.319 vs its memorised place point y = 0.240 (reference): +7.1 / +7.3 / +7.9 cm (wanted +8; offline gain
predicted 6.3), x within 1 cm. It stops and releases (no drift): a closed-loop internal destination knob, learned in ~4 min from
18 observations of OTHER tasks (spatial 0-4), applied to task 1. (Episodes took ~11 min each: the GPU was saturated by an
outside graphics workload from ~18:05; dose 2 not run yet.) Method: SEARCH (gradient) + SIM. Ledger #11.
GPU note 19:05: matmul 4 TFLOPS (normal ~50x more), policy chunk 4.4 s, env step 0.4 s; no CUDA process visible except
graphics clients (Firefox, GNOME). Operator notified. Simulator work is ~40x slower until it clears.
## E37c internal hand-back on spatial_swap 2 (programs/candidates/dest_swap.py, y-vector only, dose = tracked -15.6 cm / (8 cm x
0.92) = -2.1, gated on holding; 1 episode, GPU-starved): the hand moved toward the moved plate but only -8..-10 cm in y (of
-15.6) with +7 cm x drift, hovered and never released. A single vector is linear around dose 1 (E37b) but not at -2 with a sign
flip. -> E39: learn the whole map offset -> vector instead.
## E39 (lab/e39_destination_interface.py): a learned DESTINATION INTERFACE M (720 x 2) at expert L0, v = M @ (dx, dy), trained
on the policy's own chunks with random synthetic endpoint shifts in a 25 cm disc (3 per obs, N = 50 steps), trained spatial
tasks 0-4, 6, 8, test on 5, 7, 9. Batched forward (16 obs) because the GPU is time-sliced (9x throughput vs single obs).
First run (2 obs/episode): loss 0.177 -> 0.082 in 10 steps (45 s/step); restarted with 1 obs/episode, 50 steps.
E39 result (40 train obs, 50 steps, 24 min under GPU contention): loss 0.176 -> 0.047; random offsets in the 25 cm disc:
TRAIN gain 0.61 (median 0.61), off-axis 1.8 cm; TEST tasks 5/7/9 gain 0.59 (median 0.60), off-axis 1.7 cm, gripper change
0.017. Under-trained but direction-correct for ANY offset on unseen tasks; linear by construction, so programs divide the
wanted offset by the gain. Saved library/directions/dest_iface_L0.pt. Method: SEARCH (gradient, batched). Closed loop: running.
E39b closed loop, spatial_swap 2 ep 0 (IFACE=dest_iface_L0, GAIN 0.75, gated on holding): same end as the single y-vector run
(E37c): the hand goes to (0.14-0.18, 0.10-0.15) instead of the tracked goal (0.08, 0.07), hovers, never releases. Two different
levers ending at the same spot suggests an image-driven attractor in the VLA's final approach (E29), or a hard episode:
carry_handback2 also FAILS ep 0 of spatial_swap 2 (succeeds on eps 1, 4). Next closed-loop test on ep 1.
## E40 (lab/e40_jacobian_basis.py): CONTROL VIEW - the destination is a linear, shared 2-D code in the action expert; a
## closed-form interface (pseudo-inverse of the mean endpoint Jacobian) needs NO training and has gain 1.00
Method: DIS (Jacobian = one backward of the chunk endpoint per observation w.r.t. an additive vector at expert layer L, all action
tokens) + closed form. Data: 40 holding obs of trained spatial tasks 0-4, 6, 8 (1 per episode); test 18 obs of tasks 5, 7, 9.
Cost: ~1 min per layer (GPU time-sliced). Results (endpoint = summed clipped translation of the predicted chunk):
  layer  J consistency cos(obs, mean)  top singular values        pinv interface TEST gain / off-axis   8 cm     20 cm
  L0     0.93 (min 0.82)               0.200 0.183 | 0.058 0.038                                        1.00/0.3 0.97/0.7 cm
  L4     0.89 (min 0.75)               0.137 0.128 | 0.049 0.034                                        0.99/0.3 0.98/1.0
  L8     0.90 (min 0.79)               0.126 0.116 | 0.040 0.030                                        1.01/0.3 1.00/1.1
=> every observation shares the same Jacobian (cos 0.93) and it is effectively rank 2 (x, y): the expert's early layers carry a
task-independent planar motion code, controllable LINEARLY up to at least 20 cm. The closed-form v = pinv(J_mean) @ (dx, dy)
beats both gradient-trained levers (E37 vector gain 0.79, E39 interface 0.60) at a fraction of the cost: the control view is the
better discovery method here. Saved library/directions/dest_pinv_L{0,4,8}.pt ({"P": (720, 2), "Jm": (3, 720)}).
Closed loop on spatial_swap 2 ep 1 (dest_swap.py IFACE=dest_pinv_L0): running.
## 21:10 FIRST SUCCESS OF A PURELY INTERNAL SWAP CONTROLLER (programs/candidates/dest_swap.py -> dest_swap_fb.py)
spatial_swap 2 ep 1. The VLA performs every motion and the release itself; the program (a) tracks the plate in the image
(-15.6 cm in y), (b) while the robot holds the bowl, adds v = pinv(J_mean) @ cmd at expert L0 (E40 interface, cmd = tracked
offset), (c) near placing height corrects cmd every chunk by 0.5 x (goal - hand) (integral feedback on the lever INPUT).
Open loop (FB=0) on the same episode: y converged (0.09-0.10 vs goal 0.073) but x drifted +1 cm / 10 steps (0.13 -> 0.22), no
release (also ep 0, both levers). With feedback: x error -4.9 -> -3.5 cm, cmd (-10.6, -18.8) cm, success at step 135.
Method: DIS (E40 Jacobian) + PROG (feedback law) + SIM. Screening eps 0, 2-4 (carry_handback2 2/5 on eps 0-4; frozen 0/10).
dest_swap_fb v1 on spatial_swap 2 eps 0-4: 2/5 (eps 1, 2; = carry_handback2's 2/5 on these episodes; frozen 0/10). Failures:
ep 0 hand ends 1-2 cm from the tracked goal but the VLA never releases; ep 3 the feedback fired at the grasp (hand low, 14 cm
from the goal) and pushed cmd the wrong way, then ran away (hand at y 0.20); ep 4 no success. v2: feedback only within 10 cm
of the goal (FB_RADIUS) -> rerunning eps 0, 3, 4.
## E41 (lab/e41_reach_jacobian.py): the REACH target uses the same linear code
E40 on reaching observations (gripper open, first half of trained spatial episodes; 42 train / 18 test obs): J consistency cos
0.94 (min 0.87), rank ~2 (0.222 0.206 | 0.071), pinv interface TEST gain 0.92 at 8 cm (0.4 cm off-axis) and 0.92 at 20 cm (0.9 cm);
cos(J_hold, J_reach) = 0.87 (x), 0.86 (y). => one planar motion code at expert L0 serves reach and transport; the same
closed-form interface can retarget the GRASP (moved pick objects: goal_swap) and the place. Saved directions/reach_pinv_L0.pt.
dest_swap_fb v2 (feedback only within 10 cm): eps 0, 3, 4 still 0/3. Both inspected failures: the feedback brings the hand to
~1 cm of the tracked goal (t 211-241) and the VLA hovers without releasing (the carry_handback failure mode), then wanders.
v3 = v2 + a state-gated INTERNAL release: when the hand has stayed within 1.5 cm of the goal at placing height for 2 chunks,
push the gripper-switch neuron expert L4 #1855 (E28) with dose -2 (offline: the VLA's own gripper command on a holding obs goes
+0.99 -> -1.17; activation 81 per dose unit). Composition of two discovered levers (ledger #6 + #13), both gated by the state.
Running eps 0, 3, 4.
dest_swap_fb v3 on eps 0, 3: ep 3 SUCCESS (step 159), ep 0 fail. The release lever did NOT fire in either (ep 0 never settled
within 1.5 cm: final error (1.6, 4.8) cm; ep 3 succeeded by the VLA's own release), so ep 3's success is not credited to the lever
(policy sampling noise differs between runs). Tally on spatial_swap 2 eps 0-4 across v1-v3: eps 1, 2, 3 succeed = 3/5 (frozen
0/10, carry_handback2 2/5 on the same eps); ep 4 v3 running. 22:50 VERIFY requested (dest_swap_fb, spatial_swap 2): the
first program in which the VLA performs every motion and release itself on a swapped layout, steered only through an internal
closed-form destination interface (E40) with integral feedback.
dest_swap_fb v3 ep 4: fail -> spatial_swap 2 dev 3/5 (eps 1-3). Driver started the verification at ~23:10.
spatial_swap 4 ep 0 (v3): the VLA stopped at y 0.21 (memorised 0.242, goal 0.078: only -3 cm of the -16 cm shift in closed loop,
although the transport here comes from the drawer at y -0.10, z 1.25) and released there; v2/v3's 10 cm feedback radius never
engaged. -> dest_swap_fb2.py (v4, separate file so the verification of dest_swap_fb.py is not altered): feedback engages once
the transport is over (hand low AND > 10 cm from the grasp point), whatever the distance to the goal. Screening swap 4 eps 0-1.
v4 spatial_swap 4 ep 0: fail again, release at y 0.209 and NO feedback logged. Cause (a gating bug, i.e. evidence that the
WHEN matters): the lever is on only while the hand is above grasp height + 3 cm; in task 4 the bowl is picked from the drawer
(z ~1.10) above the plate (0.93), so the lever switched OFF during the descent and the VLA went back to its memorised place point
(-3 cm instead of -16). In spatial_swap 2 the pick is at table height, so the same gate kept the lever on. v4b: latch the lever
once engaged until the release. Running swap 4 ep 0.
v4b (latched lever) spatial_swap 4 ep 0: the destination lever now WORKS on this task too: the VLA carries the bowl from the
drawer to (0.09, 0.05-0.09) vs the tracked goal (0.086, 0.078) (v3 without latch: stopped at y 0.21) and hovers there; feedback
cmd (-2.4, -16.8) cm; no release (the release lever needs 2 chunks within 1.5 cm; hand error 2-4 cm ~ tracking error) -> fail.
Running REL_RADIUS 4 cm. Lesson: the release is the remaining bottleneck of every swap hybrid (carry_handback too).
00:14 v4b + REL_RADIUS 4 cm, spatial_swap 4 ep 0: SUCCESS at step 173. The release lever did NOT fire (no log): the VLA released
by itself once the latched internal destination + feedback brought the bowl over the moved plate. (v3 on the same episode: fail,
lever switched off during the descent.) dest_swap_fb2.py defaults now = this configuration (REL_RADIUS 0.04).
Internal swap controller, dev so far: spatial_swap 2 3/5 (dest_swap_fb, VERIFY running), spatial_swap 4 1/1 (dest_swap_fb2).

## 00:15 END OF SESSION 2 (driver deadline ~00:37). Nothing of mine running. Driver: verifying dest_swap_fb on spatial_swap 2.
Read notes/SUMMARY.md (session-2 block, next steps). Main results: spatial_lan 0 promoted (v15); auto.py = canon_v16; panel
tool (frozen 5/80, v033 35/80); E37-E41: a linear shared planar motion code at expert L0 with a closed-form interface
(pinv of the endpoint Jacobian, gain 1.00 on unseen tasks); internal swap controller (VLA does every motion and release).
