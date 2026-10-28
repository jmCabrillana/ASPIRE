# Discovery ledger: how each intervention was found (operator paper target 1)

Every library intervention, with the experiment that discovered it, the data it used, its cost, and whether the seed library
already had it. "Seed" = the library before any agent work (git faf67ec: steering by mean-difference directions, gates,
ablate/scale, LoRA helpers, the example program). Costs: sim = simulator episodes, lab = offline forward passes / GPU time.
Methods: DIS = dissection (patching, ablation, attribution), SIM = simulator screening of a hypothesis, SEARCH = automated search
(gradient / grid / evolutionary), LM = external language model, PERC = image perception, PROG = hand-written control logic.

| # | intervention (library entry) | discovered by (experiment) | method | data used | cost to discover | in seed? | held-out result |
|---|---|---|---|---|---|---|---|
| 1 | paraphrase failures are lexical; translate into the training dialect (`canon`) | E01 behaviour clustering, E02 contextual nearest neighbours of paraphrase tokens | DIS + LM | 10 goal paraphrases, trained instructions | ~30 lab min, 40 sim eps | no | goal_lan 7 1->29/30, 0 11->26, ... (9 tasks) |
| 2 | noisy-channel rewrite scored by Llama-2-7B + unseen-n-gram prior (v5-v7) | canon dev set (24 own paraphrases), F1 per version | SEARCH (grid over scorers/priors) + LM | own dev set, no benchmark wording | ~2 h, ~120 sim eps | no | spatial_lan 1/3/5/6 22-27/30 |
| 3 | the verb lives in its own token's KV (no summary token) | E15 residual patching per token per layer | DIS | 3 rewrite pairs x 2 obs | ~200 forward passes | no | (enables 4) |
| 4 | layer-1 closed-form lexical edit (`lexedit`, `edits/*.pt`) | E16 layer sweep of rank-k MLP edits vs the patching bound | DIS + SEARCH (layer sweep) | 1 sentence pair, 2 observations | ~150 passes; fit 1 s | locate-and-edit helper only (steer/edit.py); the layer and keys were found | goal_lan 7 29/30, 9 26/30, 8 25/30 (in-network) |
| 5 | exact in-network word deletion by pad-masking | E22 attention masking vs pad-masking (R 0.61 vs 0.995) | DIS | 4 observations | ~40 passes | no | goal_lan 8 25/30 (with 4) |
| 6 | gripper switch neuron expert L4 #1855 | E28 attribution patching on the gripper command, dosed pushes | DIS (attribution) | 6 reach obs | ~60 passes | no (Neurons tool given, neuron found) | not used in a program yet |
| 7 | images set the phase, not the place (E27, E29) -> hand back to the VLA with a virtual frame (`carry_handback2/4`) | E27 input swaps reach/hold, E29 blinding the expert, vf_place failure analysis | DIS + SIM + PERC + PROG | 45 obs pairs; 3 frozen trained eps per task (references) | ~1 h lab, ~150 sim eps | virtual frame idea seeded by the operator's prompt list; hand-back found | spatial_swap 2/4/5/0/9/7: 0 -> 8-20/30 |
| 8 | voting over references for tracking (E34) | E30/E34 tracking error analysis on swap frames | PERC + SEARCH (settings) | 3 reference frames | ~30 lab min | no | dev only (no gain overall) |
| 9 | v15 fallback fix + coordination article | failure trace of auto.py (v14) on spatial_lan 0 (wrong frame rewrite) | SIM trace inspection | 10 sim eps | ~15 min, 20 sim eps | no | spatial_lan 0 10 -> 22/30 (PROMOTED) |
| 10 | composition rule: a sub-skill must start from the training start pose INCLUDING orientation | open_then_place screens (goal_lan 3): drift / wrong object (bottle) / bowl, by pose reset | SIM + frames | 25 sim eps | ~40 min | no | (goal_lan 3 still 0) |

Session 2 entries are added below as they are made (same columns).

| 11 | destination vector at action-expert layer 0 (`directions/dest_expert0_y8.pt`, `programs/candidates/dest_vector.py`, `dest_swap.py`) | E37 site sweep (9 sites, gradient search per site), E37b closed loop | SEARCH (gradient through the frozen VLA) + SIM | 18 holding obs of 5 trained spatial tasks; target = own chunk with endpoint +8 cm | ~35 lab min (sweep) / 4 min (one site); 3 sim eps | no (optimize_vector added by the operator as a tool; site, data, target design and use found) | closed loop: VLA releases +7.1..+7.9 cm away by itself (3/3); swap test running |
| 12 | canon v16 deletion composition | development panel (v033 auto 0/5 on goal_lan 8) | panel EVAL + composition of entries 1 and 5 | none | 10 sim eps | no | goal_lan 8 9/10 dev |
| 13 | closed-form destination interface v = pinv(J_mean) (dx, dy) at expert L0/L4/L8 (`directions/dest_pinv_L*.pt`) | E40: endpoint Jacobian per observation, consistency + SVD, pinv | DIS (control view: Jacobian) | 40 holding obs, 7 trained tasks | ~1 GPU min per layer, no training | no | offline gain 1.00, 0.3 cm off-axis on unseen tasks (8 cm), 0.97 (20 cm); closed loop pending |
| 14 | internal closed-loop controller for swapped layouts (`dest_swap_fb.py`): tracked offset -> pinv interface at expert L0, integral feedback on the lever input | E37c/E39b open-loop failures (x drift) -> feedback law | DIS (13) + PROG + SIM | 1 reference per task (as carry_handback) | ~15 sim eps | no | spatial_swap 2 dev 3/5 (verification running); v4b (latched gate): spatial_swap 4 1/1 |
