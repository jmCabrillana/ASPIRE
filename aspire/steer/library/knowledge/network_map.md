# Map of smolvla_libero: what lives where (with evidence)

Architecture facts (code): prefix of 177 tokens = 64 agentview (rotated 180 deg, 8x8 grid after pixel shuffle of a 32x32
SigLIP grid), 64 wrist, 48 language positions (instruction tokens first, padding masked), 1 state token. 16 VLM layers (first
half of SmolVLM2-500M, fine-tuned) run once per chunk on the prefix; the action expert (16 layers, width 720) runs 10 flow
steps (t = 1 -> 0.1, t = 1 is noise); even expert layers self-attend over [prefix KV | action tokens], odd layers
cross-attend to the prefix KV of the same layer index. Actions: 7-d, world-aligned axes, ~0.0095 m per unit per step.

## Language
| what | where | evidence |
|---|---|---|
| word identity | the token's own residual, all layers; known words keep cos 0.93-1.00 to their trained-context representation at every layer | E02 |
| unknown verbs ('switch', 'move', 'slide') | represented like object nouns (compartment, drawer) at L2-L12; no synonym structure (fine-tuning removed general English) | E02 |
| possessive "'s" | reads like end-of-instruction ('\n') at L4-L10 | E02 |
| route of the verb to the action expert | the verb token's OWN KV entries at every VLM layer; patching only the verb after L0 restores 96% of a rewrite, after L8 50%, after L13 10%, after L15 0% | E15 |
| other tokens / summary | patching all non-verb tokens with the rewrite's residuals restores ~0% at every layer: no summary token carries the verb | E15 |
| state token | small share (11-18% restored by patching it at L2-L8) | E15 |
| best layer to write a lexical fix | L1 down projection: closed-form edit fitted on 2 obs reaches the patching bound (R 0.71-0.95) on new obs, side effect 0.12-0.20 cm; L0 keys are not context-invariant | E16 |
| what a weight edit can and cannot fix | substitutions yes (lexedit_live goal_lan 7 19/20, text unchanged); insertions / reorderings no (spatial_lan 1 0/5, goal_lan 0 0/10 vs 19/20 and 18/20 with the text rewrite): edits change existing tokens' meaning, they cannot create tokens or positions | E16, sim |
| who reads a word in the expert | the cross-attention (odd) layers; self-attention (even) layers <= 0.03. Destination 'stove': L11 (0.45), L7 (0.23); object 'bowl': L9, L15; verb 'open': L9; 'wine': L3, L5 | E23 |
| how the expert reads the instruction | as (token KV, rotary position) pairs: masking two tokens as keys gives R 0.61, marking them as padding (positions close up) R 0.995; masking only in the VLM pass R 0.01 | E22 |
| sensitivity to each word | deleting one token moves the first chunk by: nouns 3.8x noise, newline 2.9x, adjectives 2.9x, prepositions 2.7x, verbs 2.6x, 'the' 2.6x, 'and' 1.4x, 'it' 1.1x: wordings are memorised as exact patterns | E25 |
| an unknown verb vs no verb | deleting the unknown verb is worse (R -0.2 to -3.1) than keeping it: the verb slot must be filled with a known verb | E35 |
| neurons for a word | dense: reconstructing the switch->turn correction at L1 needs >256 neurons (k=16: R 0.09; k=256: R 0.53) | E19 |
| reuse of a lexical fact | a layer-1 'place'->'put' edit learned on one sentence (2 obs) transfers to other 'place' sentences (R 0.69-0.82) | E18 |
| general English knowledge | outside the policy: base SmolVLM2 (500M) knows 'switch on ~ turn on'; Llama-2-7B scores paraphrase equivalence well | lm_probe, canon dev set |

## Goal / target position
| what | where | evidence |
|---|---|---|
| current position | proprio state (eef), slope about -1 under state shifts in x/y | E05 |
| reach target | linearly decodable (R2 ~0.9) from language tokens, state token and expert; weakly from agentview image tokens (0.2-0.5) and wrist (0.15-0.25). Confounded with task identity | E11 |
| causal handle on the target | NOT the probe directions (pushing +-4 sd moves the reach <= 2 cm); NOT image attention (E06); state shifts saturate (E10) and go out of distribution in closed loop (virtual_frame 0/5) | E06, E10, E12 |
| destination (goal scene) | a smooth function of the destination noun's representation: interpolating r('plate') -> r('stove') at the noun token moves the endpoint along the line (0.15 / 0.55 / 0.9 at a = .25/.5/.75, < 1 cm off-line) | E20 |
| destination (spatial scenes) | hard-wired: any destination noun ends at the plate's trained spot (training never varied it) | E21 |
| images in the expert | used heavily for HOW to move (blind to wrist: 10x noise path change, both images 19x; grasp moves 9-14 cm) but not for WHERE: another scene's image does not retarget (E17) | E29, E17 |
| swapped layouts | the policy goes to memorised positions (spatial_swap: correct bowl, trained plate spot) | traces, knowledge/swap.md |

| a steerable MOTION code | the action expert's first layer: one learned additive vector at expert L0 (all action tokens) shifts the predicted place point by 79% of 8 cm on unseen tasks/scenes with 0.2 cm off-axis error; L4-L12 59-66%; the same search in VLM layers (all tokens or the state token) <= 6% | E37 |

| the destination code is LINEAR and SHARED | Jacobian of the chunk endpoint w.r.t. an additive vector at expert L0/L4/L8: same for every observation (cos 0.89-0.93 to the mean), effectively rank 2 (x, y); its pseudo-inverse moves the predicted place point by exactly the requested (dx, dy) on unseen tasks (gain 1.00, 0.3 cm off-axis at 8 cm; 0.97 at 20 cm) | E40 |
| the REACH uses the same code | on reaching observations the endpoint Jacobian at expert L0 is as consistent (cos 0.94), rank ~2, aligned with the holding one (cos 0.86); pinv gain 0.92 on unseen tasks | E41 |
| closed loop with the code | gated on holding, the VLA carries and releases ~7.5 cm away by itself (dose 1, E37b); for 15 cm swaps open loop drifts in x and never releases, integral feedback on the lever input succeeds 3/5 (spatial_swap 2) | dest_swap_fb |

## Phase (reach vs transport)
| what | where | evidence |
|---|---|---|
| task phase | redundantly in the images (both images: 54% of the switch to 'holding' behaviour; wrist 22%, agentview 15%) and the gripper state (16%); not the eef position (5%) | E27 |

## Motor
| what | where | evidence |
|---|---|---|
| gripper timing | a single action-expert neuron, L4 #1855: dose -1 keeps the gripper open (no close in the chunk), dose +2 closes it at once; threshold behaviour (+1 no effect) | E28 |

## Vision
| what | where | evidence |
|---|---|---|
| object identity per location | SigLIP patch features (vision tower, before the connector) match the same object after it moved (cos 0.26-0.54 for the bowl) | E08, E14 |
| noun -> image grounding in the VLM | attention from 'bowl'/'stove' tokens to agentview patches is diffuse (3-19% of the noun's attention) and does not track moved objects | E07 |

## Behaviour statistics
| what | evidence |
|---|---|
| seed (flow noise) spread of chunks does not distinguish understood from nonsense instructions | E03 |
| a paraphrase with an unknown verb does not fall back to one known skill; it produces a mixture (lan7: grasp loop at a fixed spot) | E01 |
