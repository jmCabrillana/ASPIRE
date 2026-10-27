# Swapped layouts (LIBERO-PRO *_swap): what the frozen policy does and what moves it

Layouts (BDDL, read for notes only): goal_swap permutes object positions DIFFERENTLY PER TASK (e.g. task 1: bowl <-> bottle, cabinet <-> stove;
task 8: bowl -> cream-cheese spot, plate -> bottle spot); spatial_swap keeps the bowl to pick where it was and moves the plate (exchanged with the cookie box or the other
bowl). Frozen success 0.0 everywhere.

## Mechanism
- The policy goes to memorised positions: in spatial_swap it grasps the right bowl (unmoved) and carries it to the trained
  plate position (traces, 3 tasks x 3 eps). In goal_swap it grasps at the bowl's trained spot (now the wine bottle).
- The reach target is linearly decodable from the instruction tokens (R2 0.94), the state token (0.9) and the expert (0.9),
  hardly from the agentview image tokens (0.2-0.5) (E11): the target is retrieved from the instruction, not read from the image.
- Not levers: boosting expert attention to image tokens (E06), pushing probe directions (E12, <= 2 cm), shifting the state
  (E10: saturating; virtual_frame 0/5 on goal_swap 1 and 8: out-of-distribution states).
- The current position is read from the state (E05: shifting the state's eef shifts the reach with slope about -1).

## Perception that works without simulator state
- The policy's own SigLIP patch features find moved objects (E08) and track a destination region (E14: 2 cm on
  spatial_swap 0).
- Agentview camera model (scene.project/backproject) for pixel <-> world; actions: 0.0095 m per unit per step, world axes.

## Tracking quality is the bottleneck (E14, E30, E33, E34)
- When the tracked destination displacement is right, carry_handback2 succeeds ~40-50%; when it is wrong, 0%.
- Single reference frame: median error ~2-5 cm but confident zero-shift failures in ~1/3 of frames (E30); a median over
  tracker settings does not help (E30); a plate prototype averaged over references is worse (E33, context matters).
- Voting over 3 reference frames (trained episodes 0-2) cuts > 10 cm failures 4/9 -> 2/9, median 1.2 cm (E34) -> carry_handback3.

## What the policy needs to place (E27, E29)
- The phase/release is read from the IMAGES (both images: 54% of the switch to 'holding' behaviour); the final local motion is
  shaped by the wrist view (blind expert: 10x noise path change). So the policy releases only when it SEES the destination
  under the object; a pure state shift (vf_place) carries the bowl around but never releases.
- LIBERO 'On' needs the object centre within 3 cm of the target centre: open-loop scripted placement (carry_to_target) misses.

## Programs
- `carry_to_target.py`: VLA picks; program carries and releases: 0/5 (precision), 1/3 even on the trained layout.
- `vf_place.py`: virtual frame during transport only: 0/5 (no release: images never show the plate).
- `carry_handback.py` / `carry_handback2.py` (best): VLA picks; program carries the object above the image-tracked
  destination (and descends to 2 cm above the reference place height); then hands control back to the VLA with its state
  shifted by the tracked displacement, so it believes it is at its memorised place point while the cameras show the real
  plate below: the VLA does its own final alignment and release. spatial_swap 0: 5/15, 5: 5/15, 4: 8/15, 2: 6/15 (frozen 0/10).
- `carry_handback4.py` (best): + local search: if the VLA does not release within 25 steps of the hand-back (it hovers when
  its wrist view does not show the object over the destination), nudge 2-3 cm and hand back again: spatial_swap 0 9/15,
  2 5/5, 9 4/5, 4 3/5, 5 1/5. Failure modes:
  tracking misses (confident zero-shift matches when another object now stands at the old spot, E30) and occasional
  wandering after the hand-back.
