---
name: localize
description: Perception-preflight pattern — clear the arm from the camera's view before re-observing the scene, to avoid self-occlusion corrupting re-localization.
---

# Localize — Perception Preflight

## Clear the arm's view before re-observing

Any program that moves the end-effector near an object and then calls
`get_observation()` again to re-localize (retry loops, multi-step
manipulation, repeated grasp-and-release cycles) should move the arm out of
the camera's field of view first:

```python
open_gripper()
goto_home_joint_position()   # get the arm out of the camera's field of view
obs = get_observation()      # now safe to re-segment / re-localize
```

**Provenance:** `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65,
fix-loop round 0 of `library_lex_s10_r5`. Without the reset, a trial's
localized object center drifted across repeated grasp cycles within the same
trial (e.g. world x: 0.624 -> 0.587 -> 0.570, with an implausible z ~ -0.005
on the last cycle — below the table surface), consistent with the arm
partially occluding the agentview camera after its own motion. Adding the
reset between cycles removed the drift pattern in the traces (pass count on
this 15-seed sample was unchanged, 6/15 either way — this is a correctness
fix for the re-localization step, not something that will always move the
headline number).

## Sanity-filter a re-localized static target by expected geometry, not score alone

Right after a pick, re-localizing a static scene fixture (table, stovetop,
shelf, drawer front) by text prompt can accidentally re-find the object
you're now holding, if it's newly visible (e.g. lifted into view) and
happens to score highest. Filter candidates by a plausible geometric property
of the real target (e.g. flatness for a tabletop/stovetop) before trusting
the top segmentation score:

```python
def locate_flat_surface(prompts, max_extent_z=0.02):
    candidates = []  # (score, mask, obb)
    for prompt in prompts:
        masks = segment_sam3_text_prompt(rgb, prompt)
        cand = best_mask(masks)
        if cand is None or cand.get("score", 0.0) <= 0.0:
            continue
        pts = mask_to_world_points(cand["mask"], depth, intrinsics, extrinsics)
        obb = get_oriented_bounding_box_from_3d_points(pts) if pts is not None and len(pts) >= 5 else None
        candidates.append((cand["score"], cand, obb))
    flat = [c for c in candidates if c[2] is not None and float(np.asarray(c[2]["extent"])[2]) <= max_extent_z]
    pool = flat if flat else candidates
    score, mask, obb = max(pool, key=lambda c: c[0])
    return mask, obb
```

**Provenance:** `libero_goal_swap/put_the_bowl_on_the_stove`, dev seeds
51-65, fix-loop round 3 of `library_lex_s10_r5`. On 3 of 15 dev seeds, the
"burner" text prompt's top-scoring mask was the just-lifted bowl itself
(`extent_z` ~0.06, bowl height) rather than the stovetop (`extent_z` ~0.006,
flat). Adding the flatness filter fixed 2 of those 3 seeds (confirmed
correct flat-surface detection afterward); the third still failed, but at
the earlier pick stage, unrelated to this fix.
