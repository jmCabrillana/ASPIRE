---
name: grasp
description: Gripper feedback caveats discovered during fix-loop debugging. close_gripper() return value, and when gripper-open-fraction is (and isn't) a useful grasp-success signal.
---

# Grasp — Gripper Feedback

## close_gripper() does not return gripper width

`close_gripper()` returns `None`. The `gripper_width` field visible in
`trace.json`'s `close_gripper` entries comes from external trace/sandbox
instrumentation around the call, not from the function's return value — it is
not readable by generated code. Do not branch retry logic on an assumed
return value from `close_gripper()`.

If grasp-quality feedback is needed inside generated code, use real
proprioception instead:

```python
obs = get_observation()
gripper_open_fraction = obs["robot_cartesian_pos"][7]  # 0 = closed, 1 = open
```

**Provenance:** `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65, fix-loop
round 0 of `library_lex_s10_r5`. An iteration that assumed `close_gripper()`
returned a width never triggered its retry path (confirmed via `gripper_width
= None` in the eval log) while still applying an unconditional extra-descend
offset, regressing pass rate from 4/10 (probe) to 4/15.

## Gripper-open-fraction doesn't discriminate grasp success on thin/small objects

`obs["robot_cartesian_pos"][7]` is real signal, but on a thin object (a
burner knob) it reads ~0.008-0.05 on nearly every trial regardless of whether
the trial ultimately passes — the object simply doesn't hold the gripper much
open either way. Don't gate a retry/branch on this signal for thin or small
objects (knobs, dials, buttons, thin handles) without first checking, for the
object class at hand, that the signal actually separates pass from fail.

**Provenance:** same source as above. An iteration that retried whenever this
fraction was `< 0.03` fired on almost every trial (pass and fail alike),
adding motion/time with no discriminating power; net pass rate 5/15, no
better than the baseline.

**Extension — an anomalously *high* reading doesn't discriminate either.**
On the same task, `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65,
fix-loop round 4 of `library_lex_s10_r5`: several trials showed one cycle's
post-close gripper width spike to 0.27-1.0 (vs. the normal thin-knob range of
~0.005-0.05). This spike appeared on both passing trials (52, 55, 56, 61) and
failing trials (58, 60, 62, 63, 64) — don't add a retry/abort gate on a
*high* reading either, for thin/small-object grasps, without first checking
it separates outcomes for the specific object/task.

## A confirmed lift does not mean the held orientation is safe to change

Never reorient the wrist while an object is held only by a rim/edge/friction
grasp (not a firm parallel-jaw clamp through its full width). Compute the
grasp quaternion once at pick time and reuse it unchanged for every later
waypoint (pregrasp, grasp, lift, place-above, place, retreat) — do not
introduce a second, separately-chosen orientation (e.g. a "nice" fixed
top-down quat) for the placement motion:

```python
ok, grasp_quat = try_pick(obb, descend_frac, yaw)
held_quat = grasp_quat  # save it

# every placement waypoint reuses held_quat — never a second target quat
place_quat = held_quat
safe_move(solve_ik(place_above, place_quat))
safe_move(solve_ik(place_pos, place_quat))
open_gripper()
```

**Provenance:** `libero_goal_swap/put_the_bowl_on_the_stove`, dev seeds
51-65, fix-loop round 3 of `library_lex_s10_r5`. A zero-shot program grasped
the bowl with one quaternion and then forced a different, hardcoded top-down
quaternion for placement; it achieved a confirmed lift (height rise
~0.21-0.23m) on 10/10 probed seeds yet placed successfully on 0/10 — the
reorientation itself was isolated as the cause (place XY/height were
otherwise unchanged). Removing the reorientation (reusing the grasp quat
throughout) raised the pass rate to 2-3/15 on the same pick strategy. A
confirmed lift only proves the grasp sufficed for a straight vertical
transport, not for an off-axis wrist rotation.
