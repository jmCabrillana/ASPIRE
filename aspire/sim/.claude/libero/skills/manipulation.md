---
name: manipulation
description: Contact-task patterns — ratcheted grasp/turn/release cycling for rotations a single grip can't sustain (knobs, dials, valves).
---

# Manipulation — Ratcheted Turning

## Trigger

A task requires a rotation that one grasp-and-sweep motion cannot reliably
complete in a single attempt: knobs, dials, valves, or anything turned past a
friction-limited grip span where a single continuous sweep tends to slip
partway through.

## Pattern: independent grasp/turn/release cycles, re-localizing fresh each time

Instead of one continuous grasp + large-angle sweep, run several independent
grasp -> turn -> release cycles, re-observing the scene between cycles (see
[localize.md](localize.md) for why the arm must clear the camera's view
first):

```python
def turn_cycle(grasp_pos, grasp_quat, pivot_center, turn_deg, n_steps, direction_sign):
    open_gripper()
    joints = solve_ik(grasp_pos + np.array([0.0, 0.0, 0.08]), grasp_quat)
    if joints is None: return False
    move_to_joints(joints)
    joints = solve_ik(grasp_pos, grasp_quat)
    if joints is None: return False
    move_to_joints(joints)
    close_gripper()
    radius_vec = grasp_pos - pivot_center
    for step in range(1, n_steps + 1):
        angle = direction_sign * np.deg2rad(turn_deg) * (step / n_steps)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rx = cos_a * radius_vec[0] - sin_a * radius_vec[1]
        ry = sin_a * radius_vec[0] + cos_a * radius_vec[1]
        new_pos = pivot_center + np.array([rx, ry, radius_vec[2]])
        new_quat = quat_mul(quat_about_z(angle), grasp_quat)
        joints = solve_ik(new_pos, new_quat)
        if joints is None: break
        move_to_joints(joints)
    open_gripper()
    goto_home_joint_position()  # clear the camera's view before re-observing
    return True

# main loop: several independent cycles, re-localizing fresh each time
for cycle in range(3):
    grasp_pos, grasp_quat, pivot_center = locate_object()  # fresh get_observation() + segment + plan_grasp
    if grasp_pos is None: continue
    turn_cycle(grasp_pos, grasp_quat, pivot_center, turn_deg=90.0, n_steps=6, direction_sign=-1.0)
```

**Provenance:** `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65,
fix-loop round 0 of `library_lex_s10_r5`. Replacing a single grasp + 90deg
sweep with 3 independent 70deg grasp/turn/release cycles raised dev pass rate
from 4/10 (single-sweep probe) to 6/15, recovering several seeds (53, 56, 59,
61, 62) that the single-sweep baseline failed. The remaining ~60% failure
rate on this task looked like genuine contact-physics slip stochasticity, not
a further diagnosable bug — this pattern raises but does not guarantee
success on thin-grip turning tasks. May also generalize to "regrasp and
repeat" for long push/pull transport motions; unverified.

(`turn_deg` above was retuned to 90 in round 1 — see "Tune turn_deg before
n_cycles" below; the call-site parameter is the only thing that changed, the
`turn_cycle()` skeleton itself is unchanged.)

## Tune turn_deg before n_cycles

**Trigger:** the ratchet pattern above is not reaching 15/15 even though each
cycle's grasp looks solid (`gripper_width` small/firm in `trace.json`).
Before adding more cycles, try increasing the per-cycle `turn_deg` instead —
each release point appears to give back some of the turned position
(spring-back/detent), so more release points can net *less* cumulative
rotation even with a larger total-degree budget.

**Evidence:** `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65, fix-loop
round 1 of `library_lex_s10_r5`. Raising `turn_deg` 70 -> 90 at fixed
`n_cycles=3` took this task from 5/15 to 11/15. Simultaneously *lowering*
`turn_deg` to 55 and raising `n_cycles` to 5 dropped it to 2/15. Holding
`turn_deg=90` fixed and raising `n_cycles` 3 -> 4 also regressed, to 9/15.
There are diminishing/negative returns on `turn_deg` too (110deg regressed
slightly to 10/15 and produced one step-budget crash) — this is a "tune
`turn_deg` before tuning `n_cycles`" ordering, not an unbounded "bigger is
always better."

**Round 2 re-check (same task/seeds):** re-running the unmodified
`turn_deg=90, n_cycles=3, n_steps=6` program reached 13/15 (its best
recorded score). Two follow-up attempts to chase the remaining 2 failures —
raising `n_steps` 6 -> 10 for a finer per-cycle sweep, and adding a one-shot
re-grasp retry triggered on an "air grasp" gripper-width outlier — both
regressed (7/15 and 12/15 respectively). On a task already clearing ~85%+,
adding per-cycle overhead (finer IK resolution, a mid-cycle retry/extra
`get_observation()`) costs episode step-budget headroom and can net *fewer*
completed cycles/rotation even when individually well-motivated; re-check
headroom before adding such logic to an already-tuned ratchet loop. The
retry's trigger condition also never fired on these seeds — `gripper_width`
stayed uniformly small (~0.01-0.03) on both the passing and the 2 failing
trials, with no outlier, reinforcing the caveat below that this signal
doesn't discriminate pass/fail for a thin knob. The residual 2/15 failures
showed no mechanical signal at all (perception, grasp, and IK all succeeded
in every cycle of every trial) — this is read as genuine contact-physics
stochasticity, not a further diagnosable bug within this method.

## Guard move_to_joints / goto_home_joint_position against mid-episode termination

**Trigger:** any multi-cycle or multi-step program that calls `move_to_joints`
or `goto_home_joint_position` repeatedly and could plausibly run the
simulator past its episode step budget (`max_steps` in the env config; each
`move_to_joints_blocking` call can itself use up to ~120 sim steps). Once the
episode has terminated, a further call raises `ValueError: executing action
in terminated episode` deep in the LIBERO/robosuite stack, which crashes the
whole trial instead of just ending the program's own loop. A crash scores the
same reward (0) as a graceful stop, so there is no reason to let the
exception propagate.

```python
def safe_move(joints):
    try:
        move_to_joints(joints)
        return True
    except Exception as exc:
        print(f"move_to_joints raised, stopping early: {exc}")
        return False

# inside any per-cycle motion helper:
if not safe_move(joints):
    return False   # end this cycle/attempt cleanly instead of crashing
```

Wrap `goto_home_joint_position()` the same way wherever it is called after a
sequence of motions, and break out of any outer retry/cycle loop when a cycle
returns `False`.

**Evidence:** `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65, fix-loop
round 1 of `library_lex_s10_r5`. An earlier zero-shot variant (an adaptive
ratchet with an extra direction-probe cycle, up to 5 cycles total) crashed on
most seeds from exactly this failure mode. Adding the guard reduced but did
not fully eliminate the crash class — one iteration at 110deg/8-steps-per-cycle
still crashed once via a differently-typed caller path outside `safe_move`'s
direct coverage — so iterations that push cycle count/degree aggressively
should keep headroom under the step budget rather than relying solely on the
guard.

## Prefer a top-down-aligned grasp pose for any pure-Z-axis rotation task

**Trigger:** any `turn_cycle`-style skeleton above, where the grasp pose is
chosen from `plan_grasp`'s candidate list. The motion computes every waypoint
by rotating strictly about world +Z, which implicitly assumes a top-down
grip — a raw `np.argmax(grasp_scores)` pick can silently return an off-axis
candidate that fights that assumption.

```python
try:
    top_down_world, top_down_score = select_top_down_grasp(
        grasp_poses_cam, grasp_scores, extrinsics, vertical_threshold=0.7,
    )
except Exception:
    top_down_world = None

if top_down_world is not None:
    grasp_world = top_down_world
else:
    top_idx = int(np.argmax(grasp_scores))
    grasp_world = extrinsics @ grasp_poses_cam[top_idx]

grasp_pos, grasp_quat = decompose_transform(grasp_world)
```

**Evidence:** `libero_goal_swap/turn_on_the_stove`, dev seeds 51-65, fix-loop
round 4 of `library_lex_s10_r5`. Replacing raw-argmax grasp selection with
`select_top_down_grasp(..., vertical_threshold=0.7)` (falling back to raw
argmax only if nothing clears the threshold) raised pass rate 8/15 -> 11/15
with no other change; the top-down candidate was found and used on 40/40
grasp cycles across all 15 trials (the fallback branch never fired in this
run).

**Anti-pattern:** an unconditional geometric sanity filter (e.g. rejecting
non-"compact" masks) applied on every re-localization, overriding an
already-correctly-scoring mask, regressed hard (1/10 on probe) in the same
round. A geometric sanity filter like this should only run as a fallback
when the primary localization path fails outright — never as an
unconditional override.
