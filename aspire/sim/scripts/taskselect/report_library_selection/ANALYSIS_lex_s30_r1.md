# Analysis of `lex`, 30 seeds, 5 repairs (`library_lex_s30_r1`) against the earlier runs

Date: 2026-10-16. Numbers: `analysis_runs.py`. The `unsolved` (random) arm of this repetition is still running, so **this
is not yet a `lex` against random comparison**. It places the first long run next to the earlier ones and says how large a
difference the noise allows us to see.

## 1. Quantitative

Metric: mean pass rate over the 27 programs (9 tasks x 3), same 9 tasks in every run.

| run | design | round 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|---|
| `lex` r1 (pilot) | 15 seeds, 3 repairs | 0.091 | 0.064 | 0.057 | 0.141 | | |
| `unsolved` r1 (pilot, yesterday) | same | 0.091 | 0.067 | 0.148 | 0.052 | | |
| `lex` r2 | same | 0.091 | 0.104 | 0.126 | 0.128 | | |
| `unsolved` r2 | same | 0.091 | 0.111 | 0.128 | 0.099 | | |
| **`lex` s30 r1** | 30 seeds, 5 repairs, seeds 51-65 | 0.032 | 0.049 | 0.094 | 0.052 | 0.089 | **0.156** |
| same run, seeds 66-80 only | never used by any repair | 0.025 | 0.074 | 0.069 | 0.044 | 0.077 | **0.138** |

- **Noise floor** (five independent round-0 generations from the empty library, seeds 51-65): mean 0.065, SD 0.029
  (0.058 and 0.029 without the hand-patched one). One run's mean moves by that much with no change to the library.
- **The difference of two single runs has SD 0.042.** At 5% significance and 80% power the smallest difference we can reliably
  see is 0.117 with one run per arm, 0.083 with two, 0.067 with three, 0.048 with six. The campaign (3 repetitions) can
  therefore show a gap of about 0.07 or more, not smaller.
- **This run rose from 0.028 to 0.147** over the five repairs (+0.119, 95% interval [+0.009, +0.247], bootstrap over tasks and
  seeds). On seeds 66-80, which no repair used, it rose from 0.025 to 0.138, so the gain is not repair-seed overfitting.
  The final value is 3.1 noise SDs above the mean of fresh round-0 generations.
- **But the path is noisy.** The five round steps are +0.034, +0.019, -0.033, +0.035, +0.064. Their SD (0.036) is about the
  noise SD. Only the last step is clearly beyond it. The linear trend is +0.018 per round.
- **Against the earlier final rounds** (0.141, 0.052, 0.128, 0.099; mean 0.105) the final 0.156 is higher than all four, but
  only 1.2 noise SDs above their mean. It had two more repairs. This run also started lower (0.032 against 0.091), so
  "gain over round 0" overstates it. We cannot say 5 repairs beat 3.
- **Crashed trials fell to 0 to 19% (4% in the final round)**, against 13 to 43% in `lex` r1. This is the effect of putting the
  API facts in the writers' prompts. The pilot's point-prompt defect is therefore gone from the new runs.
- **No task was solved on all 30 seeds** in any round. Best single-task counts of 90 trials in the final round: salad dressing
  51, turn on the stove 30, bowl on the stove 19.

## 2. Qualitative

**What the repairs found (five repairs, 1 of 5 passed its 15 development seeds with the kept program):**

| round | task repaired | diagnosis | library addition |
|---|---|---|---|
| 0 | open the middle drawer | handle count unstable; but handle choice was not what decided pass or fail | do not add motion churn without checking known-good seeds; ordinal targets; never retry from a stale pose |
| 1 | bowl on the stove | an absolute GraspNet score threshold rejects every grasp; 4 pick attempts help (5 to 11 of 15) | more retries beat tuning; no absolute score filter |
| 2 | bowl on the stove again | two clusters of placement target from one stove mask (height bimodality) | height-bimodality guard; density-mode centroid |
| 3 | bowl between plate and ramekin | "ramekin" gets score 0.001, so the betweenness test never ran and the wrong bowl was picked; grip check read at the wrong moment | read the **settled** gripper width, not the return value of `close_gripper()`; no score filter on landmark objects |
| 4 | salad dressing | clean trace, "inside footprint", reward 0: release dynamics, not targeting | one run's pass count cannot validate a change; release-dynamics signature |

**One library element clearly changed the programs.** The settled-width rule (added after round 3) was used by 0 of 27
programs before it, then by **23 of 27 in round 4 and 24 of 27 in round 5**. The programs that use it score higher than those
that do not (0.126 against 0.043), but the "without" group is 7 programs spread over three tasks, so this proves adoption, not
benefit. Its best case is salad dressing: 3 passes (round 3) to 42 and 51 (rounds 4 and 5), but that task had also just been
repaired, with the same development seeds in the repair.

**Knowledge transfers across tasks.** The stove-bowl and drawer repairs lifted a task nobody repaired: "turn on the stove" went
5, 11, 14, 17, 10, 30 of 90. Generic grasp-retry and placement-target rules (density mode, pick attempts) reached its programs
(density-mode code appears in round 3, the settled-width check in all three round-5 programs).

**Comparison with the earlier runs.**

- Yesterday's `unsolved` runs added mostly **perception prompts and learned-planner rules** (appearance prompts, wrist-camera
  re-localisation, IK-stall escapes). This `lex` run added mostly **grasp-verification and retry rules** and placement-target
  statistics. The `lex` pilots (r1 and r2) also stayed in grasp/placement territory (thickness filter, no reorientation after
  grasp), so the style difference follows the first repaired task more than the selection rule.
- Libraries grow with repairs: 4.7 KB (L01) to 17.6 KB (L05) here, 15 KB at the end of the 3-repair pilots. The same task can
  be picked twice (bowl on the stove in rounds 1 and 2 here; the pilots picked the stove or bowl tasks in both arms).
- The `lex` rule chose the bowl on the stove twice because its disagreement stayed high after a failed repair. That is
  consistent with the rule, and in the pilot the stove repair also looked like noise (identical programs scored 7 and 13 of 15).
- `lex` r2 and the unsolved arm of r2 both ended around 0.10 to 0.13: there is no visible gap between rules at 15 seeds and
  3 repairs, which matches the noise level above.

## 3. What can be claimed

- Supported: five repairs on 30 seeds raised the mean pass rate from 0.03 to 0.15 in this run, outside the noise (about 3 SD)
  and on seeds the repairs never used. A generic rule was adopted by almost every later program and helped a task that was
  not repaired.
- Not supported: that `lex` beats random selection (the paired arm has not finished); that 5 repairs beat 3; that any single
  rule caused the gain.
- Next: the unsolved arm of this repetition (running), then two more repetitions. Only a mean gap of about 0.07 or more
  between arms, over three repetitions, would be distinguishable from noise.
