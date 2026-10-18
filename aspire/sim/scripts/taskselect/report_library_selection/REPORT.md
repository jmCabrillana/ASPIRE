# Library build with task selection: first comparison and noise floor

Date: 2026-10-14 onwards. One RTX 5090. Model: `claude-sonnet-5` for every writer and repair agent.
Figures: `figs/` (make them with `make_figures.py`). Code: `scripts/taskselect/library_loop.py`.

## Summary

- We built a skill library from nothing in two ways, one run each: the next task to repair chosen by
  **disagreement** (`lex`), and chosen **at random among unsolved tasks** (`unsolved`).
- **No difference can be claimed.** Each round runs 27 programs (9 tasks x 3) for 15 trials each, 405 trials. The final pass
  rate is 0.141 (`lex`) against 0.052 (`unsolved`), but three independent generations of the *same* empty library already
  range from 0.035 to 0.091, and every 95% interval of the gap contains 0.
- Two clear library effects were seen, one in each arm, and both are real mechanisms, not selection effects:
  a perception rule that was used and helped, and a perception prompt that helped and then was not reused.
- A pipeline defect (point-prompt API misuse, never fixed in the writers' instructions) crashed up to 30% of the `lex` trials in
  rounds 1 to 3. It biases the comparison against `lex`. It is fixed in all later runs.

## Proposition and hypotheses

**Setting.** A language model writes a Python program for a robot task from a skill library (markdown files of rules and
helpers). When programs fail, an agent repairs one task, and the lesson is written back into the library. The library is the
only thing that carries knowledge from one round to the next. Repairing is expensive (about an hour of agent and simulator
time per task), so a build repairs only a few tasks. **Which task to repair next decides what the library learns.**

**Proposition.** The most informative task to repair is the one where the current programs **disagree**: some pass and some
fail on the same seeds, many unsolved tasks fail at the same stage, and the failures have varied traces. Disagreement says that
the task is neither hopeless nor already solved, and that a fix at a shared failure stage will carry over to other tasks.
Repairing by this rule (`lex`) should build a better library than repairing a random unsolved task (`unsolved`) at the same
number of repairs.

**Hypotheses** (all about the library after the same number of repairs, same tasks, same writers):

| id | statement | measure | how it can fail |
|---|---|---|---|
| H1 (primary) | the final library of `lex` gives a higher pass rate than that of `unsolved` | mean pass rate over 27 programs (9 tasks x 3), and over 6 programs per task, on 30 seeds | gap within the noise, or negative |
| H2 | `lex` improves a library faster along the rounds | gain over round 0, mean over rounds 1 to 5 | no gain difference |
| H3 | `lex` lessons transfer: tasks that were not repaired improve | per-task passes, unrepaired tasks | gains only on repaired tasks |
| H0 (noise) | one run's mean moves by 0.03 (SD) with no change in the library, so an effect smaller than the detectable gap cannot be shown | six round-0 generations from the empty library | (this is a measured fact, not a claim) |

**What can be concluded.** The smallest gap that two arms can reliably show (5% level, 80% power) is 0.106 with 1
repetition, 0.075 with 2, 0.061 with 3, 0.053 with 4, 0.043 with 6. A result below that is reported as "not shown", not as
"no effect".

## Method: the library, the loop and the update rule

**The library.** Markdown files in `.claude/libero/skills/` (`grasp`, `localize`, `manipulation`, `transport`, `README`).
Writers read them as context; programs never import them. A build starts from the empty library `L00`, and each repair
produces the next one (`L01`, `L02`, ...). Every version is stored in `outputs/<run>/library/Lnn`.

**Algorithm.**

```
L <- empty library
repeat 5 times:
    P <- 3 fresh programs per task, written from L          # 9 tasks
    run P on seeds 51..80                                   # 810 trials
    U <- tasks where no program passes every seed
    t <- pick(U)                                            # the only difference between arms
    findings <- repair(t)                                   # up to 5 tries, dev seeds 51..65
    L <- L + lessons(findings)                              # add rules, never remove
final: 3 fresh programs per task from L, run on seeds 51..80

pick(U), arm "unsolved":  a random task of U
pick(U), arm "lex":       the task of U with the largest (split, shares, trace), compared in that order
    split  = how much programs disagree per seed (entropy of how many pass)
    shares = how many tasks of U fail at the same stage
    trace  = how varied the failure traces are
```

A trial's failure stage comes from its API trace only: `crash`, `perceive` (object not found), `no_motion`, `no_close`, `air`
(closed on nothing), `slip` (held, then lost), `held_end`, `released` (placed wrong). `LEX.md` has a worked example.

**Repair.** The agent reads the passing and failing traces of the task's 3 programs, then writes up to 5 task-level programs
(no seed branches, no hard-coded coordinates), each run on the 15 dev seeds and diagnosed from its failed trials. The kept
program is the one with the most passes (ties: the later one). It writes `findings.md`: root causes, pass counts and
"Generalizable Patterns".

**Library update rule (promotion).** After the repair, the coordinator reads `findings.md`, edits the skill files and logs it in a
ledger (`skill_promotions.jsonl`: files changed, reason). Rules in force:
1. A lesson is added as a short rule with its trigger and evidence, in the file of its category (grasp, localise, place,
   carry). Generic patterns are preferred over task facts.
2. No rule is ever removed or corrected: the library only grows (4.7 KB after the first repair, 17.6 KB after five in one run).
3. The kept program's pass count on the dev seeds is the only evidence required; there is no test on other seeds.
4. Skills may use the allowed APIs only (no simulator state, no `.bddl`/`.xml` reads).

**Protocol and controls.**

| item | value |
|---|---|
| pool | 9 swap tasks: goal (3), spatial (3), object (3); 21 more swap tasks are held out and unused so far |
| probe seeds | 51-80 (30), every round; repair development seeds are 51-65 |
| repetition | round 0 is written once and shared by both arms; arm order alternates; 5 repetitions planned |
| same in both arms | model (`claude-sonnet-5`), K, writers' prompts, repair budget, seeds, API facts note |
| differs | only the selection rule |
| noise control | six independent round-0 generations from the empty library, and a re-measurement of each final library with 3 more programs per task |

## Setup

| item | value |
|---|---|
| pool | 9 swap tasks: goal (3), spatial (3), object (3), `library_9_swap` |
| start | empty skill library `L00`; round 0 shared by both arms (27 programs, 9 patched by hand for API errors, see Defects) |
| per round | 3 fresh zero-shot programs per task from the current library, probed on seeds 51-65 (405 trials) |
| repair | one task per round, up to 5 iterations from its best program, then skill promotion |
| rounds | 3 repairs, then a final probe (4 probes per arm) |
| selection | `lex`: highest per-seed split, then most tasks sharing the failure stage, then trace variety (explained, with a worked example, in `LEX.md`). `unsolved`: uniform among tasks with no program passing 15 of 15 |
| differences between arms | `unsolved` probed with 4 workers, `lex` with 2 (speed only; outcomes are not deterministic at either) |

## 1. The two arms

![arms](figs/fig1_arms_by_round.png)

| round | `lex` pass rate | `unsolved` pass rate | `lex` crashed trials | `unsolved` crashed trials |
|---|---|---|---|---|
| 0 (shared) | 0.091 | 0.091 | 13% | 13% |
| 1 | 0.064 | 0.067 | 43% | 19% |
| 2 | 0.057 | 0.148 | 24% | 4% |
| 3 (final) | 0.141 | 0.052 | 18% | 32% |

- "Pass rate" is the passes of all 27 programs (9 tasks x 3 programs, 15 trials each) divided by the 405 trials.
- The best program per task is not used as a metric: a maximum over 3 noisy programs moves by 0.07 to 0.09 between identical
  generations.
- Solved tasks (one program passes 15 of 15): at most 1 of 9 in any round, in both arms.
- Repairs: `lex` picked salad dressing, bowl between plate and ramekin, turn on the stove. `unsolved` picked bowl next to the
  ramekin, cream cheese, bowl on the stove. Each arm solved one of its three repair tasks.
- The sign of the difference flips between rounds (+0.089 in round 3, -0.091 in round 2).

![tasks](figs/fig2_task_by_round.png)

The per-task view (passes of 45 trials: 3 programs x 15 seeds) shows how unstable single tasks are: salad dressing in `lex`
goes 14, 18, **0**, 6; cream cheese in `unsolved` goes 0, 8, **36**, 1. Neither can be explained by the library alone.

## 2. Noise floor

![noise](figs/fig3_noise_floor.png)

Five independent generations of programs from the same empty library, same pool, same seeds (51-65). Each generation is 27
programs (9 tasks x 3) with 15 trials each, 405 trials. The plot shows, for each generation and for the final round of each
arm, the mean and SD over the 27 programs of each program's pass rate (passes of 15).

| generation | mean ± SD over the 27 programs | passes of 405 |
|---|---|---|
| fresh batch a | 0.035 ± 0.087 | 14 |
| fresh batch b | 0.074 ± 0.119 | 30 |
| fresh batch c | 0.091 ± 0.178 | 37 |
| fresh batch d | 0.032 ± 0.077 | 13 |
| fresh batch e | 0.077 ± 0.191 | 31 |
| `lex`, round 3 | 0.141 ± 0.250 | |
| `unsolved`, round 3 | 0.052 ± 0.167 | |

- The SD is large against the mean: most programs pass 0 of 15 and a few pass most seeds.
- The mean of the five generations is 0.062; the SD between generations is 0.027. That is the noise on one run's mean.
- The two arms start from a round 0 that was written once and shared; 9 of its programs needed a hand patch for API errors (see
  Defects), so it is not used in this noise estimate.
- Execution noise adds to this: re-running identical programs on the same seeds changed 11 of 15 per-seed outcomes for one
  program (earlier control).

**Reading the arms against this.** The difference between two single runs has an SD of about 0.04 (0.027 x sqrt 2 = 0.038).

| round | gap `lex` - `unsolved` | 95% interval (same tasks, seeds resampled) |
|---|---|---|
| 1 | -0.002 | [-0.101, 0.121] |
| 2 | -0.091 | [-0.291, 0.052] |
| 3 (final) | +0.089 | [-0.032, 0.269] |

All three intervals contain 0. The final-round gap is about 2 noise SDs, but it appears in **both directions** (round 2 favours `unsolved` by the
same amount). With one run per arm, three noise points and several metrics and rounds looked at, that cannot be read as an
effect of the selection rule. It does show that library changes or generation variance move the metric by that much between
rounds. A comparison needs several repetitions per arm, which is what the campaign now running provides.

## 3. The two libraries

| | `lex` (L03) | `unsolved` (L03) |
|---|---|---|
| files | grasp, localize, manipulation, transport, README: 18 KB | localize, manipulation: 15 KB |
| edits | additions only, no rule removed or corrected | additions only |
| general rules | grasp plausibility filter; proprioceptive grasp check; min-thickness filter; landmark exclusion | out-of-vocabulary prompts are not "absent"; wrist-camera re-localisation with a displacement gate; floor-based drop height; learned planner then forced top-down |
| task-specific | push-flick for the stove knob; `DROP_CLEARANCE = 0.11` | gate values 0.08 and 0.12 m with the cream-cheese basket as evidence |

Neither library is clearly better. `lex` covers more skill categories because its three repairs hit different failure kinds.

## 4. Which library elements measurably helped

| element | evidence | verdict |
|---|---|---|
| **min-thickness filter** (`lex`, from the stove repair) | no round-1 or round-2 stove program implements a thickness filter (one comment mentions "flat"); all three round-3 programs do; passes 0/1/5 become 8/10/14 | best evidence in the data |
| **describe by appearance** (`unsolved`, from the ramekin repair) | all 3 cream-cheese programs in rounds 1 and 2 use the prompt "white and blue box"; round 2 passes 11/15/10 | helped, then lost: no round-3 program used it (1 pass of 45), with the rule still in the library |
| learned planner then forced top-down (`unsolved`, round 2) | programs calling `plan_grasp` rose from 13 to 19 of 27; "No grasp candidates found" 31 times, "All grasp attempts failed" 31 times; crashed trials 4% to 32% | correlation only; harmful in that round |
| drop clearance, yaw fallbacks | no durable effect | neutral |

Why a better element appeared at all:

1. **The valuable elements are perception diagnoses**: why the object was not found or was wrongly picked. Control tweaks
   (clearances, yaw fallbacks, planner settings) gave variable results.
2. **A repair's apparent gain can be pure noise.** In the stove repair, iterations 2 and 5 are byte-identical programs (checked with
   `cmp`) and scored 7/15 and 13/15; the agent itself promoted a caution about this.
3. **Knowledge is not reliably reapplied.** A prompt that worked in two rounds was unused in the third with the library
   unchanged. The library stored the rule, not the concrete prompt; writers are not required to follow it.

## 5. Defects and deviations (read before trusting the numbers)

- **Point-prompt API misuse recurred in every later round.** The function takes one `(x, y)` pair; programs passed a list. The
  allowed-API list in `CLAUDE.md` says `points`, which invites it. I patched round 0 by hand and did not add the warning to
  the writers' instructions as intended. Trials with the resulting 422 error:

  | arm | round 1 | round 2 | round 3 |
  |---|---|---|---|
  | `lex` | 120 | 90 | 60 |
  | `unsolved` | 30 | 15 | 0 |

- **Round 0 of the two arms was patched by hand** (9 programs: missing `import numpy as np`, list-wrapped point prompt, dict passed as
  a mask); later rounds and the noise batches were not.
- **Molmo is off**, as in the upstream TC1 runs. It was removed from 11 round-0 programs and the writers were told not to use it.
- **Worker count** differed between the arms (2 against 4); a control showed it is not a source of bias.
- **One run per arm.** Nothing here is a significance statement.

## Where things are, by experiment id

All paths are under `aspire/sim/outputs/` (gitignored). The live directory `.claude/libero/skills/` is only a working
copy: each run saves its own libraries, so it is restored to the committed version when no run is active.

| experiment id | what it is | libraries | probes and selections | repairs |
|---|---|---|---|---|
| `library_lex_r1` | arm `lex`, 3 repairs + final probe | `library/L00` (empty) to `L03` | `round_NN/probe.json`, `round_NN/selection.json` | `round_NN/fixloop/<suite>/<task>/findings.md`, `skill_promotions.jsonl` |
| `library_unsolved_r1` | arm `unsolved` (also holds the shared round 0) | `library/L00` to `L03` | same | same |
| `library_noise_a`, `library_noise_b` | fresh round-0 generations from the empty library | `library/L00` | `round_00/probe.json` | none |
| `library_lex_r2`, `library_unsolved_r2` | pilot repetition 2 (15 seeds, 3 repairs; shared round 0) | `library/L00` to `L03` | same | same |
| `library_lex_s30_r<k>`, `library_unsolved_s30_r<k>` | campaign (k = 1 to 5); `outputs/remeasure/<run>` holds the 3 extra programs per task and their probe (30 seeds 51-80, 5 repairs, 6 probes; shared round 0 per repetition) | `library/L00` to `L05` | same | same |
| `skill_snapshots/family_start` | skills after the first single-task fix loop; equals the committed `.claude/libero/skills` | 5 files | none | none |

Each run also has `state.json` (pool, rounds, settings) and `coordinator.md` (the exact prompt it was started with).
Round 0 of `library_lex_r1` is a byte-identical copy of the shared one (`probe.json` compared with `cmp`).
