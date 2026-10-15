# The `lex` rule: which task to repair next

`lex` is the "disagreement" arm. It looks at the programs that were just written and run for every unsolved task, and
picks the task where the programs **disagree**, because a repair has contrasting evidence to learn from there.

Code: `scripts/taskselect/library_loop.py` (formula in the header, branch in `cmd_select`), `core.probe_score`
(split, trace) and `core.task_mode` (failure stage).

## What it needs

For the current library, K = 3 programs per task, each run on seeds 51-65. Only **unsolved** tasks compete: a task is
solved when one program passes all 15 seeds.

## The three keys, in order

Each unsolved task gets a score list `(split, shares, trace)`. The task with the largest list wins, comparing the
first number, then the second only on a tie, then the third.

| key | name | what it measures | high means |
|---|---|---|---|
| 1 | **split** | mean over seeds of the entropy of "how many of the 3 programs pass this seed" (0 when all agree, 1 bit when it is split) | some programs can do it and others cannot: a repair can see a working and a failing run side by side |
| 2 | **shares** | how many unsolved tasks have the same most common failing stage as this task (crash, perceive, air, slip, released, ...) | one fix may help several tasks at once |
| 3 | **trace** | entropy of the stage plus failure message across the programs | the programs fail in different ways |

If two tasks have exactly the same list, a salt-seeded random pick breaks the tie (the salt is `r<repetition>`).

## Worked example: round 0 of repetition 1 (`library_lex_r1/round_00/selection.json`)

| task | split | shares | trace | picked |
|---|---|---|---|---|
| salad dressing | **0.857** | 1 | 1.54 | **yes** |
| turn on the stove | 0.735 | 5 | 0.91 | |
| bowl between plate and ramekin | 0.245 | 5 | 0.95 | |
| bowl from table center | 0.184 | 1 | 1.36 | |
| bowl next to the ramekin | 0.122 | 5 | 0.29 | |
| bowl on the stove | 0.061 | 2 | 0.76 | |
| open middle drawer, cream cheese, alphabet soup | 0.000 | 5, 5, 2 | 1.01, 0.06, 1.05 | |

Salad dressing wins on key 1 alone: its three programs pass 0, 14 and 0 of 15, so the seeds are split almost everywhere.
The other keys are only consulted when split ties, which here happens only among the three tasks with split 0.

## Why this order

1. **Split first**, because without any passing program there is no contrast to learn from. Cheng Yee's committee analysis
   (`FINDINGS.md`) found split puts a repairable task first in 55 of 56 committees.
2. **Shares second**: among tasks with equal split, prefer the failure that most tasks have, hoping a skill transfers.
3. **Trace last**: prefer the task whose programs fail in the most varied ways.

## Known weaknesses

- **Split counts noise as disagreement.** A program that passes 9 of 15 on random seeds looks like a split by itself; identical
  reruns differ on up to 11 of 15 seeds (`REPORT.md`, noise floor).
- **It can choose a nearly solved task.** Salad dressing at 14 of 15 was chosen first, and its repair had little left to
  teach.
- **Shares only helps once the repairable tasks run out**: usually about 2 tasks per round have any passing program, so keys 2
  and 3 decide the later picks.
- It ranks tasks by what the programs did, not by what the repair will learn; whether that helps is exactly what the
  repetitions are for.

## Comparison arm

`unsolved` uses the same pool of unsolved tasks and picks one uniformly at random (salt-seeded). Everything else is
identical, including the probe and the repair.
