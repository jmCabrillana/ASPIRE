# Task selection: results and plan

Date: 2026-10-05. Machine: one RTX 5090. Model: `claude-sonnet-5`. Tools: [DESIGN.md](DESIGN.md).

## Summary

- We cannot predict from the code which programs pass a task.
- We can measure it at low cost. One seed for each task is sufficient.
- Our goal is now to get good programs in fewer rounds. We measure this with regret.
- The idea fits best in the LIBERO-90 library build. There, each round is expensive and the task order is free.

## Words

| word | meaning |
|---|---|
| trial | One program does one task with one seed. A trial takes 6 to 10 seconds. |
| round | We write 8 programs, test them on one task, and keep the 3 best. |
| split | On one seed, some programs pass and some programs fail. |
| split score | A number for how much a task splits the programs (`D_seed` in DESIGN.md). We add it over the rounds. |
| probe | We run all 8 programs on one seed of each task. Then we select a task. |
| family score | The pass rate of the best program on all tasks of the family. |
| regret | The target score minus the family score, added over all rounds. Low regret means fewer rounds. |

## Results

All results are from one family (`goal_swap`, 10 tasks) and 8 programs that we did not tune.

| question | result |
|---|---|
| Do the programs get different results? | Yes, on 5 tasks. On the other 5 tasks, all programs fail. 28 of 400 trials pass. |
| Which programs pass? | Two programs have 24 of the 28 passes. They pass on different tasks. |
| Is a result stable? | Yes. The same program with the same seed gives the same result. |
| Can the model predict pass or fail from code and images? | No. The ranking score (AUC) is 0.44 and 0.52. A score of 0.5 is chance. |
| Can it predict after it sees the results of one task? | Sometimes. AUC is 0.82 in one run and 0.62 in the other run. |
| Does a probe find the tasks with a split? | Yes. It selects the best task for each of the 5 seeds that we tried. |
| Is the probe better than a random task? | Yes for the split score: 1.87 to 2.10 against 0.88 to 1.02 in 5 rounds. |
| Is it better when random gets the same number of trials? | Yes. Random then gets 1.31 to 1.51. |
| Is "most splits" different from "pass rate near 50%"? | No. The two rules select the same tasks here. |
| Did the fix loop solve the task "bowl on stove"? | No. 0 of 50 seeds pass. The final code is the same as the first code. |

## Results with traces

Pass or fail is one bit. A trace shows more: how far the program got. We reduce each trial to a stage:
no motion, no grasp attempt, closed on air, object lost, object held, object released, pass.
The trace score of a task is high when the 8 programs get to different stages.

| question | result |
|---|---|
| Does the trace score separate the 5 tasks where all programs fail? | Yes. Wine bottle on rack 1.11, cream cheese 0.65, top drawer 0.43, turn on stove 0.33, push plate 0.00. |
| Does it agree with the split score on the other tasks? | Yes. The rank correlation of the two scores is 0.92. |
| Is one seed sufficient? | Yes. The rank correlation between one seed and the other four seeds is 0.73 to 0.90. |
| What changes in the selection? | After the tasks with a split, the split probe selects a random task. The trace probe selects wine bottle on rack. |
| Is the trace probe better than random? | For the trace score in 5 rounds: 5.69 against 3.74. With the split score, the two probes are equal (1.87). |

- On wine bottle on rack, 5 programs hold and release the bottle and 3 programs close on air. No program passes.
- The stages are for grasp tasks. For a push task, a closed hand is correct, but the stage says "closed on air".
- We did not show that a high trace score makes the next programs better.

## What the results show

- Prediction from code does not work for the first round. Measurement works.
- With a probe, selection is simple. It is almost the same as a random task plus a quick check.
- We did not test a third method: test all tasks on 1 or 2 seeds in each round, with no selection.
- We did not show that a task with a split makes the next programs better. Only online runs can show this.
- Selection is useful only when three conditions are true:
  1. We can select the task.
  2. We cannot test all tasks.
  3. The task that we select changes what the programs learn.
- In one search on 10 tasks, condition 2 is false. On a real robot, condition 1 is false.

## New goal: fewer rounds

- We compare methods by regret, not by the split score.
- After each round, we measure the family score: the 3 parents on all tasks, on seeds 71 to 73 (`score-plan`, `score-pick`).
  For 10 tasks this is 90 trials.
- We report the regret and the number of rounds to get to a target score.
- A method can select the same task again (`--allow-revisit`, for the random rule and the probe rule).
- `summarize_family_run.py` prints the regret and the round that gets to the target.
- We do not know the best possible score. We compare each method with the best method that we ran.

## Where the idea fits

The LIBERO-90 library build (`.claude/libero/zeroshot-transfer/`) does this:

1. It takes the 90 tasks in a random order. The order is fixed in `ordering.txt`.
2. It runs one build loop for each task. One loop can take 90 minutes.
3. It saves the skill library after each 5 tasks.
4. It tests each saved library on the long LIBERO tasks, with no repair.

The three conditions are true here. The order is free. A build loop is expensive. The order changes the library.

**Claim to test:** a task order from a probe gives less regret than the random order.

## Plan

| step | what we do | what it tells us |
|---|---|---|
| 0 | Probe all 90 tasks with the current library. Use a small number of general programs and one seed. | If there is a signal at the start. |
| 1 | Run 10 build loops on 30 tasks. Compare the random order with the probe order. | If the order changes the library. |
| 2 | Compare the selection rules: most splits, largest gap between best and mean, all-fail first. | Which rule gives the lowest regret. |
| 3 | Test the saved libraries on the long tasks. Measure the regret for each order. | If the claim is true. |

Step 0 costs one session to write the programs and about one hour of trials. This is an estimate.
Step 1 costs about 10 to 20 hours for each order. This is an estimate.

## Risks

- The build makes one program for each task. A probe needs programs that can try all tasks. We did not try this at scale.
- The best tasks to build can be the tasks where all programs fail. Then "most splits" is the wrong rule.
- Step 0 can show that almost all tasks fail. Then selection helps only later in the build.
- One GPU cannot run the full build of 90 tasks for each order.
