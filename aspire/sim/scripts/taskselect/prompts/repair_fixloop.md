Capped fix loop for ONE task of a library build. Working directory: aspire/sim. GPU: {{GPU}}.

SUITE = {{SUITE}}   TASK = {{TASK}}   seeds = 51-65   budget = {{ITERS}} iterations
TASK_DIR  = {{TASK_DIR}}
PROBE_DIR = {{PROBE_DIR}}   (this round's zero-shot programs for the task, already run on seeds 51-65)

Read first, from .claude/libero/fix-loop/subagent-prompt.md: "Context" (forbidden and allowed APIs, venv, perception
servers), "Stage 1 / Step 2 — Diagnose each failed seed" (how to read trace.json and keyframes) and "Step 5 — Write
findings.md". Read the skill library .claude/libero/skills/ (it may be empty). Everything else in that file (Stage 0,
the per-seed replay limit, BLOCKED files, /tmp paths, outputs/working_codes, GPU numbers, Stage 2) does NOT apply here.

Starting point: the programs in PROBE_DIR/candidate_*/code.py and their trials in PROBE_DIR/candidate_*/eval/trial_*
(trace.json, keyframes; the outcome is in the directory name). Compare passing and failing trials across programs
before you write anything: where one program passes a seed that another fails, the difference localises the bug.

One iteration (at most {{ITERS}}; stop early when an iteration passes 15/15):
  1. Write ONE task-level program to TASK_DIR/iter_NN/candidate_A/code.py (NN = 01, 02, ...), from the best evidence so
     far. No seed-specific branches, no hardcoded scene coordinates.
  2. Run it on all development seeds, and wait for TASK_DIR/iter_NN/iter_summary.json:
       MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 .venv-libero/bin/python3 scripts/libero/evosearch_eval.py \
         --iter-dir TASK_DIR/iter_NN --suite {{SUITE}} --task {{TASK}} --trials 15 \
         --trial-seeds 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 \
         --sim-gpus {{GPU}} --parallel-per-gpu 2 --no-highlights > TASK_DIR/iter_NN/eval.log 2>&1
  3. Diagnose its failed seeds from TASK_DIR/iter_NN/candidate_A/eval/trial_* only.
No other simulator runs: no replay_trial.py, no REPL, no extra seeds. A crashed evaluation may be re-run once with the
same command (delete iter_NN/candidate_A/eval first); it does not count as an iteration.

The program kept for the task is chosen by library_loop.py repaired: the iteration with the most passes (ties: the later
one). Do not copy or rename programs.

Finally write TASK_DIR/findings.md (Step 5 format): what failed, what fixed it, the pass count of every iteration, and a
"Generalizable Patterns" section the coordinator can promote into the skill library. Do not edit .claude/libero/skills/.
Never read other outputs/ directories than TASK_DIR and PROBE_DIR, and do not write to docs/logs/.
Return: pass count per iteration and the path of findings.md.
