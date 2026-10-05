Run LIBERO Evolutionary Search over a TASK FAMILY (not one task) per .claude/libero/evosearch/INSTRUCTIONS.md, as the
{{ARM}} arm, repetition {{REP}}, of a controlled comparison. The new part is the task of each round: it is chosen by
scripts/taskselect/select_task.py (rule {{ARM}}), never by you or the subagent. Design: scripts/taskselect/DESIGN.md.
Approved deviations below; do not stop on them in preflight.

Fixed inputs
- FAMILY={{FAMILY}} (relation: {{RELATION}}, {{N_TASKS}} tasks, listed in scripts/taskselect/families.json). Round-0 task: {{ROUND0}}.
- RUN_DIR = outputs/family_{{ARM}}_r{{REP}}/{{FAMILY}}     (this arm + repetition only)
- Reference run for iter_00: {{REF_RUN}}
- Candidates are GENERAL programs: each reads env.handle.task_language and dispatches to skills, so one program is
  meaningful on every family task. All candidates of a round are evaluated on the SAME task.

Frozen settings (identical in every arm)
- K = 8 candidates per round (candidate_A..candidate_H). Dev seeds only: {{DEV_SEEDS}}. Rounds iter_00..iter_04, no early stop.
- Parents: after round NN is evaluated run
    .venv/bin/python3 scripts/tc1/select_parents.py select --rule top3 --iter-dir $RUN_DIR/iter_NN
  Every iter_NN+1 candidate derives only from those 3 parents; record $RUN_DIR/iter_NN+1/parents.json as in
  scripts/tc1/evosearch_prompt.md and run `select_parents.py verify --iter-dir $RUN_DIR/iter_NN+1` until it prints OK.
- Pass rates are comparable only inside a round (different tasks); do not compare them across rounds.
- Skill library is FROZEN: nobody edits .claude/libero/skills/. Run `bash scripts/tc1/skills_snapshot.sh check <tag>`
  in preflight and in the final report with tag `family_start`; if that snapshot does not exist yet, create it in preflight with
  `bash scripts/tc1/skills_snapshot.sh save family_start` (it is shared by all arms and repetitions).
- Subagent model: dispatch with NO model override (inherits this session's model, claude-sonnet-5).

Round procedure (exact order)
1. Round 0 (iter_00).
   - Task: .venv/bin/python3 scripts/taskselect/select_task.py select --family {{FAMILY}} --run-dir $RUN_DIR --round 0 --rule {{ARM}}
   - If a reference run is given: .venv/bin/python3 scripts/taskselect/select_task.py init-from --family {{FAMILY}} --run-dir $RUN_DIR --from-run <reference>
     then skip to "parents". Otherwise ONE subagent writes iter_00/candidate_A..H/code.py: 8 diverse general programs.
   - Evaluate: print the command with `select_task.py eval-cmd --run-dir $RUN_DIR --round 0 --gpu 0 --parallel 2`, launch it detached
     (setsid nohup <cmd> > $RUN_DIR/iter_00/eval.log 2>&1 &) and poll for iter_00/iter_summary.json with short Bash calls.
2. Parents of the round just evaluated (select_parents.py select, above). Then the family score of the round:
     .venv/bin/python3 scripts/taskselect/select_task.py score-plan --family {{FAMILY}} --run-dir $RUN_DIR --round N --gpu 0 --parallel 2
   run each printed command (one at a time, detached), then
     .venv/bin/python3 scripts/taskselect/select_task.py score-pick --run-dir $RUN_DIR --round N
   It is a measurement only: nobody reads iter_NN/score/ (not you, not the subagent) and it never changes parents or tasks.
3. Round N >= 1:
   a. ONE subagent writes iter_N/candidate_A..H/code.py from the parents only, learning from the traces of every task
      visited so far (and nothing else). It may replay candidates only on visited tasks. It does not know the next task.
   b. Choose the task of round N for the programs just written:
{{ARM_SECTION}}
   c. .venv/bin/python3 scripts/taskselect/select_task.py verify --family {{FAMILY}} --run-dir $RUN_DIR      (must print OK)
   d. Evaluate with the command from `select_task.py eval-cmd --run-dir $RUN_DIR --round N --gpu 0 --parallel 2`, detached as in step 1.
      If an eval dies, keep its log as iter_NN/eval_crashed_<HHMM>.log, clear that round's partial outputs, re-run the whole round
      with the same task.json (do not re-select), and tell me.
4. After iter_04: Stage 2 below.

Stage 2 (you, after the last round; sequential, detached, resumable)
- Tournament of the last round's 3 parents on every family task:
    .venv/bin/python3 scripts/tc1/select_parents.py select --rule top3 --iter-dir $RUN_DIR/iter_04
    .venv/bin/python3 scripts/taskselect/select_task.py final-plan --family {{FAMILY}} --run-dir $RUN_DIR --gpu 0 --parallel 2
  run each printed command (one at a time, detached), then
    .venv/bin/python3 scripts/taskselect/select_task.py final-pick --run-dir $RUN_DIR
- Held-out of the winner on every family task, seeds {{HELDOUT_SEEDS}}:
    .venv/bin/python3 scripts/taskselect/select_task.py heldout-cmds --family {{FAMILY}} --run-dir $RUN_DIR --gpu 0
  run each printed command (detached, resumable), then
    .venv/bin/python3 scripts/taskselect/select_task.py heldout-summary --family {{FAMILY}} --run-dir $RUN_DIR
  and report per-task and macro-average held-out pass rate.

Isolation (runs must be independent)
- Read-only inputs: scripts/taskselect/descriptors/, families.json, the reference run's iter_00 (only through init-from).
- Never open, list or search any other run: outputs/family_*_r* except your own RUN_DIR, outputs/claude_*, outputs/aspire_*.
  Brief every subagent with the same rule. Do not save memories about this run.
- CLAUDE.md "Self-Evolve"/"Log Everything" are suspended: do not read or write docs/logs/, do not edit CLAUDE.md,
  runbooks or skills. Findings go to $RUN_DIR/findings.md only.
- Forbidden-API rule applies to programs AND to task selection: nothing from simulator internal state, no .bddl/.xml reads.

Hardware (local workstation, one RTX 5090 shared by everything)
- GPU 0 hosts SAM3, GraspNet, PyRoKi (ports 8114-8116, already running), the sim workers and the evals. Never start/stop the servers.
- Use .venv-libero/bin/python3 for sim scripts; MUJOCO_GL=egl and TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 are exported.
  No API key; model access is the logged-in subscription. If a replay hits CUDA OOM, report it, do not retry in a loop.

Agents
- Subagents: at most ONE at a time, only for writing candidates (steps 1 and 3a). No Explore/Plan/helper agents.
  Neither you nor a subagent runs evaluations except through the commands above.

Resuming
- First run `.venv/bin/python3 scripts/taskselect/select_task.py verify --family {{FAMILY}} --run-dir $RUN_DIR`.
  A round with task.json but no iter_summary.json: re-run its eval with the same task.json.
  Finished rounds are never redone and task.json is never regenerated (select refuses to overwrite).

Final report
- Per round: task, rule, best/mean dev pass rate, committee disagreement (summarize with
  .venv/bin/python3 scripts/taskselect/summarize_family_run.py $RUN_DIR), parents. Tournament winner and macro pass rate,
  held-out per task and macro-average, skills check result, anything that deviated.

Never push to any remote. Start with the preflight report and wait for my go.
