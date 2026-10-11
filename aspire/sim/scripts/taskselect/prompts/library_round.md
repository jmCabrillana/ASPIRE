One round of the library build with disagreement-based task selection (scripts/taskselect/library_loop.py).
RUN = outputs/library_{{RULE}}_r{{REP}}   POOL = {{POOL}}   ROUND = {{ROUND}}   RULE = {{RULE}}

0. Library: .venv/bin/python3 scripts/taskselect/library_loop.py activate $RUN   (skip for ROUND 0: plan creates RUN and activates L_0)
   Every arm keeps its own library under $RUN/library/. Never edit .claude/libero/skills/ outside step 4.
1. Plan:   .venv/bin/python3 scripts/taskselect/library_loop.py plan $RUN --round $ROUND {{POOL_FLAG}}
   For every directory it prints, write {{K}} programs candidate_A.. candidate_{{K_LAST}}/code.py for that task, each a
   fresh attempt from the CURRENT skill library (.claude/libero/skills/), in the replay format, differing in strategy.
   One short subagent per task, no debugging replays: these are zero-shot programs.
2. Probe:  .venv/bin/python3 scripts/taskselect/library_loop.py probe $RUN --round $ROUND --gpu 0
3. Select: .venv/bin/python3 scripts/taskselect/library_loop.py select $RUN --round $ROUND --rule {{RULE}} --salt r{{REP}}
4. Repair the selected task with the run's repair method (selection.json: repair.method; same for every arm).
   Use the paths that select printed (also in $RUN/round_$ROUND/selection.json: repair); they replace the runbooks'
   outputs/libero_fix_loop, outputs/claude_evosearch and outputs/libero_fix_loop_debug paths. ONE subagent, GPU 0.
   It skips every outputs/working_codes copy, never runs held-out seeds or Stage 2, and never edits skills.
   fixloop:   .claude/libero/fix-loop/subagent-prompt.md for this one task (as scripts/tc1/fixloop_prompt.md),
              debug seeds 51-65, TASK_DIR = repair.task_dir, DEBUG_DIR = repair.debug_dir. Output: repair.code (fix_code.py).
   evosearch: .claude/libero/evosearch/subagent-prompt.md with the frozen settings of scripts/tc1/evosearch_prompt.md:
              K = 8, dev seeds 51-65, stop at best >= 80% or after 5 rounds (iter_00..iter_04),
              RUN_DIR = repair.task_dir/<timestamp>; iter_00 candidate_A = repair.start_code verbatim (write a fresh
              candidate_A when it is null); parents by scripts/tc1/select_parents.py select --rule top3 and verify;
              final pick by select_parents.py best --run-dir RUN_DIR, copied to repair.code (evosearch_best_code.py).
              Launch every evosearch_eval.py detached (setsid nohup) and poll for iter_summary.json.
   Both write repair.task_dir/findings.md. Then promote skills as the coordinator does, with the run's ledger:
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir <repair.build_dir> begin  --suite S --task T
     (edit .claude/libero/skills/ from repair.task_dir/findings.md)
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir <repair.build_dir> finish --suite S --task T
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir <repair.build_dir> verify --suite S --task T
   Nothing else edits skills.
5. Record: .venv/bin/python3 scripts/taskselect/library_loop.py repaired $RUN --round $ROUND
   (reads repair.code, checks it on seeds 71 72, and saves the promoted library as the next L).
6. Checkpoint (when ROUND+1 is in the run's checkpoints, and once after ROUND 0's plan for n = 0):
   .venv/bin/python3 scripts/taskselect/library_loop.py eval-plan $RUN
   write {{K}} zero-shot programs per printed directory exactly as in step 1 (these are the held-out tasks B), then
   .venv/bin/python3 scripts/taskselect/library_loop.py eval $RUN --gpu 0
7. Next round: ROUND+1 from step 0.

Report after the last round: .venv/bin/python3 scripts/taskselect/library_loop.py report $RUN
Isolation: never read other outputs/library_* runs, outputs/libero_fix_loop*, outputs/claude_evosearch*,
outputs/aspire_*_eval*, outputs/working_codes or docs/logs/;
brief every subagent with the same rule. CLAUDE.md's Self-Evolve and Log Everything rules are suspended: skills change only
in step 4, and nothing is written to docs/logs/. Every arm runs steps 0-2 and 4-6 identically.
