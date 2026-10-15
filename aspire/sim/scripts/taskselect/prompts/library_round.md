Library build with task selection (scripts/taskselect/library_loop.py), arm RULE = {{RULE}}, repetition {{REP}}.
RUN = outputs/library_{{RULE}}_r{{REP}}   L = .venv/bin/python3 scripts/taskselect/library_loop.py
Start:  claude --model claude-sonnet-5 "$(sed -e 's/{{''RULE}}/order/g' -e 's/{{''REP}}/1/g' -e 's/{{''ROUNDS}}/5/g' \
            scripts/taskselect/prompts/library_round.md)"
Every subagent inherits the session model (no model override). Fresh session per run; never --continue across runs.

Round 0 creates the run (same flags in every arm of a comparison):
  $L plan $RUN --round 0 --pool library_9 --library empty      (defaults: --k 3 --repair fixloop --repair-iters 5)
Then for ROUND = 0 .. {{ROUNDS}}-1:

0. $L activate $RUN                         (skip for ROUND 0; plan already activated L00)
1. $L plan $RUN --round $ROUND              (ROUND 0: already done above)
   For every directory it prints, write K programs candidate_A, candidate_B, ... /code.py for that task (K = state.json k),
   each a fresh zero-shot attempt from the CURRENT skill library (.claude/libero/skills/), in the replay format, differing
   in strategy. One short subagent per task: no replay_trial.py, no evaluation, no other outputs read.
   Molmo is not available in this setup: programs must never call point_prompt_molmo (it is absent from the
   namespace and crashes the trial); use SAM3 text/point prompts and geometry instead.
   API facts every program must respect: segment_sam3_point_prompt(rgb, point_coords) takes ONE (x, y) pair of floats,
   never a list of points (CLAUDE.md's "points" wording is wrong); start every program with `import numpy as np`.
2. $L probe $RUN --round $ROUND --gpu 0     (every program on seeds 51-65)
3. $L select $RUN --round $ROUND --rule {{RULE}} --salt r{{REP}}
4. Repair the selected task with ONE subagent on GPU 0, with the paths in round_$ROUND/selection.json "repair"; they
   replace the runbooks' outputs/libero_fix_loop*, outputs/claude_evosearch and outputs/working_codes paths. It never runs
   held-out seeds or Stage 2 and never edits skills. Budget: repair.iterations iterations, stop early at 15/15.
   fixloop:   pass round_$ROUND/repair_prompt.md verbatim as the subagent prompt (rendered by select from
              prompts/repair_fixloop.md). repaired picks the kept program and enforces the iteration budget.
   evosearch: .claude/libero/evosearch/subagent-prompt.md with scripts/tc1/evosearch_prompt.md's settings: K = 8, seeds
              51-65, one iteration = one round (iter_00 ..), RUN_DIR = repair.task_dir/<timestamp>, iter_00 candidate_A =
              repair.start_code verbatim, parents by scripts/tc1/select_parents.py select --rule top3 + verify, final pick
              by select_parents.py best, copied to repair.code (evosearch_best_code.py). Evals detached (setsid nohup).
   Both write repair.task_dir/findings.md (evosearch: its step 7c, although Stage 2 is skipped).
   Then promote, as the fix-loop coordinator does, into the run's ledger (B = repair.build_dir, S/T = suite/task):
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir B begin  --suite S --task T
     edit .claude/libero/skills/ from repair.task_dir/findings.md
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir B finish --suite S --task T  [--reason "no-op: ..."]
5. $L repaired $RUN --round $ROUND         (saves the live library as $RUN/library/L<ROUND+1>)

No evaluation during the build: every library stays in $RUN/library/Lnn and is evaluated later (activate / eval-plan /
eval --after n). $L report $RUN prints the rounds.
Isolation: never read other outputs/library_* runs, outputs/libero_fix_loop*, outputs/claude_evosearch*,
outputs/aspire_*_eval*, outputs/working_codes or docs/logs/; brief every subagent with the same rule. CLAUDE.md's
Self-Evolve and Log Everything rules are suspended: skills change only in step 4, nothing is written to docs/logs/.
