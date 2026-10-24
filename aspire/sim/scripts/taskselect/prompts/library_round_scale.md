Library build with random task choice among probed unsolved tasks (scripts/taskselect/library_loop.py), repetition {{REP}}.
RUN = outputs/scale_{{ARM}}_r{{REP}}   L = .venv/bin/python3 scripts/taskselect/library_loop.py
Every subagent inherits the session model (no model override). Fresh session per run; never --continue across runs.

The run already exists: `plan --round 0` was run for you (pool libero_90: 70 build tasks, 20 held-out tasks that you never touch; the
candidate tasks of round 0 are sampled) and the live skill library is L00 (empty). Do NOT run plan --round 0.
Then for ROUND = 0 .. {{ROUNDS}}-1:

0. $L activate $RUN                         (skip for ROUND 0)
1. $L plan $RUN --round $ROUND --candidates {{CANDIDATES}} --salt r{{REP}}      (skip for ROUND 0: already done)
   It samples {{CANDIDATES}} unsolved, not yet repaired tasks. For every directory it prints, write 3 programs candidate_A,
   candidate_B, candidate_C /code.py for that task (K = state.json k), each a fresh zero-shot attempt from the CURRENT skill
   library (.claude/libero/skills/), in the replay format, differing in strategy. One short subagent per task directory (at most 4 at
   a time): no replay_trial.py, no evaluation, no other outputs read.
   Molmo is not available in this setup: programs must never call point_prompt_molmo (it is absent from the namespace and crashes
   the trial); use SAM3 text/point prompts and geometry instead.
   API facts every program must respect: segment_sam3_point_prompt(rgb, point_coords) takes ONE (x, y) pair of floats, never a list
   of points (CLAUDE.md's "points" wording is wrong); start every program with `import numpy as np`.
2. $L probe $RUN --round $ROUND --gpu 0 --seeds 66 67 68 69 70
3. $L select $RUN --round $ROUND --rule unsolved --salt r{{REP}}
4. Repair the selected task with ONE subagent on GPU 0, with the paths in round_$ROUND/selection.json "repair". It never runs
   held-out seeds and never edits skills. Budget: repair.iterations iterations, stop early at 15/15. Pass round_$ROUND/repair_prompt.md
   verbatim as the subagent prompt (rendered by select from prompts/repair_fixloop.md). repaired picks the kept program and enforces
   the iteration budget. It writes repair.task_dir/findings.md.
   Then promote, as the fix-loop coordinator does, into the run's ledger (B = repair.build_dir, S/T = suite/task):
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir B begin  --suite S --task T
     edit .claude/libero/skills/ from repair.task_dir/findings.md
     .venv/bin/python3 scripts/libero/record_skill_promotion.py --build-dir B finish --suite S --task T  [--reason "no-op: ..."]
5. $L repaired $RUN --round $ROUND         (saves the live library as $RUN/library/L<ROUND+1>)

After ROUND {{ROUNDS}}-1's `repaired` there is NO final probe and nothing else to do: print `$L report $RUN` and stop. The held-out
measurements are made by a separate driver from the saved libraries.
Isolation: never read other outputs/ runs, outputs/libero_fix_loop*, outputs/claude_evosearch*, outputs/aspire_*_eval*,
outputs/working_codes or docs/logs/; brief every subagent with the same rule. CLAUDE.md's Self-Evolve and Log Everything rules are
suspended: skills change only in step 4, nothing is written to docs/logs/.

WAITING RULES for every subagent (add these lines after the verbatim repair prompt in your dispatch message): never end your turn
while an evaluation is running. Wait inside one tool call:
  until [ -f <iter_dir>/iter_summary.json ]; do sleep 10; done      (Bash timeout 600000; repeat the call until it returns)
A crashed evaluation may be re-run once as the repair prompt says. If a subagent returns without a kept program or without
findings.md, do NOT dispatch a second one: record it with `repaired` if the iterations allow, otherwise stop and report.

Never push. Do not change .claude/ other than skills in step 4, and never edit scripts/taskselect/.
