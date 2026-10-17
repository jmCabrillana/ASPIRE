Write a second, independent set of zero-shot programs for the FINAL skill library of a finished library build
(scripts/taskselect/remeasure.py). Start immediately; there is no preflight gate. Working directory: aspire/sim.

LIBRARY = {{LIB}}   (read every .md file in it; this is the skill library to write from. Do NOT read .claude/libero/skills/,
it holds a different library.)
OUT = {{DIR}}

For each of the 9 directories under OUT/programs/ (one per task; the name is <suite>__<task>), write 3 programs:
  <dir>/candidate_A/code.py, <dir>/candidate_B/code.py, <dir>/candidate_C/code.py
Use ONE subagent per task directory (9 subagents, at most 3 running at a time). Brief each subagent with:
- The task: suite and task from the directory name. In *_swap suites object positions change per seed: localise objects
  from perception every run; never hard-code coordinates.
- Read CLAUDE.md, .claude/libero/CLAUDE.md, .claude/libero/api-reference.md and the LIBRARY files above. Allowed APIs only;
  nothing from simulator internal state; never read .bddl/.xml/.urdf files. The runtime instruction is env.handle.task_language.
- Each program is one self-contained file in the replay format of scripts/libero/replay_trial.py --args.replay-code,
  written for THIS task, using the library's rules where they apply. The 3 programs must differ in strategy (perception
  prompts and fallbacks, grasp method and height, approach path, placement height, checks and retries), not only in wording.
- Zero-shot: no replay_trial.py, no evosearch_eval.py, no simulator, no evaluation of any kind. Do not read anything under
  outputs/ except the LIBRARY directory above, and nothing under docs/logs/ or scripts/tc1/.
- Do not edit .claude/, CLAUDE.md, docs/ or any script; no memories; no git commands that change anything; never push.
- Molmo is not available in this setup: never call point_prompt_molmo (it is absent from the namespace and crashes the
  trial); use SAM3 text/point prompts and geometry instead.
- API facts: segment_sam3_point_prompt(rgb, point_coords) takes ONE (x, y) pair of floats, never a list of points
  (CLAUDE.md's "points" wording is wrong); start every program with `import numpy as np`.

When all 27 files exist and are non-empty, write OUT/WRITER_DONE containing one line per program
("<task dir>/candidate_X: <strategy in a few words>"), then stop. Do not run anything else.
