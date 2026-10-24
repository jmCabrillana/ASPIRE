Write fresh zero-shot programs for the held-out tasks of a skill-library build, from ONE fixed skill library
(scripts/taskselect/scale/measure.py). Start immediately; there is no preflight gate. Working directory: aspire/sim.

LIBRARY = {{LIB}}   (read every .md file in this directory when briefing yourself; it is the skill library to write from. Do NOT
read .claude/libero/skills/, it holds a different library. If the directory is empty the library is empty on purpose.)
OUT = {{DIR}}
K = {{K}} programs per task, written by {{GROUPS}} independent subagents of 3 programs each.

For each task directory under OUT/programs/ (one per task; the name is <suite>__<task>) and each group g = 1 .. {{GROUPS}},
the programs of group g are <dir>/candidate_NN/code.py with NN = 3g-2, 3g-1, 3g (two digits: 01, 02, 03, 04, ...).
First list which code.py files already exist and are non-empty; dispatch a subagent only for a (task, group) with a missing file
(this run may have been restarted). Dispatch at most 4 subagents at a time. Brief each subagent with:
- The task: suite and task from the directory name; the runtime instruction is env.handle.task_language. In some suites objects
  move per seed: localise objects from perception every run, never hard-code coordinates.
- Read CLAUDE.md, .claude/libero/CLAUDE.md, .claude/libero/api-reference.md and the LIBRARY files above. Allowed APIs only; nothing
  from simulator internal state; never read .bddl/.xml/.urdf files.
- Each program is one self-contained file in the replay format of scripts/libero/replay_trial.py --args.replay-code, written for
  THIS task, using the library's rules where they apply. Its 3 programs must differ in strategy (perception prompts and fallbacks,
  grasp method and height, approach path, placement height, checks and retries), not only in wording. It does not know about the
  other groups and must not look at their files.
- Zero-shot: no replay_trial.py, no evosearch_eval.py, no simulator, no evaluation of any kind. Do not read anything under outputs/
  except the LIBRARY directory above, and nothing under docs/logs/ or scripts/tc1/.
- Do not edit .claude/, CLAUDE.md, docs/ or any script; no memories; no git commands that change anything; never push.
- Molmo is not available: never call point_prompt_molmo (it is absent from the namespace and crashes the trial); use SAM3 text/point
  prompts and geometry instead.
- API facts: segment_sam3_point_prompt(rgb, point_coords) takes ONE (x, y) pair of floats, never a list of points (CLAUDE.md's
  "points" wording is wrong); start every program with `import numpy as np`.

If a subagent returns without its files, dispatch it once more; if it fails again, leave the files missing (the driver reports them).
When every code.py exists and is non-empty, write OUT/WRITER_DONE (one line per task: "<dir>: <n> programs"), then stop. Do not run anything else.
