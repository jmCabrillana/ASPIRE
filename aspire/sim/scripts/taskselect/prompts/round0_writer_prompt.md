Write the 8 round-0 candidate programs for the task family {{FAMILY}} (scripts/taskselect/DESIGN.md). They are used to
measure how much programs disagree across the family's tasks. Start immediately; there is no preflight gate.

Output
- RUN = outputs/family_matrix/{{FAMILY}}
- $RUN/iter_00/candidate_A/code.py ... candidate_H/code.py  (8 files), then $RUN/notes.md, then an empty file $RUN/DONE.
- Tasks: run `.venv/bin/python3 scripts/taskselect/select_task.py context --family {{FAMILY}} --run-dir $RUN --round 0 > /dev/null`
  and read $RUN/iter_00/context.json: every task's runtime instruction ("language") and reset-frame image ("scene_image";
  look at the images).

What a candidate is
- ONE self-contained Python file in the replay format of scripts/libero/replay_trial.py --args.replay-code.
- GENERAL: it reads env.handle.task_language at run time and handles EVERY task of the family (dispatch on the instruction).
  It must make a real attempt on each task and never crash on an instruction it does not expect.
- The 8 must differ in strategy, not wording: e.g. perception prompts and fallbacks, top-down grasp vs plan_grasp, grasp
  height and width checks, how drawers / knobs / pushes are done, placement heights, retries. notes.md: one line per candidate
  saying what its strategy is and how it differs from the others.

Rules
- Read first: CLAUDE.md, .claude/libero/CLAUDE.md, .claude/libero/api-reference.md, .claude/libero/skills/. Allowed APIs only;
  nothing from simulator internal state; never read .bddl/.xml/.urdf files.
- Debug replays: only with scripts/libero/replay_trial.py, only trials (seeds) 61-65, GPU 0, at most 40 replays in total,
  --args.output-dir $RUN/debug. Never run seeds 1-60. Count your replays in notes.md.
- Do not edit .claude/, CLAUDE.md, docs/ or any script. Do not read other runs under outputs/ (only $RUN). No memories, no other
  agents, no git commands that change anything, never push.
- SAM3, GraspNet and PyRoKi already run on ports 8114-8116: never start or stop them. Use .venv-libero/bin/python3 with
  MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=0.
- Stop after about 90 minutes of work even if some candidates are weak: write what you have, notes.md and DONE.
