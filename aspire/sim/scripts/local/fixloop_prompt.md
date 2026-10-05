Run the LIBERO fix loop per .claude/libero/fix-loop/QUICKSTART.md with these approved deviations
(do not stop on them in preflight):

Scope
- ONE task only: SUITE=libero_goal_swap TASK=put_the_bowl_on_the_stove.
  gen_progress.py will list other pending tasks — ignore them, never dispatch or evaluate them.

Hardware (local workstation, one RTX 5090 shared by everything -- no SLURM)
- The reference topology in QUICKSTART.md (8 GPUs) does not apply; this single approved deviation
  replaces it. GPU 0 = SAM3, GraspNet, PyRoKi (ports 8114-8116, already running, CUDA 12.8 /
  torch cu128), the Stage 1 worker and the Stage 2 eval. Never start/stop the servers.
- Not a Docker/SLURM host: /tmp is not wiped, but keep scratch code and replay outputs under outputs/
  (e.g. $TASK_DIR/attempts/, outputs/libero_fix_loop_debug/) so everything stays with the run.
- Use .venv-libero/bin/python3 for sim/perception scripts; env vars MUJOCO_GL=egl and
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 are already exported. No API key is set; model access is the
  logged-in subscription.
- Another job on this machine may share the GPU memory; if a replay fails with CUDA OOM, report it
  instead of retrying in a loop.

Subagents
- Exactly ONE Stage 1 subagent in total, for this task, on GPU 0. Fill the subagent-prompt.md
  template with GPU: 0 and add the /tmp rule above to it.
- Do not spawn any other agents (no Explore/Plan/helper agents) — read the runbooks yourself.
- If the subagent returns without fix_code.py or findings.md, do NOT re-dispatch; tell me and wait.

Resuming (an earlier session may have ended mid-task)
- Regenerate progress first and check outputs/libero_fix_loop/libero_goal_swap/put_the_bowl_on_the_stove/.
- Stage 1 files but no fix_code.py: resume in ONE new subagent that reads those files first.
- fix_code.py present: skip Stage 1; do skill promotion if `record_skill_promotion.py verify` fails,
  then Stage 2 with --resume (finished seeds are skipped).
- Never redo a finished stage.

Then, as in main-agent-prompt.md: skill promotion (begin → edit skills → finish → verify),
Stage 2 on seeds 1-50 with --resume in the background on GPU 0, regenerate progress, and report
per the Completion Report — for this one task.

Never push to any remote. Start with the preflight report and wait for my go.
