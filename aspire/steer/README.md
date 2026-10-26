# Code-as-Steering

A coding agent builds a library of programs that steer a **frozen** vision-language-action model (SmolVLA) at inference time,
using what it learns in simulation (LIBERO / LIBERO-PRO). The VLA weights never change; all adaptation lives in the library.

This folder is separate from `aspire/sim` (code-as-policies). It reuses the repo's LIBERO-PRO install and the Claude agent setup.

## Question

Can a coding agent discover, verify and compose programs that change a frozen VLA's internal behaviour (activation steering) so that
it solves tasks it fails on, and do these programs transfer to new tasks?

## Library entry

Each entry is executable: an applicability condition `p(s)`, a steering program `f(s)` (which direction, layer and strength to add
to the VLA's hidden states, as a function of the observation), and a check of whether it helped. The agent discovers entries by
controlled simulator interventions and composes them for new tasks.

## Milestones (results first)

1. **Baseline.** Frozen `lerobot/smolvla_libero` on LIBERO (trained suites) and LIBERO-PRO (perturbed suites): success per task.
   The perturbed suites are where steering can help.
2. **Steering works at all.** Hooks on SmolVLA's VLM and action-expert layers; steering vectors from contrasts (successful vs failed
   rollouts, or instruction pairs); a fixed sweep of layer x strength on failing tasks. Does any intervention raise success?
3. **State-dependent steering.** The same directions applied only in some phases (e.g. before or after the grasp).
4. **Agent-written steering programs.** The coding agent gets the steering API and the simulator, writes and tests programs, and keeps
   the verified ones in a library; reuse on new tasks.

## Layout

| path | what |
|---|---|
| `setup_env.sh` | creates `.venv` (lerobot + SmolVLA + LIBERO-PRO from `../sim/cap/third_party`) |
| `steer/` | Python package: rollout loop, hooks, steering API |
| `scripts/` | experiments (baseline, sweeps) |
| `outputs/` | results (gitignored) |

Safety: generated code runs with full Python access, as in `aspire/sim`; same boundary (see the top-level `CLAUDE.md`).
