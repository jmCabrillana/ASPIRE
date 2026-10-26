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
| `setup_env.sh` | creates `.venv`: a copy of `../sim/.venv-libero` (torch cu128, LIBERO-PRO, robosuite) plus `lerobot[smolvla]` |
| `steer/rollout.py` | episode loop identical to lerobot's LIBERO eval; steering programs and traces plug in here |
| `steer/hooks.py` | activation steering (MLP-output hooks, VLM and action-expert layers) and recording |
| `steer/introspect.py` | per-chunk internals: attention shares of the action tokens per segment, MLP norms, projections |
| `steer/directions.py`, `steer/library.py` | contrast directions; the library (directions, programs, context API) |
| `scripts/baseline.py` | frozen success rate per task |
| `scripts/record.py`, `scripts/make_direction.py`, `scripts/sweep.py`, `scripts/pilot.py` | record activations, build directions, fixed layer x strength sweeps |
| `scripts/run_program.py`, `scripts/verify.py`, `scripts/agent_round.py` | evaluate a program on development states (0-19), verify on held-out states (20-49), one agent round |
| `prompts/agent.md` | instructions of the steering agent |
| `library/` | directions, programs (candidates and promoted), notes, `INDEX.md` (verified results) |
| `doc/code_as_steering.tex` | idea, implementation and plan (`scripts/tables.py` then `tectonic`) |
| `outputs/` | results (gitignored) |

Quick start:

```bash
./setup_env.sh
.venv/bin/python scripts/baseline.py --suites libero_goal libero_goal_lan --episodes 10
.venv/bin/python scripts/pilot.py --trained libero_goal --perturbed libero_goal_lan
.venv/bin/python scripts/agent_round.py --suite libero_goal_lan --task 0 --budget 12
```

Safety: generated code runs with full Python access, as in `aspire/sim`; same boundary (see the top-level `CLAUDE.md`).
