# Code-as-Steering

A coding agent builds a library of programs that steer a **frozen** vision-language-action model (SmolVLA) at inference time,
using what it learns in simulation (LIBERO / LIBERO-PRO). The VLA weights never change; all adaptation lives in the library.
The design document is `doc/code_as_steering.pdf` (source `doc/code_as_steering.tex`).

This folder is separate from `aspire/sim` (code-as-policies). It reuses the repo's LIBERO-PRO install and the Claude agent setup.

## Question

Can a coding agent dissect a frozen VLA, discover interventions that make it solve tasks it fails (activation steering, neuron
pushes, attention edits, instruction rewrites, action edits, small adapters, and combinations of them), verify them, keep them as
reusable code, and reuse them on new tasks, getting close to what one-shot fine-tuning would give?

## How it works

1. **Frozen policy in the loop.** `lerobot/smolvla_libero` (450M; a 16-layer VLM prefix encoder, width 960, and a 16-layer
   flow-matching action expert, width 720, that denoises 50-action chunks in 10 steps). Rollouts copy lerobot's LIBERO evaluation.
2. **Open programs.** A program may define `setup(ctx)`, `program(t, obs, internals, ctx)` (steering terms before each action
   chunk), `on_step(t, obs, action, ctx)` (replace actions) and `on_end(result, ctx)`. `ctx` exposes the whole model, hooks and
   patches (undone after the episode), the instruction, per-episode memory and free logging. Two hard rules: no simulator ground
   truth, and frozen weights identical after every episode (checked).
3. **Introspection.** Per chunk: attention shares of the action tokens on agentview / wrist / instruction / state / action tokens,
   MLP norms, projections on named directions, plus anything the program logs.
4. **Fast experiments.** The offline lab predicts action chunks on saved observations in about 0.5 s (fixed noise, so differences
   are causal), captures and swaps activations, attributes action features to single MLP neurons and pushes neurons with the dose
   recipe used for LLM neuron steering; LoRA adapters train in seconds and detach exactly.
5. **Library.** `library/steerlib/` is real Python (interventions, gates, analysis, instruction tools), indexed with evidence;
   directions, adapters, programs, notes and the agent's lab notebook live beside it.
6. **Agent and verification.** The research agent (`prompts/explorer.md`) works unattended for hours: diagnoses, runs causal
   experiments, writes programs, screens them on 5-10 development episodes (initial states 0-19) and requests verification.
   The driver verifies on held-out initial states 20-49 against the frozen VLA (one-sided Fisher test) and records the result in
   `library/INDEX.md`. A shorter ASPIRE-style round (`prompts/agent.md`: 3 candidates, repairs from traces, library update) also exists.

## Status (2026-10-26)

- Frozen baseline (10 episodes per task): `libero_goal` 0.84 mean; `libero_goal_lan` (paraphrased instructions) 0.0-0.8 per
  task; every `_swap` task tried so far (object positions swapped) 0.0. Table in the design document.
- A first fixed steering sweep (trained-minus-paraphrased mean MLP output, VLM and expert layers) had no effect; the literature
  review explains why (direction averaged over identical image tokens, strength tiny relative to per-token norms).
- The research agent is running (`scripts/explore.py`, 8 hours). Its first finding: the fine-tuned 16-layer VLM has lost general
  English, so paraphrase verbs it never saw land near unrelated nouns; it built `steerlib/canon.py`, which rewrites an instruction
  into the policy's training vocabulary (target task's twin excluded) with the base SmolVLM2 as a meaning check. Not verified yet.

## Layout

| path | what |
|---|---|
| `setup_env.sh` | creates `.venv`: a copy of `../sim/.venv-libero` (torch cu128, LIBERO-PRO, robosuite) plus `lerobot[smolvla]` and `peft` |
| `steer/rollout.py` | episode loop identical to lerobot's LIBERO eval; programs, introspection, traces and saved observations plug in here |
| `steer/hooks.py` | additive steering on MLP outputs (VLM and action-expert layers) and activation recording |
| `steer/introspect.py` | per-chunk internals: attention shares per segment, MLP norms, projections |
| `steer/directions.py`, `steer/library.py` | contrast directions; the program API (`Context`), direction files, program loading, weight check |
| `steer/lab.py` | offline lab: chunk prediction, capture, edit and swap activations; `Neurons`: attribution patching, dosed pushes |
| `steer/interp.py` | classic interpretability: activation bank over saved episodes, neuron selectivity (AUROC / correlation), max-activating examples, linear probes (grouped by episode), word-selective neurons |
| `steer/lora.py` | LoRA adapters: attach, train on saved episodes, save, load, detach |
| `scripts/baseline.py` | frozen success rate per task |
| `scripts/record.py`, `make_direction.py`, `sweep.py`, `pilot.py` | record activations, build directions, fixed layer x strength sweeps |
| `scripts/run_program.py` | evaluate a program (or `none`) on development episodes; `--trace`, `--save-obs` |
| `scripts/verify.py` | held-out verification (episodes 20-49), Fisher test, promotion into the library |
| `scripts/explore.py` | runs the research agent unattended and verifies what it submits |
| `scripts/agent_round.py` | one ASPIRE-style round on one task |
| `scripts/tables.py` | LaTeX tables for the design document |
| `prompts/explorer.md`, `prompts/agent.md` | agent instructions |
| `library/` | `steerlib/` (code + README index), `directions/`, `adapters/`, `programs/`, `tools/`, `notes/` (JOURNAL.md), `INDEX.md`, `LITERATURE.md`, `LITERATURE_REVIEW.md` |
| `doc/` | design document |
| `outputs/` | results (gitignored) |

## Quick start

```bash
./setup_env.sh
.venv/bin/python scripts/baseline.py --suites libero_goal libero_goal_lan --episodes 10
.venv/bin/python scripts/run_program.py --program none --suite libero_goal_lan --task 7 --episodes 0-9 --trace --save-obs
.venv/bin/python scripts/explore.py --hours 8          # research agent + verification
.venv/bin/python scripts/tables.py && (cd doc && ~/miniconda3/envs/tex/bin/tectonic code_as_steering.tex)
```

Safety: generated code runs with full Python access, as in `aspire/sim`; same boundary (see the top-level `CLAUDE.md`).
