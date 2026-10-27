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

## Status (2026-10-27)

First research session (agent, 8 h, one GPU): **21 tasks promoted** on held-out initial states (20-49, 30 episodes, Fisher test;
`library/INDEX.md`, table and details in the design document):
- **Paraphrased instructions, 15 tasks:** `goal_lan` 0, 6, 7, 8, 9 and `spatial_lan` 1-9 (e.g. goal_lan 7: 1/30 -> 29/30;
  spatial_lan 1-9 mostly 0/30 -> 19-27/30).
- **Swapped positions, 6 tasks** (frozen 0/30 everywhere): `spatial_swap` 0, 2, 4, 5, 7, 9 (8-20/30). `goal_swap` still unsolved.
- **In-network steering, text unchanged:** a closed-form rank-one edit of VLM layer 1 teaches "switch" = "turn", fitted live on one
  observation (goal_lan 7: 1 -> 29/30; goal_lan 9: 15 -> 26/30); a "place" = "put" fact learned on one sentence transfers to others
  (goal_lan 8: 0 -> 25/30).
- **Mechanisms** (`library/knowledge/network_map.md`): paraphrase failures are lexical (unknown verbs look like object nouns); the
  action expert reads the instruction as (token, position) pairs through its cross-attention layers, the verb only in its own
  token; targets are memorised per instruction (images steer how, not where); destinations interpolate smoothly along the
  destination noun's representation; one action-expert neuron (layer 4, #1855) controls gripper timing.
- **Learning:** one self-generated success trains an adapter that teaches a paraphrase (0/10 -> 3/5), but adapters did not transfer
  across tasks and 7 hybrid successes were not enough for a swapped layout.
- Caveats: the swap wins are hybrids (the program carries the object to the image-tracked destination and hands back to the VLA);
  some rewrites coincide with the target's trained wording through other sentences; the agentview camera model is used as a
  calibration constant (pending review).

Next: library-transfer comparison on the unseen `libero_10_lan` (seed vs grown library, same agent and budget), then an
equal-budget LoRA baseline for sample efficiency.

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
| `scripts/tables.py` | LaTeX tables for the design document (baseline, sweeps, verified results) |
| `scripts/compare_library.py` | library-transfer comparison: seed vs grown library on unseen tasks, isolated arm copies |
| `prompts/explorer.md`, `prompts/agent.md` | agent instructions |
| `library/` | `steerlib/` (code + README index), `directions/`, `adapters/`, `programs/`, `tools/`, `notes/` (JOURNAL.md), `INDEX.md`, `LITERATURE.md`, `LITERATURE_REVIEW.md` |
| `doc/` | design document |
| `reports/activity/` | activity logbook: `activity_<library version>.html` (one per version) and `latest.html`, built by `scripts/activity.py` |
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
