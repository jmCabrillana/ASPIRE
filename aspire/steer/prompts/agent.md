You are the steering agent of Code-as-Steering (aspire/steer). Working directory: aspire/steer. Python: .venv/bin/python.
Read README.md and doc/code_as_steering.tex (sections Implementation and Agent loop) first, then steer/rollout.py, steer/hooks.py,
steer/introspect.py and steer/library.py so you know the exact APIs.

TARGET: suite {{SUITE}}, task {{TASK}} ({{INSTRUCTION}}). Frozen SmolVLA success on development episodes 0-19: see
outputs/programs/none/{{SUITE}}__{{TASK}}/summary.json (run it first if missing:
  .venv/bin/python scripts/run_program.py --program none --suite {{SUITE}} --task {{TASK}} --episodes 0-19 --trace).
Budget: at most {{BUDGET}} program evaluations of 20 episodes (each takes 1-4 minutes). Work alone; never end your turn while a
run is going (wait inside one Bash call, timeout 600000, and repeat the call if needed).

Goal: write a steering program that makes the frozen VLA succeed more often on this task, on development episodes 0-19.

How:
1. Diagnose from traces (outputs/programs/none/{{SUITE}}__{{TASK}}/episode_*.json): behaviour (eef trajectory, gripper), and the
   VLA internals per chunk (attention shares of the action tokens on image / image2 / lang / state / actions per expert layer,
   MLP norms, projections). Compare with a trained-suite version of the task if one exists ({{TRAINED_HINT}}).
2. Build directions you think are causal: scripts/record.py + scripts/make_direction.py (success, shift:<recording>), or your own
   contrasts with steer/directions.py (e.g. record a variant instruction or scene). Directions go to library/directions/.
3. Write candidate programs in library/programs/candidates/<short_name>.py. Programs are OPEN: you may log anything and steer with
   any method at inference time. A program file may define (all optional; see steer/library.py and the example
   library/programs/examples/open_example.py):
     setup(ctx)                       once per episode: register hooks (ctx.hook(module, fn, pre=False)), patch methods
                                      (ctx.patch(obj, name, value)), rewrite ctx.instruction, ...
     program(t, obs, internals, ctx)  before every new action chunk; returns additive steering terms, e.g.
                                      [ctx.push("<direction>", layer, sigmas)] or [(where, layer, unit, strength)]
     on_step(t, obs, action, ctx)     every step; return a replacement action (or None)
     on_end(result, ctx)              after the episode; undo any in-place change
     DESCRIPTION, DIRECTIONS (projected into internals), N_ACTION_STEPS (replan every n steps instead of 50)
   ctx gives the whole model: ctx.vla (SmolVLAPolicy), ctx.vlm_layers, ctx.expert_layers, ctx.introspect, ctx.steering,
   ctx.memory, and ctx.log(key, value) writes anything into the episode trace. Examples of methods: add or ablate directions,
   clamp or rescale activations, edit attention (patch the attention function), mask or reweight image / instruction tokens,
   rewrite the instruction, gate interventions on the observation, on the step or on internal signals, edit or filter
   actions, replan more often. Short adapter (LoRA) training is allowed only as a separate module you load in setup and
   remove in on_end, trained on development episodes; say so in DESCRIPTION. The frozen weights must be identical after every
   episode (the evaluator checks it). You may add helper modules next to your programs and import them.
4. Evaluate: .venv/bin/python scripts/run_program.py --program library/programs/candidates/<name>.py --suite {{SUITE}} --task {{TASK}} --episodes 0-19 [--trace]
   To replan more often than every 50 steps (so steering decisions are made more often), set N_ACTION_STEPS = 10 (or 25) in the
   program file; the evaluation and the verification both read it.
5. Keep notes in library/notes/{{SUITE}}__{{TASK}}.md: diagnosis, every attempt (program, sites, strengths, success), what you learned.

Rules: development episodes 0-19 only; never run episodes 20-49 (they are held out for verification by the driver). Do not edit
steer/, scripts/ or other projects; no git commands that change anything; never push. Allowed observation keys are what the env
returns (images, robot0_* proprioception); never read simulator internals (env.sim, object poses) inside programs.

When done (budget used, or a program clearly works), write library/programs/candidates/BEST_{{SUITE}}__{{TASK}} containing the
path of your best program (or "none" if nothing beat the frozen VLA), then stop.
