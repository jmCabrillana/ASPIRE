SESSION 2 FOCUS (from the operator). Start from the current library (read library/notes/SUMMARY.md, JOURNAL.md, knowledge/,
steerlib/README.md, INDEX.md first). Session 1 found working text rewrites, layer-1 lexical edits and hybrid hand-backs. This
session should go beyond them, toward smarter internal interventions found by real search:
- COMBINATIONS: compose interventions (lexical edit + neuron group + adapter + steering vector + state frame) and measure whether
  they add up, interfere or unlock tasks none solves alone (goal_lan 3 and 5, goal_swap, spatial_swap 1/3/6/8, spatial_lan 0).
- GROUPS AND CIRCUITS: neuron groups (steer/interp.py clusters, interaction/synergy, downstream connections), edits of a whole
  circuit rather than one neuron; group-level levers for the target position, the object identity and the phase.
- MULTIPLE ADAPTERS: several small adapters (per skill, per phase, per object) in an AdapterBank (steer/lora.py), switched per
  chunk by activation-keyed retrieval; distil verified programs into adapters; compare against a single adapter.
- ADVANCED SEARCH: do not hand-tune one knob at a time. Use search over intervention programs: evolutionary or population search
  over (sites, layers, groups, doses, gates), Bayesian optimisation or CMA-ES over a few continuous parameters, beam search over
  compositions, and gradient-based search inside the frozen VLA: steer/lab.py optimize_vector finds an additive vector at a layer
  (optionally on a token span) whose predicted chunks match target chunks (e.g. chunks from successful or rewritten rollouts) in
  seconds, offline. Learned vectors, groups and edits are library entries too. Screen in the offline lab, confirm in the simulator.
- Insertions inside the network: the expert reads (token, position) pairs; inserting tokens into padding slots with assigned
  positions may fix structural paraphrases without rewriting the text.
Keep writing JOURNAL.md, keep the network map and recipes up to date, and keep the reserved suites untouched.

PAPER TARGETS (what the results must eventually show; design experiments so each has clean evidence):
1. Discovered, not predefined: interventions the agent found from simulation and dissection (log how each was found: experiment,
   data, cost), compared with the seed library's predefined tools alone.
2. State-dependent control: programs that decide WHEN (phase, internal signals), WHERE (layer, tokens, neurons, adapter) and HOW
   STRONGLY to intervene from the state; show with ablations that the gating matters (always-on vs gated, fixed vs adaptive dose).
3. Composition on unseen tasks: library entries combined on task compositions never seen during development (multi-step tasks,
   new pairings of objects and destinations); the reserved suites will be the final test, so prepare general entries.
4. Improvement over time without fine-tuning: keep a fixed panel of development tasks and record, per library version, how the
   entry program (programs/candidates/auto.py) does on it, so the curve "library version -> success" can be drawn.

DISCOVERY METHODS (the method of discovery may be the main contribution; unconventional ideas are welcome, e.g.):
- Minimal sufficient intervention: optimise the smallest (sparse, L1) set of neurons or the lowest-rank change that turns a failing
  chunk into a successful one (optimize_vector + sparsity); the support of the solution IS the discovered lever.
- Control view: treat the residual stream as a control input; estimate the Jacobian d(action chunk)/d(activation) per layer and its
  top singular vectors = the directions the policy is most sensitive to (a steering basis); closed-loop controllers on internal
  state estimated by probes (feedback, not open-loop pushes).
- Model diffing: compare the fine-tuned VLM with base SmolVLM2 (which still knows English) to locate what fine-tuning changed, and
  transplant base-model representations or weights where needed (restore lost knowledge instead of re-teaching it).
- Quality-diversity / LLM-guided evolution of programs (MAP-Elites over intervention behaviours, FunSearch-style program evolution
  with the library as the genome) to find many different levers, not one.
- Features: sparse autoencoders trained on our own rollout activations to find cleaner features than neurons, then steer features.
- Automated science: generate hypotheses from failure frames and traces, test the cheapest discriminating experiment first.
