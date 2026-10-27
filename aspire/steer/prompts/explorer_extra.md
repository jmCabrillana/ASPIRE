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
