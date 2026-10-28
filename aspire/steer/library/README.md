# Library index

| path | what |
|---|---|
| `steerlib/` | reusable Python (indexed in `steerlib/README.md`) |
| `knowledge/language.md` | how the policy reads instructions; why paraphrases fail (lexical); what fixes it |
| `knowledge/swap.md` | swapped layouts: memorised targets, what is and is not a lever, image-only perception |
| `knowledge/agentview_P.npy`, `agentview_H.npy` | failed self-calibrations (kept for the record; scene.py uses the camera model) |
| `references/` | trained-layout references for swap programs (tools/make_reference.py) |
| `programs/candidates/auto.py` | ENTRY POINT for a new task: canon_v16 for *_lan, carry_handback4 for *_swap (see knowledge/recipes.md) |
| `programs/candidates/` | programs: `canon_rewrite.py` (v2), `canon_v5/6/7.py`, `virtual_frame.py` (failed), `carry_to_target.py`; `programs/probes/` feasibility probes (not candidates) |
| `notes/SUMMARY.md` | state of the research: verified results, mechanisms, rules (read first) |
| `notes/JOURNAL.md` | lab notebook (chronological) |
| `knowledge/recipes.md` | what to try first on a new target, in order, with evidence |
| `knowledge/network_map.md` | map of the network: what lives where, with evidence |
| `edits/` | stored lexical weight edits (learned facts) + facts.json |
| `adapters/` | LoRA adapters distilled from self-generated successes (+ json with sample efficiency) |
| `notes/lab/` | offline lab experiments (e01..), instruction lists, canon outputs |
| `tools/` | `simq.sh` (simulator queue, 2 parallel), `lm_probe.py`, `canon_tune.py` + `canon_devset.json`, `frames.py`, `calib*.py`, `make_reference.py` |
| `INDEX.md` | held-out verification results (written by the driver) |
| `knowledge/discoveries.md` | discovery ledger: every intervention, the experiment that found it, method, data, cost, seed or not |
| `notes/panel.md` + `tools/panel.py` | fixed 16-task development panel scored per library version (curve version -> success) |
| `directions/` | learned motion levers (session 2): `dest_expert0_y8.pt` (E37 vector), `dest_iface_L0.pt` (E39), `dest_pinv_L{0,4,8}.pt` (E40 closed-form destination interface), `reach_pinv_L0.pt` (E41) |
| `programs/candidates/dest_swap_fb.py` | internal swap controller: VLA does all motion, program feeds the tracked offset into the E40 interface + integral feedback |
| `programs/candidates/open_then_place.py` | skill composition for 'open X and put Y in it' (goal_lan 3; unsolved, see recipes) |
| `steerlib/levers.py` | minimal sufficient intervention: sparse (L1) gradient search over MLP neurons; per-chunk dose knob for programs |
