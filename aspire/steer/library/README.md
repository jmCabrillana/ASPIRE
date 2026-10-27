# Library index

| path | what |
|---|---|
| `steerlib/` | reusable Python (indexed in `steerlib/README.md`) |
| `knowledge/language.md` | how the policy reads instructions; why paraphrases fail (lexical); what fixes it |
| `knowledge/swap.md` | swapped layouts: memorised targets, what is and is not a lever, image-only perception |
| `knowledge/agentview_P.npy`, `agentview_H.npy` | failed self-calibrations (kept for the record; scene.py uses the camera model) |
| `references/` | trained-layout references for swap programs (tools/make_reference.py) |
| `programs/candidates/auto.py` | ENTRY POINT for a new task: canon_v13 for *_lan, carry_handback2 for *_swap (see knowledge/recipes.md) |
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
