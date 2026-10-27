# Recipes (what to try first on a new target, in order, with evidence)

## A paraphrased instruction (*_lan)
1. Diagnose (seconds, offline): `steerlib.analysis.tokens(x)` and the dialect (`canon.Dialect((trained suite, task))`, twin
   excluded). Unknown words in verb slots, possessives, missing articles, unseen n-grams = the failure (E02, E25).
2. Pure SUBSTITUTION paraphrase (same structure, one unknown word, e.g. 'switch on the stove', 'place the bowl on the stove'):
   - in-network: `programs/candidates/lexfacts.py` if a stored fact covers the word (library/edits/facts.json; transfer, no
     data needed), else `lexedit_live.py` (fits a layer-1 edit on the first observation). Evidence: goal_lan 7 held-out 29/30;
     goal_lan 1/2/4/9 19/20, 19/20, 17/20, 17/20 dev.
   - text: `canon_v5.py` / `canon_v7.py` works equally (goal_lan 7 29/30).
3. Paraphrase with an extra out-of-dialect word ('on top of the plate'): `lexfacts_mask.py` (fact + exact pad-deletion of the
   word when its n-gram is not in the dialect): goal_lan 8 18/20 dev.
4. STRUCTURAL paraphrase (insertions, reordering, possessives, verb phrases 'pick up'): text rewrite is required:
   `canon_v7.py`, then `canon_v12.py` (frame-fill fallback when a verb stays untranslated), then `canon_v13.py` (dialect
   noun-phrase completions: 'black bowl', 'drawer of the wooden cabinet'). Evidence: spatial_lan 1/5/6 held-out 22/30, 24/30,
   27/30 (v6); spatial_lan 7 14/20, 9 8/20 (v12), 4 6/10, 8 11/20 (v13) dev; goal_lan 0 held-out 26/30 (v5).
5. Precompute the canon rewrite in ONE process before parallel simulator runs (Llama-2-7B per process = CUDA OOM).
6. If the program succeeds on a task, one successful episode distilled into a LoRA adapter (tools/train_adapter.py) teaches
   the raw paraphrase to the frozen VLA (3/5 from 1 episode, 4/5 from 3) -> sample-efficiency data point; adapters do not
   transfer across tasks.
7. Out of reach so far: a verb that only the twin task uses (goal_lan 5 'push'); two-step skills whose second step is only in
   the twin (goal_lan 3 'put ... inside').

## A swapped layout (*_swap)
Best so far: `carry_handback4.py` (destination moved, pick unchanged: spatial_swap 0 9/15, 2 5/5, 9 4/5, 4 3/5; v2 without
the nudge search: 0/5 5/15, 4 8/15, 2 6/15); for a moved pick
object `approach_handback.py` (under test). Key: let the VLA do the final alignment and release with its cameras SEEING the
target, its believed position shifted to its memorised point (E27, E29). Weak point: tracking (E14, E30).
Earlier: not solved. What is known (knowledge/swap.md, network_map.md): targets are memorised per instruction; the image barely moves
them; destination interpolation works only among destinations trained with the same object; spatial destinations are
hard-wired. Image-only perception that works: SigLIP feature tracking (E08, E14) + agentview camera model. Placement must be
within 3 cm of the target centre (LIBERO 'On'), so open-loop carry controllers need sub-patch precision.

## Evaluation protocol that worked
Screen 5-10 dev episodes -> confirm on 20 -> VERIFY file (held-out 20-49). Two simulator jobs at a time
(library/tools/simq.sh, chaining via a script that waits on outputs/simq.log). Offline lab first (seconds per experiment).

## Applying the library to a NEW suite (e.g. the reserved transfer suites, run by the operator)
1. `*_lan`: `.venv/bin/python library/tools/precompute_canon.py <suite>` (ONE process; fills the canon cache with Llama-2-7B),
   then run `library/programs/candidates/auto.py` (dispatches to canon_v13). For substitution-only paraphrases the in-network
   alternatives are `lexedit_live.py` / `lexfacts.py` (same cache).
2. `*_swap`: build references first: frozen runs of the TRAINED suite with `--save-obs --trace` (e.g. eps 0-2 per task), then
   `library/tools/make_reference.py <trained suite> <task>` (+ the `_frames.npy` of the first frames for voting), then run
   `auto.py` (dispatches to carry_handback2 when the reference exists). Works when only the destination moved (spatial_swap
   type); a moved pick object needs approach_handback.py (weaker).
3. Screen 5-10 dev episodes, confirm on 20, request verification.
