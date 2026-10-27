"""Precompute the canon v14 rewrites (normalisation + canon v7 + frame-fill fallback + completions) for every task of a *_lan suite in ONE
process, so that parallel simulator runs of canon programs only hit the cache (each cache miss loads Llama-2-7B, ~13 GB;
two at once = CUDA OOM). Usage: precompute_canon.py <suite>   (prints instruction -> rewrite)"""
import sys
from types import SimpleNamespace
sys.path[:0] = ['library', '.', 'library/programs/candidates']
from steer.rollout import load_task
import canon_v14 as V
suite = sys.argv[1]
i = 0
while True:
    try:
        t = load_task(suite, i)
    except Exception:
        break
    ctx = SimpleNamespace(task=SimpleNamespace(suite=suite, task_id=i, instruction=t.instruction), instruction=t.instruction,
                          log=lambda *a: None, memory={})
    V.setup(ctx)
    print(i, repr(t.instruction), "->", repr(ctx.instruction), flush=True)
    i += 1
