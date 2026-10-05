      You design a proxy, then a script applies it. Goal: the next task on which the 8 programs just written disagree most,
      so that the next round learns from pass/fail contrasts on the same task.
      - Before round 1 only: write $RUN_DIR/proxy.py defining  def score(ctx) -> dict[str, float]  (higher = more expected
        disagreement) with a score for EVERY id in ctx["unvisited"]. It is then FROZEN (hash-locked): never change it, never use --unlock.
      - ctx (JSON, see `select_task.py context ...`): tasks (id, suite, task, language, scene_image path), visited, unvisited,
        history (per visited round: outcomes of each candidate per seed, committee metrics), committee (candidate code paths),
        parents. It must be deterministic, standard library only, no network, and may read the files named in ctx.
      - Design it from what is known after round 0 (programs, round-0 outcomes, descriptors). Suggested proxies: a surrogate
        success model per program fitted on visited tasks; similarity to tasks that already split the committee; a cheap comparison
        of what each program would do on the task's descriptor. State the proxy and why it should track disagreement in findings.md.
      - Each round N >= 1: .venv/bin/python3 scripts/taskselect/select_task.py select --family {{FAMILY}} --run-dir $RUN_DIR --round N --rule harness --proxy $RUN_DIR/proxy.py
        The script runs the proxy, takes the argmax and records every score. Do not override its choice.
