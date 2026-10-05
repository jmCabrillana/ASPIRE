      You are the selector. Goal: the next task on which the 8 programs just written disagree most (some pass, some fail),
      because the difference between a passing and a failing program on the same task is what the next round learns from.
      A task where all would pass or all would fail teaches little.
      1. .venv/bin/python3 scripts/taskselect/select_task.py context --family {{FAMILY}} --run-dir $RUN_DIR --round N > /dev/null
         then read $RUN_DIR/iter_NN/context.json: unvisited tasks (runtime instruction + scene image path, look at the images),
         the committee's code paths, and for every visited task the per-candidate outcomes and disagreement.
      2. Read the 8 programs and the traces of visited tasks. Decide from programs, traces and descriptors ONLY: do not run any
         candidate, do not replay any unvisited task, do not use simulator internals.
      3. Write $RUN_DIR/iter_NN/task_choice.json:
         {"task": "<suite>/<task> (an unvisited id)", "predicted_pass": {"candidate_A": 0.0-1.0, ... all 8 ...},
          "rationale": "which program feature makes them succeed or fail differently on this task"}
         predicted_pass is your probability that each program passes a typical dev seed of that task; it is scored afterwards.
      4. .venv/bin/python3 scripts/taskselect/select_task.py select --family {{FAMILY}} --run-dir $RUN_DIR --round N --rule prompt
         If it rejects the file, fix the file, never the script.
