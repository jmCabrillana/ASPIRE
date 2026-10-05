      Measure, do not guess: run the 8 programs just written on one seed of every unvisited task, then let the script pick.
      1. .venv/bin/python3 scripts/taskselect/select_task.py probe-plan --family {{FAMILY}} --run-dir $RUN_DIR --round N --gpu 0 --parallel 2
         Run each printed command (one at a time, detached as in step 1). Re-running probe-plan prints only the unfinished ones.
      2. .venv/bin/python3 scripts/taskselect/select_task.py select --family {{FAMILY}} --run-dir $RUN_DIR --round N --rule probe --salt r{{REP}}
         It picks the task where the programs differ most (a fixed pseudo-random task if they differ nowhere).
      Do not read the probe results to change the programs or the choice; the candidates are frozen once probe-plan has run.
