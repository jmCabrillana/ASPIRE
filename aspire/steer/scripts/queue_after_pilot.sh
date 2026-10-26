#!/bin/bash
# After the language pilot: one agent round on the pilot's task, then resume the remaining baseline suites.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/after_goal_baseline.sh" >/dev/null || pgrep -f "scripts/pilot.py" >/dev/null; do sleep 30; done
TASK=$(grep -oP '^task \K[0-9]+' outputs_pilot_lan.log | head -1)
.venv/bin/python scripts/agent_round.py --suite libero_goal_lan --task "${TASK:-0}" --budget 12 > outputs_agent_lan.log 2>&1
.venv/bin/python scripts/baseline.py --suites libero_spatial libero_spatial_swap libero_spatial_lan libero_object libero_object_swap libero_object_lan --episodes 10 > outputs_baseline2.log 2>&1
