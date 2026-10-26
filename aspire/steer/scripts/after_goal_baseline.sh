#!/bin/bash
# Wait for the goal family of the baseline, stop the baseline (it resumes later), run the language-shift pilot.
cd "$(dirname "$0")/.."
until [ "$(ls outputs/baseline/libero_goal_lan__*.json 2>/dev/null | wc -l)" -ge 10 ]; do sleep 30; done
for p in $(pgrep -f "scripts/baseline.py --suites"); do kill $p; done
sleep 5
.venv/bin/python scripts/pilot.py --trained libero_goal --perturbed libero_goal_lan
