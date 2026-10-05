#!/usr/bin/env bash
# Start the one-task fix loop in a tmux session (attach with: tmux attach -t aspire-fixloop).
# Starts the perception servers if they are down. Prompt ends at a preflight report; you reply "go".
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.local/bin:$PATH"
export ASPIRE_ROOT="$PWD" PYTHON_ROOT="$(cd ../.. && pwd)" HF_HOME="$HOME/.cache/huggingface"
export MUJOCO_GL=egl TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ASPIRE_PERCEPTION_PYTHON=.venv-libero/bin/python3
export SAM3_GPU=0 GRASPNET_GPU=0 SIM_GPU=0
unset ANTHROPIC_API_KEY   # use the logged-in subscription (generated code must not see a key)

up() { [ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$1/health")" != 000 ]; }
if ! { up 8114 && up 8115 && up 8116; }; then
  bash scripts/common/start_perception_servers.sh --no-molmo --gpu-sam3 0 --gpu-graspnet 0
fi
tmux has-session -t aspire-fixloop 2>/dev/null && { echo "session exists: tmux attach -t aspire-fixloop"; exit 0; }
tmux new-session -d -s aspire-fixloop -c "$PWD" \
  "claude --model claude-sonnet-5 \"\$(cat scripts/local/fixloop_prompt.md)\"; bash"
echo "started. attach: tmux attach -t aspire-fixloop"
