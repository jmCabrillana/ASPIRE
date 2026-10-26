#!/bin/bash
# Environment for Code-as-Steering: a copy of aspire/sim/.venv-libero (torch cu128, LIBERO-PRO, robosuite) plus lerobot + SmolVLA.
# Copying avoids re-downloading torch; the original venv is not touched.
set -e
cd "$(dirname "$0")"
export UV_HTTP_TIMEOUT=600
[ -d .venv ] || cp -a ../sim/.venv-libero .venv
uv pip install --python .venv/bin/python -c constraints.txt "lerobot[smolvla]==0.6.1"
.venv/bin/python -c "import torch, lerobot, libero; print(torch.__version__, torch.cuda.is_available(), lerobot.__version__)"
