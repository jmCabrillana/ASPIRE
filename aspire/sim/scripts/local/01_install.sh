#!/usr/bin/env bash
# Local (non-SLURM) install of .venv and .venv-libero on a Blackwell GPU box. Run from aspire/sim.
# Needs the isolated CUDA 12.8 conda env described in scripts/local/cuda_env.sh.
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ASPIRE_ROOT="$PWD" PYTHON_ROOT="$(cd ../.. && pwd)"
source scripts/local/cuda_env.sh

bash scripts/common/apply_contact_graspnet_patch.sh
uv python install 3.10 3.12
[ -d .venv ] || uv venv .venv --python 3.10
uv sync --locked

[ -d .venv-libero ] || uv venv .venv-libero --python 3.12
source .venv-libero/bin/activate
uv sync --locked --active --extra libero --extra contactgraspnet --extra dev
deactivate

mkdir -p "$HOME/.libero"
cat > "$HOME/.libero/config.yaml" <<YAML
benchmark_root: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero
bddl_files: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/bddl_files
init_states: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/init_files
datasets: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/datasets
assets: ${ASPIRE_ROOT}/cap/third_party/LIBERO-PRO/libero/libero/assets
YAML

.venv-libero/bin/python3 -c "import libero, robosuite, sam3, contact_graspnet_pytorch; print('libero env ok')"
.venv-libero/bin/python3 -c "import torch; x=torch.randn(512,512,device='cuda'); print('torch', torch.__version__, 'cuda ok', (x@x).sum().item())"
echo INSTALL_DONE
