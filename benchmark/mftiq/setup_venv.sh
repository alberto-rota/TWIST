#!/usr/bin/env bash
# One-time setup for the MFTIQ benchmark venv on FAU NHR nodes.
#
# MFTIQ upstream pins torch==2.0.1+cu117, but this cluster only ships CUDA 12.x
# toolkits and gcc/15.2.0 (too new for nvcc). We therefore:
#   * install torch 2.1.2+cu121 (minor CUDA mismatch with cuda/12.8 is fine)
#   * compile spatial-correlation-sampler with system gcc 11.5 (/usr/bin/gcc)
#   * target A40 sm_86 via TORCH_CUDA_ARCH_LIST
#
# Usage (from repo root, on a node with network + a cuda module):
#   bash benchmark/mftiq/setup_venv.sh
#   cd benchmark/methods/MFTIQ && bash download_model.sh && cd -
#   sbatch benchmark/mftiq/benchmark_mftiq_a40.sbatch
set -euo pipefail

WS_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${BENCH_VENV:-$WS_DIR/benchmark/mftiq/.venv}"
MFTIQ_DIR="$WS_DIR/benchmark/methods/MFTIQ"
TORCH_INDEX="https://download.pytorch.org/whl/cu121"

export http_proxy="${http_proxy:-http://proxy.nhr.fau.de:80}"
export https_proxy="${https_proxy:-http://proxy.nhr.fau.de:80}"

module load python
# gcc/15.2.0 breaks nvcc; use the system compiler instead.
module unload gcc 2>/dev/null || true
module load cuda/12.8.1
export CUDA_HOME="${CUDA_HOME:-/apps/cuda/12.8.1}"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (pip install uv or use the login-node install)." >&2
  exit 1
fi

echo "Creating venv at $VENV_DIR"
uv venv "$VENV_DIR" --python 3.11
PY="$VENV_DIR/bin/python"

echo "Installing torch 2.1.2+cu121 (FAU-cluster-compatible; upstream pins 2.0.1+cu117)"
uv pip install --python "$PY" \
  "torch==2.1.2" "torchvision==0.16.2" \
  --index-url "$TORCH_INDEX"

echo "Installing MFTIQ runtime deps (skip upstream torch pin)"
uv pip install --python "$PY" \
  "numpy<2" opencv-python einops ipdb tqdm Pillow scipy rich kornia \
  pycolormap-2d matplotlib lz4 cmocean pypng ninja "setuptools<81" wheel

echo "Installing MFTIQ package (no deps — torch already pinned above)"
uv pip install --python "$PY" --no-deps -e "$MFTIQ_DIR"

echo "Compiling spatial-correlation-sampler (needs torch present + system gcc 11.5)"
uv pip install --python "$PY" --no-build-isolation "spatial-correlation-sampler==0.4.0"

echo "Installing xformers for torch 2.1.2+cu121"
uv pip install --python "$PY" "xformers==0.0.23.post1" --index-url "$TORCH_INDEX"

echo "Smoke-importing MFTIQ config (no GPU required for this step)"
(
  cd "$MFTIQ_DIR"
  "$PY" -c "from MFTIQ.config import load_config; load_config('configs/MFTIQ4_RAFT_200k_cfg.py'); print('import OK')"
)

echo "Done. Activate with: source $VENV_DIR/bin/activate"
echo "Fetch checkpoints: cd benchmark/methods/MFTIQ && bash download_model.sh"
