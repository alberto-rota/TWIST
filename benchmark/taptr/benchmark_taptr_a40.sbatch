#!/bin/bash -l

#SBATCH --job-name=taptr_bench
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --output=/home/hpc/v120bb/v120bb18/job_outputs/twist_%x_%j.out
#SBATCH --begin=now

# Benchmark TAPTRv3 on the TWIST eval datasets and log to W&B as run 'taptr'
# (tag 'benchmark'), through the SAME evaluator the TWIST model uses -> directly
# comparable eval/* metrics. See benchmark/taptr/taptr.py.
#
#   sbatch benchmark/taptr/benchmark_taptr_a40.sbatch
#
# Prereqs (do once, NOT inside this job):
#   * Check out the v3 branch:  cd benchmark/methods/TAPTR && git checkout v3
#   * Compile TAPTR's custom CUDA ops (needs nvcc on a GPU node / build node):
#       cd benchmark/methods/TAPTR/models/dino/ops && python setup.py install
#   * TAPTRv3 checkpoint (Google Drive) -> weights/taptr/TAPTRv3_resnet50_512x512.pth
#       (ResNet-50 512x512 variant; pip install gdown then `gdown <file-id>` — see README).
#   * Deps in benchmark/methods/TAPTR/requirements.txt; likely a DEDICATED venv
#     (set BENCH_VENV). TAPTR is the highest-risk adapter -> first run is a smoke test:
#       python benchmark/taptr/taptr.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb

set -euo pipefail
unset SLURM_EXPORT_ENV

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

export WS_DIR=/anvme/workspace/v120bb18-twist
export DATASET_DIR="$WS_DIR/DATA"
export RESULTS_DIR="$WS_DIR/results"
export WEIGHTS_DIR="$WS_DIR/weights"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

module load python
cd "$WS_DIR"
BENCH_VENV="${BENCH_VENV:-$WS_DIR/.venv}"
source "$BENCH_VENV/bin/activate"

set -a; [ -f .env ] && . ./.env; set +a

NPROC=${SLURM_GPUS_ON_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$(( CPUS / NPROC ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
echo "host=$(hostname) gpu=a40 NPROC=$NPROC CPUS=$CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS venv=$BENCH_VENV"

python benchmark/taptr/taptr.py
