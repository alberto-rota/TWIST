#!/bin/bash -l

#SBATCH --job-name=chrono_bench
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

# Benchmark chrono on the TWIST eval datasets and log to W&B (tag
# 'benchmark'), through the SAME evaluator the TWIST model uses -> directly
# comparable eval/* metrics. See benchmark/chrono/chrono.py.
#
#   sbatch benchmark/chrono/benchmark_chrono_a40.sbatch
#
# Prereqs (do once, NOT inside this job):
#   * Chrono checkpoint (Google Drive) -> weights/chrono/chrono_base.ckpt via gdown
#     (see README; no direct URL). DINOv2 backbone auto-downloads via torch.hub.
#   * Deps include xformers + lightning; likely needs a DEDICATED venv (set BENCH_VENV).
#     Source cloned at benchmark/methods/Chrono (top-level 'models' is namespace-isolated).

set -euo pipefail
unset SLURM_EXPORT_ENV

# --- network (W&B + any one-time downloads via the FAU proxy) ---
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# --- paths (failsafes; same vars live in .env, loaded below) ---
export WS_DIR=/anvme/workspace/v120bb18-twist
export DATASET_DIR="$WS_DIR/DATA"
export RESULTS_DIR="$WS_DIR/results"
export WEIGHTS_DIR="$WS_DIR/weights"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# --- environment ---
# Per-method env: defaults to the shared TWIST .venv. If this method's deps
# conflict with the TWIST env, point BENCH_VENV at a dedicated venv (see README).
module load python
cd "$WS_DIR"
BENCH_VENV="/anvme/workspace/v120bb18-twist/benchmark/chrono/.venv"
source "$BENCH_VENV/bin/activate"

set -a; [ -f .env ] && . ./.env; set +a

NPROC=${SLURM_GPUS_ON_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$(( CPUS / NPROC ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
echo "host=$(hostname) gpu=a40 NPROC=$NPROC CPUS=$CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS venv=$BENCH_VENV"

# --- benchmark ---
python benchmark/chrono/chrono.py
