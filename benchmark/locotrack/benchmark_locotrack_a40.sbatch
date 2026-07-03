#!/bin/bash -l

#SBATCH --job-name=locotrack_bench
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

# Benchmark LocoTrack on the TWIST eval datasets and log to W&B as run 'locotrack'
# (tag 'benchmark'), through the SAME evaluator the TWIST model uses -> directly
# comparable eval/* metrics. See benchmark/locotrack/locotrack.py.
#
#   sbatch benchmark/locotrack/benchmark_locotrack_a40.sbatch
#
# Prereqs (do once, NOT inside this job):
#   * LocoTrack checkpoint -> weights/locotrack/locotrack_base.ckpt (auto-downloaded
#     on first run via the proxy if absent; or fetch manually, see the README note).
#   * The LocoTrack source is already cloned at benchmark/methods/locotrack and is
#     put on sys.path by the script (no pip install of the package needed). Its
#     deps (torch, einops) are expected in $BENCH_VENV.

set -euo pipefail
unset SLURM_EXPORT_ENV

# --- network (W&B + one-time checkpoint download via the FAU proxy) ---
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
# Per-method env: defaults to the shared TWIST .venv. If LocoTrack's deps conflict
# with the TWIST env, point BENCH_VENV at a dedicated venv (see the README note).
module load python
cd "$WS_DIR"
BENCH_VENV="/anvme/workspace/v120bb18-twist/benchmark/locotrack/.venv"
source "$BENCH_VENV/bin/activate"

set -a; [ -f .env ] && . ./.env; set +a

NPROC=${SLURM_GPUS_ON_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$(( CPUS / NPROC ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
echo "host=$(hostname) gpu=a40 NPROC=$NPROC CPUS=$CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS venv=$BENCH_VENV"

# --- benchmark ---
python benchmark/locotrack/locotrack.py 
