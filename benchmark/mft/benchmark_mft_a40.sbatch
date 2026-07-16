#!/bin/bash -l

#SBATCH --job-name=mft_bench
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

# Benchmark MFT (Multi-Flow dense Tracker, WACV 2024) on the TWIST eval datasets
# and log to W&B as run 'mft' (tag 'benchmark'), through the SAME evaluator the
# TWIST model uses -> directly comparable eval/* metrics. MFT is dense and chains
# RAFT flow over several temporal deltas per frame, forward from each query
# frame's template -> it is heavy (one init+rollout per distinct first-visible
# frame; multi-group clips re-roll the flow), hence 48h. See benchmark/mft/mft.py.
#
#   sbatch benchmark/mft/benchmark_mft_a40.sbatch
#
# Runs on the shared TWIST .venv (needs scipy + ipdb, in pyproject.toml). The
# RAFT-OU flow checkpoint ships inside the clone (benchmark/methods/MFT/
# checkpoints/) -- no download needed. Needs a GPU (MFT's RAFT wrapper is
# cuda-only).

set -euo pipefail
unset SLURM_EXPORT_ENV                       # rebuild the job env cleanly

# --- network (W&B logging via the FAU proxy) ---
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# --- paths (failsafes; the same vars live in .env and are loaded below) -----
export WS_DIR=/anvme/workspace/v120bb18-twist
export DATASET_DIR="$WS_DIR/DATA"
export RESULTS_DIR="$WS_DIR/results"
export WEIGHTS_DIR="$WS_DIR/weights"

# Fewer allocator fragments; quiet the HF tokenizer fork warning.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# --- environment ------------------------------------------------------------
module load python
cd "$WS_DIR"
BENCH_VENV="${BENCH_VENV:-$WS_DIR/.venv}"
source "$BENCH_VENV/bin/activate"

# WANDB_API_KEY / HF_TOKEN come from .env (not echoed here).
set -a; [ -f .env ] && . ./.env; set +a

# --- diagnostics ------------------------------------------------------------
NPROC=${SLURM_GPUS_ON_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$(( CPUS / NPROC ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
echo "host=$(hostname) gpu=a40 NPROC=$NPROC CPUS=$CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS venv=$BENCH_VENV"

# --- benchmark --------------------------------------------------------------
# Defaults: all IS_EVAL_DATASET datasets, TAP-Vid queried-first, fp32, W&B run
# 'mft' tagged 'benchmark' in twisteam/twist (config: benchmark/mft/mft.yaml).
# Add --datasets / --max-clips / --no-wandb to restrict or smoke-test.
python benchmark/mft/mft.py 
