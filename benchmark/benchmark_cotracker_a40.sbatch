#!/bin/bash -l

#SBATCH --job-name=cotracker_bench
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mail-user=alberto1.rota@polimi.it
#SBATCH --mail-type=ALL
#SBATCH --output=/home/hpc/v120bb/v120bb18/job_outputs/twist_%x_%j.out
#SBATCH --begin=now

# Benchmark CoTracker3 (offline, scaled) on the TWIST eval datasets and log the
# results to W&B as run 'cotracker3' (tag 'benchmark'), through the SAME evaluator
# the TWIST model uses -> directly comparable eval/* metrics. See benchmark/cotracker.py.
#
#   sbatch .sbatch_scripts/benchmark_cotracker_a40.sbatch
#
# First run downloads the CoTracker3 checkpoint (via the proxy) and caches it to
# weights/cotracker/scaled_offline.pth; later runs reuse it.

set -euo pipefail
unset SLURM_EXPORT_ENV                       # rebuild the job env cleanly

# --- network (W&B + the one-time CoTracker checkpoint download via the FAU proxy) ---
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
source "$WS_DIR/.venv/bin/activate"

# WANDB_API_KEY / HF_TOKEN come from .env (not echoed here).
set -a; [ -f .env ] && . ./.env; set +a

# --- diagnostics ------------------------------------------------------------
NPROC=${SLURM_GPUS_ON_NODE:-1}
CPUS=${SLURM_CPUS_PER_TASK:-8}
export OMP_NUM_THREADS=$(( CPUS / NPROC ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
echo "host=$(hostname) gpu=a40 NPROC=$NPROC CPUS=$CPUS OMP_NUM_THREADS=$OMP_NUM_THREADS"

# --- benchmark --------------------------------------------------------------
# Defaults: all IS_EVAL_DATASET datasets, TAP-Vid queried-first, fp32, W&B run
# 'cotracker3' tagged 'benchmark' in twisteam/twist (config: benchmark/cotracker.yaml).
# Add --datasets / --max-clips / --no-wandb to restrict or smoke-test.
python benchmark/cotracker.py --workers "$CPUS"
