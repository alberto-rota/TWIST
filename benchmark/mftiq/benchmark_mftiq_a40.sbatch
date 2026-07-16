#!/bin/bash -l

#SBATCH --job-name=mftiq_bench
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

# Benchmark MFTIQ (WACV 2025) on the TWIST eval datasets and log to W&B as run
# 'mftiq' (tag 'benchmark'), through the SAME evaluator the TWIST model uses ->
# directly comparable eval/* metrics (and comparable to the 'mft' baseline).
# Dense forward flow-chaining with a decoupled learned occlusion/uncertainty
# module; one init+rollout per distinct first-visible frame -> heavy, hence 48h.
# See benchmark/mftiq/mftiq.py.
#
#   sbatch benchmark/mftiq/benchmark_mftiq_a40.sbatch
#
# Prereqs (do once, NOT inside this job -- see benchmark/README.md):
#   * Dedicated venv (conflicts with the shared .venv; includes a compiled
#     spatial-correlation-sampler — must be built with system gcc 11.5 on FAU):
#       bash benchmark/mftiq/setup_venv.sh
#   * UOM + flow checkpoints fetched into the clone:
#       cd benchmark/methods/MFTIQ && bash download_model.sh

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
# Dedicated venv REQUIRED (see header / setup_venv.sh).
module load python
cd "$WS_DIR"
BENCH_VENV="${BENCH_VENV:-$WS_DIR/benchmark/mftiq/.venv}"
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
# 'mftiq' tagged 'benchmark' in twisteam/twist (config: benchmark/mftiq/mftiq.yaml).
# Add --datasets / --max-clips / --no-wandb to restrict or smoke-test.
python benchmark/mftiq/mftiq.py 
