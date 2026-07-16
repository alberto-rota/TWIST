#!/bin/bash -l

#SBATCH --job-name=tapnext_bench
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

# Benchmark TAPNext (BootsTAPNext, online/causal) on the TWIST eval datasets and
# log the results to W&B as run 'tapnext' (tag 'benchmark'), through the SAME
# evaluator the TWIST model uses -> directly comparable eval/* metrics. TAPNext
# is truly frame-by-frame recurrent (one full forward per frame, no windowing),
# so it is heavier than CoTracker-online on long clips (e.g. RoboTAP's
# ~1200-frame clips = ~1200 forward passes each) -- hence 48h. See
# benchmark/tapnext/tapnext.py.
#
#   sbatch benchmark/tapnext/benchmark_tapnext_a40.sbatch
#
# Needs a GPU unconditionally -- TAPNext hardcodes device='cuda' for its
# recurrent blocks at construction, so it cannot even be built on CPU.
#
# First run downloads the TAPNext checkpoint (via the proxy, public GCS, no
# auth) and caches it to weights/tapnext/bootstapnext_ckpt.npz; later runs
# reuse it.

set -euo pipefail
unset SLURM_EXPORT_ENV                       # rebuild the job env cleanly

# --- network (W&B + the one-time TAPNext checkpoint download via the FAU proxy) ---
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
# 'tapnext' tagged 'benchmark' in twisteam/twist (config: benchmark/tapnext/tapnext.yaml).
# Add --datasets / --max-clips / --no-wandb to restrict or smoke-test.
python benchmark/tapnext/tapnext.py 
