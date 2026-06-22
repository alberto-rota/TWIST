#!/bin/bash
#SBATCH --job-name=podprep
#SBATCH --array=0-7                 # 8 parallel shards; each handles sequences[shard::8]
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/podprep_%A_%a.out
#SBATCH --error=logs/podprep_%A_%a.err

# Asynchronous, resumable PointOdyssey -> shared-layout (index.json + per-clip
# .npz) re-packaging. Launch with:   sbatch --array=0-7 pointodysseyprep.sbatch
# Each array task is one shard; --shard_id / --num_shards are auto-read from
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT. Re-launching a failed/preempted
# task is safe: already-saved clips are skipped, never recomputed.
#
# PointOdyssey already ships GROUND-TRUTH point tracks, so NO CoTracker/GPU
# inference runs here -- this is a CPU/IO-bound conversion (read anno.npz + decode
# JPEGs + write clips). This cluster is GPU-only, so we still request the smallest
# GPU (a40:1); it is used solely for the optional frame resize and is otherwise
# idle. The bottleneck is JPEG decode + disk, hence the higher --cpus-per-task.
#
# Sequences are long (~1.5k-2k frames); we cut fixed-length 48-frame clips and
# keep 400 GT points per clip (chosen among those visible & in-frame at the clip's
# first frame), matching the Cholec80 / SurgT layout. Output is read by the same
# CoTrackerTracksDataset as every other dataset.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

VENV=.venv/bin/python
DATA_DIR=DATA/PointOdissey/train
OUT_ROOT=DATA/PointOdissey/gt_tracks

# Reduce allocator fragmentation for the (optional) resize on GPU.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
[ -f .env ] && . ./.env
set +a

$VENV pointodyssey_data_prep.py \
    --data_dir "$DATA_DIR" \
    --out_root "$OUT_ROOT" \
    --anno_glob '*/anno.npz' \
    --clip_len 48 --frame_stride 1 --num_points 400 \
    --device cuda
