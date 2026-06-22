#!/bin/bash
#SBATCH --job-name=endotappprep
#SBATCH --array=0-7                 # 8 parallel shards; each handles videos[shard::8]
#SBATCH --gres=gpu:
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/endotappprep_%A_%a.out
#SBATCH --error=logs/endotappprep_%A_%a.err

# Asynchronous, resumable EndoTAPP -> CoTracker pre-processing.
# Launch with:   sbatch --array=0-7 endotappprep.sbatch
# Each array task is one shard; --shard_id / --num_shards are auto-read from
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT. Re-launching a failed/preempted
# task is safe: already-saved clips are skipped, never recomputed.
#
# EndoTAPP is mono (only a 'left' view), native 1920x1080 @ 30 fps, with short
# sequences (~90-190 frames). We track the whole sub-sampled sequence as a
# single clip (--clip_len 0) and resize 1920x1080 -> 480x854 (same 16:9, no
# distortion; matches the Cholec80 resolution).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

VENV=.venv/bin/python
VIDEOS_DIR=DATA/EndoTAPP
VIDEO_GLOB='tissue_dataset_seqs/*/left/seq*/frames/*-visible.mp4'
OUT_ROOT=DATA/EndoTAPP/cotracker_tracks

set -a
[ -f .env ] && . ./.env
set +a

$VENV cotracker_tracks_prep.py \
    --videos_dir "$VIDEOS_DIR" \
    --video_glob "$VIDEO_GLOB" \
    --out_root   "$OUT_ROOT" \
    --clip_len 0 --frame_stride 1 --grid_size 20 --resize 480 854 \
    --device cuda
