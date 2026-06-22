#!/bin/bash
#SBATCH --job-name=surgtprep
#SBATCH --array=0-7                 # 8 parallel shards; each handles videos[shard::8]
#SBATCH --gres=gpu:
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/surgtprep_%A_%a.out
#SBATCH --error=logs/surgtprep_%A_%a.err

# Asynchronous, resumable SurgT -> CoTracker pre-processing.
# Launch with:   sbatch --array=0-7 surgtprep.sbatch
# Each array task is one shard; --shard_id / --num_shards are auto-read from
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT. Re-launching a failed/preempted
# task is safe: already-saved clips are skipped, never recomputed.
#
# SurgT stacks each stereo pair into one frame, but the axis differs per case:
# the .mp4 cases (1-7,12) stack VERTICALLY (1280x2048, each eye 1280x1024), the
# .avi cases (8-11) stack HORIZONTALLY at smaller, varied resolutions. We read
# the axis from each video's sibling info.yaml (--stereo auto), split the stack,
# and track BOTH eyes as independent streams written to <case>/<clip>/left/ and
# <case>/<clip>/right/. Every eye is resized to 512x640. To process a single eye
# instead, add e.g. `--stereo_eyes left`.
#
# Unlike EndoTAPP, SurgT sequences are LONG (median ~930, up to ~2200 frames).
# Feeding a whole sequence to the offline CoTracker OOMs a 48 GB GPU, so we cut
# fixed-length 48-frame clips (same as Cholec80, memory-safe), yielding several
# clips per video.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

VENV=.venv/bin/python
VIDEOS_DIR=DATA/SurgT
# SurgT is mixed: cases 1-7,12 are vertical-stacked video.mp4 (1280x1024/eye),
# cases 8-11 are horizontal-stacked video.avi (smaller, varied resolutions).
# Match both; --stereo auto reads each video's stack axis from its info.yaml.
VIDEO_GLOB='case_*/*/video.*'
OUT_ROOT=DATA/SurgT/cotracker_tracks

# Reduce allocator fragmentation for the large correlation volumes.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

set -a
[ -f .env ] && . ./.env
set +a

$VENV cotracker_tracks_prep.py \
    --videos_dir "$VIDEOS_DIR" \
    --video_glob "$VIDEO_GLOB" \
    --out_root   "$OUT_ROOT" \
    --stereo auto --stereo_eyes left right \
    --clip_len 48 --frame_stride 1 --grid_size 20 --resize 512 640 \
    --device cuda
