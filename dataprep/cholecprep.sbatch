#!/bin/bash
#SBATCH --job-name=cholecprep
#SBATCH --array=0-7                 # 8 parallel shards; each handles videos[shard::8]
#SBATCH --gres=gpu:
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/cholecprep_%A_%a.out
#SBATCH --error=logs/cholecprep_%A_%a.err

# Asynchronous, resumable Cholec80 -> CoTracker pre-processing.
# Launch with:   sbatch --array=0-7 cholecprep.sbatch
# Each array task is one shard; --shard_id / --num_shards are auto-read from
# SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT. Re-launching a failed/preempted
# task is safe: already-saved clips are skipped, never recomputed.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

VENV=.venv/bin/python
VIDEOS_DIR=DATA/cholec80/videos
OUT_ROOT=DATA/cholec80/cotracker_tracks

set -a
[ -f .env ] && . ./.env
set +a

# Each shard rebuilds index.json from the full output tree when it finishes
# (atomic + scan-based, so this is race-free). Whichever shard finishes last
# leaves a complete manifest. To be fully explicit you can also run a final
# `--dependency=afterok` job:  cholec80_data_prep.py --out_root "$OUT_ROOT" --build_index
$VENV cholec80_data_prep.py \
    --videos_dir "$VIDEOS_DIR" \
    --out_root   "$OUT_ROOT" \
    --clip_len 48 --frame_stride 5 --grid_size 20 --resize 480 854 \
    --device cuda
