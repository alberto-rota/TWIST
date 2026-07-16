#!/bin/bash
# ============================================================================
# TWIST -> DGX Spark transfer (run FROM the cluster login node).
#
#   Usage:  bash spark/pack_for_spark.sh user@spark-host:/data/twist [--dry-run]
#
# Rsyncs everything the Spark needs (see spark/README.md §1): code, the
# twist_surgical_mix_r512 run dir (checkpoints + run_state), the surgical
# train + eval datasets, and the DINOv3 HF cache. ~640 GB total — run it in
# tmux; every rsync is resumable (-aP), so re-running continues.
#
# NOT copied: Kubric / PointOdyssey / DynamicReplica (~570 G, pretraining is
# done), results/ other than the handoff run, wandb/, .venv (x86 — rebuild on
# ARM per README §2).
# ============================================================================
set -euo pipefail

DEST=${1:?usage: pack_for_spark.sh user@host:/data/twist [--dry-run]}
DRY=""
[ "${2:-}" = "--dry-run" ] && DRY="--dry-run"

WS=/anvme/workspace/v120bb18-twist
RSYNC="rsync -aP $DRY"

echo "### 1/4 code (repo minus data/results/venv) ###"
$RSYNC \
    --exclude DATA --exclude results --exclude wandb --exclude .venv \
    --exclude workshops --exclude logs --exclude __pycache__ \
    --exclude 'inference.ipynb' \
    "$WS/" "$DEST/"

echo "### 2/4 handoff run dir (checkpoints + run_state) ###"
$RSYNC "$WS/results/twist_surgical_mix_r512" "$DEST/results/"

echo "### 3/4 datasets (surgical train + eval; ~640 GB — the long part) ###"
for d in \
    cholec80/cotracker_tracks \
    SurgT/cotracker_tracks \
    EndoTAPP/gt_tracks \
    STIRChallenge_2024/gt_tracks \
    SurgicalMotion/gt_tracks \
    VLsurgPT/gt_tracks \
    tapvid_davis/gt_tracks \
; do
    echo "--- DATA/$d ---"
    $RSYNC --relative "$WS/DATA/./$d" "$DEST/DATA/"
done

echo "### 4/4 DINOv3 encoder (HF cache; place under ~/.cache/huggingface/hub on the Spark) ###"
$RSYNC "$HOME/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m" \
    "$DEST/hf_hub/"

echo
echo "DONE. On the Spark: follow spark/README.md §2 (env) then §3 (resume)."
echo "Reminder: edit .env — remove FAU proxy lines, repoint DATASET_DIR/RESULTS_DIR/WEIGHTS_DIR."
