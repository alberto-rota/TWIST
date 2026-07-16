#!/bin/bash
# ============================================================================
# Seed a FRESH TWIST specialization run from an existing checkpoint (the
# documented --start-stage 1 shortcut from config/surgical_domain.yaml).
#
#   Usage:  bash spark/seed_from_pretrain.sh <NEW_RUN_NAME> <path/to/best.pt>
#   Then:   python train.py config/spark_surgical.yaml \
#               --EXPERIMENT_NAME=<NEW_RUN_NAME> --start-stage 1
#
# Creates $RESULTS_DIR/<NEW_RUN_NAME>/stage0_mix_pretrain/best.pt so the
# engine's cross-stage weight carry picks the checkpoint up as stage 1's init.
# ============================================================================
set -euo pipefail

RUN=${1:?usage: seed_from_pretrain.sh <NEW_RUN_NAME> <path/to/best.pt>}
CKPT=${2:?usage: seed_from_pretrain.sh <NEW_RUN_NAME> <path/to/best.pt>}
RESULTS=${RESULTS_DIR:-./results}

[ -f "$CKPT" ] || { echo "ERROR: checkpoint not found: $CKPT"; exit 1; }

DEST="$RESULTS/$RUN/stage0_mix_pretrain"
if [ -e "$DEST/best.pt" ]; then
    echo "ERROR: $DEST/best.pt already exists — refusing to overwrite a seeded run."
    exit 1
fi
mkdir -p "$DEST"
cp -v "$CKPT" "$DEST/best.pt"

echo
echo "Seeded. Launch stage 1 with:"
echo "  python train.py config/spark_surgical.yaml --EXPERIMENT_NAME=$RUN --start-stage 1"
