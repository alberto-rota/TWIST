# TWIST → DGX Spark handoff

Cluster access (8× A100-80) ends soon; the DGX Spark (GB10, 128 GB **unified**
memory, aarch64, single device) takes over **only the surgical specialization**.
This directory is the complete handoff: what to copy, how to set up the ARM
environment, and how to resume training from whatever state the cluster chain
reached.

## 0. What the cluster chain leaves behind

The chain (SURG8 → TWCHAIN2..4 → TWEVAL, see
`.sbatch_scripts/chain_pretrain_finetune.sh`) drives run
**`twist_surgical_mix_r512`** (`config/surgical_domain.yaml`):

- **stage 0 `mix_pretrain`** — Kubric+PointOdyssey+DynamicReplica, 50 ep (~27 h
  remained from ep23; finishes in the first chain window or two).
- **stage 1 `surgical_finetune`** — Cholec80+SurgT pseudo-labels, 20 ep (~40 h).
- **TWEVAL** — final uncapped `evaluate.py` (quotable numbers).

Depending on when access ends, the run dir will be: stage-1 complete /
stage-1 partial / stage-0 only. **All three cases resume with the same command**
(§3) — `run_state.json` + `last.pt` encode the state.

## 1. What to copy to the Spark (~640 GB; use `pack_for_spark.sh`)

| What | Path (cluster) | Size |
|---|---|---|
| Code | `/anvme/workspace/v120bb18-twist` (minus DATA/results/wandb/.venv) | ~50 MB |
| Run dir (checkpoints + run_state) | `results/twist_surgical_mix_r512/` | ~5 GB |
| Cholec80 pseudo-labels (train) | `DATA/cholec80/cotracker_tracks` | 487 G |
| SurgT pseudo-labels (train) | `DATA/SurgT/cotracker_tracks` | 113 G |
| EndoTAPP GT (eval) | `DATA/EndoTAPP/gt_tracks` | 1.2 G |
| STIR Challenge (eval) | `DATA/STIRChallenge_2024/gt_tracks` | 26 G |
| SurgicalMotion (eval) | `DATA/SurgicalMotion/gt_tracks` | 0.7 G |
| VLsurgPT (eval) | `DATA/VLsurgPT/gt_tracks` | 6 G |
| TAP-Vid DAVIS (forgetting monitor) | `DATA/tapvid_davis/gt_tracks` | 1.6 G |
| DINOv3 encoder (HF cache) | `~/.cache/huggingface/hub/models--facebook--dinov3-vitl16-pretrain-lvd1689m` | ~1.2 G |
| Secrets | `.env` (edit per §2!) | — |

Kubric/PointOdyssey/DynamicReplica are **not** needed (pretraining is done);
that alone saves ~570 G.

## 2. Environment on the Spark (aarch64 + Blackwell — the cluster .venv does NOT transfer)

```bash
# 1. uv + venv (Python 3.13, same as cluster)
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ~/twist && uv venv --python 3.13 && source .venv/bin/activate

# 2. PyTorch FIRST, from the CUDA-13 index (GB10 = sm_121; needs cu130 aarch64 wheels)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 3. The rest from pyproject (torch already satisfied, won't be re-resolved)
uv pip install -e . --no-deps
uv pip install dotmap einops flow-vis "imageio[ffmpeg]" matplotlib opencv-python \
    python-dotenv scipy tensorboard tqdm "transformers>=5" "wandb[media]"

# 4. Verify before anything else
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If the pip wheels don't cover sm_121 yet, fall back to NVIDIA's NGC PyTorch
container (`nvcr.io/nvidia/pytorch:<latest>-py3`) and `pip install` the deps
inside it.

**`.env` edits (important):**
- **DELETE the FAU proxy lines** (`http_proxy`/`https_proxy` → the Spark has
  direct internet; the proxy would break W&B/HF).
- Point the roots at the Spark's disk:
  `DATASET_DIR=/path/to/DATA`, `RESULTS_DIR=/path/to/results`,
  `WEIGHTS_DIR=/path/to/weights`.
- Keep `WANDB_API_KEY` and `HF_TOKEN`.

Put the copied DINOv3 folder under `~/.cache/huggingface/hub/` (or let it
re-download — HF_TOKEN required, it's a gated repo).

## 3. Resume/continue training (all cluster-end states, same command)

```bash
# Smoke first (CPU-safe, no GPU/W&B):
python train.py -b

# THE continuation command — resumes wherever the cluster stopped
# (mid-stage-1 -> continues last.pt; stage-0-only -> starts stage 1 with the
#  weight carry; run complete -> no-op, use the fresh-run path below):
python train.py config/spark_surgical.yaml --resume-run twist_surgical_mix_r512
```

`config/spark_surgical.yaml` = `surgical_domain.yaml` with Spark sizing:
BATCH_SIZE 4 (single device, unified memory headroom — try 6 after a
`peak_mem_gb` check), WORKERS 8, PIN_MEMORY off, **stage-1 data capped**
(CHOLEC80 3000 + SURGT 1000 ≈ 3.6k clips ≈ 6–8 h/epoch on GB10; uncapped would
be ~40 h/epoch). LOSS/LR identical to the validated recipe. NO torchrun —
plain `python` (single device).

**If the cluster finished stage 1** and you want further specialization
experiments (more epochs, different caps, LR tweaks):

```bash
bash spark/seed_from_pretrain.sh twist_spark_ft results/twist_surgical_mix_r512/stage1_surgical_finetune/best.pt
python train.py config/spark_surgical.yaml --EXPERIMENT_NAME=twist_spark_ft --start-stage 1
```

(Each new experiment: fresh EXPERIMENT_NAME, one change vs the base config —
Law 2. Load `.claude/skills/twist-experiment-playbook` before designing any.)

## 4. Evaluation on the Spark

```bash
# Uncapped quotable numbers (slow on GB10 — hours; resumable via --tag):
python evaluate.py --run twist_surgical_mix_r512 --tag spark

# Quick per-benchmark check:
python evaluate.py --run twist_surgical_mix_r512 --datasets STIR_CHALLENGE,ENDOTAPP_GT --max-clips 30
```

Headline protocol reminders (`.claude/skills/twist-evaluate`): TAP-Vid
"queried first"; canonical published-table numbers additionally need
stretch@256; STIR uses thresholds [4,8,16,32,64] and excludes AJ/OA from MEAN.
The surgical bar = **zero-shot CoTracker3 through this same evaluator**
(no published surgical CoTracker number exists — the comparison is the result).

## 5. Expectations on GB10 (be realistic)

- bf16 compute ≈ **under half of ONE A100**, memory bandwidth ~273 GB/s
  (A100: ~2 TB/s) → per-step time will be dominated by the frozen-DINOv3
  forward + dataloading. The capped stage-1 epoch ≈ 6–8 h.
- 128 GB is **unified** — the OS, dataloader workers, and CUDA all share it.
  If you see host-side OOM/thrash, drop WORKERS before dropping BATCH_SIZE.
- Long runs: just leave it running (`tmux`); `RESUME: last` makes any
  interruption resumable, same as on the cluster.
