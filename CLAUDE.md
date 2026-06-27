# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**TWIST** (Tracking via World-model State Transitions) reframes dense surgical tissue-point tracking as a **passive, observation-corrected state-space world model**: tracked points are the *state*, the next frame is the *exogenous input*. A separable learned **transition prior** (frame-free tissue dynamics, handles occlusion by rollout) is corrected by an **observation update** from each new frame. The full concept, novelty positioning, dataset/phase strategy, and key papers are in `CORE.md` — read it before any architecture or research-framing work. Hardware budget: 1× A100 40GB.

## Working conventions (important)

- **Build step by step.** Do not implement large swaths in one go — build one component, confirm it works, then move on.
- **Every codebase change ships with a workshop notebook.** For each addition/change, write and *execute* `workshops/NN_<topic>.ipynb` that demonstrates it by **calling the exact modules the real runs use** (`utilities.config.load_and_process_config`, `create_datasets_from_config`, `create_model_from_config`, `Engine`, …) — not a mock. The user reviews progress through these. Build via `nbformat` then `jupyter nbconvert --execute --inplace`; set `http_proxy=http://proxy.nhr.fau.de:80` for execution. First cell should `os.chdir` to repo root and add it to `sys.path`. `workshops/` is gitignored.
- This codebase deliberately **mirrors `/anvme/workspace/v120bb18-unreflectanything`** (entrypoints, config-from-sweep loading, engine design). When in doubt about a pattern, that repo is the reference.

## Commands

```bash
# Train (login node is CPU-only — for real runs submit the SLURM job below)
python train.py                         # uses config/train.yaml
python train.py config/schedule.yaml    # multi-stage phased schedule
python train.py config/smoke.yaml       # CPU smoke run (AMP off, no scheduler)
python train.py -b                      # boot mode: tiny smoke + no-download CNN encoder, no W&B

# Override any field (dotted keys for nested ones), type-coerced to the existing type
python train.py --DATASETS.KUBRIC.MAX_POINTS=512 --EPOCHS=10 --no-wandb

# Resume a phased run at its first unfinished stage (run identity = EXPERIMENT_NAME)
python train.py config/schedule.yaml --resume-run <EXPERIMENT_NAME>
python train.py config/schedule.yaml --start-stage 1   # force a starting stage

# W&B sweep (sweep_agent.py passes wandb.config straight into run_pipeline)
wandb sweep config/sweep.yaml           # -> SWEEP_ID
wandb agent <entity>/<project>/<id>

# Real GPU training: submit to SLURM (A100-40). Re-submitting auto-resumes from last.pt.
sbatch train_a100.sbatch                # job output -> /home/hpc/v120bb/v120bb18/job_outputs/
```

`uv`-managed env (`.venv`, Python 3.13): use `source .venv/bin/activate` (the sbatch scripts do `module load python` first). **There is no test suite** — verification is done through executed workshop notebooks and the boot/smoke runs.

## Architecture

**Single orchestrator, two entrypoints.** Everything routes through `main.run_pipeline(mode, config)`. `train.py` calls it with CLI parsing; `sweep_agent.py` calls it with `config=dict(wandb.config)` (the *exact same path*). `run_pipeline` loops over schedule **stages**: per stage it builds datasets → dataloaders → model + loss → runs `Engine(...).fit()` → checkpoints → marks the stage complete.

**Config system** (`utilities/config.py`). Configs are **W&B-sweep-format YAML** (`parameters: {KEY: {value: ...}}`), flattened to `{KEY: value}` and loaded into a `DotMap`. The same file therefore drives a plain run and a sweep. CLI `--KEY=val` and dotted `--A.B.C=val` overrides are coerced to the existing value's type; W&B sweeps deliver nested overrides as flat dotted keys that get folded back into the nested structure. `boot_mode` (`-b`) shrinks batch/epochs/datasets and forces the no-download CNN encoder (login node has no GPU/network for DINOv3).

**Phased training schedule.** A config's `STAGES` list is an ordered schedule (Kubric pretrain → long-horizon → surgical → …). `resolve_stage_config(cfg, i)` overlays stage `i`'s override block onto the base config — **top-level keys replace, not merge** (so a stage's `DATASETS` and `MODEL` fully define that phase). A config without `STAGES` is one implicit stage. Run state lives in `$RESULTS_DIR/<EXPERIMENT_NAME>/run_state.json` (`utilities/runs.py`); `--resume-run` continues at `first_incomplete_stage`. Cross-stage: a rerun of the same stage resumes from `last.pt` (model+optim+scaler+epoch); a fresh later stage **carries the previous stage's weights** (`strict=False`).

**Datasets** (`dataset/`). `DATASET_DEFAULTS` in `dataset/wrappers.py` is the single source of truth for *where* each dataset lives (`ROOT_DIR`) and *which reader* serves it (`READER`); configs only override per-experiment sampling. `create_datasets_from_config` merges defaults < `ALL_DATASETS` overrides < per-dataset config, splits sequences (fractional+seed or explicit lists), and forwards only the config keys the reader's `__init__` accepts (UPPER_CASE → snake_case). Every reader yields the **canonical tracking item dict** documented in `dataset/__init__.py`: `frames (T,3,H,W)`, `tracks (T,N,2)` pixel xy, `visibility (T,N)`, `queries (N,3)=(t,x,y)`, `frame_size (2,)`, `video`, `clip_idx`, optional `depths`. **Every dataset (including CT3Kubric) is now served by the single shared reader** `dataset/cotracker.py::CoTrackerTracksDataset` over the common `index.json` + per-clip `.npz` layout — there is no longer a dataset-specific reader. CT3Kubric ships in a bespoke per-sequence layout, so `ct3kubric_data_prep.py` **converts** it into the shared layout (transpose to frame-major, embed frames, drop the model-unused depth) under `DATA/CT3Kubric/cotracker_tracks`; the other `*_data_prep.py` scripts produce the same layout directly.

**World model** (`models/`, RSSM-style). `TrackerWorldModel`: frozen encoder → frame-free `TransitionModel` (per-point GRU + inter-point self-attention, bounded tanh displacement) → `ObservationModel` (local cost-volume cross-attention; separate corr/vis/logvar heads) → soft visibility-gated state. `models/encoder.py` `FrozenFrameEncoder` wraps DINOv3 (default `facebook/dinov3-vitl16-pretrain-lvd1689m`, HF-cached/offline) with a no-download CNN fallback for CPU/smoke/boot; works internally in normalized [-1,1] coords, returns pixels. `models/losses.py` `TrackerLoss` = **direct Huber position (must dominate)** + visibility BCE + KL(posterior‖prior) with KL-balancing + free-bits. `models/metrics.py` `tracking_metrics` = TAP metrics (EPE, δ_avg, OA, AJ); headline is `val/epe`. Models are built via `create_model_from_config`/`create_loss_from_config` (dynamic `getattr(models, MODEL_CLASS)`), never instantiated directly.

**Training engine** (`utilities/engine.py`). Trains one stage end to end. Per-component optimizer groups (`encoder.*` → `RGB_ENCODER_LR`; rest → `LR`; `RGB_ENCODER_LR==0` or `FREEZE_BACKBONE` → encoder frozen, no group). AMP = bf16 on A100 / fp16+GradScaler elsewhere / off on CPU. All schedules are **pure functions of epoch** (resume-safe): cosine LR + linear warmup, KL weight annealed `KL_WEIGHT_START`→`MODEL.LOSS.KL_WEIGHT`, teacher forcing for the first `TEACHER_FORCING_EPOCHS`. Per epoch: train → validate → checkpoint `last.pt`/`best.pt` under `$RESULTS_DIR/<run>/stage{idx}_{name}/`. W&B opens one run per schedule and logs per-epoch scalars + a periodic pred-vs-GT `wandb.Video` (`utilities.visualization.render_comparison_frames`).

## Environment & paths

`.env` (gitignored, loaded by `utilities/env.py` `load_env()`) holds `WANDB_API_KEY`, `HF_TOKEN`, FAU proxy vars, and the path roots `DATASET_DIR` / `RESULTS_DIR` / `WEIGHTS_DIR`. Config YAMLs refer to data with `$DATASET_DIR/...` placeholders that `expand_path` resolves (`DATASET_DIR` defaults to `./DATA`, `RESULTS_DIR` to `./results`). `DATA/`, `results/`, `weights/`, `logs/`, `wandb/`, and `workshops/` are gitignored.

## Status / next

Built and verified: Kubric data pipeline, phased/resumable orchestration, world model, training engine, A100 launcher, **benchmark evaluation** (`utilities/evaluation.py` + `evaluate.py`). **Next: wire the surgical datasets** (Cholec80 / EndoTAPP / SurgT — prep scripts `cholec80_data_prep.py` / `cotracker_tracks_prep.py` exist and share an `index.json` layout but aren't yet in `DATASET_DEFAULTS`) into a later schedule stage, then add **STIR evaluation**. Surgical readers slot in by adding a `DATASET_DEFAULTS` entry + a stage `DATASETS` block (the cross-stage weight carry already supports it).

**Benchmark evaluation** (`utilities/evaluation.py`). Scores the model on the datasets a config flags `IS_EVAL_DATASET: True` (per-dataset, default `False`; the TAP-Vid / RoboTAP / EndoTAPP-GT registry entries default `True`) and reports, **meaned per dataset**, the headline TAP metrics — **EPE / Delta AVG / Average Jaccard / Occlusion Accuracy** (reusing `models.metrics.tracking_metrics`, the same definitions the engine monitors, so eval `epe` is directly comparable to training `val/epe`) plus **Time (ms/frame)** (timed around the forward, CUDA warm-up excluded). Emits a **CSV under the run dir** (rows = datasets + a `MEAN` row, cols = metrics) and, when W&B is active, a **`wandb.Table`** + per-dataset/mean scalars. One scorer, three entry points: standalone `python evaluate.py --run <EXPERIMENT_NAME> [--wandb]` (rebuilds the model from the checkpoint's embedded config), the live-model `evaluate_and_report(...)`, and the engine hooks **`EVAL_AT_END`** (after each stage) / **`EVAL_EVERY`** (every N epochs, for monitoring). Eval scores whole datasets on rank 0 only (DDP-safe via a barrier); cap with `EVAL_MAX_CLIPS` for a quick smoke.

### Eval protocol (must match CoTracker / TAP-Vid to be comparable)

We follow the **TAP-Vid "queried first"** protocol (the one CoTracker reports): each GT point is **queried at its first *visible* frame**, and metrics are scored **only on the frames strictly after that query frame** (`EVAL_QUERY_MODE: first`, the benchmark default). This matters because the readers store **occluded GT coordinates as `(0,0)`** (placeholder; `visibility=False`) — and ~**17 % of TAP-Vid-DAVIS** and ~**24 % of RoboTAP** points are occluded at frame 0. The TWIST model takes one query frame per forward, so `first` mode groups a clip's points by their first-visible frame and runs **one forward per group** (timing's ms/frame still uses a single all-points pass).

How occluded GT is (correctly) handled in `models.metrics.tracking_metrics`: occluded frames are **masked out by GT visibility** — `epe`/`delta` sum only over visible-and-evaluated points, and in Average Jaccard an occluded point can only ever be a *false positive* (if the model predicts it visible), never a distance term — so the `(0,0)` placeholder **does not** corrupt the metric *at occluded frames*. The damage the `(0,0)` does is only as a **query coordinate** (a point queried at frame 0 while occluded there is tracked from `(0,0)`); querying at the first *visible* frame is the fix. Definitions match TAP-Vid exactly: `OA = mean(pred_vis == gt_vis)` over evaluated frames; `δ_avg` = fraction of visible points within {1,2,4,8,16}px; `AJ = TP/(TP+FP+FN)` combining position + visibility.

### Metrics to beat (the bar for "competitive with CoTracker")

Headline targets are **CoTracker3 trained on Kubric** (TWIST's own pretraining regime — the apples-to-apples bar); original **CoTracker** is the secondary reference. Higher is better for AJ / δ_avg / OA; "off"/"on" = offline/online variant. Numbers from CoTracker3 (arXiv:2410.11831), TAP-Vid "queried first".

| Benchmark (metric) | CoTracker (Kub) | CoTracker3 off (Kub) | CoTracker3 on (Kub) |
|---|---|---|---|
| TAP-Vid Kinetics — AJ / δ_avg | 49.6 / 64.3 | 53.5 / 66.5 | 54.1 / 66.6 |
| TAP-Vid DAVIS — AJ / δ_avg | 67.4 / 78.9 | **74.0 / 84.9** | 71.1 / 81.9 |
| TAP-Vid RGB-Stacking — AJ / δ_avg | 61.8 / 76.1 | 63.3 / 76.2 | 64.5 / 76.7 |
| Dynamic Replica — δ_avg(vis) / δ_avg(occ) | 68.9 / 37.6 | 69.8 / 41.8 | 72.9 / 41.0 |
| RoboTAP — δ_avg / OA | 70.6 / 87.0 | 73.4 / 87.1 | 73.7 / 87.1 |

Surgical benchmarks (VLsurgPT / SurgicalMotion / EndoTAPP / SurgT) have **no published CoTracker number** — TWIST's surgical-adapted result is the contribution there; report it against the **zero-shot CoTracker3** baseline run through this same evaluator.
