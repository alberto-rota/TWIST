# TWIST — Tracking via World-model State Transitions

TWIST reframes **dense surgical tissue-point tracking** as a **passive, observation-corrected state-space world model** rather than a correspondence-chaining problem. The tracked points are the *state* $s_t$; each new frame $I_{t+1}$ is the *exogenous input* that drives the transition $s_{t+1} = f(s_t, I_{t+1})$. A separable, frame-free **transition prior** (learned tissue dynamics, handles occlusion by rollout) is corrected by an **observation update** from each new frame.

This buys capabilities standard trackers (TAPIR, CoTracker) lack: occlusion handling by dynamics rollout, future-trajectory prediction, and counterfactual simulation. The full concept, novelty positioning, and dataset/phase strategy are in [`CORE.md`](CORE.md).

**Hardware budget:** 1× NVIDIA A100 (40 GB).

## Architecture

RSSM-style world model, built end to end (`models/`):

```
frozen encoder  →  transition model        →  observation model           →  visibility-gated state
(DINOv3, frozen)   (per-point GRU +            (local cost-volume cross-       (soft gate fuses prior
                    inter-point attention,      attention; corr / vis /         + observation update)
                    bounded Δ displacement)     logvar heads)
```

- `models/encoder.py` — `FrozenFrameEncoder` wraps DINOv3 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, HF-cached/offline), with a no-download CNN fallback for CPU/smoke/boot.
- `models/world_model.py` — `TrackerWorldModel` (transition + observation + soft visibility gate).
- `models/losses.py` — `TrackerLoss` = Huber position (dominant) + visibility BCE + KL(posterior‖prior) with KL-balancing + free-bits.
- `models/metrics.py` — TAP metrics (EPE, δ_avg, OA, AJ); headline is `val/epe`.

Models are built via `create_model_from_config` / `create_loss_from_config`, never instantiated directly.

## Project structure

```
main.py                 single orchestrator: run_pipeline(mode, config) loops over schedule stages
train.py                CLI entrypoint  → run_pipeline (config parsing + dotted overrides)
sweep_agent.py          W&B sweep entrypoint → run_pipeline (config = dict(wandb.config))

config/                 W&B-sweep-format YAML (parameters: {KEY: {value: ...}})
  train.yaml            default single-stage run
  schedule.yaml         multi-stage phased schedule (STAGES list)
  smoke.yaml            CPU smoke run (AMP off, no scheduler)
  sweep.yaml            W&B sweep definition

dataset/                canonical tracking item dict (see dataset/__init__.py)
  wrappers.py           DATASET_DEFAULTS — single source of truth for ROOT_DIR + READER per dataset
  kubric.py             Kubric reader (the only reader wired in today)
  sampling.py splits.py collate.py    point sampling, sequence splits, padded collation

models/                 world model, encoder, losses, metrics (see above)

utilities/
  config.py             load_and_process_config, create_*_from_config, override coercion
  engine.py             Engine.fit() — trains one stage end to end (AMP, optim groups, schedules)
  runs.py               run_state.json bookkeeping for resumable phased runs
  visualization.py      pred-vs-GT comparison videos for W&B
  env.py log.py         .env loading / logging

dataprep/               offline dataset preparation scripts + SLURM launchers (*.sbatch)
CORE.md                 concept, novelty, dataset/phase strategy (read before research work)
```

### Config system

Configs are **W&B-sweep-format YAML** (`parameters: {KEY: {value: ...}}`), flattened to `{KEY: value}` and loaded into a `DotMap` — so the same file drives both a plain run and a sweep. CLI `--KEY=val` and dotted `--A.B.C=val` overrides are type-coerced to the existing value. Data paths use `$DATASET_DIR/...` placeholders resolved at load time.

### Phased, resumable training

A config's `STAGES` list is an ordered schedule (Kubric pretrain → long-horizon → surgical → …); a config without `STAGES` is one implicit stage. `resolve_stage_config(cfg, i)` overlays stage `i`'s override block onto the base config (**top-level keys replace, not merge**). Run state lives in `$RESULTS_DIR/<EXPERIMENT_NAME>/run_state.json`. Re-running a stage resumes from `last.pt`; a fresh later stage **carries the previous stage's weights** (`strict=False`).

## Setup

```bash
# uv-managed env (.venv, Python 3.13)
uv sync
source .venv/bin/activate            # sbatch scripts do `module load python` first
```

Create a `.env` (gitignored, loaded by `utilities/env.py`) with:

```bash
WANDB_API_KEY=...
HF_TOKEN=...                         # for DINOv3 weights
# FAU proxy (cluster only)
http_proxy=http://proxy.nhr.fau.de:80
https_proxy=http://proxy.nhr.fau.de:80
# path roots (default to ./DATA, ./results, ./weights)
DATASET_DIR=...
RESULTS_DIR=...
WEIGHTS_DIR=...
```

## How to replicate

### 1. Prepare data

Dataset readers expect a prepared layout under `$DATASET_DIR`. The only reader wired in today is **Kubric** (reads `DATA/CT3Kubric`). Prep scripts and SLURM launchers live in `dataprep/`:

```bash
python dataprep/ct3kubric_data_prep.py          # Kubric (wired in)
sbatch dataprep/cholecprep.sbatch               # surgical sets (prep ready, not yet wired)
sbatch dataprep/pointodysseyprep.sbatch
sbatch dataprep/surgtprep.sbatch
sbatch dataprep/endotappprep.sbatch
```

### 2. Train

The login node is CPU-only — use the smoke/boot modes there and submit real runs to SLURM (A100-40).

```bash
python train.py                                 # config/train.yaml
python train.py config/schedule.yaml            # multi-stage phased schedule
python train.py config/smoke.yaml               # CPU smoke run
python train.py -b                              # boot mode: tiny smoke, no-download encoder, no W&B

# override any field (type-coerced); dotted keys for nested
python train.py --DATASETS.KUBRIC.MAX_POINTS=512 --EPOCHS=10 --no-wandb

# resume a phased run at its first unfinished stage (identity = EXPERIMENT_NAME)
python train.py config/schedule.yaml --resume-run <EXPERIMENT_NAME>
python train.py config/schedule.yaml --start-stage 1
```

Real GPU training is submitted to SLURM; re-submitting auto-resumes from `last.pt`. Job output goes to `/home/hpc/v120bb/v120bb18/job_outputs/`.

### 3. Sweeps

```bash
wandb sweep config/sweep.yaml                   # -> SWEEP_ID
wandb agent <entity>/<project>/<id>             # sweep_agent.py forwards wandb.config to run_pipeline
```

Checkpoints (`last.pt` / `best.pt`) are written per epoch under `$RESULTS_DIR/<run>/stage{idx}_{name}/`. There is no test suite — verification is done through the executed `workshops/` notebooks and the boot/smoke runs.

## Status

Built and verified: Kubric data pipeline, phased/resumable orchestration, world model, training engine, A100 launcher. **Next:** wire the surgical datasets (Cholec80 / EndoTAPP / SurgT) into a later schedule stage, then add **STIR evaluation** (the headline surgical point-tracking benchmark; **Endo-TTAP** is the in-domain baseline to beat).
