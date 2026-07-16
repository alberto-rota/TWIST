---
name: twist-launch-run
description: Launch, resume, or babysit TWIST GPU training on the FAU SLURM cluster. Use before submitting any training job or sweep, when a job died or hit the 24h wall, when sizing batches for a GPU type, or when planning how many GPUs/resubmits an experiment needs.
---

# TWIST launch & babysit

Login node is CPU-only; all real runs go through SLURM. Up to **8× A100-80** as
independent single-GPU jobs (one run per GPU; DDP scripts exist but the 8-GPU DDP
run was the one that collapsed — the workhorse is single-GPU).

## Launchers (pick one)

```bash
# 1. Generic single-config launcher (THE workhorse for ablations/one-offs):
sbatch .sbatch_scripts/train_cfg_a100_80.sbatch config/<file>.yaml

# 2. Dedicated long-run launcher (hardcodes train_best.yaml):
sbatch .sbatch_scripts/train_best_a100_80.sbatch

# 3. W&B sweep agents (N cells in parallel = submit N agent jobs):
wandb sweep config/<sweep>.yaml            # -> twisteam/twist/<SWEEP_ID>
jsub -w 'wandb agent twisteam/twist/<SWEEP_ID>' agent_gpu/singlegpu/A100_80GB.sbatch
#   (template lives under .sbatch_scripts/agent_gpu/; repeat the jsub N times;
#    add `--count 1` to the agent cmd for exactly one trial per job)

# Monitor:
squeue -u $USER                            # job states
ls -t ~/job_outputs/ | head                # stdout/err: twist_<jobname>_<jobid>.out
```

All launchers: 24 h wall, FAU proxy exported, `.env` sourced (WANDB_API_KEY,
HF_TOKEN, DATASET_DIR/RESULTS_DIR/WEIGHTS_DIR), `module load python` + `.venv`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Wall-clock math (plan resubmits BEFORE launching)

Uncapped-Kubric bs6 @512 ≈ **3.9–4.3 h/epoch** (incl. the epoch-0 monitor eval)
⇒ **~5 epochs per 24 h wall**. So: 20-epoch ranking run ≈ 4 submits; 50-epoch long
run ≈ 10 submits. Consequences:

- Anything >5 epochs needs **fixed EXPERIMENT_NAME + `RESUME: last`** — a resubmit
  of the same sbatch then CONTINUES (model+optim+scaler+epoch from `last.pt`, W&B
  run resumed). Blank name + `scratch` restarts from zero under a new name each
  submit (this lost absurd-dust-35 and both 640/768 resolution runs).
- **Sweep agents do NOT survive the wall**: trials run with blank names + scratch;
  a re-submitted agent starts the NEXT/new trial, not the dead one. Grid cells that
  must train past ~5 epochs should either (a) cap data to fit the wall (the
  fine-loc sweep pattern), (b) be judged at matched partial epochs, or (c) be
  converted to standalone fixed-name configs launched via the generic launcher
  (the `config/ablation_*.yaml` pattern — preferred).
- Timeout resubmit playbook: `sbatch` the same script again — nothing else. Verify
  in W&B that the run RESUMED (same run id via WANDB_RESUME) rather than forked.

## Resume semantics (get this wrong and you silently train garbage)

- Run identity = `EXPERIMENT_NAME` → `$RESULTS_DIR/<name>/run_state.json`;
  `--resume-run <name>` continues a phased schedule at `first_incomplete_stage`.
- Re-running the same stage resumes `last.pt`; a fresh later stage carries the
  previous stage's weights with `strict=False`.
- **`strict=False` is a footgun**: resuming a checkpoint into a CHANGED architecture
  loads silently, dropping mismatched weights. Architecture/flag change (COARSE,
  STATS, VIS_INPUT, head shapes) ⇒ fresh EXPERIMENT_NAME + `RESUME: scratch`, always.
- A resumed run inherits the old checkpoint's training history — if its diagnostics
  look unlike a fresh sibling (e.g. twist_best's low gate vs the sweep cells),
  suspect the inherited checkpoint before suspecting the config.

## Sizing & smoke

- A100-80: BATCH_SIZE **6** @512 ≈ 75 GB peak (measured). Push 8 only after a
  peak_mem check; drop to 4 on OOM. A100-40: 2–3 @512. Higher IMAGE_SIZE (640/768):
  start bs4 and smoke first.
- Smoke on the login node BEFORE burning a GPU slot: `python train.py -b`
  (boot: tiny data, no-download CNN encoder, no W&B) or `python train.py
  config/smoke.yaml`. For a GPU mem check: 2-step run with `--EPOCHS=1
  --MAX_STEPS_PER_EPOCH=2`, read `peak_mem_gb` from the log.

## Pre-launch checklist

1. Loss-balance pre-flight done? (`w_pos` will dominate; total loss positive —
   see twist-experiment-playbook Law 1.)
2. Fresh vs resume decided per the rules above? EXPERIMENT_NAME unique if fresh?
3. One change vs the base config? (Law 2 — diff against `config/train_best.yaml`.)
4. Boot/smoke run green after the code change? Workshop notebook written/executed
   (twist-workshop skill)?
5. Resubmit plan for the wall (epochs ÷ 5, who resubmits)?
6. Eval monitor knobs sane? (`EVAL_EVERY: 10`, `EVAL_MAX_CLIPS: 30` for training
   runs — uncapped eval inside a training job wastes hours.)
7. After ~30 min: job RUNNING, W&B run appeared, first `train/epoch/w_*` shares
   sane, no NaN grad_norm.
