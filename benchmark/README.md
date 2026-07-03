# Baseline benchmarks

Each baseline tracker is run through the **exact same evaluator the TWIST model
uses** (`utilities.evaluation.evaluate_and_report`), so its numbers are directly
comparable to the TWIST runs' `eval/*` metrics: same datasets (everything the
registry flags `IS_EVAL_DATASET: True`), same TAP-Vid "queried first" protocol,
same metric definitions (`models.metrics`), same CSV + W&B table + Post-Occlusion
Recovery artifacts. Each method logs to W&B under its own **run name** tagged
**`benchmark`** in `twisteam/twist`, and writes its CSVs to
`$RESULTS_DIR/<run-name>/`.

## Layout

```
benchmark/
  common.py                  # shared driver: CLI, W&B run, evaluate-and-report,
                             #   adapter helpers, namespace isolation
  methods/                   # upstream source clones (gitignored, each has its own .git)
    co-tracker/  locotrack/  tapnet/  track_on/  Chrono/  TAPTR/
    Python-SuPer/  SurgT_benchmarking/
  <method>/                  # the benchmark *harness* for one method (tracked):
    <method>.py              #   model adapter + build_adapter(); calls common.run
    <method>.yaml            #   W&B-sweep-format config (geometry, model, eval protocol)
    benchmark_<method>_a40.sbatch
```

A method script does three things: fix `sys.path` for its upstream package,
define `build_adapter(cfg, device) -> nn.Module` wrapping the predictor to the
TWIST forward contract `model(frames, queries) -> {"coords", "vis_logits"}`, and
call `common.run(...)`. Everything else is shared.

## Run a benchmark

```bash
# all eval datasets -> W&B run + CSVs under results/<run-name>/
sbatch benchmark/<method>/benchmark_<method>_a40.sbatch

# locally / on a login GPU:
python benchmark/<method>/<method>.py                          # full
python benchmark/<method>/<method>.py --datasets TAPVID_DAVIS --max-clips 5   # smoke
python benchmark/<method>/<method>.py --no-wandb               # CSV only
python benchmark/<method>/<method>.py --image-size 256         # override geometry
```

Common flags (all methods, defined in `common.py`): `--datasets`, `--max-clips`,
`--batch-size`, `--image-size`, `--crop`, `--workers`, `--max-steps`,
`--query-mode`, `--name`, `--tags`, `--out-dir`, `--no-wandb`, `--amp`, `--cpu`,
`--checkpoint`.

## Status

| Method | Harness | Variant benchmarked | Notes |
|---|---|---|---|
| CoTracker3 | ✅ `cotracker/` | online (scaled) | works as before; source moved to `methods/co-tracker` |
| LocoTrack  | ✅ `locotrack/` | base (PyTorch) | clean |
| TAPIR      | ✅ `tapir/`     | BootsTAPIR offline | clean (torch path, no JAX) |
| TAPNext    | ✅ `tapnext/`   | BootsTAPNext | online/causal only, needs a GPU **even to construct** the model (upstream hardcodes `device='cuda'`); frame-by-frame (no windowing), so the heaviest online baseline on long clips; needs `einops` (added to `pyproject.toml`); untested end-to-end (no GPU on the login node) — smoke-test first |
| LiteTracker | ✅ `litetracker/` | LiteTracker (MICCAI'25) | training-free runtime re-opt of CoTracker3-online; loads CoTracker3's **exact** `scaled_online.pth` (0 missing/0 unexpected) so it reuses the cached `weights/cotracker/` ckpt (no separate download); online/causal frame-by-frame; runs on shared `.venv` (torch/einops/numpy/cv2). GPU-smoke-verified: DAVIS EPE 6.95/δ 0.494/AJ 0.391/OA 0.945 @ 12ms/frame (~3× faster than CoTracker/TAPNext) |
| Track-On   | ✅ `trackon/`   | track_on_r | needs gated DINOv3 + mmcv (dedicated venv) |
| Chrono     | ✅ `chrono/`    | ViT-B | top-level `models` is namespace-isolated; needs xformers (dedicated venv) |
| TAPTRv3    | ✅ `taptr/`     | resnet50 512×512 | **code lives in branch `v3`** (checked out); needs compiled CUDA ops; `models`/`main` namespace-isolated; highest-risk adapter — smoke-test first |
| MFT        | ✅ `mft/`       | RAFT-OU (WACV'24) | dense flow-chaining, online/causal, **forward-only from template** so `supports_query_times=False` (1 init+rollout per distinct first-visible frame); RAFT checkpoint ships **in the clone** (no download); runs on shared `.venv` (adds `scipy`+`ipdb`); config/ckpt are relative paths so the tracker builds with CWD at the clone root. GPU-smoke-verified: DAVIS EPE 7.71/δ 0.529/AJ 0.373/OA 0.917 @ 456ms/frame (dense = slow) |
| MFTIQ      | ✅ `mftiq/`     | MFTIQ4 RAFT (WACV'25) | MFT successor (decoupled learned occlusion/uncertainty = UOM); same forward-only driving as MFT. **Dedicated venv** (`torch==2.0.1` + `xformers`/`kornia`/compiled `spatial-correlation-sampler`) + `download_model.sh` for UOM/flow ckpts; untested end-to-end — smoke-test first |
| Python-SuPer | ❌ N/A | — | **does not fit the TAP interface** — RGBD deformable surfel reconstruction; needs stereo/depth + per-sequence optimization, no arbitrary query-point API |
| SurgT_benchmarking | ❌ N/A | — | **not a model** — it's a competing stereo benchmark *harness* (you implement `run_method()`); uses its own data/protocol |

## One-time setup (do these yourself — heavy / network; not run inside the jobs)

The harness auto-downloads weights where a direct URL exists; the rest need a
manual fetch. Put all weights under `$WEIGHTS_DIR` (`weights/`).

```bash
# proxy for any download on the FAU nodes
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80

# --- CoTracker3 (auto-downloads on first run; or fetch manually) ---
mkdir -p weights/cotracker && \
  wget -O weights/cotracker/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth

# --- LocoTrack (auto-downloads on first run; or fetch manually) ---
mkdir -p weights/locotrack && \
  wget -O weights/locotrack/locotrack_base.ckpt \
  https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt

# --- BootsTAPIR (auto-downloads on first run; or fetch manually) ---
mkdir -p weights/tapir && \
  wget -O weights/tapir/bootstapir.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt

# --- BootsTAPNext (auto-downloads on first run; or fetch manually; public, no auth) ---
mkdir -p weights/tapnext && \
  wget -O weights/tapnext/bootstapnext_ckpt.npz \
  https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz

# --- LiteTracker: no separate weights. It loads CoTracker3's exact scaled_online.pth
#     (0 missing/0 unexpected) -> reuses weights/cotracker/scaled_online.pth above.

# --- MFT: no download. The RAFT-OU flow checkpoint ships inside the clone at
#     benchmark/methods/MFT/checkpoints/ (raft-things-sintel-kubric-...pth).

# --- MFTIQ (dedicated venv + checkpoints; see "Dedicated environments" below) ---
cd benchmark/methods/MFTIQ && bash download_model.sh && cd -   # UOM + RAFT/flow ckpts -> its checkpoints/

# --- Track-On (manual; pick track_on_r) ---
mkdir -p weights/trackon && \
  wget -O weights/trackon/track_on_r.pt \
  "https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt?download=true"
# Also request access to the gated backbone and make sure HF_TOKEN can read it:
#   https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m

# --- Chrono (manual; Google Drive, needs gdown) ---
pip install gdown && mkdir -p weights/chrono && \
  gdown 1XYOr5pVncEAgyWcQZ_TjgvqLTcexdUQr -O weights/chrono/chrono_base.ckpt   # ViT-B

# --- TAPTRv3 (manual; Google Drive, needs gdown) ---
pip install gdown && mkdir -p weights/taptr && \
  gdown 19iql2VTqGIeoyg_wt3JjpohszN5UE6s1 -O weights/taptr/TAPTRv3_resnet50_512x512.pth
```

### Dedicated environments

The shared TWIST `.venv` covers CoTracker, LocoTrack, TAPIR, TAPNext,
LiteTracker and MFT (pure-torch paths; their `einops` / `scipy` / `ipdb` deps are
already in `pyproject.toml`). **Track-On** (needs `mmcv==2.2.0`, `timm`,
`transformers`), **Chrono** (needs `xformers`, `lightning`) and **MFTIQ** (pins
`torch==2.0.1` + needs `xformers`, `kornia`, a compiled `spatial-correlation-sampler`)
have heavier/conflicting deps — give each its own venv and point the job at it:

```bash
python -m venv .venv-trackon && source .venv-trackon/bin/activate
pip install -r benchmark/methods/track_on/requirements.txt
# then: BENCH_VENV=$PWD/.venv-trackon sbatch benchmark/trackon/benchmark_trackon_a40.sbatch

# MFTIQ (dedicated venv; spatial-correlation-sampler must be compiled — on FAU
# use the helper, which pins torch 2.1.2+cu121 + system gcc 11.5 for nvcc):
bash benchmark/mftiq/setup_venv.sh
cd benchmark/methods/MFTIQ && bash download_model.sh && cd -
# then: sbatch benchmark/mftiq/benchmark_mftiq_a40.sbatch
```

(Every sbatch reads `BENCH_VENV`, defaulting to the shared `.venv` — MFTIQ
defaults to `benchmark/mftiq/.venv`.)

### TAPTRv3 (extra setup)

Its code lives in branch `v3` (already checked out) and it needs custom CUDA ops
compiled before the model will import:

```bash
cd benchmark/methods/TAPTR && git checkout v3            # already done
cd benchmark/methods/TAPTR/models/dino/ops && python setup.py install   # needs nvcc/GPU
```

The `benchmark/taptr/` harness is wired (resnet50 512×512). It is the highest-risk
adapter (DETR-style target-dict seeding, untested end-to-end) — smoke-test first:

```bash
python benchmark/taptr/taptr.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb
```

### TAPNext (extra setup / caveat)

Unlike every other baseline here, TAPNext cannot be constructed on CPU at all —
upstream `tapnet/tapnext/tapnext_torch.py::TRecViTBlock` hardcodes
`device='cuda'` for its recurrent blocks at `__init__` time, so there is no
`--cpu` path even for a smoke test; `build_adapter` raises a clear error instead
of the upstream `RuntimeError: No CUDA GPUs are available`. Smoke-test on a GPU
node first:

```bash
python benchmark/tapnext/tapnext.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb
```
