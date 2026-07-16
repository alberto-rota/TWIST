# `SETUP_TRACKERS.md` — one-time install / weight / dataset commands

Copy-pasteable, ordered, **grouped per method**. These are the *heavy / network*
steps deliberately kept out of the SLURM jobs (dep installs, CUDA/C++ builds,
weight + dataset downloads). Run them yourself on a node with network + a GPU
where noted. Every command was cross-checked against the upstream README as of
2026-07; **differences from the task brief / older notes are flagged inline**.

> **Almost everything here is already integrated.** All 8 target trackers have a
> working `benchmark/<method>/` harness (see the summary table at the bottom and
> `benchmark/README.md`). What remains is the *heavy setup* below — mostly weight
> downloads and the two dedicated venvs (Track-On, MFTIQ).

## 0. Global prerequisites (do first)

```bash
cd /anvme/workspace/v120bb18-twist

# FAU proxy — REQUIRED for every download / pip / git on the compute+login nodes
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

module load python                       # the sbatch jobs do this too
source .venv/bin/activate                # shared TWIST venv (covers most methods)

mkdir -p weights                         # $WEIGHTS_DIR; one subdir per method below
```

Upstream sources are already cloned under `benchmark/methods/` (each keeps its own
`.git`, all gitignored). If any is missing, re-clone it into that exact path:

```bash
git -C benchmark/methods clone https://github.com/facebookresearch/co-tracker      co-tracker
git -C benchmark/methods clone https://github.com/cvlab-kaist/locotrack            locotrack
git -C benchmark/methods clone https://github.com/google-deepmind/tapnet           tapnet
git -C benchmark/methods clone https://github.com/gorkaydemir/track_on             track_on
git -C benchmark/methods clone https://github.com/Wardsy/Chrono                    Chrono   # ViT tracker
git -C benchmark/methods clone https://github.com/IDEA-Research/TAPTR              TAPTR
git -C benchmark/methods clone https://github.com/ImFusionGmbH/lite-tracker        lite-tracker
git -C benchmark/methods clone https://github.com/serycjon/MFT                     MFT
git -C benchmark/methods clone https://github.com/serycjon/MFTIQ                   MFTIQ
```

---

# A. Surgical / endoscopy methods (priority)

## A1. LiteTracker  — `benchmark/litetracker/`  ✅ integrated, GPU-verified

- **Env**: shared `.venv` (needs only `torch`/`einops`/`numpy`/`cv2`, all present).
  Upstream ships a `uv`-managed env (`uv sync`); **we bypass it** and reuse the
  shared venv, so *no install step*.
- **Weights**: NONE of its own. LiteTracker loads CoTracker3's **exact**
  `scaled_online.pth` (0 missing/0 unexpected) → reuses `weights/cotracker/`
  from A5. Do A5 (CoTracker3) first; nothing else here.

```bash
# nothing to install / download — just make sure A5's checkpoint exists:
ls weights/cotracker/scaled_online.pth
# smoke (GPU node):
python benchmark/litetracker/litetracker.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb
```

## A2. MFT  — `benchmark/mft/`  ✅ integrated, GPU-verified

- **Env**: shared `.venv` (adds `scipy`+`ipdb`, already in `pyproject.toml`).
- **Weights**: NONE to download — the RAFT-OU flow checkpoint **ships inside the
  clone** at `benchmark/methods/MFT/checkpoints/raft-things-sintel-kubric-*.pth`.
- **CUDA-only**: MFT's RAFT wrapper hardcodes `.cuda()`; no CPU smoke path.

```bash
ls benchmark/methods/MFT/checkpoints/*.pth        # confirm the bundled RAFT ckpt
python benchmark/mft/mft.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb   # GPU node
```

## A3. MFTIQ  — `benchmark/mftiq/`  ✅ integrated, ⚠️ untested end-to-end

MFT successor (decoupled learned occlusion/uncertainty = UOM). **Dedicated venv +
compiled `spatial-correlation-sampler` + downloaded backbones.** Heaviest setup.

- **torch/CUDA**: upstream pins `torch==2.0.1+cu117`; the FAU helper installs
  **`torch==2.1.2+cu121`** instead (this cluster only has CUDA 12.x + gcc 15.2,
  too new for nvcc), compiles `spatial-correlation-sampler==0.4.0` with system
  gcc 11.5, targets A40 `sm_86`. **Flag**: this torch/CUDA pin differs from the
  upstream README on purpose — required for FAU.

```bash
# 1) build the dedicated venv (network + a cuda module; ~10-20 min, compiles ext):
bash benchmark/mftiq/setup_venv.sh              # -> benchmark/mftiq/.venv

# 2) fetch UOM + flow backbones (RAFT / FlowFormer++ / NeuFlow) into the clone:
cd benchmark/methods/MFTIQ && bash download_model.sh && cd -
#   downloads (verified against the repo's download_model.sh):
#     checkpoints/UOM_bs4_200k.pth
#     checkpoints/flowformerpp-sintel.pth
#     checkpoints/raft-things-sintel-kubric-splitted-...-non-occluded-base-sintel.pth
#     src/MFTIQ/NeuFlow/neuflow_sintel.pth
#     src/MFTIQ/NeuFlow_v2/neuflow_mixed.pth

# 3) smoke (GPU node; the job defaults BENCH_VENV to benchmark/mftiq/.venv):
python benchmark/mftiq/mftiq.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb
```

---

# B. General-domain online TAP methods

## B1. CoTracker3 (online)  — `benchmark/cotracker/`  ✅ integrated

- **Env**: shared `.venv`.
- **Weights**: the **online** scaled checkpoint (causal sliding window). Auto-
  downloads on first run, or fetch manually:

```bash
mkdir -p weights/cotracker
wget -O weights/cotracker/scaled_online.pth \
  https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth
python benchmark/cotracker/cotracker.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb
```

## B2. Online / causal TAPIR  — `benchmark/tapir_online/`  ✅ NEW (this task), ⚠️ untested

The causal sibling of the offline `benchmark/tapir/` baseline. Uses the **Online
BootsTAPIR** checkpoint (`use_casual_conv=True`) and tracks frame-by-frame with a
carried causal context (the DeepMind `pytorch_live_demo.py` regime).

- **Env**: shared `.venv` (torch only; the tapnet torch path needs no JAX).
- **Weights**: the **causal** checkpoint — a *different file* from offline
  `bootstapir.pt`. Auto-downloads on first run, or fetch manually:

```bash
mkdir -p weights/tapir
# Online BootsTAPIR (PyTorch, CAUSAL). Verified against the tapnet README model table:
wget -O weights/tapir/causal_bootstapir.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/causal_bootstapir_checkpoint.pt
# smoke (GPU node). Default geometry is 256 (its trained res); 512 works but is heavier:
python benchmark/tapir_online/tapir_online.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb
```

> ⚠️ **Untested end-to-end** (no GPU on the login node). The adapter mirrors the
> official online loop (`get_query_features` → `construct_initial_causal_state` →
> per-frame `estimate_trajectories(..., causal_context=...)`); smoke-test the
> above before trusting numbers.

## B3. (offline) TAPIR / BootsTAPIR  — `benchmark/tapir/`  ✅ integrated

The offline whole-clip baseline (kept alongside B2 for the online-vs-offline
delta). Auto-downloads, or:

```bash
mkdir -p weights/tapir
wget -O weights/tapir/bootstapir.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt
```

## B4. TAPNext (BootsTAPNext)  — `benchmark/tapnext/`  ✅ integrated, ⚠️ GPU-only

Online/causal, next-token-prediction tracker. **Cannot even be constructed on
CPU** — upstream hardcodes `device='cuda'`; no `--cpu` smoke path.

```bash
mkdir -p weights/tapnext
wget -O weights/tapnext/bootstapnext_ckpt.npz \
  https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz
python benchmark/tapnext/tapnext.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb   # GPU node
```

## B5. Track-On / Track-On-R / Track-On2  — `benchmark/trackon/`  ✅ integrated

Online / causal memory tracker. Harness defaults to **Track-On-R** (Kubric +
real-world fine-tuned — the strongest checkpoint).

- **Env — dedicated venv.** ⚠️ **The upstream README changed**: it now prescribes
  a **conda/mamba** env (`pytorch==2.4.1` + `mmcv==2.2.0` from the openmmlab wheel
  index) rather than the plain `pip install -r requirements.txt` that
  `benchmark/README.md` still shows. Use the upstream-documented recipe:

```bash
# upstream-documented (README): conda/mamba
mamba create -n track_on_r python=3.12 && mamba activate track_on_r
mamba install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install -r benchmark/methods/track_on/requirements.txt
# then point the job at it:  BENCH_VENV=<that env's prefix> sbatch benchmark/trackon/benchmark_trackon_a40.sbatch
#   (if you prefer venv over conda: python -m venv .venv-trackon && source .../activate,
#    then the mmcv wheel + requirements above — same packages.)
```

- **Weights** (verified against the README model table):

```bash
mkdir -p weights/trackon
# Track-On-R (default in the harness):
wget -O weights/trackon/track_on_r.pt \
  "https://huggingface.co/gorkaydemir/track_on_r/resolve/main/track_on_r.pt?download=true"
# Optional — Track-On2 (Kubric-only). Same wrapper; set TRACKON.CHECKPOINT to this file:
wget -O weights/trackon/trackon2_dinov3_checkpoint.pt \
  "https://huggingface.co/gorkaydemir/track_on2/resolve/main/trackon2_dinov3_checkpoint.pt?download=true"
```

- **Gated backbone**: Track-On ships **no DINOv3 weights** (license). Request
  access to `facebook/dinov3-vits16plus-pretrain-lvd1689m` on HF and
  `huggingface-cli login` (or ensure `HF_TOKEN` can read it) — it auto-downloads
  on first run.

```bash
python benchmark/trackon/trackon.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb   # GPU node
```

---

# (Other general baselines already present — for completeness)

## LocoTrack  — `benchmark/locotrack/`  ✅ (shared `.venv`)
```bash
mkdir -p weights/locotrack
wget -O weights/locotrack/locotrack_base.ckpt \
  https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_base.ckpt
```

## Chrono  — `benchmark/chrono/`  ✅ (dedicated venv: `xformers`, `lightning`; Google-Drive weights)
```bash
python -m venv .venv-chrono && source .venv-chrono/bin/activate
pip install xformers lightning    # + the repo's requirements
pip install gdown && mkdir -p weights/chrono
gdown 1XYOr5pVncEAgyWcQZ_TjgvqLTcexdUQr -O weights/chrono/chrono_base.ckpt   # ViT-B
deactivate
```

## TAPTRv3  — `benchmark/taptr/`  ✅ (⚠️ compiled CUDA ops; branch `v3`; highest-risk)
```bash
cd benchmark/methods/TAPTR && git checkout v3                       # already checked out
cd benchmark/methods/TAPTR/models/dino/ops && python setup.py install   # needs nvcc/GPU
cd /anvme/workspace/v120bb18-twist
pip install gdown && mkdir -p weights/taptr
gdown 19iql2VTqGIeoyg_wt3JjpohszN5UE6s1 -O weights/taptr/TAPTRv3_resnet50_512x512.pth
python benchmark/taptr/taptr.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb   # GPU node
```

---

# C. Datasets (staging only — do NOT bulk-download blindly; large)

Every dataset is scored through the shared reader after a `*_data_prep.py` repack
(`assets/dataprep/`). Prep runs CPU/IO-only on the login node.

## C1. STIR — the STIR-challenge baseline data (headline surgical eval)

Two data sources + the official loader/metrics repos:

```bash
# --- loader + metrics (clone; used by prep + as the official protocol reference) ---
git -C benchmark/methods clone https://github.com/athaddius/STIRLoader   STIRLoader
git -C benchmark/methods clone https://github.com/athaddius/STIRMetrics  STIRMetrics
#   STIRMetrics scores TAP-Vid delta over pixel thresholds [4,8,16,32,64] (native res).

# --- STIR Challenge 2024 held-out test split (Zenodo record 14803158) ---
#   Public; the STIR_CHALLENGE registry entry expects it under DATA/STIRChallenge_2024/.
#   Browse/download the record's archive, then extract to DATA/STIRChallenge_2024/:
#     https://zenodo.org/records/14803158
mkdir -p DATA/STIRChallenge_2024
# wget/zenodo_get the archive(s) from the record above into DATA/STIRChallenge_2024/ and unpack.

# --- STIR original (STIROrig, the larger release) — IEEE DataPort, LOGIN-GATED ---
#   No anonymous wget: sign in to IEEE DataPort, accept terms, download manually:
#     https://ieee-dataport.org/open-access/stir-surgical-tattoos-infrared
#   Place under DATA/STIRFull/ (the STIR_FULL registry entry is currently commented out).

# --- repack to the shared layout (CPU/IO; login node) ---
python assets/dataprep/stir_data_prep.py       # see its --help for --src_root/--out_root
#   -> DATA/STIRChallenge_2024/gt_tracks/{index.json, <seq>/clip_00000.npz}
```

STIR eval knobs are already set in `dataset/wrappers.py::STIR_CHALLENGE`
(`EVAL_THRESHOLDS [4..64]`, `EVAL_VISIBLE_ONLY`, AJ/OA excluded from the mean —
GT exists only on start/end frames).

## C2. VL-SurgPT — ✅ already downloaded, prepped, and wired

VL-SurgPT ([arXiv:2511.12026](https://arxiv.org/abs/2511.12026), AAAI 2026;
project [szupc.github.io/VL-SurgPT](https://szupc.github.io/VL-SurgPT/)) is a
surgical tracking **dataset** (no method code released). It is **already staged**:
`DATA/VLsurgPT/{export_tissue_new,export_instrument_new}` downloaded, repacked by
`assets/dataprep/vlsurgpt_data_prep.py` into `DATA/VLsurgPT/gt_tracks/`
(**908 clips = 754 tissue + 154 instrument**, verified loadable), and registered
as `VLSURGPT` (`IS_EVAL_DATASET: True`) in `dataset/wrappers.py`. Nothing to run.

If you ever need to re-prep (e.g. re-download from the project page first):
```bash
python assets/dataprep/vlsurgpt_data_prep.py \
    --src_root DATA/VLsurgPT --out_root DATA/VLsurgPT/gt_tracks --resize 540 960
```
No STIR-style `EVAL_VISIBLE_ONLY` knob is needed: the prep stores **only the
annotated keyframes** as the clip's frames, so every stored frame carries genuine
GT visibility (unlike STIR's full-video/endpoint-only GT).

---

# Running a full benchmark (after setup)

```bash
# one method, all IS_EVAL_DATASET datasets -> W&B run + CSVs under results/<run-name>/
sbatch benchmark/<method>/benchmark_<method>_a40.sbatch
#   dedicated-venv methods:  BENCH_VENV=<prefix> sbatch benchmark/<method>/benchmark_<method>_a40.sbatch
```
