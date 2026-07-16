# `benchmark/` integration convention (inferred from the code)

This is the contract every baseline tracker in `benchmark/` conforms to, inferred
by reading `common.py`, `README.md`, and the existing harnesses
(`cotracker/`, `tapir/`, `tapnext/`, `trackon/`, `litetracker/`, `mft/`,
`mftiq/`, `locotrack/`, `chrono/`, `taptr/`). **Anything you add must match it.**
It exists so every method is scored by the *exact same* evaluator the TWIST model
uses (`utilities.evaluation.evaluate_and_report`) — same datasets, same TAP-Vid
"queried first" protocol, same metric definitions (`models.metrics`), same
CSV + W&B table + Post-Occlusion-Recovery artifacts — making the numbers directly
comparable.

## 1. The forward contract (the one thing every adapter must satisfy)

A method is wrapped as an `nn.Module` whose `forward` matches the TWIST world
model:

```python
model(frames, queries, point_mask=None) -> {"coords":     (B, T, N, 2),   # pixel xy
                                             "vis_logits":  (B, T, N)}       # sigmoid>0.5 = visible
```

- **`frames`**: `(B, T, 3, H, W)`, `uint8` in `[0,255]` — or float `[0,1]` if a
  config set `FRAMES_AS_FLOAT`. Use `common.frames_to_255_float(frames)` to
  normalize to float `[0,255]` regardless.
- **`queries`**: `(B, N, 3) = (t, x, y)` in **pixels** (t = query frame index,
  x/y in the frame given at the eval `TARGET_SIZE`).
- **`point_mask`**: `(B, N)` or `None` — accept it and **ignore** it; the
  evaluator masks padded / non-evaluated points itself.
- **coords** are **xy pixel** coordinates at the *input* `(H, W)` resolution
  (feed frames at the eval size and predictions line up with GT directly — no
  rescale unless the model resizes internally, in which case rescale back).
- **vis_logits** are real-valued; the evaluator thresholds `sigmoid(logit) > 0.5`.
  If the method makes a hard visible/occluded decision, emit `±10` via
  `common.vis_bool_to_logits(vis_bool, coords_like)`.

### `supports_query_times` (class attribute) — the single most important flag

Set it on the adapter class; the evaluator reads it to decide how to drive the
model under "queried first":

- **`True`** (default when omitted): the model can seed **per-point query times
  in one forward** (a point due later is embedded as "unknown" until its query
  frame). The evaluator scores the whole clip in **one pass**. Used by
  CoTracker, TAPIR-offline, TAPNext, LiteTracker, LocoTrack, Track-On.
- **`False`**: the model tracks **forward-only from a single template/query
  frame** (dense flow chaining, causal state). The evaluator groups a clip's
  points by first-visible frame and calls the adapter **once per distinct query
  frame** `f` — every point in that call shares `f`; you `init` at `f` and roll
  forward to the clip end. Fill frames `≤ f` with a placeholder (never scored).
  Used by **MFT, MFTIQ, and the new Online TAPIR**.

## 2. Three-file harness layout (per method, all tracked in git)

```
benchmark/<method>/
    <method>.py                    # adapter + build_adapter(); calls common.run(...)
    <method>.yaml                  # W&B-sweep-format config (geometry, model, eval protocol)
    benchmark_<method>_a40.sbatch  # SLURM launcher (reads BENCH_VENV, runs the .py)
```

The upstream source clone lives under `benchmark/methods/<upstream-name>/`
(gitignored, keeps its own `.git`); the harness never edits it.

### `<method>.py` structure (copy an existing one)

```python
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))   # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))   # <repo>/benchmark for `import common`
import common
REPO_ROOT = common.setup_method_paths(__file__, "<upstream-subdir>")  # add benchmark/methods/<subdir>

class MethodAdapter(torch.nn.Module):
    supports_query_times = True | False
    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_): ...

def build_adapter(cfg, device) -> MethodAdapter:
    # construct upstream predictor, load/download weights, wrap, .to(device).eval()
    ...

if __name__ == "__main__":
    sys.exit(common.run(build_adapter,
        default_config=DEFAULT_CONFIG,
        description="...",
        checkpoint_key="<METHOD>.CHECKPOINT",   # or None if weights load differently
        default_name="<method>"))
```

`build_adapter(cfg, device) -> nn.Module` is the **only** per-method logic.
`common.run(...)` owns everything shared: CLI parsing, config load + overrides,
geometry logging, device selection, the W&B run (named + tagged `benchmark`), and
the `evaluate_and_report(...)` call.

### `sys.path` isolation helpers (`common.py`)

- `common.setup_method_paths(__file__, *src_subdirs)` — puts repo root + the
  upstream source dir(s) on `sys.path`, and **drops the script's own dir** so
  e.g. `cotracker.py` cannot shadow the upstream `cotracker` package. Use this
  for methods whose top-level module names **don't** collide with TWIST's.
- `common.import_isolated("<subdir>", *shadow_names)` — a context manager for
  methods that ship top-level packages colliding with TWIST's (`models`,
  `dataset`, `utils`, `main`, …). Inside the `with`, the method dir is first on
  `sys.path`, repo root is removed, and the colliding names are popped from
  `sys.modules` so the method imports cleanly; on exit TWIST's modules are
  restored while your already-imported objects keep their bindings. Used by
  Track-On (`dataset`/`model`/`utils`), Chrono (`models`/`model_utils`),
  TAPTR (`models`/`main`).

## 3. Coordinate / tensor helpers (`common.py`) — reuse these, don't re-derive

| Helper | Maps |
|---|---|
| `frames_to_255_float(frames)` | reader frames → float `[0,255]` |
| `frames_to_bthwc_norm(frames)` | reader `(B,T,3,H,W)` → TAPIR-style `(B,T,H,W,3)` in `[-1,1]` |
| `queries_txy_to_tyx(queries)` | TWIST `(t,x,y)` → TAPIR `(t,y,x)` |
| `vis_bool_to_logits(vis, coords_like, logit=10.)` | bool visibility → hard `±logit` |
| `tapir_outputs_to_twist(tracks_bnt2, occ_logits, expected_dist=None)` | TAPIR-family output (`tracks (B,N,T,2)` xy + occlusion/dist logits) → the TWIST dict, incl. the `(1-σ(occ))(1-σ(dist))>0.5` visibility rule |

## 4. Config (`<method>.yaml`) — W&B-sweep format, like every config in the repo

`parameters: {KEY: {value: ...}}`, flattened + loaded into a `DotMap` by
`utilities.config.load_and_process_config`. Conventional keys:

- **Run/W&B**: `EXPERIMENT_NAME` (= W&B run name + `$RESULTS_DIR/<name>/` dir),
  `WANDB_PROJECT: twist`, `WANDB_ENTITY: twisteam`, `WANDB_TAGS: [benchmark]`.
- **Geometry**: `IMAGE_SIZE` (square frame side → dataset `TARGET_SIZE`),
  `CROP` (native-pixel `x0,y0,x1,y1` before resize, or `null`).
- **Model block**: one namespaced block `METHOD: {value: {...}}` holding
  `CHECKPOINT` (relative to repo root or `$WEIGHTS_DIR`), an optional
  `CHECKPOINT_URL` (auto-downloaded + cached on first run if the file is absent),
  and any construction kwargs (`MODEL_KWARGS`, `VARIANT`, …).
- **Eval protocol**: `EVAL_QUERY_MODE: first`, `EVAL_AMP`, `EVAL_BATCH_SIZE`,
  `EVAL_WORKERS`, `EVAL_RECOVERY(+_WINDOW)`, `EVAL_TIMING_BATCHES`.

Shared CLI flags (from `common.build_parser`, override the config):
`--config --checkpoint --datasets --max-clips --batch-size --image-size --crop
--workers --max-steps --query-mode --name --tags --out-dir --no-wandb --amp
--cpu`. Unknown `--KEY=val` / `--A.B.C=val` args pass straight through as
type-coerced config overrides.

## 5. Weights layout

All weights live under `$WEIGHTS_DIR` (defaults to `weights/`), one subdir per
method: `weights/<method>/<file>`. Two loading patterns:

1. **Auto-download**: config carries `CHECKPOINT` + `CHECKPOINT_URL`; the harness
   fetches to the cache path on first run if absent (CoTracker, TAPIR, TAPNext,
   LocoTrack, LiteTracker→CoTracker's ckpt, Online TAPIR).
2. **Manual-only**: gated / Google-Drive / bundled weights (Track-On DINOv3,
   Chrono, TAPTR, MFT-in-clone, MFTIQ `download_model.sh`). `build_adapter` raises
   a clear `FileNotFoundError` pointing at `SETUP_TRACKERS.md`.

Mark any not-yet-present weight in the adapter with a `# TODO(weights): ...`
comment and a `FileNotFoundError` that names the file + points at SETUP.

## 6. Environments (`BENCH_VENV`)

Every `.sbatch` reads `BENCH_VENV` (default = the shared TWIST `.venv`). Methods
whose deps are compatible with the shared env (CoTracker, LocoTrack, TAPIR,
TAPNext, LiteTracker, MFT, Online TAPIR) run on it directly. Methods with
heavier/conflicting pins get a **dedicated venv** the job points to:
Track-On (`mmcv`/`timm`/`transformers` + gated DINOv3), Chrono (`xformers`,
`lightning`), MFTIQ (`torch==2.1.2` + `xformers`/`kornia`/compiled
`spatial-correlation-sampler`; use `benchmark/mftiq/setup_venv.sh`). TAPTR needs
custom CUDA ops compiled. See `SETUP_TRACKERS.md` for exact commands + constraints.

## 7. How a NEW method is added (checklist)

1. Clone upstream → `benchmark/methods/<name>/` (gitignored).
2. `benchmark/<method>/<method>.py` — adapter to the §1 contract + `build_adapter`,
   ending in `common.run(...)`. Choose `supports_query_times` honestly (§1).
   Reuse §2/§3 helpers; isolate imports if names collide.
3. `benchmark/<method>/<method>.yaml` — copy the closest existing config; set the
   model block + geometry + eval protocol.
4. `benchmark/<method>/benchmark_<method>_a40.sbatch` — copy an existing one; only
   the final `python benchmark/<method>/<method>.py` line changes (+ `BENCH_VENV`
   note if it needs a dedicated env).
5. Add setup commands to `SETUP_TRACKERS.md` (weights, deps, compilation caveats),
   **verified against the upstream README**.
6. Add a row to `benchmark/README.md`'s status table.
7. Smoke-test on a GPU node:
   `python benchmark/<method>/<method>.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb`.

## 8. Eval datasets (for reference — not a `benchmark/` concern)

Which datasets a run scores is **not** set here — it's the registry
`dataset/wrappers.py::DATASET_DEFAULTS`. A dataset is scored by every benchmark
harness iff its entry flags `IS_EVAL_DATASET: True` (TAP-Vid DAVIS / RGB-Stacking /
Kinetics, RoboTAP, and the surgical sets EndoTAPP-GT, VL-SurgPT, STIR-Challenge,
SurgicalMotion). All datasets go through the single shared reader
`dataset.cotracker.CoTrackerTracksDataset` over an `index.json` + per-clip `.npz`
layout; a `*_data_prep.py` script under `assets/dataprep/` repacks each raw
dataset into that layout. Sparse-GT surgical sets that annotate only endpoint
frames of the *full* video (STIR) set `EVAL_VISIBLE_ONLY` + exclude AJ/OA from the
cross-dataset mean; sets that store only annotated **keyframes** as the clip's
frames (EndoTAPP-GT, VL-SurgPT) need no such knob because every stored frame
carries genuine GT visibility.
