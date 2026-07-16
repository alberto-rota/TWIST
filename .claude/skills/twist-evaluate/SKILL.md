---
name: twist-evaluate
description: Run and interpret TWIST benchmark evaluation and compare against CoTracker/TAP baselines. Use for evaluate.py runs, any "how good is the model" question, quoting TAP-Vid/RoboTAP/STIR/EndoTAPP numbers, POR recovery metrics, or running baselines through benchmark/. Load BEFORE quoting or comparing any number.
---

# TWIST evaluation & baselines

One scorer for everything: `utilities/evaluation.py` (reuses
`models.metrics.tracking_metrics`, so eval `epe` is directly comparable to training
`val/epe`). Three entry points: standalone `python evaluate.py --run
<EXPERIMENT_NAME> [--wandb]` (rebuilds the model from the checkpoint's embedded
config), live `evaluate_and_report(...)`, and engine hooks `EVAL_AT_END` /
`EVAL_EVERY` (monitor). Baselines route through the SAME evaluator via
`benchmark/` — that is what makes our comparisons valid.

## Protocol laws (violate these and the number is meaningless)

1. **Queried-first** (`EVAL_QUERY_MODE: first`): each point queried at its first
   *visible* frame, scored strictly after it. Mandatory because readers store
   occluded GT coords as `(0,0)` — ~17% of DAVIS / ~24% of RoboTAP points are
   occluded at frame 0 and would otherwise be tracked from (0,0). Occluded frames
   are masked out of EPE/δ by GT visibility and only ever count as false positives
   in AJ, so the placeholder never corrupts metrics — only queries.
2. **Whole-video eval** (since 2026-06-30): `build_eval_dataset` forces
   clip_len=None. Numbers produced BEFORE that fix scored 24-frame windows and are
   NOT comparable — this includes snowy-sweep-1's "eval/MEAN EPE 11.19" and the
   "TWIST 11.2 vs CoTracker3 20.1" claim. Never mix the eras.
3. **Published-table comparisons need the canonical TAP-Vid geometry**:
   `RESIZE_MODE: stretch` at 256² (anisotropic squash, no crop). The internal
   default (cover@512, aspect-preserving + center-crop) discards ~44% of DAVIS
   width and cost CoTracker3 δ 0.849→0.58 in-harness. In-harness numbers are for
   ranking OUR runs and same-evaluator baselines only.
4. Training-time eval is a CAPPED monitor (`EVAL_MAX_CLIPS: 30`); final numbers
   come from uncapped `evaluate.py`. Files: `evaluation_stage0_ep{N+1}.csv` under
   the run dir (`ep1` = after the first epoch — near-untrained; expect huge EPE).

## Reading the CSV

Rows = datasets + MEAN; cols = EPE / Delta AVG / Average Jaccard / Occlusion
Accuracy / Time (ms/frame). Interpretation rules:

- **MEAN EPE is drift-dominated**: whole-video RoboTAP (~250 frames) can sit in the
  1000s of px early in training and swamps the mean. Rank runs on per-dataset
  delta_avg/AJ and on short-clip rows (DynamicReplica) + watch RoboTAP EPE as the
  long-horizon-drift proxy. Falling RoboTAP EPE = the prior is stabilizing.
- **STIR**: GT visible only at first+last frame. Scored `visible_only`
  (`EVAL_VISIBLE_ONLY: True` registry key — without it AJ/OA degenerate to
  ~1/(T-1)); δ over the official thresholds [4,8,16,32,64]px (`EVAL_THRESHOLDS`);
  AJ/OA excluded from the MEAN row (`EVAL_EXCLUDE_FROM_MEAN`). Internally
  comparable; the official leaderboard additionally needs native-res uncropped
  scoring (open TODO).
- **EndoTAPP_GT**: real human GT, but the prep must have been re-run by the user
  (`assets/dataprep/endotapp_gt_prep.py`, 250 seqs) — if `DATA/EndoTAPP/gt_tracks`
  is empty the dataset silently can't load. **SurgT is train-only** (CoTracker
  pseudo-labels; scoring it against CoTracker would be circular). TAPVID_KINETICS
  is scored offline only (too big for the in-training monitor).
- Time (ms/frame) is timed around the forward, warm-up excluded, single all-points
  pass. TWIST ~11–12 ms in-harness.

## Post-Occlusion Recovery (POR) — the thesis metric

`models.metrics.recovery_metrics`, on by default (`EVAL_RECOVERY`). Per occlusion
event: **snap-back** (first re-emergence frame) and **sustained** (visible run
after it, window `EVAL_RECOVERY_WINDOW`=8), each as EPE and δ, length-weighted
w(L)=L, plus a by-length curve. Case A (real data): visible frames only. Case B
(`HAS_OCCLUDED_GT`: KUBRIC/DYNAMICREPLICA/POINTODYSSEY): adds through-occlusion
`tho_epe/tho_delta` + drift-vs-frames-since-onset curve (rollout quality).
Artifacts: `recovery.csv`, `recovery_by_length.csv`, `recovery_drift.csv` +
`eval/<ds>/por_*` scalars. This is the metric no baseline optimizes — report it
prominently; expect the coarse re-acquisition ablation to move snap-back first.

## The bars

Published CoTracker3-on-Kubric (canonical protocol — compare only stretch@256
whole-video numbers): DAVIS AJ 74.0 / δ 84.9 (off), 71.1/81.9 (on); Kinetics
53.5/66.5; RGB-Stacking 63.3/76.2; RoboTAP δ 73.4 / OA 87.1; DynamicReplica
δ_vis 69.8 / δ_occ 41.8. Full table in CLAUDE.md.

In-harness (cover@512, whole-video, same evaluator — OUR apples-to-apples set),
DAVIS: TAPNext EPE 5.09 / δ .523 / AJ .396 / OA .936 @41ms; LiteTracker 6.95 /
.494 / .391 / .945 @12ms; MFT 7.71 / .529 / .373 / .917 @456ms. These are the
numbers TWIST must approach on DAVIS while winning occlusion/surgical/latency.

## Running baselines

`benchmark/<method>/{<method>.py,.yaml,sbatch}` wraps each predictor to the TWIST
contract `model(frames,queries)->{"coords","vis_logits"}` and calls
`benchmark/common.py::run` (W&B-tagged `benchmark`). Wired: cotracker, locotrack,
tapir, tapnext, litetracker, mft (+mftiq/trackon/chrono/taptr needing dedicated
venvs/compiles — see benchmark/README.md and the twist-baseline-benchmarks
memory for per-method quirks: TAPNext can't build on CPU, MFT needs BGR + clone-root
CWD, Chrono/TAPTR need `common.import_isolated` for the `models` namespace clash).
Zero-shot CoTracker3 on the surgical sets through this harness is the designated
baseline for the paper's surgical table.
