---
name: twist-diagnose-run
description: Diagnose TWIST training-run health from W&B or logs. Use when checking on a live or finished run, when metrics look flat/weird/collapsed, when deciding kill-vs-continue, or when comparing runs. Contains the metric map with healthy reference values, the collapse-signature catalog with historical run IDs, and the exact W&B query recipe.
---

# TWIST run diagnosis

## Pulling metrics (the recipe that works)

W&B entity/project = **twisteam/twist** (older runs may sit under `alberto_rota`).
From the repo root (login node, proxy required):

```bash
export $(grep -E "WANDB_API_KEY" .env | xargs)
export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
.venv/bin/python - <<'EOF'
import wandb
api = wandb.Api(timeout=60)
r = api.run("twisteam/twist/<RUN_ID>")          # or api.runs("twisteam/twist", order="-created_at")
print(r.name, r.state, dict(r.summary))
h = r.history(keys=["val/epoch/epe","val/epoch/gate_occ","_runtime"], pandas=False)
EOF
```

Key namespaces: `train/epoch/*` (incl. per-term `w_*` loss shares, `grad_norm`,
`gate`, `tf_prob`, `obs_kept`), `val/epoch/*`, `val/epoch/rollout/*` (prior-only
forecast), `eval/<dataset>/*` + `eval/MEAN/*` (benchmark monitor), `eval/*/por_*`.

**Sweep-run gotcha:** swept values live under FLAT DOTTED config keys —
`r.config["LOSS.USE_OCCLUDED_GT"]` — while `r.config["LOSS"]` shows the base block.
Reading the nested dict misidentifies the cell. Cheap fingerprint: a run with
`val/epoch/epe_occ` present in its summary trained with USE_OCCLUDED_GT=True
(the diagnostic only exists when occluded frames carry loss).

On-disk artifacts: `$RESULTS_DIR/<run>/` → `run_state.json`,
`stage{i}_{name}/{last,best}.pt` (best = lowest **val/epe**),
`evaluation_stage0_ep{N+1}.csv` + `recovery*.csv` (monitor eval fires at epoch 0
and every `EVAL_EVERY`; `ep1` = after the FIRST epoch — early, nearly-untrained
numbers; don't panic over them). SLURM logs: `~/job_outputs/`.

## Metric map (healthy vs alarm — uncapped Kubric, bs6 @512, the standard recipe)

| Metric | Meaning | Healthy | Alarm |
|---|---|---|---|
| `val/epoch/epe` | headline, best.pt selector (24f Kubric clips) | ep0 ~20 → ep3 ~15 → ep8 ~14 → ep14 ~10–13, still falling at ep19 | >25 @ep3; rising for 3+ epochs; oscillating ±10px |
| `val/epoch/delta_avg` | mean δ@{1,2,4,8,16} | ~0.45–0.52 by ep3, ~0.5+ later | falling while epe falls (inconsistency) |
| `delta_1px`/`2px` | tight precision | ~0.09 / 0.15–0.19 — the known wall, moves for no fine-loc knob | expecting them to move without an encoder-resolution lever |
| `gate_vis` vs `gate_occ` | Kalman gate split by GT vis — THE gate-health chart | separation: vis 0.6–1.0, occ falling (@ep3 refs: vis 0.58–0.70, occ 0.34–0.54) | both ≈1.0 (gate blind to occlusion, xrj5v46a end-state); both ≈0 (obs dead / riding prior); gate collapse to ~0.01 mid-run |
| `epe_occ` | EPE on occluded-but-supervised frames (only when USE_OCCLUDED_GT) | ~19–22 @ep3, trending down | absent when you expected occGT on; ≫ val/epe × 3 |
| `val/epoch/rollout/epe` | prior-only forecast (observe 4, coast) | falls monotonic: 49→37 over 15ep (with ROLLOUT_WEIGHT 2); ~26 @ep3 on current recipe | erratic 60–110+ (runaway); flat = prior not learning |
| `rollout/motion_ratio` | prior travel vs GT | →1.0 (0.9–1.4) | spikes 2–5 = velocity runaway |
| `motion_ratio` (corrected) | pred travel / GT travel, POOLED Σpred/Σgt | ~0.8–1.6 (inflated while tf_prob>0) | ≪1 (~0.1–0.2) = static collapse; note: pre-2026-07 logs used mean-of-ratios which blows up to ~40 on near-static clips — an ARTIFACT, not physics |
| `stuck_frac` | moving GT points the model leaves ~static | <0.05 | >0.3 with low motion_ratio = static collapse |
| `grad_norm` | | O(1) | spikes to e13 = destabilization (xrj5v46a); check `w_*` immediately |
| `train/epoch/w_*` | weighted loss shares | `w_pos` dominant, total loss positive | `w_unc` large-negative / total negative = Law-1 violation, kill |
| `coarse_gate` | global re-acquisition gate (COARSE runs) | low on clean tracking, spikes on re-acquisition | pinned high (coarse hijacking the window) |
| `kl_raw` / `train/kl` | pre/post-clamp KL | sits at free-bits floor — EXPECTED | treating it as a health signal at all |
| `eval/MEAN/epe` (monitor) | whole-video benchmark, 30-clip cap | dominated by long-video drift (RoboTAP can be 1000s of px early); watch it FALL | comparing it to pre-2026-06-30 chopped-clip numbers (e.g. snowy's 11.19) |

Timing reference: ~3.9–4.3 h/epoch (incl. ep-0 monitor eval) ⇒ a 24 h wall ≈ 5
epochs. Judge nothing before the curricula end: teacher forcing ep<3, obs-dropout
ramp ep<5.

## Collapse-signature catalog

1. **Static-query collapse** — run `37ek3go9`. Fingerprint: pred travel ≪ GT
   (2.9 vs 15.4 px), EPE ≈ GT motion magnitude, prior saturates tanh·max_step,
   gate ≈ 1, obs emits a constant counter-correction; loss negative from ep1
   (UNC-dominated). Root causes fixed (cost-volume obs, velocity prior, UNC 0);
   has not recurred (stuck_frac ~0.02). If it recurs: check MAX_CORR and UNC first.
2. **Loss-imbalance / uncertainty collapse** — run `xrj5v46a` (=woven-hill-75),
   ended val/epe 65, δ 0.155, gates pinned 1.0. Fingerprint: healthy → ep9+
   oscillation between regimes, grad_norm e13 spikes, gate →0.01, rollout 300–800px,
   total loss negative with `w_unc ≈ −45`, `w_pos ≈ 0.02`. Cause: HUBER_DELTA 0.02
   + UNC_WEIGHT 0.5. Fix: restore position dominance; features one at a time.
3. **Velocity runaway (prior)** — `w6x8eg6f`, and ctrl `8c5afg1k`. Fingerprint:
   val/epe fine, but rollout epe 60–112 erratic, rollout motion_ratio 2–4.8.
   Fix: ROLLOUT_WEIGHT 2 (validated by `lap32y0x`); VEL_DECAY only if it persists.
4. **Vis–gate coupling blowup** — pre-2026-06-20 design. Fingerprint: periodic
   epoch-scale collapses (OA→0.2, EPE→40) when BCE drives σ(vis)→0 and the obs
   shuts off globally. The gate is a separate head trained by position loss ONLY —
   never re-couple it to the vis logit.
5. **Infra losses masquerading as training failure**: blank EXPERIMENT_NAME +
   RESUME=scratch on a >24h job = each resubmit restarts from zero under a new W&B
   name (lost absurd-dust-35, both 640/768 res runs). A "finished" W&B run with an
   empty summary (e.g. `cgm7oaaj`) usually means it died before logging — check
   `~/job_outputs/twist_*_<jobid>.out`. Sweep-agent trials do NOT auto-resume
   across walls.

## Kill-or-continue rules

- Negative total loss, e13 grad spikes, or gate collapse to ~0 mid-run → **kill now**,
  fix balance, relaunch fresh (a collapsed run never recovered in this project).
- Ranking runs: compare cells at MATCHED epochs (nodes differ up to 2.6× in
  clips/s — wall-clock comparisons lie). A cell >2px val/epe behind its sibling at
  matched epoch and diverging → reallocate its GPU.
- Flat val/epe but falling rollout/epe (or vice versa) is fine — different branches
  saturate at different times; the corrected headline historically plateaus (~ep8–15)
  before the prior does.
- Don't kill for ugly `eval/MEAN/epe` alone — it's drift-dominated; look at
  per-dataset delta_avg/AJ and short-clip rows (DynamicReplica) first.

## Reference runs (name → what it proves)

| Run | Identity | Verdict |
|---|---|---|
| `fqbpeml7` snowy-sweep-1 | RAD24/MC0/RW2 cell | Stable-base provenance; old-protocol eval MEAN EPE 11.19/AJ 0.258 (NOT comparable to whole-video numbers) |
| `t7imej6u` eternal-sweep-2 | 1-GPU mixed-data run | val/epe 10.1 @ep14 then crashed; RAD=16 (inferior); historical yardstick only |
| `lap32y0x` vs `8c5afg1k` | rollout-loss A/B | ROLLOUT_WEIGHT=2 validated |
| `0kg2atga` (sweep) | fine-loc 2³ grid | RAD24/MC0/RW2 win; δ_1px hypothesis falsified |
| `xrj5v46a` woven-hill-75 | v2 all-on, 8-GPU DDP | THE collapse; Law 1 + Law 2 |
| `jewj2vlk` twist_v2_kubric_r512 | v2 all-on, 1-GPU | crashed @ep8 (infra) while still healthy (val 14.3) — distinct from the DDP collapse |
| `1k939vwu/6puudrhg/f2b1hydr/oqvm3asp` | supervision 2×2 cells | live — see the live-runs memory for the cell map |
| `ngdpapij` twist_best_kubric_r512 | 50ep stable-base long run | live; resumed a 07-02 checkpoint — watch its low-gate anomaly (gate_vis 0.26 @ep3 vs ~0.6 in sweeps) |
