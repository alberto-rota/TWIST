---
name: twist-experiment-playbook
description: TWIST research strategy and experiment design. Use when deciding what to run next, designing or editing a training config, judging whether a result is good, planning ablations or sweeps, or asked about project goals, hypotheses, or how to beat CoTracker. Load BEFORE proposing any new experiment or config change.
---

# TWIST experiment playbook

The intellectual state of the project: what we're trying to prove, what is already
proven/falsified (with the run IDs that prove it), and the hard-won laws that every
new experiment must respect. Architecture/commands live in CLAUDE.md; this file is
the *judgment*.

## The thesis and the win condition

TWIST is an **online, causal** point tracker built as a state-space world model:
a frame-free transition prior (per-point GRU + inter-point attention, constant-velocity
+ bounded delta) corrected each frame by a cost-volume observation, fused by a
per-point Kalman gate. The bet: **explicit dynamics beat feed-forward matchers
exactly where matching fails — occlusion, re-acquisition, long-horizon drift.**

Winning is therefore measured on THREE axes, in priority order:

1. **Occlusion skill** (the differentiator): `epe_occ`↓, `gate_occ`↓ while `gate_vis`
   stays high, `val/rollout/*`↓, POR snap/sustained↑, through-occlusion `tho_epe`↓.
   No baseline trains for this; it is the paper's story.
2. **Surgical benchmarks** (the contribution): STIR_CHALLENGE / EndoTAPP_GT /
   SurgicalMotion vs **zero-shot CoTracker3 run through the same evaluator**.
   No published CoTracker numbers exist here — our comparison IS the result.
3. **TAP-Vid parity** (the credibility bar): CoTracker3-Kubric numbers
   (DAVIS AJ 74.0 / δ 84.9 offline; 71.1/81.9 online — full table in CLAUDE.md).
   Parity is necessary, not sufficient. TWIST's ms/frame (~11–12 in-harness) already
   beats TAPNext (41) and MFT (456); LiteTracker (12) is the latency rival.

Known character of the model: **robust but imprecise**. It rarely diverges
catastrophically on short clips (the prior anchors it) but tight thresholds are the
persistent wall (δ_1px ≈ 0.09, δ_2px ≈ 0.15–0.19 across EVERY config tried), and on
**whole-video** eval the prior can run away over hundreds of frames (RoboTAP EPE in
the thousands early in training). Robustness is bankable; precision and long-horizon
stability are the open fronts.

## The Laws (violations have each destroyed at least one run)

1. **Position dominance.** The direct Huber position loss must dominate the total
   gradient. `w_pos` must be the largest logged `w_*` share and total loss must stay
   positive. Proof: run `xrj5v46a` (= W&B `woven-hill-75`) collapsed because
   HUBER_DELTA 0.02 shrank the position gradient ~10× (Huber grad ∝ delta in the L1
   regime) while UNC_WEIGHT 0.5 made `w_unc ≈ −45` dominate a negative total loss.
   Corollary: if you ever shrink HUBER_DELTA, rescale POS_WEIGHT up or auxiliaries
   down in the same change — never ship the shrink alone with new auxiliaries.
2. **One change at a time.** `xrj5v46a` stacked all v2 features at once and lost to a
   14-epoch 1-GPU run. The repo's own discipline (sweep_supervision 2×2, the three
   `config/ablation_*.yaml`) exists because of this. Every new feature rides on the
   stable base (`config/train_best.yaml`) alone before it is combined.
3. **Comparable numbers require the canonical protocol.** Published-table comparisons
   need `RESIZE_MODE: stretch` @ 256², TAP-Vid "queried first", whole videos. The
   internal harness (cover@512) mutilates DAVIS (CoTracker3 δ 0.849 published → 0.58
   in-harness). In-harness numbers rank OUR runs against each other and against
   baselines run through the SAME evaluator — never against published tables.
4. **Eval-protocol epochs.** Numbers produced before 2026-06-30 used 24-frame chopped
   eval clips (the `build_eval_dataset` bug); post-fix eval is whole-video. The two
   eras are NOT comparable: snowy-sweep-1's "eval/MEAN EPE 11.19" is old-protocol;
   the same model family scores MEAN EPE in the hundreds under whole-video eval
   (RoboTAP drift dominates). Check the date/protocol before comparing any two numbers.
5. **Run identity is EXPERIMENT_NAME.** Fixed name + `RESUME: last` for anything that
   outlives a 24h wall; blank name + `scratch` forks a fresh run each submit (lost
   absurd-dust-35 and both 640/768 resolution runs). Architecture changed ⇒ fresh
   name, never resume (checkpoint load is `strict=False` — a mismatched resume
   *silently* drops weights).
6. **KL is dead.** KL always pins at the free-bits floor (zero gradient) because both
   prior and posterior are directly GT-supervised — it cannot discriminate healthy
   agreement from collapse and is not a lever. train_best carries KL_WEIGHT 0.05
   harmlessly (floor ⇒ no gradient); don't tune it, don't expect anything from it,
   don't resurrect ELBO reasoning. `kl_raw` is a passive diagnostic only.
7. **The gate learns occlusion only if occluded frames carry loss.** `USE_OCCLUDED_GT`
   (default **True** in `models/losses.py` since commit 202cf56) routes position loss
   through pos_valid (Kubric stores real coords on its ~28% occluded frames). Without
   it the gate has no gradient at occluded frames and stays ≈1 everywhere. Also: the
   vis logit must NEVER double as the gate (separate `gate_head`; re-coupling caused
   the 2026-06-20 OA→0.2 blowups).
8. **val ≠ eval.** `val/epe` is 24-frame Kubric clips; benchmark eval is whole videos.
   A run can look healthy on val and be diverging on long videos. Watch both; the
   eval MEAN EPE is dominated by long-video drift (RoboTAP, 250+ frames), so rank
   runs on per-dataset EPE and delta_avg/AJ, not MEAN EPE alone.

## Hypothesis ledger

**VALIDATED (keep; do not re-litigate):**
- Multi-step rollout loss `ROLLOUT_WEIGHT: 2` — A/B `lap32y0x` (ON) vs `8c5afg1k`
  (ctrl): rollout EPE 49→37 monotonic vs erratic 60–112, motion_ratio →1.0 vs spikes
  2–4.8, no cost to corrected EPE. The single clearest win in the project.
- Fine-loc levers settled by sweep `0kg2atga`: RADIUS_PX **24** (16 is worse, −0.027 AJ),
  MAX_CORR **0**, ROLLOUT_WEIGHT 2 > 4. Do not re-grid.
- Cost-volume soft-argmax observation + constant-velocity prior + neutral separate
  gate head (the post-37ek3go9 redesign) — static collapse never recurred
  (stuck_frac ~0.02 since).
- Scheduled sampling (tf_prob anneal), per-point occlusion-shaped obs-dropout
  (contiguous spans, OBS_DROPOUT 0.3 / SPAN [3,8]), on-device encoder preprocess,
  whole-video eval, STIR visible_only scoring, POR metric suite.
- Through-occlusion supervision (F1) is *plausibly* positive and now default-on —
  early 2×2 read supports it (see live-runs memory), final verdict pending.

**FALSIFIED (do not retry):**
- Fine-localization knobs as the δ_1px bottleneck (0kg2atga: δ_1px 0.09 flat across
  all 8 cells). Suspected true bottleneck: DINOv3 patch-16 feature stride (32×32 grid
  @512) — the untested levers are a learned feature upsampler, mid-layer features,
  or finer effective grid.
- KL/free-bits tuning as a dynamics lever (30fxhk0n, w6x8eg6f: floor-pinned always).
- ROLLOUT_VEL_DECAY < 1 as a default (workshop 21: hurts constant-velocity motion;
  reserve for a demonstrated motion_ratio≫1 runaway only).
- HUBER_DELTA 0.02 stacked with UNC_WEIGHT 0.5 (xrj5v46a collapse — Law 1).
- Whole-frame i.i.d. obs dropout as occlusion training (trains 1-step coasting only;
  the per-point contiguous-span version is the correct regime).
- Resuming v1 checkpoints into v2 architectures (silent strict=False drops).

**OPEN (the live frontier, 2026-07-07):**
- Supervision 2×2 (F1 × F4a) — RUNNING; cells mapped in the live-runs memory.
  Early read @ep3: δ=0.2 beats δ=0.02 on val/epe (14.9/15.2 vs 16.1/18.2), occGT
  slightly ahead at fixed δ, F4a-alone (deft-4) worst incl. rollout 41 vs 26.
  Judge at matched epochs; cells will hit the 24h wall around ep5.
- Ablations queued (jobs 3817243–45): `abl_ce_r512` (+local corr-CE 0.5),
  `abl_coarse_r512` (+COARSE re-acquisition), `abl_redetect_r512` (COARSE + CE 0.5 +
  GLOBAL_CE 0.25) — each ONE block on the stable base, 20ep, fresh scratch runs.
- UNC_WEIGHT reintroduction: currently 0. If tried again: tiny (≤0.05), with warmup,
  watching `w_unc` share; it trains only the logvar heads (Huber is detached inside
  the NLL) so it is never worth destabilizing the run for.
- Long-horizon stability: CLIP_LEN 48, length curriculum, and/or the coarse stage as
  eval-time re-anchoring — targets the whole-video RoboTAP drift that now dominates
  eval MEAN EPE. Likely the biggest single eval-number win available.
- Sub-2px precision: feature upsampler / mid-layer DINOv3 features (deferred lever).
- Surgical fine-tune stage (Cholec80/EndoTAPP/SurgT pseudo-labels) + zero-shot
  CoTracker3 surgical baseline; STIR native-res scoring for the official leaderboard.

## Config recipes

- **The stable base is `config/train_best.yaml`** = snowy-sweep-1 (`fqbpeml7`)
  verbatim: RAD 24 / MC 0 / K 9, HUBER_DELTA 0.2, POS 10 / PRIOR 0.5 / VIS 0.5 /
  ROLLOUT 2 / UNC 0 / CE 0, LR 3e-4 cosine, warmup 3, TF 3, OBS_DROPOUT 0.3 over 5,
  bs6 @512 (~75 GB on A100-80), Kubric-only uncapped, eval sets held out. Every
  ablation copies this file and flips ONE block (see `config/ablation_*.yaml` for
  the pattern, including the provenance header — keep writing those headers).
- **Loss-balance pre-flight** (before ANY launch that touches LOSS or adds a head):
  eyeball expected `w_*` shares — `w_pos` largest, total positive, new auxiliary
  ≤ ~20% of `w_pos`. After 1–2 epochs, verify in W&B (`train/epoch/w_*`). A negative
  total loss is an automatic kill-and-rebalance.
- 20-epoch runs RANK levers; only 50-epoch runs on the merged winner produce
  quotable numbers (runs were still improving at ep19 in every sweep so far).
- When a ranking result is in: merge winning flags into train_best.yaml (new
  EXPERIMENT_NAME), launch 50ep, and only then run the canonical-protocol eval.

## How to decide "is this result good?"

1. Same protocol era? (Law 4) Same epochs? Undertrained comparisons lie.
2. Better than the sibling cell at matched epoch on val/epe AND not worse on
   val/rollout/epe and gate separation?
3. Eval: per-dataset — DAVIS/RGB-Stacking delta_avg & AJ up? RoboTAP EPE (drift
   proxy) down? Surgical rows up? POR snap/sustained up?
4. Nothing pathological: loss positive, grad_norm sane, gate separated, motion_ratio
   ~1, stuck_frac <0.05? (Fingerprints: twist-diagnose-run skill.)
5. Only then: merge the lever, long-run it, and re-baseline.
