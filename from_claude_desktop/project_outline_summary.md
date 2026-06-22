# Dense Point Tracking as a State-Space World Model — Project Outline & Key Concepts

**Owner:** Alberto Rota (Politecnico di Milano) · **Hardware budget:** 1× NVIDIA A40 (48 GB) · **Last updated:** 2026-06-17

---

## 1. One-Paragraph Summary

The project reframes **dense tissue-point tracking** in surgical video as a **passive, observation-corrected state-space world model** rather than a correspondence-chaining problem. The tracked points are the model's *state*; each newly observed frame is the *exogenous input* that drives a state transition. The model separates a **transition prior** (learned tissue dynamics that can run frame-free, e.g. through occlusion) from an **observation update** (correction from the new frame). This yields capabilities standard trackers lack: occlusion handling by dynamics rollout, future-trajectory prediction, and counterfactual simulation. The defensible novelty is the *intersection* — passive observation-corrected filtering over real tracked tissue points, with a separable learned tissue-dynamics prior, in the surgical domain.

---

## 2. Motivation: Why a State-Space World Model over Standard Tracking

Standard dense trackers (TAPIR, CoTracker) are **correspondence chains** with no explicit persistent state and no dynamics prior. They handle occlusion by classification and re-matching, never by predicting where a point *should* go. The state-space formulation instead treats tracking as a differentiable Bayesian filter, which:

- **Handles occlusion by rollout** — the transition prior carries occluded points forward from state alone (instruments, smoke, blood).
- **Predicts future trajectories** — the dynamics prior runs without observations.
- **Enables counterfactual simulation** — perturb one coordinate and propagate forward.
- **Admits physical priors** — tissue elasticity / incompressibility can be injected directly into the transition model.

No existing surgical method frames tracking this way, which is the opening this project targets.

---

## 3. State-Space Framing

- **State** $s_t$ — set of $N$ point coordinates `(N, 2)` or `(N, 3)`, the tracked tissue points.
- **"Action"** $a_t$ — the next observed frame $I_{t+1}$, the exogenous signal driving the transition.
- **World model** — $s_{t+1} = f(s_t, I_{t+1})$.

Two separable components:

- **Transition model** $p(s_{t+1} \mid s_t)$ — tissue-dynamics prior; runs *without* observations, handling occlusion by rollout.
- **Observation model** $p(s_{t+1} \mid s_t, I_{t+1})$ — corrects coordinates from the new frame.

---

## 4. Proposed Architecture (RSSM-inspired)

Frozen **EndoFM / DINOv2** frame encoder → coordinate tokens **cross-attend** to frame features (observation model) → lightweight **transformer/GRU dynamics prior** (transition model) → **MLP decoder** to coordinate displacement $\Delta p$.

**Training:** coordinate-prediction loss + **KL(dynamics-only ‖ observation-updated)** to force a useful dynamics prior.

A pure-tracking sketch (closed-loop recurrent particle filter with transition + observation update + learned visibility head, tensor-shape annotated) lives in `surgical_tap_filter_sketch.py`; the architecture/supervision diagram is in `dense_tracking_world_model_diagram.svg`.

---

## 5. Novelty Assessment (deep lit review, 2026-06-15) — defensible, but position carefully

The raw ingredients are **not** novel on their own:

- Transition + observation decomposition trained end-to-end = textbook (RSSM/Dreamer, Deep Kalman Filters, Differentiable Particle Filters). **Do not claim as novel.**
- "Next observation as the driving input / passive action-free world model" has classical precedent (observation-driven SSM, exogenous-input filters). **Reframing, not reinvention.**
- Particles/keypoints as world-model state with rollout already exists — **Latent Particle World Models (LPWM, ICLR 2026 Oral, 2603.04553)**, but it *abolishes* tracking (no real observed points). **What Happens Next (ICLR 2026, 2509.21592)** forecasts dense point trajectories open-loop from one image and argues tracks > pixels, but has **no observation/correction term**. **TAPNext (ICCV 2025, 2504.05579)** is the only mainstream tracker with a real recurrent SSM state, but not a separable frame-free dynamics prior.
- **Surgical domain: no one frames tracking as a state-space WM.** Surgical WMs (SurgSora, Surgical Vision WM, SurgVeo, SurgWorld) are pixel/latent generators. Surgical trackers (SuPer family, Endo-TTAP 2503.22394, EndoTracker MICCAI 2025, STIR/SurgT baselines) handle occlusion by classification + re-matching, never by dynamics rollout. **Endo-TTAP is the key in-domain baseline to beat.**

**Defensible novelty = the intersection:** passive, observation-corrected filtering over *real* tracked tissue points + a separable learned tissue-dynamics transition prior (frame-free occlusion rollout) + frame-as-observation + counterfactual rollout, in the surgical domain. Related work must explicitly differentiate from LPWM, What Happens Next, TAPNext, and Endo-TTAP.

---

## 6. Benchmarks & Datasets

The data strategy spans out-of-domain synthetic pretraining (dense supervision), surgical self-supervised adaptation (abundant in-vivo video, no labels), optional 3D grounding, and held-out surgical evaluation. No single surgical dataset offers dense per-frame point-track supervision, so the plan pretrains the filter where dense ground truth exists, adapts it unsupervised on surgical video, then evaluates on sparse-label surgical benchmarks.

### Dataset descriptions

- **Kubric / MOVi (TAP-Vid-Kubric)** [synthetic, out-of-domain]. About 11k procedurally generated videos of rigid objects, with pixel-perfect dense point tracks, depth, flow, and visibility. Standard pretraining source for trackers. Provides the dense, fully observed supervision needed to train the observation and transition modules together and to validate the filter end to end.
- **PointOdyssey** [synthetic, out-of-domain]. Large-scale long-horizon synthetic tracks (sequences of roughly 1k to 4k frames) with dense ground truth. Best source for pretraining the frame-free dynamics prior and stress-testing long-range rollout and occlusion handling.
- **Hamlyn** [real, laparoscopy, stereo]. Large volume of in-vivo stereo surgical video with tissue deformation and no tracking labels. Primary self-supervised surgical adaptation set: photometric and cycle-consistency objectives adapt the filter to surgical appearance and tissue dynamics. Stereo also supports a depth-consistency signal.
- **SurgT** [real, laparoscopy, stereo; MICCAI 2022 challenge]. 157 stereo endoscopic videos from 20 clinical cases with stereo calibration; designed explicitly to encourage unsupervised methods (no annotated training data). Test set has hidden bounding-box tracking labels. Used both for unsupervised surgical training and as a held-out tracking benchmark.
- **SCARED** [real, porcine, stereo; EndoVis 2019]. da Vinci Xi captures with structured-light dense depth ground truth (7 train + 2 test, 1280x1024). The most widely used endoscopic depth benchmark. Optional phase: grounds a 3D `(N, 3)` state, validates lifting 2D tracks to 3D, and supports geometric (incompressibility) priors.
- **STIR (Surgical Tattoos in Infrared)** [real, in-vivo + ex-vivo, stereo; IEEE TMI 2024, 2024 STIR Challenge]. Tissue points tattooed with IR-fluorescent ICG dye, persistent but invisible to visible-spectrum algorithms; hundreds of stereo clips with start and end frames labelled (over 3,000 points). The headline surgical point-tracking benchmark, evaluated by 2D/3D endpoint error and accuracy. This is where the method is measured against the in-domain baseline.

### Which dataset in which phase

| Phase | Goal | Datasets | Supervision | Notes |
|---|---|---|---|---|
| 1 — Pretrain filter | Train observation + transition end to end; sanity-check shapes and the filter loop | Kubric / MOVi, TAP-Vid (DAVIS, Kinetics) for eval | Dense GT tracks | Fully observed; isolates architecture bugs before any domain shift |
| 2 — Pretrain dynamics prior | Long-horizon rollout, occlusion handling, KL(dynamics ‖ observation-updated) | PointOdyssey | Dense GT, long sequences | Forces a useful frame-free transition prior |
| 3 — Surgical adaptation | Adapt appearance + tissue dynamics to surgery, unsupervised | Hamlyn (primary), SurgT (unsupervised) | Self-supervised (photometric, cycle, stereo) | No tracking labels; matches SurgT's unsupervised design intent |
| 4 — 3D grounding (optional) | Lift state to `(N, 3)`, validate depth / incompressibility priors | SCARED, Hamlyn stereo | Structured-light depth GT | Only if pursuing the 3D-state variant |
| 5 — Evaluation | Headline point-tracking metrics vs. baselines | STIR (primary), SurgT held-out test | Sparse start/end GT (STIR), bbox GT (SurgT) | Report STIR 2D/3D endpoint error; TAP-Vid as out-of-domain generalization check |

**Key in-domain baseline to beat:** Endo-TTAP (2503.22394), benchmarked on STIR.

---

## 7. Key Papers

**Closest prior art / positioning anchors:** LPWM (2603.04553, ICLR 2026 Oral — particles as WM state, *abolishes* tracking) · What Happens Next (2509.21592, ICLR 2026 — open-loop dense trajectory forecasting, *no correction term*) · TAPNext (2504.05579, ICCV 2025 — recurrent SSM tracker, *not a separable dynamics prior*).

**In-domain trackers / baselines:** Endo-TTAP (2503.22394, *baseline to beat*) · EndoTracker (MICCAI 2025) · SuPer family · STIR / SurgT challenge baselines.

**Backbone / representation:** EndoFM (DINO, 33K endoscopic clips) · DINOv2.

**Method lineage (cite, do not claim):** RSSM / Dreamer · Deep Kalman Filters · Differentiable Particle Filters (Jonschkowski, RSS 2018).

---

## 8. Project Assets (in this folder)

- `surgical_tap_filter_sketch.py` — closed-loop recurrent particle-filter tracking sketch (transition + observation + visibility head; tensor-shape annotated).
- `dense_tracking_world_model_diagram.svg` — architecture / supervision diagram.
- `LPWM.pdf`, `WHN.pdf` — key prior-art papers for novelty positioning.
