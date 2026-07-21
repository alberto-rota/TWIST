"""TWIST training engine.

The piece that actually trains a stage: it takes the model + loss + dataloaders
that ``main.run_pipeline`` builds for one schedule stage and runs the epoch loop,
then checkpoints into the run dir and marks the stage complete so a resumed run
continues at the next stage (loading these weights).

Design (a lean version of the unreflectanything engine):

* **Per-component optimizer groups** — encoder params (``encoder.*``) get
  ``RGB_ENCODER_LR``; everything else gets the base ``LR``. ``RGB_ENCODER_LR == 0``
  (or ``FREEZE_BACKBONE``) leaves the encoder frozen, so it contributes no group.
* **AMP** — bf16 autocast on an A100 (no grad scaler needed), fp16 + ``GradScaler``
  elsewhere, disabled on CPU.
* **Schedules** — cosine LR with linear warmup; KL weight annealed from
  ``KL_WEIGHT_START`` up to ``LOSS.KL_WEIGHT`` over ``KL_ANNEAL_EPOCHS``;
  **scheduled sampling** — a per-point teacher-forcing probability annealed
  linearly 1->0 over ``TEACHER_FORCING_EPOCHS`` (feed GT positions often early,
  then let the filter run on its own predictions; no hard cliff). All are pure
  functions of the epoch, so they are correct after a resume with no extra state.
* **Checkpoints** — ``last.pt`` every epoch and ``best.pt`` on the best monitored
  metric (val EPE), written under ``<run_dir>/stage{idx}_{name}/``. A rerun of the
  same stage resumes from ``last.pt`` (model + optimizer + scaler + epoch); a fresh
  later stage carries the **weights** of the previous completed stage.
* **W&B** — per-epoch scalars (headline ``val/epe``) and a periodic pred-vs-GT
  ``wandb.Video``; a no-op when W&B is disabled/unavailable.

``Engine(...).fit()`` runs one stage and returns its best metrics.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import tracking_metrics
from utilities.log import get_logger
from utilities.runs import (
    find_cross_run_checkpoint,
    is_cross_run_resume,
    mark_stage_complete,
    parse_resume_value,
    resolve_source_run_dir,
)

logger = get_logger(__name__).set_context("ENGINE")


# --------------------------------------------------------------------------- #
# W&B (optional; shared by a whole run, not per stage)
# --------------------------------------------------------------------------- #
def init_wandb(config: Any, run_dir: Optional[Path] = None, run_id: Optional[str] = None) -> Tuple[Any, bool]:
    """Return ``(run, owned)``. ``owned`` is True only when *we* started the run.

    Disabled by ``NO_WANDB``. Inside a sweep (``sweep_agent.py`` already opened a
    run) the active run is reused and ``owned`` is False, so we don't finish it.
    Any failure (offline node, import error) degrades to ``(None, False)`` — the
    engine simply trains without logging.

    ``run_id`` (optional — e.g. a checkpoint's saved ``wandb_run_id``) resumes that
    *exact* run instead of opening a new one, so out-of-process logging (standalone
    ``evaluate.py``) lands on the same run/chart as training rather than a
    same-named duplicate. Falls back to opening a fresh run if the resume fails
    (run id stale/deleted, offline, etc.).
    """
    if bool(config.get("NO_WANDB", False)):
        return None, False
    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        logger.warning(f"wandb unavailable ({e}); training without logging")
        return None, False
    if getattr(wandb, "run", None) is not None:
        return wandb.run, False                          # sweep already opened it
    project = str(config.get("WANDB_PROJECT", "Twist"))
    entity = str(config.get("WANDB_ENTITY", "twisteam"))
    wdir = str(run_dir) if run_dir is not None else None
    if run_id:
        try:
            run = wandb.init(project=project, entity=entity, id=run_id, resume="must", dir=wdir)
            return run, True
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not resume W&B run '{run_id}' ({e}); opening a new run instead")
    try:
        cfg = config.toDict() if hasattr(config, "toDict") else dict(config)
        run = wandb.init(
            project=project,
            name=config.get("EXPERIMENT_NAME", None),
            entity=entity,
            config=cfg,
            dir=wdir,
        )
        return run, True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"wandb.init failed ({e}); training without logging")
        return None, False


def finish_wandb(run: Any, owned: bool) -> None:
    if run is not None and owned:
        try:
            run.finish()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Optimizer with per-component learning rates
# --------------------------------------------------------------------------- #
_OPTIMIZERS = {
    "adamw": torch.optim.AdamW,
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}


def build_optimizer(model: nn.Module, config: Any) -> torch.optim.Optimizer:
    """Two param groups: the encoder (``encoder.*``) at ``RGB_ENCODER_LR`` and the
    rest at ``LR``. Frozen params (``requires_grad == False``) are skipped, so a
    frozen encoder simply contributes nothing."""
    mc = config.get("MODEL", {})
    mc = mc.toDict() if hasattr(mc, "toDict") else dict(mc)
    enc_cfg = dict(mc.get("RGB_ENCODER", {}) or {})
    base_lr = float(config.get("LR", 3.0e-4))
    enc_lr = float(enc_cfg.get("RGB_ENCODER_LR", 0.0) or 0.0)
    wd = float(config.get("WEIGHT_DECAY", 1.0e-4))

    # Unwrap DDP so parameter names don't carry the "module." prefix.
    m = getattr(model, "module", model)
    enc_params, rest_params = [], []
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        (enc_params if name.startswith("encoder.") else rest_params).append(p)

    groups = []
    if rest_params:
        groups.append({"params": rest_params, "lr": base_lr, "weight_decay": wd, "name": "model"})
    if enc_params and enc_lr > 0:
        groups.append({"params": enc_params, "lr": enc_lr, "weight_decay": wd, "name": "encoder"})

    opt_name = str(config.get("OPTIMIZER", "adamw")).lower()
    opt_cls = _OPTIMIZERS.get(opt_name, torch.optim.AdamW)
    kwargs = {} if opt_cls is torch.optim.SGD else {"betas": (0.9, 0.999)}
    if not groups:  # everything frozen -- keep a valid (empty) optimizer
        logger.warning("no trainable parameters -- optimizer has no param groups")
        groups = [{"params": [], "lr": base_lr}]
    optimizer = opt_cls(groups, **kwargs)
    for g in optimizer.param_groups:
        n = sum(p.numel() for p in g["params"])
        logger.info(f"  optim group '{g.get('name', '?')}': lr={g['lr']:.2e}  params={n:,}")
    return optimizer


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def stage_dir(run_dir: Path, idx: int, name: str) -> Path:
    base = f"stage{idx}"
    # append the phase name only when it adds information (named multi-stage runs);
    # the implicit single stage is just "stage0", never "stage0_stage0".
    if name and name != base:
        base = f"{base}_{name}"
    return Path(run_dir) / base


def _prev_stage_checkpoint(run_dir: Path, idx: int) -> Optional[Path]:
    """``best.pt`` (else ``last.pt``) of the highest stage ``< idx`` that has one."""
    run_dir = Path(run_dir)
    for j in range(idx - 1, -1, -1):
        for d in sorted(run_dir.glob(f"stage{j}_*")):
            for fn in ("best.pt", "last.pt"):
                if (d / fn).exists():
                    return d / fn
    return None


# --------------------------------------------------------------------------- #
# DDP helpers
# --------------------------------------------------------------------------- #
def _get_sampler(loader: Optional[DataLoader]):
    """Return the loader's sampler if it has ``set_epoch`` (DistributedSampler), else None."""
    if loader is None:
        return None
    sampler = getattr(loader, "sampler", None)
    return sampler if hasattr(sampler, "set_epoch") else None


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Engine:
    """Train one schedule stage end to end."""

    def __init__(
        self,
        config: Any,
        model: nn.Module,
        loss_fn: nn.Module,
        loaders: Dict[str, Optional[DataLoader]],
        device: torch.device,
        run_dir: Path,
        stage_idx: int = 0,
        stage_name: str = "stage0",
        wandb_run: Any = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.config = config
        self.model = model
        self.loss_fn = loss_fn
        self.loaders = loaders
        self.device = device
        self.run_dir = Path(run_dir)
        self.stage_idx = int(stage_idx)
        self.stage_name = str(stage_name)
        self.wandb = wandb_run
        self._rank = rank
        self._world_size = world_size
        self._is_ddp = world_size > 1
        # DistributedSamplers need set_epoch() each epoch for correct shuffling.
        self._train_sampler = _get_sampler(loaders.get("train"))

        self.epochs = int(config.get("EPOCHS", 1))
        self.grad_clip = float(config.get("GRAD_CLIP", 1.0))
        self.log_every = int(config.get("LOG_EVERY", 20))
        self.viz_every = int(config.get("VIZ_EVERY", 5))          # epochs between val gifs (0 off)
        self.viz_val_clips = max(1, int(config.get("VIZ_VAL_CLIPS", 1)))  # # distinct val clips logged per viz epoch
        self.viz_every_batches = int(config.get("VIZ_EVERY_BATCHES", 0))  # train-step viz cadence (0 off)
        self.viz_dense_spacing = int(config.get("VIZ_DENSE_SPACING", 0))  # dense-grid query spacing px (0 off)
        self.viz_dense_max_points = int(config.get("VIZ_DENSE_MAX_POINTS", 400))  # safety cap on grid size
        self.viz_frames = int(config.get("VIZ_FRAMES", 24))       # cap clip length (short)
        self.viz_max_points = int(config.get("VIZ_MAX_POINTS", 48))
        self.viz_size = int(config.get("VIZ_SIZE", 256))          # square px of the logged frames
        self.viz_tail = int(config.get("VIZ_TAIL", 12))           # pred-track trail length (frames)
        self.viz_dpi = int(config.get("VIZ_DPI", 56))             # fallback dpi when VIZ_SIZE unset
        self.viz_fps = int(config.get("VIZ_FPS", 8))
        self._viz_keys_defined: set = set()                       # unused; kept for checkpoint compat
        # Independent RNG for "different every time" clip sampling (NOT the seeded
        # torch generator, so viz picks vary across calls within one run).
        import random as _random
        self._viz_rng = _random.Random()
        # W&B video logging (mp4 encode + upload) is offloaded to a single
        # background thread on rank 0 so it never blocks the training loop /
        # DDP collective schedule. Only one viz is in flight at a time; a new
        # one is dropped while the previous is still encoding (prevents backlog).
        self._viz_executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="wandb-viz")
            if (self._rank == 0 and self.wandb is not None) else None
        )
        self._viz_future: Optional[Future] = None
        self.val_every = max(1, int(config.get("VAL_EVERY", 1)))
        self.max_steps = int(config.get("MAX_STEPS_PER_EPOCH", 0))  # 0 -> all
        self.max_val_steps = int(config.get("MAX_VAL_STEPS", 0))    # 0 -> all (smoke caps this)
        self.patience = int(config.get("EARLY_STOP_PATIENCE", 0))
        # checkpoint policy: "scratch" (ignore ckpts, fresh) | "last" | "best"
        # | "<source-run>" | "<source-run>:last|best" (warm-start weights from another run)
        self.resume_raw = str(config.get("RESUME", "last"))
        self.resume_token, self.resume_ckpt_hint = parse_resume_value(self.resume_raw)
        self.resume_mode = self.resume_token.lower()
        torch.manual_seed(int(config.get("SEED", 42)))

        # LR schedule
        self.scheduler = str(config.get("LR_SCHEDULER", "cosine")).lower()
        self.warmup_epochs = int(config.get("WARMUP_EPOCHS", 0))
        self.min_lr_ratio = float(config.get("MIN_LR_RATIO", 0.05))

        # KL annealing (target read from the loss the config built)
        self.kl_target = float(getattr(loss_fn, "kl_weight", 0.05))
        self.kl_start = float(config.get("KL_WEIGHT_START", 0.0))
        self.kl_anneal = int(config.get("KL_ANNEAL_EPOCHS", 0))

        # teacher-forcing curriculum
        self.tf_epochs = int(config.get("TEACHER_FORCING_EPOCHS", 0))

        # rollout (frame-free forecast) evaluation: observe ROLLOUT_OBSERVE_STEPS
        # frames after the query, then predict the rest prior-only. Isolates the
        # transition prior's standalone quality -- the occlusion/forecast thesis
        # the headline (fully-observed) EPE never exercises. Logged as
        # ``val/epoch/rollout/*`` (EPE/delta/AJ on the forecast region only).
        self.rollout_eval = bool(config.get("ROLLOUT_EVAL", False))
        self.rollout_observe = int(config.get("ROLLOUT_OBSERVE_STEPS", 4))
        # multi-step rollout TRAINING loss: when LOSS.ROLLOUT_WEIGHT>0 the model also
        # forecasts frame-free from a clean observed state and the loss supervises it
        # vs GT -- trains the prior to forecast through occlusion, the cure for the
        # rollout divergence the eval exposes. Optional ROLLOUT_OBSERVE_MIN/MAX
        # randomize the TRAIN-time observe length per batch, so the prior practices
        # coasting from varied state quality rather than always from the same step;
        # eval keeps the fixed ROLLOUT_OBSERVE_STEPS protocol for comparability.
        self.rollout_loss = float(getattr(loss_fn, "rollout_weight", 0.0)) > 0.0
        ro_min, ro_max = config.get("ROLLOUT_OBSERVE_MIN"), config.get("ROLLOUT_OBSERVE_MAX")
        self.rollout_observe_range = (
            (int(ro_min), int(ro_max)) if ro_min is not None and ro_max is not None else None)
        if self.rollout_observe_range and not (1 <= self.rollout_observe_range[0]
                                               <= self.rollout_observe_range[1]):
            raise ValueError(f"ROLLOUT_OBSERVE_MIN/MAX invalid: {self.rollout_observe_range}")

        # observation-dropout curriculum: drop the frame correction during TRAINING
        # so the loss at those steps trains the prior directly (forces a competent,
        # load-bearing prior instead of letting the gate open to ~1 and bypass it).
        # Dropping is PER-POINT in contiguous spans of OBS_DROPOUT_SPAN=[lo,hi]
        # frames — simulated occlusion: a masked point coasts on the transition
        # (and on its still-observing neighbours via the inter-point attention),
        # exactly the regime real occlusion puts it in at inference. OBS_DROPOUT is
        # the expected dropped fraction of post-query steps, ramped 0 -> OBS_DROPOUT
        # over OBS_DROPOUT_EPOCHS (the model first learns to observe, then to coast).
        self.obs_dropout = float(config.get("OBS_DROPOUT", 0.0))
        self.obs_dropout_epochs = int(config.get("OBS_DROPOUT_EPOCHS", 0))
        span = config.get("OBS_DROPOUT_SPAN", [3, 8])
        span = [int(v) for v in (span if isinstance(span, (list, tuple)) else [span, span])]
        if not (1 <= span[0] <= span[1]):
            raise ValueError(f"OBS_DROPOUT_SPAN must be [lo, hi] with 1 <= lo <= hi, got {span}")
        self.obs_dropout_span = (span[0], span[1])

        # benchmark evaluation (utilities.evaluation): the standalone TAP metrics
        # (Delta AVG / Average Jaccard / Occlusion Accuracy / ms-per-frame) on the
        # IS_EVAL_DATASET-flagged datasets, written as a CSV + a W&B table.
        #   EVAL_AT_END    -- run once after this stage's training finishes.
        #   EVAL_EVERY > 0 -- also run every N epochs after validation (monitoring).
        # Both default off, so existing runs are unaffected.
        self.eval_at_end = bool(config.get("EVAL_AT_END", False))
        self.eval_every = int(config.get("EVAL_EVERY", 0))
        # EVAL_EVERY_EXCLUDE: dataset name(s) to skip in the per-epoch monitoring eval
        # only (still scored at EVAL_AT_END) -- e.g. an expensive benchmark whose long
        # clips would slow every epoch. Passed to evaluate_and_report on periodic calls.
        self.eval_every_exclude = config.get("EVAL_EVERY_EXCLUDE", None)
        self.eval_max_clips = config.get("EVAL_MAX_CLIPS", None)
        self.eval_batch_size = int(config.get("EVAL_BATCH_SIZE", 1))
        self.eval_workers = int(config.get("EVAL_WORKERS", 0))
        # eval scores the whole dataset by default (independent of MAX_VAL_STEPS,
        # which only caps the per-epoch validation loop). Cap with EVAL_MAX_CLIPS /
        # EVAL_MAX_STEPS explicitly for a quick eval-path smoke.
        self.eval_max_steps = int(config.get("EVAL_MAX_STEPS", 0))

        # pseudo-GT synthetic supervision (novel-view generation from single frames,
        # dataset/pseudo_gt.py): when PSEUDO_GT.ENABLED, each TRAIN batch's real clips
        # are REPLACED on-device by synthetic clips generated from their frame-0 source
        # image (dense known tracks + visibility, through-occlusion coords). Default
        # OFF -> real batches pass through unchanged, so existing runs are byte-
        # identical. Generation runs MoGe + point-cloud warping on the training
        # DEVICE (GPU); it is impractically slow on CPU, so enable only on GPU runs.
        pg = config.get("PSEUDO_GT", None)
        try:
            pg = pg.toDict() if hasattr(pg, "toDict") else (dict(pg) if pg is not None else None)
        except Exception:  # noqa: BLE001
            pg = None
        self._pseudo_gt_cfg: Dict[str, Any] = pg or {}
        self.pseudo_gt_enabled = bool(self._pseudo_gt_cfg.get("ENABLED", False))
        if self.pseudo_gt_enabled and self._rank == 0:
            logger.info(
                "PSEUDO_GT ENABLED: train batches replaced by synthetic novel-view "
                f"clips (depth_source={self._pseudo_gt_cfg.get('DEPTH_SOURCE', 'moge')}, "
                f"grid_size={self._pseudo_gt_cfg.get('GRID_SIZE', 32)})"
            )

        # AMP: bf16 (no scaler) on A100-class GPUs, else fp16 + scaler, off on CPU
        self.amp = bool(config.get("AMP", True)) and device.type == "cuda"
        bf16 = self.amp and torch.cuda.is_bf16_supported()
        self.amp_dtype = torch.bfloat16 if bf16 else torch.float16
        self.use_scaler = self.amp and not bf16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.optimizer = build_optimizer(model, config)
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]

        self.dir = stage_dir(self.run_dir, self.stage_idx, self.stage_name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = math.inf
        self.start_epoch = 0
        self._bad_epochs = 0
        self.history: list = []   # per-epoch metric records (for notebooks / debugging)
        self._train_perf: Dict[str, float] = {}   # last epoch's compute-efficiency metrics

        # static model-size stats (logged once at fit start)
        _m = self._unwrap_model()
        self.n_params = sum(p.numel() for p in _m.parameters())
        self.n_trainable = sum(p.numel() for p in _m.parameters() if p.requires_grad)

    # -- DDP helpers --------------------------------------------------------- #
    def _is_main_process(self) -> bool:
        return self._rank == 0

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying module, stripping any DDP wrapper."""
        return self.model.module if self._is_ddp else self.model

    def _barrier(self) -> None:
        """All-rank sync point (no-op outside DDP).

        Used to resync after rank-0-only work (viz, checkpointing) so the other
        ranks don't race ahead into the next gradient all-reduce and trip the
        NCCL watchdog timeout while rank 0 is still busy.
        """
        if self._is_ddp:
            import torch.distributed as dist
            dist.barrier()

    def _all_reduce_dict(self, d: Dict[str, float]) -> Dict[str, float]:
        """Average a metrics dict across all DDP ranks. Safe against NaN values."""
        if not self._is_ddp:
            return d
        import torch.distributed as dist
        keys = sorted(d.keys())
        # Replace NaN with 0 before reduce (NaN poisons all_reduce).
        t = torch.tensor([d[k] if d[k] == d[k] else 0.0 for k in keys],
                         dtype=torch.float64, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= self._world_size
        return {k: t[i].item() for i, k in enumerate(keys)}

    # -- schedules ----------------------------------------------------------- #
    def _lr_mult(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        if self.scheduler != "cosine":
            return 1.0
        prog = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
        prog = min(max(prog, 0.0), 1.0)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    def _kl_weight(self, epoch: int) -> float:
        if self.kl_anneal <= 0:
            return self.kl_target
        frac = min(1.0, epoch / self.kl_anneal)
        return self.kl_start + (self.kl_target - self.kl_start) * frac

    def _tf_prob(self, epoch: int) -> float:
        """Scheduled-sampling teacher-forcing probability: linear 1->0 over
        ``TEACHER_FORCING_EPOCHS`` epochs, then 0 (pure free-running)."""
        if self.tf_epochs <= 0:
            return 0.0
        return max(0.0, 1.0 - epoch / self.tf_epochs)

    def _obs_dropout(self, epoch: int) -> float:
        """Per-step observation-dropout probability: linear 0 -> ``OBS_DROPOUT`` over
        ``OBS_DROPOUT_EPOCHS`` epochs, then constant (0 disables the curriculum)."""
        if self.obs_dropout <= 0:
            return 0.0
        if self.obs_dropout_epochs <= 0:
            return self.obs_dropout
        return self.obs_dropout * min(1.0, epoch / self.obs_dropout_epochs)

    def _apply_schedules(self, epoch: int) -> Tuple[float, float, float, float]:
        mult = self._lr_mult(epoch)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * mult
        kw = self._kl_weight(epoch)
        self.loss_fn.kl_weight = kw
        tf_prob = self._tf_prob(epoch)
        obs_p = self._obs_dropout(epoch)
        lr0 = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        return lr0, kw, tf_prob, obs_p

    # -- batch prep ---------------------------------------------------------- #
    def _prep(self, batch: Dict[str, Any]):
        d = self.device
        frames = batch["frames"].to(d, non_blocking=True)
        queries = batch["queries"].float().to(d, non_blocking=True)
        tgt = {
            "tracks": batch["tracks"].float().to(d, non_blocking=True),
            "visibility": batch["visibility"].to(d, non_blocking=True),
        }
        for k in ("time_mask", "point_mask", "pos_valid"):
            if k in batch and batch[k] is not None:
                tgt[k] = batch[k].to(d, non_blocking=True)
        return frames, queries, tgt

    def _metric_point_mask(self, queries: torch.Tensor, tgt: Dict[str, torch.Tensor]
                           ) -> torch.Tensor:
        """``(B,N)`` bool mask of the points whose track is scoreable: GT-visible at
        their own query frame (and real under ``point_mask``).

        A point occluded at its query frame was queried at whatever coordinate the
        reader stored there — on real-GT datasets the ``(0,0)`` occluded placeholder —
        so its predicted track is garbage by construction, yet it would be scored at
        the frames where it later becomes visible (the val-pollution mechanism: the
        eval-registry datasets set MAX_POINTS None, which bypasses the
        visible-at-query candidate filter in dataset.sampling). This mirrors the
        TAP-Vid rule the benchmark evaluator already applies (query at a visible
        frame); it guards the *metrics* only — the loss masks itself by ``pos_valid``.
        """
        vis = tgt["visibility"].bool()                              # (B,T,N)
        qt = queries[..., 0].long().clamp(0, vis.shape[1] - 1)      # (B,N) per-point query frame
        vis_at_q = vis.gather(1, qt.unsqueeze(1)).squeeze(1)        # (B,N)
        pm = tgt.get("point_mask")
        return vis_at_q & pm.bool() if pm is not None else vis_at_q

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp)

    def _sync(self) -> None:
        """Block until queued CUDA work finishes (no-op on CPU) for accurate timing."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # -- train / validate ---------------------------------------------------- #
    @torch.no_grad()
    def _gate_by_visibility(self, out: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                            qf: int, point_mask: Optional[torch.Tensor] = None
                            ) -> Tuple[float, float]:
        """Mean Kalman gate split by GT visibility — the gate-health diagnostic.

        A functioning gate should sit HIGH on visible points (trust the match) and
        drop toward 0 on GT-occluded ones (coast on the prior; the window shows
        the occluder, not the point). Only post-query steps where the observation
        was actually APPLIED count (dropout-masked steps have a forced -10 logit
        that would fake a healthy occluded gate). Returns NaN where a batch has
        no qualifying entries.
        """
        if "gate_logits" not in out:
            return float("nan"), float("nan")
        g = torch.sigmoid(out["gate_logits"].float())        # (B,T,N)
        vis = tgt["visibility"].bool()
        b, t, n = vis.shape
        if point_mask is None:
            point_mask = tgt.get("point_mask")
        mask = torch.ones_like(vis)
        if tgt.get("time_mask") is not None:
            mask &= tgt["time_mask"].bool()[:, :, None]
        if point_mask is not None:
            mask &= point_mask.bool()[:, None, :]
        mask &= torch.arange(t, device=vis.device)[None, :, None] > qf   # post-query only
        if "observed" in out:
            mask &= out["observed"] > 0.5
        m_vis, m_occ = mask & vis, mask & ~vis
        gate_vis = float(g[m_vis].mean()) if bool(m_vis.any()) else float("nan")
        gate_occ = float(g[m_occ].mean()) if bool(m_occ.any()) else float("nan")
        return gate_vis, gate_occ

    def _build_observe_mask(self, T: int, n_points: int, qf: int, p: float
                            ) -> Optional[torch.Tensor]:
        """``(T, N)`` bool keep-mask for PER-POINT, occlusion-shaped observation
        dropout.

        Each post-query (step, point) starts a dropped span with probability
        ``p / mean(span)`` and the span lasts ``Uniform[span_lo, span_hi]`` steps,
        so the expected dropped fraction of post-query steps is ~``p`` (slightly
        less where spans overlap or clip at the end). Spans are per-POINT: a
        masked point must coast on the transition — and on its still-observing
        neighbours through the inter-point attention — while the rest of the
        batch keeps observing, which is the regime real occlusion creates at
        inference (the old mask dropped whole frames, so neighbours were always
        blind together and the carry-your-neighbour mechanism went untrained).
        Pre-query / query frames stay True. ``OBS_DROPOUT_SPAN: [1, 1]``
        degenerates to i.i.d. per-point dropout. Returns None when ``p <= 0``.
        """
        if p <= 0 or T <= qf + 1:
            return None
        lo, hi = self.obs_dropout_span
        q = min(1.0, p / max(0.5 * (lo + hi), 1.0))     # span-start rate -> ~p dropped
        starts = torch.rand(T, n_points, device=self.device) < q
        starts[:qf + 1] = False
        lengths = torch.randint(lo, hi + 1, (T, n_points), device=self.device)
        dropped = torch.zeros(T, n_points, dtype=torch.bool, device=self.device)
        for off in range(hi):                            # unroll spans (hi is small)
            active = starts & (lengths > off)
            dropped[off:] |= active[:T - off] if off else active
        keep = ~dropped
        keep[:qf + 1] = True
        return keep

    def _maybe_pseudo_gt(self, epoch: int, frames: torch.Tensor,
                         queries: torch.Tensor, tgt: Dict[str, torch.Tensor]):
        """When ``PSEUDO_GT.ENABLED``, REPLACE the real batch with synthetic
        novel-view clips generated on-device from each clip's query-frame source
        image; otherwise return ``(frames, queries, tgt)`` unchanged (exact no-op,
        the default — so a disabled run is byte-identical to before this feature).

        The synthetic batch matches :meth:`_prep`'s contract, so the rest of the
        training step (model call, teacher forcing, rollout, loss) is source-
        agnostic. ``pos_valid`` is the in-frame mask (the warped 3-D query carries a
        real coordinate through occlusion). Generation runs under ``no_grad`` — the
        clips are targets and the frozen encoder re-encodes the frames.
        """
        if not self.pseudo_gt_enabled:
            return frames, queries, tgt
        from dataset.pseudo_gt import (
            assemble_pseudo_batch,
            deformation_config_from_run_config,
            generate_pseudo_tracks,
            occluder_config_from_run_config,
            trajectory_config_from_run_config,
        )
        pg = self._pseudo_gt_cfg
        T = int(frames.shape[1])
        qf = max(0, min(int(queries[0, 0, 0].item()), T - 1))   # source = query frame
        traj = trajectory_config_from_run_config(
            {"PSEUDO_GT_TRAJECTORY": pg.get("TRAJECTORY") or None}, n_frames=T)
        deform = deformation_config_from_run_config(
            {"PSEUDO_GT_DEFORMATION": pg.get("DEFORMATION") or None})
        occ = occluder_config_from_run_config(
            {"PSEUDO_GT_OCCLUDERS": pg.get("OCCLUDERS") or None})
        grid_size = int(pg.get("GRID_SIZE", 32))
        margin = float(pg.get("GRID_MARGIN_FRAC", 0.03))
        depth_source = str(pg.get("DEPTH_SOURCE", "moge"))
        moge_name = str(pg.get("MOGE_MODEL_NAME", "Ruicheng/moge-2-vits-normal"))
        clips = [
            generate_pseudo_tracks(
                frames[b, qf], n_frames=T, grid_size=grid_size,
                seed=int(epoch * 100003 + b * 17 + 1), device=self.device,
                depth_source=depth_source, moge_model_name=moge_name,
                trajectory=traj, deformation=deform, occluders=occ,
                grid_margin_frac=margin,
            )
            for b in range(int(frames.shape[0]))
        ]
        return assemble_pseudo_batch(clips, self.device)

    def train_epoch(self, epoch: int, tf_prob: float, obs_dropout: float = 0.0) -> Dict[str, float]:
        loader = self.loaders.get("train")
        if loader is None:
            return {}
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        # Advance per-epoch point resampling on the train readers (no-op unless
        # RESAMPLE_POINTS_PER_EPOCH is on). ConcatDataset -> each sub-reader; val
        # readers are deliberately never advanced (fixed val point set).
        _tds = getattr(loader, "dataset", None)
        for _reader in (getattr(_tds, "datasets", None) or ([_tds] if _tds is not None else [])):
            if hasattr(_reader, "set_epoch"):
                _reader.set_epoch(epoch)
        self.model.train()
        if getattr(self.model, "encoder", None) is not None and getattr(self.model.encoder, "frozen", False):
            self.model.encoder.eval()                    # keep the frozen backbone in eval
        agg = {"loss": 0.0, "pos": 0.0, "prior": 0.0, "unc": 0.0, "unc_prior": 0.0,
               "vis": 0.0, "kl": 0.0, "kl_raw": 0.0, "ce": 0.0, "gce": 0.0, "epe": 0.0,
               "rollout": 0.0, "rollout_epe": 0.0, "w_rollout": 0.0,
               "w_pos": 0.0, "w_prior": 0.0, "w_unc": 0.0, "w_vis": 0.0, "w_kl": 0.0,
               "w_ce": 0.0, "w_gce": 0.0,
               "gate": 0.0, "obs_kept": 0.0}
        pt_sum, gt_travel_sum = 0.0, 0.0   # pooled travel for a stable epoch motion_ratio
        # NaN-safe means (separate counters, since a batch can legitimately lack the
        # entries — e.g. no visible points, no occluded-but-supervised frames):
        # the TAP metrics + the occlusion diagnostics (epe_occ from the loss;
        # gate|visible vs gate|occluded — THE gate-health chart: a working Kalman
        # gate should sit high on visible points and drop toward 0 on occluded ones).
        m_keys = ("average_jaccard", "delta_avg", "occlusion_accuracy",
                  "epe_occ", "gate_vis", "gate_occ", "coarse_gate")
        m_sum = {k: 0.0 for k in m_keys}
        m_cnt = {k: 0 for k in m_keys}
        n = 0
        gn_sum, gn_cnt = 0.0, 0   # grad-norm aggregated separately (skip non-finite overflow steps)
        # --- compute-efficiency accounting (excludes dataloading + diagnostics) ---
        # Timed with CUDA events, which are enqueued on the stream and do NOT block
        # the host: the old per-step torch.cuda.synchronize() pair drained the GPU
        # twice every step, destroying CPU/GPU overlap (the CPU could not build the
        # next batch while the GPU ran). We instead record a start/end event per step
        # and read their elapsed times after a SINGLE synchronize at epoch end, so
        # ms/image stays accurate with zero per-step serialization.
        use_cuda = self.device.type == "cuda"
        compute_s = 0.0          # model fwd+bwd+step only (filled from events at epoch end on CUDA)
        cuda_spans: list = []    # (start_event, end_event) per step
        n_clips_done, n_frames = 0, 0
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        for step, batch in enumerate(loader):
            if self.max_steps and step >= self.max_steps:
                break
            frames, queries, tgt = self._prep(batch)
            # pseudo-GT: replace the real batch with synthetic novel-view clips when
            # PSEUDO_GT.ENABLED (default OFF -> unchanged pass-through).
            frames, queries, tgt = self._maybe_pseudo_gt(epoch, frames, queries, tgt)
            # observation-dropout curriculum: per-point contiguous spans of dropped
            # frame corrections (simulated occlusion) — the loss there trains the
            # prior + the neighbour-carry attention directly.
            qf0 = max(0, min(int(queries[0, 0, 0].item()), frames.shape[1] - 1))
            observe_mask = self._build_observe_mask(frames.shape[1], queries.shape[1],
                                                    qf0, obs_dropout)
            obs_kept = float(observe_mask[qf0 + 1:].float().mean()) if observe_mask is not None else 1.0
            # rollout-loss observe length: fixed (eval protocol) or, with
            # ROLLOUT_OBSERVE_MIN/MAX, resampled per batch for varied coast starts.
            if not self.rollout_loss:
                rollout_observe = None
            elif self.rollout_observe_range is not None:
                lo, hi = self.rollout_observe_range
                rollout_observe = int(torch.randint(lo, hi + 1, (1,)).item())
            else:
                rollout_observe = self.rollout_observe
            self.optimizer.zero_grad(set_to_none=True)
            if use_cuda:
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
            else:
                _t0 = time.perf_counter()
            with self._autocast():
                out = self.model(frames, queries, point_mask=tgt.get("point_mask"),
                                 observe_mask=observe_mask,
                                 tf_prob=tf_prob,
                                 gt_tracks=tgt["tracks"] if tf_prob > 0.0 else None,
                                 rollout_observe=rollout_observe)
                total, parts = self.loss_fn(out, tgt)
            # Gradient norm is computed ONCE, fused into the clip call (clip_grad_norm_
            # returns the pre-clip global L2 norm), instead of a separate per-parameter
            # Python-loop reduction that synced the device to host for every tensor.
            if self.use_scaler:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = self._grad_norm_and_clip()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                grad_norm = self._grad_norm_and_clip()
                self.optimizer.step()
            if use_cuda:
                ev1.record()
                cuda_spans.append((ev0, ev1))
            else:
                compute_s += time.perf_counter() - _t0
            n_clips_done += int(frames.shape[0])             # B clips
            n_frames += int(frames.shape[0] * frames.shape[1])  # B*T frames (= "images")

            # mean Kalman gate (diagnostic: now free to learn a low gain where the
            # observation is noisier than the motion; not pinned open as before)
            gate = float(torch.sigmoid(out["gate_logits"]).mean().detach()) if "gate_logits" in out else float("nan")
            mpm = self._metric_point_mask(queries, tgt)      # scoreable points only (visible at query)
            gate_vis, gate_occ = self._gate_by_visibility(out, tgt, qf0, point_mask=mpm)
            coarse_gate = (float(out["coarse_gate"].mean().detach())
                           if "coarse_gate" in out else float("nan"))
            # train motion_ratio: how far the model's OWN predictions travel vs GT
            # (inflated while tf_prob>0 since the state is GT-fed; judge once tf->0).
            # The per-batch ratio-of-means is the step-log value; the EPOCH number is
            # POOLED (Σpred/Σgt below) so a near-static clip can't blow it up.
            with torch.no_grad():
                c = out["coords"].float()
                pd = torch.linalg.norm(c - c[:, :1], dim=-1)
                gd = torch.linalg.norm(tgt["tracks"] - tgt["tracks"][:, :1], dim=-1)
                vm = tgt["visibility"].bool()
                if vm.any():
                    pd_sum = float(pd[vm].sum()); gd_sum = float(gd[vm].sum())
                    motion_ratio = pd_sum / gd_sum if gd_sum > 1e-6 else float("nan")
                    pt_sum += pd_sum; gt_travel_sum += gd_sum
                else:
                    motion_ratio = float("nan")
            # full TAP metrics on the train batch (recorded coords are ALWAYS the
            # model's own predictions, so this is valid even while teacher-forced)
            tm = tracking_metrics(out["coords"], tgt["tracks"], out["vis_logits"],
                                  tgt["visibility"], tgt.get("time_mask"), mpm)
            tm = {**tm, "epe_occ": parts["epe_occ"], "gate_vis": gate_vis,
                  "gate_occ": gate_occ, "coarse_gate": coarse_gate}
            for k in m_keys:
                v = tm.get(k, float("nan"))
                if v == v:                                   # finite
                    m_sum[k] += v; m_cnt[k] += 1
            agg["loss"] += float(total.detach())
            for k in ("pos", "prior", "unc", "unc_prior", "vis", "kl", "kl_raw", "ce", "gce",
                      "epe", "rollout", "rollout_epe",
                      "w_pos", "w_prior", "w_unc", "w_vis", "w_kl", "w_ce", "w_gce",
                      "w_rollout"):
                agg[k] += float(parts[k])
            agg["gate"] += gate
            agg["obs_kept"] += obs_kept
            if grad_norm == grad_norm:                   # finite (GradScaler overflow steps -> NaN, skipped)
                gn_sum += grad_norm; gn_cnt += 1
            n += 1
            if self._is_main_process() and self.log_every and step % self.log_every == 0:
                self._log_train_step_console(epoch, step, len(loader), total, parts,
                                             grad_norm, gate, motion_ratio, tf_prob)
                if self.wandb is not None:
                    self.wandb.log({
                        "train/step_loss": float(total.detach()),
                        "train/step_epe": float(parts["epe"]),
                        "train/step_pos": float(parts["pos"]),
                        "train/step_prior": float(parts["prior"]),
                        "train/step_vis": float(parts["vis"]),
                        "train/step_kl": float(parts["kl"]),
                        "train/step_unc": float(parts["unc"]),
                        "train/step_ce": float(parts["ce"]),
                        "train/step_epe_occ": float(parts["epe_occ"]),
                        "train/step_rollout": float(parts["rollout"]),
                        "train/step_rollout_epe": float(parts["rollout_epe"]),
                        "train/step_grad_norm": grad_norm,
                        "train/step_gate": gate,
                        "train/step_gate_vis": gate_vis,
                        "train/step_gate_occ": gate_occ,
                        "train/step_motion_ratio": motion_ratio,
                        "schedules/lr": self.optimizer.param_groups[0]["lr"],
                        "schedules/kl_weight": self.loss_fn.kl_weight,
                        "schedules/tf_prob": tf_prob,
                        "schedules/obs_dropout": obs_dropout,
                    })
            # periodic in-training tracks viz (random clip of the CURRENT batch).
            # Rank-0-only forward; the barrier resyncs all ranks afterwards so the
            # others don't race into the next all-reduce while rank 0 visualizes.
            if (self.viz_every_batches and step > 0 and step % self.viz_every_batches == 0):
                self._log_tracks(epoch, split="train", batch=batch, step=step)
                self.model.train()                           # _log_tracks switched to eval
                if getattr(self.model, "encoder", None) is not None and getattr(self.model.encoder, "frozen", False):
                    self.model.encoder.eval()
                self._barrier()
        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["motion_ratio"] = pt_sum / gt_travel_sum if gt_travel_sum > 1e-6 else float("nan")
        for k in m_keys:                                     # NaN-safe TAP metric means
            out[k] = m_sum[k] / m_cnt[k] if m_cnt[k] else float("nan")
        out["grad_norm"] = gn_sum / gn_cnt if gn_cnt else float("nan")
        # Fold the per-step CUDA event timings into compute_s with ONE synchronize
        # (the only host/device sync the timing path now costs per epoch).
        if use_cuda and cuda_spans:
            torch.cuda.synchronize(self.device)
            compute_s = sum(s.elapsed_time(e) for s, e in cuda_spans) / 1e3  # ms -> s
        peak_mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if self.device.type == "cuda" else float("nan")
        self._train_perf = self._compute_perf(compute_s, n_frames, n_clips_done, peak_mem_gb)
        return self._all_reduce_dict(out)

    def _compute_perf(self, compute_s: float, n_frames: int, n_clips: int,
                      peak_mem_gb: float) -> Dict[str, float]:
        """Epoch compute-efficiency metrics, DDP-aware.

        Ranks process distinct shards *in parallel*, so the right reductions are:
          * counts (frames/clips)  -> SUM   (work done by the whole job)
          * compute-seconds        -> SUM (for mean per-image cost) and MAX (≈ the
                                      epoch wall-clock, since ranks overlap)
          * peak GPU memory        -> MAX   (the worst-rank ceiling, the real limit)

        From those: ``ms_per_image`` / ``ms_per_clip`` = total-compute / total-work
        (mean per-GPU cost) and ``images_per_s`` / ``clips_per_s`` = total-work /
        wall-clock (the *aggregate* system throughput, which scales with GPUs).
        With ``world_size == 1`` every reduction is a no-op, so single-GPU numbers
        are unchanged.
        """
        s_sum = s_max = float(compute_s)
        nf, nc = float(n_frames), float(n_clips)
        mem_max = float(peak_mem_gb)
        if self._is_ddp:
            import torch.distributed as dist
            summed = torch.tensor([compute_s, nf, nc], dtype=torch.float64, device=self.device)
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
            s_sum, nf, nc = summed.tolist()
            maxed = torch.tensor(
                [compute_s, mem_max if mem_max == mem_max else 0.0],  # NaN -> 0 for the reduce
                dtype=torch.float64, device=self.device,
            )
            dist.all_reduce(maxed, op=dist.ReduceOp.MAX)
            s_max, mem_max = maxed.tolist()
        bs = float(self.config.get("BATCH_SIZE", 0))
        return {
            "ms_per_image": 1e3 * s_sum / max(nf, 1.0),     # one frame == one image (per-GPU cost)
            "ms_per_clip": 1e3 * s_sum / max(nc, 1.0),
            "images_per_s": nf / s_max if s_max > 0 else float("nan"),   # aggregate across ranks
            "clips_per_s": nc / s_max if s_max > 0 else float("nan"),
            "compute_s": s_max,                              # ≈ epoch compute wall-clock
            "peak_mem_gb": mem_max,                          # worst-rank ceiling
            "batch_size": bs,                                # per-GPU
            "effective_batch_size": bs * self._world_size,   # global (DDP-aware)
            "world_size": float(self._world_size),
        }

    def _grad_norm_and_clip(self) -> float:
        """Clip gradients to ``GRAD_CLIP`` and return their global pre-clip L2 norm
        in a SINGLE pass (assumes grads are already unscaled).

        ``torch.nn.utils.clip_grad_norm_`` already computes the total L2 norm with a
        fused ``foreach`` kernel and returns it, so we reuse that value instead of a
        second, separate reduction. The old ``_grad_norm`` looped over parameters
        calling ``float(p.grad.norm(2))`` — a device->host sync for *every* tensor —
        and then ``clip_grad_norm_`` recomputed the same norm; this collapses both
        into one fused kernel and one host sync. With ``GRAD_CLIP <= 0`` we pass
        ``max_norm=inf`` so nothing is clipped but the norm is still measured.

        Returns NaN if non-finite (a GradScaler overflow step on the fp16 path leaves
        inf/NaN grads, even though ``scaler.step`` then skips the update) so the
        spurious value is dropped from the epoch aggregate rather than poisoning it.
        """
        max_norm = self.grad_clip if self.grad_clip > 0 else float("inf")
        total = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        g = float(total)
        return g if math.isfinite(g) else float("nan")

    def _forecast_mask(self, queries: torch.Tensor, tgt: Dict[str, torch.Tensor], T: int) -> torch.Tensor:
        """``(B,T)`` bool mask of the frame-free *forecast* region: steps more than
        ``rollout_observe`` after each clip's query frame (intersected with the real
        time_mask). Restricts the rollout metrics to the prior-only horizon."""
        ar = torch.arange(T, device=self.device)[None, :]          # (1,T)
        qfb = queries[:, 0, 0].long()[:, None]                     # (B,1) query frame per clip
        fmask = (ar - qfb) > self.rollout_observe                  # (B,T) forecast region
        tmask = tgt.get("time_mask")
        if tmask is not None:
            fmask = fmask & tmask.bool()
        return fmask

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        loader = self.loaders.get("val")
        if loader is None:
            return {}
        self.model.eval()
        agg: Dict[str, float] = {"loss": 0.0}
        cnt: Dict[str, int] = {"loss": 0}
        for vi, batch in enumerate(loader):
            if self.max_val_steps and vi >= self.max_val_steps:
                break
            frames, queries, tgt = self._prep(batch)
            with self._autocast():
                out = self.model(frames, queries, point_mask=tgt.get("point_mask"))
                total, parts = self.loss_fn(out, tgt)
            mpm = self._metric_point_mask(queries, tgt)      # scoreable points only (visible at query)
            m = tracking_metrics(out["coords"], tgt["tracks"], out["vis_logits"],
                                 tgt["visibility"], tgt.get("time_mask"), mpm)
            qf0 = max(0, min(int(queries[0, 0, 0].item()), frames.shape[1] - 1))
            gate_vis, gate_occ = self._gate_by_visibility(out, tgt, qf0, point_mask=mpm)
            m = {**m, "epe_occ": parts["epe_occ"], "gate_vis": gate_vis, "gate_occ": gate_occ}
            if "coarse_gate" in out:
                m["coarse_gate"] = float(out["coarse_gate"].mean())
            agg["loss"] += float(total.detach()); cnt["loss"] += 1
            for k, v in m.items():                       # epe, delta_avg, OA, AJ, per-threshold deltas
                if v == v:                               # drop NaN
                    agg[k] = agg.get(k, 0.0) + v
                    cnt[k] = cnt.get(k, 0) + 1
            # --- frame-free rollout eval (prior-only forecast horizon) ---
            if self.rollout_eval:
                with self._autocast():
                    out_r = self.model(frames, queries, point_mask=tgt.get("point_mask"),
                                       observe_steps=self.rollout_observe)
                fmask = self._forecast_mask(queries, tgt, frames.shape[1])
                m_r = tracking_metrics(out_r["coords"], tgt["tracks"], out_r["vis_logits"],
                                       tgt["visibility"], fmask, mpm)
                for k, v in m_r.items():
                    rk = f"rollout/{k}"
                    if v == v:
                        agg[rk] = agg.get(rk, 0.0) + v
                        cnt[rk] = cnt.get(rk, 0) + 1
        local = {k: (agg[k] / cnt[k] if cnt.get(k) else float("nan")) for k in agg}
        reduced = self._all_reduce_dict(local)
        # pooled motion_ratio = mean(pred_travel)/mean(gt_travel), robust to near-static
        # clips (which blow up the per-batch ratio-of-means). Same for rollout eval.
        for pre in ("", "rollout/"):
            pt, gt = reduced.get(pre + "pred_travel"), reduced.get(pre + "gt_travel")
            if pt is not None and gt is not None and gt == gt and gt > 1e-6:
                reduced[pre + "motion_ratio"] = pt / gt
            reduced.pop(pre + "pred_travel", None)
            reduced.pop(pre + "gt_travel", None)
        return reduced

    # -- benchmark evaluation ------------------------------------------------ #
    @torch.no_grad()
    def _run_evaluation(self, epoch: int, tag: str, periodic: bool = False) -> None:
        """Run the standalone benchmark evaluation on the selected eval datasets
        (rank-0 only, on the unwrapped model) and write its CSV + W&B table.

        ``periodic`` marks the per-epoch monitoring call (``EVAL_EVERY``): it forwards
        ``EVAL_EVERY_EXCLUDE`` so an expensive dataset is dropped from the every-epoch
        pass while the end-of-stage (``EVAL_AT_END``) pass still scores it.

        Other ranks wait at the trailing barrier so they don't race into the next
        gradient all-reduce while rank 0 evaluates. Never raises into the loop.
        """
        if not self._is_main_process():
            self._barrier()
            return
        try:
            from utilities.evaluation import evaluate_and_report
            evaluate_and_report(
                self._unwrap_model(), self.config, self.device, self.run_dir,
                wandb_run=self.wandb, tag=tag, epoch=epoch,
                max_clips=self.eval_max_clips, batch_size=self.eval_batch_size,
                num_workers=self.eval_workers, amp=self.amp, amp_dtype=self.amp_dtype,
                max_steps=self.eval_max_steps,
                exclude_datasets=(self.eval_every_exclude if periodic else None),
            )
        except Exception as e:  # noqa: BLE001 -- eval must never crash training
            logger.warning(f"benchmark evaluation skipped ({e})")
        finally:
            self._barrier()

    # -- qualitative tracks viz ---------------------------------------------- #
    def _random_batch(self, split: str):
        """A *random* batch from ``split`` (different every call). Falls back across
        splits. Returns the batch dict, or None."""
        loader = self.loaders.get(split) or self.loaders.get("val") or self.loaders.get("train")
        if loader is None:
            return None
        try:
            import itertools
            n = len(loader)                                  # DataLoader has __len__
            k = self._viz_rng.randrange(n) if n > 1 else 0
            return next(itertools.islice(loader, k, None))
        except Exception:  # noqa: BLE001 -- iterable/len edge cases
            try:
                return next(iter(loader))
            except Exception:  # noqa: BLE001
                return None

    @torch.no_grad()
    def _render_tracks_pair(self, epoch: int, split: str, batch,
                            step: Optional[int] = None, label: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Render one clip to its two ``VIZ_SIZE``² uint8 viz arrays (no logging):

          * ``compare``     — predicted (○) vs GT (△) points, with error lines
          * ``pred_tracks`` — predicted tracks with fading motion trails

        ``label`` (e.g. a dataset name) prefixes the on-frame title. The model runs
        free (no teacher forcing) so the viz reflects true tracking. Returns the
        arrays; the caller namespaces the keys and submits them.
        """
        from utilities.visualization import render_comparison_frames, render_track_frames
        frames, queries, tgt = self._prep(batch)
        b = self._viz_rng.randrange(frames.shape[0])     # random clip in the batch
        model = self._unwrap_model()                     # unwrapped: no DDP sync on this pass
        model.eval()
        with self._autocast():
            out = model(frames, queries, point_mask=tgt.get("point_mask"))
        tf = min(self.viz_frames, frames.shape[1]) if self.viz_frames > 0 else frames.shape[1]
        # Render only the scoreable points (GT-visible at their query frame): a point
        # occluded at its query was queried at the reader's placeholder coord and its
        # track is garbage by construction — hiding it matches what the metrics score.
        keep = self._metric_point_mask(queries, tgt)[b].cpu()
        if not bool(keep.any()):
            keep = torch.ones_like(keep)
        pred_vis = (torch.sigmoid(out["vis_logits"][b, :tf]) > 0.5).cpu()[:, keep]
        gt_xy = tgt["tracks"][b, :tf].cpu()[:, keep]
        pr_xy = out["coords"][b, :tf].float().cpu()[:, keep]
        gt_vis = tgt["visibility"][b, :tf].cpu()[:, keep]
        src = batch["frames"][b, :tf]
        tag = (f"{label} " if label else "") + f"{self.stage_name} {split} ep{epoch}" \
            + (f" b{step}" if step is not None else "")
        compare = render_comparison_frames(
            src, gt_xy, pr_xy, gt_visibility=gt_vis, pred_visibility=pred_vis,
            max_points=self.viz_max_points, out_size=self.viz_size, title=tag,
        )
        tracks = render_track_frames(
            src, pr_xy, visibility=pred_vis, tail=self.viz_tail,
            max_points=self.viz_max_points, out_size=self.viz_size, title=f"{tag} pred",
        )
        return {"compare": compare, "pred_tracks": tracks}

    @torch.no_grad()
    def _log_tracks(self, epoch: int, split: str, batch=None, step: Optional[int] = None,
                    idx: Optional[int] = None, label: Optional[str] = None) -> None:
        """Render + submit a single clip's videos (the train-step batch-viz path).

        The per-epoch validation/train viz instead goes through
        :meth:`_log_epoch_viz`, which batches every clip into one submission so the
        single-flight viz executor cannot drop part of the set.
        """
        if not self._is_main_process() or self.wandb is None:
            return
        if batch is None:
            batch = self._random_batch(split)
        if batch is None:
            return
        try:
            pair = self._render_tracks_pair(epoch, split, batch, step=step, label=label)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"tracks viz skipped ({e})")
            return
        suffix = f"_{label}" if label is not None else (f"_{idx}" if idx is not None else "")
        self._log_video(epoch, split, pair, suffix=suffix)

    # -- per-epoch qualitative viz (one batched W&B submission) -------------- #
    def _val_subdatasets(self):
        """``[(name, dataset), ...]`` for the val loader's sub-datasets.

        Unwraps a ``ConcatDataset`` into its readers (each tagged with
        ``dataset_name`` by ``create_datasets_from_config``); a single dataset is
        returned as one entry. Empty list when there is no val loader.
        """
        loader = self.loaders.get("val")
        ds = getattr(loader, "dataset", None) if loader is not None else None
        if ds is None:
            return []
        subs = getattr(ds, "datasets", None) or [ds]     # ConcatDataset.datasets, else itself
        return [(getattr(s, "dataset_name", None) or s.__class__.__name__, s) for s in subs]

    def _collate_one(self, item: Dict[str, Any]):
        """Collate a single dataset item into a batch of 1 using the val loader's
        collate_fn (``pad_collate`` for variable-length clips, else default)."""
        from torch.utils.data._utils.collate import default_collate
        loader = self.loaders.get("val")
        collate = getattr(loader, "collate_fn", None) if loader is not None else None
        return (collate or default_collate)([item])

    @torch.no_grad()
    def _collect_val_per_dataset(self, epoch: int) -> Dict[str, np.ndarray]:
        """Render ``VIZ_VAL_CLIPS`` clip(s) **per val dataset** so every dataset is
        shown, returning namespaced media (``val/videos/compare_<DATASET>`` ...).

        Falls back to a single random val clip if the val loader exposes no
        identifiable sub-datasets.
        """
        media: Dict[str, np.ndarray] = {}
        subsets = self._val_subdatasets()
        if not subsets:
            batch = self._random_batch("val")
            if batch is not None:
                try:
                    pair = self._render_tracks_pair(epoch, "val", batch)
                    media["val/videos/compare"] = pair["compare"]
                    media["val/videos/pred_tracks"] = pair["pred_tracks"]
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"val tracks viz skipped ({e})")
            return media
        for name, sub in subsets:
            n = len(sub)
            if n == 0:
                continue
            for j in range(self.viz_val_clips):
                sfx = name if self.viz_val_clips == 1 else f"{name}_{j}"
                try:
                    k = self._viz_rng.randrange(n)
                    pair = self._render_tracks_pair(epoch, "val", self._collate_one(sub[k]), label=sfx)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"val viz failed for {name} ({e})")
                    continue
                media[f"val/videos/compare_{sfx}"] = pair["compare"]
                media[f"val/videos/pred_tracks_{sfx}"] = pair["pred_tracks"]
        return media

    @torch.no_grad()
    def _collect_dense(self, epoch: int, split: str, batch=None) -> Dict[str, np.ndarray]:
        """Qualitative DENSE tracking: query a regular grid of points (spacing
        ``VIZ_DENSE_SPACING`` px) at the first frame and render the predicted
        tracks. Inspection only — these grid points have no GT, so nothing is
        scored; it shows how the field flows where the sparse GT can't. Returns
        ``{"<split>/videos/dense_tracks": (T,3,H,W)}`` (empty when disabled/failed).
        """
        if (not self._is_main_process() or self.wandb is None
                or self.viz_dense_spacing <= 0):
            return {}
        if batch is None:
            batch = self._random_batch(split)
        if batch is None:
            return {}
        try:
            from utilities.visualization import render_track_frames
            frames, _, _ = self._prep(batch)
            b = self._viz_rng.randrange(frames.shape[0])
            clip = frames[b:b + 1]                              # (1,T,3,H,W)
            h, w = int(clip.shape[-2]), int(clip.shape[-1])
            sp = self.viz_dense_spacing
            ys = torch.arange(sp // 2, h, sp, device=clip.device)
            xs = torch.arange(sp // 2, w, sp, device=clip.device)
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1).float()   # (N,2) xy
            if grid.shape[0] > self.viz_dense_max_points:      # safety cap (even subsample)
                sel = torch.linspace(0, grid.shape[0] - 1, self.viz_dense_max_points).round().long()
                grid = grid[sel]
            queries = torch.cat([grid.new_zeros(grid.shape[0], 1), grid], dim=-1).unsqueeze(0)  # (1,N,3)
            model = self._unwrap_model(); model.eval()
            with self._autocast():
                out = model(clip, queries)
            tf = min(self.viz_frames, clip.shape[1]) if self.viz_frames > 0 else clip.shape[1]
            pred_vis = (torch.sigmoid(out["vis_logits"][0, :tf]) > 0.5).cpu()
            pr_xy = out["coords"][0, :tf].float().cpu()
            src = batch["frames"][b, :tf]
            tag = f"{self.stage_name} {split} ep{epoch} dense({sp}px,{grid.shape[0]}pts)"
            dense = render_track_frames(
                src, pr_xy, visibility=pred_vis, tail=self.viz_tail,
                max_points=grid.shape[0], out_size=self.viz_size, title=tag,
            )
            return {f"{split}/videos/dense_tracks": dense}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"dense tracks viz skipped ({e})")
            return {}

    @torch.no_grad()
    def _log_epoch_viz(self, epoch: int) -> None:
        """Render the whole per-epoch qualitative viz and submit it as ONE W&B media
        row: a clip **per val dataset** (>=1 video per dataset), a random train clip,
        and the optional dense grids. Batching into a single submission means the
        single-flight viz executor never logs only part of the set.
        """
        if not self._is_main_process() or self.wandb is None:
            return
        media: Dict[str, np.ndarray] = {}
        media.update(self._collect_val_per_dataset(epoch))
        train_batch = self._random_batch("train")           # ... and a random train clip
        if train_batch is not None:
            try:
                pair = self._render_tracks_pair(epoch, "train", train_batch)
                media["train/videos/compare"] = pair["compare"]
                media["train/videos/pred_tracks"] = pair["pred_tracks"]
            except Exception as e:  # noqa: BLE001
                logger.warning(f"train tracks viz skipped ({e})")
        media.update(self._collect_dense(epoch, "val"))      # no-op unless VIZ_DENSE_SPACING > 0
        media.update(self._collect_dense(epoch, "train"))
        self._submit_media(epoch, media)

    def _log_video(self, epoch: int, split: str, arrays: Dict[str, np.ndarray],
                   suffix: str = "") -> None:
        """Namespace ``arrays`` as ``<split>/videos/<KEYNAME><suffix>`` and submit
        them as one media row (used by the train-step batch-viz path). ``suffix``
        (e.g. ``"_0"``) keeps several clips of the same split/epoch on distinct keys.
        """
        media = {f"{split}/videos/{k}{suffix}": v for k, v in arrays.items()}
        self._submit_media(epoch, media)

    def _submit_media(self, epoch: int, media: Dict[str, np.ndarray]) -> None:
        """Encode a ``{full_wandb_key: (T,3,H,W) uint8}`` media row to mp4 and log it.

        The mp4 encode + ``wandb.log`` is dispatched to a background thread so it
        never blocks the training loop (and, in DDP, never stalls the other ranks
        at the next gradient all-reduce). Only one row is encoded at a time; if a
        previous one is still in flight the new row is dropped rather than queued,
        so a slow encoder can never build an unbounded backlog. The whole epoch's
        viz is submitted as a single row so this single-flight drop is all-or-nothing
        (never logs only part of the per-dataset set).
        """
        if not media:
            return
        media = {k: np.ascontiguousarray(v) for k, v in media.items()}  # detach from caller's buffers
        if self._viz_executor is None:                       # no async path -> log inline
            self._encode_and_log_media(epoch, media)
            return
        if self._viz_future is not None and not self._viz_future.done():
            logger.debug("viz drop: previous W&B video still encoding")
            return
        self._viz_future = self._viz_executor.submit(self._encode_and_log_media, epoch, media)

    def _encode_and_log_media(self, epoch: int, media: Dict[str, np.ndarray]) -> None:
        """Encode each array to an mp4 ``wandb.Video`` and push the row to W&B.
        Runs on the viz worker thread; never raises into the training loop."""
        try:
            import wandb
            row: Dict[str, Any] = {"epoch": epoch}
            for key, arr in media.items():
                row[key] = wandb.Video(arr, fps=self.viz_fps, format="mp4")
            self.wandb.log(row)
        except Exception as e:  # noqa: BLE001 -- logging must never crash training
            logger.warning(f"W&B video log skipped ({e})")

    def _flush_viz(self, timeout: float = 120.0) -> None:
        """Block until any in-flight background video log finishes (rank 0 only).

        Called before a stage ends so the last clips are not lost when the W&B run
        is closed. Bounded by ``timeout`` so a wedged encoder can't hang shutdown.
        """
        fut, self._viz_future = self._viz_future, None
        if fut is None:
            return
        try:
            fut.result(timeout=timeout)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"viz flush incomplete ({e})")

    # -- checkpoint ---------------------------------------------------------- #
    def _ckpt(self, epoch: int) -> Dict[str, Any]:
        cfg = self.config.toDict() if hasattr(self.config, "toDict") else dict(self.config)
        return {
            "epoch": epoch,
            "model": self._unwrap_model().state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "stage_idx": self.stage_idx,
            "stage_name": self.stage_name,
            "config": cfg,
            "wandb_run_id": getattr(self.wandb, "id", None),
        }

    @staticmethod
    def _load_compatible(m: nn.Module, ckpt_state: Dict[str, Any]) -> Tuple[int, int, int]:
        """Load only the checkpoint tensors whose shape matches the current model.

        ``load_state_dict(strict=False)`` tolerates missing/unexpected *keys* but
        still hard-fails on a *shape* mismatch for a key present in both. That
        breaks resuming across a (minor) architecture change — e.g. the transition
        ``to_token`` input growing from ``[feat,hidden,pos]`` to
        ``[feat,hidden,pos,vel]`` (1282->1284). We instead drop shape-incompatible
        tensors and let them keep their fresh init, then load the rest non-strictly.
        Image size never changes any parameter shape, so switching resolutions
        (256<->448<->...) is always compatible.
        """
        model_state = m.state_dict()
        skipped = [k for k, v in ckpt_state.items()
                   if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)]
        compatible = {k: v for k, v in ckpt_state.items() if k not in skipped}
        missing, unexpected = m.load_state_dict(compatible, strict=False)
        if skipped:
            logger.warning(f"reinitialized {len(skipped)} shape-mismatched param(s) on resume "
                           f"(e.g. {skipped[0]}): {skipped}")
        return len(missing), len(unexpected), len(skipped)

    def _maybe_resume(self) -> None:
        """Restore weights/optimizer per ``RESUME`` policy.

        * ``"scratch"`` — ignore every checkpoint, train from epoch 0.
        * ``"last"`` / ``"best"`` — in-stage resume from this run's checkpoint.
        * ``"<source-run>"`` or ``"<source-run>:last|best"`` — warm-start model
          weights from another run (epoch 0, fresh optimizer). When the source run
          dir equals this run dir, ``last``/``best`` in-run semantics apply.

        When no in-stage checkpoint applies, carry the previous stage's weights.
        """
        m = self._unwrap_model()   # load into the underlying module (no "module." prefix)
        if self.resume_mode == "scratch":
            logger.info(f"RESUME=scratch -- training stage {self.stage_idx} from scratch "
                        "(checkpoints ignored)")
            return

        cross_run = is_cross_run_resume(self.resume_raw)
        if cross_run:
            source_dir = resolve_source_run_dir(self.resume_token, create=False)
            same_run = source_dir.resolve() == self.run_dir.resolve()
            if not same_run:
                which = self.resume_ckpt_hint or str(self.config.get("RESUME_CHECKPOINT", "best")).lower()
                ckpt_path = find_cross_run_checkpoint(source_dir, self.stage_idx, which)
                if ckpt_path is None:
                    logger.warning(
                        f"RESUME={self.resume_token!r}: no checkpoint in {source_dir}; "
                        "training from scratch"
                    )
                    return
                ck = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                n_missing, n_unexpected, n_skipped = self._load_compatible(m, ck["model"])
                logger.info(
                    f"warm-started stage {self.stage_idx} from {source_dir.name}/{ckpt_path.parent.name}/{ckpt_path.name} "
                    f"(cross-run; epoch 0; missing={n_missing}, unexpected={n_unexpected})"
                )
                if n_skipped:
                    logger.warning(
                        f"reinitialized {n_skipped} shape-mismatched param(s) on cross-run warm-start"
                    )
                return

        if self.resume_mode == "best":
            ckpt_path = self.dir / "best.pt"
            if not ckpt_path.exists():
                ckpt_path = self.dir / "last.pt"
        else:                                            # "last" (default) or unknown
            if self.resume_mode not in ("last",):
                logger.warning(f"unknown RESUME={self.resume_raw!r}; defaulting to 'last'")
            ckpt_path = self.dir / "last.pt"
        last = ckpt_path
        if last.exists():
            ck = torch.load(last, map_location=self.device, weights_only=False)
            # tolerate minor architecture changes (added heads, grown token input,
            # resolution switch) instead of hard-failing on resume.
            n_missing, n_unexpected, n_skipped = self._load_compatible(m, ck["model"])
            if n_missing or n_unexpected:
                logger.warning(f"resumed last.pt non-strictly "
                               f"(missing={n_missing}, unexpected={n_unexpected})")
            if n_skipped:
                # A reinitialized param invalidates its saved optimizer moments
                # (shape would mismatch at step time); start the optimizer fresh.
                # Schedules are pure functions of epoch, so this stays resume-safe.
                logger.warning("optimizer/scaler state reset (architecture changed since checkpoint)")
            else:
                try:
                    self.optimizer.load_state_dict(ck["optimizer"])
                    self.scaler.load_state_dict(ck["scaler"])
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"optimizer/scaler state not restored ({e})")
            self.best_metric = ck.get("best_metric", math.inf)
            self.start_epoch = int(ck.get("epoch", -1)) + 1
            logger.info(f"resuming stage {self.stage_idx} from {last.name} at epoch {self.start_epoch}")
            return
        prev = _prev_stage_checkpoint(self.run_dir, self.stage_idx)
        if prev is not None:
            ck = torch.load(prev, map_location=self.device, weights_only=False)
            n_missing, n_unexpected, _ = self._load_compatible(m, ck["model"])
            logger.info(f"carried weights from {prev.parent.name}/{prev.name} "
                        f"(missing={n_missing}, unexpected={n_unexpected})")

    # -- fit ----------------------------------------------------------------- #
    def fit(self) -> Dict[str, float]:
        self._maybe_resume()
        if self.start_epoch >= self.epochs:
            if self._is_main_process():
                logger.info(f"stage {self.stage_idx} already at epoch {self.start_epoch}/{self.epochs}; nothing to train")
                mark_stage_complete(self.run_dir, self.stage_idx, self.stage_name)
            self._barrier()
            return {"epe": self.best_metric}

        has_val = self.loaders.get("val") is not None
        if self._is_main_process():
            logger.info(f"training stage {self.stage_idx} '{self.stage_name}' for {self.epochs} epochs "
                        f"(amp={self.amp}/{self.amp_dtype if self.amp else '-'}, device={self.device.type}, "
                        f"world_size={self._world_size})")
            logger.info(f"model size: {self.n_params/1e6:.2f}M params "
                        f"({self.n_trainable/1e6:.2f}M trainable)")
            if self.wandb is not None:
                try:
                    self.wandb.summary["performance/params_total_M"] = self.n_params / 1e6
                    self.wandb.summary["performance/params_trainable_M"] = self.n_trainable / 1e6
                except Exception:  # noqa: BLE001
                    pass
        last_val: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.epochs):
            lr0, kw, tf_prob, obs_p = self._apply_schedules(epoch)
            tr = self.train_epoch(epoch, tf_prob=tf_prob, obs_dropout=obs_p)
            do_val = has_val and (epoch % self.val_every == 0 or epoch == self.epochs - 1)
            val = self.validate() if do_val else {}
            if val:
                last_val = val

            monitor = val.get("epe") if val else tr.get("epe", math.inf)
            improved = monitor == monitor and monitor < self.best_metric  # not NaN and better
            if improved:
                self.best_metric = monitor
                self._bad_epochs = 0
            else:
                self._bad_epochs += 1

            self.history.append({"epoch": epoch, "lr": lr0, "kl_weight": kw, "tf_prob": tf_prob,
                                 "obs_dropout": obs_p,
                                 **{f"train_{k}": v for k, v in tr.items()},
                                 **{f"val_{k}": v for k, v in val.items()}})
            self._log_console(epoch, lr0, kw, tf_prob, obs_p, tr, val)
            self._log_perf_console(epoch)
            if self._is_main_process() and self.wandb is not None:
                row = {"epoch": epoch, "stage": self.stage_idx,
                       "schedules/lr": lr0, "schedules/kl_weight": kw,
                       "schedules/tf_prob": tf_prob, "schedules/obs_dropout": obs_p,
                       **{f"train/epoch/{k}": v for k, v in tr.items()},
                       **{f"val/epoch/{k}": v for k, v in val.items()},
                       **{f"performance/{k}": v for k, v in self._train_perf.items()}}
                self.wandb.log(row)
            if self.viz_every and (epoch % self.viz_every == 0 or epoch == self.epochs - 1):
                # one batched submission (rank-0): a clip per val DATASET (>=1 video
                # per dataset, each dataset-keyed), a train clip, and dense grids.
                self._log_epoch_viz(epoch)

            if self._is_main_process():
                torch.save(self._ckpt(epoch), self.dir / "last.pt")
                if improved:
                    torch.save(self._ckpt(epoch), self.dir / "best.pt")

            # Resync after the rank-0-only epoch viz + checkpoint so the other
            # ranks don't enter the next epoch's all-reduce while rank 0 is busy.
            self._barrier()

            # periodic benchmark evaluation (monitoring): the full TAP metrics on
            # the IS_EVAL_DATASET datasets every EVAL_EVERY epochs. Its own CSV /
            # W&B table snapshot, tagged with the epoch.
            if self.eval_every and epoch % self.eval_every == 0:
                self._run_evaluation(epoch, tag=f"{self.stage_name}_ep{epoch + 1}", periodic=True)

            if self.patience and self._bad_epochs >= self.patience:
                if self._is_main_process():
                    logger.info(f"early stop: no improvement for {self.patience} epochs")
                break

        if self._is_main_process():
            mark_stage_complete(self.run_dir, self.stage_idx, self.stage_name)
            logger.info(f"stage {self.stage_idx} '{self.stage_name}' complete -- best epe={self.best_metric:.3f}px "
                        f"(checkpoints in {self.dir})")
        self._flush_viz()                       # finish pending W&B video before the run can close
        if self._viz_executor is not None:
            self._viz_executor.shutdown(wait=True)
            self._viz_executor = None
        # end-of-stage benchmark evaluation (canonical CSV per stage + W&B table)
        if self.eval_at_end:
            self._run_evaluation(self.epochs - 1, tag=self.stage_name)
        self._barrier()  # all ranks wait before returning (next stage must see completion marker)
        return last_val or {"epe": self.best_metric}

    def _log_train_step_console(self, epoch, step, n_batches, total, parts,
                                grad_norm, gate, motion_ratio, tf_prob) -> None:
        """Per-batch progress line (epoch + batch + live metrics), à la unreflectanything."""
        logger.info(
            f"E {epoch + 1:>3}/{self.epochs}  B {step + 1:>4}/{n_batches:<4}  "
            f"loss={float(total.detach()):.4f} epe={float(parts['epe']):.2f}px "
            f"pos={float(parts['pos']):.4f} vis={float(parts['vis']):.4f} kl={float(parts['kl']):.3f} "
            f"mr={motion_ratio:.2f} gate={gate:.2f} |g|={grad_norm:.2f} "
            f"lr={self.optimizer.param_groups[0]['lr']:.2e} tf={tf_prob:.2f}"
        )

    def _log_console(self, epoch, lr, kw, tf_prob, obs_p, tr, val) -> None:
        if not self._is_main_process():
            return
        line = f"[stage{self.stage_idx} {self.stage_name}] epoch {epoch + 1}/{self.epochs}"
        if tr:
            line += (f"  train: loss={tr.get('loss', float('nan')):.3f} epe={tr.get('epe', float('nan')):.2f}px "
                     f"mr={tr.get('motion_ratio', float('nan')):.2f} gate={tr.get('gate', float('nan')):.2f} "
                     f"AJ={tr.get('average_jaccard', float('nan')):.3f} "
                     f"|g|={tr.get('grad_norm', float('nan')):.2f}")
        if val:
            line += (f"  val: loss={val.get('loss', float('nan')):.3f} epe={val.get('epe', float('nan')):.2f}px "
                     f"mr={val.get('motion_ratio', float('nan')):.2f} stuck={val.get('stuck_frac', float('nan')):.2f} "
                     f"δ={val.get('delta_avg', float('nan')):.3f} OA={val.get('occlusion_accuracy', float('nan')):.3f} "
                     f"AJ={val.get('average_jaccard', float('nan')):.3f}")
            if "rollout/epe" in val:
                line += f" | roll_epe={val.get('rollout/epe', float('nan')):.2f}px roll_δ={val.get('rollout/delta_avg', float('nan')):.3f}"
        line += f"  lr={lr:.2e} kl_w={kw:.3f} tf={tf_prob:.2f} obs_drop={obs_p:.2f}  best_epe={self.best_metric:.2f}"
        logger.info(line)

    def _log_perf_console(self, epoch) -> None:
        """One-line compute-efficiency summary for the epoch (rank-0 only)."""
        if not self._is_main_process() or not self._train_perf:
            return
        p = self._train_perf
        mem = p.get("peak_mem_gb", float("nan"))
        ws = int(p.get("world_size", 1))
        ddp = f"  x{ws}gpu(eff_bs={int(p.get('effective_batch_size', 0))})" if ws > 1 else ""
        logger.info(
            f"perf epoch {epoch + 1}/{self.epochs}  "
            f"{p.get('ms_per_image', float('nan')):.2f} ms/image  "
            f"{p.get('ms_per_clip', float('nan')):.1f} ms/clip  "
            f"{p.get('images_per_s', float('nan')):.1f} img/s  "
            f"peak_mem={mem:.2f}GB  params={self.n_params/1e6:.2f}M{ddp}"
        )
