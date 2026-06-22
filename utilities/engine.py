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
  ``KL_WEIGHT_START`` up to ``MODEL.LOSS.KL_WEIGHT`` over ``KL_ANNEAL_EPOCHS``;
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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import tracking_metrics
from utilities.log import get_logger
from utilities.runs import mark_stage_complete

logger = get_logger(__name__).set_context("ENGINE")


# --------------------------------------------------------------------------- #
# W&B (optional; shared by a whole run, not per stage)
# --------------------------------------------------------------------------- #
def init_wandb(config: Any, run_dir: Optional[Path] = None) -> Tuple[Any, bool]:
    """Return ``(run, owned)``. ``owned`` is True only when *we* started the run.

    Disabled by ``NO_WANDB``. Inside a sweep (``sweep_agent.py`` already opened a
    run) the active run is reused and ``owned`` is False, so we don't finish it.
    Any failure (offline node, import error) degrades to ``(None, False)`` — the
    engine simply trains without logging.
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
    try:
        cfg = config.toDict() if hasattr(config, "toDict") else dict(config)
        run = wandb.init(
            project=str(config.get("WANDB_PROJECT", "twist")),
            name=str(config.get("EXPERIMENT_NAME", "run")),
            config=cfg,
            dir=str(run_dir) if run_dir is not None else None,
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
    return Path(run_dir) / f"stage{idx}_{name}"


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
        self.viz_every = int(config.get("VIZ_EVERY", 5))
        self.viz_frames = int(config.get("VIZ_FRAMES", 24))       # cap clip length (short)
        self.viz_max_points = int(config.get("VIZ_MAX_POINTS", 48))
        self.viz_dpi = int(config.get("VIZ_DPI", 56))             # low resolution
        self.viz_fps = int(config.get("VIZ_FPS", 8))
        self.val_every = max(1, int(config.get("VAL_EVERY", 1)))
        self.max_steps = int(config.get("MAX_STEPS_PER_EPOCH", 0))  # 0 -> all
        self.patience = int(config.get("EARLY_STOP_PATIENCE", 0))
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

    # -- DDP helpers --------------------------------------------------------- #
    def _is_main_process(self) -> bool:
        return self._rank == 0

    def _unwrap_model(self) -> nn.Module:
        """Return the underlying module, stripping any DDP wrapper."""
        return self.model.module if self._is_ddp else self.model

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

    def _apply_schedules(self, epoch: int) -> Tuple[float, float, float]:
        mult = self._lr_mult(epoch)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base * mult
        kw = self._kl_weight(epoch)
        self.loss_fn.kl_weight = kw
        tf_prob = self._tf_prob(epoch)
        lr0 = self.optimizer.param_groups[0]["lr"] if self.optimizer.param_groups else 0.0
        return lr0, kw, tf_prob

    # -- batch prep ---------------------------------------------------------- #
    def _prep(self, batch: Dict[str, Any]):
        d = self.device
        frames = batch["frames"].to(d, non_blocking=True)
        queries = batch["queries"].float().to(d, non_blocking=True)
        tgt = {
            "tracks": batch["tracks"].float().to(d, non_blocking=True),
            "visibility": batch["visibility"].to(d, non_blocking=True),
        }
        for k in ("time_mask", "point_mask"):
            if k in batch and batch[k] is not None:
                tgt[k] = batch[k].to(d, non_blocking=True)
        return frames, queries, tgt

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp)

    # -- train / validate ---------------------------------------------------- #
    def train_epoch(self, epoch: int, tf_prob: float) -> Dict[str, float]:
        loader = self.loaders.get("train")
        if loader is None:
            return {}
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(epoch)
        self.model.train()
        if getattr(self.model, "encoder", None) is not None and getattr(self.model.encoder, "frozen", False):
            self.model.encoder.eval()                    # keep the frozen backbone in eval
        agg = {"loss": 0.0, "pos": 0.0, "prior": 0.0, "unc": 0.0, "vis": 0.0, "kl": 0.0, "epe": 0.0,
               "w_pos": 0.0, "w_prior": 0.0, "w_unc": 0.0, "w_vis": 0.0, "w_kl": 0.0,
               "gate": 0.0, "motion_ratio": 0.0}
        n = 0
        gn_sum, gn_cnt = 0.0, 0   # grad-norm aggregated separately (skip non-finite overflow steps)
        for step, batch in enumerate(loader):
            if self.max_steps and step >= self.max_steps:
                break
            frames, queries, tgt = self._prep(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                out = self.model(frames, queries, point_mask=tgt.get("point_mask"),
                                 tf_prob=tf_prob,
                                 gt_tracks=tgt["tracks"] if tf_prob > 0.0 else None)
                total, parts = self.loss_fn(out, tgt)
            grad_norm = 0.0
            if self.use_scaler:
                self.scaler.scale(total).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = self._grad_norm()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total.backward()
                grad_norm = self._grad_norm()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            # mean Kalman gate (diagnostic: now free to learn a low gain where the
            # observation is noisier than the motion; not pinned open as before)
            gate = float(torch.sigmoid(out["gate_logits"]).mean().detach()) if "gate_logits" in out else float("nan")
            # train motion_ratio: how far the model's OWN predictions travel vs GT
            # (inflated while tf_prob>0 since the state is GT-fed; judge once tf->0).
            with torch.no_grad():
                c = out["coords"].float()
                pd = torch.linalg.norm(c - c[:, :1], dim=-1)
                gd = torch.linalg.norm(tgt["tracks"] - tgt["tracks"][:, :1], dim=-1)
                vm = tgt["visibility"].bool()
                motion_ratio = float((pd[vm].mean() / gd[vm].mean().clamp_min(1e-6))) if vm.any() else float("nan")
            agg["loss"] += float(total.detach())
            for k in ("pos", "prior", "unc", "vis", "kl", "epe", "w_pos", "w_prior", "w_unc", "w_vis", "w_kl"):
                agg[k] += float(parts[k])
            agg["gate"] += gate
            agg["motion_ratio"] += motion_ratio
            if grad_norm == grad_norm:                   # finite (GradScaler overflow steps -> NaN, skipped)
                gn_sum += grad_norm; gn_cnt += 1
            n += 1
            if self._is_main_process() and self.wandb is not None and self.log_every and step % self.log_every == 0:
                self.wandb.log({
                    "train/step_loss": float(total.detach()),
                    "train/step_epe": float(parts["epe"]),
                    "train/step_pos": float(parts["pos"]),
                    "train/step_prior": float(parts["prior"]),
                    "train/step_vis": float(parts["vis"]),
                    "train/step_kl": float(parts["kl"]),
                    "train/step_unc": float(parts["unc"]),
                    "train/step_grad_norm": grad_norm,
                    "train/step_gate": gate,
                    "train/step_motion_ratio": motion_ratio,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "kl_weight": self.loss_fn.kl_weight,
                    "tf_prob": tf_prob,
                })
        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["grad_norm"] = gn_sum / gn_cnt if gn_cnt else float("nan")
        return self._all_reduce_dict(out)

    def _grad_norm(self) -> float:
        """Global L2 norm of model gradients (assumes grads are already unscaled).

        Returns NaN if non-finite (a GradScaler overflow step on the fp16 path leaves
        inf/NaN grads here, even though ``scaler.step`` then skips the update) so the
        spurious value is dropped from the epoch aggregate rather than poisoning it.
        """
        total = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total += float(p.grad.detach().norm(2)) ** 2
        g = total ** 0.5
        return g if math.isfinite(g) else float("nan")

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        loader = self.loaders.get("val")
        if loader is None:
            return {}
        self.model.eval()
        agg: Dict[str, float] = {"loss": 0.0}
        cnt: Dict[str, int] = {"loss": 0}
        for batch in loader:
            frames, queries, tgt = self._prep(batch)
            with self._autocast():
                out = self.model(frames, queries, point_mask=tgt.get("point_mask"))
                total, _ = self.loss_fn(out, tgt)
            m = tracking_metrics(out["coords"], tgt["tracks"], out["vis_logits"],
                                 tgt["visibility"], tgt.get("time_mask"), tgt.get("point_mask"))
            agg["loss"] += float(total.detach()); cnt["loss"] += 1
            for k, v in m.items():                       # epe, delta_avg, OA, AJ, per-threshold deltas
                if v == v:                               # drop NaN
                    agg[k] = agg.get(k, 0.0) + v
                    cnt[k] = cnt.get(k, 0) + 1
        local = {k: (agg[k] / cnt[k] if cnt.get(k) else float("nan")) for k in agg}
        return self._all_reduce_dict(local)

    # -- qualitative video --------------------------------------------------- #
    @torch.no_grad()
    def _log_video(self, epoch: int, split: str = "val") -> None:
        """Log a short, low-res pred-vs-GT gif for ``split`` (``"val"`` | ``"train"``).

        Always runs the model free-running (no teacher forcing) so the gif shows
        the filter's true tracking quality — including on a *train* batch, which
        makes train-time progress and overfit/instability visible as it happens.
        """
        if not self._is_main_process():
            return
        loader = self.loaders.get(split) or self.loaders.get("val") or self.loaders.get("train")
        if loader is None or self.wandb is None:
            return
        try:
            import wandb

            from utilities.visualization import render_comparison_frames
            batch = next(iter(loader))
            frames, queries, tgt = self._prep(batch)
            # Use the unwrapped model to avoid DDP sync on a single-rank forward pass.
            model = self._unwrap_model()
            model.eval()
            with self._autocast():
                out = model(frames, queries, point_mask=tgt.get("point_mask"))
            # Render only the first clip, trimmed to a short, low-res sub-sequence
            # so the gif stays light to upload and quick to eyeball.
            tf = min(self.viz_frames, frames.shape[1]) if self.viz_frames > 0 else frames.shape[1]
            pred_vis = (torch.sigmoid(out["vis_logits"][0, :tf]) > 0.5)
            arr = render_comparison_frames(
                batch["frames"][0, :tf], tgt["tracks"][0, :tf].cpu(), out["coords"][0, :tf].cpu(),
                gt_visibility=tgt["visibility"][0, :tf].cpu(), pred_visibility=pred_vis.cpu(),
                max_points=self.viz_max_points, dpi=self.viz_dpi,
                title=f"{self.stage_name} {split} ep{epoch}",
            )
            self.wandb.log({f"{split}/tracks": wandb.Video(arr, fps=self.viz_fps, format="gif"),
                            "epoch": epoch})
        except Exception as e:  # noqa: BLE001
            logger.warning(f"video logging skipped ({e})")

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

    def _maybe_resume(self) -> None:
        """In-stage resume from ``last.pt``; else carry the previous stage's weights."""
        last = self.dir / "last.pt"
        m = self._unwrap_model()   # load into the underlying module (no "module." prefix)
        if last.exists():
            ck = torch.load(last, map_location=self.device, weights_only=False)
            # strict=False so a minor architecture change (e.g. the added gate head)
            # resumes from a compatible checkpoint instead of hard-failing.
            missing, unexpected = m.load_state_dict(ck["model"], strict=False)
            if missing or unexpected:
                logger.warning(f"resumed last.pt with strict=False "
                               f"(missing={len(missing)}, unexpected={len(unexpected)})")
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
            missing, unexpected = m.load_state_dict(ck["model"], strict=False)
            logger.info(f"carried weights from {prev.parent.name}/{prev.name} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # -- fit ----------------------------------------------------------------- #
    def fit(self) -> Dict[str, float]:
        self._maybe_resume()
        if self.start_epoch >= self.epochs:
            if self._is_main_process():
                logger.info(f"stage {self.stage_idx} already at epoch {self.start_epoch}/{self.epochs}; nothing to train")
                mark_stage_complete(self.run_dir, self.stage_idx, self.stage_name)
            if self._is_ddp:
                import torch.distributed as dist
                dist.barrier()
            return {"epe": self.best_metric}

        has_val = self.loaders.get("val") is not None
        if self._is_main_process():
            logger.info(f"training stage {self.stage_idx} '{self.stage_name}' for {self.epochs} epochs "
                        f"(amp={self.amp}/{self.amp_dtype if self.amp else '-'}, device={self.device.type}, "
                        f"world_size={self._world_size})")
        last_val: Dict[str, float] = {}
        for epoch in range(self.start_epoch, self.epochs):
            lr0, kw, tf_prob = self._apply_schedules(epoch)
            tr = self.train_epoch(epoch, tf_prob=tf_prob)
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
                                 **{f"train_{k}": v for k, v in tr.items()},
                                 **{f"val_{k}": v for k, v in val.items()}})
            self._log_console(epoch, lr0, kw, tf_prob, tr, val)
            if self._is_main_process() and self.wandb is not None:
                row = {"epoch": epoch, "stage": self.stage_idx, "lr": lr0,
                       "kl_weight": kw, "tf_prob": tf_prob,
                       **{f"train/{k}": v for k, v in tr.items()},
                       **{f"val/{k}": v for k, v in val.items()}}
                self.wandb.log(row)
            if self.viz_every and (epoch % self.viz_every == 0 or epoch == self.epochs - 1):
                self._log_video(epoch, split="val")     # qualitative val gif (rank-0 only)
                self._log_video(epoch, split="train")   # ... and a train-batch gif

            if self._is_main_process():
                torch.save(self._ckpt(epoch), self.dir / "last.pt")
                if improved:
                    torch.save(self._ckpt(epoch), self.dir / "best.pt")

            if self.patience and self._bad_epochs >= self.patience:
                if self._is_main_process():
                    logger.info(f"early stop: no improvement for {self.patience} epochs")
                break

        if self._is_main_process():
            mark_stage_complete(self.run_dir, self.stage_idx, self.stage_name)
            logger.info(f"stage {self.stage_idx} '{self.stage_name}' complete -- best epe={self.best_metric:.3f}px "
                        f"(checkpoints in {self.dir})")
        if self._is_ddp:
            import torch.distributed as dist
            dist.barrier()  # all ranks wait before returning (next stage must see completion marker)
        return last_val or {"epe": self.best_metric}

    def _log_console(self, epoch, lr, kw, tf_prob, tr, val) -> None:
        if not self._is_main_process():
            return
        line = f"[stage{self.stage_idx} {self.stage_name}] epoch {epoch + 1}/{self.epochs}"
        if tr:
            line += (f"  train: loss={tr.get('loss', float('nan')):.3f} epe={tr.get('epe', float('nan')):.2f}px "
                     f"mr={tr.get('motion_ratio', float('nan')):.2f} gate={tr.get('gate', float('nan')):.2f} "
                     f"|g|={tr.get('grad_norm', float('nan')):.2f}")
        if val:
            line += (f"  val: loss={val.get('loss', float('nan')):.3f} epe={val.get('epe', float('nan')):.2f}px "
                     f"mr={val.get('motion_ratio', float('nan')):.2f} stuck={val.get('stuck_frac', float('nan')):.2f} "
                     f"δ={val.get('delta_avg', float('nan')):.3f} OA={val.get('occlusion_accuracy', float('nan')):.3f} "
                     f"AJ={val.get('average_jaccard', float('nan')):.3f}")
        line += f"  lr={lr:.2e} kl_w={kw:.3f} tf={tf_prob:.2f}  best_epe={self.best_metric:.2f}"
        logger.info(line)
