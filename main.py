#!/usr/bin/env python
"""TWIST pipeline entry point.

``run_pipeline`` is the single orchestrator shared by:
  * ``python train.py [config.yaml] [--KEY=val ...] [-b]``  (CLI)
  * ``sweep_agent.py``  (W&B sweep -> passes ``config=wandb.config``)

Each schedule **stage** runs end to end: load config -> build datasets ->
dataloaders -> world model + loss -> :class:`utilities.engine.Engine` trains it,
checkpoints into the run dir, and marks the stage complete so a resumed run
continues at the next stage (carrying the previous stage's weights). W&B is
opened once per run (or reused inside a sweep) and logs every stage.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta
from typing import Any, Dict, Optional

import torch.distributed as dist

from utilities.config import (
    build_dataloaders,
    create_datasets_from_config,
    create_loss_from_config,
    create_model_from_config,
    get_stages,
    load_and_process_config,
    resolve_stage_config,
)
from utilities.engine import Engine, finish_wandb, init_wandb
from utilities.env import load_env
from utilities.log import get_logger
from utilities.runs import (
    first_incomplete_stage,
    load_run_state,
    resolve_experiment_name,
    resolve_run_dir,
    resolve_source_run_dir,
)

logger = get_logger(__name__).set_context("MAIN")


def _build_parser(mode: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"{mode.capitalize()} TWIST")
    p.add_argument("config_file", nargs="?", default=None,
                   help="path to a config YAML (defaults to config/<mode>.yaml)")
    p.add_argument("--config", "-c", default=None, help="config YAML (overridden by positional)")
    p.add_argument("--boot", "-b", action="store_true",
                   help="boot mode: tiny CPU-sanity run (cnn encoder + shrunk model)")
    p.add_argument("--smoke", "-s", action="store_true",
                   help="smoke mode: real config, 1 epoch, batch 1, very few train/val iters + clips")
    p.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    p.add_argument("--max-batches", type=int, default=2,
                   help="batches to stream per split in the data dry-run")
    p.add_argument("--resume-run", default=None, metavar="NAME",
                   help="continue an existing run at its first unfinished stage")
    p.add_argument("--start-stage", type=int, default=None,
                   help="explicitly start the schedule at this stage index")
    p.add_argument("--single", "--singlegpu", action="store_true", help="force single-GPU (disable DDP even if torchrun sets RANK)")
    p.add_argument("--ddp", action="store_true",
                   help="enable DistributedDataParallel; requires launching with torchrun")
    return p


def _resolve_config_path(path: Optional[str], mode: str) -> str:
    """Search cwd then config/ for the YAML; default to config/<mode>.yaml."""
    path = path or os.path.join("config", f"{mode}.yaml")
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    for d in ("", "config", "configs"):
        cand = os.path.join(os.getcwd(), d, os.path.basename(path)) if d else os.path.join(os.getcwd(), path)
        if os.path.isfile(cand):
            return cand
    return path  # let the loader raise a clear FileNotFoundError


def _dryrun_split(loaders: Dict[str, Any], split: str, max_batches: int) -> None:
    """Stream up to ``max_batches`` batches and report shapes / stats / timing."""
    import torch

    dl = loaders.get(split)
    if dl is None:
        logger.info(f"[{split}] empty -- skipped")
        return
    logger.info(f"[{split}] {len(dl.dataset)} clips in {len(dl)} batches; streaming {min(max_batches, len(dl))} ...")
    t0 = time.time()
    for bi, batch in enumerate(dl):
        if bi >= max_batches:
            break
        if bi == 0:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    extra = ""
                    if k == "tracks":
                        extra = f"  x[{v[...,0].min():.0f},{v[...,0].max():.0f}] y[{v[...,1].min():.0f},{v[...,1].max():.0f}]"
                    elif k == "visibility":
                        extra = f"  visible={v.float().mean():.1%}"
                    logger.info(f"    {k:11s} {tuple(v.shape)!s:22s} {str(v.dtype):14s}{extra}")
                else:
                    logger.info(f"    {k:11s} {v}")
    dt = time.time() - t0
    n = min(max_batches, len(dl))
    logger.info(f"[{split}] streamed {n} batches in {dt:.2f}s ({dt / max(n,1):.2f}s/batch)")


def _has_experiment_name(cfg: Any) -> bool:
    """True when the config carries a usable (non-empty) ``EXPERIMENT_NAME`` string."""
    raw = cfg.get("EXPERIMENT_NAME", None)
    name = str(raw).strip() if isinstance(raw, str) else ""
    return name not in ("", "None")


def _sync_run_name(name: str, is_ddp: bool, is_main: bool, device) -> str:
    """Broadcast rank-0's resolved run name to every rank (no-op without DDP)."""
    if not is_ddp:
        return name
    obj = [name if is_main else None]
    dist.broadcast_object_list(obj, src=0, device=device)
    return obj[0]


def _lr_scaling_factor(mode: str, world_size: int) -> float:
    """LR multiplier for a DDP ``world_size`` under the selected scaling rule.

    - ``"none"``   -> 1.0 (LR is invariant to GPU count). Use with **Regime A**:
      hold the *global* batch fixed by setting per-GPU ``BATCH_SIZE = global/N``.
      DDP averages gradients, so the per-step update is *mathematically identical*
      to the single-GPU run (no BatchNorm in this model to break the equivalence);
      you get ~Nx wall-clock for free and lose nothing in learning quality.
    - ``"sqrt"``   -> sqrt(N). Use with **Regime B** (per-GPU batch fixed, so the
      effective batch grows Nx). The square-root rule is the right one for adaptive
      optimizers (AdamW): Adam already normalizes by the gradient second moment, so
      the batch<->LR coupling is weaker than SGD's and the linear rule overshoots.
    - ``"linear"`` -> N. The Goyal et al. 2017 SGD rule. Kept for completeness;
      empirically it overshoots here (2-GPU 6e-4 underperformed the 3e-4 baseline,
      4-GPU 4e-3 diverged), so it is NOT the default.
    """
    if world_size <= 1:
        return 1.0
    mode = (mode or "none").strip().lower()
    if mode == "linear":
        return float(world_size)
    if mode == "sqrt":
        return float(world_size) ** 0.5
    if mode == "none":
        return 1.0
    raise ValueError(f"Unknown LR_SCALING mode {mode!r}; expected none|sqrt|linear")


def _apply_lr_scaling(cfg: Any, world_size: int, is_main: bool) -> None:
    """Scale the learning rate(s) for DDP per ``cfg['LR_SCALING']`` (default ``none``).

    The LR in the config is treated as the **single-GPU** value. On a single GPU
    (``world_size <= 1``) this is always a no-op regardless of mode. The factor is
    chosen by :func:`_lr_scaling_factor`; see its docstring for Regime A/B guidance.

    Mutates ``cfg`` in place. Applied to the resolved stage config, so a non-zero
    encoder LR (e.g. an unfrozen backbone in a fine-tuning stage) is scaled too; a
    frozen encoder (0.0) is left untouched.
    """
    mode = cfg.get("LR_SCALING", "none")
    factor = _lr_scaling_factor(mode, world_size)
    if factor == 1.0:
        if is_main and world_size > 1:
            logger.info(
                f"LR scaling: mode={mode!r} x {world_size} GPUs -> factor 1.0 "
                f"(LR unchanged; hold global batch fixed with BATCH_SIZE=global/N)"
            )
        return

    base_lr = cfg.get("LR", None)
    if base_lr is not None:
        scaled = float(base_lr) * factor
        cfg["LR"] = scaled
        if is_main:
            logger.info(
                f"LR scaling: mode={mode!r} config LR {float(base_lr):.3e} "
                f"(single-GPU) x {world_size} GPUs (factor {factor:.3f}) -> {scaled:.3e}"
            )

    mc = cfg.get("MODEL", None)
    enc = mc.get("RGB_ENCODER", None) if mc is not None else None
    if enc is not None:
        enc_lr = float(enc.get("RGB_ENCODER_LR", 0.0) or 0.0)
        if enc_lr > 0:
            enc_scaled = enc_lr * factor
            enc["RGB_ENCODER_LR"] = enc_scaled
            if is_main:
                logger.info(
                    f"LR scaling: mode={mode!r} config RGB_ENCODER_LR {enc_lr:.3e} "
                    f"(single-GPU) x {world_size} GPUs (factor {factor:.3f}) -> {enc_scaled:.3e}"
                )


def run_pipeline(mode: str = "train", config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the (currently data-only) TWIST pipeline.

    Args:
        mode: "train" or "test" (only the data path differs cosmetically for now).
        config: a flat ``{KEY: value}`` dict (e.g. from a W&B sweep). When given,
            CLI args are NOT parsed.

    Returns:
        ``{"config", "datasets", "loaders"}`` so callers (notebooks, the sweep
        agent, the future engine) can use the assembled objects directly.
    """
    load_env()

    resume_run = None
    start_stage = None
    force_single = False
    if config is not None:
        cfg = load_and_process_config(config=config)
        max_batches = 2
    else:
        args, unknown = _build_parser(mode).parse_known_args()
        cfg_path = _resolve_config_path(args.config_file or args.config, mode)
        cfg = load_and_process_config(
            config_path=cfg_path, unknown_args=unknown,
            boot_mode=args.boot, smoke_mode=args.smoke,
        )
        if args.no_wandb:
            cfg.NO_WANDB = True
        max_batches = args.max_batches
        resume_run = args.resume_run
        start_stage = args.start_stage
        force_single = getattr(args, "single", False)

    # ---------------------------------------------------------------------- #
    # DDP setup: activate when torchrun sets RANK env var (or --ddp was passed)
    # unless --single/--singlegpu was explicitly requested.
    # ---------------------------------------------------------------------- #
    import torch

    is_ddp = ("RANK" in os.environ) and not force_single
    rank = 0
    world_size = 1
    local_rank = 0
    if is_ddp:
        # Generous collective timeout (default 60 min, override via DDP_TIMEOUT_MIN).
        # The default 10 min is too short here: validate() ends in a single metrics
        # all-reduce after a long, no-sync val loop (ROLLOUT_EVAL doubles forwards,
        # MAX_VAL_STEPS uncapped), so cross-rank dataloading skew over the shared FS
        # can leave the fast rank waiting >10 min and trip the NCCL watchdog. The
        # EVAL_AT_END / EVAL_EVERY path is worse: ranks 1.. wait at _barrier() while
        # rank 0 runs the whole-dataset benchmark solo. This timeout governs both.
        ddp_timeout_min = float(os.environ.get("DDP_TIMEOUT_MIN", "60"))
        dist.init_process_group(backend="nccl",
                                timeout=timedelta(minutes=ddp_timeout_min))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        cfg.RANK = rank
        cfg.WORLD_SIZE = world_size
        cfg.LOCAL_RANK = local_rank
        if rank == 0:
            logger.info(f"DDP enabled: world_size={world_size}, backend=nccl, "
                        f"collective_timeout={ddp_timeout_min:.0f}min")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0

    # Set OMP_NUM_THREADS and intra-op threads: split available cores across ranks.
    # Overrides torchrun's OMP_NUM_THREADS=1 default (set per-process, so each rank
    # gets its fair share rather than fighting over all cores).
    try:
        affinity = os.sched_getaffinity(os.getpid())
        n_cores = len(affinity)
        omp_threads = max(1, n_cores // world_size)
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        torch.set_num_threads(omp_threads)
        if is_main:
            logger.info(
                f"CPU cores available: {n_cores}  OMP_NUM_THREADS={omp_threads} (cores/world_size)",
                context="DDP" if is_ddp else "MAIN",
            )
    except Exception:
        if is_main:
            logger.warning("Could not read CPU affinity; OMP_NUM_THREADS unchanged")

    # Resolve output run name: explicit EXPERIMENT_NAME wins; cross-run RESUME can
    # derive via RESUME_NAME template ({source}). --resume-run only adopts the
    # source name when EXPERIMENT_NAME is still unset after that.
    resolved_name = resolve_experiment_name(cfg)
    if resolved_name is not None:
        cfg.EXPERIMENT_NAME = resolved_name

    schedule_run_dir = None
    if resume_run:
        if not _has_experiment_name(cfg):
            cfg.EXPERIMENT_NAME = resume_run
        schedule_run_dir = resolve_source_run_dir(resume_run, create=False)
        if is_main and _has_experiment_name(cfg) and str(cfg.EXPERIMENT_NAME).strip() != resume_run:
            logger.info(
                f"--resume-run {resume_run!r}: schedule from source run; "
                f"writing to {cfg.EXPERIMENT_NAME!r}"
            )

    # When no EXPERIMENT_NAME is given, adopt W&B's friendly run name
    # (adjective-noun-number) so the local results dir matches the W&B run. That
    # name only exists after wandb.init, so rank 0 opens the run now (before
    # run_dir is known) and broadcasts the resolved name to the other ranks.
    wandb_run, owns_wandb = None, False
    if not _has_experiment_name(cfg):
        if is_main:
            wandb_run, owns_wandb = init_wandb(cfg, run_dir=None)
        name = getattr(wandb_run, "name", None) or time.strftime("run-%Y%m%d_%H%M%S")
        name = _sync_run_name(name, is_ddp, is_main, device)
        cfg.EXPERIMENT_NAME = name
        if wandb_run is not None:
            try:
                wandb_run.config.update({"EXPERIMENT_NAME": name}, allow_val_change=True)
            except Exception:  # noqa: BLE001
                pass

    run_dir = resolve_run_dir(cfg)
    stages = get_stages(cfg)

    # Where to start in the schedule: explicit flag > resume (first unfinished) > 0.
    if start_stage is not None:
        start = start_stage
    elif resume_run:
        src = schedule_run_dir if schedule_run_dir is not None else run_dir
        start = first_incomplete_stage(src, len(stages))
    else:
        start = 0

    if is_main:
        done = set(load_run_state(run_dir).get("completed", []))
        logger.info(f"=== TWIST [{mode}] :: {cfg.get('EXPERIMENT_NAME', 'run')}  ({len(stages)} stage(s)) ===")
        logger.info(f"run dir: {run_dir}")
        for i, s in enumerate(stages):
            flag = "done" if i in done else ("-> start" if i == start else "")
            logger.info(f"  stage {i}: {s.get('NAME', f'stage{i}')} {('['+flag+']') if flag else ''}")
    if start >= len(stages):
        if is_main:
            logger.info(f"all {len(stages)} stage(s) already complete -- nothing to do")
            finish_wandb(wandb_run, owns_wandb)
        if is_ddp:
            dist.destroy_process_group()
        return {"config": cfg, "run_dir": str(run_dir), "stages": [], "loaders": None}

    # One W&B run for the whole schedule, rank-0 only (already opened above for an
    # auto-named run; otherwise opened here, reusing a sweep's run if present).
    if is_main and wandb_run is None:
        wandb_run, owns_wandb = init_wandb(cfg, run_dir)

    last = None
    try:
        for i in range(start, len(stages)):
            scfg = resolve_stage_config(cfg, i)
            if is_main:
                logger.info(f"--- stage {i + 1}/{len(stages)} :: {scfg.STAGE_NAME} ---")
            # Scale the LR for DDP per LR_SCALING (default 'none'; config LR == single-GPU value).
            _apply_lr_scaling(scfg, world_size, is_main)
            datasets = create_datasets_from_config(scfg)
            loaders = build_dataloaders(scfg, datasets, rank=rank, world_size=world_size)
            if is_main:
                _dryrun_split(loaders, "train", max_batches)

            model = create_model_from_config(scfg, device)
            loss_fn = create_loss_from_config(scfg, device)

            # Wrap with DDP after moving to device (done inside create_model_from_config).
            if is_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP
                model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

            engine = Engine(
                scfg, model, loss_fn, loaders, device, run_dir,
                stage_idx=i, stage_name=scfg.STAGE_NAME, wandb_run=wandb_run,
                rank=rank, world_size=world_size,
            )
            metrics = engine.fit()
            last = {"config": scfg, "datasets": datasets, "loaders": loaders,
                    "model": model, "loss_fn": loss_fn, "device": str(device),
                    "metrics": metrics}
    finally:
        if is_main:
            finish_wandb(wandb_run, owns_wandb)
        if is_ddp:
            dist.destroy_process_group()

    if is_main:
        logger.info(f"=== run '{cfg.get('EXPERIMENT_NAME', 'run')}' done: all {len(stages)} stage(s) trained ===")
    return {"config": cfg, "run_dir": str(run_dir), "stages": stages, **(last or {})}
