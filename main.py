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
from utilities.runs import first_incomplete_stage, load_run_state, resolve_run_dir

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

    # A run is identified by EXPERIMENT_NAME; --resume-run points at an existing one.
    if resume_run:
        cfg.EXPERIMENT_NAME = resume_run

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
        start = first_incomplete_stage(run_dir, len(stages))
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
