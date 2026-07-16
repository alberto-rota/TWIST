#!/usr/bin/env python
"""Shared driver for the TWIST baseline benchmarks.

Every baseline (CoTracker, LocoTrack, Chrono, TAPIR, Track-On, TAPTR, ...) is run
through the *exact same* evaluator the TWIST model uses
(:func:`utilities.evaluation.evaluate_and_report`), so the reported numbers are
directly comparable to the TWIST runs' ``eval/*`` metrics: same datasets
(whatever the registry flags ``IS_EVAL_DATASET``), same TAP-Vid "queried first"
protocol, same metric definitions (:mod:`models.metrics`), same CSV + W&B table.
The *only* thing that changes per method is the model — a thin adapter wrapping
the upstream predictor to the TWIST forward contract::

    model(frames, queries, point_mask=None) -> {"coords": (B,T,N,2) px,
                                                 "vis_logits": (B,T,N)}

where ``frames`` is ``(B,T,3,H,W)`` uint8 ``[0,255]`` (or float ``[0,1]`` if a
config set ``FRAMES_AS_FLOAT``) and ``queries`` is ``(B,N,3) = (t,x,y)`` in
pixels (see :class:`models.world_model.TrackerWorldModel`).

A method's benchmark script only has to (1) fix ``sys.path`` so its upstream
package imports, (2) define a ``build_adapter(cfg, device) -> nn.Module``, and
(3) call :func:`run`. This module owns everything else the three-file pipeline
shares: the CLI, the geometry logging, device selection, the W&B run (named +
tagged here so the baselines group together), and the evaluate-and-report call.

This file is a *library*, not a script — import it, don't run it. Each method
script adds ``<repo>/benchmark`` to ``sys.path`` and does ``import common``.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional

import torch

from utilities.config import load_and_process_config
from utilities.env import expand_path, load_env
from utilities.evaluation import evaluate_and_report
from utilities.log import get_logger

logger = get_logger(__name__).set_context("BENCH")

# Repo root = two levels up from benchmark/common.py (benchmark/ -> repo).
REPO_ROOT = Path(__file__).resolve().parent.parent
METHODS_DIR = REPO_ROOT / "benchmark" / "methods"


# --------------------------------------------------------------------------- #
# sys.path setup — call from a method script *before* importing its package.
# --------------------------------------------------------------------------- #
def setup_method_paths(script_file: str, *src_subdirs: str) -> Path:
    """Put the repo root and a method's upstream source dir(s) on ``sys.path``,
    and drop the running script's own directory so a script named e.g.
    ``cotracker.py`` cannot shadow the upstream ``cotracker`` package.

    ``src_subdirs`` are paths under ``benchmark/methods`` (e.g. ``"co-tracker"``
    or ``"tapnet"``). Returns the repo root. Idempotent.
    """
    here = str(Path(script_file).resolve().parent)
    sys.path[:] = [p for p in sys.path if p not in ("", ".", here)]
    heads = [str(REPO_ROOT)] + [str(METHODS_DIR / s) for s in src_subdirs]
    for p in reversed(heads):           # preserve given order at the front
        if p not in sys.path:
            sys.path.insert(0, p)
    return REPO_ROOT


@contextmanager
def import_isolated(src_subdir: str, *shadow_names: str):
    """Import a method package whose **top-level module names collide with TWIST's**
    (e.g. Chrono ships a top-level ``models`` package, same name as ``models/``
    that the evaluator already imported).

    Inside the ``with`` block: the method's source dir is first on ``sys.path``,
    the repo root is temporarily removed (otherwise Python prefers TWIST's
    same-named *regular* packages over the method's namespace packages), and
    the colliding top-level names are removed from ``sys.modules`` so the
    method's versions import cleanly. On exit, ``sys.path`` and TWIST's modules
    are restored — the method objects you imported keep their already-bound
    references (e.g. Chrono's ``LocoTrack`` still points at Chrono's
    ``models.utils``), so the model works afterwards while the evaluator keeps
    TWIST's ``models``.

        with common.import_isolated("Chrono", "models", "model_utils"):
            from models.locotrack_model import LocoTrack
    """
    method_dir = str(METHODS_DIR / src_subdir)
    repo_root = str(REPO_ROOT)
    saved_path = list(sys.path)
    saved_mods = {}

    def _pop(names):
        out = {}
        for name in names:
            keys = [name] + [k for k in list(sys.modules) if k.startswith(name + ".")]
            for k in keys:
                out[k] = sys.modules.pop(k, None)
        return out

    saved_mods = _pop(shadow_names)
    # Drop repo root so a method dir without __init__.py (namespace package)
    # is not shadowed by TWIST's same-named regular package on sys.path.
    sys.path[:] = [p for p in sys.path if p not in (repo_root, "")]
    if method_dir not in sys.path:
        sys.path.insert(0, method_dir)
    try:
        yield
    finally:
        sys.path[:] = saved_path
        _pop(shadow_names)                      # drop the method's versions
        for k, v in saved_mods.items():         # restore TWIST's
            if v is not None:
                sys.modules[k] = v


# --------------------------------------------------------------------------- #
# W&B run — opened here so we control name + tags (engine.init_wandb does not
# pass tags). Named after the method, tagged 'benchmark' so baselines group up.
# --------------------------------------------------------------------------- #
def open_wandb_run(cfg: Any, run_dir: Path, name: str, tags: List[str]):
    try:
        import wandb
    except Exception as e:  # noqa: BLE001
        logger.warning(f"wandb unavailable ({e}); writing CSV only")
        return None
    try:
        cfg_dict = cfg.toDict() if hasattr(cfg, "toDict") else dict(cfg)
        run = wandb.init(
            project=str(cfg.get("WANDB_PROJECT", "twist")),
            entity=str(cfg.get("WANDB_ENTITY", "twisteam")),
            name=name,
            tags=tags,
            config=cfg_dict,
            dir=str(run_dir),
        )
        logger.info(f"W&B run '{name}' tags={tags} -> {run.url}")
        return run
    except Exception as e:  # noqa: BLE001
        logger.warning(f"wandb.init failed ({e}); writing CSV only")
        return None


# --------------------------------------------------------------------------- #
# CLI — identical across methods. ``checkpoint_key`` is the dotted config key the
# --checkpoint shortcut writes to (e.g. COTRACKER.CHECKPOINT); methods that load
# weights differently can pass checkpoint_key=None and read --checkpoint via cfg.
# --------------------------------------------------------------------------- #
def build_parser(default_config: Path, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(default_config),
                   help=f"benchmark config YAML (default {default_config})")
    p.add_argument("--checkpoint", default=None,
                   help="model checkpoint path (overrides the method's config key; downloaded if absent)")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names (default: all IS_EVAL_DATASET)")
    p.add_argument("--max-clips", type=int, default=None, help="cap clips per dataset")
    p.add_argument("--batch-size", type=int, default=None, help="eval batch size (default: config)")
    p.add_argument("--image-size", type=int, default=None,
                   help="square frame side in px (sets top-level IMAGE_SIZE -> dataset TARGET_SIZE)")
    p.add_argument("--crop", default=None,
                   help="native-pixel crop x0,y0,x1,y1 before resize (sets top-level CROP)")
    p.add_argument("--workers", type=int, default=None, help="dataloader workers (default: config)")
    p.add_argument("--max-steps", type=int, default=0, help="cap batches per dataset (0=all)")
    p.add_argument("--query-mode", choices=("first", "frame0"), default=None,
                   help="TAP-Vid query protocol (default: config EVAL_QUERY_MODE / 'first')")
    p.add_argument("--name", default=None, help="W&B run name (default: config EXPERIMENT_NAME)")
    p.add_argument("--tags", default=None, help="comma-separated W&B tags (default: config WANDB_TAGS)")
    p.add_argument("--out-dir", default=None, help="override where the CSVs are written")
    p.add_argument("--no-wandb", action="store_true", help="skip W&B, write CSV only")
    p.add_argument("--amp", dest="amp", action="store_true", default=None,
                   help="force autocast on (default: config EVAL_AMP / off)")
    p.add_argument("--cpu", action="store_true", help="force CPU (slow; smoke only)")
    return p


def _log_geometry(cfg: Any) -> None:
    oad = cfg.get("DATASETS", {}).get("OVERRIDE_ALL_DATASETS", {})
    crop = cfg.get("CROP")
    img_size = cfg.get("IMAGE_SIZE")
    target = oad.get("TARGET_SIZE") if oad else None
    crop_s = f"CROP={list(crop)}" if crop is not None else "CROP=off"
    if img_size is not None:
        logger.info(f"eval geometry: {crop_s}, IMAGE_SIZE={img_size} -> TARGET_SIZE={target}")
    elif crop is not None:
        logger.info(f"eval geometry: {crop_s}, resize=off (IMAGE_SIZE unset)")
    else:
        logger.info("eval geometry: native clip (CROP and IMAGE_SIZE unset)")


# --------------------------------------------------------------------------- #
# The shared main(). A method script calls this with its build_adapter callback.
# --------------------------------------------------------------------------- #
def run(
    build_adapter: Callable[[Any, torch.device], torch.nn.Module],
    *,
    default_config: Path,
    description: str,
    checkpoint_key: Optional[str] = None,
    default_name: str = "benchmark",
) -> int:
    """Load config + CLI overrides, build the method adapter, evaluate on the
    TWIST eval datasets, and report (CSV under the run dir + W&B table/scalars).

    ``build_adapter(cfg, device) -> nn.Module`` is the only per-method piece: it
    constructs the upstream predictor (loading / downloading weights) and wraps
    it to the TWIST forward contract. Returns a process exit code.
    """
    load_env()
    args, unknown = build_parser(default_config, description).parse_known_args()

    overrides = list(unknown)
    if args.checkpoint and checkpoint_key:
        overrides.append(f"--{checkpoint_key}={args.checkpoint}")
    if args.image_size is not None:
        overrides.append(f"--IMAGE_SIZE={args.image_size}")
    if args.crop is not None:
        parts = [c.strip() for c in args.crop.split(",")]
        if len(parts) != 4:
            raise SystemExit("--crop requires 4 comma-separated integers: x0,y0,x1,y1")
        overrides.append(f"--CROP=[{','.join(parts)}]")
    cfg = load_and_process_config(config_path=args.config, unknown_args=overrides)
    _log_geometry(cfg)

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("running on CPU — baselines are slow here; use for smoke checks only")

    name = args.name or str(cfg.get("EXPERIMENT_NAME", default_name))
    tags = ([t.strip() for t in args.tags.split(",") if t.strip()] if args.tags
            else list(cfg.get("WANDB_TAGS", ["benchmark"])) or ["benchmark"])

    run_dir = Path(args.out_dir) if args.out_dir else Path(expand_path(f"$RESULTS_DIR/{name}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = build_adapter(cfg, device)

    dataset_names = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    query_mode = args.query_mode or str(cfg.get("EVAL_QUERY_MODE", "first"))
    amp = args.amp if args.amp is not None else bool(cfg.get("EVAL_AMP", False))
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get("EVAL_BATCH_SIZE", 1))
    workers = args.workers if args.workers is not None else int(cfg.get("EVAL_WORKERS", 0))

    wandb_run = None if args.no_wandb else open_wandb_run(cfg, run_dir, name, tags)
    try:
        results = evaluate_and_report(
            adapter, cfg, device, run_dir,
            wandb_run=wandb_run,
            dataset_names=dataset_names,
            max_clips=args.max_clips,
            batch_size=batch_size,
            num_workers=workers,
            max_steps=args.max_steps,
            query_mode=query_mode,
            amp=amp,
        )
    finally:
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:  # noqa: BLE001
                pass

    if not results:
        logger.error("no results (no eval datasets present on disk?)")
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Small adapter helpers shared by several methods.
# --------------------------------------------------------------------------- #
def frames_to_255_float(frames: torch.Tensor) -> torch.Tensor:
    """Reader frames -> float ``[0,255]``. uint8 stays [0,255]; a float tensor the
    reader gave as ``[0,1]`` (FRAMES_AS_FLOAT) is scaled up by 255."""
    video = frames.float()
    if torch.is_floating_point(frames) and float(video.max()) <= 1.5:
        video = video * 255.0
    return video


def vis_bool_to_logits(vis: torch.Tensor, coords_like: torch.Tensor,
                       logit: float = 10.0) -> torch.Tensor:
    """Map a boolean/0-1 visibility tensor to hard ``±logit`` so the evaluator's
    ``sigmoid(logit) > 0.5`` reproduces the method's own visibility decision."""
    return torch.where(vis.bool(),
                       coords_like.new_full((), logit),
                       coords_like.new_full((), -logit)).float()


# --------------------------------------------------------------------------- #
# TAPIR-family helpers (LocoTrack / TAPIR / Chrono share these conventions:
# video as (B,T,H,W,3) in [-1,1], queries as (t,y,x), tracks as (B,N,T,2) xy).
# --------------------------------------------------------------------------- #
def frames_to_bthwc_norm(frames: torch.Tensor) -> torch.Tensor:
    """Reader frames ``(B,T,3,H,W)`` -> TAPIR-style ``(B,T,H,W,3)`` float in
    ``[-1,1]`` (``video/255*2-1``)."""
    video = frames_to_255_float(frames) / 255.0 * 2.0 - 1.0
    return video.permute(0, 1, 3, 4, 2).contiguous()


def queries_txy_to_tyx(queries: torch.Tensor) -> torch.Tensor:
    """TWIST queries ``(B,N,3)=(t,x,y)`` -> TAPIR ``(t,y,x)``."""
    return queries[..., [0, 2, 1]].contiguous()


def tapir_outputs_to_twist(tracks_bnt2: torch.Tensor, occlusion_logits: torch.Tensor,
                           expected_dist_logits: Optional[torch.Tensor] = None,
                           logit: float = 10.0) -> dict:
    """Map a TAPIR-family output (``tracks (B,N,T,2)`` xy + occlusion logits, and
    optionally an ``expected_dist`` uncertainty logit) to the TWIST contract
    ``{"coords": (B,T,N,2), "vis_logits": (B,T,N)}``.

    Visibility follows the TAPIR demos: ``visible = (1-sig(occ))*(1-sig(dist))
    > 0.5`` when ``expected_dist`` is given, else ``sig(occ) < 0.5`` (occlusion
    logit: higher = more occluded)."""
    coords = tracks_bnt2.permute(0, 2, 1, 3).contiguous()       # (B,N,T,2)->(B,T,N,2)
    if expected_dist_logits is not None:
        vis = (1 - torch.sigmoid(occlusion_logits)) * (1 - torch.sigmoid(expected_dist_logits)) > 0.5
    else:
        vis = torch.sigmoid(occlusion_logits) < 0.5
    vis = vis.permute(0, 2, 1).contiguous()                     # (B,N,T)->(B,T,N)
    return {"coords": coords.float(), "vis_logits": vis_bool_to_logits(vis, coords, logit)}
