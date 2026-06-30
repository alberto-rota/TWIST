#!/usr/bin/env python
"""Benchmark CoTracker3 on the TWIST eval datasets, logged to W&B as ``cotracker3``.

This runs Meta's **CoTracker3** (offline, scaled) through the *exact same*
evaluator the TWIST model uses (:mod:`utilities.evaluation`), so the reported
numbers are directly comparable to the TWIST runs' ``eval/*`` metrics: same
datasets (whatever the registry flags ``IS_EVAL_DATASET``), same TAP-Vid
"queried first" protocol, same metric definitions (``models.metrics``), same CSV
and W&B table. The *only* thing swapped is the model — a thin adapter
(:class:`CoTrackerAdapter`) wraps ``CoTrackerPredictor`` to the TWIST forward
contract ``model(frames, queries, point_mask=None) -> {"coords", "vis_logits"}``.

This is the methodology CLAUDE.md prescribes for baselines: the surgical
benchmarks have no published CoTracker number, so the comparison point is
*zero-shot CoTracker3 run through this same evaluator*.

    python benchmark/cotracker.py                         # all eval datasets -> W&B run 'cotracker3'
    python benchmark/cotracker.py --no-wandb              # CSV only, no W&B
    python benchmark/cotracker.py --datasets TAPVID_DAVIS --max-clips 5   # quick smoke
    python benchmark/cotracker.py --checkpoint weights/cotracker/scaled_offline.pth
    python benchmark/cotracker.py --config benchmark/cotracker.yaml --MODEL... (any --KEY override)

Outputs land under ``$RESULTS_DIR/<EXPERIMENT_NAME>/`` (``evaluation.csv`` +
recovery CSVs) and, with W&B on, an ``eval/metrics`` table + ``eval/<ds>/<metric>``
scalars on a run named ``cotracker3`` tagged ``benchmark`` in the TWIST project.

Heavy: needs a GPU. On the FAU nodes export ``http_proxy=http://proxy.nhr.fau.de:80``
for the one-time CoTracker checkpoint download (cached to ``COTRACKER.CHECKPOINT``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
COTRACKER_DIR = REPO_ROOT / "co-tracker"
# This file is literally named cotracker.py, and Python puts the running script's
# own directory (benchmark/) on sys.path[0] — which would shadow the real
# `cotracker` package (co-tracker/cotracker/) with this very file. Drop the
# script dir, then put co-tracker + repo root at the front.
_HERE = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path if p not in ("", ".", _HERE)]
for p in (str(REPO_ROOT), str(COTRACKER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utilities.config import load_and_process_config            # noqa: E402
from utilities.env import expand_path, load_env                 # noqa: E402
from utilities.evaluation import evaluate_and_report            # noqa: E402
from utilities.log import get_logger                            # noqa: E402

logger = get_logger(__name__).set_context("BENCH")

DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "cotracker.yaml"


# --------------------------------------------------------------------------- #
# Model adapter: CoTrackerPredictor -> the TWIST model forward contract
# --------------------------------------------------------------------------- #
class CoTrackerAdapter(torch.nn.Module):
    """Wrap ``CoTrackerPredictor`` so the TWIST evaluator can drive it unchanged.

    The evaluator calls ``model(frames, queries, point_mask=None)`` and expects a
    dict ``{"coords": (B,T,N,2) px, "vis_logits": (B,T,N)}`` (see
    :class:`models.world_model.TrackerWorldModel`). CoTracker returns
    ``(tracks, vis_bool)``; we map the boolean visibility to a hard logit
    (``±VIS_LOGIT``) so the evaluator's ``sigmoid(logit) > 0.5`` threshold
    reproduces CoTracker's own visibility decision exactly.

    Frames arrive as the reader yields them — uint8 ``[0,255]`` by default (or
    float ``[0,1]`` if a config set ``FRAMES_AS_FLOAT``). CoTracker wants
    ``[0,255]`` float (it does ``2*(video/255)-1`` internally), so we cast to
    float and rescale a ``[0,1]`` tensor up by 255. ``point_mask`` is accepted and
    ignored: CoTracker tracks every query it is given, and the evaluator masks out
    padded / non-evaluated points itself.
    """

    VIS_LOGIT = 10.0

    def __init__(self, predictor: torch.nn.Module, *, backward_tracking: bool = False):
        super().__init__()
        self.predictor = predictor
        self.backward_tracking = bool(backward_tracking)

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = frames.float()
        if torch.is_floating_point(frames) and float(video.max()) <= 1.5:
            video = video * 255.0                       # reader gave [0,1] -> CoTracker wants [0,255]
        tracks, vis = self.predictor(
            video, queries=queries.float(), backward_tracking=self.backward_tracking,
        )
        vis_logits = torch.where(
            vis.bool(),
            tracks.new_full((), self.VIS_LOGIT),
            tracks.new_full((), -self.VIS_LOGIT),
        )
        return {"coords": tracks.float(), "vis_logits": vis_logits.float()}


def build_cotracker_adapter(cfg: Any, device: torch.device) -> CoTrackerAdapter:
    """Build the CoTracker3 offline predictor (loading / downloading weights) and
    wrap it in :class:`CoTrackerAdapter`, moved to ``device`` and in eval mode."""
    from cotracker.predictor import CoTrackerPredictor

    ct = cfg.get("COTRACKER", {})
    ck_raw = ct.get("CHECKPOINT") or "weights/cotracker/scaled_offline.pth"
    url = ct.get("CHECKPOINT_URL") or (
        "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth")
    window_len = int(ct.get("WINDOW_LEN", 60))
    backward = bool(ct.get("BACKWARD_TRACKING", False))

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if ckpt.exists():
        logger.info(f"loading CoTracker3 offline from {ckpt}")
        predictor = CoTrackerPredictor(checkpoint=str(ckpt), offline=True,
                                       window_len=window_len, v2=False)
    else:
        logger.info(f"checkpoint {ckpt} absent -> downloading {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        predictor = CoTrackerPredictor(checkpoint=None, offline=True,
                                       window_len=window_len, v2=False)
        predictor.model.load_state_dict(state_dict)
        try:                                            # cache for next time
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, ckpt)
            logger.info(f"cached checkpoint -> {ckpt}")
        except OSError as e:
            logger.warning(f"could not cache checkpoint ({e})")

    adapter = CoTrackerAdapter(predictor, backward_tracking=backward).to(device).eval()
    logger.info(f"CoTracker3 offline ready on {device} "
                f"(window_len={window_len}, backward_tracking={backward})")
    return adapter


# --------------------------------------------------------------------------- #
# W&B run (named 'cotracker3', tagged 'benchmark') — opened here so we control
# the name + tags (utilities.engine.init_wandb does not pass tags).
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
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Benchmark CoTracker3 on the TWIST eval datasets")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help=f"benchmark config YAML (default {DEFAULT_CONFIG})")
    p.add_argument("--checkpoint", default=None,
                   help="CoTracker .pth (overrides COTRACKER.CHECKPOINT; downloaded if absent)")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names (default: all IS_EVAL_DATASET)")
    p.add_argument("--max-clips", type=int, default=None, help="cap clips per dataset")
    p.add_argument("--batch-size", type=int, default=None, help="eval batch size (default: config)")
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


def main() -> int:
    load_env()
    args, unknown = _build_parser().parse_known_args()

    overrides = list(unknown)
    if args.checkpoint:
        overrides += [f"--COTRACKER.CHECKPOINT={args.checkpoint}"]
    cfg = load_and_process_config(config_path=args.config, unknown_args=overrides)

    device = torch.device("cpu") if args.cpu else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        logger.warning("running on CPU — CoTracker is slow here; use for smoke checks only")

    name = args.name or str(cfg.get("EXPERIMENT_NAME", "cotracker3"))
    tags = ([t.strip() for t in args.tags.split(",") if t.strip()] if args.tags
            else list(cfg.get("WANDB_TAGS", ["benchmark"])) or ["benchmark"])

    run_dir = Path(args.out_dir) if args.out_dir else Path(
        expand_path(f"$RESULTS_DIR/{name}"))
    run_dir.mkdir(parents=True, exist_ok=True)

    adapter = build_cotracker_adapter(cfg, device)

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


if __name__ == "__main__":
    sys.exit(main())
