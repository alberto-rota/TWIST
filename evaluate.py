#!/usr/bin/env python
"""Standalone evaluation entry point.

Loads a trained checkpoint, rebuilds the model from its embedded config, and
computes the headline TAP metrics (Delta AVG / Average Jaccard / Occlusion
Accuracy / ms-per-frame) on the ``IS_EVAL_DATASET``-flagged datasets, then
writes a CSV under the run dir and (optionally) a W&B table.

    # evaluate a specific checkpoint
    python evaluate.py results/<run>/stage0/best.pt

    # resolve best.pt (else last.pt) inside a run by EXPERIMENT_NAME
    python evaluate.py --run <EXPERIMENT_NAME>

    # restrict / cap / log to W&B
    python evaluate.py <ckpt> --datasets TAPVID_DAVIS,ROBOTAP --max-clips 50 --wandb

    # override any saved-config field (type-coerced), e.g. force the CPU encoder
    python evaluate.py <ckpt> --MODEL.RGB_ENCODER.ENCODER=cnn

The same scoring runs automatically inside training when ``EVAL_AT_END`` /
``EVAL_EVERY`` are set (see utilities.engine); this script is the offline path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from utilities.env import expand_path, load_env
from utilities.evaluation import evaluate_checkpoint
from utilities.log import get_logger

logger = get_logger(__name__).set_context("EVAL")


def _resolve_run_checkpoint(run_name: str, prefer: str = "best") -> Optional[Path]:
    """Find ``best.pt`` (else ``last.pt``) of the highest stage in a run dir.

    ``run_name`` is an EXPERIMENT_NAME under ``$RESULTS_DIR`` or a direct path to
    a run dir. Returns the checkpoint path, or None if none is found.
    """
    import os

    cand = Path(run_name)
    if not cand.is_dir():
        results = os.environ.get("RESULTS_DIR") or str(Path.cwd() / "results")
        cand = Path(expand_path(results)) / run_name
    if not cand.is_dir():
        return None
    order = ("best.pt", "last.pt") if prefer == "best" else ("last.pt", "best.pt")
    # highest stage first (carries the most-trained weights)
    for stage_dir in sorted(cand.glob("stage*"), reverse=True):
        for fn in order:
            if (stage_dir / fn).exists():
                return stage_dir / fn
    return None


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate a trained TWIST checkpoint")
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="path to a .pt checkpoint (or use --run)")
    p.add_argument("--run", default=None, metavar="NAME",
                   help="resolve best.pt (else last.pt) inside this run / EXPERIMENT_NAME")
    p.add_argument("--prefer", choices=("best", "last"), default="best",
                   help="which checkpoint to prefer when resolving --run (default best)")
    p.add_argument("--datasets", default=None,
                   help="comma-separated dataset names to evaluate (default: all IS_EVAL_DATASET)")
    p.add_argument("--max-clips", type=int, default=None,
                   help="cap clips per dataset (default: all)")
    p.add_argument("--batch-size", type=int, default=1, help="eval batch size (default 1)")
    p.add_argument("--workers", type=int, default=0, help="dataloader workers (default 0)")
    p.add_argument("--max-steps", type=int, default=0, help="cap batches per dataset (0=all)")
    p.add_argument("--tag", default="", help="CSV name tag -> evaluation_<tag>.csv")
    p.add_argument("--out-dir", default=None, help="override where the CSV is written")
    p.add_argument("--wandb", action="store_true", help="open a W&B run and log the table")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p


def main() -> int:
    load_env()
    args, unknown = _build_parser().parse_known_args()

    ckpt = args.checkpoint
    if ckpt is None and args.run:
        resolved = _resolve_run_checkpoint(args.run, prefer=args.prefer)
        if resolved is None:
            logger.error(f"no checkpoint found for run '{args.run}'")
            return 1
        ckpt = str(resolved)
        logger.info(f"resolved run '{args.run}' -> {ckpt}")
    if ckpt is None:
        logger.error("provide a checkpoint path or --run NAME")
        return 1
    if not Path(ckpt).exists():
        logger.error(f"checkpoint not found: {ckpt}")
        return 1

    device = None
    if args.cpu:
        import torch
        device = torch.device("cpu")

    dataset_names = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    results = evaluate_checkpoint(
        ckpt,
        device=device,
        unknown_args=unknown,
        out_dir=args.out_dir,
        use_wandb=args.wandb,
        tag=args.tag,
        dataset_names=dataset_names,
        max_clips=args.max_clips,
        batch_size=args.batch_size,
        num_workers=args.workers,
        max_steps=args.max_steps,
    )
    if not results:
        logger.error("evaluation produced no results (no eval datasets available?)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
