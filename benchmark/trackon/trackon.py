#!/usr/bin/env python
"""Benchmark Track-On on the TWIST eval datasets, logged to W&B as ``track_on``.

Runs Track-On (online / causal point tracker) through the *same* evaluator the
TWIST model uses (:mod:`utilities.evaluation`) so the numbers are directly
comparable. Shared plumbing lives in :mod:`benchmark.common`; this file only
builds the adapter.

    python benchmark/trackon/trackon.py                       # all eval datasets -> W&B 'track_on'
    python benchmark/trackon/trackon.py --no-wandb
    python benchmark/trackon/trackon.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Heavy: needs a GPU, the Track-On checkpoint, and the gated DINOv3 backbone
(see benchmark/trackon/trackon.yaml).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__, "track_on")     # add benchmark/methods/track_on

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "trackon" / "trackon.yaml"


class TrackOnAdapter(torch.nn.Module):
    """Wrap Track-On's ``Predictor`` to the TWIST forward contract.

    Track-On already matches the TWIST layout closely: it takes ``video
    (1,T,3,H,W)`` float ``[0,255]`` and ``queries (1,N,3)=(t,x,y)`` in pixels, and
    returns ``tracks (1,T,N,2)`` xy + ``visibility (1,T,N)`` bool at the original
    pixel space. Only the batch dim differs — the predictor is batch-1, so we loop
    over clips and re-stack."""

    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        device = frames.device
        video = common.frames_to_255_float(frames)      # (B,T,3,H,W) float [0,255]
        queries = queries.float()                       # (B,N,3) = (t,x,y)
        B = video.shape[0]
        tr, vi = [], []
        for b in range(B):
            tracks, vis = self.predictor(video[b:b + 1], queries=queries[b:b + 1])
            tr.append(tracks)                           # (1,T,N,2) — Predictor returns CPU
            vi.append(vis)                              # (1,T,N)
        tracks = torch.cat(tr, dim=0).to(device)
        vis = torch.cat(vi, dim=0).to(device)
        return {"coords": tracks.float(), "vis_logits": common.vis_bool_to_logits(vis, tracks)}


def build_adapter(cfg: Any, device: torch.device) -> TrackOnAdapter:
    # Track-On ships top-level ``dataset`` / ``model`` / ``utils`` packages that
    # collide with TWIST's ``dataset/`` on sys.path — isolate like LocoTrack/Chrono.
    with common.import_isolated("track_on", "dataset", "model", "utils"):
        from model.trackon_predictor import Predictor
        from utils.train_utils import load_args_from_yaml
        from utilities.env import expand_path

        to = cfg.get("TRACKON", {})
        ck_raw = to.get("CHECKPOINT") or "weights/trackon/track_on_r.pt"
        cfg_path = to.get("CONFIG")
        support_grid = int(to.get("SUPPORT_GRID_SIZE", 20))

        ckpt = Path(expand_path(str(ck_raw)))
        if not ckpt.is_absolute():
            ckpt = REPO_ROOT / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Track-On checkpoint not found at {ckpt}. Download it first "
                "(see benchmark/trackon/trackon.yaml).")

        model_args = None
        if cfg_path:
            cfg_path = Path(expand_path(str(cfg_path)))
            if not cfg_path.is_absolute():
                cfg_path = REPO_ROOT / cfg_path
            model_args = load_args_from_yaml(str(cfg_path))
            logger.info(f"Track-On model args from {cfg_path}")
        else:
            logger.info("Track-On model args: built-in Predictor defaults (DINOv3-s+)")

        predictor = Predictor(model_args, checkpoint_path=str(ckpt),
                              support_grid_size=support_grid)
        adapter = TrackOnAdapter(predictor).to(device).eval()
        logger.info(f"Track-On ready on {device} (checkpoint={ckpt.name}, support_grid={support_grid})")
        return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark Track-On on the TWIST eval datasets",
        checkpoint_key="TRACKON.CHECKPOINT",
        default_name="track_on",
    ))
