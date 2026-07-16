#!/usr/bin/env python
"""Benchmark LiteTracker (MICCAI 2025) on the TWIST eval datasets, logged to
W&B as ``litetracker``.

LiteTracker (arXiv:2504.09904, ImFusion) is a **training-free runtime
re-optimisation of CoTracker3-online for low-latency tissue tracking**: it keeps
CoTracker3's *exact* weights (it loads the same ``scaled_online.pth`` — the
state_dict matches 0 missing / 0 unexpected) but rebuilds the inference loop to
run truly frame-by-frame with a temporal memory buffer + EMA-flow track
initialisation, ~7x faster than CoTracker3-online. Benchmarking it here through
the *same* evaluator the TWIST model uses (:mod:`utilities.evaluation`) makes its
numbers directly comparable, and — since it shares CoTracker3's weights — gives a
clean read on what the runtime optimisations cost (or don't) in accuracy vs the
``cotracker3`` online baseline already in this folder.

Shared plumbing lives in :mod:`benchmark.common`; this file only builds the
adapter.

    python benchmark/litetracker/litetracker.py                       # all eval datasets -> W&B 'litetracker'
    python benchmark/litetracker/litetracker.py --no-wandb
    python benchmark/litetracker/litetracker.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Runs on the shared TWIST ``.venv`` (the model needs only torch / einops / numpy /
cv2, all present) and reuses the cached CoTracker3 online checkpoint at
``weights/cotracker/scaled_online.pth`` — no separate weights download.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__, "lite-tracker") # add benchmark/methods/lite-tracker

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "litetracker" / "litetracker.yaml"


# --------------------------------------------------------------------------- #
# Model adapter: LiteTracker -> the TWIST model forward contract
# --------------------------------------------------------------------------- #
class LiteTrackerAdapter(torch.nn.Module):
    """Wrap LiteTracker to the TWIST forward contract. LiteTracker is online and
    stateful: it takes **one frame at a time**, ``(1,3,H,W)`` float in ``[0,255]``
    (it internally resizes to its ``model_resolution`` and rescales predicted
    coords back to ``(H,W)``), plus ``queries (1,N,3)=(t,x,y)`` in ``(H,W)`` pixel
    space passed on every call. Each call returns the *current* frame's
    ``coords (1,1,N,2)`` xy (original-input pixels), ``vis (1,1,N)`` bool (already
    thresholded internally: ``sigmoid(vis)*sigmoid(conf) > 0.6``) and ``conf``.

    Per-point query times are handled natively by the model: a point is
    initialised exactly when ``online_ind`` reaches its query frame, so passing
    the full query set once (each with its true query time) and stepping through
    the clip tracks every point in a single pass — the frames before a point's
    query are never scored under the "queried first" protocol. Hence
    ``supports_query_times = True`` and the whole clip runs in one frame-by-frame
    loop, same shortcut as CoTracker/TAPIR/TAPNext.

    The tracker's recurrent buffers are per-clip, so the batch is looped one clip
    at a time with ``init_video_online_processing()`` (reset) before each, same
    pattern as :class:`CoTrackerAdapter` / :class:`TapNextAdapter`.

    ``point_mask`` is accepted and ignored: LiteTracker tracks every query it is
    given, and the evaluator masks padded / non-evaluated points itself."""

    VIS_LOGIT = 10.0
    supports_query_times = True

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = common.frames_to_255_float(frames)      # (B,T,3,H,W) [0,255]
        queries = queries.float()
        B = video.shape[0]
        outs = [self._forward_one(video[b:b + 1], queries[b:b + 1]) for b in range(B)]
        tracks = torch.cat([o[0] for o in outs], dim=0)   # (B,T,N,2) xy
        vis = torch.cat([o[1] for o in outs], dim=0)      # (B,T,N) bool
        vis_logits = common.vis_bool_to_logits(vis, tracks, self.VIS_LOGIT)
        return {"coords": tracks.float(), "vis_logits": vis_logits}

    @torch.no_grad()
    def _forward_one(self, video: torch.Tensor, queries: torch.Tensor):
        """Track one clip. ``video (1,T,3,H,W)`` [0,255], ``queries (1,N,3)=(t,x,y)``
        pixel coords -> ``(tracks (1,T,N,2) xy, vis (1,T,N) bool)`` at the input
        ``(H,W)`` resolution."""
        T = int(video.shape[1])
        self.model.init_video_online_processing()       # reset recurrent buffers for this clip
        tracks, vis = [], []
        for t in range(T):
            coords_t, vis_t, _ = self.model(video[:, t], queries=queries)   # (1,1,N,2),(1,1,N)
            tracks.append(coords_t)
            vis.append(vis_t)
        tracks = torch.cat(tracks, dim=1)                # (1,T,N,2) xy
        vis = torch.cat(vis, dim=1).bool()               # (1,T,N)
        return tracks, vis


def build_adapter(cfg: Any, device: torch.device) -> LiteTrackerAdapter:
    from src.lite_tracker import LiteTracker
    from utilities.env import expand_path

    lt = cfg.get("LITETRACKER", {})
    # LiteTracker keeps CoTracker3's weights verbatim -> default to the checkpoint the
    # cotracker/ benchmark already caches; downloaded from CHECKPOINT_URL if still absent.
    ck_raw = lt.get("CHECKPOINT") or "weights/cotracker/scaled_online.pth"
    url = lt.get("CHECKPOINT_URL") or (
        "https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth")
    mr = lt.get("MODEL_RESOLUTION")
    model_resolution: Tuple[int, int] = (
        (int(mr[0]), int(mr[1])) if mr is not None else (384, 512))    # (H, W); CoTracker3 default

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if ckpt.exists():
        logger.info(f"loading LiteTracker (CoTracker3 online weights) from {ckpt}")
        state_dict = torch.load(str(ckpt), map_location="cpu")
    else:
        logger.info(f"checkpoint {ckpt} absent -> downloading {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        try:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, ckpt)
            logger.info(f"cached checkpoint -> {ckpt}")
        except OSError as e:
            logger.warning(f"could not cache checkpoint ({e})")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    model = LiteTracker(model_resolution=model_resolution)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        logger.info("load_state_dict: exact match (0 missing / 0 unexpected)")
    model = model.to(device).eval()

    adapter = LiteTrackerAdapter(model).to(device).eval()
    logger.info(f"LiteTracker ready on {device} (model_resolution={model_resolution})")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark LiteTracker on the TWIST eval datasets",
        checkpoint_key="LITETRACKER.CHECKPOINT",
        default_name="litetracker",
    ))
