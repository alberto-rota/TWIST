#!/usr/bin/env python
"""Benchmark MFT (Multi-Flow dense Tracker, WACV 2024) on the TWIST eval datasets,
logged to W&B as ``mft``.

MFT (Neoral, Šerých, Matas; arXiv:2305.12998) tracks *every pixel* by chaining
optical flow from the query frame to the current frame over multiple temporal
``deltas`` and picking, per pixel, the most reliable chain (lowest predicted
uncertainty, non-occluded). It is **dense and online/causal**: you ``init`` it on
a template frame and feed subsequent frames one at a time; any 2D query point on
the template frame is then read off the accumulated flow field. Benchmarked here
through the *same* evaluator the TWIST model uses (:mod:`utilities.evaluation`)
for directly-comparable numbers.

Because MFT chains flow strictly **forward from its template frame**, it cannot
place queries at arbitrary per-point times in one pass — so ``supports_query_times
= False`` and the "queried first" evaluator drives it with **one init+rollout per
distinct first-visible frame** (each group's points share a template frame). See
:class:`MFTAdapter`.

    python benchmark/mft/mft.py                       # all eval datasets -> W&B 'mft'
    python benchmark/mft/mft.py --no-wandb
    python benchmark/mft/mft.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Runs on the shared TWIST ``.venv`` (needs ``scipy`` + ``ipdb``, added to
``pyproject.toml``, on top of torch / einops / numpy / opencv). The RAFT-OU flow
checkpoint ships **inside the clone** at ``benchmark/methods/MFT/checkpoints/`` —
no download. MFT resolves its config/checkpoint by *relative* path, so the
tracker is built with the CWD temporarily set to the clone root.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__, "MFT")          # add benchmark/methods/MFT

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "mft" / "mft.yaml"
MFT_SRC_DIR = common.METHODS_DIR / "MFT"


@contextmanager
def _chdir(path: Path):
    """MFT's configs reference the flow config + RAFT checkpoint by *relative*
    path (``configs/flow/...``, ``checkpoints/...``), resolved against the CWD.
    Build the tracker with the CWD at the clone root so those resolve; the loaded
    RAFT weights live in memory afterwards, so tracking needs no CWD."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


# --------------------------------------------------------------------------- #
# Model adapter: MFT (dense flow-chaining tracker) -> the TWIST forward contract
# --------------------------------------------------------------------------- #
class MFTAdapter(torch.nn.Module):
    """Wrap MFT to the TWIST forward contract. MFT is dense, stateful and
    forward-only: ``tracker.init(bgr_frame)`` sets the template, each
    ``tracker.track(bgr_frame)`` advances one frame, and
    ``convert_to_point_tracking(meta.result, q_xy)`` reads the tracked position +
    occlusion of arbitrary template-frame query points ``(N,2)=(x,y)`` off the
    accumulated flow field. Frames are opencv ``(H,W,3)`` uint8 **BGR** at native
    resolution (MFT does not resize internally), so predictions line up with GT at
    the eval ``TARGET_SIZE`` directly — no coordinate rescale.

    ``supports_query_times = False``: MFT chains flow forward from the template,
    so a point can only be tracked on frames *after* its query frame. The
    evaluator therefore calls this adapter **once per distinct first-visible
    frame**, every point in the call sharing that query frame ``f``; we
    ``init`` at ``f`` and roll forward to the clip end, scattering each frame's
    tracked coords / visibility. (The loop over unique query frames inside is
    defensive — in the "queried first" protocol each call already carries a single
    ``f``.) Frames at or before ``f`` are filled with the query coordinate as a
    harmless placeholder — the evaluator only scores frames strictly after ``f``.

    Visibility = ``occlusion <= OCCLUSION_THRESHOLD`` (MFT's occlusion score is
    0..1, higher = more occluded; the demo thresholds at 0.5). ``point_mask`` is
    accepted and ignored; the evaluator masks padded / non-evaluated points."""

    VIS_LOGIT = 10.0
    supports_query_times = False

    def __init__(self, tracker, occlusion_threshold: float = 0.5):
        super().__init__()
        self.tracker = tracker
        self.occlusion_threshold = float(occlusion_threshold)

    @staticmethod
    def _frames_to_bgr(video_1thwc_or_frames: torch.Tensor):
        """Reader clip ``(T,3,H,W)`` (RGB, uint8 or float) -> list of ``T``
        contiguous ``(H,W,3)`` uint8 **BGR** numpy arrays (opencv convention MFT's
        RAFT wrapper expects; it flips BGR->RGB internally)."""
        vid = common.frames_to_255_float(video_1thwc_or_frames)     # (T,3,H,W) [0,255]
        vid = vid.round().clamp_(0, 255).to(torch.uint8)
        rgb = vid.permute(0, 2, 3, 1).cpu().numpy()                 # (T,H,W,3) RGB
        return [np.ascontiguousarray(rgb[t, :, :, ::-1]) for t in range(rgb.shape[0])]  # ->BGR

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        from MFT.point_tracking import convert_to_point_tracking

        B, T = frames.shape[:2]
        N = queries.shape[1]
        device = frames.device
        queries = queries.float()

        coords = torch.zeros((B, T, N, 2), dtype=torch.float32, device=device)
        vis = torch.zeros((B, T, N), dtype=torch.bool, device=device)

        for b in range(B):
            frames_bgr = self._frames_to_bgr(frames[b])
            q_t = queries[b, :, 0].round().long()                   # (N,) per-point query frame
            q_xy = queries[b, :, 1:3]                               # (N,2) x,y on the query frame
            for f in torch.unique(q_t).tolist():
                f = int(f)
                sel = (q_t == f).nonzero(as_tuple=True)[0]          # points queried at frame f
                group_xy = q_xy[sel]                                # (n,2) on device
                # Template frame: identity -> coords are the query coords, visible.
                coords[b, : f + 1, sel] = group_xy
                vis[b, f, sel] = True
                self.tracker.init(frames_bgr[f], start_frame_i=f)
                for tt in range(f + 1, T):
                    meta = self.tracker.track(frames_bgr[tt])
                    c, occ = convert_to_point_tracking(meta.result, group_xy)   # (n,2),(n,) numpy
                    coords[b, tt, sel] = torch.as_tensor(c, dtype=torch.float32, device=device)
                    vis[b, tt, sel] = torch.as_tensor(
                        occ, device=device) <= self.occlusion_threshold

        vis_logits = common.vis_bool_to_logits(vis, coords, self.VIS_LOGIT)
        return {"coords": coords, "vis_logits": vis_logits}


def build_adapter(cfg: Any, device: torch.device) -> MFTAdapter:
    mft = cfg.get("MFT", {})
    config_name = str(mft.get("CONFIG", "configs/MFT_cfg.py"))
    occ_thr = float(mft.get("OCCLUSION_THRESHOLD", 0.5))

    if device.type != "cuda":
        # MFT's RAFT wrapper hardcodes self.device='cuda' and calls .cuda() on tensors.
        raise RuntimeError("MFT requires a CUDA device (its RAFT flow wrapper is cuda-only). "
                           "Run on a GPU node (no --cpu path).")

    from MFT.config import load_config
    with _chdir(MFT_SRC_DIR):
        logger.info(f"loading MFT config {config_name} (CWD={MFT_SRC_DIR})")
        config = load_config(config_name)
        tracker = config.tracker_class(config)                     # RAFTWrapper loads its ckpt here
    logger.info(f"MFT ready on {device} (config={config_name}, deltas={list(config.deltas)}, "
                f"occlusion_threshold={occ_thr})")
    return MFTAdapter(tracker, occlusion_threshold=occ_thr)


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark MFT on the TWIST eval datasets",
        checkpoint_key=None,
        default_name="mft",
    ))
