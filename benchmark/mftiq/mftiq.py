#!/usr/bin/env python
"""Benchmark MFTIQ (Multi-Flow Tracker with Independent Matching Quality, WACV
2025) on the TWIST eval datasets, logged to W&B as ``mftiq``.

MFTIQ (Šerých, Neoral, Matas; arXiv:2411.09551) is MFT's successor: same dense
forward flow-chaining backbone, but the per-pixel occlusion+uncertainty used to
select the best temporal ``delta`` is estimated by a **separate learned module
(UOM)** decoupled from the flow network, letting any off-the-shelf optical-flow
model be plugged in. This harness benchmarks the **RAFT-flow** variant
(``MFTIQ4_RAFT_200k_cfg.py``) — the closest apples-to-apples to the ``mft``
baseline in this folder — through the *same* evaluator the TWIST model uses
(:mod:`utilities.evaluation`).

Like MFT, MFTIQ chains flow **forward from its template frame**, so it cannot
place queries at arbitrary per-point times in one pass — ``supports_query_times
= False`` and the "queried first" evaluator drives it with one init+rollout per
distinct first-visible frame. The adapter is the MFT adapter with MFTIQ's
package; see :class:`MFTIQAdapter` and :mod:`benchmark.mft.mft`.

    python benchmark/mftiq/mftiq.py --datasets TAPVID_DAVIS --max-clips 3 --no-wandb   # smoke
    python benchmark/mftiq/mftiq.py                       # all eval datasets -> W&B 'mftiq'

Heavy deps: needs a **dedicated venv** with ``xformers`` / ``kornia`` / a compiled
``spatial-correlation-sampler`` and downloaded UOM + flow checkpoints. On FAU NHR
use ``bash benchmark/mftiq/setup_venv.sh`` (torch 2.1.2+cu121; upstream pins
2.0.1+cu117 which cannot compile extensions against the cluster CUDA toolkit).
Point the job at the venv via ``BENCH_VENV`` (see benchmark/README.md).
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
# MFTIQ's package lives under the clone's src/ (installed via `pip install .`, but
# add it to the path too so an editable / uninstalled checkout also imports).
REPO_ROOT = common.setup_method_paths(__file__, "MFTIQ/src")

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "mftiq" / "mftiq.yaml"
MFTIQ_SRC_DIR = common.METHODS_DIR / "MFTIQ"


@contextmanager
def _chdir(path: Path):
    """MFTIQ's config references the flow config + UOM / RAFT checkpoints by
    *relative* path (``configs/...``, ``checkpoints/...``), resolved against the
    CWD. Build the tracker with the CWD at the clone root so those resolve; the
    loaded weights live in memory afterwards, so tracking needs no CWD."""
    prev = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(prev)


# --------------------------------------------------------------------------- #
# Model adapter: MFTIQ (dense flow-chaining tracker) -> the TWIST forward contract
# --------------------------------------------------------------------------- #
class MFTIQAdapter(torch.nn.Module):
    """Wrap MFTIQ to the TWIST forward contract — identical driving to
    :class:`benchmark.mft.mft.MFTAdapter` (MFTIQ shares MFT's ``init`` / ``track``
    / ``convert_to_point_tracking`` API): dense, stateful, forward-only from a
    template frame; opencv ``(H,W,3)`` uint8 **BGR** frames at native resolution
    (no internal resize, so predictions line up with GT at ``TARGET_SIZE``).

    ``supports_query_times = False``: the evaluator calls this once per distinct
    first-visible frame ``f`` (every point in the call sharing ``f``); we ``init``
    at ``f`` and roll forward, scattering each frame's tracked coords / visibility.
    Frames at/before ``f`` are placeholder-filled (only frames strictly after
    ``f`` are scored). Visibility = ``occlusion <= OCCLUSION_THRESHOLD``.
    ``point_mask`` is accepted and ignored."""

    VIS_LOGIT = 10.0
    supports_query_times = False

    def __init__(self, tracker, occlusion_threshold: float = 0.5):
        super().__init__()
        self.tracker = tracker
        self.occlusion_threshold = float(occlusion_threshold)

    @staticmethod
    def _frames_to_bgr(clip_thwc: torch.Tensor):
        """Reader clip ``(T,3,H,W)`` (RGB, uint8 or float) -> list of ``T``
        contiguous ``(H,W,3)`` uint8 **BGR** numpy arrays."""
        vid = common.frames_to_255_float(clip_thwc)                 # (T,3,H,W) [0,255]
        vid = vid.round().clamp_(0, 255).to(torch.uint8)
        rgb = vid.permute(0, 2, 3, 1).cpu().numpy()                 # (T,H,W,3) RGB
        return [np.ascontiguousarray(rgb[t, :, :, ::-1]) for t in range(rgb.shape[0])]  # ->BGR

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        from MFTIQ.point_tracking import convert_to_point_tracking

        B, T = frames.shape[:2]
        N = queries.shape[1]
        device = frames.device
        queries = queries.float()

        coords = torch.zeros((B, T, N, 2), dtype=torch.float32, device=device)
        vis = torch.zeros((B, T, N), dtype=torch.bool, device=device)

        for b in range(B):
            frames_bgr = self._frames_to_bgr(frames[b])
            q_t = queries[b, :, 0].round().long()
            q_xy = queries[b, :, 1:3]
            for f in torch.unique(q_t).tolist():
                f = int(f)
                sel = (q_t == f).nonzero(as_tuple=True)[0]
                group_xy = q_xy[sel]
                coords[b, : f + 1, sel] = group_xy
                vis[b, f, sel] = True
                self.tracker.init(frames_bgr[f], start_frame_i=f)
                for tt in range(f + 1, T):
                    meta = self.tracker.track(frames_bgr[tt])
                    c, occ = convert_to_point_tracking(meta.result, group_xy)
                    coords[b, tt, sel] = torch.as_tensor(c, dtype=torch.float32, device=device)
                    vis[b, tt, sel] = torch.as_tensor(
                        occ, device=device) <= self.occlusion_threshold

        vis_logits = common.vis_bool_to_logits(vis, coords, self.VIS_LOGIT)
        return {"coords": coords, "vis_logits": vis_logits}


def build_adapter(cfg: Any, device: torch.device) -> MFTIQAdapter:
    mq = cfg.get("MFTIQ", {})
    config_name = str(mq.get("CONFIG", "configs/MFTIQ4_RAFT_200k_cfg.py"))
    occ_thr = float(mq.get("OCCLUSION_THRESHOLD", 0.5))

    if device.type != "cuda":
        raise RuntimeError("MFTIQ requires a CUDA device (its flow + UOM modules are cuda-only). "
                           "Run on a GPU node (no --cpu path).")

    from MFTIQ.config import load_config
    with _chdir(MFTIQ_SRC_DIR):
        logger.info(f"loading MFTIQ config {config_name} (CWD={MFTIQ_SRC_DIR})")
        config = load_config(config_name)
        tracker = config.tracker_class(config)                     # flow + UOM weights load here
    logger.info(f"MFTIQ ready on {device} (config={config_name}, deltas={list(config.deltas)}, "
                f"occlusion_threshold={occ_thr})")
    return MFTIQAdapter(tracker, occlusion_threshold=occ_thr)


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark MFTIQ on the TWIST eval datasets",
        checkpoint_key=None,
        default_name="mftiq",
    ))
