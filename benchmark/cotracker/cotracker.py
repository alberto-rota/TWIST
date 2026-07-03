#!/usr/bin/env python
"""Benchmark CoTracker3 on the TWIST eval datasets, logged to W&B as ``cotracker3``.

Runs Meta's **CoTracker3** through the *exact same* evaluator the TWIST model
uses (:mod:`utilities.evaluation`) so the numbers are directly comparable. All
the shared plumbing (CLI, W&B run, evaluate-and-report) lives in
:mod:`benchmark.common`; this file only builds the model adapter — a thin wrapper
(:class:`CoTrackerAdapter`) mapping ``CoTrackerPredictor`` to the TWIST forward
contract ``model(frames, queries, point_mask=None) -> {"coords", "vis_logits"}``.

    python benchmark/cotracker/cotracker.py                       # all eval datasets -> W&B 'cotracker3'
    python benchmark/cotracker/cotracker.py --no-wandb            # CSV only
    python benchmark/cotracker/cotracker.py --datasets TAPVID_DAVIS --max-clips 5   # smoke
    python benchmark/cotracker/cotracker.py --checkpoint weights/cotracker/scaled_offline.pth
    python benchmark/cotracker/cotracker.py --image-size 512      # or set IMAGE_SIZE in cotracker.yaml

Heavy: needs a GPU. On the FAU nodes export ``http_proxy=http://proxy.nhr.fau.de:80``
for the one-time CoTracker checkpoint download (cached to ``COTRACKER.CHECKPOINT``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

# Bootstrap sys.path so `import common` (and, through it, `utilities`) resolves:
# repo root for utilities.*, <repo>/benchmark for common itself. setup_method_paths
# then adds the CoTracker source and drops this script's own dir (so this file,
# cotracker.py, can't shadow the upstream `cotracker` package).
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__, "co-tracker")

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "cotracker" / "cotracker.yaml"


# --------------------------------------------------------------------------- #
# Model adapter: CoTrackerPredictor -> the TWIST model forward contract
# --------------------------------------------------------------------------- #
class CoTrackerAdapter(torch.nn.Module):
    """Wrap a CoTracker predictor so the TWIST evaluator can drive it unchanged.

    The evaluator calls ``model(frames, queries, point_mask=None)`` and expects a
    dict ``{"coords": (B,T,N,2) px, "vis_logits": (B,T,N)}``. CoTracker returns
    ``(tracks, vis_bool)``; we map the boolean visibility to a hard logit so the
    evaluator's ``sigmoid(logit) > 0.5`` reproduces CoTracker's decision exactly.

    Two backends, selected by ``online``:

    * **offline** (``CoTrackerPredictor``) — one forward over the whole clip.
    * **online** (``CoTrackerOnlinePredictor``) — the *streaming* model, fed a
      sliding window of ``2*step`` frames advancing by ``step`` (the protocol in
      ``co-tracker/online_demo.py``); the causal, real-time variant.

    ``point_mask`` is accepted and ignored: CoTracker tracks every query it is
    given, and the evaluator masks padded / non-evaluated points itself.
    """

    VIS_LOGIT = 10.0
    # CoTracker's predictors (offline and online alike) accept queries at arbitrary
    # per-point frames, so the "queried first" evaluator can track every point in ONE
    # forward (each at its own first-visible frame) instead of one forward per distinct
    # first-visible frame — far fewer passes on multi-group clips, and more faithful
    # (all points tracked jointly, as CoTracker's own eval does). See
    # utilities.evaluation._first_visible_eval.
    supports_query_times = True

    def __init__(self, predictor: torch.nn.Module, *, online: bool,
                 backward_tracking: bool = False, add_support_grid: bool = True):
        super().__init__()
        self.predictor = predictor
        self.online = bool(online)
        self.backward_tracking = bool(backward_tracking)
        self.add_support_grid = bool(add_support_grid)

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = common.frames_to_255_float(frames)     # CoTracker wants [0,255] float
        queries = queries.float()
        # CoTracker's predictors (and the support-grid concat) assume a batch of 1,
        # so process each clip independently and re-stack.
        B = video.shape[0]
        outs = [self._forward_one(video[b:b + 1], queries[b:b + 1]) for b in range(B)]
        tracks = torch.cat([o[0] for o in outs], dim=0)   # (B,T,N,2)
        vis = torch.cat([o[1] for o in outs], dim=0)      # (B,T,N)
        vis_logits = common.vis_bool_to_logits(vis, tracks, self.VIS_LOGIT)
        return {"coords": tracks.float(), "vis_logits": vis_logits}

    @torch.no_grad()
    def _forward_one(self, video, queries):
        """Track one clip (``video (1,T,3,H,W)`` / ``queries (1,N,3)``) -> ``(tracks
        (1,T,N,2), vis (1,T,N))``. CoTracker is batch-1 only, so this is the unit the
        batched :meth:`forward` loops over."""
        if self.online:
            return self._online_forward(video, queries)
        return self.predictor(
            video, queries=queries, backward_tracking=self.backward_tracking)

    @torch.no_grad()
    def _online_forward(self, video, queries):
        """Stream ``video`` (1,T,3,H,W) through ``CoTrackerOnlinePredictor`` and
        return the full-length ``(tracks (1,T,N,2), vis (1,T,N))``.

        Mirrors ``co-tracker/online_demo.py`` exactly: the model is initialised
        once (``is_first_step=True`` registers the queries + grid, returns
        ``None``), then fed windows of ``2*step`` frames advancing by ``step``;
        the streaming model accumulates and returns the trajectory over all frames
        seen. A final tail call covers a length not divisible by ``step``.
        """
        pred = self.predictor
        s = int(pred.step)
        T = int(video.shape[1])
        asg = self.add_support_grid
        tracks = vis = None
        is_first = True
        i = 0
        for i in range(T):
            if i % s == 0 and i != 0:
                chunk = video[:, max(0, i - 2 * s):i]
                tracks, vis = pred(chunk, is_first_step=is_first, queries=queries,
                                   grid_size=0, add_support_grid=asg)
                is_first = False
        tail_lo = max(0, T - (i % s) - s - 1)
        tracks, vis = pred(video[:, tail_lo:], is_first_step=is_first, queries=queries,
                           grid_size=0, add_support_grid=asg)
        if tracks is None:                              # clip shorter than one window
            pred(video, is_first_step=True, queries=queries, grid_size=0, add_support_grid=asg)
            tracks, vis = pred(video, is_first_step=False, queries=queries,
                               grid_size=0, add_support_grid=asg)
        return tracks, vis


def build_adapter(cfg: Any, device: torch.device) -> CoTrackerAdapter:
    """Build the CoTracker3 predictor (loading / downloading weights) and wrap it
    in :class:`CoTrackerAdapter`, moved to ``device`` and in eval mode.

    The variant (``COTRACKER.VARIANT``) selects offline vs **online**: a name
    containing ``online`` builds the streaming ``CoTrackerOnlinePredictor``."""
    from cotracker.predictor import CoTrackerOnlinePredictor, CoTrackerPredictor
    from utilities.env import expand_path

    ct = cfg.get("COTRACKER", {})
    variant = str(ct.get("VARIANT", "scaled_online")).lower()
    online = "online" in variant
    default_ck = f"weights/cotracker/{variant}.pth"
    ck_raw = ct.get("CHECKPOINT") or default_ck
    url = ct.get("CHECKPOINT_URL") or (
        f"https://huggingface.co/facebook/cotracker3/resolve/main/{variant}.pth")
    window_len = int(ct.get("WINDOW_LEN", 16 if online else 60))
    backward = bool(ct.get("BACKWARD_TRACKING", False))
    kind = "online" if online else "offline"

    def _make(checkpoint):
        if online:
            return CoTrackerOnlinePredictor(checkpoint=checkpoint, offline=False,
                                            window_len=window_len, v2=False)
        return CoTrackerPredictor(checkpoint=checkpoint, offline=True,
                                  window_len=window_len, v2=False)

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if ckpt.exists():
        logger.info(f"loading CoTracker3 {kind} from {ckpt}")
        predictor = _make(str(ckpt))
    else:
        logger.info(f"checkpoint {ckpt} absent -> downloading {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        predictor = _make(None)
        predictor.model.load_state_dict(state_dict)
        try:                                            # cache for next time
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, ckpt)
            logger.info(f"cached checkpoint -> {ckpt}")
        except OSError as e:
            logger.warning(f"could not cache checkpoint ({e})")

    adapter = CoTrackerAdapter(predictor, online=online,
                               backward_tracking=backward).to(device).eval()
    logger.info(f"CoTracker3 {kind} ready on {device} (variant={variant}, "
                f"window_len={window_len}"
                + (f", step={predictor.step}" if online else f", backward_tracking={backward}")
                + ")")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark CoTracker3 on the TWIST eval datasets",
        checkpoint_key="COTRACKER.CHECKPOINT",
        default_name="cotracker3",
    ))
