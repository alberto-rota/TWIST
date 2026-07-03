#!/usr/bin/env python
"""Benchmark TAPNext (BootsTAPNext, PyTorch) on the TWIST eval datasets, logged
to W&B as ``tapnext``.

Runs DeepMind's TAPNext (arXiv:2504.05579) through the *exact same* evaluator
the TWIST model uses (:mod:`utilities.evaluation`) so the numbers are directly
comparable. Shared plumbing lives in :mod:`benchmark.common`; this file only
builds the model adapter.

TAPNext is **online / causal by construction**: it tracks by propagating a
recurrent state one frame at a time (no offline whole-clip mode exists). It
also natively supports **per-point query times** within that same frame-by-frame
pass — a point not yet due is embedded as an "unknown" token and only starts
predicting at its own query frame (see ``TapNextAdapter`` docstring) — so the
"queried first" evaluator can still score a clip in a single forward pass, same
one-pass shortcut as CoTracker/TAPIR (``supports_query_times = True``).

    python benchmark/tapnext/tapnext.py                       # all eval datasets -> W&B 'tapnext'
    python benchmark/tapnext/tapnext.py --no-wandb
    python benchmark/tapnext/tapnext.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Needs a **GPU unconditionally** — unlike the other baselines here, TAPNext
cannot even be constructed on CPU: upstream ``tapnext_torch.py`` hardcodes
``device='cuda'`` for its recurrent (LRU) blocks at ``__init__`` time. There is
no ``--cpu`` smoke path; ``build_adapter`` raises early with an explicit error
instead of the upstream ``RuntimeError: No CUDA GPUs are available``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__, "tapnet")       # add benchmark/methods/tapnet

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "tapnext" / "tapnext.yaml"


# --------------------------------------------------------------------------- #
# Model adapter: TAPNext -> the TWIST model forward contract
# --------------------------------------------------------------------------- #
class TapNextAdapter(torch.nn.Module):
    """Wrap TAPNext to the TWIST forward contract. TAPNext wants one frame at a
    time, ``(1,1,Hs,Ws,3)`` float in ``[-1,1]`` at its trained resolution
    (``interp_size``, 256x256 for BootsTAPNext — fixed by the positional
    embeddings baked into the checkpoint), plus ``query_points (1,N,3)=(t,y,x)``
    in that same resolution on the very first call.

    Per-point query times are handled *inside* TAPNext's own token embedding
    (``embed_queries`` in ``tapnext_torch.py``): a point whose query frame is
    still in the future is fed the "unknown" token (no prediction task yet);
    exactly on its query frame it gets the point-query token; after that it
    gets the "mask" token (continue predicting via the recurrent state). So
    passing the *full* query set (each at its true, possibly-later, query time)
    on the very first frame and then stepping through the rest of the clip
    tracks every point correctly in one pass — the per-frame-before-its-query
    output is never scored (the "queried first" protocol only scores frames
    after each point's query frame).

    Raw model output is ``tracks (1,1,N,2) = (y,x)`` in interp-resolution pixel
    space (confirmed against the working reference adapter in
    ``benchmark/methods/track_on/ensemble/tapnext/tapnext_predictor.py``, which
    flips the channel order before rescaling by width/height ratios
    respectively) — flipped to ``(x,y)`` and rescaled back to the input
    ``(H,W)`` here.

    ``point_mask`` is accepted and ignored: TAPNext tracks every query it is
    given, and the evaluator masks padded / non-evaluated points itself. Batch
    is looped one clip at a time (TAPNext's recurrent state bookkeeping is
    simplest kept per-clip, same pattern as :class:`CoTrackerAdapter`)."""

    VIS_LOGIT = 10.0
    supports_query_times = True

    def __init__(self, model: torch.nn.Module, interp_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.model = model
        self.interp_size = (int(interp_size[0]), int(interp_size[1]))

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
        """Track one clip. ``video (1,T,3,H,W)`` [0,255], ``queries (1,N,3)
        =(t,x,y)`` pixel coords -> ``(tracks (1,T,N,2) xy, vis (1,T,N) bool)``,
        both at the input ``(H,W)`` resolution."""
        _, T, _, H, W = video.shape
        Hs, Ws = self.interp_size

        vid = F.interpolate(video[0], size=(Hs, Ws), mode="bilinear")   # (T,3,Hs,Ws)
        vid = vid.unsqueeze(0).permute(0, 1, 3, 4, 2)                    # (1,T,Hs,Ws,3)
        vid = vid / 255.0 * 2.0 - 1.0

        q = queries.clone()
        q[..., 1] = q[..., 1] * (Ws / W)                # query x -> interp space
        q[..., 2] = q[..., 2] * (Hs / H)                # query y -> interp space
        q_tyx = torch.stack([q[..., 0], q[..., 2], q[..., 1]], dim=-1)   # (1,N,3) t,y,x

        tracks_yx, vis_raw, state = [], [], None
        for t in range(T):
            frame = vid[:, t:t + 1]
            if state is None:
                trk, _, vis_logit, state = self.model(video=frame, query_points=q_tyx)
            else:
                trk, _, vis_logit, state = self.model(video=frame, state=state)
            tracks_yx.append(trk)          # (1,1,N,2) yx, interp space
            vis_raw.append(vis_logit)      # (1,1,N,1)

        tracks = torch.cat(tracks_yx, dim=1).flip(-1)                # (1,T,N,2) yx -> xy
        vis = (torch.cat(vis_raw, dim=1).squeeze(-1) > 0)             # (1,T,N)

        tracks = tracks.clone()
        tracks[..., 0] *= W / Ws
        tracks[..., 1] *= H / Hs
        return tracks, vis


def build_adapter(cfg: Any, device: torch.device) -> TapNextAdapter:
    from tapnet.tapnext.tapnext_torch import TAPNext
    from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint
    from utilities.env import expand_path

    if device.type != "cuda":
        raise RuntimeError(
            "TAPNext hardcodes device='cuda' for its recurrent blocks at "
            "construction (upstream benchmark/methods/tapnet/tapnet/tapnext/"
            "tapnext_torch.py::TRecViTBlock) -- it cannot be built on CPU, not "
            "even for a smoke test. Run this on a GPU node (no --cpu path)."
        )

    tn = cfg.get("TAPNEXT", {})
    variant = str(tn.get("VARIANT", "bootstapnext")).lower()
    image_size = tuple(int(s) for s in tn.get("MODEL_IMAGE_SIZE", [256, 256]))
    ck_raw = tn.get("CHECKPOINT") or f"weights/tapnext/{variant}_ckpt.npz"
    url = tn.get("CHECKPOINT_URL") or (
        "https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz")

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.exists():
        if not url:
            raise FileNotFoundError(
                f"TAPNext checkpoint not found at {ckpt} and no TAPNEXT.CHECKPOINT_URL "
                "set. Download it first (see benchmark/tapnext/tapnext.yaml).")
        logger.info(f"checkpoint {ckpt} absent -> downloading {url}")
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(ckpt))

    logger.info(f"loading TAPNext ({variant}) from {ckpt}")
    model = TAPNext(image_size=image_size)          # builds its LRU blocks on 'cuda' internally
    model = restore_model_from_jax_checkpoint(model, str(ckpt))
    model.eval()

    adapter = TapNextAdapter(model, interp_size=image_size).to(device).eval()
    logger.info(f"TAPNext ({variant}) ready on {device} (image_size={image_size})")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark TAPNext (BootsTAPNext) on the TWIST eval datasets",
        checkpoint_key="TAPNEXT.CHECKPOINT",
        default_name="tapnext",
    ))
