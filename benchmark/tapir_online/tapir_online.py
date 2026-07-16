#!/usr/bin/env python
"""Benchmark **Online / causal TAPIR** (Online BootsTAPIR, PyTorch) on the TWIST
eval datasets, logged to W&B as ``tapir_online``.

This is the *causal* sibling of :mod:`benchmark.tapir.tapir` (which wraps the
**offline** whole-clip BootsTAPIR). Online TAPIR tracks strictly frame-by-frame:
it builds per-point *query features* on the query frame, then advances one frame
at a time carrying a **causal context** (the temporal-refinement recurrent
state), never looking at future frames — the DeepMind live-demo regime
(``tapnet/pytorch_live_demo.py``). Benchmarked here through the *same* evaluator
the TWIST model uses (:mod:`utilities.evaluation`) so the numbers are directly
comparable — and comparable to the offline ``bootstapir`` baseline already in
this folder, isolating what causality costs in accuracy.

The online API (all on :class:`tapnet.torch.tapir_model.TAPIR`):

    feature_grids  = model.get_feature_grids(frames_bthwc_m11, is_training=False)
    query_features = model.get_query_features(frames, is_training=False,
                                              query_points=q_tyx, feature_grids=...)
    causal_state   = model.construct_initial_causal_state(N, len(qf.resolutions)-1)
    # then per frame:
    traj = model.estimate_trajectories(hw, is_training=False, feature_grids=...,
                                       query_features=qf, query_points_in_video=None,
                                       causal_context=causal_state, get_causal_context=True)
    tracks = traj["tracks"][-1]          # (B,N,1,2) xy  (final-resolution head)
    causal_state = traj["causal_context"]

Because online TAPIR chains state strictly **forward from each point's query
frame**, it cannot seed arbitrary per-point query times in one shared pass, so
``supports_query_times = False`` and the "queried first" evaluator drives it with
**one init+rollout per distinct first-visible frame** (same contract as MFT /
LiteTracker's forward-only mode). See :class:`TapirOnlineAdapter`.

    python benchmark/tapir_online/tapir_online.py                     # all eval datasets -> W&B 'tapir_online'
    python benchmark/tapir_online/tapir_online.py --no-wandb
    python benchmark/tapir_online/tapir_online.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Heavy: needs a GPU + the **Online BootsTAPIR** checkpoint
(``causal_bootstapir_checkpoint.pt`` — the causal one, NOT the offline
``bootstapir_checkpoint_v2.pt``). See benchmark/tapir_online/tapir_online.yaml.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__, "tapnet")       # add benchmark/methods/tapnet

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "tapir_online" / "tapir_online.yaml"

# Online BootsTAPIR construction kwargs. Same head config as offline BootsTAPIR
# (ResNet18 + 4 extra convs) but with the *causal* temporal conv enabled — this
# is what the causal_bootstapir checkpoint was trained with.
ONLINE_BOOTSTAPIR_KWARGS = dict(
    pyramid_level=1, extra_convs=True, softmax_temperature=10.0,
    bilinear_interp_with_depthwise_conv=False, use_casual_conv=True,   # sic: upstream spelling
)


def _to_device(obj, device):
    """Move a nested list/tuple/dict of tensors (TAPIR's causal_context) to
    ``device`` without a dm-tree dependency."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj


class TapirOnlineAdapter(torch.nn.Module):
    """Wrap Online TAPIR to the TWIST forward contract.

    Online TAPIR consumes frames as ``(B,T,H,W,3)`` float in ``[-1,1]`` (we
    pre-normalise once with :func:`common.frames_to_bthwc_norm`, so the model's
    own ``preprocess_frames`` must NOT be re-applied) and query points as
    ``(t,y,x)``; it resizes nothing internally, so predictions come back at the
    input ``(H,W)`` and line up with the GT at the eval ``TARGET_SIZE`` directly.
    Output ``tracks`` are ``(B,N,T,2) = (x,y)`` at the final-resolution head.

    ``supports_query_times = False``: online TAPIR chains its causal state forward
    from the query frame, so a point is only tracked on frames *after* it. The
    evaluator therefore calls this adapter **once per distinct first-visible
    frame** ``f`` (every point in the call sharing ``f``); we build query features
    on frame ``f``, construct the initial causal state, then step ``f+1..T`` one
    frame at a time, scattering each frame's coords / visibility. (The inner loop
    over unique query frames is defensive — under "queried first" each call
    already carries a single ``f``.) Frames at/before ``f`` are filled with the
    query coordinate as a harmless placeholder — the evaluator only scores frames
    strictly after ``f``.

    Visibility follows the TAPIR demos:
    ``visible = (1-sig(occ))*(1-sig(expected_dist)) > 0.5``.
    ``point_mask`` is accepted and ignored; the evaluator masks padded /
    non-evaluated points itself."""

    VIS_LOGIT = 10.0
    supports_query_times = False

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = common.frames_to_bthwc_norm(frames)             # (B,T,H,W,3) [-1,1]
        B, T, H, W, _ = video.shape
        N = queries.shape[1]
        device = frames.device
        q = queries.float()
        q_t = q[:, :, 0].round().long()                         # (B,N) query frame per point
        q_xy = q[:, :, 1:3]                                     # (B,N,2) x,y on the query frame

        coords = torch.zeros((B, T, N, 2), dtype=torch.float32, device=device)
        vis = torch.zeros((B, T, N), dtype=torch.bool, device=device)

        for b in range(B):
            for f in torch.unique(q_t[b]).tolist():
                f = int(f)
                sel = (q_t[b] == f).nonzero(as_tuple=True)[0]   # points queried at frame f
                # (t=0 within the single-frame init clip, then (y,x))
                q_tyx = torch.stack(
                    [torch.zeros(len(sel), device=device),
                     q_xy[b, sel, 1], q_xy[b, sel, 0]], dim=-1).unsqueeze(0)   # (1,n,3)
                # Template frame: identity — coords are the query coords, visible.
                coords[b, : f + 1, sel] = q_xy[b, sel]
                vis[b, f, sel] = True
                self._track_group(video[b, f], video[b], f, T, q_tyx, coords, vis, b, sel)

        vis_logits = common.vis_bool_to_logits(vis, coords, self.VIS_LOGIT)
        return {"coords": coords, "vis_logits": vis_logits}

    @torch.no_grad()
    def _track_group(self, init_frame, video_b, f, T, q_tyx, coords, vis, b, sel):
        """Init query features on frame ``f`` and roll causally to the clip end,
        scattering ``coords``/``vis`` for the points in ``sel``.

        ``init_frame`` (H,W,3) [-1,1], ``video_b`` (T,H,W,3) [-1,1], ``q_tyx``
        (1,n,3)=(0,y,x)."""
        m = self.model
        init = init_frame.unsqueeze(0).unsqueeze(0)             # (1,1,H,W,3)
        fg = m.get_feature_grids(init, is_training=False)
        qf = m.get_query_features(init, is_training=False,
                                  query_points=q_tyx, feature_grids=fg)
        cc = m.construct_initial_causal_state(len(sel), len(qf.resolutions) - 1)
        cc = _to_device(cc, init_frame.device)

        for tt in range(f + 1, T):
            frame = video_b[tt].unsqueeze(0).unsqueeze(0)       # (1,1,H,W,3)
            fg_t = m.get_feature_grids(frame, is_training=False)
            traj = m.estimate_trajectories(
                frame.shape[-3:-1], is_training=False, feature_grids=fg_t,
                query_features=qf, query_points_in_video=None,
                causal_context=cc, get_causal_context=True,
            )
            cc = traj["causal_context"]
            tracks = traj["tracks"][-1]                         # (1,n,1,2) xy
            occ = traj["occlusion"][-1]                         # (1,n,1)
            expd = traj["expected_dist"][-1]                    # (1,n,1)
            visible = ((1 - torch.sigmoid(occ)) * (1 - torch.sigmoid(expd)) > 0.5)
            coords[b, tt, sel] = tracks[0, :, 0]
            vis[b, tt, sel] = visible[0, :, 0]


def build_adapter(cfg: Any, device: torch.device) -> TapirOnlineAdapter:
    from tapnet.torch import tapir_model
    from utilities.env import expand_path

    if device.type != "cuda":
        logger.warning("Online TAPIR on CPU is extremely slow (frame-by-frame); smoke only.")

    tp = cfg.get("TAPIR_ONLINE", {})
    variant = str(tp.get("VARIANT", "causal_bootstapir")).lower()
    kw = tp.get("MODEL_KWARGS")
    kwargs = (kw.toDict() if hasattr(kw, "toDict") else dict(kw)) if kw else dict(ONLINE_BOOTSTAPIR_KWARGS)
    ck_raw = tp.get("CHECKPOINT") or f"weights/tapir/{variant}.pt"
    url = tp.get("CHECKPOINT_URL")

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if ckpt.exists():
        logger.info(f"loading Online TAPIR ({variant}) from {ckpt}")
        state_dict = torch.load(str(ckpt), map_location="cpu")
    elif url:
        logger.info(f"checkpoint {ckpt} absent -> downloading {url}")
        state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        try:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, ckpt)
            logger.info(f"cached checkpoint -> {ckpt}")
        except OSError as e:
            logger.warning(f"could not cache checkpoint ({e})")
    else:
        # TODO(weights): download the Online BootsTAPIR (causal) checkpoint first.
        raise FileNotFoundError(
            f"Online TAPIR checkpoint not found at {ckpt} and no TAPIR_ONLINE.CHECKPOINT_URL "
            "set. Download causal_bootstapir_checkpoint.pt first "
            "(see benchmark/tapir_online/tapir_online.yaml and SETUP_TRACKERS.md).")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = tapir_model.TAPIR(**kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        logger.info("load_state_dict: exact match (0 missing / 0 unexpected)")
    adapter = TapirOnlineAdapter(model).to(device).eval()
    logger.info(f"Online TAPIR ({variant}) ready on {device} (use_casual_conv=True)")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark Online / causal TAPIR on the TWIST eval datasets",
        checkpoint_key="TAPIR_ONLINE.CHECKPOINT",
        default_name="tapir_online",
    ))
