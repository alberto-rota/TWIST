#!/usr/bin/env python
"""Benchmark Chrono on the TWIST eval datasets, logged to W&B as ``chrono``.

Runs Chrono (DINOv2 + long-range temporal adapter point tracker) through the
*same* evaluator the TWIST model uses (:mod:`utilities.evaluation`) so the
numbers are directly comparable. Shared plumbing lives in :mod:`benchmark.common`.

Namespace note: Chrono ships a **top-level ``models`` package** — the same name
as TWIST's ``models/`` that the evaluator imports. :func:`common.import_isolated`
imports Chrono's model with its source dir first on ``sys.path`` and TWIST's
``models`` temporarily hidden, then restores TWIST's afterwards (Chrono's already
built object keeps its own bound references). See benchmark/common.py.

    python benchmark/chrono/chrono.py                       # all eval datasets -> W&B 'chrono'
    python benchmark/chrono/chrono.py --no-wandb
    python benchmark/chrono/chrono.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Heavy: needs a GPU, xformers, the DINOv2 backbone (auto-downloaded via torch.hub),
and the Chrono checkpoint (see benchmark/chrono/chrono.yaml).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__)                 # Chrono is added via import_isolated

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "chrono" / "chrono.yaml"


class ChronoAdapter(torch.nn.Module):
    """Wrap Chrono to the TWIST forward contract.

    Chrono wants ``video (B,T,H,W,3)`` in ``[-1,1]`` and ``queries (B,N,3) =
    (t,y,x)`` in pixels at the *working* resolution. It returns ``tracks
    (B,N,T,2)`` xy plus occlusion logits. Long clips are tiled temporally —
    DINO runs on all ``T`` frames at once and memory / cost scale linearly
    with ``T``."""

    # Chrono accepts per-point query frames in one forward (offline TAPIR-style),
    # so the "queried first" evaluator scores a clip in one pass instead of one
    # DINO forward per distinct first-visible frame — critical for Kinetics / RoboTAP.
    supports_query_times = True

    def __init__(
        self,
        model: torch.nn.Module,
        resolution=(256, 256),
        query_chunk_size: int = 64,
        max_temporal_frames: int = 128,
    ):
        super().__init__()
        self.model = model
        self.resolution = tuple(resolution)
        self.query_chunk_size = int(query_chunk_size)
        self.max_temporal_frames = int(max_temporal_frames)

    def _to_work_resolution(
        self,
        video: torch.Tensor,                                    # (1,T,H,W,3)
        qp_tyx: torch.Tensor,                                     # (1,N,3)
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        """Resize once per clip; scale queries to working resolution."""
        H, W = int(video.shape[2]), int(video.shape[3])
        rh, rw = self.resolution
        if (H, W) == (rh, rw):
            return video, qp_tyx, (H, W)
        B, T = video.shape[:2]
        v = rearrange(video, 'b t h w c -> (b t) c h w')
        v = F.interpolate(v, self.resolution, mode='bilinear', align_corners=False)
        v = rearrange(v, '(b t) c h w -> b t h w c', b=B, t=T)
        qp = qp_tyx.clone()
        qp[..., 1] = qp[..., 1] / H * rh                        # y
        qp[..., 2] = qp[..., 2] / W * rw                        # x
        return v, qp, (H, W)

    def _tracks_to_orig(self, tracks: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
        """``tracks (1,N,T,2)`` at working res -> input pixel space."""
        H, W = orig_hw
        rh, rw = self.resolution
        if (H, W) == (rh, rw):
            return tracks
        scale = tracks.new_tensor([W / rw, H / rh])             # (2,) xy
        return tracks * scale

    def _run_model(
        self,
        video: torch.Tensor,                                    # (1,T,rh,rw,3)
        qp_tyx: torch.Tensor,                                     # (1,n,3) t,y,x
    ) -> dict:
        out = self.model.forward(video, qp_tyx, query_chunk_size=self.query_chunk_size)
        pred_occ = torch.sigmoid(out['occlusion'])
        pred_ed = torch.sigmoid(out['expected_dist'])
        occ = (1 - (1 - pred_occ) * (1 - pred_ed)) > 0.5
        return dict(tracks=out['tracks'], occlusion=occ)

    def _forward_clip(
        self,
        video: torch.Tensor,                                    # (1,T,H,W,3)
        qp_tyx: torch.Tensor,                                     # (1,N,3) t,y,x
        point_mask: Optional[torch.Tensor],                       # (1,N) or None
    ) -> dict:
        """Run Chrono on one clip, tiling temporally when ``T`` is large."""
        video, qp_tyx, orig_hw = self._to_work_resolution(video, qp_tyx)
        T = int(video.shape[1])
        N = int(qp_tyx.shape[1])
        if T <= self.max_temporal_frames:
            out = self._run_model(video, qp_tyx)
            out['tracks'] = self._tracks_to_orig(out['tracks'], orig_hw)
            return out

        device = video.device
        tracks = video.new_zeros(1, N, T, 2)
        occ = torch.ones(1, N, T, dtype=torch.bool, device=device)  # True = occluded
        t_query = qp_tyx[0, :, 0]                               # (N,)
        started = torch.zeros(N, dtype=torch.bool, device=device)
        usable = torch.ones(N, dtype=torch.bool, device=device)
        if point_mask is not None:
            usable &= point_mask[0].bool()

        seg_start = 0
        while seg_start < T:
            seg_end = min(seg_start + self.max_temporal_frames, T)
            seg_video = video[:, seg_start:seg_end]             # (1,t_seg,rh,rw,3)

            active: List[int] = []
            seg_q: List[List[float]] = []
            for n in range(N):
                if not usable[n]:
                    continue
                tq = int(t_query[n].item())
                if tq >= seg_end:
                    continue
                if tq < seg_start:
                    if not started[n]:
                        continue
                    prev_f = max(seg_start - 1, 0)
                    x, y = tracks[0, n, prev_f, 0].item(), tracks[0, n, prev_f, 1].item()
                    seg_q.append([0.0, y, x])                   # (t,y,x) re-query at window start
                else:
                    seg_q.append([
                        float(tq - seg_start),
                        float(qp_tyx[0, n, 1].item()),
                        float(qp_tyx[0, n, 2].item()),
                    ])
                    started[n] = True
                active.append(n)

            if seg_q:
                sq = torch.tensor(seg_q, device=device, dtype=qp_tyx.dtype).unsqueeze(0)
                out = self._run_model(seg_video, sq)
                n_act = len(active)
                seg_len = seg_end - seg_start
                tracks[0, active, seg_start:seg_end] = out['tracks'][0, :n_act, :seg_len]
                occ[0, active, seg_start:seg_end] = out['occlusion'][0, :n_act, :seg_len]

            seg_start = seg_end

        tracks = self._tracks_to_orig(tracks, orig_hw)
        return dict(tracks=tracks, occlusion=occ)

    @staticmethod
    def _to_twist(out: dict) -> dict:
        coords = out['tracks'].permute(0, 2, 1, 3).contiguous()  # (B,N,T,2)->(B,T,N,2)
        visible = (~out['occlusion'].bool()).permute(0, 2, 1).contiguous()  # ->(B,T,N)
        return {"coords": coords.float(), "vis_logits": common.vis_bool_to_logits(visible, coords)}

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = common.frames_to_bthwc_norm(frames)             # (B,T,H,W,3) in [-1,1]
        qp = common.queries_txy_to_tyx(queries.float())         # (B,N,3) t,y,x
        outs = []
        for b in range(video.shape[0]):
            pm = point_mask[b:b + 1] if point_mask is not None else None
            outs.append(self._forward_clip(video[b:b + 1], qp[b:b + 1], pm))
        out = {
            "tracks": torch.cat([o["tracks"] for o in outs], dim=0),
            "occlusion": torch.cat([o["occlusion"] for o in outs], dim=0),
        }
        return self._to_twist(out)


def build_adapter(cfg: Any, device: torch.device) -> ChronoAdapter:
    from utilities.env import expand_path

    ch = cfg.get("CHRONO", {})
    dino_size = str(ch.get("DINO_SIZE", "base")).lower()
    dino_reg = bool(ch.get("DINO_REG", False))
    adapter_ch = int(ch.get("ADAPTER_INTERMED_CHANNELS", 128))
    resolution = tuple(ch.get("RESOLUTION", [256, 256]))
    chunk = int(ch.get("QUERY_CHUNK_SIZE", 64))
    image_size = int(cfg.get("IMAGE_SIZE", 256))
    # DINO is O(T); at 256² an A40 can hold more frames per tile than at 512².
    default_max_t = max(64, 128 * 256 // image_size)
    max_t = int(ch.get("MAX_TEMPORAL_FRAMES", default_max_t))
    ck_raw = ch.get("CHECKPOINT") or f"weights/chrono/chrono_{dino_size}.ckpt"

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Chrono checkpoint not found at {ckpt}. Download it first "
            "(see benchmark/chrono/chrono.yaml).")

    # Import + construct inside the isolation window so Chrono's top-level `models`
    # resolves to Chrono's (not TWIST's); the built object then works after exit.
    with common.import_isolated("Chrono", "models", "model_utils", "data"):
        from models.locotrack_model import LocoTrack
        model = LocoTrack(dino_size=dino_size, dino_reg=dino_reg,
                          adapter_intermed_channels=adapter_ch)
        raw = torch.load(str(ckpt), map_location="cpu")
        state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")

    adapter = ChronoAdapter(
        model, resolution=resolution, query_chunk_size=chunk, max_temporal_frames=max_t,
    ).to(device).eval()
    logger.info(
        f"Chrono (dino_size={dino_size}) ready on {device} "
        f"(resolution={resolution}, max_t={max_t}, query_chunk={chunk})")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark Chrono on the TWIST eval datasets",
        checkpoint_key="CHRONO.CHECKPOINT",
        default_name="chrono",
    ))
