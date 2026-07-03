#!/usr/bin/env python
"""Benchmark LocoTrack on the TWIST eval datasets, logged to W&B as ``locotrack``.

Runs LocoTrack (PyTorch port) through the *same* evaluator the TWIST model uses
(:mod:`utilities.evaluation`) so the numbers are directly comparable. Shared
plumbing (CLI, W&B, evaluate-and-report) lives in :mod:`benchmark.common`; this
file only builds the model adapter.

    python benchmark/locotrack/locotrack.py                       # all eval datasets -> W&B 'locotrack'
    python benchmark/locotrack/locotrack.py --no-wandb
    python benchmark/locotrack/locotrack.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Heavy: needs a GPU + the LocoTrack checkpoint (see benchmark/locotrack/locotrack.yaml).

Namespace note: LocoTrack's PyTorch port ships a **top-level ``models`` package**
(the same name as TWIST's ``models/`` that the evaluator imports).
:func:`common.import_isolated` imports LocoTrack with its source dir first on
``sys.path`` and TWIST's ``models`` temporarily hidden, then restores TWIST's
afterwards. See benchmark/common.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))                       # <repo> for utilities.*
sys.path.insert(0, str(_HERE.parents[1]))                       # <repo>/benchmark for `import common`
import common                                                   # noqa: E402
REPO_ROOT = common.setup_method_paths(__file__)                 # locotrack_pytorch via import_isolated

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "locotrack" / "locotrack.yaml"


class LocoTrackAdapter(torch.nn.Module):
    """Wrap a LocoTrack model to the TWIST forward contract.

    LocoTrack wants ``video (B,T,H,W,3)`` in ``[-1,1]`` and ``queries (B,N,3) =
    (t,y,x)`` in pixels, and returns ``tracks (B,N,T,2)`` xy at the *input*
    resolution plus ``occlusion`` / ``expected_dist`` logits. It resizes
    internally and reports tracks in the fed pixel space, so we pass frames at
    the eval ``TARGET_SIZE`` and the coords already line up with the GT.

    Long clips (RoboTAP ~1.2k frames) cannot fit the full-sequence feature grids
    and cost volumes in GPU memory even with a small ResNet chunk size. Clips
    longer than ``max_temporal_frames`` are processed in contiguous temporal
    windows; continuing points are re-queried at each window boundary with the
    previous window's predicted position."""

    # LocoTrack accepts per-point query frames in one forward (offline TAPIR-style),
    # so the "queried first" evaluator can score a clip in one pass instead of one
    # forward per distinct first-visible frame — critical for long clips (RoboTAP).
    supports_query_times = True

    def __init__(
        self,
        model: torch.nn.Module,
        query_chunk_size: int = 64,
        max_temporal_frames: int = 128,
        refinement_resolutions: Optional[Sequence[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.model = model
        self.query_chunk_size = int(query_chunk_size)
        self.max_temporal_frames = int(max_temporal_frames)
        self.refinement_resolutions = (
            list(refinement_resolutions) if refinement_resolutions is not None else None)

    def _run_model(
        self,
        video: torch.Tensor,                                    # (1,T,H,W,3)
        qp_tyx: torch.Tensor,                                     # (1,n,3)
    ) -> dict:
        kw: dict = dict(query_chunk_size=self.query_chunk_size)
        if self.refinement_resolutions is not None:
            kw["refinement_resolutions"] = self.refinement_resolutions
        return self.model(video, qp_tyx, **kw)

    def _forward_clip(
        self,
        video: torch.Tensor,                                    # (1,T,H,W,3)
        qp_tyx: torch.Tensor,                                     # (1,N,3) t,y,x
        point_mask: Optional[torch.Tensor],                       # (1,N) or None
    ) -> dict:
        """Run LocoTrack on one clip, tiling temporally when ``T`` is large."""
        T = int(video.shape[1])
        N = int(qp_tyx.shape[1])
        if T <= self.max_temporal_frames:
            return self._run_model(video, qp_tyx)

        device = video.device
        tracks = video.new_zeros(1, N, T, 2)
        occ = video.new_zeros(1, N, T)
        expd = video.new_zeros(1, N, T)
        t_query = qp_tyx[0, :, 0]
        started = torch.zeros(N, dtype=torch.bool, device=device)
        usable = torch.ones(N, dtype=torch.bool, device=device)
        if point_mask is not None:
            usable &= point_mask[0].bool()

        seg_start = 0
        while seg_start < T:
            seg_end = min(seg_start + self.max_temporal_frames, T)
            seg_video = video[:, seg_start:seg_end]             # (1,t_seg,H,W,3)

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
                tracks[0, active, seg_start:seg_end] = out["tracks"][0, :n_act, :seg_len]
                occ[0, active, seg_start:seg_end] = out["occlusion"][0, :n_act, :seg_len]
                if "expected_dist" in out:
                    expd[0, active, seg_start:seg_end] = out["expected_dist"][0, :n_act, :seg_len]

            seg_start = seg_end
            if device.type == "cuda":
                torch.cuda.empty_cache()

        return dict(tracks=tracks, occlusion=occ, expected_dist=expd)

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = common.frames_to_bthwc_norm(frames)             # (B,T,H,W,3)
        qp = common.queries_txy_to_tyx(queries.float())
        outs = []
        for b in range(video.shape[0]):
            pm = point_mask[b:b + 1] if point_mask is not None else None
            outs.append(self._forward_clip(video[b:b + 1], qp[b:b + 1], pm))
        out = {
            "tracks": torch.cat([o["tracks"] for o in outs], dim=0),
            "occlusion": torch.cat([o["occlusion"] for o in outs], dim=0),
        }
        if all("expected_dist" in o for o in outs):
            out["expected_dist"] = torch.cat([o["expected_dist"] for o in outs], dim=0)
        result = common.tapir_outputs_to_twist(
            out["tracks"], out["occlusion"], out.get("expected_dist"))
        if video.is_cuda:
            torch.cuda.empty_cache()
        return result


def build_adapter(cfg: Any, device: torch.device) -> LocoTrackAdapter:
    from utilities.env import expand_path

    lt = cfg.get("LOCOTRACK", {})
    model_size = str(lt.get("MODEL_SIZE", "base")).lower()
    ck_raw = lt.get("CHECKPOINT") or f"weights/locotrack/locotrack_{model_size}.ckpt"
    url = lt.get("CHECKPOINT_URL") or (
        "https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/"
        f"resolve/main/locotrack_{model_size}.ckpt")
    chunk = int(lt.get("QUERY_CHUNK_SIZE", 64))
    image_size = int(cfg.get("IMAGE_SIZE", 512))
    default_feat_chunk = max(2, int(10 * 256 * 256 / (image_size * image_size)))
    feat_chunk = int(lt.get("FEATURE_EXTRACTOR_CHUNK_SIZE", default_feat_chunk))
    # LocoTrack demo caps at 300 frames @ 256²; scale down for 512² eval.
    default_max_t = max(96, 384 * 256 // image_size)
    max_t = int(lt.get("MAX_TEMPORAL_FRAMES", default_max_t))
    # Skip the 512² refinement pyramid when eval feeds 512 frames — halves feature
    # memory; coords are still reported in the input pixel space via train2orig.
    refin = lt.get("REFINEMENT_RESOLUTIONS")
    if refin is not None:
        refin = [tuple(r) for r in refin]
    elif image_size > 256:
        refin = [(256, 256)]

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if ckpt.exists():
        logger.info(f"loading LocoTrack-{model_size} from {ckpt}")
        raw = torch.load(str(ckpt), map_location="cpu")
    else:
        logger.info(f"checkpoint {ckpt} absent -> downloading {url}")
        raw = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        try:
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(raw, ckpt)
            logger.info(f"cached checkpoint -> {ckpt}")
        except OSError as e:
            logger.warning(f"could not cache checkpoint ({e})")

    state_dict = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    with common.import_isolated("locotrack/locotrack_pytorch", "models", "locotrack_pytorch"):
        from models.locotrack_model import LocoTrack
        model = LocoTrack(
            model_size=model_size,
            feature_extractor_chunk_size=feat_chunk,
        )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    adapter = LocoTrackAdapter(
        model,
        query_chunk_size=chunk,
        max_temporal_frames=max_t,
        refinement_resolutions=refin,
    ).to(device).eval()
    logger.info(
        f"LocoTrack-{model_size} ready on {device} "
        f"(max_t={max_t}, feature_extractor_chunk={feat_chunk}, "
        f"query_chunk={chunk}, refinement={refin or 'auto'})")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark LocoTrack on the TWIST eval datasets",
        checkpoint_key="LOCOTRACK.CHECKPOINT",
        default_name="locotrack",
    ))
