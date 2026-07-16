#!/usr/bin/env python
"""Benchmark TAPIR / BootsTAPIR (PyTorch) on the TWIST eval datasets, logged to
W&B as ``tapir`` (or ``bootstapir``).

Runs DeepMind's TAPIR through the *same* evaluator the TWIST model uses
(:mod:`utilities.evaluation`) so the numbers are directly comparable. Shared
plumbing lives in :mod:`benchmark.common`; this file only builds the adapter.

This wraps the **offline** TAPIR model (one forward over the whole clip). The
``MODEL_KWARGS`` in the config select the variant — the defaults match
**BootsTAPIR** (the strong baseline, PyTorch checkpoint available).

    python benchmark/tapir/tapir.py                       # all eval datasets -> W&B 'bootstapir'
    python benchmark/tapir/tapir.py --no-wandb
    python benchmark/tapir/tapir.py --datasets TAPVID_DAVIS --max-clips 5   # smoke

Heavy: needs a GPU + the TAPIR checkpoint (see benchmark/tapir/tapir.yaml).
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
REPO_ROOT = common.setup_method_paths(__file__, "tapnet")       # add benchmark/methods/tapnet

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "tapir" / "tapir.yaml"

# BootsTAPIR offline construction kwargs (tapnet colab demos).
BOOTSTAPIR_KWARGS = dict(
    pyramid_level=1, extra_convs=True, softmax_temperature=10.0,
    bilinear_interp_with_depthwise_conv=False,
)


class TapirAdapter(torch.nn.Module):
    """Wrap TAPIR to the TWIST forward contract. TAPIR wants ``video (B,T,H,W,3)``
    in ``[-1,1]`` and ``queries (B,N,3)=(t,y,x)``; it resizes internally and
    returns ``tracks (B,N,T,2)`` xy at the input resolution + occlusion /
    expected_dist logits, so frames fed at the eval ``TARGET_SIZE`` line up with
    the GT directly."""

    # Offline TAPIR accepts per-point query frames in one forward (same as LocoTrack),
    # so the "queried first" evaluator scores a clip in one pass.
    supports_query_times = True

    def __init__(
        self,
        model: torch.nn.Module,
        query_chunk_size: int = 32,
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
        """Run TAPIR on one clip, tiling temporally when ``T`` is large."""
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
        if video.is_cuda:
            torch.cuda.empty_cache()
        return common.tapir_outputs_to_twist(
            out["tracks"], out["occlusion"], out.get("expected_dist"))


def build_adapter(cfg: Any, device: torch.device) -> TapirAdapter:
    from tapnet.torch import tapir_model
    from utilities.env import expand_path

    tp = cfg.get("TAPIR", {})
    variant = str(tp.get("VARIANT", "bootstapir")).lower()
    kw = tp.get("MODEL_KWARGS")
    kwargs = (kw.toDict() if hasattr(kw, "toDict") else dict(kw)) if kw else dict(BOOTSTAPIR_KWARGS)
    chunk = int(tp.get("QUERY_CHUNK_SIZE", 32))
    image_size = int(cfg.get("IMAGE_SIZE", 512))
    default_feat_chunk = max(2, int(10 * 256 * 256 / (image_size * image_size)))
    feat_chunk = int(tp.get("FEATURE_EXTRACTOR_CHUNK_SIZE", default_feat_chunk))
    default_max_t = max(96, 384 * 256 // image_size)
    max_t = int(tp.get("MAX_TEMPORAL_FRAMES", default_max_t))
    refin = tp.get("REFINEMENT_RESOLUTIONS")
    if refin is not None:
        refin = [tuple(r) for r in refin]
    elif image_size > 256:
        refin = [(256, 256)]
    kwargs.setdefault("feature_extractor_chunk_size", feat_chunk)
    ck_raw = tp.get("CHECKPOINT") or f"weights/tapir/{variant}.pt"
    url = tp.get("CHECKPOINT_URL")

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if ckpt.exists():
        logger.info(f"loading TAPIR ({variant}) from {ckpt}")
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
        raise FileNotFoundError(
            f"TAPIR checkpoint not found at {ckpt} and no TAPIR.CHECKPOINT_URL set. "
            "Download it first (see benchmark/tapir/tapir.yaml).")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model = tapir_model.TAPIR(**kwargs)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    adapter = TapirAdapter(
        model,
        query_chunk_size=chunk,
        max_temporal_frames=max_t,
        refinement_resolutions=refin,
    ).to(device).eval()
    logger.info(
        f"TAPIR ({variant}) ready on {device} "
        f"(max_t={max_t}, feature_extractor_chunk={feat_chunk}, "
        f"query_chunk={chunk}, refinement={refin or 'auto'})")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark TAPIR / BootsTAPIR on the TWIST eval datasets",
        checkpoint_key="TAPIR.CHECKPOINT",
        default_name="bootstapir",
    ))
