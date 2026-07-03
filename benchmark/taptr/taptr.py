#!/usr/bin/env python
"""Benchmark TAPTRv3 on the TWIST eval datasets, logged to W&B as ``taptr``.

Runs TAPTRv3 (DETR-style Track-Any-Point transformer) through the *same*
evaluator the TWIST model uses (:mod:`utilities.evaluation`) so the numbers are
directly comparable. Shared plumbing lives in :mod:`benchmark.common`.

This is the most involved adapter: TAPTR is a detection-transformer that seeds
its point queries from a ``targets`` dict (normalized ``cx,cy,w,h`` point-boxes +
per-frame tracking masks; the first ``True`` frame of each point's mask is its
query frame), takes the video as a ``NestedTensor`` of ImageNet-normalized
frames, and returns normalized ``pred_boxes`` + occlusion logits. We rebuild the
model exactly as TAPTR's ``main.py`` does (its ``get_args_parser`` + config +
``build_model_main``), then drive it like its ``evaluate.py`` eval loop.

Namespace note: TAPTR ships top-level ``models`` / ``datasets`` / ``util`` /
``main`` packages — ``models`` and ``main`` collide with TWIST's. They are
imported inside :func:`common.import_isolated` so TAPTR's versions resolve at
build time and TWIST's are restored afterwards.

Prereqs (see benchmark/taptr/taptr.yaml + the sbatch note): the v3 branch checked
out at benchmark/methods/TAPTR, the custom CUDA ops compiled
(``cd benchmark/methods/TAPTR/models/dino/ops && python setup.py install``), and
the TAPTRv3 checkpoint downloaded. **Highest-risk adapter — verify on first run.**

    python benchmark/taptr/taptr.py                       # all eval datasets -> W&B 'taptr'
    python benchmark/taptr/taptr.py --datasets TAPVID_DAVIS --max-clips 5   # smoke
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
REPO_ROOT = common.setup_method_paths(__file__)                 # TAPTR added via import_isolated

logger = common.logger
DEFAULT_CONFIG = REPO_ROOT / "benchmark" / "taptr" / "taptr.yaml"

# ImageNet normalization (TAPTR datasets/kubric.py).
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class TaptrAdapter(torch.nn.Module):
    """Wrap a TAPTRv3 model to the TWIST forward contract.

    Per clip (TAPTR is batch-1): ImageNet-normalize the frames and wrap them in a
    ``NestedTensor``; build the ``targets`` dict that seeds one query per TWIST
    query point (location at its query frame, tracking mask True from the query
    frame onward); run ``streaming_forward`` (causal, the eval default) or the
    offline ``forward``; read ``full_seq_output`` -> de-normalize ``pred_boxes``
    to the input pixel space and turn the occlusion logit into a visibility logit.
    """

    def __init__(self, model, nested_fn, *, streaming: bool = True,
                 mini_box_size: float = 5.0, occ_threshold: float = 0.45):
        super().__init__()
        self.model = model
        self._nested_fn = nested_fn
        self.streaming = bool(streaming)
        self.mini_box_size = float(mini_box_size)
        self.occ_threshold = float(occ_threshold)
        self.register_buffer("_mean", torch.tensor(_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(_STD).view(1, 3, 1, 1), persistent=False)

    def _build_targets(self, queries, T, H, W, device):
        """queries (N,3)=(t,x,y) px -> TAPTR targets dict (normalized cx,cy,w,h)."""
        N = queries.shape[0]
        t = queries[:, 0].long().clamp_(0, T - 1)
        cx = (queries[:, 1] / (W - 1)).clamp_(0, 1)
        cy = (queries[:, 2] / (H - 1)).clamp_(0, 1)
        w = torch.full((N,), self.mini_box_size / (W - 1), device=device)
        h = torch.full((N,), self.mini_box_size / (H - 1), device=device)
        box = torch.stack([cx, cy, w, h], dim=-1)               # (N,4) at the query frame
        pt_boxes = box[:, None, :].expand(N, T, 4).contiguous()  # repeat across time (init style)
        frame_idx = torch.arange(T, device=device)[None, :].expand(N, T)
        track_mask = frame_idx >= t[:, None]                    # True from the query frame onward
        pt_labels = torch.ones(N, T, device=device)             # seed all visible; model overrides
        return {
            "pt_boxes": pt_boxes.float(),
            "pt_labels": pt_labels.float(),
            "pt_tracking_mask": track_mask.bool(),
            "query_frames": t.int(),
            "num_real_pt": torch.tensor(N, device=device),
        }

    @torch.no_grad()
    def _forward_one(self, frames_t3hw, queries_n3):
        """Track one clip. frames (T,3,H,W) float [0,255] -> coords (T,N,2) px,
        visible (T,N) bool."""
        device = frames_t3hw.device
        T, _, H, W = frames_t3hw.shape
        video = (frames_t3hw / 255.0 - self._mean) / self._std  # ImageNet norm
        samples = self._nested_fn(video[None, ...]).to(device)  # (1,T,3,H,W) NestedTensor
        targets = self._build_targets(queries_n3, T, H, W, device)

        if self.streaming:
            out, _ = self.model.streaming_forward(samples, [targets])
            full = out["full_seq_output"]
            pred_boxes = full["pred_boxes"].permute(1, 0, 2)            # (N,T,4)
            occ = full["pred_logits"].permute(1, 0, 2)[..., 1]
        else:
            out, _ = self.model(samples, [targets])
            full = out["full_seq_output"]
            pred_boxes = full["pred_boxes"][0].permute(1, 0, 2)         # (N,T,4)
            occ = full["pred_logits"][0].permute(1, 0, 2)[..., 1]
        occluded = occ.sigmoid() > self.occ_threshold                  # (N,T)

        scale = pred_boxes.new_tensor([W - 1, H - 1])
        tracks = pred_boxes[..., :2] * scale                           # (N,T,2) px
        coords = tracks.permute(1, 0, 2).contiguous()                  # (T,N,2)
        visible = (~occluded).permute(1, 0).contiguous()               # (T,N)
        return coords, visible

    @torch.no_grad()
    def forward(self, frames, queries, point_mask=None, **_):
        video = common.frames_to_255_float(frames)              # (B,T,3,H,W)
        queries = queries.float()
        B = video.shape[0]
        cs, vs = [], []
        for b in range(B):
            c, v = self._forward_one(video[b], queries[b])
            cs.append(c)
            vs.append(v)
        coords = torch.stack(cs, dim=0)                         # (B,T,N,2)
        visible = torch.stack(vs, dim=0)                        # (B,T,N)
        return {"coords": coords.float(),
                "vis_logits": common.vis_bool_to_logits(visible, coords)}


def build_adapter(cfg: Any, device: torch.device) -> TaptrAdapter:
    from utilities.env import expand_path

    tp = cfg.get("TAPTR", {})
    cfg_file = str(tp.get("CONFIG", "benchmark/methods/TAPTR/config/TAPTRv3_resnet50_512x512.py"))
    cfg_path = Path(expand_path(cfg_file))
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    ck_raw = tp.get("CHECKPOINT") or "weights/taptr/TAPTRv3_resnet50_512x512.pth"
    streaming = bool(tp.get("STREAMING", True))
    mini_box = float(tp.get("MINI_BOX_SIZE", 5.0))
    occ_thr = float(tp.get("OCC_THRESHOLD", 0.45))

    ckpt = Path(expand_path(str(ck_raw)))
    if not ckpt.is_absolute():
        ckpt = REPO_ROOT / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(
            f"TAPTRv3 checkpoint not found at {ckpt}. Download it first "
            "(see benchmark/taptr/taptr.yaml).")
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"TAPTR config {cfg_path} not found — is the v3 branch checked out? "
            "(cd benchmark/methods/TAPTR && git checkout v3)")

    # Build the model exactly as TAPTR's main.py does, with its top-level packages
    # isolated from TWIST's same-named ones.
    with common.import_isolated("TAPTR", "models", "datasets", "util", "main"):
        from main import build_model_main, get_args_parser
        from util.misc import nested_temporal_tensor_from_tensor_list
        from util.slconfig import SLConfig

        args = get_args_parser().parse_args(["-c", str(cfg_path)])
        slcfg = SLConfig.fromfile(str(cfg_path))
        for k, v in slcfg._cfg_dict.to_dict().items():
            if not hasattr(args, k):
                setattr(args, k, v)
        args.device = str(device)
        args.eval = True
        if not hasattr(args, "masks"):
            args.masks = False

        model, _, _ = build_model_main(args)
        raw = torch.load(str(ckpt), map_location="cpu")
        if isinstance(raw, dict) and "model" in raw:
            state_dict = raw["model"]
        elif isinstance(raw, dict) and "ema_model" in raw:
            state_dict = {k.replace("module.", "", 1): v for k, v in raw["ema_model"].items()}
        else:
            state_dict = raw
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        logger.warning(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
    adapter = TaptrAdapter(model, nested_temporal_tensor_from_tensor_list,
                           streaming=streaming, mini_box_size=mini_box,
                           occ_threshold=occ_thr).to(device).eval()
    logger.info(f"TAPTRv3 ready on {device} (config={cfg_path.name}, "
                f"streaming={streaming}, occ_thr={occ_thr})")
    return adapter


if __name__ == "__main__":
    sys.exit(common.run(
        build_adapter,
        default_config=DEFAULT_CONFIG,
        description="Benchmark TAPTRv3 on the TWIST eval datasets",
        checkpoint_key="TAPTR.CHECKPOINT",
        default_name="taptr",
    ))
