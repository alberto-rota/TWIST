"""Frozen frame encoders + coordinate-sampling helpers.

The world model needs dense per-frame features ``(B, C, Hf, Wf)`` from RGB frames
and the ability to bilinearly sample those features at sub-pixel point locations.

Two encoder variants, one interface (:class:`FrozenFrameEncoder`):

* ``"dino"`` — DINOv3 (default ``facebook/dinov3-vitl16-pretrain-lvd1689m``,
  dim 1024, patch 16). The backbone wrapper :class:`DINOv3` is ported from the
  sibling project ``unreflectanything`` (``models.py``) so the encoding API
  (config dict, ``preprocess_image``, ``tokens_to_feature_maps``) matches what
  is already in use there.
* ``"cnn"`` — a tiny from-scratch conv stack needing **no download**, for CPU
  smoke tests / boot mode (the login node has no GPU).

Coordinate convention everywhere: pixels ``(x, y)`` with origin top-left in the
clip's frame size ``(H, W)``; :func:`normalize_coords` maps them to the ``[-1, 1]``
space the network operates in and that ``grid_sample`` expects (``align_corners=True``).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Coordinate <-> grid helpers (single source of truth for normalization)
# --------------------------------------------------------------------------- #
def normalize_coords(xy: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Pixel ``(x, y)`` -> ``[-1, 1]`` (align_corners=True convention)."""
    h, w = hw
    scale = xy.new_tensor([max(w - 1, 1), max(h - 1, 1)])
    return 2.0 * xy / scale - 1.0


def denormalize_coords(xy_n: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """``[-1, 1]`` -> pixel ``(x, y)`` (inverse of :func:`normalize_coords`)."""
    h, w = hw
    scale = xy_n.new_tensor([max(w - 1, 1), max(h - 1, 1)])
    return (xy_n + 1.0) * scale / 2.0


def sample_features(feats: torch.Tensor, xy: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinearly sample ``feats (B,C,Hf,Wf)`` at points ``xy (B,N,2)`` (pixels).

    Returns ``(B, N, C)``. ``padding_mode="border"`` so points near/over the edge
    sample the boundary feature (and keep a gradient) instead of zeros.
    """
    b, c, _, _ = feats.shape
    n = xy.shape[1]
    grid = normalize_coords(xy, hw).reshape(b, n, 1, 2)              # (B,N,1,2)
    samp = F.grid_sample(feats, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return samp.squeeze(-1).permute(0, 2, 1).contiguous()           # (B,N,C)


def sample_window(
    feats: torch.Tensor, center_xy: torch.Tensor, hw: Tuple[int, int], k: int, radius_px: float
) -> torch.Tensor:
    """Sample a ``k x k`` pixel window of features around each point.

    Builds a ``k x k`` grid of pixel offsets in ``[-radius_px, radius_px]`` around
    ``center_xy (B,N,2)`` and bilinearly samples ``feats (B,C,Hf,Wf)`` at all of
    them. Returns ``(B, N, k*k, C)`` — the local cost-volume the observation model
    attends over. Plain torch (no einops).
    """
    b, c, _, _ = feats.shape
    n = center_xy.shape[1]
    lin = torch.linspace(-radius_px, radius_px, k, device=feats.device, dtype=center_xy.dtype)
    oy, ox = torch.meshgrid(lin, lin, indexing="ij")                # (k,k) each
    offs = torch.stack([ox, oy], dim=-1).reshape(1, 1, k * k, 2)    # (1,1,k*k,2)
    pts = center_xy.unsqueeze(2) + offs                             # (B,N,k*k,2)
    grid = normalize_coords(pts, hw).reshape(b, n * k * k, 1, 2)    # (B,N*k*k,1,2)
    samp = F.grid_sample(feats, grid, mode="bilinear", align_corners=True, padding_mode="border")
    return samp.squeeze(-1).permute(0, 2, 1).reshape(b, n, k * k, c).contiguous()


# --------------------------------------------------------------------------- #
# DINOv3 backbone (ported from unreflectanything/models.py — keep API in sync)
# --------------------------------------------------------------------------- #
class DINOv3(nn.Module):
    """Configurable DINOv3 backbone returning dense patch feature maps.

    Ported (lightly trimmed) from ``unreflectanything/models.py`` so the encoding
    API is identical: a config dict, ``preprocess_image`` (HF processor: resize to
    a square ``image_size`` + ImageNet normalize), and ``tokens_to_feature_maps``
    (drop CLS + 4 register tokens, reshape to ``(B, C, Hf, Wf)``).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.config = {
            "model_name": "facebook/dinov3-vitl16-pretrain-lvd1689m",
            "image_size": 256,
            "freeze_backbone": True,
            "return_last_hidden_state": True,
            "return_all_hidden_states": False,
            "return_selected_layers": None,
            "return_patch_tokens_only": True,
            "return_as_feature_maps": False,
            "return_cls_token": False,
            "return_register_tokens": False,
            **config,
        }

        self.dinov3 = AutoModel.from_pretrained(self.config["model_name"])
        self.processor = AutoImageProcessor.from_pretrained(self.config["model_name"])
        self.processor.size = {
            "height": self.config["image_size"],
            "width": self.config["image_size"],
        }

        if self.config["freeze_backbone"]:
            for p in self.dinov3.parameters():
                p.requires_grad = False

        self.feature_dim = self.dinov3.config.hidden_size   # 1024 for ViT-L/16
        self.patch_size = self.dinov3.config.patch_size     # 16 for DINOv3
        self.dinov3.config.image_size = self.config["image_size"]

    def get_patch_spatial_dims(self, h: int, w: int) -> Tuple[int, int]:
        return h // self.patch_size, w // self.patch_size

    def tokens_to_feature_maps(self, hidden, batch_size, patch_h, patch_w):
        # DINOv3 token layout: [CLS, 4 register tokens, patch tokens...]
        patch_tokens = hidden[:, 5:]                                 # (B, Np, C)
        patch_tokens = patch_tokens.transpose(1, 2).contiguous()    # (B, C, Np)
        return patch_tokens.view(batch_size, self.feature_dim, patch_h, patch_w)

    def forward(self, rgb_image: torch.Tensor) -> dict:
        """``rgb_image (B,3,H,W)`` preprocessed for DINOv3 (H,W divisible by patch)."""
        b, _, h, w = rgb_image.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0, (
            f"H,W ({h},{w}) must be divisible by patch {self.patch_size}"
        )
        patch_h, patch_w = self.get_patch_spatial_dims(h, w)
        need_all = self.config["return_all_hidden_states"] or self.config["return_selected_layers"] is not None
        outputs = self.dinov3(rgb_image, output_hidden_states=need_all)

        result = {}
        if self.config["return_last_hidden_state"]:
            last = outputs.last_hidden_state
            if self.config["return_as_feature_maps"]:
                last = self.tokens_to_feature_maps(last, b, patch_h, patch_w)
            result["last_hidden_state"] = last
        if self.config["return_cls_token"]:
            result["cls_token"] = outputs.last_hidden_state[:, 0]
        if self.config["return_selected_layers"] is not None:
            all_hidden = outputs.hidden_states
            if self.config["return_patch_tokens_only"]:
                all_hidden = [h_[:, 5:] for h_ in all_hidden]
            sel = [all_hidden[i] for i in self.config["return_selected_layers"]]
            if self.config["return_as_feature_maps"]:
                sel = [self.tokens_to_feature_maps(h_, b, patch_h, patch_w) for h_ in sel]
            result["selected_hidden_states"] = sel
        return result

    def preprocess_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """``(B,3,H,W)`` in ``[0,1]`` -> resized to ``image_size`` + ImageNet-normalized."""
        import PIL.Image

        img = (image_tensor.clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        pil = [PIL.Image.fromarray(img[i], mode="RGB") for i in range(img.shape[0])]
        processed = self.processor(images=pil, return_tensors="pt")["pixel_values"]
        return processed.to(image_tensor.device)


# --------------------------------------------------------------------------- #
# Lightweight CNN backbone (no download; CPU smoke / boot)
# --------------------------------------------------------------------------- #
class _CNNBackbone(nn.Module):
    """Tiny strided conv stack producing stride-``patch_size`` dense features.

    ``patch_size`` must be a power of two; ``log2(patch_size)`` stride-2 conv
    blocks bring the resolution down to the feature grid. Random init is fine —
    it exists to exercise shapes + gradient flow on CPU, not for accuracy.
    """

    def __init__(self, feature_dim: int = 64, patch_size: int = 8) -> None:
        super().__init__()
        n_down = int(round(math.log2(patch_size)))
        assert 2 ** n_down == patch_size, f"cnn patch_size must be a power of 2, got {patch_size}"
        chans = [3] + [max(32, feature_dim // 2)] * (n_down - 1) + [feature_dim]
        layers = []
        for i in range(n_down):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, stride=2, padding=1), nn.GELU()]
        self.net = nn.Sequential(*layers)
        self.feature_dim = feature_dim
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,H,W)->(B,C,H/p,W/p)
        return self.net(x)


# --------------------------------------------------------------------------- #
# Unified frozen encoder
# --------------------------------------------------------------------------- #
class FrozenFrameEncoder(nn.Module):
    """Dense frame encoder: ``frames (B,3,H,W)`` -> ``feats (B,C,Hf,Wf)``.

    Config keys (UPPER from the YAML are lowercased by the builder):
        variant: "dino" | "cnn"
        model_name, image_size            (dino)
        feature_dim, patch_size           (cnn; for dino they come from the backbone)
        freeze_backbone, encoder_lr       (frozen when freeze_backbone or encoder_lr == 0)

    Accepts uint8 or float frames; scales uint8/255 to ``[0,1]`` and (dino) runs
    the HF processor's ImageNet normalization. ``.feature_dim`` / ``.patch_size``
    expose the grid the downstream modules need.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        cfg = {k.lower(): v for k, v in config.items()}
        self.variant = cfg.get("variant", cfg.get("encoder", "dino"))
        encoder_lr = cfg.get("encoder_lr", 0.0)
        freeze = bool(cfg.get("freeze_backbone", True)) or (encoder_lr in (0, 0.0, None))
        self.frozen = freeze

        # Which transformer layer to read as the dense feature map. ``None`` = the
        # final hidden state (v1, default). An int selects a mid-block hidden state
        # (a precision lever: earlier DINOv3 layers can carry finer spatial detail
        # than the semantics-tuned last block). Negative indexes from the end
        # (-1 == last == None-equivalent, -4 == 4th-from-last). ``dino`` only.
        fl = cfg.get("feature_layer", None)
        self.feature_layer = None if fl in (None, "", "null") else int(fl)

        # Multi-layer fusion: a LIST of transformer layers to read as separate dense
        # maps (fused downstream in the world model). When set it takes precedence
        # over the single ``feature_layer`` and ``forward`` returns a *list* of K
        # ``(B,C,Hf,Wf)`` maps instead of one. ``dino`` only. Negative indexes from
        # the end (like ``feature_layer``).
        fls = cfg.get("feature_layers", None)
        self.feature_layers = (
            None if fls in (None, "", "null") or len(fls) == 0 else [int(x) for x in fls]
        )

        if self.variant == "cnn":
            self.backbone = _CNNBackbone(
                feature_dim=int(cfg.get("feature_dim", 64)),
                patch_size=int(cfg.get("patch_size", 8)),
            )
            self.feature_dim = self.backbone.feature_dim
            self.patch_size = self.backbone.patch_size
        elif self.variant in ("dino", "dinov3", "dinov2"):
            dino_cfg = {
                "model_name": cfg.get("model_name", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
                "image_size": int(cfg.get("image_size", 256)),
                "freeze_backbone": freeze,
                "return_as_feature_maps": True,
            }
            if self.feature_layers is not None:
                # Return K mid-transformer hidden states as separate feature maps
                # (fused downstream). return_patch_tokens_only=False so the 5-token
                # (CLS+4 reg) prefix is stripped exactly once inside
                # tokens_to_feature_maps.
                dino_cfg["return_last_hidden_state"] = False
                dino_cfg["return_selected_layers"] = list(self.feature_layers)
                dino_cfg["return_patch_tokens_only"] = False
            elif self.feature_layer is None:
                dino_cfg["return_last_hidden_state"] = True
            else:
                # Return one mid-transformer hidden state as the feature map. NB:
                # keep return_patch_tokens_only=False so the 5-token (CLS+4 reg)
                # prefix is stripped exactly once — inside tokens_to_feature_maps.
                dino_cfg["return_last_hidden_state"] = False
                dino_cfg["return_selected_layers"] = [self.feature_layer]
                dino_cfg["return_patch_tokens_only"] = False
            self.backbone = DINOv3(dino_cfg)
            self.feature_dim = self.backbone.feature_dim
            self.patch_size = self.backbone.patch_size
        else:
            raise ValueError(f"unknown encoder variant {self.variant!r} (use 'dino' or 'cnn')")

        if self.frozen:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

        # ImageNet statistics as (non-persistent) buffers: the dino path
        # preprocesses ON-DEVICE (resize when needed + normalize) instead of the
        # HF processor's GPU->CPU->PIL->GPU round-trip, which throttled every
        # training step and inflated the timed ms/frame at eval.
        self.register_buffer("_img_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("_img_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
                             persistent=False)

    @staticmethod
    def _to_unit(frames: torch.Tensor) -> torch.Tensor:
        """uint8 or float frames -> float in ``[0, 1]``."""
        if frames.dtype == torch.uint8:
            return frames.float() / 255.0
        f = frames.float()
        return f / 255.0 if float(f.max()) > 1.5 else f

    def _preprocess_dino(self, x: torch.Tensor) -> torch.Tensor:
        """``(B,3,H,W)`` in ``[0,1]`` -> DINOv3 input, entirely on-device.

        Matches the HF processor semantics (resize to the square ``image_size``
        with bicubic + antialias, then ImageNet-normalize). With the standard
        config the dataset already loads frames at ``IMAGE_SIZE`` (the top-level
        knob sets both), so the resize is skipped and this is a pure normalize.
        """
        size = int(self.backbone.config["image_size"])
        if x.shape[-2:] != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bicubic",
                              align_corners=False, antialias=True).clamp(0, 1)
        return (x - self._img_mean) / self._img_std

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames (B,3,H,W)`` -> dense features ``(B, C, Hf, Wf)``."""
        ctx = torch.no_grad() if self.frozen else nullcontext()
        with ctx:
            x = self._to_unit(frames)
            if self.variant == "cnn":
                return self.backbone(x)
            x = self._preprocess_dino(x)                   # on-device resize + normalize
            out = self.backbone(x)
            if self.feature_layers is not None:
                return out["selected_hidden_states"]       # list[K] of (B, C, Hf, Wf)
            if self.feature_layer is None:
                return out["last_hidden_state"]            # (B, C, Hf, Wf)
            return out["selected_hidden_states"][0]        # (B, C, Hf, Wf)
