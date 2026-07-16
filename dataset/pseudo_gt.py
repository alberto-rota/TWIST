"""Pseudo ground-truth dense track generation for point tracking.

Generates synthetic video sequences with known dense point correspondences
from a single RGB image by combining estimated geometry, smooth random
camera trajectories, and time-varying 3D scene deformations.

Usage::

    from dataset.pseudo_gt import (
        PseudoGTGenerator, TrajectoryConfig, DeformationConfig, GridConfig,
    )

    gen = PseudoGTGenerator(448, 448, device="cuda")
    result = gen.generate(
        image=I0, depth=D0, intrinsics=K,
        trajectory=TrajectoryConfig(n_frames=24),
        deformation=DeformationConfig(),
        grid=GridConfig(grid_size=32),
        seed=42,
    )
    PseudoGTGenerator.log_to_rerun(result)
    PseudoGTGenerator.render_video(result, "tracks.mp4")

For the high-level TWIST entry point that runs geometry + generation in one
call and returns the canonical tracking item dict, use
:func:`generate_pseudo_tracks` (defined at the bottom of this module).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from geometry.projections import BackProject, Project
from geometry.transforms import euler2mat

__all__ = [
    "ScalarOrFloatRange",
    "ScalarOrIntRange",
    "TrajectoryConfig",
    "DeformationConfig",
    "GridConfig",
    "OccluderConfig",
    "PseudoGTResult",
    "PseudoGTGenerator",
    "trajectory_config_from_run_config",
    "deformation_config_from_run_config",
    "occluder_config_from_run_config",
    "generate_pseudo_tracks",
    "assemble_pseudo_batch",
    "clear_pseudo_track_caches",
]


# ── sampling helpers ────────────────────────────────────────────────────────


def _uniform(g: torch.Generator, lo: float, hi: float, device) -> float:
    """Sample a scalar uniformly in [lo, hi]."""
    return lo + (hi - lo) * torch.rand(1, device=device, generator=g).item()


def _uniform_int(g: torch.Generator, lo: int, hi: int, device) -> int:
    """Sample an integer uniformly in [lo, hi] inclusive."""
    return torch.randint(lo, hi + 1, (1,), device=device, generator=g).item()


ScalarOrFloatRange = Union[float, Tuple[float, float]]
ScalarOrIntRange = Union[int, Tuple[int, int]]


def _sample_f(g: torch.Generator, x: ScalarOrFloatRange, device) -> float:
    """Fixed float or uniform sample from ``[lo, hi]``."""
    if isinstance(x, tuple):
        return _uniform(g, float(x[0]), float(x[1]), device)
    return float(x)


def _sample_i(g: torch.Generator, x: ScalarOrIntRange, device) -> int:
    """Fixed int or uniform integer in ``[lo, hi]`` inclusive."""
    if isinstance(x, tuple):
        return _uniform_int(g, int(x[0]), int(x[1]), device)
    return int(x)


def _gaussian_smooth_1d(
    signal: torch.Tensor, sigma: float, device
) -> torch.Tensor:
    """Smooth a ``[T, C]`` signal along dim-0 with a 1-D Gaussian kernel."""
    T, C = signal.shape
    if T <= 1:
        return signal

    ks = max(int(4 * sigma) | 1, 3)
    if ks % 2 == 0:
        ks += 1
    # ``F.pad(..., mode="reflect")`` requires padding < length on that axis.
    # Here length is ``T``; pad = ks // 2, so enforce ks <= 2 * T - 1 (odd).
    max_ks = max(2 * T - 1, 1)
    if max_ks % 2 == 0:
        max_ks -= 1
    ks = min(ks, max_ks)
    if ks < 3:
        return signal

    ax = torch.arange(ks, device=device, dtype=torch.float32) - ks // 2
    kernel = torch.exp(-0.5 * (ax / max(sigma, 0.1)) ** 2)
    kernel = (kernel / kernel.sum()).view(1, 1, ks)  # [1, 1, ks]

    sig = signal.T.unsqueeze(1)  # [C, 1, T]  (C acts as batch)
    pad = ks // 2
    sig = F.pad(sig, (pad, pad), mode="reflect")
    out = F.conv1d(sig, kernel)  # [C, 1, T]
    return out.squeeze(1).T  # [T, C]


def _catmull_rom_interp(
    waypoints: torch.Tensor,  # [W, C]
    T: int,
    device,
) -> torch.Tensor:
    """Catmull-Rom spline interpolation of ``W`` waypoints to ``T`` samples.

    Returns ``[T, C]``.
    """
    W, C = waypoints.shape
    t_out = torch.linspace(0, W - 1, T, device=device)  # [T]

    # pad waypoints with duplicated endpoints for boundary tangents
    pts = torch.cat(
        [waypoints[:1], waypoints, waypoints[-1:]], dim=0
    )  # [W+2, C]

    # find segment index for each output sample
    seg = torch.clamp(t_out.long(), 0, W - 2)  # [T]  segment 0..W-2
    frac = t_out - seg.float()  # [T]  local param in [0, 1)

    p0 = pts[seg]      # [T, C]   (padded index: seg+0 in original = seg in padded)
    p1 = pts[seg + 1]  # [T, C]
    p2 = pts[seg + 2]  # [T, C]
    p3 = pts[seg + 3]  # [T, C]

    tt = frac.unsqueeze(1)   # [T, 1]
    tt2 = tt * tt
    tt3 = tt2 * tt

    # Catmull-Rom basis (tau = 0.5)
    out = 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * tt
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * tt2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * tt3
    )  # [T, C]
    return out


def _resample_1d(
    signal: torch.Tensor,  # [T, C]
    new_indices: torch.Tensor,  # [T]  float indices into dim-0
    device,
) -> torch.Tensor:
    """Linearly resample ``signal`` at fractional ``new_indices``."""
    T, C = signal.shape
    idx0 = new_indices.long().clamp(0, T - 2)
    idx1 = (idx0 + 1).clamp(0, T - 1)
    frac = (new_indices - idx0.float()).unsqueeze(1)  # [T, 1]
    return signal[idx0] * (1.0 - frac) + signal[idx1] * frac  # [T, C]


def _make_temporal_profiles(
    K: int,
    T: int,
    sigma: float,
    rng: torch.Generator,
    device,
) -> torch.Tensor:
    """Smoothed, enveloped random temporal profiles in ``[-1, 1]``.

    Returns shape ``[K, T]``.
    """
    raw = torch.randn(K, T, device=device, generator=rng)
    smoothed = _gaussian_smooth_1d(raw.T, sigma, device).T  # [K, T]
    envelope = torch.sin(torch.linspace(0, torch.pi, T, device=device))  # [T]
    smoothed = smoothed * envelope.unsqueeze(0)
    mx = smoothed.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    return smoothed / mx  # [K, T]


# ── configuration dataclasses ──────────────────────────────────────────────


@dataclass
class TrajectoryConfig:
    """Distribution parameters for stochastic camera trajectory generation.

    Each numeric field may be a **scalar** (fixed) or **``(lo, hi)``** (sampled
    on every :meth:`PseudoGTGenerator.generate` call).

    Attributes
    ----------
    n_frames : int
        Number of output frames (temporal length of the sequence).
    z_bias_range
        Starting-point Z offset.  Positive ⇒ camera starts *closer* to the
        surface (zoom-in); negative ⇒ *farther* (zoom-out).
    complexity_range
        Controls trajectory morphology on a ``[0, 1]`` scale.

        * **0** — gentle, nearly straight-line motion (high temporal
          smoothing, small rotations).
        * **1** — winding path with sharp direction changes and large
          rotations.
    translation_range
        Per-frame translation magnitude (scene-metric units, per axis).
    rotation_range_deg
        Per-frame rotation magnitude (degrees, per axis).
    forward_bias_range
        Constant +Z drift per frame (endoscopic forward-zoom feel).
    speed_scale_range
        Global speed multiplier applied to all deltas.
    still_fraction_range
        Fraction of the trajectory spent in near-still / slow-motion segments.
    """

    n_frames: int = 24

    z_bias_range: ScalarOrFloatRange = (-0.1, 0.1)
    complexity_range: ScalarOrFloatRange = (0.3, 0.7)

    translation_range: ScalarOrFloatRange = (0.02, 0.08)
    rotation_range_deg: ScalarOrFloatRange = (2.0, 8.0)
    forward_bias_range: ScalarOrFloatRange = (0.005, 0.025)

    speed_scale_range: ScalarOrFloatRange = (0.5, 1.5)
    still_fraction_range: ScalarOrFloatRange = (0.0, 0.15)

    # Hard safety lower bound on the closest camera-to-cloud distance as a
    # fraction of the scene's initial nearest-point distance.  When any
    # frame of the sampled trajectory would breach this clearance, ALL
    # translations are globally scaled down by a single bisected factor
    # so that the worst-case frame exactly matches the bound.  Prevents
    # the 'novel view becomes mostly black because the camera flew into
    # the cloud' failure (points become too sparse in screen-space for
    # the median-filter inpainting to cope with).  Set ``0`` to disable.
    min_scene_clearance_frac: float = 0.5


@dataclass
class DeformationConfig:
    """Distribution parameters for stochastic scene deformation.

    ``*_range`` fields follow the same scalar vs ``(lo, hi)`` convention as
    :class:`TrajectoryConfig`.

    Attributes
    ----------
    n_deformers_range
        Number of independent deformation control points.
    sigma_frac_range
        Gaussian kernel radius as a fraction of the scene's spatial extent.
    amplitude_frac_range
        Peak displacement as a fraction of the scene's spatial extent
        (used by *drag* and *inflate* deformers).
    drag_weight, inflate_weight, twist_weight : float
        Relative probabilities for assigning each deformer's primary type.
    twist_max_deg_range
        Maximum Rodrigues-rotation angle for *twist*-type deformers.
    temporal_smooth_range
        Gaussian smoothing σ (in frames) for the deformation temporal profiles.
    """

    n_deformers_range: ScalarOrIntRange = (2, 5)

    sigma_frac_range: ScalarOrFloatRange = (0.08, 0.25)
    amplitude_frac_range: ScalarOrFloatRange = (0.005, 0.03)

    drag_weight: float = 0.5
    inflate_weight: float = 0.3
    twist_weight: float = 0.2

    twist_max_deg_range: ScalarOrFloatRange = (1.0, 5.0)
    temporal_smooth_range: ScalarOrFloatRange = (2.0, 6.0)

    # Optional per-type overrides for sigma / amplitude.  When ``None``
    # (the default) the shared ``sigma_frac_range`` / ``amplitude_frac_range``
    # is used instead — keeps backward compatibility.  Per-type overrides
    # let you sculpt qualitatively different behaviours, e.g. large-smooth-
    # low-amplitude inflations + small-concentrated-strong twists without
    # cross-contaminating drag.
    drag_sigma_frac_range: Optional[ScalarOrFloatRange] = None
    drag_amplitude_frac_range: Optional[ScalarOrFloatRange] = None
    inflate_sigma_frac_range: Optional[ScalarOrFloatRange] = None
    inflate_amplitude_frac_range: Optional[ScalarOrFloatRange] = None
    twist_sigma_frac_range: Optional[ScalarOrFloatRange] = None

    # Hard cap on inflate peak 3-D displacement as a fraction of the
    # scene extent.  Inflation moves points radially outward from the
    # control centre; once the step exceeds the mean 3-D spacing of
    # nearby points the screen-space cloud thins out enough to produce
    # black holes even after inpainting.  Effective amplitude is clamped
    # to ``min(sampled_amp, max_inflate_displacement_frac * extent)``.
    max_inflate_displacement_frac: float = 0.015

    # Softens the unit-radial direction near the control centre so that
    # points within a tiny ball of the centre don't receive the full
    # ``1 / ||d||`` spike.  Implemented as ``d / sqrt(||d||^2 + eps^2)``
    # with ``eps = inflate_radial_softness * extent``.  Bigger values =
    # smoother / bubblier inflations; very small = sharper spikes.
    inflate_radial_softness: float = 0.05


@dataclass
class OccluderConfig:
    """Stochastic 2-D occluder sprites composited into the rendered clip.

    Occluders produce **dense pixel-accurate visibility negatives**: the
    novel-view depth-only visibility almost never flags occlusions (single
    depth layer, small parallax, smooth deformations), so the BCE head has
    no negatives to learn from.  Each occluder is a soft Gaussian blob that

    * moves along a straight line (with optional sinusoidal perturbation),
    * has a raised-cosine temporal lifetime (fades in / out),
    * is alpha-composited on top of the warped frames,
    * marks any query whose ``(u, v)`` falls under ``alpha > vis_alpha_threshold``
      as ``visibility = False`` at that frame,
    * multiplies ``frame_valid`` by ``(1 - alpha)`` so downstream masks
      discard the occluded pixels for position / descriptor losses.

    Attributes
    ----------
    n_range
        Number of occluders sampled per clip (inclusive int range).
        ``[0, 0]`` disables the feature.
    size_frac_range
        Gaussian sigma as a fraction of ``min(H, W)``.
    motion_frac_range
        Displacement from start to end centre as a fraction of ``min(H, W)``.
        Sampled per occluder; the direction is uniform on the unit circle.
    sinusoid_frac_range
        Amplitude of a sinusoidal perpendicular perturbation added to the
        linear path (fraction of ``min(H, W)``); ``0`` disables.
    lifetime_frac_range
        Fraction of the window length ``T`` that each occluder is active.
    appearance_value_range
        Per-channel RGB value range for the solid-colour base of the sprite.
    appearance_noise_std
        Per-pixel additive Gaussian noise std on top of the base colour
        (keeps the appearance non-trivial so the backbone can't cheat on
        "flat colour → occluded"); clamped to ``[0, 1]`` after compositing.
    vis_alpha_threshold
        Query is marked occluded when sampled ``alpha >`` this value.
    apply_to_frame_valid
        When ``True``, multiplies ``frame_valid`` by ``(1 - alpha)``.
    """

    n_range: ScalarOrIntRange = (1, 3)
    size_frac_range: ScalarOrFloatRange = (0.05, 0.15)
    motion_frac_range: ScalarOrFloatRange = (0.0, 0.5)
    sinusoid_frac_range: ScalarOrFloatRange = (0.0, 0.05)
    lifetime_frac_range: ScalarOrFloatRange = (0.3, 0.9)
    appearance_value_range: ScalarOrFloatRange = (0.0, 1.0)
    appearance_noise_std: float = 0.05
    vis_alpha_threshold: float = 0.5
    apply_to_frame_valid: bool = True

    # Domain-aware colour sampling.  When ``True`` each occluder's base
    # RGB is sampled as the mean colour of a random square patch of the
    # **source image** (endoscopic colours only, no out-of-domain blue
    # sprites on red tissue), optionally multiplied by a brightness
    # jitter.  When ``False`` falls back to the legacy uniform
    # ``appearance_value_range`` sampler.
    sample_color_from_image: bool = True
    color_patch_frac_range: ScalarOrFloatRange = (0.02, 0.08)
    color_brightness_range: ScalarOrFloatRange = (0.6, 1.2)


@dataclass
class GridConfig:
    """Dense query-point grid layout.

    Attributes
    ----------
    grid_size : int
        The grid is ``grid_size × grid_size`` (total Q = grid_size²).
    margin_frac : float
        Margin from image edges as a fraction of the image dimension.
    """

    grid_size: int = 32
    margin_frac: float = 0.03


def _mapping_get(obj: Any, key: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _coerce_scalar_or_float_range(value: Any, *, field_name: str) -> ScalarOrFloatRange:
    """Parse a YAML value into a fixed float or ``(lo, hi)`` pair."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{field_name}: expected length-2 sequence [lo, hi] or a scalar, got {value!r}"
            )
        return (float(value[0]), float(value[1]))
    if isinstance(value, bool):
        raise TypeError(f"{field_name}: expected float, int, or [lo, hi], not bool")
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        f"{field_name}: expected float, int, or [lo, hi], got {type(value).__name__}"
    )


def _coerce_scalar_or_int_range(value: Any, *, field_name: str) -> ScalarOrIntRange:
    """Parse a YAML value into a fixed int or ``(lo, hi)`` inclusive int pair."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{field_name}: expected length-2 sequence [lo, hi] or a scalar, got {value!r}"
            )
        return (int(value[0]), int(value[1]))
    if isinstance(value, bool):
        raise TypeError(f"{field_name}: expected int or [lo, hi] ints, not bool")
    if isinstance(value, (int, float)):
        return int(value)
    raise TypeError(
        f"{field_name}: expected int or [lo, hi] ints, got {type(value).__name__}"
    )


def trajectory_config_from_run_config(config: Any, *, n_frames: int) -> TrajectoryConfig:
    """Merge optional ``PSEUDO_GT_TRAJECTORY`` from run config with defaults.

    ``n_frames`` is always taken from the caller (batch temporal length ``T``),
    not from YAML, so pseudo clips match the training / val window.

    Parameters
    ----------
    config
        Flat mapping (e.g. DotMap). Reads subtree ``PSEUDO_GT_TRAJECTORY`` if set.
    n_frames
        Output trajectory length (typically ``TRACKING_SEQUENCE_LENGTH``).
    """
    base = TrajectoryConfig(n_frames=int(n_frames))
    sec = _mapping_get(config, "PSEUDO_GT_TRAJECTORY")
    if sec is None:
        return base
    overrides: dict = {}
    for fname in (
        "z_bias_range",
        "complexity_range",
        "translation_range",
        "rotation_range_deg",
        "forward_bias_range",
        "speed_scale_range",
        "still_fraction_range",
    ):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        overrides[fname] = _coerce_scalar_or_float_range(raw, field_name=fname)
    raw = _mapping_get(sec, "min_scene_clearance_frac")
    if raw is not None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(
                "min_scene_clearance_frac: expected float, got "
                f"{type(raw).__name__}"
            )
        overrides["min_scene_clearance_frac"] = float(raw)
    return replace(base, **overrides) if overrides else base


def deformation_config_from_run_config(config: Any) -> DeformationConfig:
    """Merge optional ``PSEUDO_GT_DEFORMATION`` from run config with defaults."""
    base = DeformationConfig()
    sec = _mapping_get(config, "PSEUDO_GT_DEFORMATION")
    if sec is None:
        return base
    overrides: dict = {}
    raw_nd = _mapping_get(sec, "n_deformers_range")
    if raw_nd is not None:
        overrides["n_deformers_range"] = _coerce_scalar_or_int_range(
            raw_nd, field_name="n_deformers_range",
        )
    for fname in (
        "sigma_frac_range",
        "amplitude_frac_range",
        "twist_max_deg_range",
        "temporal_smooth_range",
        "drag_sigma_frac_range",
        "drag_amplitude_frac_range",
        "inflate_sigma_frac_range",
        "inflate_amplitude_frac_range",
        "twist_sigma_frac_range",
    ):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        overrides[fname] = _coerce_scalar_or_float_range(raw, field_name=fname)
    for fname in (
        "drag_weight", "inflate_weight", "twist_weight",
        "max_inflate_displacement_frac", "inflate_radial_softness",
    ):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{fname}: expected float, got {type(raw).__name__}")
        overrides[fname] = float(raw)
    return replace(base, **overrides) if overrides else base


def occluder_config_from_run_config(config: Any) -> Optional["OccluderConfig"]:
    """Merge optional ``PSEUDO_GT_OCCLUDERS`` from run config with defaults.

    Returns ``None`` when the config block is absent **or** when
    ``n_range`` explicitly disables occluders (max ≤ 0). Returning ``None``
    is the canonical "feature off" signal consumed by :meth:`generate`.
    """
    sec = _mapping_get(config, "PSEUDO_GT_OCCLUDERS")
    if sec is None:
        return None
    base = OccluderConfig()
    overrides: dict = {}

    raw_n = _mapping_get(sec, "n_range")
    if raw_n is not None:
        overrides["n_range"] = _coerce_scalar_or_int_range(
            raw_n, field_name="n_range",
        )

    for fname in (
        "size_frac_range",
        "motion_frac_range",
        "sinusoid_frac_range",
        "lifetime_frac_range",
        "appearance_value_range",
        "color_patch_frac_range",
        "color_brightness_range",
    ):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        overrides[fname] = _coerce_scalar_or_float_range(raw, field_name=fname)
    raw = _mapping_get(sec, "sample_color_from_image")
    if raw is not None:
        overrides["sample_color_from_image"] = bool(raw)

    for fname in ("appearance_noise_std", "vis_alpha_threshold"):
        raw = _mapping_get(sec, fname)
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{fname}: expected float, got {type(raw).__name__}")
        overrides[fname] = float(raw)

    raw = _mapping_get(sec, "apply_to_frame_valid")
    if raw is not None:
        overrides["apply_to_frame_valid"] = bool(raw)

    cfg = replace(base, **overrides) if overrides else base

    # "n_range max <= 0" → feature explicitly disabled.
    n_hi = cfg.n_range[1] if isinstance(cfg.n_range, tuple) else cfg.n_range
    if int(n_hi) <= 0:
        return None
    return cfg


@dataclass
class PseudoGTResult:
    """Complete output of a single pseudo ground-truth generation call.

    All tensors live on the same device that was used during generation.
    """

    # core tracking data
    frames: torch.Tensor  # [T, 3, H, W]
    tracks: torch.Tensor  # [T, Q, 2]
    visibility: torch.Tensor  # [T, Q]  bool
    query_pixels: torch.Tensor  # [Q, 2]
    # Novel-view RGB validity from Project holemask (1 = reliable, 0 = hole / inpaint)
    frame_valid: torch.Tensor  # [T, 1, H, W] float in [0, 1]

    # camera
    poses: torch.Tensor  # [T, 4, 4]
    intrinsics: torch.Tensor  # [1, 3, 3]

    # source data
    source_image: torch.Tensor  # [1, 3, H, W]
    depth: torch.Tensor  # [1, 1, H, W]

    # 3-D data (for visualisation)
    cloud_ref: torch.Tensor  # [1, 4, N]
    cloud_rgb: torch.Tensor  # [1, 3, N]
    query_3d_ref: torch.Tensor  # [1, 4, Q]
    clouds_deformed: List[torch.Tensor]  # T × [1, 4, N]
    queries_deformed: List[torch.Tensor]  # T × [1, 4, Q]

    # spatial dimensions
    height: int
    width: int

    zbuf_debug: Optional[torch.Tensor] = None  # [T, H, W, 3] uint8 RGB


# ── generator ──────────────────────────────────────────────────────────────


class PseudoGTGenerator:
    """One-call pseudo ground-truth track generator.

    Instantiate once (sets up ``BackProject`` / ``Project`` buffers for a
    given resolution), then call :meth:`generate` with different images and
    configs.
    """

    def __init__(self, height: int, width: int, device: str = "cuda"):
        self.height = height
        self.width = width
        self.device = device
        self._backproject = BackProject(height, width).to(device)
        self._project = Project(height, width).to(device)

    # ── public API ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        trajectory: Optional[TrajectoryConfig] = None,
        deformation: Optional[DeformationConfig] = None,
        grid: Optional[GridConfig] = None,
        occluders: Optional[OccluderConfig] = None,
        seed: Optional[int] = None,
        randomize_trajectory: bool = True,
        visibility_z_tol_frac: float = 0.04,
        visibility_z_abs_min: float = 1e-3,
        visibility_depth_dilate: int = 1,
        visibility_query_patch_rad: int = 0,
        visibility_temporal_window: int = 3,
        visibility_splat_rad: int = 1,
        visibility_respect_frame_valid: bool = True,
        frame_valid_erode_px: int = 0,
        novel_view_median_kernel: int = 7,
    ) -> PseudoGTResult:
        """Generate pseudo-GT tracks from a single image + geometry.

        Parameters
        ----------
        image : Tensor [1, 3, H, W]
            Source RGB image normalised to ``[0, 1]``.
        depth : Tensor [1, 1, H, W]
            Metric depth (**not** normalised to ``[0, 1]``).
        intrinsics : Tensor [1, 3, 3]
            Camera intrinsic matrix.
        trajectory : TrajectoryConfig, optional
        deformation : DeformationConfig, optional
        grid : GridConfig, optional
        occluders : OccluderConfig, optional
            Stochastic 2-D sprite occluders; ``None`` disables the feature
            (default, matches the pre-occluder behaviour).
        seed : int, optional
            Seeds the main RNG (deformation, etc.).  If *None*, non-deterministic.
        randomize_trajectory : bool, default ``True``
            If *True*, camera path uses a separate RNG with fresh entropy each
            call (different path even when ``seed`` is fixed).  If *False*, the
            trajectory shares the main RNG (fully reproducible with ``seed``).
        visibility_z_tol_frac, visibility_z_abs_min
            Relative and absolute depth tolerances for z-buffer visibility.
        visibility_depth_dilate
            Odd kernel size for min-pooling the splatted depth map.
        visibility_query_patch_rad
            Pixel patch radius when reading the z-buffer at each query.
        visibility_temporal_window
            Odd temporal max-pool width on visibility; ``1`` disables.
        frame_valid_erode_px
            Erosion radius (pixels) on ``frame_valid`` to shrink valid regions away
            from splat / inpaint boundaries; ``0`` disables.

        Returns
        -------
        PseudoGTResult
        """
        trajectory = trajectory or TrajectoryConfig()
        deformation = deformation or DeformationConfig()
        grid = grid or GridConfig()

        dev = self.device
        H, W = self.height, self.width
        T = trajectory.n_frames

        rng = torch.Generator(device=dev)
        if seed is not None:
            rng.manual_seed(seed)
        else:
            rng.seed()

        if randomize_trajectory:
            rng_traj = torch.Generator(device=dev)
            rng_traj.seed()
        else:
            rng_traj = rng

        invK = torch.inverse(intrinsics)  # [1, 3, 3]

        # ── back-project scene cloud + query grid ───────────────────────
        bp = self._backproject(image, depth, invK)
        X0 = bp["xyz1"]  # [1, 4, H*W]
        C0 = bp["rgb"]  # [1, 3, H*W]

        margin = max(1, int(min(H, W) * grid.margin_frac))
        gx = torch.linspace(margin, W - margin, grid.grid_size, device=dev)
        gy = torch.linspace(margin, H - margin, grid.grid_size, device=dev)
        gy_g, gx_g = torch.meshgrid(gy, gx, indexing="ij")
        Q0_px = torch.stack(
            [gx_g.reshape(-1), gy_g.reshape(-1)], dim=-1
        ).unsqueeze(0)  # [1, Q, 2]

        bp_q = self._backproject(image, depth, invK, points_match=Q0_px)
        Q0_3d = bp_q["points_match_3d"]  # [1, 4, Q]

        # ── trajectory + deformation ────────────────────────────────────
        poses = self._build_trajectory(trajectory, rng_traj, dev)  # [T, 4, 4]
        poses = PseudoGTGenerator._clamp_trajectory_to_scene(
            poses,
            X0[0, :3, :],          # [3, N]
            min_clearance_frac=float(trajectory.min_scene_clearance_frac),
        )
        deform_fn = self._build_deformation(
            deformation, X0[:, :3, :], T, rng, dev
        )

        # ── render loop ─────────────────────────────────────────────────
        all_frames: List[torch.Tensor] = []
        all_tracks: List[torch.Tensor] = []
        all_vis: List[torch.Tensor] = []
        all_X: List[torch.Tensor] = []
        all_Q: List[torch.Tensor] = []
        all_zbuf_dbg: List[torch.Tensor] = []
        all_frame_valid: List[torch.Tensor] = []

        for t in range(T):
            Xt = deform_fn(X0, t)  # [1, 4, H*W]
            Qt = deform_fn(Q0_3d, t)  # [1, 4, Q]

            Pt = poses[t].unsqueeze(0)  # [1, 4, 4]

            proj_out = self._project(
                Xt, C0, intrinsics, Pt,
                points_match_3d=Qt,
                return_mask=True,
                median_kernel_size=novel_view_median_kernel,
            )

            Pt_inv = torch.inverse(Pt)
            Xt_cam = torch.bmm(Pt_inv, Xt)  # [1, 4, N]
            Qt_cam = torch.bmm(Pt_inv, Qt)  # [1, 4, Q]
            _, vis_t, zbuf_dbg_t = self._compute_visibility(
                Xt_cam,
                Qt_cam,
                intrinsics,
                H,
                W,
                z_tol_frac=visibility_z_tol_frac,
                z_abs_min=visibility_z_abs_min,
                dilate_k=visibility_depth_dilate,
                query_patch_rad=visibility_query_patch_rad,
                splat_rad=visibility_splat_rad,
            )

            all_frames.append(proj_out["warped"])
            all_tracks.append(proj_out["matches"])
            hm = proj_out.get("mask")
            if hm is None:
                fv = torch.ones(1, 1, H, W, device=dev, dtype=torch.float32)
            else:
                fv = hm[:, :1, :, :].to(dtype=torch.float32).clamp(0.0, 1.0)
            all_frame_valid.append(fv)
            all_vis.append(vis_t)
            all_X.append(Xt)
            all_Q.append(Qt)
            all_zbuf_dbg.append(zbuf_dbg_t)

        vis_stack = torch.cat(all_vis, dim=0)  # [T, Q]
        if visibility_temporal_window > 1:
            vis_stack = PseudoGTGenerator._smooth_visibility_temporal(
                vis_stack, visibility_temporal_window
            )

        frame_valid_stack = torch.cat(all_frame_valid, dim=0)  # [T, 1, H, W]
        frame_valid_stack = PseudoGTGenerator._erode_valid_mask_2d(
            frame_valid_stack, frame_valid_erode_px
        )

        # Fold the novel-view holemask into the visibility GT: a query
        # whose projected pixel lands in a rendering hole (frame_valid == 0)
        # cannot be a reliable positive because the RGB there is either
        # black or median-inpainted garbage — mark it occluded.
        if visibility_respect_frame_valid:
            vis_stack = PseudoGTGenerator._mask_visibility_by_frame_valid(
                vis_stack, frame_valid_stack,
                tracks=torch.cat(all_tracks, dim=0),
            )

        frames_stack = torch.cat(all_frames, dim=0)  # [T, 3, H, W]
        tracks_stack = torch.cat(all_tracks, dim=0)  # [T, Q, 2]

        if occluders is not None:
            frames_stack, vis_stack, frame_valid_stack = (
                PseudoGTGenerator._apply_occluders(
                    frames=frames_stack,
                    tracks=tracks_stack,
                    visibility=vis_stack,
                    frame_valid=frame_valid_stack,
                    cfg=occluders,
                    source_image=image,
                    rng=rng,
                    device=dev,
                )
            )

        return PseudoGTResult(
            frames=frames_stack,                         # [T, 3, H, W]
            tracks=tracks_stack,                          # [T, Q, 2]
            visibility=vis_stack,                        # [T, Q]
            query_pixels=Q0_px.squeeze(0),               # [Q, 2]
            frame_valid=frame_valid_stack,
            poses=poses,
            intrinsics=intrinsics,
            source_image=image,
            depth=depth,
            cloud_ref=X0,
            cloud_rgb=C0,
            query_3d_ref=Q0_3d,
            clouds_deformed=all_X,
            queries_deformed=all_Q,
            height=H,
            width=W,
            zbuf_debug=torch.cat(all_zbuf_dbg, dim=0),  # [T, H, W, 3]
        )

    # ── trajectory ──────────────────────────────────────────────────────

    @staticmethod
    def _build_trajectory(
        cfg: TrajectoryConfig,
        rng: torch.Generator,
        device,
    ) -> torch.Tensor:
        """Sample a smooth random SE(3) camera trajectory.

        Uses a **waypoint-spline** strategy: sample ``n_waypoints`` random
        6-DoF poses, then interpolate with cubic B-spline to ``T`` frames.
        The *complexity* knob controls how many waypoints (more = more
        direction changes and rotations).

        Returns ``[T, 4, 4]`` cumulative poses (frame-0 includes only the
        *z_bias* offset).
        """
        T = cfg.n_frames

        complexity = _sample_f(rng, cfg.complexity_range, device)
        z_bias = _sample_f(rng, cfg.z_bias_range, device)
        trans_ampl = _sample_f(rng, cfg.translation_range, device)
        rot_ampl_deg = _sample_f(rng, cfg.rotation_range_deg, device)
        rot_ampl = torch.deg2rad(torch.tensor(rot_ampl_deg, device=device))
        fwd_bias = _sample_f(rng, cfg.forward_bias_range, device)
        spd_scale = _sample_f(rng, cfg.speed_scale_range, device)
        still_frac = _sample_f(rng, cfg.still_fraction_range, device)

        rot_scale = 0.3 + 0.9 * complexity  # 0→0.3x  1→1.2x

        # per-axis random scale  [3]
        trans_ax = trans_ampl * (
            0.5 + torch.rand(3, device=device, generator=rng)
        )
        rot_ax = rot_ampl * rot_scale * (
            0.5 + torch.rand(3, device=device, generator=rng)
        )

        # number of waypoints driven by complexity: 2 (gentle) .. ~T/3 (winding)
        n_wp = max(2, int(2 + (T / 3 - 2) * complexity))

        # random waypoint poses in 6-DoF (translation + euler)
        wp = torch.randn(n_wp, 6, device=device, generator=rng)  # [n_wp, 6]
        wp[:, :3] *= trans_ax * spd_scale
        wp[:, 3:] *= rot_ax
        wp[:, 2] += fwd_bias * T / max(n_wp, 1)

        # cumulative sum so waypoints form a path
        wp = torch.cumsum(wp, dim=0)  # [n_wp, 6]
        # prepend origin
        wp = torch.cat([torch.zeros(1, 6, device=device), wp], dim=0)  # [n_wp+1, 6]

        # cubic B-spline interpolation to T frames
        poses_6dof = _catmull_rom_interp(wp, T, device)  # [T, 6]

        # speed profile with optional still / slow segments
        speed = PseudoGTGenerator._make_speed_profile(
            T, still_frac, complexity, rng, device
        )  # [T]

        # apply speed modulation: re-parameterise arc by integrating speed
        arc = torch.cumsum(speed, dim=0)         # [T]
        arc = arc / arc[-1]                       # normalise to [0, 1]
        arc = arc * (T - 1)                       # map back to frame indices
        # re-sample poses_6dof at the new arc positions (linear interp)
        poses_6dof = _resample_1d(poses_6dof, arc, device)  # [T, 6]

        poses_6dof[0] = 0.0
        poses_6dof[:, 2] += z_bias

        return euler2mat(poses_6dof)  # [T, 4, 4]

    @staticmethod
    def _make_speed_profile(
        T: int,
        still_frac: float,
        complexity: float,
        rng: torch.Generator,
        device,
    ) -> torch.Tensor:
        """Smooth speed envelope ``[T]`` in ``(0, 1]`` with near-still dips."""
        speed = torch.ones(T, device=device)
        if still_frac < 0.01 or T < 6:
            return speed

        n_still = max(2, int(T * still_frac))
        n_seg = max(1, min(3, int(1 + 2 * complexity)))
        frames_per = max(2, n_still // n_seg)
        t_idx = torch.arange(T, device=device, dtype=torch.float32)

        for _ in range(n_seg):
            ctr = _uniform_int(rng, 2, max(3, T - 2), device)
            hw = max(frames_per // 2, 1)
            dip = torch.exp(-0.5 * ((t_idx - ctr) / hw) ** 2)
            speed = speed * (1.0 - 0.95 * dip)

        return speed.clamp(min=0.02)

    @staticmethod
    def _clamp_trajectory_to_scene(
        poses: torch.Tensor,            # [T, 4, 4] cam-to-world in src-cam frame
        cloud_xyz: torch.Tensor,        # [3, N] in src-cam frame
        min_clearance_frac: float,
        n_subsample: int = 2048,
        n_bisect_iter: int = 20,
    ) -> torch.Tensor:
        r"""Globally scale translations so the camera never enters the cloud.

        The source camera is at the origin, so the initial clearance is
        ``z_min = min_i ||cloud[i]||``.  The target clearance is
        ``thr = min_clearance_frac * z_min``.  We bisect a scalar
        ``s in (0, 1]`` applied to every frame's translation:

        .. math::

            s^{*} = \max \{ s \in (0, 1]\; : \; \min_{t, i} \| c_i - s \cdot \tau_t \| \ge \mathrm{thr} \}.

        Scaling is uniform to preserve the trajectory's *shape* (only the
        amplitude shrinks) and to keep the motion recognisable.  Rotations
        are untouched.

        Subsamples the cloud to ``n_subsample`` points (uniform at random)
        for the pairwise distance call to stay cheap even for full-frame
        clouds with ``N ≈ H·W ≈ 2·10^5``.
        """
        if min_clearance_frac <= 0.0:
            return poses
        T = poses.shape[0]
        dev = poses.device
        N = cloud_xyz.shape[1]

        if N > n_subsample:
            idx = torch.randint(0, N, (n_subsample,), device=dev)
            cloud_s = cloud_xyz[:, idx].T  # [M, 3]
        else:
            cloud_s = cloud_xyz.T  # [N, 3]

        z_min = cloud_s.norm(dim=1).min().item()
        thr = float(min_clearance_frac) * float(z_min)

        orig_trans = poses[:, :3, 3].clone()  # [T, 3]

        def worst_dist(scale: float) -> float:
            cam = scale * orig_trans  # [T, 3]
            d = torch.cdist(cam.unsqueeze(0), cloud_s.unsqueeze(0)).squeeze(0)
            return d.min().item()

        if worst_dist(1.0) >= thr:
            return poses  # nothing to do

        # Bisect in [0, 1] for the largest s such that worst_dist(s) >= thr.
        # s = 0 always yields worst_dist = z_min >= thr by construction.
        s_lo, s_hi = 0.0, 1.0
        for _ in range(n_bisect_iter):
            s_mid = 0.5 * (s_lo + s_hi)
            if worst_dist(s_mid) >= thr:
                s_lo = s_mid
            else:
                s_hi = s_mid
        scale = max(s_lo, 1e-3)

        poses = poses.clone()
        poses[:, :3, 3] = scale * orig_trans
        return poses

    @staticmethod
    def _mask_visibility_by_frame_valid(
        vis: torch.Tensor,           # [T, Q] bool
        frame_valid: torch.Tensor,   # [T, 1, H, W] float in [0, 1]
        tracks: torch.Tensor,        # [T, Q, 2] pixel coords (x, y)
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """AND visibility with ``frame_valid >= threshold`` sampled at each track.

        Novel-view pixels that ended up as holes (``frame_valid == 0``) are
        rendered either black or with median-inpainted garbage — any query
        projecting there cannot be a reliable visibility positive, so we
        flip it to False.
        """
        T, _, H, W = frame_valid.shape
        gx = 2.0 * tracks[..., 0].clamp(0, max(W - 1, 1)) / max(W - 1, 1) - 1.0
        gy = 2.0 * tracks[..., 1].clamp(0, max(H - 1, 1)) / max(H - 1, 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [T, Q, 1, 2]
        v_at_q = F.grid_sample(
            frame_valid, grid, mode='bilinear',
            padding_mode='zeros', align_corners=True,
        ).squeeze(1).squeeze(-1)  # [T, Q]
        return vis & (v_at_q >= threshold)

    # ── deformation ─────────────────────────────────────────────────────

    @staticmethod
    def _build_deformation(
        cfg: DeformationConfig,
        cloud_xyz: torch.Tensor,  # [1, 3, N]
        T: int,
        rng: torch.Generator,
        device,
    ):
        """Build a deformation closure ``fn(X, t) -> X_def``.

        The returned callable accepts ``X  [1, 3or4, N]`` and an integer
        frame index ``t`` and returns a deformed copy.  Type is sampled
        first so sigma / amplitude can use per-type overrides from ``cfg``
        (e.g. big-smooth-low-amplitude inflations + small-strong twists).
        Inflation peak displacement is additionally hard-clamped to
        ``cfg.max_inflate_displacement_frac * extent`` to prevent
        screen-space tearing.
        """
        K = _sample_i(rng, cfg.n_deformers_range, device)
        if K == 0:
            return lambda X, t: X  # identity

        xyz = cloud_xyz[0]  # [3, N]
        N = xyz.shape[1]
        centroid = xyz.mean(dim=1, keepdim=True)  # [3, 1]
        extent = torch.quantile((xyz - centroid).norm(dim=0), 0.9).item()

        # control-point centres sampled from the cloud
        idx = torch.randint(0, N, (K,), device=device, generator=rng)
        centres = xyz[:, idx].T  # [K, 3]

        # ── sample type first so per-type sigma/amp can be used ─────────
        total_w = cfg.drag_weight + cfg.inflate_weight + cfg.twist_weight
        p_drag = cfg.drag_weight / total_w
        p_infl = cfg.inflate_weight / total_w
        types = torch.zeros(K, dtype=torch.long, device=device)
        for k in range(K):
            r = torch.rand(1, device=device, generator=rng).item()
            if r < p_drag:
                types[k] = 0
            elif r < p_drag + p_infl:
                types[k] = 1
            else:
                types[k] = 2

        # per-type range lookup with fallback to shared range
        sigma_ranges = (
            cfg.drag_sigma_frac_range or cfg.sigma_frac_range,     # drag
            cfg.inflate_sigma_frac_range or cfg.sigma_frac_range,  # inflate
            cfg.twist_sigma_frac_range or cfg.sigma_frac_range,    # twist
        )
        amp_ranges = (
            cfg.drag_amplitude_frac_range or cfg.amplitude_frac_range,     # drag
            cfg.inflate_amplitude_frac_range or cfg.amplitude_frac_range,  # inflate
            cfg.amplitude_frac_range,  # twist — strength driven by twist_max_deg_range
        )

        sigma_list: List[float] = []
        amp_list: List[float] = []
        max_inflate_disp = float(cfg.max_inflate_displacement_frac) * extent
        for k in range(K):
            tk = int(types[k].item())
            sigma_list.append(extent * _sample_f(rng, sigma_ranges[tk], device))
            amp = extent * _sample_f(rng, amp_ranges[tk], device)
            if tk == 1:  # inflate — cap amplitude to avoid pixel holes
                amp = min(amp, max_inflate_disp)
            amp_list.append(amp)

        sigmas = torch.tensor(sigma_list, device=device, dtype=torch.float32)
        amplitudes = torch.tensor(amp_list, device=device, dtype=torch.float32)

        drag_dir = F.normalize(
            torch.randn(K, 3, device=device, generator=rng), dim=1
        )  # [K, 3]

        twist_axis = F.normalize(
            torch.randn(K, 3, device=device, generator=rng), dim=1
        )  # [K, 3]
        twist_max = torch.deg2rad(
            torch.tensor(
                [_sample_f(rng, cfg.twist_max_deg_range, device) for _ in range(K)],
                device=device,
            )
        )  # [K]

        t_smooth = _sample_f(rng, cfg.temporal_smooth_range, device)
        profiles = _make_temporal_profiles(K, T, t_smooth, rng, device)  # [K, T]

        # softened-radial epsilon for inflate: eps = soft_frac * extent
        radial_eps = float(cfg.inflate_radial_softness) * max(extent, 1e-6)

        # ── closure ----------------------------------------------------------

        def _deform(X: torch.Tensor, t: int) -> torch.Tensor:
            has_homo = X.shape[1] == 4
            pts = X[:, :3, :]  # [1, 3, M]
            M = pts.shape[2]
            prof = profiles[:, t]  # [K]

            diff = pts[0].T.unsqueeze(0) - centres.unsqueeze(1)  # [K, M, 3]
            dist_sq = (diff ** 2).sum(dim=2)  # [K, M]
            w = torch.exp(
                -dist_sq / (2.0 * sigmas.unsqueeze(1) ** 2)
            )  # [K, M]

            disp = torch.zeros(1, 3, M, device=device)

            for k in range(K):
                scale_k = w[k] * prof[k] * amplitudes[k]  # [M]

                if types[k] == 0:  # drag
                    disp = disp + (
                        scale_k.unsqueeze(0) * drag_dir[k].unsqueeze(1)
                    ).unsqueeze(0)  # [1, 3, M]

                elif types[k] == 1:  # inflate / deflate — softened radial
                    # d / sqrt(||d||^2 + eps^2) instead of d / ||d||:
                    # bounded direction near the centre → no spike, plus
                    # the magnitude grows smoothly from 0 at the centre,
                    # avoiding the tiny near-singularity dense region.
                    denom = torch.sqrt(dist_sq[k] + radial_eps ** 2).unsqueeze(1)  # [M, 1]
                    radial = diff[k] / denom  # [M, 3]
                    disp = disp + (
                        scale_k.unsqueeze(1) * radial
                    ).T.unsqueeze(0)  # [1, 3, M]

                else:  # twist (Rodrigues)
                    angle_k = prof[k] * twist_max[k]
                    if angle_k.abs() < 1e-7:
                        continue
                    ax = twist_axis[k]  # [3]
                    d = diff[k]  # [M, 3]
                    ca, sa = torch.cos(angle_k), torch.sin(angle_k)
                    cross = torch.stack([
                        ax[1] * d[:, 2] - ax[2] * d[:, 1],
                        ax[2] * d[:, 0] - ax[0] * d[:, 2],
                        ax[0] * d[:, 1] - ax[1] * d[:, 0],
                    ], dim=1)  # [M, 3]
                    dot = (ax.unsqueeze(0) * d).sum(1, keepdim=True)  # [M, 1]
                    rot_d = d * ca + cross * sa + ax * dot * (1 - ca)
                    tw = (rot_d - d) * w[k].unsqueeze(1)  # [M, 3]
                    disp = disp + tw.T.unsqueeze(0)

            out = pts + disp
            if has_homo:
                return torch.cat([out, X[:, 3:4, :]], dim=1)
            return out

        return _deform

    # ── synthetic occluders ─────────────────────────────────────────────

    @staticmethod
    def _sample_occluder_colors(
        source_image: torch.Tensor,   # [1, 3, H, W] in [0, 1]
        cfg: "OccluderConfig",
        K: int,
        rng: torch.Generator,
        device,
    ) -> torch.Tensor:
        """Per-occluder base RGB in ``[0, 1]^3`` of shape ``[K, 3]``.

        When ``cfg.sample_color_from_image`` is True each colour is the
        mean of a random square patch of the source image, multiplied by
        a brightness jitter sampled from ``cfg.color_brightness_range``.
        This keeps occluders inside the dataset's colour manifold (no
        out-of-domain blue sprites on red tissue).  Otherwise falls back
        to the legacy uniform sampler bounded by
        ``cfg.appearance_value_range``.
        """
        if not getattr(cfg, "sample_color_from_image", False):
            return torch.stack([
                torch.tensor(
                    [_sample_f(rng, cfg.appearance_value_range, device) for _ in range(3)],
                    device=device, dtype=torch.float32,
                )
                for _ in range(K)
            ])  # [K, 3]

        _, _, H, W = source_image.shape
        min_dim = float(min(H, W))

        colors = torch.empty(K, 3, device=device, dtype=torch.float32)
        for k in range(K):
            patch_frac = _sample_f(rng, cfg.color_patch_frac_range, device)
            patch_size = max(2, int(round(patch_frac * min_dim)))
            cy = _uniform_int(rng, 0, max(H - patch_size, 0), device)
            cx = _uniform_int(rng, 0, max(W - patch_size, 0), device)
            patch = source_image[0, :, cy:cy + patch_size, cx:cx + patch_size]
            color = patch.reshape(3, -1).mean(dim=1)  # [3]
            brightness = _sample_f(rng, cfg.color_brightness_range, device)
            colors[k] = (color * brightness).clamp(0.0, 1.0)
        return colors

    @staticmethod
    def _apply_occluders(
        frames: torch.Tensor,        # [T, 3, H, W] float in [0, 1]
        tracks: torch.Tensor,        # [T, Q, 2] pixel coords (x, y)
        visibility: torch.Tensor,    # [T, Q] bool
        frame_valid: torch.Tensor,   # [T, 1, H, W] float
        cfg: "OccluderConfig",
        source_image: torch.Tensor,  # [1, 3, H, W] RGB in [0, 1]
        rng: torch.Generator,
        device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Composite stochastic 2-D occluder sprites into the rendered clip.

        Produces dense pixel-accurate visibility negatives for the BCE head.

        Pipeline (fully vectorized across frames and pixels; Python loop only
        over the small number of occluders K ≤ max(n_range) ≈ 5):

        1. Sample K occluders, each with a linear-plus-sinusoidal 2-D centre
           trajectory, a raised-cosine temporal lifetime envelope, a solid
           base colour and small per-pixel Gaussian noise.
        2. Build per-frame alpha  ``α_t(x, y) ∈ [0, 1]``  via the
           order-independent over approximation
           ``α = 1 - ∏_k (1 - α_k)`` and a corresponding RGB weighted
           by each sprite's alpha contribution.
        3. Blend   ``I'_t = I_t (1 - α_t) + C_t α_t``.
        4. Sample ``α_t`` at each track location (bilinear) and mark
           ``visibility[t, q] = False`` where ``α > vis_alpha_threshold``.
        5. Multiply ``frame_valid *= (1 - α)`` (optional) so that downstream
           composite supervision masks also discard occluded positions.

        Returns:
            frames_out  [T, 3, H, W]
            vis_out     [T, Q]  bool
            valid_out   [T, 1, H, W]
        """
        T, _, H, W = frames.shape
        _, Q, _ = tracks.shape

        n_range = cfg.n_range if isinstance(cfg.n_range, tuple) else (int(cfg.n_range), int(cfg.n_range))
        K = _uniform_int(rng, int(n_range[0]), int(n_range[1]), device)
        if K <= 0:
            return frames, visibility, frame_valid

        min_dim = float(min(H, W))

        # Per-occluder sampled parameters --------------------------------
        sizes_px = torch.stack([
            torch.tensor(
                _sample_f(rng, cfg.size_frac_range, device) * min_dim,
                device=device, dtype=torch.float32,
            )
            for _ in range(K)
        ])  # [K]  (Gaussian sigma in pixels)

        motion_frac = torch.stack([
            torch.tensor(_sample_f(rng, cfg.motion_frac_range, device),
                         device=device, dtype=torch.float32)
            for _ in range(K)
        ])  # [K]
        sinusoid_frac = torch.stack([
            torch.tensor(_sample_f(rng, cfg.sinusoid_frac_range, device),
                         device=device, dtype=torch.float32)
            for _ in range(K)
        ])  # [K]
        life_frac = torch.stack([
            torch.tensor(_sample_f(rng, cfg.lifetime_frac_range, device),
                         device=device, dtype=torch.float32)
            for _ in range(K)
        ])  # [K]
        base_rgb = PseudoGTGenerator._sample_occluder_colors(
            source_image=source_image, cfg=cfg, K=K, rng=rng, device=device,
        )  # [K, 3]

        # Start / end pixel centres (allow sliding from slightly off-screen).
        margin = 0.15 * min_dim
        start_px = torch.stack([
            torch.stack([
                torch.rand((), device=device, generator=rng) * (W + 2 * margin) - margin,
                torch.rand((), device=device, generator=rng) * (H + 2 * margin) - margin,
            ])
            for _ in range(K)
        ])  # [K, 2]
        # Uniform direction on the circle × motion_frac * min_dim
        theta = torch.stack([
            torch.rand((), device=device, generator=rng) * 2.0 * torch.pi
            for _ in range(K)
        ])  # [K]
        direction = torch.stack([theta.cos(), theta.sin()], dim=-1)  # [K, 2]
        end_px = start_px + direction * (motion_frac.unsqueeze(-1) * min_dim)

        # Sinusoid phase + unit-normal perpendicular to direction
        phase = torch.stack([
            torch.rand((), device=device, generator=rng) * 2.0 * torch.pi
            for _ in range(K)
        ])  # [K]
        perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)  # [K, 2]

        # Temporal lifetime: raised-cosine bump of half-width = life_frac * T / 2
        # centred at a random frame inside the clip.
        t_mid = torch.stack([
            torch.tensor(
                _uniform(rng, 0.0, float(T - 1), device),
                device=device, dtype=torch.float32,
            )
            for _ in range(K)
        ])  # [K]
        t_half = (life_frac * float(T) / 2.0).clamp_min(1.0)  # [K]

        t_idx = torch.arange(T, device=device, dtype=torch.float32)  # [T]
        u = t_idx.view(1, T) / max(T - 1, 1)  # [1, T]  normalised 0..1 for traj param
        # Centres: [K, T, 2]
        centres = (
            start_px.unsqueeze(1) * (1.0 - u.unsqueeze(-1))
            + end_px.unsqueeze(1) * u.unsqueeze(-1)
            + perp.unsqueeze(1) * (
                (sinusoid_frac * min_dim).view(K, 1, 1)
                * torch.sin(u.unsqueeze(-1) * 2.0 * torch.pi + phase.view(K, 1, 1))
            )
        )  # [K, T, 2]

        # Raised-cosine active envelope: cos^2(π/2 · phase) inside window, 0 outside
        phase_t = (t_idx.view(1, T) - t_mid.view(K, 1)) / t_half.view(K, 1)  # [K, T]
        in_window = (phase_t.abs() <= 1.0).float()
        active = torch.cos(0.5 * torch.pi * phase_t.clamp(-1.0, 1.0)) ** 2 * in_window  # [K, T]

        # Per-occluder alpha and accumulated over-composite ---------------
        yy = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
        xx = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)

        alpha_acc = torch.zeros(T, H, W, device=device, dtype=torch.float32)
        rgb_acc = torch.zeros(T, 3, H, W, device=device, dtype=torch.float32)
        w_acc = torch.zeros(T, H, W, device=device, dtype=torch.float32)

        for k in range(K):
            cx = centres[k, :, 0].view(T, 1, 1)  # [T, 1, 1]
            cy = centres[k, :, 1].view(T, 1, 1)
            s2 = (2.0 * sizes_px[k].clamp_min(1.0) ** 2)
            alpha_k = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / s2)  # [T, H, W]
            alpha_k = alpha_k * active[k].view(T, 1, 1)
            # Unconditional numerical clamp against fp drift
            alpha_k = alpha_k.clamp(0.0, 1.0)

            rgb_k = base_rgb[k].view(1, 3, 1, 1).expand(T, 3, H, W)
            rgb_acc = rgb_acc + rgb_k * alpha_k.unsqueeze(1)
            w_acc = w_acc + alpha_k
            alpha_acc = alpha_acc + alpha_k * (1.0 - alpha_acc)

        alpha = alpha_acc.clamp(0.0, 1.0)                 # [T, H, W]
        occ_rgb = rgb_acc / w_acc.clamp_min(1e-6).unsqueeze(1)  # [T, 3, H, W]

        # Optional per-pixel noise for appearance diversity (clamped 0..1).
        noise_std = float(cfg.appearance_noise_std)
        if noise_std > 0.0:
            noise = torch.randn(
                T, 3, H, W, device=device, dtype=torch.float32, generator=rng,
            ) * noise_std
            occ_rgb = (occ_rgb + noise).clamp(0.0, 1.0)

        # Composite into RGB ---------------------------------------------
        alpha_bchw = alpha.unsqueeze(1)  # [T, 1, H, W]
        frames_out = frames * (1.0 - alpha_bchw) + occ_rgb * alpha_bchw
        frames_out = frames_out.clamp(0.0, 1.0)

        # Sample alpha at each track location (bilinear) -----------------
        # tracks: [T, Q, 2] in pixel coords (x, y). Build a per-frame grid
        # of shape [T, Q, 1, 2] normalised to [-1, 1].
        gx = 2.0 * tracks[..., 0].clamp(0.0, float(max(W - 1, 1))) / float(max(W - 1, 1)) - 1.0
        gy = 2.0 * tracks[..., 1].clamp(0.0, float(max(H - 1, 1))) / float(max(H - 1, 1)) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [T, Q, 1, 2]
        alpha_at_q = F.grid_sample(
            alpha_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(1).squeeze(-1)  # [T, Q]

        covered = alpha_at_q > float(cfg.vis_alpha_threshold)  # [T, Q]
        vis_out = visibility & (~covered)

        if cfg.apply_to_frame_valid:
            valid_out = (frame_valid * (1.0 - alpha_bchw)).clamp(0.0, 1.0)
        else:
            valid_out = frame_valid

        return frames_out, vis_out, valid_out

    # ── validity (novel-view holes) ─────────────────────────────────────

    @staticmethod
    def _erode_valid_mask_2d(
        valid: torch.Tensor,  # [T, 1, H, W] float 0..1
        erode_px: int,
    ) -> torch.Tensor:
        """Erode valid regions by dilating the invalid set (max-pool on 1 - valid)."""
        if erode_px is None or int(erode_px) <= 0:
            return valid
        k = 2 * int(erode_px) + 1
        inv = 1.0 - valid
        inv_d = F.max_pool2d(inv, kernel_size=k, stride=1, padding=int(erode_px))
        return (1.0 - inv_d).clamp(0.0, 1.0)

    # ── visibility ──────────────────────────────────────────────────────

    @staticmethod
    def _query_neighborhood_zbuf_min(
        zbuf_flat: torch.Tensor,  # [1, H * W]
        H: int,
        W: int,
        u: torch.Tensor,  # [1, Q]  long
        v: torch.Tensor,  # [1, Q]  long
        rad: int,
    ) -> torch.Tensor:
        """Min dilated z-buffer over a ``(2*rad+1)²`` patch; returns ``[1, Q]``."""
        if rad <= 0:
            return zbuf_flat.gather(1, v * W + u)
        dev = u.device
        offs = torch.arange(-rad, rad + 1, device=dev, dtype=torch.long)
        uu = u.unsqueeze(2).unsqueeze(3) + offs.view(1, 1, -1, 1)
        vv = v.unsqueeze(2).unsqueeze(3) + offs.view(1, 1, 1, -1)
        uu = (uu + torch.zeros_like(vv)).clamp(0, W - 1)
        vv = (vv + torch.zeros_like(uu)).clamp(0, H - 1)
        lin = (vv * W + uu).reshape(1, -1)
        z_nb = zbuf_flat.gather(1, lin).view(1, u.shape[1], -1)
        z_nb = z_nb.masked_fill(torch.isinf(z_nb), float("inf"))
        return z_nb.min(dim=-1).values

    @staticmethod
    def _smooth_visibility_temporal(vis: torch.Tensor, window: int) -> torch.Tensor:
        """OR-pool visibility over ``window`` frames (odd), ``[T, Q]``."""
        if window <= 1 or vis.shape[0] < 2:
            return vis
        if window % 2 == 0:
            window = window + 1
        w = window // 2
        x = vis.float().permute(1, 0).unsqueeze(1)  # [Q, 1, T]
        x = F.pad(x, (w, w), mode="replicate")
        y = F.max_pool1d(x, kernel_size=window, stride=1, padding=0)
        return y.squeeze(1).permute(1, 0) > 0.5

    @staticmethod
    def _compute_visibility(
        cloud_cam: torch.Tensor,  # [1, 4, N]
        query_cam: torch.Tensor,  # [1, 4, Q]
        K: torch.Tensor,  # [1, 3, 3]
        H: int,
        W: int,
        z_tol_frac: float = 0.04,
        z_abs_min: float = 1e-3,
        dilate_k: int = 1,
        query_patch_rad: int = 0,
        splat_rad: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Float z-buffer visibility + debug RGB overlay ``[1, H, W, 3]`` uint8.

        Strategy
        --------
        For every 3-D cloud point ``c_i`` with camera-frame depth ``z_i``
        we **splat to a ``(2·splat_rad+1)²`` pixel neighbourhood** around
        its projected pixel with ``scatter_reduce(amin)`` on ``z_i``.  This
        drastically reduces z-buffer holes caused by the "one point per
        pixel" undersampling (after projection + rotation + deformation
        the source cloud no longer covers every output pixel, especially
        for inflations where point spacing grows).

        A query ``q`` with depth ``z_q`` projecting to pixel ``(u_q, v_q)``
        is declared **visible** iff:

        .. math::

            z_q > 0 \;\land\; z_q \le \mathrm{zbuf}(u_q, v_q) + \max(\tau_{\mathrm{frac}} \cdot |z_q|, \tau_{\mathrm{abs}})

        i.e. ``q`` is at most ``z_tol`` behind the *nearest* splat over
        that pixel (small positive slack absorbs splat quantisation).

        Differences vs. the pre-fix implementation:

        1. **Multi-pixel splat** (``splat_rad=1`` → 3×3) instead of one
           pixel + 5×5 min-pool dilation.  Min-pool dilation *leaked*
           near-field depths into neighbouring background pixels, causing
           spurious occlusions at depth discontinuities (queries near the
           edges of closer surfaces were flagged occluded even though no
           surface is actually in front of them).
        2. **No query neighbourhood min** (``query_patch_rad=0`` default).
           The old 3×3 patch at the query combined with the 5×5 dilation
           created a 7×7 "nearest depth hunt" that grabbed foreground
           from far away — too conservative.
        3. **Bilinear zbuf lookup** at the query's subpixel coordinate
           when ``query_patch_rad <= 0``.  Eliminates the integer-round
           jitter that made visibility flicker frame-to-frame on queries
           sitting between pixel centres.
        """
        dev = cloud_cam.device
        B = 1

        proj_c = torch.bmm(K, cloud_cam[:, :3, :])  # [1, 3, N]
        uv_c = proj_c[:, :2, :] / proj_c[:, 2:3, :].clamp(min=1e-6)
        z_c = cloud_cam[:, 2, :]  # [1, N]

        u_c0 = uv_c[:, 0, :].round().long()
        v_c0 = uv_c[:, 1, :].round().long()

        # Multi-pixel splat with scatter_reduce(amin).  Build
        # (lin_idx, z_value) pairs of shape [B, N * Ks] by replicating each
        # point across its splat neighbourhood; out-of-bounds / behind-camera
        # entries are masked to +inf so they never win the amin reduce.
        splat_rad = max(int(splat_rad), 0)
        if splat_rad == 0:
            offs_u = torch.zeros(1, device=dev, dtype=torch.long)
            offs_v = torch.zeros(1, device=dev, dtype=torch.long)
        else:
            r = splat_rad
            ax = torch.arange(-r, r + 1, device=dev, dtype=torch.long)
            offs_v, offs_u = torch.meshgrid(ax, ax, indexing="ij")
            offs_u = offs_u.reshape(-1)
            offs_v = offs_v.reshape(-1)
        Ks = offs_u.numel()

        uu = u_c0.unsqueeze(-1) + offs_u.view(1, 1, -1)  # [B, N, Ks]
        vv = v_c0.unsqueeze(-1) + offs_v.view(1, 1, -1)  # [B, N, Ks]
        in_bounds_c = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        uu = uu.clamp(0, W - 1)
        vv = vv.clamp(0, H - 1)
        lin = (vv * W + uu).reshape(B, -1)                          # [B, N*Ks]
        z_rep = z_c.unsqueeze(-1).expand(-1, -1, Ks).reshape(B, -1)  # [B, N*Ks]
        INF = float("inf")
        z_rep = torch.where(
            in_bounds_c.reshape(B, -1) & (z_rep > 0),
            z_rep,
            torch.full_like(z_rep, INF),
        )

        zbuf_min = torch.full((B, H * W), INF, device=dev)
        zbuf_min.scatter_reduce_(1, lin, z_rep, reduce="amin", include_self=True)

        # Optional extra dilation: only active when ``dilate_k > 1``.
        dilate_k = max(int(dilate_k), 1)
        if dilate_k % 2 == 0:
            dilate_k += 1
        zbuf_2d = zbuf_min.view(B, 1, H, W)
        if dilate_k > 1:
            pad = dilate_k // 2
            zbuf_2d = -F.max_pool2d(
                -zbuf_2d, kernel_size=dilate_k, stride=1, padding=pad,
            )
        zbuf = zbuf_2d.view(B, H * W)

        # Debug overlay: red = no splat, yellow = conflicting depths at pixel,
        # green = consistent coverage.
        finite_mask = (z_rep < INF).float()
        hit_count = torch.zeros(B, H * W, device=dev)
        hit_count.scatter_reduce_(
            1, lin, finite_mask, reduce="sum", include_self=False,
        )
        zbuf_max = torch.full((B, H * W), 0.0, device=dev)
        z_rep_finite = torch.where(
            z_rep < INF, z_rep, torch.zeros_like(z_rep)
        )
        zbuf_max.scatter_reduce_(
            1, lin, z_rep_finite, reduce="amax", include_self=True,
        )
        gap_mask = hit_count < 0.5
        spread = (zbuf_max - zbuf_min.clamp(max=1e6)).abs()
        med_z = zbuf_min.clamp(min=1e-6)
        conflict_mask = (~gap_mask) & (spread > z_tol_frac * med_z)

        dbg = torch.zeros(B, H * W, 3, device=dev, dtype=torch.uint8)
        dbg[gap_mask] = torch.tensor([255, 0, 0], device=dev, dtype=torch.uint8)
        dbg[conflict_mask] = torch.tensor(
            [255, 220, 0], device=dev, dtype=torch.uint8
        )
        dbg[(~gap_mask) & (~conflict_mask)] = torch.tensor(
            [0, 0, 0], device=dev, dtype=torch.uint8
        )
        zbuf_debug = dbg.view(B, H, W, 3)

        proj_q = torch.bmm(K, query_cam[:, :3, :])  # [1, 3, Q]
        uv_q = proj_q[:, :2, :] / proj_q[:, 2:3, :].clamp(min=1e-6)
        z_q = query_cam[:, 2, :]  # [1, Q]

        in_bounds = (
            (uv_q[:, 0, :] >= 0)
            & (uv_q[:, 0, :] < W)
            & (uv_q[:, 1, :] >= 0)
            & (uv_q[:, 1, :] < H)
        )

        if query_patch_rad > 0:
            u_q = uv_q[:, 0, :].round().long().clamp(0, W - 1)
            v_q = uv_q[:, 1, :].round().long().clamp(0, H - 1)
            zbuf_at_q = PseudoGTGenerator._query_neighborhood_zbuf_min(
                zbuf, H, W, u_q, v_q, query_patch_rad,
            )
        else:
            # Bilinear lookup on a "safe" zbuf (finite in-bounds, large at
            # holes).  We use a large sentinel (1e9) in place of +inf so
            # grid_sample's linear interpolation stays finite, then restore
            # +inf afterwards to keep "no splat nearby" queries occluded.
            zbuf_safe = torch.where(
                torch.isinf(zbuf), torch.full_like(zbuf, 1e9), zbuf,
            ).view(B, 1, H, W)
            gx = 2.0 * uv_q[:, 0, :].clamp(0, W - 1) / max(W - 1, 1) - 1.0
            gy = 2.0 * uv_q[:, 1, :].clamp(0, H - 1) / max(H - 1, 1) - 1.0
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(2)  # [B, Q, 1, 2]
            zbuf_at_q = F.grid_sample(
                zbuf_safe, grid, mode="bilinear",
                padding_mode="border", align_corners=True,
            ).squeeze(1).squeeze(-1)  # [B, Q]
            zbuf_at_q = torch.where(
                zbuf_at_q > 1e8,
                torch.full_like(zbuf_at_q, INF),
                zbuf_at_q,
            )

        z_tol = torch.maximum(
            z_tol_frac * z_q.abs(),
            z_q.new_tensor(z_abs_min),
        )
        visible = in_bounds & (z_q > 0) & (z_q <= zbuf_at_q + z_tol)

        return uv_q.permute(0, 2, 1), visible, zbuf_debug

    # ── visualisation helpers ───────────────────────────────────────────

    @staticmethod
    def log_to_rerun(
        result: PseudoGTResult,
        *,
        subsample_cloud: int = 8,
        n_tracks_3d: int = 128,
    ) -> None:
        """Log a complete :class:`PseudoGTResult` to an **already initialised**
        Rerun recording.

        Parameters
        ----------
        result : PseudoGTResult
        subsample_cloud : int
            Point-cloud subsampling factor for the per-frame 3-D cloud.
        n_tracks_3d : int
            Number of query trajectories to draw as 3-D polylines.
        """
        import rerun as rr
        from matplotlib import colormaps

        T = result.frames.shape[0]
        Q = result.tracks.shape[1]
        H, W = result.height, result.width
        K_np = result.intrinsics[0].cpu().numpy()

        cmap = colormaps["hsv"]
        qcolors = (
            cmap(np.linspace(0, 1, Q, endpoint=False))[:, :3] * 255
        ).astype(np.uint8)  # [Q, 3]

        # static: pinhole
        rr.log(
            "world/camera",
            rr.Pinhole(image_from_camera=K_np, width=W, height=H),
            static=True,
        )

        # static: camera path
        cam_xyz = (
            torch.stack([result.poses[t, :3, 3] for t in range(T)])
            .cpu()
            .numpy()
        )  # [T, 3]
        rr.log(
            "world/cam_path",
            rr.LineStrips3D([cam_xyz], colors=[[0, 200, 255]]),
            static=True,
        )

        # static: 3-D track polylines (subsampled)
        show_idx = np.linspace(0, Q - 1, min(n_tracks_3d, Q), dtype=int)
        for qi in show_idx:
            pts = (
                torch.stack([result.queries_deformed[t][0, :3, qi] for t in range(T)])
                .cpu()
                .numpy()
            )  # [T, 3]
            rr.log(
                f"world/tracks3d/{qi}",
                rr.LineStrips3D([pts], colors=[qcolors[qi].tolist()]),
                static=True,
            )

        # per-frame
        sub = subsample_cloud
        for t in range(T):
            rr.set_time_sequence("frame", t)

            Xt = result.clouds_deformed[t][0, :3, ::sub].T.cpu().numpy()
            Ct = (
                (result.cloud_rgb[0, :, ::sub].T.cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            rr.log("world/cloud", rr.Points3D(Xt, colors=Ct, radii=0.002))

            P = result.poses[t].cpu().numpy()
            rr.log(
                "world/camera",
                rr.Transform3D(translation=P[:3, 3], mat3x3=P[:3, :3]),
            )

            img = (
                (result.frames[t].permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            rr.log("world/camera/image", rr.Image(img))

            uv = result.tracks[t].cpu().numpy()  # [Q, 2]
            vis = result.visibility[t].cpu().numpy()  # [Q]
            vc = qcolors.copy()
            vc[~vis] = [60, 60, 60]
            rr.log(
                "world/camera/tracks2d",
                rr.Points2D(uv, colors=vc, radii=2.0),
            )

    @staticmethod
    def render_video(
        result: PseudoGTResult,
        path: str,
        *,
        fps: int = 8,
        trail_length: int = 8,
        log_to_rerun: bool = True,
        zbuf_overlay_alpha: float = 0.0,
        predicted_visibility: Optional[torch.Tensor] = None,
        predicted_tracks: Optional[torch.Tensor] = None,
        draw_legend: bool = True,
    ) -> None:
        """Write an MP4 video with track dots and coloured trails.

        Visibility is rendered explicitly:

        * **visible GT** — solid filled circle, bright trail.
        * **occluded GT** — hollow coloured ring + semi-transparent fill +
          dashed / faded trail. The point is still shown at its (warped)
          ground-truth location so you can see *where* the tracker should
          recover the track to once visibility returns.

        When ``predicted_visibility`` is given, each point also gets an
        outer confusion-matrix ring with the colour convention

        =============================  =================================
        prediction vs. GT              outer ring colour (BGR)
        =============================  =================================
        visible (TP) / occluded (TN)   green  — correct visibility
        predicted visible but occluded red    — false positive (FP)
        predicted occluded but visible orange — false negative (FN)
        =============================  =================================

        Args:
            result:                A :class:`PseudoGTResult`.
            path:                  Output ``.mp4`` file path.
            fps:                   Output frame rate.
            trail_length:          Polyline history length.
            log_to_rerun:          Also log annotated frames to Rerun if
                a recording is active.
            zbuf_overlay_alpha:    Blend ``[0, 1]`` for z-buffer debug
                overlay; ``0`` = off. When ``> 0`` the output is
                side-by-side: tracks | tracks + overlay.
            predicted_visibility:  Optional bool/float tensor ``[T, Q]``
                (or ``[Q, T]``). When given, a coloured outer ring
                encodes prediction-vs-GT confusion.
            predicted_tracks:      Optional ``[T, Q, 2]`` tensor with the
                predicted pixel coordinates. When given, the predicted
                point is drawn as a small white diamond so you can
                eyeball the position error next to the GT dot.
            draw_legend:           Draw an in-frame legend (top-right).
        """
        import cv2
        from matplotlib import colormaps

        T = result.frames.shape[0]
        Q = result.tracks.shape[1]
        H, W = result.height, result.width

        cmap = colormaps["hsv"]
        colors_bgr = (
            cmap(np.linspace(0, 1, Q, endpoint=False))[:, :3] * 255
        ).astype(np.uint8)[:, ::-1].copy()

        import tempfile, subprocess, shutil
        _has_ffmpeg = shutil.which("ffmpeg") is not None
        tmp_path = tempfile.mktemp(suffix=".mp4") if _has_ffmpeg else path

        alpha = float(np.clip(zbuf_overlay_alpha, 0.0, 1.0))
        has_zbuf = alpha > 0 and result.zbuf_debug is not None
        if has_zbuf:
            zbuf_np = result.zbuf_debug.cpu().numpy()

        out_w = W * 2 if has_zbuf else W
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (out_w, H))

        trk = result.tracks.cpu().numpy().astype(np.int32)  # [T, Q, 2]
        vis = result.visibility.cpu().numpy().astype(bool)  # [T, Q]

        # Optional prediction tensors — accept [T, Q] or [Q, T] for visibility.
        pred_vis_np: Optional[np.ndarray] = None
        if predicted_visibility is not None:
            pv = predicted_visibility.detach().cpu()
            if pv.dtype != torch.bool:
                pv = pv > 0.5
            pv = pv.numpy()
            if pv.shape == (Q, T):
                pv = pv.T
            if pv.shape != (T, Q):
                raise ValueError(
                    f"predicted_visibility must have shape [T={T}, Q={Q}] "
                    f"or [Q, T]; got {tuple(pv.shape)}"
                )
            pred_vis_np = pv

        pred_trk_np: Optional[np.ndarray] = None
        if predicted_tracks is not None:
            pt = predicted_tracks.detach().cpu().numpy()
            if pt.shape != (T, Q, 2):
                raise ValueError(
                    f"predicted_tracks must have shape [T={T}, Q={Q}, 2]; "
                    f"got {tuple(pt.shape)}"
                )
            pred_trk_np = pt.astype(np.int32)

        try:
            import rerun as rr
            has_rr = log_to_rerun
        except ImportError:
            has_rr = False

        # Pre-computed drawing helpers ---------------------------------
        def _dashed_polyline(
            img_bgr: np.ndarray,
            pts_xy: np.ndarray,  # [L, 2] int32
            color_bgr: tuple,
            thickness: int = 1,
            dash: int = 4,
            gap: int = 3,
        ) -> None:
            """Draw a dashed polyline (cv2 has no native dashed stroke)."""
            if pts_xy.shape[0] < 2:
                return
            for k in range(pts_xy.shape[0] - 1):
                p1 = pts_xy[k]
                p2 = pts_xy[k + 1]
                seg = np.linalg.norm(p2 - p1)
                if seg < 1e-3:
                    continue
                n = max(1, int(np.ceil(seg / (dash + gap))))
                t_vals = np.linspace(0, 1, 2 * n + 1)
                for m in range(n):
                    a = p1 + (p2 - p1) * t_vals[2 * m]
                    b = p1 + (p2 - p1) * t_vals[2 * m + 1]
                    cv2.line(
                        img_bgr,
                        (int(a[0]), int(a[1])),
                        (int(b[0]), int(b[1])),
                        color_bgr, thickness, cv2.LINE_AA,
                    )

        def _draw_legend(img_bgr: np.ndarray) -> None:
            lines = [
                ("visible GT (filled)",     (255, 255, 255),   "filled"),
                ("occluded GT (hollow)",    (255, 255, 255),   "hollow"),
            ]
            if pred_vis_np is not None:
                lines.extend([
                    ("pred = GT (green)",       (0, 200, 0),   "ring"),
                    ("false positive (red)",    (0, 0, 255),   "ring"),
                    ("false negative (orange)", (0, 140, 255), "ring"),
                ])
            x0, y0 = img_bgr.shape[1] - 210, 10
            pad = 6
            h = 18 * len(lines) + pad * 2
            overlay = img_bgr.copy()
            cv2.rectangle(
                overlay, (x0 - pad, y0 - pad), (img_bgr.shape[1] - 2, y0 + h),
                (0, 0, 0), -1,
            )
            cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0.0, img_bgr)
            for i, (label, col, kind) in enumerate(lines):
                yy = y0 + 12 + 18 * i
                cx = x0 + 8
                if kind == "filled":
                    cv2.circle(img_bgr, (cx, yy - 4), 4, col, -1, cv2.LINE_AA)
                elif kind == "hollow":
                    cv2.circle(img_bgr, (cx, yy - 4), 4, col, 1, cv2.LINE_AA)
                else:
                    cv2.circle(img_bgr, (cx, yy - 4), 5, col, 1, cv2.LINE_AA)
                cv2.putText(
                    img_bgr, label, (cx + 12, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA,
                )

        for t in range(T):
            img = (
                (result.frames[t].permute(1, 2, 0).cpu().numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            canvas = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            t0 = max(0, t - trail_length)

            # -- Trails: solid for visible-to-visible, dashed for boundaries --
            for qi in range(Q):
                pts_win = trk[t0 : t + 1, qi]           # [L, 2]
                v_win = vis[t0 : t + 1, qi]             # [L]
                if pts_win.shape[0] < 2:
                    continue
                color = tuple(int(c) for c in colors_bgr[qi])
                faded = tuple(int(0.5 * c + 0.5 * 90) for c in color)

                # iterate segment-by-segment
                for k in range(pts_win.shape[0] - 1):
                    p1 = pts_win[k]
                    p2 = pts_win[k + 1]
                    v1, v2 = bool(v_win[k]), bool(v_win[k + 1])
                    if v1 and v2:
                        cv2.line(
                            canvas, (int(p1[0]), int(p1[1])),
                            (int(p2[0]), int(p2[1])), color, 1, cv2.LINE_AA,
                        )
                    else:
                        # at least one endpoint occluded → dashed faint line
                        _dashed_polyline(
                            canvas, np.stack([p1, p2], axis=0), faded,
                            thickness=1, dash=3, gap=3,
                        )

            # -- Current-frame markers ---------------------------------
            for qi in range(Q):
                center = (int(trk[t, qi, 0]), int(trk[t, qi, 1]))
                color = tuple(int(c) for c in colors_bgr[qi])
                is_vis = bool(vis[t, qi])

                if is_vis:
                    # Filled visible marker
                    cv2.circle(canvas, center, 3, color, -1, cv2.LINE_AA)
                    cv2.circle(canvas, center, 3, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    # Hollow occluded marker: semi-transparent fill
                    # + coloured outline so the user still sees *where*
                    # the point should be while it is occluded.
                    overlay = canvas.copy()
                    cv2.circle(overlay, center, 3, color, -1, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0.0, canvas)
                    cv2.circle(canvas, center, 4, color, 1, cv2.LINE_AA)

                # Optional prediction overlay ring: outer confusion ring.
                if pred_vis_np is not None:
                    pred = bool(pred_vis_np[t, qi])
                    if pred == is_vis:
                        ring_bgr = (0, 200, 0)       # green — correct
                    elif pred and not is_vis:
                        ring_bgr = (0, 0, 255)       # red — FP
                    else:
                        ring_bgr = (0, 140, 255)     # orange — FN
                    cv2.circle(canvas, center, 6, ring_bgr, 1, cv2.LINE_AA)

                # Optional predicted position marker (small white diamond).
                if pred_trk_np is not None:
                    px, py = pred_trk_np[t, qi, 0], pred_trk_np[t, qi, 1]
                    pts = np.array([
                        [px, py - 3], [px + 3, py], [px, py + 3], [px - 3, py],
                    ], dtype=np.int32)
                    cv2.polylines(
                        canvas, [pts], True, (255, 255, 255), 1, cv2.LINE_AA,
                    )

            if draw_legend:
                _draw_legend(canvas)

            if has_zbuf:
                overlay_rgb = zbuf_np[t]
                overlay_bgr = overlay_rgb[:, :, ::-1].copy()
                blended = cv2.addWeighted(
                    canvas, 1.0, overlay_bgr, alpha, 0.0
                )
                cv2.putText(
                    blended, "zbuf", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
                )
                frame_out = np.concatenate([canvas, blended], axis=1)
            else:
                frame_out = canvas

            writer.write(frame_out)

            if has_rr:
                rr.set_time_sequence("frame", t)
                rr.log(
                    "video/annotated",
                    rr.Image(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)),
                )
                if has_zbuf:
                    rr.log("video/zbuf_debug", rr.Image(overlay_rgb))

        writer.release()

        if _has_ffmpeg:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_path,
                 "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", "-crf", "23", path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                check=True,
            )
            os.remove(tmp_path)


# ── high-level TWIST entry point ─────────────────────────────────────────────
#
# One clean function the rest of the codebase calls to turn a single RGB image
# into a synthetic clip with dense pseudo-GT tracks. It stitches together the
# two heavy objects — the MoGe geometry pipeline and the PseudoGTGenerator —
# behind a process-wide cache so repeated calls at the same resolution neither
# reload the ~300 MB MoGe checkpoint nor rebuild the projection buffers.

# device-keyed caches: (H, W, device) -> PseudoGTGenerator ; and
# (model_name, device, H, W) -> GeometryPipeline. Never reload MoGe per call.
_PSEUDO_GEN_CACHE: dict = {}
_GEOMETRY_PIPELINE_CACHE: dict = {}


def clear_pseudo_track_caches() -> None:
    """Drop the cached :class:`PseudoGTGenerator` / ``GeometryPipeline`` objects.

    Frees the MoGe checkpoint and projection buffers held by
    :func:`generate_pseudo_tracks`. Handy in notebooks / after switching device.
    """
    _PSEUDO_GEN_CACHE.clear()
    _GEOMETRY_PIPELINE_CACHE.clear()


def _get_pseudo_generator(H: int, W: int, device: str) -> "PseudoGTGenerator":
    key = (int(H), int(W), str(device))
    gen = _PSEUDO_GEN_CACHE.get(key)
    if gen is None:
        gen = PseudoGTGenerator(H, W, device=str(device))
        _PSEUDO_GEN_CACHE[key] = gen
    return gen


def _get_geometry_pipeline(model_name: str, device: str, H: int, W: int):
    key = (str(model_name), str(device), int(H), int(W))
    pipe = _GEOMETRY_PIPELINE_CACHE.get(key)
    if pipe is None:
        # Lazy import: only the MoGe path touches ``geometry.pipeline`` (and
        # therefore ``moge``), so the external-depth path stays importable and
        # runnable on machines without ``moge`` installed (e.g. the CPU login node).
        from geometry.pipeline import GeometryPipeline

        pipe = GeometryPipeline(
            geometry_model_name=model_name,
            height=H,
            width=W,
            device=str(device),
            # Pseudo-GT needs METRIC depth — never min-max normalised.
            return_normalized_depth=False,
        )
        _GEOMETRY_PIPELINE_CACHE[key] = pipe
    return pipe


def _as_single_image(image: torch.Tensor) -> torch.Tensor:
    """Coerce ``image`` to ``[1, 3, H, W]`` float in ``[0, 1]`` (single clip)."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            f"image must be [3, H, W] or [1, 3, H, W]; got shape {tuple(image.shape)}"
        )
    if image.shape[0] != 1:
        raise ValueError(
            "generate_pseudo_tracks generates ONE clip from ONE source image; "
            f"got a batch of {image.shape[0]}. Call it per image and stack the "
            "results (the generator is inherently per-image)."
        )
    if not torch.is_floating_point(image):
        image = image.float() / 255.0
    return image


@torch.no_grad()
def generate_pseudo_tracks(
    image: torch.Tensor,
    n_frames: int = 24,
    grid_size: int = 32,
    seed: Optional[int] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
    depth_source: str = "moge",
    depth: Optional[torch.Tensor] = None,
    intrinsics: Optional[torch.Tensor] = None,
    moge_model_name: str = "Ruicheng/moge-2-vits-normal",
    trajectory: Optional[TrajectoryConfig] = None,
    deformation: Optional[DeformationConfig] = None,
    occluders: Optional[OccluderConfig] = None,
    grid_margin_frac: float = 0.03,
    **generate_kwargs: Any,
) -> dict:
    """Turn a single RGB image into a synthetic clip with dense pseudo-GT tracks.

    Runs monocular geometry estimation (MoGe) once, then novel-view point-cloud
    warping with a random camera trajectory + optional scene deformation to
    produce a video whose dense 2-D point correspondences and visibility masks
    are known by construction.

    Parameters
    ----------
    image : Tensor ``[3, H, W]`` or ``[1, 3, H, W]``
        Source RGB image. Float tensors are assumed already in ``[0, 1]``;
        integer tensors are divided by 255.
    n_frames : int
        Temporal length ``T`` of the generated clip.
    grid_size : int
        Dense query grid is ``grid_size × grid_size`` (``Q = grid_size²``).
    seed : int, optional
        Seeds deformation/occluder RNG (the camera path still uses fresh entropy
        each call unless ``randomize_trajectory=False`` is forwarded).
    device : str or torch.device, optional
        Where to build the pipeline and return tensors. Defaults to the image's
        device, falling back to ``"cuda"`` when available else ``"cpu"``.
    depth_source : {"moge", "external"}
        ``"moge"`` (default) estimates depth + intrinsics with
        :class:`geometry.pipeline.GeometryPipeline`. ``"external"`` skips it and
        uses the caller-supplied ``depth`` (**metric**, never normalised) and
        ``intrinsics`` — use this when depth/K already exist upstream.
    depth : Tensor, optional
        Required for ``depth_source="external"``: metric depth ``[H, W]``,
        ``[1, H, W]`` or ``[1, 1, H, W]``.
    intrinsics : Tensor, optional
        Required for ``depth_source="external"``: camera matrix ``[3, 3]`` or
        ``[1, 3, 3]``.
    moge_model_name : str
        HF id of the MoGe checkpoint (``depth_source="moge"`` only).
    trajectory, deformation, occluders : config dataclasses, optional
        Passed through to :meth:`PseudoGTGenerator.generate`. ``trajectory``'s
        ``n_frames`` is always overridden by the ``n_frames`` argument here so
        the clip length is authoritative.
    grid_margin_frac : float
        Query-grid margin from the image edge (fraction of the image dimension).
    **generate_kwargs
        Forwarded verbatim to :meth:`PseudoGTGenerator.generate` (e.g.
        ``randomize_trajectory=False`` for fully reproducible camera paths, or
        the ``visibility_*`` tolerances).

    Returns
    -------
    dict of tensors (all on ``device``) following the TWIST canonical tracking
    item layout (see ``dataset/__init__.py``):

    ==============  ====================  ==========================================
    key             shape                 meaning
    ==============  ====================  ==========================================
    ``frames``      ``[T, 3, H, W]``      float ``[0, 1]`` novel-view RGB
    ``tracks``      ``[T, Q, 2]``         float pixel ``(x, y)`` per frame
    ``visibility``  ``[T, Q]``            bool GT visibility
    ``queries``     ``[Q, 3]``            float ``(t=0, x, y)`` query at frame 0
    ``query_pixels````[Q, 2]``            float source query ``(x, y)``
    ``frame_size``  ``[2]``               long ``(H, W)``
    ``intrinsics``  ``[1, 3, 3]``         camera matrix used for warping
    ``depth``       ``[1, 1, H, W]``      metric source depth
    ``frame_valid`` ``[T, 1, H, W]``      float novel-view validity (1=reliable)
    ``result``      ``PseudoGTResult``    the full raw result (3-D clouds, poses,
                                          zbuf debug) for viz / advanced use
    ==============  ====================  ==========================================
    """
    if depth_source not in ("moge", "external"):
        raise ValueError(
            f"depth_source must be 'moge' or 'external', got {depth_source!r}"
        )

    image = _as_single_image(image)
    if device is None:
        device = image.device if image.is_cuda else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    device = torch.device(device)
    dev_str = str(device)

    H, W = int(image.shape[2]), int(image.shape[3])
    image = image.to(device)

    # ── geometry: depth + intrinsics ────────────────────────────────────────
    if depth_source == "moge":
        pipe = _get_geometry_pipeline(moge_model_name, dev_str, H, W)
        # Strict tensor contract: metric depth, never min-max normalised.
        depth_t, _normals, K = pipe.compute_geometry(image, return_normalized=False)
        depth_t = depth_t.to(device)          # [1, 1, H, W]
        K = K.to(device)                      # [1, 3, 3]
    else:
        if depth is None or intrinsics is None:
            raise ValueError(
                "depth_source='external' requires both `depth` and `intrinsics`."
            )
        depth_t = depth
        while depth_t.ndim < 4:
            depth_t = depth_t.unsqueeze(0)    # [H,W]->[1,H,W]->[1,1,H,W]
        if depth_t.shape[-2:] != (H, W):
            raise ValueError(
                f"external depth spatial size {tuple(depth_t.shape[-2:])} != "
                f"image size {(H, W)}"
            )
        depth_t = depth_t.to(device=device, dtype=torch.float32)
        K = intrinsics
        if K.ndim == 2:
            K = K.unsqueeze(0)                # [3,3] -> [1,3,3]
        K = K.to(device=device, dtype=torch.float32)

    # ── generate ─────────────────────────────────────────────────────────────
    gen = _get_pseudo_generator(H, W, dev_str)
    if trajectory is None:
        traj_cfg = TrajectoryConfig(n_frames=int(n_frames))
    else:
        traj_cfg = replace(trajectory, n_frames=int(n_frames))
    grid_cfg = GridConfig(grid_size=int(grid_size), margin_frac=float(grid_margin_frac))

    res = gen.generate(
        image=image,
        depth=depth_t,
        intrinsics=K,
        trajectory=traj_cfg,
        deformation=deformation,
        grid=grid_cfg,
        occluders=occluders,
        seed=seed,
        **generate_kwargs,
    )

    # ── assemble the canonical item dict ──────────────────────────────────────
    query_pixels = res.query_pixels.to(device)              # [Q, 2]
    Q = query_pixels.shape[0]
    queries = torch.cat(
        [torch.zeros(Q, 1, device=device, dtype=query_pixels.dtype), query_pixels],
        dim=1,
    )                                                        # [Q, 3] = (t=0, x, y)

    return {
        "frames": res.frames.to(device),                    # [T, 3, H, W]
        "tracks": res.tracks.to(device),                    # [T, Q, 2]
        "visibility": res.visibility.to(device),            # [T, Q] bool
        "queries": queries,                                 # [Q, 3]
        "query_pixels": query_pixels,                       # [Q, 2]
        "frame_size": torch.tensor([H, W], device=device, dtype=torch.long),
        "intrinsics": K,                                    # [1, 3, 3]
        "depth": depth_t,                                   # [1, 1, H, W] metric
        "frame_valid": res.frame_valid.to(device),          # [T, 1, H, W]
        "result": res,                                      # full PseudoGTResult
    }


def assemble_pseudo_batch(
    clips: List[dict],
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Stack per-clip :func:`generate_pseudo_tracks` outputs into a TWIST batch.

    Turns a list of ``B`` single-clip dicts (each with the same ``T``, ``Q``,
    ``H``, ``W``) into the ``(frames, queries, tgt)`` triple the engine feeds to
    the model — matching what ``Engine._prep`` produces from a real dataloader
    batch, so the training step is source-agnostic.

    Parameters
    ----------
    clips : list of dict
        Each element is a return value of :func:`generate_pseudo_tracks`.
    device : optional
        Device for the assembled tensors; defaults to the first clip's device.

    Returns
    -------
    frames : Tensor ``[B, T, 3, H, W]`` float ``[0, 1]``
    queries : Tensor ``[B, Q, 3]`` float ``(t=0, x, y)``
    tgt : dict with
        ``tracks`` ``[B, T, Q, 2]`` float,
        ``visibility`` ``[B, T, Q]`` bool,
        ``pos_valid`` ``[B, T, Q]`` bool — frames whose coords are supervisable.
        For pseudo-GT the warped 3-D query has a real coordinate wherever it
        projects **inside the frame** (through occlusion), so ``pos_valid`` is the
        in-frame mask — the ``has_occluded_gt=True`` convention the loss expects
        (see ``dataset/base.py`` / ``models/losses.py``).
    """
    if not clips:
        raise ValueError("assemble_pseudo_batch got an empty clip list")
    if device is None:
        device = clips[0]["frames"].device
    device = torch.device(device)

    frames = torch.stack([c["frames"] for c in clips], dim=0).to(device)       # [B,T,3,H,W]
    tracks = torch.stack([c["tracks"] for c in clips], dim=0).to(device).float()  # [B,T,Q,2]
    visibility = torch.stack([c["visibility"] for c in clips], dim=0).to(device)  # [B,T,Q]
    queries = torch.stack([c["queries"] for c in clips], dim=0).to(device).float()  # [B,Q,3]

    H, W = int(frames.shape[-2]), int(frames.shape[-1])
    xs, ys = tracks[..., 0], tracks[..., 1]
    inframe = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)                       # [B,T,Q]

    tgt = {
        "tracks": tracks,
        "visibility": visibility.bool(),
        "pos_valid": inframe,
    }
    return frames, queries, tgt
