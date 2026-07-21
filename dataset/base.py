#!/usr/bin/env python
"""Shared, config-driven base for every TWIST tracking dataset.

All readers expose the *same* read-time API -- the full sampling / geometry /
query parameter set that the Kubric reader pioneered -- so a single YAML config
surface drives them uniformly. The only thing a concrete reader implements is
how to enumerate clips (``self.index``) and how to load one raw clip
(:meth:`BaseTracksDataset._load_raw_clip`); everything downstream (point
sub-sampling, query construction, crop, resize, offscreen masking, dtype
finalisation) lives here and is therefore identical across datasets.

Returned item -- the canonical TWIST tracking dict::

    frames      (T, 3, H, W)  uint8 | float[0,1]   (only if ``load_frames``)
    tracks      (T, N, 2)     float32  pixel coords (x, y)
    visibility  (T, N)        bool     per-point visibility
    pos_valid   (T, N)        bool     frames whose COORDS are supervisable: visible,
                                       or (``has_occluded_gt``) occluded-but-in-frame.
                                       The loss masks its position terms with this, so
                                       synthetic sets with full GT (Kubric/PO/DynRep)
                                       supervise tracking *through* occlusion.
    queries     (N, 3)        float32  (t, x, y) at the clip's query frame
    frame_size  (2,)          long     final (H, W) of the (cropped/resized) clip
    video       str                    sequence / video id
    clip_idx    int                    index of this clip within the sequence
    depths      (T, H, W)     float32  (only if available and ``load_depths``)

A concrete subclass returns from :meth:`_load_raw_clip` a *point-major* bundle
(matching the on-disk storage convention and :mod:`dataset.sampling`)::

    tracks_nt   (N, T, 2)  float  pixel tracks for the clip window
    vis_nt      (N, T)     bool   per-point visibility for the clip window
    frames      (T, H, W, 3) uint8 numpy | None   native-resolution RGB
    depths      (T, H, W)    float numpy | None    native-resolution depth
    native_hw   (H, W) | None      stored frame size (for the in-frame test)
    q           int                query frame index *within the clip window*
    video       str
    clip_idx    int
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dataset.sampling import select_point_indices

# Seed stride between epochs when ``resample_points_per_epoch`` is on. A large
# prime so consecutive epochs draw well-decorrelated point subsets. Only the
# seeded ``random`` sampler consumes it; the deterministic modes ("even"/"grid"/
# "first") ignore the seed and are unaffected by resampling.
_EPOCH_SEED_STRIDE = 1_000_003


class BaseTracksDataset(torch.utils.data.Dataset):
    """Base class owning the unified, config-driven tracking-clip pipeline.

    Parameters (all optional; defaults reproduce the historical Kubric reader):

    include, exclude : sequence of str | None
        Restrict to / drop these sequence (video) ids. Realises the train/val
        split when driven by :func:`utilities.config.create_datasets_from_config`.
    max_sequences : int | None
        Keep only the first this-many sequences (after include/exclude).
    clip_len : int | None
        Output frames per clip (after temporal subsampling). ``None`` -> the
        whole (subsampled) sequence/stored-clip is one clip.
    frame_stride : int
        Keep every ``frame_stride``-th frame (temporal subsample).
    clip_stride : int | None
        Raw-frame step between consecutive clip starts (``None`` -> the clip's
        own raw span, i.e. non-overlapping). Smaller -> overlapping clips.
    max_clips_per_video : int | None
        Cap clips taken from each sequence.
    max_clips : int | None
        Cap the *total* number of clips this reader yields (applied after
        ``max_clips_per_video`` / ``max_sequences``). ``None`` -> no cap.
    max_points : int | None
        Number of tracked points ``N`` to sample per clip. ``None`` -> keep all.
    point_sample_mode : str
        ``"even"`` | ``"random"`` | ``"grid"`` | ``"first"`` (see
        :mod:`dataset.sampling`).
    query_frame : int
        Frame index *within the clip* used to define / filter query points.
    require_visible_at_query, min_visible_frames :
        Candidate filters for point selection (see :func:`candidate_mask`).
    target_size : (int, int) | None
        Resize frames to ``(H, W)`` and scale track coordinates to match.
    resize_mode : str
        ``"cover"`` (default): aspect-preserving resize-to-cover then center-crop
        — the TWIST-internal geometry. ``"stretch"``: anisotropic resize straight
        to ``target_size`` with NO crop — the canonical TAP-Vid protocol (256x256
        squash), needed for numbers comparable to published tables.
    crop : (int, int, int, int) | None
        ``(x0, y0, x1, y1)`` pixel box applied *before* resize.
    mark_offscreen_invisible : bool
        Mark points outside the final frame not-visible (trails fade at edges).
    has_occluded_gt : bool
        The source stores *valid* GT coordinates on occluded frames (synthetic
        full GT, not a placeholder). Emits them as supervisable in ``pos_valid``.
    load_frames, load_depths, frames_as_float :
        IO toggles. ``frames_as_float`` returns frames in ``[0, 1]``.
    seed : int
        Base seed for ``point_sample_mode="random"`` (combined with the clip
        index so different clips get different -- but reproducible -- samples).
    resample_points_per_epoch : bool
        When ``True``, fold the current epoch (set via :meth:`set_epoch`) into
        the per-clip sampling seed, so each epoch draws a *different* ``N``-point
        subset from the same clip's candidate pool -- effectively a mild data
        augmentation that, over training, exposes the model to far more of the
        stored trajectories than any single ``max_points`` snapshot, at no extra
        per-batch memory. Only meaningful with ``point_sample_mode="random"``
        (the deterministic modes ignore the seed). Default ``False`` reproduces
        the historical fixed-per-clip sampling exactly. The engine advances only
        the *train* readers, so validation metrics stay on a fixed point set.
    """

    def __init__(
        self,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        max_sequences: Optional[int] = None,
        clip_len: Optional[int] = None,
        frame_stride: int = 1,
        clip_stride: Optional[int] = None,
        max_clips_per_video: Optional[int] = None,
        max_clips: Optional[int] = None,
        max_points: Optional[int] = None,
        point_sample_mode: str = "even",
        query_frame: int = 0,
        require_visible_at_query: bool = True,
        min_visible_frames: int = 1,
        target_size: Optional[Tuple[int, int]] = None,
        resize_mode: str = "cover",
        crop: Optional[Tuple[int, int, int, int]] = None,
        mark_offscreen_invisible: bool = True,
        has_occluded_gt: bool = False,
        load_frames: bool = True,
        load_depths: bool = False,
        frames_as_float: bool = False,
        seed: int = 0,
        subsample: Optional[float] = None,
        resample_points_per_epoch: bool = False,
    ):
        self.include = list(include) if include is not None else None
        self.exclude = list(exclude) if exclude is not None else None
        self.max_sequences = max_sequences
        self.frame_stride = int(frame_stride)
        self.clip_len = clip_len
        self.clip_stride = clip_stride
        self.max_clips_per_video = max_clips_per_video
        self.max_clips = max_clips
        self.max_points = max_points
        self.point_sample_mode = point_sample_mode
        self.query_frame = int(query_frame)
        self.require_visible_at_query = require_visible_at_query
        self.min_visible_frames = int(min_visible_frames)
        self.target_size = tuple(target_size) if target_size is not None else None
        if resize_mode not in ("cover", "stretch"):
            raise ValueError(f"resize_mode must be 'cover' or 'stretch', got {resize_mode!r}")
        self.resize_mode = resize_mode
        self.mark_offscreen_invisible = mark_offscreen_invisible
        self.has_occluded_gt = bool(has_occluded_gt)
        self.load_frames = load_frames
        self.load_depths = load_depths
        self.frames_as_float = frames_as_float
        self.seed = int(seed)
        # Deterministic clip-index subsampling applied after the index is built
        # (see the subclass ``_build_index``). ``None``/``>=1`` keeps everything;
        # ``0 < f < 1`` keeps an evenly-spread fraction; ``<= 0`` keeps none.
        # Used for a small, *representative* per-epoch val set (VAL_SUBSAMPLE),
        # decoupled from VAL_FRACTION (the train/val split ratio).
        self.subsample = subsample

        # Per-epoch point resampling (train-only augmentation; see the class
        # docstring). The epoch lives in a shared-memory int so that set_epoch()
        # -- called once per epoch in the MAIN process -- is visible inside the
        # DataLoader's *persistent* worker processes: those are forked ONCE (at
        # first iteration) and would never see a plain-attribute update. Fork
        # start method assumed (Linux/SLURM default; verified for this repo).
        # Allocated only when the feature is on, so default readers are unchanged
        # (the seed stays exactly ``self.seed + i`` -- see ``_sample_seed``).
        self.resample_points_per_epoch = bool(resample_points_per_epoch)
        self._epoch = mp.Value("i", 0, lock=False) if self.resample_points_per_epoch else None

        if crop is not None:
            x0, y0, x1, y1 = (int(v) for v in crop)
            if not (x1 > x0 and y1 > y0):
                raise ValueError(f"crop must be (x0,y0,x1,y1) with x1>x0, y1>y0, got {crop}")
            crop = (x0, y0, x1, y1)
        self.crop = crop

        # Subclasses MUST populate ``self.index`` (the per-clip enumeration).
        self.index: list = getattr(self, "index", [])

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    def _load_raw_clip(self, i: int) -> dict:
        """Return the raw, point-major clip bundle for index entry ``i``.

        See the module docstring for the expected keys. Subclass responsibility.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.index)

    # ------------------------------------------------------------------ #
    # Per-epoch point resampling (train-only; no-op unless enabled)
    # ------------------------------------------------------------------ #
    def set_epoch(self, epoch: int) -> None:
        """Set the current training epoch for per-epoch point resampling.

        No-op unless ``resample_points_per_epoch`` is on. The engine calls this
        on every *train* reader at the start of each epoch; validation readers
        are never advanced, so val metrics are scored on a fixed point set.
        Writes a shared-memory int so the new epoch reaches already-forked
        persistent DataLoader workers.
        """
        if self._epoch is not None:
            self._epoch.value = int(epoch)

    def _sample_seed(self, i: int) -> int:
        """Per-clip RNG seed for point sampling.

        Adds an epoch term ONLY when per-epoch resampling is on, so each epoch
        draws a different subset in the ``random`` sampler; with the feature off
        this is exactly ``self.seed + i`` (historical behaviour). Deterministic
        sample modes ignore the seed and are unaffected either way.
        """
        epoch = int(self._epoch.value) if self._epoch is not None else 0
        return self.seed + i + epoch * _EPOCH_SEED_STRIDE

    # ------------------------------------------------------------------ #
    # Geometry: crop -> resize (coords transformed to match)
    # ------------------------------------------------------------------ #
    def _apply_crop(self, frames, depths, tracks, queries):
        # frames (T,H,W,3) np|None, depths (T,H,W) np|None, tracks (T,N,2), queries (N,3)
        x0, y0, x1, y1 = self.crop
        offset = tracks.new_tensor([x0, y0])                 # (2,)
        tracks = tracks - offset                              # (T, N, 2)
        queries = queries.clone()
        queries[:, 1:] = queries[:, 1:] - offset             # (N, 3): (t, x, y)
        if frames is not None:
            frames = frames[:, y0:y1, x0:x1, :]              # (T, h, w, 3)
        if depths is not None:
            depths = depths[:, y0:y1, x0:x1]                 # (T, h, w)
        return frames, depths, tracks, queries

    def _apply_resize(self, frames, depths, tracks, queries, cur_hw):
        """Resize to ``target_size`` in one of two geometries.

        ``"cover"`` (default): aspect-preserving resize-to-cover, **then**
        center-crop. Uniform-scales (single factor, so no aspect distortion)
        until the frame *covers* ``target_size``, then center-crops to exactly
        ``target_size``; coordinates follow the same scale + crop offset. (An
        older version anisotropically squashed straight to the target, distorting
        any non-square source — the aspect-ratio warping once seen on
        PointOdyssey / DAVIS clips.)

        ``"stretch"``: anisotropic resize straight to ``target_size`` with NO
        crop — the canonical TAP-Vid benchmark geometry (e.g. DAVIS 854x480 ->
        256x256 squash). Nothing leaves the frame, so numbers are comparable to
        published TAP-Vid tables; use for benchmark-table evals only.
        """
        th, tw = self.target_size
        ch, cw = cur_hw
        # Fast path: the source is already exactly the target size (e.g. Kubric is
        # stored at 512² and trains at 512²). Both geometries reduce to identity
        # there (scale 1, no crop), so skip the ~300 ms CPU bilinear F.interpolate
        # over (T,3,H,W) and the float<->uint8 round-trip — pure wasted work.
        if (ch, cw) == (th, tw):
            return frames, depths, tracks, queries, (th, tw)
        if self.resize_mode == "stretch":
            rh, rw = th, tw                                  # direct anisotropic resize
            oy = ox = 0                                      # no crop
        else:                                                # "cover"
            s = max(th / ch, tw / cw)                        # uniform cover scale
            rh, rw = max(int(round(ch * s)), th), max(int(round(cw * s)), tw)  # resized size
            oy, ox = (rh - th) // 2, (rw - tw) // 2          # center-crop offsets
        # use the *actual* per-axis ratio (rounding rh/rw makes it differ from s
        # by <1px) so coords stay registered to the resized pixels, then offset.
        scale = tracks.new_tensor([rw / cw, rh / ch])        # (2,)
        off = tracks.new_tensor([ox, oy])                    # (2,)
        tracks = tracks * scale - off                         # (T, N, 2)
        queries = queries.clone()
        queries[:, 1:] = queries[:, 1:] * scale - off        # (N, 3)
        if frames is not None:  # (T, H, W, 3) np uint8 -> resize-to-cover -> crop
            f = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()  # (T,3,H,W)
            f = F.interpolate(f, size=(rh, rw), mode="bilinear", align_corners=False)
            f = f[:, :, oy:oy + th, ox:ox + tw]                                             # center crop
            frames = f.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()    # (T,th,tw,3)
        if depths is not None:
            d = torch.from_numpy(np.ascontiguousarray(depths)).unsqueeze(1)                 # (T,1,H,W)
            d = F.interpolate(d, size=(rh, rw), mode="nearest").squeeze(1)                  # (T,rh,rw)
            depths = d[:, oy:oy + th, ox:ox + tw].numpy()                                   # center crop
        return frames, depths, tracks, queries, (th, tw)

    # ------------------------------------------------------------------ #
    # Unified read pipeline
    # ------------------------------------------------------------------ #
    def __getitem__(self, i: int) -> dict:
        raw = self._load_raw_clip(i)
        tracks_nt = np.asarray(raw["tracks_nt"])             # (N, T, 2)
        vis_nt = np.asarray(raw["vis_nt"])                   # (N, T)
        frames = raw.get("frames")                           # (T, H, W, 3) np | None
        depths = raw.get("depths")                           # (T, H, W) np | None
        native_hw = raw.get("native_hw")                     # (H, W) | None
        q = int(raw["q"])                                    # query frame in clip

        # ---- point sub-sampling (visible-at-query candidates, configurable) ----
        sel = select_point_indices(
            tracks_nt, vis_nt, q, native_hw, self.max_points,
            mode=self.point_sample_mode,
            require_visible_at_query=self.require_visible_at_query,
            min_visible_frames=self.min_visible_frames,
            seed=self._sample_seed(i),
        )
        tracks = torch.from_numpy(np.asarray(tracks_nt[sel])).float().permute(1, 0, 2)   # (T, N, 2)
        visibility = torch.from_numpy(np.asarray(vis_nt[sel])).bool().permute(1, 0)      # (T, N)
        # queries: (t within clip, x, y) at the query frame.
        queries = torch.cat(
            [torch.full_like(tracks[q, :, :1], float(q)), tracks[q]], dim=-1
        )  # (N, 3)

        # ---- geometry: (optional manual crop) -> resize-to-cover + center-crop ----
        # The manual ``crop`` selects a native-pixel region (e.g. trimming endoscope
        # borders); the target resize then preserves aspect ratio (resize-to-cover
        # then center-crop) instead of squashing. Coords follow both transforms.
        cur_hw = tuple(native_hw) if native_hw is not None else None
        if self.crop is not None:
            frames, depths, tracks, queries = self._apply_crop(frames, depths, tracks, queries)
            x0, y0, x1, y1 = self.crop
            cur_hw = (y1 - y0, x1 - x0)
        if self.target_size is not None and cur_hw is not None:
            frames, depths, tracks, queries, cur_hw = self._apply_resize(
                frames, depths, tracks, queries, cur_hw
            )

        # ---- offscreen test + position-supervision validity ----
        # ``inframe`` is needed even when offscreen points stay "visible":
        # coordinates outside the frame are never supervisable (the feature
        # sampler border-clamps there) and must not enter the position loss.
        if cur_hw is not None:
            h, w = cur_hw
            inframe = (
                (tracks[..., 0] >= 0) & (tracks[..., 0] < w)
                & (tracks[..., 1] >= 0) & (tracks[..., 1] < h)
            )  # (T, N)
        else:
            inframe = torch.ones_like(visibility)
        # pos_valid: where the COORDS carry usable supervision. Visible-in-frame
        # always does; occluded-but-in-frame does IFF the source stores real GT
        # through occlusion (has_occluded_gt: Kubric/PointOdyssey/DynamicReplica)
        # rather than a (0,0)/pseudo-label placeholder. This is what lets the
        # loss train tracking *through* occlusion — the project's core skill.
        pos_valid = (visibility | self.has_occluded_gt) & inframe
        if self.mark_offscreen_invisible:
            visibility = visibility & inframe

        out = dict(
            tracks=tracks,
            visibility=visibility,
            pos_valid=pos_valid,
            queries=queries,
            frame_size=torch.tensor(cur_hw if cur_hw is not None else (-1, -1), dtype=torch.long),
            video=raw["video"],
            clip_idx=raw["clip_idx"],
        )
        if frames is not None:
            frames = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)  # (T,3,H,W)
            out["frames"] = frames.float() / 255.0 if self.frames_as_float else frames
        if depths is not None:
            out["depths"] = torch.from_numpy(np.ascontiguousarray(depths))               # (T, H, W)
        return out
