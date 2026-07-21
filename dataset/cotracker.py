#!/usr/bin/env python
"""Unified reader for the shared CoTracker on-disk layout (``index.json`` + per-clip ``.npz``).

Every surgical dataset (Cholec80, EndoTAPP, SurgT) and PointOdyssey is
pre-tracked offline by the ``*_data_prep.py`` scripts into the *same* layout::

    root/
        index.json                 # global manifest, one entry per stored clip
        meta.json                  # the prep configuration (resize, ...)
        <video-id>/                # flat (cholec80) or nested (case_2/1/left)
            clip_00000.npz         # tracks (T,N,2), visibility (T,N), queries (N,3), [frames (T,H,W,3)]
            ...

This reader exposes the **same config-driven API** as the Kubric reader via
:class:`dataset.base.BaseTracksDataset`. The pre-tracked grids are pre-computed
*observations*; everything else (point sub-sampling, temporal sub-windowing of a
stored clip, crop, resize to the model's square input, query selection,
offscreen masking, IO toggles) happens here at read time, so a single YAML
surface configures Kubric and the surgical datasets identically.

Item layout is the canonical TWIST tracking dict (see :class:`dataset.base.BaseTracksDataset`).
The stored clips carry no depth, so ``load_depths`` is a no-op here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from dataset.base import BaseTracksDataset


def _read_index(root: Path) -> List[dict]:
    with open(root / "index.json") as f:
        return json.load(f)


def list_sequences(root: str) -> List[str]:
    """Unique ``video`` ids present in ``root/index.json`` (sorted).

    Used by :func:`utilities.config.create_datasets_from_config` to realise the
    sequence-level train/val split. Returns ``[]`` when the dataset is absent.
    """
    root = Path(root)
    if not (root / "index.json").exists():
        return []
    seen = dict.fromkeys(e["video"] for e in _read_index(root))
    return sorted(seen)


class CoTrackerTracksDataset(BaseTracksDataset):
    """Reads the shared ``index.json`` + per-clip ``.npz`` layout.

    Serves flat ids (Cholec80) and nested ids (EndoTAPP / SurgT ``case_2/1/left``)
    alike. See :class:`dataset.base.BaseTracksDataset` for the full parameter
    documentation. Notable specifics:

    * **temporal** -- ``clip_len`` / ``frame_stride`` / ``clip_stride`` /
      ``max_clips_per_video`` sub-window each *stored* clip at read time.
      ``clip_len=None`` yields the whole (subsampled) stored clip as one
      (variable-length) clip.
    * **subset** -- ``include`` / ``exclude`` filter by ``video`` id;
      ``max_sequences`` keeps the first this-many distinct videos.
    * **geometry** -- ``crop`` then ``target_size`` resize (e.g. to the model's
      256x256 input); coordinates follow. The native frame size is read once
      from ``meta.json`` (``resize``) or a stored frame header.
    """

    def __init__(
        self,
        root: str,
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
        super().__init__(
            include=include,
            exclude=exclude,
            max_sequences=max_sequences,
            clip_len=clip_len,
            frame_stride=frame_stride,
            clip_stride=clip_stride,
            max_clips_per_video=max_clips_per_video,
            max_clips=max_clips,
            max_points=max_points,
            point_sample_mode=point_sample_mode,
            query_frame=query_frame,
            require_visible_at_query=require_visible_at_query,
            min_visible_frames=min_visible_frames,
            target_size=target_size,
            resize_mode=resize_mode,
            crop=crop,
            mark_offscreen_invisible=mark_offscreen_invisible,
            has_occluded_gt=has_occluded_gt,
            load_frames=load_frames,
            load_depths=load_depths,
            frames_as_float=frames_as_float,
            seed=seed,
            subsample=subsample,
            resample_points_per_epoch=resample_points_per_epoch,
        )
        self.root = Path(root)
        self._raw_index = _read_index(self.root)
        self.frame_hw = self._infer_frame_hw()   # stored (H, W)
        self.index = self._build_index()

    # ------------------------------------------------------------------ #
    # Frame geometry
    # ------------------------------------------------------------------ #
    def _infer_frame_hw(self) -> Optional[Tuple[int, int]]:
        """Stored frame ``(H, W)`` from ``meta.json`` (``resize``), else a frame header."""
        meta_path = self.root / "meta.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    resize = json.load(f).get("resize")
                if resize:
                    return (int(resize[0]), int(resize[1]))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if self._raw_index:
            npz = self.root / self._raw_index[0]["path"]
            with np.load(npz) as d:
                if "frames" in d.files:
                    return tuple(int(s) for s in d["frames"].shape[1:3])  # (T,H,W,3)
        return None

    # ------------------------------------------------------------------ #
    # Index: filter videos + temporal sub-windowing of each stored clip
    # ------------------------------------------------------------------ #
    def _build_index(self) -> List[dict]:
        inc = set(self.include) if self.include is not None else None
        exc = set(self.exclude) if self.exclude is not None else None

        entries: List[dict] = []
        per_video: dict = {}
        for e in self._raw_index:
            vid = e["video"]
            if inc is not None and vid not in inc:
                continue
            if exc is not None and vid in exc:
                continue

            t_stored = int(e["num_frames"])
            if self.clip_len is None:
                clip_t = len(range(0, t_stored, self.frame_stride))
                starts = [0]
            else:
                clip_t = int(self.clip_len)
                raw_span = (clip_t - 1) * self.frame_stride + 1
                if raw_span > t_stored:
                    continue  # stored clip too short for even one output clip
                step = self.clip_stride if self.clip_stride is not None else raw_span
                step = max(int(step), 1)
                starts = list(range(0, t_stored - raw_span + 1, step))

            for start in starts:
                if (self.max_clips_per_video is not None
                        and per_video.get(vid, 0) >= int(self.max_clips_per_video)):
                    break
                ci = per_video.get(vid, 0)
                ce = dict(video=vid, clip_idx=ci, path=e["path"],
                          start=int(start), clip_len=int(clip_t))
                if e.get("num_points") is not None:  # lets collate detect fixed N
                    ce["num_points"] = int(e["num_points"])
                entries.append(ce)
                per_video[vid] = ci + 1

        if self.max_sequences is not None:
            keep = list(dict.fromkeys(en["video"] for en in entries))[: int(self.max_sequences)]
            keepset = set(keep)
            entries = [en for en in entries if en["video"] in keepset]
        if self.max_clips is not None:
            entries = entries[: int(self.max_clips)]  # total-clip cap (first videos, in order)
        # Deterministic subsampling to a fraction of the clips, evenly spread over
        # the whole index (so it stays representative across videos -- unlike
        # max_clips, which takes the first videos). Used for a small per-epoch val
        # set via VAL_SUBSAMPLE; f<=0 empties it, f>=1 (or None) keeps everything.
        if self.subsample is not None:
            f = float(self.subsample)
            if f <= 0:
                entries = []
            elif f < 1.0 and len(entries) > 1:
                k = max(1, int(round(len(entries) * f)))
                if k < len(entries):
                    sel = np.linspace(0, len(entries) - 1, num=k).round().astype(int)
                    sel = sorted({int(x) for x in sel.tolist()})
                    entries = [entries[x] for x in sel]
        return entries

    # ------------------------------------------------------------------ #
    # Raw clip loading (shared pipeline does sampling / geometry / finalise)
    # ------------------------------------------------------------------ #
    def _load_raw_clip(self, i: int) -> dict:
        entry = self.index[i]
        clip_t, start = entry["clip_len"], entry["start"]
        frame_ids = start + np.arange(clip_t) * self.frame_stride          # (T,)

        # Two on-disk variants of the same clip, chosen per read:
        #   * mmap  — a sibling ``<clip>/`` dir of UNCOMPRESSED per-array ``.npy``
        #     (frames/tracks/visibility). ``np.load(mmap_mode='r')[frame_ids]``
        #     touches only the pages for the requested frames, so a 24-frame window
        #     of a 120-frame stored clip reads ~24/120 of the bytes with NO DEFLATE
        #     decompression — ~80x faster per item than the compressed path, which
        #     is what let dataloading keep the GPU fed (see assets/dataprep/
        #     transcode_to_mmap.py, which produces this layout from the .npz files).
        #   * npz   — the legacy compressed ``.npz`` (whole arrays decompressed).
        # The mmap dir is preferred when present; otherwise we fall back to the
        # ``.npz`` so un-transcoded datasets keep working unchanged.
        npz_path = self.root / entry["path"]
        mmap_dir = npz_path.with_suffix("")                                # 0000/clip_00000/
        if (mmap_dir / "tracks.npy").exists():
            tracks = np.ascontiguousarray(
                np.load(mmap_dir / "tracks.npy", mmap_mode="r")[frame_ids])       # (T, N, 2)
            visibility = np.ascontiguousarray(
                np.load(mmap_dir / "visibility.npy", mmap_mode="r")[frame_ids])   # (T, N)
            fpath = mmap_dir / "frames.npy"
            frames = (
                np.ascontiguousarray(np.load(fpath, mmap_mode="r")[frame_ids])    # (T, H, W, 3)
                if (self.load_frames and fpath.exists()) else None
            )
        else:
            with np.load(npz_path) as d:
                tracks = np.asarray(d["tracks"])[frame_ids]                # (T, N, 2)
                visibility = np.asarray(d["visibility"])[frame_ids]        # (T, N)
                frames = (
                    np.asarray(d["frames"])[frame_ids]                     # (T, H, W, 3)
                    if (self.load_frames and "frames" in d.files) else None
                )

        tracks_nt = np.ascontiguousarray(tracks.transpose(1, 0, 2))        # (N, T, 2)
        vis_nt = np.ascontiguousarray(visibility.transpose(1, 0))          # (N, T)
        q = min(self.query_frame, clip_t - 1)
        return dict(
            tracks_nt=tracks_nt,
            vis_nt=vis_nt,
            frames=frames,
            depths=None,
            native_hw=self.frame_hw,
            q=q,
            video=entry["video"],
            clip_idx=entry["clip_idx"],
        )


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    import torch

    p = argparse.ArgumentParser(description="Quick sanity check of CoTrackerTracksDataset")
    p.add_argument("--root", default="DATA/cholec80/cotracker_tracks")
    p.add_argument("--clip_len", type=int, default=None)
    p.add_argument("--frame_stride", type=int, default=1)
    p.add_argument("--max_points", type=int, default=None)
    p.add_argument("--mode", default="even")
    p.add_argument("--target_size", type=int, nargs=2, default=None)
    p.add_argument("--max_sequences", type=int, default=4)
    args = p.parse_args()

    ds = CoTrackerTracksDataset(
        args.root,
        clip_len=args.clip_len,
        frame_stride=args.frame_stride,
        max_points=args.max_points,
        point_sample_mode=args.mode,
        target_size=tuple(args.target_size) if args.target_size else None,
        max_sequences=args.max_sequences,
    )
    print(f"[cotracker] {len(ds)} clips, {len(list_sequences(args.root))} videos @ {args.root}")
    item = ds[0]
    for k, v in item.items():
        if torch.is_tensor(v):
            print(f"  {k:11s} {tuple(v.shape)!s:18s} {v.dtype}")
        else:
            print(f"  {k:11s} {v!r}")
