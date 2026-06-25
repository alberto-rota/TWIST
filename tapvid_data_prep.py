#!/usr/bin/env python
"""tapvid_data_prep.py

Convert TAP-Vid evaluation pickles (tapvid-davis, tapvid-rgb-stacking) from
their single-pickle format into the shared ``index.json`` + per-clip ``.npz``
layout used by TWIST's CoTrackerTracksDataset.

No CoTracker tracking is run here -- these datasets ship ground-truth point
tracks. The conversion is a pure format repack:

    Source (TAP-Vid pickle convention):
        points   (N, T, 2)      float32  normalized [0, 1] (x, y) -- x along width
        occluded (N, T)         bool     True = occluded
        video    (T, H, W, 3)   uint8    RGB frames

    Target (shared CoTracker layout):
        tracks     (T, N, 2)    float32  pixel (x, y)
        visibility (T, N)       bool     ~occluded
        frames     (T, H, W, 3) uint8    (optional, --no_save_frames to skip)
        queries    (N, 3)       float32  (t_first_vis, x, y)

Output layout::

    out_root/
        index.json      # global manifest
        meta.json       # run config
        <video-id>/
            clip_00000.npz

CPU/IO only -- runs on the login node.

Examples
--------
tapvid-davis::

    python tapvid_data_prep.py \\
        --pkl  DATA/tapvid_davis/tapvid_davis.pkl \\
        --out_root DATA/tapvid_davis/gt_tracks

tapvid-rgb-stacking::

    python tapvid_data_prep.py \\
        --pkl  DATA/tapvid_rgb_stacking/tapvid_rgb_stacking.pkl \\
        --out_root DATA/tapvid_rgb_stacking/gt_tracks
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Atomic IO helpers (same pattern as the Cholec80 pipeline)
# --------------------------------------------------------------------------- #
def _atomic_savez(path: Path, compress: bool, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    (np.savez_compressed if compress else np.savez)(tmp, **arrays)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# TAP-Vid -> shared layout conversion for one video
# --------------------------------------------------------------------------- #
def _tapvid_to_npz(
    points: np.ndarray,   # (N, T, 2) normalized [0,1] (x, y)
    occluded: np.ndarray, # (N, T) bool
    video: np.ndarray,    # (T, H, W, 3) uint8
    save_frames: bool,
) -> dict:
    """Convert one TAP-Vid entry to the shared npz dict."""
    T, H, W = video.shape[:3]
    N = points.shape[0]

    # (N, T, 2) normalized -> (T, N, 2) pixel coords
    tracks_nt = points.astype(np.float32)  # (N, T, 2)
    tracks_nt[:, :, 0] *= W               # x_pixel
    tracks_nt[:, :, 1] *= H               # y_pixel
    tracks = tracks_nt.transpose(1, 0, 2).astype(np.float32)  # (T, N, 2)

    # (N, T) bool occluded -> (T, N) bool visible
    visibility = (~occluded).transpose(1, 0).astype(bool)     # (T, N)

    # queries: (N, 3) = (t_first_visible, x, y) -- reader recomputes at runtime
    # but stored here for completeness / standalone use.
    queries = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        vis_frames = np.where(visibility[:, n])[0]
        t0 = int(vis_frames[0]) if len(vis_frames) > 0 else 0
        queries[n] = [t0, tracks[t0, n, 0], tracks[t0, n, 1]]

    npz = dict(
        tracks=tracks,
        visibility=visibility,
        queries=queries,
    )
    if save_frames:
        npz["frames"] = video.astype(np.uint8)  # (T, H, W, 3)
    return npz


# --------------------------------------------------------------------------- #
# Iterate over entries (dict-of-videos for davis, list for rgb_stacking)
# --------------------------------------------------------------------------- #
def _iter_entries(data) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (video_id, points, occluded, video) for each entry."""
    if isinstance(data, dict):
        for vid_id, entry in data.items():
            yield vid_id, entry["points"], entry["occluded"], entry["video"]
    elif isinstance(data, list):
        for i, entry in enumerate(data):
            vid_id = f"{i:04d}"
            yield vid_id, entry["points"], entry["occluded"], entry["video"]
    else:
        raise ValueError(f"Unknown pickle top-level type: {type(data)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process(args: argparse.Namespace) -> None:
    pkl_path = Path(args.pkl)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    entries = list(_iter_entries(data))
    print(f"[info] {len(entries)} videos -> {out_root}")

    index: List[dict] = []
    for vi, (vid_id, points, occluded, video) in enumerate(entries):
        clip_path = out_root / vid_id / "clip_00000.npz"
        if clip_path.exists() and not args.overwrite:
            print(f"  [{vi+1}/{len(entries)}] {vid_id} -- skip (exists)")
            # still add to index
            with np.load(clip_path) as d:
                num_frames, num_points = d["tracks"].shape[:2]
            index.append(dict(
                video=vid_id, clip_idx=0,
                path=f"{vid_id}/clip_00000.npz",
                num_frames=num_frames, num_points=num_points,
            ))
            continue

        print(f"  [{vi+1}/{len(entries)}] {vid_id}  {video.shape}  N={points.shape[0]}")
        npz = _tapvid_to_npz(points, occluded, video, save_frames=args.save_frames)
        _atomic_savez(clip_path, compress=args.compress, **npz)
        index.append(dict(
            video=vid_id, clip_idx=0,
            path=f"{vid_id}/clip_00000.npz",
            num_frames=int(npz["tracks"].shape[0]),
            num_points=int(npz["tracks"].shape[1]),
        ))

    _atomic_write_json(out_root / "index.json", index)
    _atomic_write_json(out_root / "meta.json", dict(
        source=str(pkl_path),
        save_frames=args.save_frames,
        compress=args.compress,
    ))
    print(f"[done] {len(index)} clips -> {out_root}/index.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", required=True, help="path to the TAP-Vid .pkl file")
    p.add_argument("--out_root", required=True, help="output root folder")
    p.add_argument("--overwrite", action="store_true", help="recompute clips that already exist")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                   help="skip storing frames (saves space; the video stays in the pickle)")
    p.add_argument("--no_compress", dest="compress", action="store_false",
                   help="uncompressed .npz (faster to write, ~3x larger)")
    p.set_defaults(save_frames=True, compress=True)
    args = p.parse_args(argv)
    process(args)


if __name__ == "__main__":
    sys.exit(main())
