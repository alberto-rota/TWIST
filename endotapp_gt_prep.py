#!/usr/bin/env python
"""endotapp_gt_prep.py

Convert the EndoTAPP ground-truth annotations (``labels.json`` + per-frame
PNG files) into the shared ``index.json`` + per-clip ``.npz`` layout used by
TWIST's CoTrackerTracksDataset.

The data in DATA/EndoTAPP provides GT point annotations at a sparse set of
keyframes for a single surgical video sequence.  We produce one clip whose
frames are the annotated PNG images, with tracks and visibility reflecting the
GT labels (None = occluded).

Source::

    DATA/EndoTAPP/
        labels.json      {"<frame_idx>": [[x, y] | null, ...], ...}
        visible_000.png
        visible_030.png
        ...  (one PNG per annotated frame)

Target (shared CoTracker layout, one clip)::

    out_root/
        index.json
        meta.json
        endotapp/
            clip_00000.npz   # tracks (T,N,2), visibility (T,N), queries (N,3),
                             # frames (T,H,W,3)  [T = number of annotated frames]

CPU/IO only -- runs on the login node.

Example::

    python endotapp_gt_prep.py \\
        --data_dir DATA/EndoTAPP \\
        --out_root DATA/EndoTAPP/gt_tracks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


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


def process(args: argparse.Namespace) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow required: pip install Pillow (or uv pip install Pillow)")

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    labels_path = data_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.json not found in {data_dir}")

    with open(labels_path) as f:
        labels = json.load(f)  # {"0": [[x,y], ...], "30": [...], ...}

    # Sort annotated frames by their integer frame index
    annotated_frames = sorted(labels.keys(), key=int)
    T = len(annotated_frames)
    if T == 0:
        raise ValueError("labels.json is empty")

    # Determine N (number of points) from the first annotated frame
    N = len(labels[annotated_frames[0]])
    print(f"[info] {T} annotated frames, {N} points per frame")

    # Load PNG frames for annotated frames
    frame_list = []
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    visibility = np.zeros((T, N), dtype=bool)

    for t, frame_key in enumerate(annotated_frames):
        frame_idx = int(frame_key)
        png_path = data_dir / f"visible_{frame_idx:03d}.png"
        if not png_path.exists():
            raise FileNotFoundError(f"Frame PNG not found: {png_path}")

        img = np.array(Image.open(png_path).convert("RGB"), dtype=np.uint8)
        frame_list.append(img)

        pts = labels[frame_key]
        for n, pt in enumerate(pts):
            if pt is None:
                visibility[t, n] = False
                # tracks[t, n] stays at 0 (placeholder for occluded)
            else:
                visibility[t, n] = True
                tracks[t, n, 0] = float(pt[0])  # x (pixel)
                tracks[t, n, 1] = float(pt[1])  # y (pixel)

    frames = np.stack(frame_list, axis=0)  # (T, H, W, 3)
    print(f"[info] frames shape: {frames.shape}")

    # queries: (N, 3) = (t_first_vis, x, y) -- reader recomputes at runtime anyway
    queries = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        vis_frames = np.where(visibility[:, n])[0]
        t0 = int(vis_frames[0]) if len(vis_frames) > 0 else 0
        queries[n] = [t0, tracks[t0, n, 0], tracks[t0, n, 1]]

    vid_id = "endotapp"
    clip_path = out_root / vid_id / "clip_00000.npz"
    npz = dict(
        tracks=tracks,
        visibility=visibility,
        queries=queries,
    )
    if args.save_frames:
        npz["frames"] = frames

    _atomic_savez(clip_path, compress=args.compress, **npz)

    index = [dict(
        video=vid_id, clip_idx=0,
        path=f"{vid_id}/clip_00000.npz",
        num_frames=T, num_points=N,
    )]
    _atomic_write_json(out_root / "index.json", index)
    _atomic_write_json(out_root / "meta.json", dict(
        data_dir=str(data_dir),
        annotated_frames=annotated_frames,
        num_frames=T,
        num_points=N,
        save_frames=args.save_frames,
        compress=args.compress,
    ))
    print(f"[done] 1 clip ({T} frames, {N} points) -> {out_root}/index.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="DATA/EndoTAPP",
                   help="folder containing labels.json and visible_*.png files")
    p.add_argument("--out_root", default="DATA/EndoTAPP/gt_tracks",
                   help="output root folder")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false")
    p.add_argument("--no_compress", dest="compress", action="store_false")
    p.set_defaults(save_frames=True, compress=True)
    args = p.parse_args(argv)
    process(args)


if __name__ == "__main__":
    sys.exit(main())
