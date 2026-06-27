#!/usr/bin/env python
"""surgicalmotion_data_prep.py

Convert the **SurgicalMotion** dataset (per-video annotation pickles + RGB
frames) into the shared ``index.json`` + per-clip ``.npz`` layout that TWIST's
:class:`dataset.cotracker.CoTrackerTracksDataset` reads -- exactly like every
other dataset in this project. These are ground-truth point tracks, so (as with
``tapvid_data_prep.py``) no CoTracker tracking is run here; this is a pure format
repack + uniform spatial resize.

Source layout (as downloaded)::

    DATA/SurgicalMotion/
        Annotation/<seq>.pkl        # one pickle per sequence
        <seq>/color/00000.png ...   # RGB frames (also embedded in the pickle)
        <seq>/mask/00000.png  ...   # segmentation masks (model-unused, ignored)

Each ``<seq>.pkl`` is a dict in the TAP-Vid *training* convention::

    video                    (1, T, H, W, 3) float32 0..255  BGR (!)  frames
    tissue / tools : dict
        query_points         (1, N, 3)    float64  (t, y, x)  at the query frame
        target_points        (1, N, T, 2) float32  (x, y)     pixel, per frame
        occluded             (1, N, T)    bool     True = occluded
        trackgroup           (1, N)       int64    (instance id, unused here)

Two point sets are annotated per sequence: ``tissue`` and ``tools``. ``--points``
selects which to keep (default ``both`` -> all annotated points, concatenated).

Conversion details (the bits that are easy to get wrong):

* **BGR -> RGB.** The embedded ``video`` is BGR (verified bit-identical to the
  RGB ``color/*.png`` after a channel flip); we flip it so stored frames are RGB
  like every other dataset.
* **coords.** ``target_points`` is already pixel ``(x, y)`` -> just transpose to
  ``(T, N, 2)``. ``visibility = ~occluded`` transposed to ``(T, N)``.
* **queries.** Stored ``(N, 3) = (t_first_visible, x, y)`` for completeness; the
  reader recomputes queries at read time, so this is cosmetic.
* **uniform resize.** SurgicalMotion ships two native sizes (640x512 and, for
  the ``case12_*`` sequences, 640x480). The shared reader infers one frame size
  per dataset root, so every clip is resized to a single ``--resize`` (H, W)
  recorded in ``meta.json``; track coordinates are scaled to match.

Target (shared CoTracker layout)::

    out_root/
        index.json                 # global manifest, one entry per sequence
        meta.json                  # this run's configuration
        <seq>/clip_00000.npz       # frames, tracks, visibility, queries

Each ``clip_00000.npz`` contains::

    frames      uint8   (T, H, W, 3)   resized RGB frames
    tracks      float32 (T, N, 2)      pixel (x, y) in the resized frame
    visibility  bool    (T, N)         per-point visibility (~occluded)
    queries     float32 (N, 3)         (t_first_visible, x, y)

CPU/IO only -- runs on the login node.

Example
-------
    python surgicalmotion_data_prep.py \\
        --src_root DATA/SurgicalMotion \\
        --out_root DATA/SurgicalMotion/gt_tracks \\
        --points both --resize 512 640
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Atomic IO helpers (same pattern as the other *_data_prep.py scripts)
# --------------------------------------------------------------------------- #
def _atomic_savez(path: Path, compress: bool, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            (np.savez_compressed if compress else np.savez)(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# --------------------------------------------------------------------------- #
# Geometry: uniform resize (frames + track coordinates together)
# --------------------------------------------------------------------------- #
def _resize_clip(
    frames: np.ndarray,   # (T, H, W, 3) uint8
    tracks: np.ndarray,   # (T, N, 2) float (x, y) pixel
    dst_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize frames to ``dst_hw`` and scale ``(x, y)`` track coords to match."""
    sh, sw = frames.shape[1:3]
    dh, dw = dst_hw
    if (sh, sw) == (dh, dw):
        return frames, tracks
    sx, sy = dw / sw, dh / sh
    tracks = tracks.copy()
    tracks[..., 0] *= sx
    tracks[..., 1] *= sy
    t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()  # (T,3,H,W)
    t = F.interpolate(t, size=(dh, dw), mode="bilinear", align_corners=False)
    frames = t.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()    # (T,dh,dw,3)
    return frames, tracks


# --------------------------------------------------------------------------- #
# SurgicalMotion pickle -> shared npz dict
# --------------------------------------------------------------------------- #
def _surgicalmotion_to_npz(data: dict, which: str) -> dict:
    """Convert one loaded SurgicalMotion pickle to the shared npz dict.

    ``which`` is ``"tissue"``, ``"tools"`` or ``"both"`` (concatenated).
    """
    video = np.asarray(data["video"])[0]                       # (T, H, W, 3) float BGR
    frames = video[..., ::-1]                                  # BGR -> RGB
    frames = np.ascontiguousarray(frames.round().clip(0, 255).astype(np.uint8))
    T = frames.shape[0]

    groups = ["tissue", "tools"] if which == "both" else [which]
    tracks_nt_parts: List[np.ndarray] = []
    vis_nt_parts: List[np.ndarray] = []
    for g in groups:
        grp = data[g]
        tp = np.asarray(grp["target_points"])[0].astype(np.float32)   # (N, T, 2) (x, y)
        occ = np.asarray(grp["occluded"])[0].astype(bool)             # (N, T)
        if tp.shape[1] != T:
            raise ValueError(f"{g}: target_points T={tp.shape[1]} != video T={T}")
        tracks_nt_parts.append(tp)
        vis_nt_parts.append(~occ)

    tracks_nt = np.concatenate(tracks_nt_parts, axis=0)        # (N, T, 2)
    vis_nt = np.concatenate(vis_nt_parts, axis=0)              # (N, T)
    N = tracks_nt.shape[0]

    tracks = np.ascontiguousarray(tracks_nt.transpose(1, 0, 2)).astype(np.float32)  # (T, N, 2)
    visibility = np.ascontiguousarray(vis_nt.transpose(1, 0)).astype(bool)          # (T, N)

    # queries: (N, 3) = (t_first_visible, x, y). The reader recomputes queries at
    # read time; stored here for completeness / standalone use (mirrors tapvid prep).
    queries = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        vis_frames = np.where(visibility[:, n])[0]
        t0 = int(vis_frames[0]) if len(vis_frames) > 0 else 0
        queries[n] = [t0, tracks[t0, n, 0], tracks[t0, n, 1]]

    return dict(frames=frames, tracks=tracks, visibility=visibility, queries=queries)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process(args: argparse.Namespace) -> None:
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    resize = tuple(args.resize)  # (H, W)

    ann_paths = sorted((src_root / "Annotation").glob("*.pkl"))
    if not ann_paths:
        raise FileNotFoundError(f"No .pkl files under {src_root / 'Annotation'}")
    print(f"[info] {len(ann_paths)} sequences -> {out_root}  (points={args.points}, resize={resize})")

    index: List[dict] = []
    for vi, ann in enumerate(ann_paths):
        vid_id = ann.stem
        clip_path = out_root / vid_id / "clip_00000.npz"

        if clip_path.exists() and not args.overwrite:
            with np.load(clip_path) as d:
                num_frames, num_points = d["tracks"].shape[:2]
            print(f"  [{vi + 1}/{len(ann_paths)}] {vid_id} -- skip (exists)")
            index.append(dict(video=vid_id, clip_idx=0, path=f"{vid_id}/clip_00000.npz",
                              num_frames=int(num_frames), num_points=int(num_points)))
            continue

        with open(ann, "rb") as f:
            data = pickle.load(f)
        npz = _surgicalmotion_to_npz(data, args.points)
        if args.save_frames:
            npz["frames"], npz["tracks"] = _resize_clip(npz["frames"], npz["tracks"], resize)
        else:
            # still scale coords to the recorded resize, drop frames
            _, npz["tracks"] = _resize_clip(npz["frames"], npz["tracks"], resize)
            del npz["frames"]

        T, N = npz["tracks"].shape[:2]
        print(f"  [{vi + 1}/{len(ann_paths)}] {vid_id}  T={T}  N={N}")
        _atomic_savez(clip_path, compress=args.compress, **npz)
        index.append(dict(video=vid_id, clip_idx=0, path=f"{vid_id}/clip_00000.npz",
                          num_frames=int(T), num_points=int(N)))

    _atomic_write_json(out_root / "index.json", index)
    _atomic_write_json(out_root / "meta.json", dict(
        source="SurgicalMotion (ground-truth tracks)",
        src_root=str(src_root),
        points=args.points,
        resize=list(resize),
        save_frames=args.save_frames,
        compressed=args.compress,
    ))
    print(f"[done] {len(index)} clips -> {out_root / 'index.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_root", default="DATA/SurgicalMotion", help="dataset root (with Annotation/ + per-seq frame folders)")
    p.add_argument("--out_root", default="DATA/SurgicalMotion/gt_tracks", help="output root folder")
    p.add_argument("--points", choices=["tissue", "tools", "both"], default="both",
                   help="which annotated point set(s) to keep (default: both, concatenated)")
    p.add_argument("--resize", type=int, nargs=2, default=[512, 640], metavar=("H", "W"),
                   help="uniform stored frame size; all clips resized to this, coords scaled to match")
    p.add_argument("--overwrite", action="store_true", help="recompute clips that already exist")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                   help="store only tracks/visibility/queries (frames re-readable from color/*.png)")
    p.add_argument("--no_compress", dest="compress", action="store_false",
                   help="uncompressed .npz (faster to write, larger)")
    p.set_defaults(save_frames=True, compress=True)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    process(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
