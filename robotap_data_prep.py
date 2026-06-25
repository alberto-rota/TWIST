#!/usr/bin/env python
"""robotap_data_prep.py

Convert RoboTAP evaluation pickles (robotap_split0.pkl .. robotap_split4.pkl)
from their single-pickle format into the shared ``index.json`` + per-clip
``.npz`` layout used by TWIST's CoTrackerTracksDataset.

RoboTAP uses the same TAP-Vid format as tapvid_davis / tapvid_rgb_stacking,
except the video arrays are stored as ``mediapy._VideoArray`` objects (a numpy
ndarray subclass). We load them via a thin stub so mediapy itself is not needed
at conversion time.

    Source (per split pickle: dict[video_id -> entry]):
        points   (N, T, 2)      float32  normalized [0, 1] (x, y)
        occluded (N, T)         bool
        video    (T, H, W, 3)   uint8    (_VideoArray subclass of ndarray)

    Target (shared CoTracker layout):
        tracks     (T, N, 2)    float32  pixel (x, y)
        visibility (T, N)       bool
        frames     (T, H, W, 3) uint8    (optional, --no_save_frames to skip)
        queries    (N, 3)       float32  (t_first_vis, x, y)

All 5 splits are merged into one ``out_root`` using the original video key as
the video id.  Duplicate keys across splits are disambiguated with a
``split{N}_`` prefix (rare, but handled).

CPU/IO only -- runs on the login node.

Example::

    python robotap_data_prep.py \\
        --splits_dir DATA/robotap \\
        --out_root   DATA/robotap/gt_tracks
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import List

import numpy as np

# Run-from-anywhere: make the sibling prep helpers importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# Reuse the shared manifest builder so the index stays in lock-step with the
# other datasets (it writes the exact same schema this script emits inline).
from cotracker_tracks_prep import build_index  # noqa: E402


# --------------------------------------------------------------------------- #
# mediapy._VideoArray stub (ndarray subclass -- no mediapy install needed)
# --------------------------------------------------------------------------- #
def _install_mediapy_stub():
    """Register a minimal mediapy stub so the pickle loads without mediapy."""
    if "mediapy" in sys.modules:
        return
    class _VideoArray(np.ndarray):
        pass
    fake = types.ModuleType("mediapy")
    fake._VideoArray = _VideoArray
    sys.modules["mediapy"] = fake


# --------------------------------------------------------------------------- #
# Atomic IO (same pattern as the Cholec80 pipeline)
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
# TAP-Vid -> shared layout conversion (same math as tapvid_data_prep)
# --------------------------------------------------------------------------- #
def _tapvid_to_npz(points, occluded, video, save_frames: bool) -> dict:
    T, H, W = video.shape[:3]
    N = points.shape[0]

    tracks_nt = np.asarray(points, dtype=np.float32)  # (N, T, 2)
    tracks_nt[:, :, 0] *= W
    tracks_nt[:, :, 1] *= H
    tracks = tracks_nt.transpose(1, 0, 2).astype(np.float32)       # (T, N, 2)
    visibility = (~np.asarray(occluded)).transpose(1, 0).astype(bool)  # (T, N)

    queries = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        vis_frames = np.where(visibility[:, n])[0]
        t0 = int(vis_frames[0]) if len(vis_frames) > 0 else 0
        queries[n] = [t0, tracks[t0, n, 0], tracks[t0, n, 1]]

    npz = dict(tracks=tracks, visibility=visibility, queries=queries)
    if save_frames:
        npz["frames"] = np.asarray(video, dtype=np.uint8)
    return npz


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process(args: argparse.Namespace) -> None:
    _install_mediapy_stub()

    splits_dir = Path(args.splits_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.build_index:
        build_index(out_root)
        return

    split_files = sorted(splits_dir.glob("robotap_split*.pkl"))
    if not split_files:
        raise FileNotFoundError(f"No robotap_split*.pkl found under {splits_dir}")
    print(f"[info] {len(split_files)} split pickles -> {out_root}")

    seen_ids: set = set()
    index: List[dict] = []

    for split_pkl in split_files:
        split_num = split_pkl.stem.replace("robotap_split", "")
        print(f"  loading {split_pkl.name} ...")
        with open(split_pkl, "rb") as f:
            data = pickle.load(f)
        print(f"    {len(data)} videos")

        for raw_id, entry in data.items():
            # Disambiguate duplicate ids across splits (rare but safe)
            vid_id = raw_id if raw_id not in seen_ids else f"split{split_num}_{raw_id}"
            seen_ids.add(vid_id)

            clip_path = out_root / vid_id / "clip_00000.npz"
            if clip_path.exists() and not args.overwrite:
                with np.load(clip_path) as d:
                    nf, np_ = d["tracks"].shape[:2]
                index.append(dict(video=vid_id, clip_idx=0,
                                  path=f"{vid_id}/clip_00000.npz",
                                  num_frames=nf, num_points=np_))
                continue

            points   = entry["points"]
            occluded = entry["occluded"]
            video    = entry["video"]
            print(f"    {vid_id}  video={video.shape}  N={points.shape[0]}")

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
        splits_dir=str(splits_dir),
        num_splits=len(split_files),
        save_frames=args.save_frames,
        compress=args.compress,
    ))
    print(f"[done] {len(index)} clips -> {out_root}/index.json")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits_dir", default="DATA/robotap",
                   help="folder containing robotap_split*.pkl files")
    p.add_argument("--out_root", default="DATA/robotap/gt_tracks",
                   help="output root folder")
    p.add_argument("--build_index", action="store_true",
                   help="only (re)build index.json from existing clips, then exit")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false")
    p.add_argument("--no_compress", dest="compress", action="store_false")
    p.set_defaults(save_frames=True, compress=True)
    args = p.parse_args(argv)
    process(args)


if __name__ == "__main__":
    sys.exit(main())
