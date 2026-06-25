#!/usr/bin/env python
"""cholec80_data_prep.py

Pre-compute CoTracker point tracks for every video in the Cholec80 dataset and
store the result in a layout that a ``torch`` ``DataLoader`` can read directly.

Each long surgical video is streamed frame-by-frame (the raw files are several
GB each, so nothing is ever fully loaded into RAM), optionally temporally
subsampled and spatially resized, then cut into fixed-length clips. CoTracker3
(offline) is run on every clip with a regular query-point grid placed on the
clip's first frame. For every clip we store the (resized) RGB frames together
with the predicted pixel tracks and visibility mask.

Output layout (``--out_root``)::

    out_root/
        index.json                 # global manifest, one entry per clip
        meta.json                  # the run configuration
        video01/
            clip_00000.npz         # frames, tracks, visibility, queries
            clip_00001.npz
            ...
        video02/
            ...

Each ``clip_xxxxx.npz`` contains:
    frames      uint8  (T, H, W, 3)   resized RGB frames
    tracks      float32(T, N, 2)      pixel coords (x, y) in the resized frame
    visibility  bool   (T, N)         per-point visibility
    queries     float32(N, 3)         query points (t, x, y) on the clip

A ready-to-use ``Cholec80TracksDataset`` is provided at the bottom of this file.

Asynchronous / parallel execution
---------------------------------
The script is safe to launch as many concurrent jobs (e.g. a SLURM array)
pointing at the same ``--out_root``:

* Work is sharded by video via ``--num_shards`` / ``--shard_id`` (auto-filled
  from ``SLURM_ARRAY_TASK_COUNT`` / ``SLURM_ARRAY_TASK_ID`` when present), so
  different jobs never touch the same files.
* Already-computed clips are skipped (existence == done, guaranteed by atomic
  writes), so a re-launched / preempted job resumes instead of recomputing.
* Each clip is written atomically (temp file + ``os.replace``); a killed job
  never leaves a half-written ``.npz`` that could look "done".
* The global ``index.json`` is *not* written concurrently: it is rebuilt by
  scanning the output tree (``--build_index``), which any job can do safely.

Example (single process)
-------
    python cholec80_data_prep.py \
        --videos_dir DATA/cholec80/videos \
        --out_root   DATA/cholec80/cotracker_tracks \
        --frame_stride 5 --clip_len 48 --grid_size 20 --resize 480 854

Example (SLURM array of 8 jobs, then merge the manifest)
-------
    sbatch --array=0-7 cholecprep.sbatch
    python cholec80_data_prep.py --out_root DATA/cholec80/cotracker_tracks --build_index
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_cotracker(device: torch.device, repo_dir: Optional[str] = None) -> torch.nn.Module:
    """Load the offline CoTracker3 predictor.

    Tries the local ``co-tracker`` checkout first (offline-friendly, uses the
    cached ``scaled_offline.pth`` checkpoint) and falls back to ``torch.hub``.
    """
    last_err: Optional[Exception] = None
    if repo_dir is not None and Path(repo_dir).exists():
        try:
            model = torch.hub.load(repo_dir, "cotracker3_offline", source="local")
            return model.to(device).eval()
        except Exception as e:  # noqa: BLE001 - fall back to remote hub
            last_err = e
    try:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        return model.to(device).eval()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not load CoTracker (local repo error: {last_err})"
        ) from e


# --------------------------------------------------------------------------- #
# Atomic IO helpers (safe for concurrent jobs sharing one out_root)
# --------------------------------------------------------------------------- #
def atomic_savez(path: Path, compress: bool, **arrays) -> None:
    """Write an ``.npz`` atomically: a temp file in the same dir + ``os.replace``.

    Passing an open file handle to ``np.savez*`` avoids the automatic ``.npz``
    suffix mangling, so the final path is exactly ``path``. ``os.replace`` is
    atomic on a single filesystem, so concurrent readers/jobs only ever see a
    complete file -- existence therefore reliably means "done".
    """
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            (np.savez_compressed if compress else np.savez)(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_json(path: Path, obj) -> None:
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
# Video streaming -> clips
# --------------------------------------------------------------------------- #
def iter_clips(
    video_path: Path,
    clip_len: int,
    frame_stride: int,
    resize: Optional[tuple[int, int]],
    max_clips: Optional[int],
) -> Iterator[np.ndarray]:
    """Yield consecutive, non-overlapping clips as uint8 arrays (T, H, W, 3).

    Frames are read lazily, subsampled by ``frame_stride`` and optionally
    resized to ``resize=(H, W)``. Only full-length clips are yielded; a trailing
    partial clip is discarded so that every saved clip has exactly ``clip_len``
    frames.
    """
    import imageio.v3 as iio

    buf: List[np.ndarray] = []
    produced = 0
    for raw_idx, frame in enumerate(iio.imiter(str(video_path), plugin="FFMPEG")):
        if raw_idx % frame_stride != 0:
            continue
        if resize is not None and frame.shape[:2] != resize:
            # cheap, dependency-free resize via torch (kept on CPU here)
            t = torch.from_numpy(frame).permute(2, 0, 1)[None].float()  # (1,3,h,w)
            t = F.interpolate(t, size=resize, mode="bilinear", align_corners=False)
            frame = t[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        buf.append(frame)
        if len(buf) == clip_len:
            yield np.stack(buf, axis=0)  # (T, H, W, 3)
            buf = []
            produced += 1
            if max_clips is not None and produced >= max_clips:
                return


# --------------------------------------------------------------------------- #
# Tracking
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def track_clip(
    model: torch.nn.Module,
    frames: np.ndarray,  # (T, H, W, 3) uint8
    grid_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run CoTracker on a single clip.

    Returns ``(tracks, visibility, queries)`` as numpy arrays with shapes
    ``(T, N, 2)`` float32, ``(T, N)`` bool and ``(N, 3)`` float32.
    """
    # (T, H, W, 3) -> (1, T, 3, H, W); CoTracker expects raw 0..255 floats
    video = (
        torch.from_numpy(frames)
        .permute(0, 3, 1, 2)[None]
        .float()
        .to(device, non_blocking=True)
    )  # (1, T, 3, H, W)

    tracks, visibility = model(video, grid_size=grid_size)  # (1,T,N,2), (1,T,N)

    # Reconstruct the query grid that the predictor placed on the first frame so
    # it can be stored alongside the tracks (queries are (t=0, x, y)).
    queries = torch.cat(
        [torch.zeros_like(tracks[:, 0, :, :1]), tracks[:, 0]], dim=-1
    )  # (1, N, 3)

    return (
        tracks[0].float().cpu().numpy(),          # (T, N, 2)
        visibility[0].bool().cpu().numpy(),       # (T, N)
        queries[0].float().cpu().numpy(),         # (N, 3)
    )


# --------------------------------------------------------------------------- #
# Work sharding (for asynchronous / SLURM-array launches)
# --------------------------------------------------------------------------- #
def resolve_shard(args: argparse.Namespace) -> tuple[int, int]:
    """Return ``(shard_id, num_shards)``.

    Explicit CLI flags win; otherwise fall back to the SLURM array environment
    so ``sbatch --array=0-N`` just works without extra wiring.
    """
    num_shards = args.num_shards
    shard_id = args.shard_id
    if num_shards is None:
        num_shards = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if shard_id is None:
        shard_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"shard_id={shard_id} out of range for num_shards={num_shards}")
    return shard_id, num_shards


def build_index(out_root: Path) -> List[dict]:
    """Rebuild ``index.json`` by scanning the output tree (race-free).

    Every clip's metadata is fully determined by its path plus the constants in
    ``meta.json`` (``clip_len`` frames, ``grid_size**2`` points), so no ``.npz``
    needs to be opened. Any job can call this safely at any time.
    """
    out_root = Path(out_root)
    with open(out_root / "meta.json") as f:
        meta = json.load(f)
    num_frames = int(meta["clip_len"])
    num_points = int(meta["grid_size"]) ** 2

    entries: List[dict] = []
    for npz in sorted(out_root.glob("*/clip_*.npz")):
        entries.append(
            dict(
                video=npz.parent.name,
                clip_idx=int(npz.stem.split("_")[1]),
                path=npz.relative_to(out_root).as_posix(),
                num_frames=num_frames,
                num_points=num_points,
            )
        )
    atomic_write_json(out_root / "index.json", entries)
    print(f"[index] {len(entries)} clips -> {out_root / 'index.json'}")
    return entries


# --------------------------------------------------------------------------- #
# Main processing loop
# --------------------------------------------------------------------------- #
def process_dataset(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # --build_index: just (re)build the manifest from existing clips and exit.
    if args.build_index:
        build_index(out_root)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available - running on CPU will be extremely slow.")

    videos_dir = Path(args.videos_dir)
    video_paths = sorted(videos_dir.glob("*.mp4"))
    if args.limit_videos is not None:
        video_paths = video_paths[: args.limit_videos]
    if not video_paths:
        raise FileNotFoundError(f"No .mp4 files found in {videos_dir}")

    shard_id, num_shards = resolve_shard(args)
    shard_videos = video_paths[shard_id::num_shards]
    print(f"[info] shard {shard_id}/{num_shards}: {len(shard_videos)}/{len(video_paths)} videos")

    resize = tuple(args.resize) if args.resize is not None else None  # (H, W)

    # meta.json is identical for every shard; write it atomically (idempotent).
    _write_meta(out_root, args)

    print(f"[info] loading CoTracker on {device} ...")
    model = load_cotracker(device, repo_dir=args.cotracker_repo)

    t_start = time.time()
    n_done = n_skipped = 0

    for vi, video_path in enumerate(shard_videos):
        vid_name = video_path.stem
        vid_out = out_root / vid_name
        vid_out.mkdir(parents=True, exist_ok=True)
        print(f"[info] ({vi + 1}/{len(shard_videos)}) {vid_name}")

        for clip_idx, frames in enumerate(
            iter_clips(
                video_path,
                clip_len=args.clip_len,
                frame_stride=args.frame_stride,
                resize=resize,
                max_clips=args.max_clips_per_video,
            )
        ):
            clip_path = vid_out / f"clip_{clip_idx:05d}.npz"

            # Existence == done (writes are atomic), so skip without recomputing.
            if clip_path.exists() and not args.overwrite:
                n_skipped += 1
                continue

            tracks, visibility, queries = track_clip(
                model, frames, grid_size=args.grid_size, device=device
            )

            kwargs = dict(
                tracks=tracks.astype(np.float32),
                visibility=visibility.astype(np.bool_),
                queries=queries.astype(np.float32),
            )
            if args.save_frames:
                kwargs["frames"] = frames.astype(np.uint8)
            atomic_savez(clip_path, args.compress, **kwargs)
            n_done += 1

    dt = time.time() - t_start
    print(f"[done] shard {shard_id}: {n_done} new, {n_skipped} skipped, in {dt:.1f}s")

    # Rebuild the manifest from whatever is on disk. This is safe to run from
    # every shard; for a guaranteed-complete index run `--build_index` once
    # after all jobs finish.
    if not args.no_index:
        build_index(out_root)


def _write_meta(out_root: Path, args: argparse.Namespace) -> None:
    atomic_write_json(
        out_root / "meta.json",
        dict(
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            grid_size=args.grid_size,
            resize=args.resize,
            save_frames=args.save_frames,
            compressed=args.compress,
        ),
    )


# --------------------------------------------------------------------------- #
# Torch Dataset
# --------------------------------------------------------------------------- #
# The reader now lives in the dataset package as the single, config-driven
# reader for this layout (full sampling / geometry / query API shared with the
# Kubric reader via dataset.base.BaseTracksDataset). ``Cholec80TracksDataset`` is
# kept as a backward-compatible alias so the historical
# ``from cholec80_data_prep import Cholec80TracksDataset`` import keeps working.
# Old call form ``Cholec80TracksDataset(root, frames_as_float=..., crop=...)`` is
# still valid (use keyword args).
from dataset.cotracker import CoTrackerTracksDataset  # noqa: E402,F401

Cholec80TracksDataset = CoTrackerTracksDataset


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_dir", default="DATA/cholec80/videos", help="folder with *.mp4 videos")
    p.add_argument("--out_root", default="DATA/cholec80/cotracker_tracks", help="output root folder")
    p.add_argument("--cotracker_repo", default="co-tracker", help="local co-tracker checkout (for offline hub load)")

    p.add_argument("--clip_len", type=int, default=48, help="frames per clip (after subsampling)")
    p.add_argument("--frame_stride", type=int, default=1, help="keep every Nth raw frame (Cholec80 is 25 fps)")
    p.add_argument("--grid_size", type=int, default=20, help="N -> N*N query points per clip")
    p.add_argument("--resize", type=int, nargs=2, default=[480, 854], metavar=("H", "W"),
                   help="resize frames to H W before tracking/saving; pass nothing to keep native")
    p.add_argument("--max_clips_per_video", type=int, default=None, help="cap clips per video (debug)")
    p.add_argument("--limit_videos", type=int, default=None, help="process only the first K videos (debug)")

    # Asynchronous / parallel sharding (auto-filled from the SLURM array env).
    p.add_argument("--num_shards", type=int, default=None,
                   help="total parallel jobs (default: $SLURM_ARRAY_TASK_COUNT or 1)")
    p.add_argument("--shard_id", type=int, default=None,
                   help="this job's index in [0, num_shards) (default: $SLURM_ARRAY_TASK_ID or 0)")
    p.add_argument("--build_index", action="store_true",
                   help="only (re)build index.json from existing clips, then exit")
    p.add_argument("--no_index", action="store_true",
                   help="do not rebuild index.json after processing (build it later with --build_index)")

    p.add_argument("--device", default="cuda")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                   help="store only tracks/visibility (frames can be re-read from the mp4)")
    p.add_argument("--no_compress", dest="compress", action="store_false",
                   help="use uncompressed .npz (faster to write, much larger)")
    p.add_argument("--overwrite", action="store_true", help="recompute clips that already exist")
    p.set_defaults(save_frames=True, compress=True)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.resize is not None and len(args.resize) == 0:
        args.resize = None
    process_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
