#!/usr/bin/env python
"""cotracker_tracks_prep.py

Generic CoTracker3 pre-processing for *nested* surgical-video datasets.

This is the sibling of ``cholec80_data_prep.py``. Cholec80 keeps every video in
one flat folder, so that script discovers ``*.mp4`` in a single directory. The
EndoTAPP and SurgT datasets instead nest videos inside a meaningful directory
hierarchy (and SurgT stacks a stereo pair into a single frame), so this script

* discovers videos with a recursive glob (``--video_glob``),
* derives a unique *video id* from each file's path relative to ``--videos_dir``
  (and mirrors that hierarchy under ``--out_root`` so nothing collides), and
* can split a vertically/horizontally stacked stereo frame into independent
  per-eye streams (``--stereo``).

Everything else -- offline CoTracker3 loading, atomic/​resumable IO, SLURM-array
sharding, the per-clip tracking call -- is imported unchanged from
``cholec80_data_prep`` so the two pipelines stay in lock-step.

Output layout (``--out_root``)::

    out_root/
        index.json                       # global manifest, one entry per clip
        meta.json                        # the run configuration
        <video-id>/                      # mirrors the source path (per eye too)
            clip_00000.npz
            clip_00001.npz
            ...

Each ``clip_xxxxx.npz`` contains (identical to the Cholec80 layout):
    frames      uint8  (T, H, W, 3)   resized RGB frames (only if --save_frames)
    tracks      float32(T, N, 2)      pixel coords (x, y) in the resized frame
    visibility  bool   (T, N)         per-point visibility
    queries     float32(N, 3)         query points (t, x, y) on the clip

Clip length
-----------
``--clip_len 0`` (the default for these short, fixed-length sequences) makes the
*entire* sub-sampled video a single, variable-length clip. A positive
``--clip_len`` reproduces the Cholec80 behaviour of cutting fixed-length,
non-overlapping clips (trailing partial clip discarded). Because clips may now
have different lengths, ``index.json`` stores each clip's ``num_frames`` /
``num_points`` (read straight from the ``.npz``) instead of deriving them from
``meta.json``.

Examples
--------
EndoTAPP (mono, 1920x1080 -> 480x854, whole sequence per clip)::

    python cotracker_tracks_prep.py \
        --videos_dir DATA/EndoTAPP \
        --video_glob 'tissue_dataset_seqs/*/left/seq*/frames/*-visible.mp4' \
        --out_root   DATA/EndoTAPP/cotracker_tracks \
        --clip_len 0 --frame_stride 1 --grid_size 20 --resize 480 854

SurgT (vertical stereo pair -> two per-eye streams, 1280x1024 -> 512 640)::

    python cotracker_tracks_prep.py \
        --videos_dir DATA/SurgT \
        --video_glob 'case_*/*/video.mp4' \
        --out_root   DATA/SurgT/cotracker_tracks \
        --stereo vertical --clip_len 0 --frame_stride 1 --resize 512 640

SLURM array of 8 shards, then merge the manifest::

    sbatch --array=0-7 endotappprep.sbatch
    python cotracker_tracks_prep.py --out_root DATA/EndoTAPP/cotracker_tracks --build_index
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the battle-tested Cholec80 helpers verbatim (keeps both pipelines in
# sync and avoids code drift). ``cholec80_data_prep`` is import-safe: all its
# side effects are guarded behind ``if __name__ == "__main__"``.
from cholec80_data_prep import (
    atomic_savez,
    atomic_write_json,
    load_cotracker,
    resolve_shard,
    track_clip,
)


# --------------------------------------------------------------------------- #
# Stereo geometry: how to slice one stacked frame into a single eye
# --------------------------------------------------------------------------- #
# A "split" is (axis, half): axis 0 -> split rows (vertical stack), axis 1 ->
# split cols (horizontal stack); half 0 -> first/top/left, 1 -> second/bottom/right.
def _slice_eye(frame: np.ndarray, split: Optional[Tuple[int, int]]) -> np.ndarray:
    """Return one eye of a stacked frame, computed from the live frame shape.

    frame: (H, W, 3). Doing this per-frame (rather than from a pre-probed size)
    keeps the streaming reader dependency-free and robust to odd dimensions.
    """
    if split is None:
        return frame
    axis, half = split
    n = frame.shape[axis]
    mid = n // 2
    sl = slice(0, mid) if half == 0 else slice(mid, n)
    return frame[sl, :, :] if axis == 0 else frame[:, sl, :]


def _resolve_stereo(stereo: str, video_path: Path) -> str:
    """Resolve the stereo stacking for a single video.

    SurgT is heterogeneous: the ``.mp4`` cases stack the pair *vertically* while
    the ``.avi`` cases stack *horizontally*. ``--stereo auto`` reads the
    authoritative ``video_stack`` field from the sibling ``info.yaml`` so each
    video is split on the correct axis. Any explicit ``--stereo`` value applies
    unchanged to every video.
    """
    if stereo != "auto":
        return stereo
    info = video_path.parent / "info.yaml"
    try:
        text = info.read_text()
    except OSError as e:  # missing/unreadable -> fail loud, never silently mis-split
        raise FileNotFoundError(f"--stereo auto needs {info} to read 'video_stack' ({e})") from e
    m = re.search(r'video_stack\s*:\s*"?(\w+)"?', text)
    if not m:
        raise ValueError(f"no 'video_stack' field in {info}")
    stack = m.group(1).lower()
    if stack not in ("vertical", "horizontal", "none"):
        raise ValueError(f"unexpected video_stack={stack!r} in {info}")
    return stack


def _stereo_eyes(stereo: str, eyes: List[str]) -> List[Tuple[str, Optional[Tuple[int, int]]]]:
    """Map (--stereo, --stereo_eyes) to a list of (eye_name, split) sub-streams.

    Returns ``[("", None)]`` for mono so the rest of the pipeline is uniform.
    """
    if stereo == "none":
        return [("", None)]
    axis = 0 if stereo == "vertical" else 1  # vertical stack -> split rows
    # Convention: first half (top / left of the stack) is the LEFT eye.
    eye_to_half = {"left": 0, "right": 1}
    streams: List[Tuple[str, Optional[Tuple[int, int]]]] = []
    for eye in eyes:
        if eye not in eye_to_half:
            raise ValueError(f"--stereo_eyes must be from {{left,right}}, got {eye!r}")
        streams.append((eye, (axis, eye_to_half[eye])))
    return streams


# --------------------------------------------------------------------------- #
# Video streaming -> clips (adds: per-eye slicing + whole-video mode)
# --------------------------------------------------------------------------- #
def iter_clips(
    video_path: Path,
    clip_len: int,
    frame_stride: int,
    resize: Optional[Tuple[int, int]],
    split: Optional[Tuple[int, int]],
    max_clips: Optional[int],
) -> Iterator[np.ndarray]:
    """Yield clips as uint8 arrays (T, H, W, 3), streaming frame-by-frame.

    Pipeline per kept frame: optional stereo ``split`` -> optional ``resize``.
    ``clip_len <= 0`` yields the whole (sub-sampled) video as one clip; a
    positive ``clip_len`` yields consecutive non-overlapping fixed-length clips
    (trailing partial clip discarded), matching the Cholec80 behaviour.
    """
    import imageio.v3 as iio

    whole_video = clip_len is None or clip_len <= 0
    buf: List[np.ndarray] = []
    produced = 0
    for raw_idx, frame in enumerate(iio.imiter(str(video_path), plugin="FFMPEG")):
        if raw_idx % frame_stride != 0:
            continue
        frame = _slice_eye(frame, split)
        if resize is not None and frame.shape[:2] != tuple(resize):
            # cheap, dependency-free resize via torch (kept on CPU here)
            t = torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1)[None].float()  # (1,3,h,w)
            t = F.interpolate(t, size=tuple(resize), mode="bilinear", align_corners=False)
            frame = t[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
        buf.append(frame)
        if not whole_video and len(buf) == clip_len:
            yield np.stack(buf, axis=0)  # (T, H, W, 3)
            buf = []
            produced += 1
            if max_clips is not None and produced >= max_clips:
                return
    if whole_video and buf:
        yield np.stack(buf, axis=0)  # (T, H, W, 3)


# --------------------------------------------------------------------------- #
# Video discovery + id derivation
# --------------------------------------------------------------------------- #
def discover_videos(videos_dir: Path, video_glob: str) -> List[Tuple[Path, str]]:
    """Return sorted ``(video_path, video_id)`` pairs.

    ``video_id`` is the file's path relative to ``videos_dir`` with the suffix
    stripped (posix, e.g. ``tissue_dataset_seqs/4/left/seq002/frames/..-visible``
    or ``case_2/1/video``). It is unique by construction and is mirrored as a
    sub-directory tree under ``out_root``.
    """
    paths = sorted(videos_dir.glob(video_glob))
    out: List[Tuple[Path, str]] = []
    for p in paths:
        vid = p.relative_to(videos_dir).with_suffix("").as_posix()
        out.append((p, vid))
    return out


# --------------------------------------------------------------------------- #
# Index (rglob + read per-clip shapes; supports variable-length clips)
# --------------------------------------------------------------------------- #
def build_index(out_root: Path) -> List[dict]:
    """Rebuild ``index.json`` by scanning the (possibly nested) output tree.

    Clip length may vary per video, so ``num_frames`` / ``num_points`` are read
    from each ``.npz`` (the ``tracks`` array is tiny: ``T*N*2`` floats). This is
    race-free and any job may call it at any time.
    """
    out_root = Path(out_root)
    entries: List[dict] = []
    for npz in sorted(out_root.rglob("clip_*.npz")):
        with np.load(npz) as d:
            num_frames, num_points = (int(x) for x in d["tracks"].shape[:2])  # (T, N, 2)
        entries.append(
            dict(
                video=npz.parent.relative_to(out_root).as_posix(),
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

    if args.build_index:
        build_index(out_root)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available - running on CPU will be extremely slow.")

    videos_dir = Path(args.videos_dir)
    videos = discover_videos(videos_dir, args.video_glob)
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]
    if not videos:
        raise FileNotFoundError(f"No videos matched {args.video_glob!r} under {videos_dir}")

    shard_id, num_shards = resolve_shard(args)
    shard_videos = videos[shard_id::num_shards]
    print(f"[info] shard {shard_id}/{num_shards}: {len(shard_videos)}/{len(videos)} videos")

    resize = tuple(args.resize) if args.resize else None  # (H, W)

    _write_meta(out_root, args)

    print(f"[info] loading CoTracker on {device} ...")
    model = load_cotracker(device, repo_dir=args.cotracker_repo)

    t_start = time.time()
    n_done = n_skipped = 0

    for vi, (video_path, vid_id) in enumerate(shard_videos):
        # Stack orientation may differ per video (vertical .mp4 vs horizontal
        # .avi in SurgT), so resolve it here rather than once for the whole run.
        eye_streams = _stereo_eyes(_resolve_stereo(args.stereo, video_path), args.stereo_eyes)
        print(f"[info] ({vi + 1}/{len(shard_videos)}) {vid_id}")
        for eye_name, split in eye_streams:
            # Mirror the source hierarchy; append the eye as a final level.
            vid_out = out_root / vid_id / eye_name if eye_name else out_root / vid_id
            vid_out.mkdir(parents=True, exist_ok=True)

            for clip_idx, frames in enumerate(
                iter_clips(
                    video_path,
                    clip_len=args.clip_len,
                    frame_stride=args.frame_stride,
                    resize=resize,
                    split=split,
                    max_clips=args.max_clips_per_video,
                )
            ):
                clip_path = vid_out / f"clip_{clip_idx:05d}.npz"
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

                # Long sequences make the offline correlation volumes huge; free
                # cached blocks between clips to curb fragmentation/peak usage.
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    dt = time.time() - t_start
    print(f"[done] shard {shard_id}: {n_done} new, {n_skipped} skipped, in {dt:.1f}s")

    if not args.no_index:
        build_index(out_root)


def _write_meta(out_root: Path, args: argparse.Namespace) -> None:
    atomic_write_json(
        out_root / "meta.json",
        dict(
            videos_dir=args.videos_dir,
            video_glob=args.video_glob,
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            grid_size=args.grid_size,
            resize=args.resize,
            stereo=args.stereo,
            stereo_eyes=args.stereo_eyes,
            save_frames=args.save_frames,
            compressed=args.compress,
        ),
    )


# --------------------------------------------------------------------------- #
# Torch Dataset (generic; reads the layout produced above)
# --------------------------------------------------------------------------- #
# The reader now lives in the dataset package as the single, config-driven
# reader for this layout (full sampling / geometry / query API shared with the
# Kubric reader via dataset.base.BaseTracksDataset). Re-exported here so the
# historical ``from cotracker_tracks_prep import CoTrackerTracksDataset`` import
# keeps working. Old call form ``CoTrackerTracksDataset(root, frames_as_float=...,
# crop=...)`` is still valid (use keyword args).
from dataset.cotracker import CoTrackerTracksDataset  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos_dir", required=True, help="dataset root to search under")
    p.add_argument("--video_glob", required=True,
                   help="recursive glob (relative to --videos_dir), e.g. 'case_*/*/video.mp4'")
    p.add_argument("--out_root", required=True, help="output root folder")
    p.add_argument("--cotracker_repo", default="co-tracker", help="local co-tracker checkout (offline hub load)")

    p.add_argument("--clip_len", type=int, default=0,
                   help="frames per clip after subsampling; <=0 -> whole video as one clip")
    p.add_argument("--frame_stride", type=int, default=1, help="keep every Nth raw frame")
    p.add_argument("--grid_size", type=int, default=20, help="N -> N*N query points per clip")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="resize (per eye) to H W before tracking/saving; omit to keep native")
    p.add_argument("--stereo", choices=["none", "vertical", "horizontal", "auto"], default="none",
                   help="split a stacked stereo frame into per-eye streams (first half = left eye); "
                        "'auto' reads each video's stack orientation from its sibling info.yaml")
    p.add_argument("--stereo_eyes", nargs="+", default=["left", "right"],
                   help="which eyes to process when --stereo is set (subset/order of left right)")
    p.add_argument("--max_clips_per_video", type=int, default=None, help="cap clips per video (debug)")
    p.add_argument("--limit_videos", type=int, default=None, help="process only the first K videos (debug)")

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
                   help="store only tracks/visibility (frames can be re-read from the video)")
    p.add_argument("--no_compress", dest="compress", action="store_false",
                   help="use uncompressed .npz (faster to write, much larger)")
    p.add_argument("--overwrite", action="store_true", help="recompute clips that already exist")
    p.set_defaults(save_frames=True, compress=True)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    process_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
