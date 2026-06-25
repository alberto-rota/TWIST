#!/usr/bin/env python
"""pointodyssey_data_prep.py

Convert the **PointOdyssey** dataset into the project's shared on-disk layout
(``index.json`` + per-clip ``.npz``), so it is read by the very same
``CoTrackerTracksDataset`` (``cotracker_tracks_prep.py``) as Cholec80 / EndoTAPP
/ SurgT and drops straight into ``dataset_demo.ipynb``.

PointOdyssey already ships **ground-truth** point tracks, so -- unlike the
surgical pipelines -- *no CoTracker / GPU inference is run here*. This is a pure
(CPU/IO-bound) re-packaging of the GT annotations into fixed-length clips.

Source layout (``--data_dir``)::

    data_dir/
        <seq>/
            rgbs/rgb_00000.jpg ... rgb_0XXXX.jpg     # (H, W, 3) uint8 frames
            anno.npz                                  # GT annotations (see below)
            info.npz, scene_info.json, depths/, masks/, normals/   # unused here
        ...

``anno.npz`` (frame-major) holds::
    trajs_2d   float32 (T, P, 2)   pixel coords (x, y); off-view points are +/-inf
    visibs     bool    (T, P)      per-frame visibility (not occluded)
    valids     bool    (T, P)      per-frame validity (point exists / defined)

Output layout (``--out_root``) -- identical to the other datasets::

    out_root/
        index.json                 # global manifest, one entry per clip
        meta.json                  # the run configuration
        <seq>/
            clip_00000.npz
            clip_00001.npz
            ...

Each ``clip_xxxxx.npz`` contains:
    frames      uint8  (T, H, W, 3)   RGB frames (only if --save_frames)
    tracks      float32(T, N, 2)      pixel coords (x, y) in the (resized) frame
    visibility  bool   (T, N)         per-point visibility
    queries     float32(N, 3)         query points (t=0, x, y) on the clip

Point subsampling
-----------------
PointOdyssey stores ~11 000 points per sequence, most of which are off-screen or
occluded for any given clip. We keep ``--num_points`` of them, drawn evenly from
the points that are **visible and inside the frame at the clip's query (first)
frame** (the standard CoTracker query convention), so the saved tracks land on
actually-trackable -- including moving foreground -- points.

Clip length
-----------
PointOdyssey sequences are long (~1.5k-2k frames). ``--clip_len 48`` (default)
cuts consecutive, non-overlapping 48-frame clips (trailing partial discarded);
``--clip_len 0`` makes the whole (sub-sampled) sequence a single clip.

Asynchronous / parallel execution (same contract as the other prep scripts)
---------------------------------------------------------------------------
Safe to launch as a SLURM array against one ``--out_root``: work is sharded by
sequence (``--num_shards`` / ``--shard_id``, auto-filled from the SLURM array
env), every clip is written atomically (existence == done -> resumable), and the
manifest is rebuilt by scanning the tree (``--build_index``).

Examples
--------
Single process (native 540x960, 48-frame clips, 400 points)::

    python pointodyssey_data_prep.py \
        --data_dir DATA/PointOdissey/train \
        --out_root DATA/PointOdissey/gt_tracks \
        --clip_len 48 --num_points 400

SLURM array of 8 shards, then merge the manifest::

    sbatch --array=0-7 pointodysseyprep.sbatch
    python pointodyssey_data_prep.py --out_root DATA/PointOdissey/gt_tracks --build_index
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the shared, battle-tested helpers so every pipeline stays in lock-step.
from cholec80_data_prep import atomic_savez, atomic_write_json, resolve_shard
from cotracker_tracks_prep import build_index


# --------------------------------------------------------------------------- #
# Sequence discovery + id derivation
# --------------------------------------------------------------------------- #
def discover_sequences(data_dir: Path, anno_glob: str) -> List[Tuple[Path, str]]:
    """Return sorted ``(anno_path, seq_id)`` pairs.

    ``seq_id`` is the annotation file's *parent* path relative to ``data_dir``
    (posix), so the source hierarchy is mirrored under ``out_root`` and ids never
    collide.
    """
    out: List[Tuple[Path, str]] = []
    for anno in sorted(data_dir.glob(anno_glob)):
        seq_id = anno.parent.relative_to(data_dir).as_posix()
        out.append((anno, seq_id))
    return out


# --------------------------------------------------------------------------- #
# Frame IO + resize (coordinates are scaled to match)
# --------------------------------------------------------------------------- #
def read_frames(seq_dir: Path, raw_idx: np.ndarray, name_tmpl: str) -> np.ndarray:
    """Load the RGB frames at ``raw_idx`` -> (T, H, W, 3) uint8."""
    import imageio.v3 as iio

    frames = [
        iio.imread(seq_dir / "rgbs" / name_tmpl.format(int(k)))[..., :3]
        for k in raw_idx
    ]
    return np.stack(frames, axis=0)  # (T, H, W, 3)


def resize_frames(frames: np.ndarray, size: Tuple[int, int], device: torch.device) -> np.ndarray:
    """Bilinearly resize (T, H, W, 3) uint8 -> (T, h, w, 3) uint8."""
    t = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()  # (T,3,H,W)
    t = t.to(device, non_blocking=True)
    t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)           # (T,3,h,w)
    return t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).cpu().numpy()  # (T,h,w,3)


# --------------------------------------------------------------------------- #
# Point selection (visible + in-frame at the clip's query frame)
# --------------------------------------------------------------------------- #
def select_points(
    q_xy: np.ndarray,   # (P, 2) coords at the query frame
    q_vis: np.ndarray,  # (P,)   visibility at the query frame
    w: int,
    h: int,
    k: int,
) -> np.ndarray:
    """Pick ``k`` point indices usable at the query frame (deterministic).

    Candidates are points that are visible *and* inside the (native) frame at the
    query frame; ``k`` of them are drawn evenly for good spatial spread. If fewer
    than ``k`` qualify, the remainder is filled with the other points.
    """
    finite = np.isfinite(q_xy).all(axis=-1)                       # (P,)
    on = (
        q_vis & finite
        & (q_xy[:, 0] >= 0) & (q_xy[:, 0] < w)
        & (q_xy[:, 1] >= 0) & (q_xy[:, 1] < h)
    )                                                             # (P,)
    sel = np.flatnonzero(on)
    if sel.size >= k:
        pick = np.linspace(0, sel.size - 1, k).round().astype(np.int64)
        return sel[pick]
    rest = np.flatnonzero(~on)
    return np.concatenate([sel, rest])[:k]


# --------------------------------------------------------------------------- #
# Per-sequence clip generator
# --------------------------------------------------------------------------- #
def iter_clips(
    anno_path: Path,
    clip_len: int,
    frame_stride: int,
    num_points: int,
    resize: Optional[Tuple[int, int]],
    max_clips: Optional[int],
    name_tmpl: str,
    save_frames: bool,
    device: torch.device,
):
    """Yield ``(frames|None, tracks, visibility, queries)`` per clip of a sequence.

    Shapes: frames (T,h,w,3) uint8, tracks (T,N,2) f32, visibility (T,N) bool,
    queries (N,3) f32. ``clip_len <= 0`` -> one whole-sequence clip.
    """
    seq_dir = anno_path.parent
    with np.load(anno_path) as a:
        trajs = a["trajs_2d"]                         # (T0, P, 2) f32
        vis = a["visibs"] & a["valids"]               # (T0, P)    bool (observed iff visible & valid)

    t0 = trajs.shape[0]
    kept = np.arange(0, t0, frame_stride)             # raw frame indices we keep
    trajs = trajs[kept]                               # (T, P, 2)
    vis = vis[kept]                                   # (T, P)
    seq_len = trajs.shape[0]

    # native resolution (for the in-frame test before any resize)
    import imageio.v3 as iio
    h0, w0 = iio.imread(seq_dir / "rgbs" / name_tmpl.format(int(kept[0]))).shape[:2]

    if resize is not None:
        hf, wf = int(resize[0]), int(resize[1])
        sx, sy = wf / w0, hf / h0
    else:
        hf, wf, sx, sy = h0, w0, 1.0, 1.0

    whole = clip_len is None or clip_len <= 0
    step = seq_len if whole else clip_len
    n_clips = 1 if whole else seq_len // clip_len

    produced = 0
    for ci in range(max(n_clips, 1)):
        s = ci * step
        e = s + step
        ct = np.asarray(trajs[s:e])                   # (L, P, 2)
        cv = np.asarray(vis[s:e])                     # (L, P)

        # choose points at the clip's query (first) frame, in NATIVE pixels
        sel = select_points(ct[0], cv[0], w0, h0, num_points)   # (N,)
        ct = ct[:, sel]                               # (L, N, 2)
        cv = cv[:, sel]                               # (L, N)

        # off-view points are stored as +/-inf -> zero them and mark not-visible
        finite = np.isfinite(ct).all(axis=-1)         # (L, N)
        ct = np.where(finite[..., None], ct, 0.0).astype(np.float32)
        cv = cv & finite

        # scale coordinates to the (optional) resized resolution
        if resize is not None:
            ct[..., 0] *= sx
            ct[..., 1] *= sy

        # a point outside the (final) frame is not observable -> not-visible
        x, y = ct[..., 0], ct[..., 1]
        cv = cv & (x >= 0) & (x < wf) & (y >= 0) & (y < hf)

        # queries: (t=0, x, y) at the clip's first frame
        queries = np.concatenate(
            [np.zeros((ct.shape[1], 1), np.float32), ct[0]], axis=1
        )                                             # (N, 3)

        frames = None
        if save_frames:
            frames = read_frames(seq_dir, kept[s:e], name_tmpl)   # (L,h0,w0,3)
            if resize is not None:
                frames = resize_frames(frames, (hf, wf), device)  # (L,hf,wf,3)

        yield frames, ct, cv, queries
        produced += 1
        if max_clips is not None and produced >= max_clips:
            return


# --------------------------------------------------------------------------- #
# Main processing loop
# --------------------------------------------------------------------------- #
def process_dataset(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.build_index:
        build_index(out_root)
        return

    data_dir = Path(args.data_dir)
    sequences = discover_sequences(data_dir, args.anno_glob)
    if args.limit_videos is not None:
        sequences = sequences[: args.limit_videos]
    if not sequences:
        raise FileNotFoundError(f"No '{args.anno_glob}' found under {data_dir}")

    shard_id, num_shards = resolve_shard(args)
    shard_seqs = sequences[shard_id::num_shards]
    print(f"[info] shard {shard_id}/{num_shards}: {len(shard_seqs)}/{len(sequences)} sequences")

    # GPU is optional here (only the resize uses it); CPU works too.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    resize = tuple(args.resize) if args.resize else None  # (H, W)

    _write_meta(out_root, args)

    t_start = time.time()
    n_done = n_skipped = 0

    for vi, (anno_path, seq_id) in enumerate(shard_seqs):
        seq_out = out_root / seq_id
        seq_out.mkdir(parents=True, exist_ok=True)
        print(f"[info] ({vi + 1}/{len(shard_seqs)}) {seq_id}")

        # Skip the (expensive) anno load entirely if every clip already exists is
        # not known up-front (clip count depends on length), so we stream clips;
        # each already-present clip is skipped cheaply below.
        for clip_idx, (frames, tracks, visibility, queries) in enumerate(
            iter_clips(
                anno_path,
                clip_len=args.clip_len,
                frame_stride=args.frame_stride,
                num_points=args.num_points,
                resize=resize,
                max_clips=args.max_clips_per_video,
                name_tmpl=args.frame_name_tmpl,
                save_frames=args.save_frames,
                device=device,
            )
        ):
            clip_path = seq_out / f"clip_{clip_idx:05d}.npz"
            if clip_path.exists() and not args.overwrite:
                n_skipped += 1
                continue

            kwargs = dict(
                tracks=tracks.astype(np.float32),
                visibility=visibility.astype(np.bool_),
                queries=queries.astype(np.float32),
            )
            if frames is not None:
                kwargs["frames"] = frames.astype(np.uint8)
            atomic_savez(clip_path, args.compress, **kwargs)
            n_done += 1

    dt = time.time() - t_start
    print(f"[done] shard {shard_id}: {n_done} new, {n_skipped} skipped, in {dt:.1f}s")

    if not args.no_index:
        build_index(out_root)


def _write_meta(out_root: Path, args: argparse.Namespace) -> None:
    atomic_write_json(
        out_root / "meta.json",
        dict(
            source="PointOdyssey (ground-truth tracks)",
            data_dir=args.data_dir,
            anno_glob=args.anno_glob,
            clip_len=args.clip_len,
            frame_stride=args.frame_stride,
            num_points=args.num_points,
            resize=args.resize,
            save_frames=args.save_frames,
            compressed=args.compress,
        ),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="DATA/PointOdissey/train",
                   help="split root that holds the <seq>/ folders")
    p.add_argument("--anno_glob", default="*/anno.npz",
                   help="glob (relative to --data_dir) matching each sequence's anno.npz")
    p.add_argument("--out_root", default="DATA/PointOdissey/gt_tracks", help="output root folder")
    p.add_argument("--frame_name_tmpl", default="rgb_{:05d}.jpg",
                   help="rgb filename template (frame index formatted in)")

    p.add_argument("--clip_len", type=int, default=48,
                   help="frames per clip after subsampling; <=0 -> whole sequence as one clip")
    p.add_argument("--frame_stride", type=int, default=1, help="keep every Nth raw frame")
    p.add_argument("--num_points", type=int, default=400, help="points kept per clip")
    p.add_argument("--resize", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="resize frames to H W (coords scaled to match); omit to keep native")
    p.add_argument("--max_clips_per_video", type=int, default=None, help="cap clips per sequence (debug)")
    p.add_argument("--limit_videos", type=int, default=None, help="process only the first K sequences (debug)")

    p.add_argument("--num_shards", type=int, default=None,
                   help="total parallel jobs (default: $SLURM_ARRAY_TASK_COUNT or 1)")
    p.add_argument("--shard_id", type=int, default=None,
                   help="this job's index in [0, num_shards) (default: $SLURM_ARRAY_TASK_ID or 0)")
    p.add_argument("--build_index", action="store_true",
                   help="only (re)build index.json from existing clips, then exit")
    p.add_argument("--no_index", action="store_true",
                   help="do not rebuild index.json after processing (build it later with --build_index)")

    p.add_argument("--device", default="cuda", help="device for the optional resize (falls back to cpu)")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                   help="store only tracks/visibility (frames can be re-read from rgbs/)")
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
