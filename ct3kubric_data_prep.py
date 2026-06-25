#!/usr/bin/env python
"""ct3kubric_data_prep.py

Convert the **CT3Kubric** synthetic dataset from its bespoke per-sequence layout
into the *shared* CoTracker on-disk layout (``index.json`` + per-clip ``.npz``)
used by every other TWIST dataset (Cholec80 / EndoTAPP / SurgT / PointOdyssey).

After conversion CT3Kubric is read by the single, config-driven
:class:`dataset.cotracker.CoTrackerTracksDataset` like everything else -- there
is no longer a CT3Kubric-specific reader. CT3Kubric already ships fully
pre-computed tracks, so (unlike the surgical pipelines) there is **no tracking
step**: this script just transposes the stored arrays into the shared layout and
embeds the RGB frames into each clip's ``.npz``.

Source layout (``--src_root``, the raw CT3Kubric tree)::

    src_root/
        0000/
            0000_trajs_2d.npy    # (N, T, 2) float32  point-major pixel tracks (x, y)
            0000_visibility.npy  # (N, T)    bool      point-major visibility
            0000.npy             # large per-scene dict   (ignored)
            0000_with_rank.npz   # camera params          (ignored)
            frames/000.png .. 119.png      # (H, W, 3) uint8 RGB frames
            depths/000.npy .. 119.npy      # (H, W)    depth maps (dropped -- see below)
        0001/ ...

Output layout (``--out_root``, the shared CoTracker layout)::

    out_root/
        index.json                 # global manifest, one entry per clip
        meta.json                  # this run's configuration
        0000/
            clip_00000.npz         # tracks (T,N,2), visibility (T,N), queries (N,3), frames (T,H,W,3)
        0001/ ...

Each CT3Kubric sequence becomes a **single** ``clip_00000.npz`` holding the whole
(optionally frame-strided) sequence -- the shared ``clip_len=0`` convention.
:class:`~dataset.cotracker.CoTrackerTracksDataset` then sub-windows it at read
time (``CLIP_LEN`` / ``FRAME_STRIDE`` / ``CLIP_STRIDE`` / ``MAX_CLIPS_PER_VIDEO``),
so the full read-time clip flexibility of the old Kubric reader is preserved.

What changes vs. the raw data
-----------------------------
* tracks / visibility are transposed point-major ``(N, T)`` -> frame-major
  ``(T, N)`` (the shared on-disk convention).
* **visibility is inverted** -- the raw ``_visibility.npy`` uses the opposite
  convention (``True`` = occluded, ``False`` = visible at the rendered surface),
  which is the inverse of the shared standard (``True`` = visible).  The
  conversion applies ``~vis`` to restore the expected semantics.
* RGB frames are embedded into the ``.npz`` (the shared reader loads frames from
  the clip file, not a sibling ``frames/`` directory).
* **depth maps are dropped** -- the shared layout carries no depth and the model
  never consumes it, so this is lossless for training. (The raw ``src_root`` is
  left untouched, so depth remains available there if ever needed.)

Asynchronous / parallel execution
----------------------------------
Safe to launch as many concurrent jobs as you like against the same
``--out_root``: work is sharded by sequence via ``--num_shards`` / ``--shard_id``
(auto-filled from ``SLURM_ARRAY_TASK_COUNT`` / ``SLURM_ARRAY_TASK_ID``), every
clip is written atomically (temp file + ``os.replace``), and ``index.json`` is
rebuilt by scanning the tree -- so existence of a ``.npz`` reliably means "done".

This is CPU/IO-only (no GPU, no network), so it runs fine on the login node.

Examples
--------
Smoke convert (first 4 sequences)::

    python ct3kubric_data_prep.py \
        --src_root DATA/CT3Kubric \
        --out_root DATA/CT3Kubric/cotracker_tracks \
        --limit_videos 4

Full convert::

    python ct3kubric_data_prep.py \
        --src_root DATA/CT3Kubric --out_root DATA/CT3Kubric/cotracker_tracks

SLURM array of 8 shards, then merge the manifest::

    sbatch --array=0-7 ct3kubricprep.sbatch
    python ct3kubric_data_prep.py --out_root DATA/CT3Kubric/cotracker_tracks --build_index

Patch existing clips converted before the visibility-inversion fix::

    python ct3kubric_data_prep.py --out_root DATA/CT3Kubric/cotracker_tracks --fix_visibility
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Reuse the battle-tested sibling helpers verbatim so all prep pipelines stay in
# lock-step (atomic/​resumable IO, SLURM sharding, the manifest scan). Both
# modules are import-safe: their side effects are guarded behind __main__.
from cholec80_data_prep import atomic_savez, atomic_write_json, resolve_shard
from cotracker_tracks_prep import build_index


# --------------------------------------------------------------------------- #
# Sequence discovery
# --------------------------------------------------------------------------- #
def discover_sequences(src_root: Path) -> List[str]:
    """Digit-named sequence folders under ``src_root`` that have a tracks file."""
    out: List[str] = []
    for p in sorted(src_root.iterdir()):
        if p.is_dir() and p.name.isdigit() and (p / f"{p.name}_trajs_2d.npy").exists():
            out.append(p.name)
    return out


# --------------------------------------------------------------------------- #
# Raw-sequence -> shared clip arrays
# --------------------------------------------------------------------------- #
def _read_frames(seq_dir: Path, frame_ids: np.ndarray) -> np.ndarray:
    import imageio.v3 as iio

    frames = [iio.imread(seq_dir / "frames" / f"{i:03d}.png")[..., :3] for i in frame_ids]
    return np.stack(frames, axis=0).astype(np.uint8)  # (T, H, W, 3)


def convert_sequence(
    seq_dir: Path,
    frame_stride: int,
    save_frames: bool,
) -> dict:
    """Load one raw CT3Kubric sequence and return the shared clip arrays.

    Returns a kwargs dict ready for :func:`atomic_savez`:
    ``tracks (T,N,2) f32``, ``visibility (T,N) bool``, ``queries (N,3) f32`` and
    (if ``save_frames``) ``frames (T,H,W,3) uint8``. Tracks/visibility are
    transposed from the stored point-major ``(N, T)`` to frame-major ``(T, N)``.

    NOTE: CT3Kubric ``_visibility.npy`` uses an **inverted** convention -- stored
    ``True`` means the point is occluded (behind the rendered surface), ``False``
    means it is at the visible surface.  We invert on load so the shared layout
    uses the standard ``True=visible`` convention.
    """
    name = seq_dir.name
    trajs = np.load(seq_dir / f"{name}_trajs_2d.npy")          # (N, T, 2) f32
    # CT3Kubric visibility is inverted: stored True = occluded, False = visible.
    # Invert here so the output follows the shared True=visible convention.
    vis = ~np.load(seq_dir / f"{name}_visibility.npy")         # (N, T)    bool

    seq_len = trajs.shape[1]
    frame_ids = np.arange(0, seq_len, max(int(frame_stride), 1))  # (T,)

    tracks = np.ascontiguousarray(
        trajs[:, frame_ids].transpose(1, 0, 2)
    ).astype(np.float32)                                        # (T, N, 2)
    visibility = np.ascontiguousarray(
        vis[:, frame_ids].transpose(1, 0)
    ).astype(np.bool_)                                          # (T, N)

    # queries: (t=0, x, y) at the first kept frame -- matches the sibling preps.
    queries = np.concatenate(
        [np.zeros((tracks.shape[1], 1), np.float32), tracks[0]], axis=1
    ).astype(np.float32)                                        # (N, 3)

    kwargs = dict(tracks=tracks, visibility=visibility, queries=queries)
    if save_frames:
        kwargs["frames"] = _read_frames(seq_dir, frame_ids)    # (T, H, W, 3) uint8
    return kwargs


def _infer_native_hw(seq_dir: Path) -> Optional[Tuple[int, int]]:
    """Native frame ``(H, W)`` via a single header read (depth npy, else PNG)."""
    dp = seq_dir / "depths" / "000.npy"
    if dp.exists():
        return tuple(int(s) for s in np.load(dp, mmap_mode="r").shape[:2])
    png = seq_dir / "frames" / "000.png"
    if png.exists():
        import imageio.v3 as iio

        return tuple(int(s) for s in iio.imread(png).shape[:2])
    return None


# --------------------------------------------------------------------------- #
# Main processing loop
# --------------------------------------------------------------------------- #
def fix_visibility_inplace(out_root: Path, args: argparse.Namespace) -> None:
    """Patch already-converted .npz files by inverting the visibility array.

    Use this to correct existing cotracker_tracks data that was converted before
    the visibility-inversion fix was applied to ``convert_sequence``.  Each clip
    is read, its ``visibility`` array is inverted (``~visibility``), and the clip
    is rewritten atomically.  Frames/tracks/queries are left untouched.
    """
    import glob

    clip_paths = sorted(out_root.glob("*/clip_*.npz"))
    shard_id, num_shards = resolve_shard(args)
    shard_clips = clip_paths[shard_id::num_shards]
    print(f"[fix-vis] shard {shard_id}/{num_shards}: patching {len(shard_clips)}/{len(clip_paths)} clips")
    t_start = time.time()
    for ci, clip_path in enumerate(shard_clips):
        with np.load(clip_path) as d:
            arrays = {k: d[k] for k in d.files}
        arrays["visibility"] = ~arrays["visibility"]
        atomic_savez(clip_path, args.compress, **arrays)
        if (ci + 1) % 20 == 0 or ci + 1 == len(shard_clips):
            print(f"[fix-vis] ({ci + 1}/{len(shard_clips)}) {clip_path.parent.name}")
    dt = time.time() - t_start
    print(f"[fix-vis] done: {len(shard_clips)} clips patched in {dt:.1f}s")


def process_dataset(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.build_index:
        build_index(out_root)
        return

    if args.fix_visibility:
        fix_visibility_inplace(out_root, args)
        return

    src_root = Path(args.src_root)
    sequences = discover_sequences(src_root)
    if args.limit_videos is not None:
        sequences = sequences[: args.limit_videos]
    if not sequences:
        raise FileNotFoundError(f"No CT3Kubric sequences found under {src_root}")

    shard_id, num_shards = resolve_shard(args)
    shard_seqs = sequences[shard_id::num_shards]
    print(f"[info] shard {shard_id}/{num_shards}: {len(shard_seqs)}/{len(sequences)} sequences")

    _write_meta(out_root, args, _infer_native_hw(src_root / sequences[0]))

    t_start = time.time()
    n_done = n_skipped = 0
    for si, name in enumerate(shard_seqs):
        seq_dir = src_root / name
        clip_path = out_root / name / "clip_00000.npz"
        if clip_path.exists() and not args.overwrite:
            n_skipped += 1
            continue
        clip_path.parent.mkdir(parents=True, exist_ok=True)

        kwargs = convert_sequence(seq_dir, frame_stride=args.frame_stride, save_frames=args.save_frames)
        atomic_savez(clip_path, args.compress, **kwargs)
        n_done += 1
        if (si + 1) % 10 == 0 or si + 1 == len(shard_seqs):
            print(f"[info] ({si + 1}/{len(shard_seqs)}) {name}: T={kwargs['tracks'].shape[0]} "
                  f"N={kwargs['tracks'].shape[1]}")

    dt = time.time() - t_start
    print(f"[done] shard {shard_id}: {n_done} new, {n_skipped} skipped, in {dt:.1f}s")

    if not args.no_index:
        build_index(out_root)


def _write_meta(out_root: Path, args: argparse.Namespace, native_hw: Optional[Tuple[int, int]]) -> None:
    atomic_write_json(
        out_root / "meta.json",
        dict(
            dataset="CT3Kubric",
            src_root=args.src_root,
            # CT3Kubric is not resized during conversion, so `resize` reports the
            # native frame size that CoTrackerTracksDataset reads as native_hw.
            resize=list(native_hw) if native_hw is not None else None,
            frame_stride=args.frame_stride,
            clip_len=0,            # whole (sub-sampled) sequence per clip
            save_frames=args.save_frames,
            compressed=args.compress,
        ),
    )


# --------------------------------------------------------------------------- #
# Backward-compat reader alias
# --------------------------------------------------------------------------- #
# CT3Kubric is now served by the shared, config-driven CoTracker reader (no
# bespoke reader). Keep the historical name importable so old code/notebooks
# (``from ct3kubric_data_prep import CT3KubricTracksDataset``) keep working --
# point it at the converted ``out_root``.
from dataset.cotracker import CoTrackerTracksDataset as CT3KubricTracksDataset  # noqa: E402,F401


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_root", default="DATA/CT3Kubric", help="raw CT3Kubric tree to convert")
    p.add_argument("--out_root", default="DATA/CT3Kubric/cotracker_tracks", help="output root folder")

    p.add_argument("--frame_stride", type=int, default=1, help="keep every Nth stored frame")
    p.add_argument("--limit_videos", type=int, default=None, help="convert only the first K sequences (debug)")

    p.add_argument("--num_shards", type=int, default=None,
                   help="total parallel jobs (default: $SLURM_ARRAY_TASK_COUNT or 1)")
    p.add_argument("--shard_id", type=int, default=None,
                   help="this job's index in [0, num_shards) (default: $SLURM_ARRAY_TASK_ID or 0)")
    p.add_argument("--build_index", action="store_true",
                   help="only (re)build index.json from existing clips, then exit")
    p.add_argument("--fix_visibility", action="store_true",
                   help="patch existing .npz files by inverting their visibility arrays "
                        "(one-time fix for data converted before the visibility-inversion bugfix)")
    p.add_argument("--no_index", action="store_true",
                   help="do not rebuild index.json after processing (build it later with --build_index)")

    p.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                   help="store only tracks/visibility/queries (no embedded frames)")
    p.add_argument("--no_compress", dest="compress", action="store_false",
                   help="use uncompressed .npz (faster to write/read, much larger)")
    p.add_argument("--overwrite", action="store_true", help="reconvert sequences that already exist")
    p.set_defaults(save_frames=True, compress=True)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    process_dataset(args)


if __name__ == "__main__":
    sys.exit(main())
