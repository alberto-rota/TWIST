#!/usr/bin/env python
"""vlsurgpt_data_prep.py

Convert the **VLsurgPT** dataset (sparse vision-language keyframe annotations +
per-sequence ``.mp4`` videos) into the shared ``index.json`` + per-clip ``.npz``
layout that TWIST's :class:`dataset.cotracker.CoTrackerTracksDataset` reads --
exactly like every other dataset in this project. These are ground-truth point
tracks, so (as with ``endotapp_gt_prep.py`` / ``surgicalmotion_data_prep.py``)
no CoTracker tracking is run here; this is a pure format repack + uniform resize.

VLsurgPT annotates a **sparse set of keyframes** per sequence (typically one
every ~30 video frames, ~1 s apart, plus the final frame), so -- like
``endotapp_gt_prep.py`` -- we treat **the annotated keyframes as the stored
clip's frames**: the corresponding images are decoded from the ``.mp4`` and laid
down as consecutive frames, with ``tracks`` / ``visibility`` taken straight from
the GT labels (``null`` = occluded / out-of-view). The result is a fully dense
(per stored-frame) clip the shared reader consumes with no special-casing. As
the annotation is sparse, VLsurgPT is registered as an **eval-only** benchmark.

Source layout (as downloaded / extracted)::

    DATA/VLsurgPT/
        export_tissue_new/<grp>/left/<seq>/        grp in {0..4}, seq = seqNNN
            Annotation/labels.json   {"<frame_idx>": [[x,y] | null, ...], ...}
            Annotation/texts.json    parallel vision-language status (unused here)
            frames/<...>ms-<...>ms-visible.mp4
        export_instrument_new/<grp>/left/<seq>/    same structure (grp = 0)
            ...

``labels.json`` keys are integer frame indices into the ``.mp4``; each value is a
length-``N`` list (one entry per tracked point), ``[x, y]`` pixel or ``null`` when
the point is occluded / out of view at that keyframe. ``N`` is constant within a
sequence (varies between sequences, ~2-17). The parallel ``texts.json`` carries
the vision-language ``location`` / ``status`` / instrument metadata; TWIST tracks
geometrically and does not consume it, so it is **dropped** in this repack.

The two subtrees are **merged into one dataset**. Each sequence becomes one clip
whose ``video`` id is tagged by source + group so ids stay unique across the
otherwise-repeating ``seqNNN`` names and the tissue/instrument split is legible::

    tissue_0_seq000, tissue_4_seq012, instrument_0_seq001, ...

Conversion details (the bits that are easy to get wrong):

* **keyframes -> frames.** Only the annotated keyframes are decoded (read
  sequentially from the ``.mp4`` so frame indexing is exact, not seek-approximate)
  and stored as consecutive frames. ``T = #keyframes`` for that sequence.
* **BGR -> RGB.** OpenCV decodes BGR; we flip to RGB so stored frames match every
  other dataset.
* **visibility.** ``vis = (label entry is not null)``. Occluded coordinates are
  carried from the nearest visible keyframe (forward- then backward-fill) so the
  stored ``(x, y)`` is always finite and plausible -- never ``NaN`` -- while
  ``visibility`` stays ``False`` there.
* **uniform resize.** VLsurgPT ships many native sizes (1920x1080, 1280x760,
  1150x700, 720x480, ...). The shared reader infers one frame size per dataset
  root, so every clip is resized to a single ``--resize`` (H, W) recorded in
  ``meta.json``; track coordinates are scaled to match.

Target (shared CoTracker layout)::

    out_root/
        index.json                 # global manifest, one entry per sequence/clip
        meta.json                  # this run's configuration
        <video-id>/clip_00000.npz  # frames, tracks, visibility, queries

Each ``clip_00000.npz`` contains::

    frames      uint8   (T, H, W, 3)   resized RGB keyframes
    tracks      float32 (T, N, 2)      pixel (x, y) in the resized frame
    visibility  bool    (T, N)         per-point visibility (label is not null)
    queries     float32 (N, 3)         (t_first_visible, x, y)

CPU/IO only -- no GPU, no network. Runs on the login node.

Example
-------
    python vlsurgpt_data_prep.py \\
        --src_root DATA/VLsurgPT \\
        --out_root DATA/VLsurgPT/gt_tracks \\
        --resize 540 960
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# (identical to surgicalmotion_data_prep._resize_clip)
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
# Video decode: read exactly the requested frame indices (sequential = exact)
# --------------------------------------------------------------------------- #
def _read_frames_at(mp4: Path, indices: List[int]) -> Dict[int, np.ndarray]:
    """Decode ``indices`` (RGB uint8) from ``mp4`` by a single sequential pass.

    Sequential ``cap.read()`` keeps frame indexing exact -- ``cap.set(POS_FRAMES)``
    seeking lands on the nearest keyframe with many codecs. Returns a dict
    ``frame_idx -> (H, W, 3)``; indices past the end of the video are simply absent.
    """
    import cv2

    want = sorted(set(int(i) for i in indices))
    out: Dict[int, np.ndarray] = {}
    if not want:
        return out
    last = want[-1]
    wantset = set(want)
    cap = cv2.VideoCapture(str(mp4))
    try:
        fi = 0
        while fi <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if fi in wantset:
                out[fi] = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
            fi += 1
    finally:
        cap.release()
    return out


# --------------------------------------------------------------------------- #
# Coordinate carry-fill: occluded coords <- nearest visible keyframe
# --------------------------------------------------------------------------- #
def _compute_queries(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    """``(N, 3) = (t_first_visible, x, y)`` from final-scale tracks.

    The reader recomputes queries at read time; stored here for completeness /
    standalone use (mirrors endotapp/tapvid/surgicalmotion prep).
    """
    N = tracks.shape[1]
    queries = np.zeros((N, 3), dtype=np.float32)
    for n in range(N):
        vis_frames = np.where(visibility[:, n])[0]
        t0 = int(vis_frames[0]) if len(vis_frames) > 0 else 0
        queries[n] = [t0, tracks[t0, n, 0], tracks[t0, n, 1]]
    return queries


def _carry_fill(tracks: np.ndarray, visibility: np.ndarray) -> None:
    """In-place: replace occluded ``(x, y)`` with the nearest visible value.

    Forward-fill then backward-fill, per point. Points never visible keep their
    ``(0, 0)`` placeholder (they get filtered / masked downstream). ``visibility``
    is left untouched -- only the coordinate values change.
    """
    T, N = visibility.shape
    for n in range(N):
        vis = visibility[:, n]
        if not vis.any():
            continue
        filled = vis.copy()
        last: Optional[np.ndarray] = None
        for t in range(T):  # forward fill
            if filled[t]:
                last = tracks[t, n].copy()
            elif last is not None:
                tracks[t, n] = last
                filled[t] = True
        nxt: Optional[np.ndarray] = None
        for t in range(T - 1, -1, -1):  # backward fill the leading gap
            if vis[t]:
                nxt = tracks[t, n].copy()
            elif nxt is not None and not filled[t]:
                tracks[t, n] = nxt
                filled[t] = True


# --------------------------------------------------------------------------- #
# One sequence -> shared npz dict
# --------------------------------------------------------------------------- #
def _sequence_to_npz(seq_dir: Path, save_frames: bool) -> Optional[dict]:
    """Build the shared npz dict for one VLsurgPT sequence (native resolution).

    Returns ``None`` (with a warning) for sequences whose labels/video are
    missing or unusable, so the run skips them instead of aborting.
    """
    labels_path = seq_dir / "Annotation" / "labels.json"
    mp4s = sorted((seq_dir / "frames").glob("*.mp4"))
    if not labels_path.exists() or not mp4s:
        print(f"  [skip] {seq_dir} -- missing labels.json or .mp4")
        return None

    with open(labels_path) as f:
        labels = json.load(f)
    if not labels:
        print(f"  [skip] {seq_dir} -- empty labels.json")
        return None

    keyframes = sorted((int(k) for k in labels), key=int)
    N = max(len(labels[str(k)]) for k in keyframes)

    decoded = _read_frames_at(mp4s[0], keyframes)
    keyframes = [k for k in keyframes if k in decoded]  # drop keyframes past video end
    if not keyframes:
        print(f"  [skip] {seq_dir} -- no decodable keyframes")
        return None

    T = len(keyframes)
    frames = np.stack([decoded[k] for k in keyframes], axis=0)        # (T, H, W, 3) RGB
    tracks = np.zeros((T, N, 2), dtype=np.float32)
    visibility = np.zeros((T, N), dtype=bool)
    for t, k in enumerate(keyframes):
        pts = labels[str(k)]
        for n in range(N):
            pt = pts[n] if n < len(pts) else None
            if pt is None or pt[0] is None:
                continue                                             # occluded -> stays (0,0), invisible
            tracks[t, n, 0] = float(pt[0])
            tracks[t, n, 1] = float(pt[1])
            visibility[t, n] = True
    _carry_fill(tracks, visibility)

    out = dict(tracks=tracks, visibility=visibility)
    if save_frames:
        out["frames"] = frames
    else:
        out["_frames_for_resize"] = frames  # carried only to scale coords, dropped before save
    return out


# --------------------------------------------------------------------------- #
# Enumerate sequences across both subtrees, building tagged video ids
# --------------------------------------------------------------------------- #
def _iter_sequences(src_root: Path) -> List[Tuple[str, Path]]:
    """``[(video_id, seq_dir), ...]`` for both export subtrees, ids tagged + sorted.

    ``video_id = "<kind>_<grp>_<seq>"`` (e.g. ``tissue_0_seq000``) so the merged
    dataset keeps unique ids and the tissue/instrument origin stays legible.
    """
    subtrees = {"tissue": "export_tissue_new", "instrument": "export_instrument_new"}
    seqs: List[Tuple[str, Path]] = []
    for kind, sub in subtrees.items():
        base = src_root / sub
        if not base.is_dir():
            print(f"[warn] {base} not found -- skipping {kind}")
            continue
        for grp in sorted(os.listdir(base)):
            left = base / grp / "left"
            if not left.is_dir():
                continue
            for seq in sorted(os.listdir(left)):
                seq_dir = left / seq
                if seq_dir.is_dir():
                    seqs.append((f"{kind}_{grp}_{seq}", seq_dir))
    return seqs


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def process(args: argparse.Namespace) -> None:
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    resize = tuple(args.resize)  # (H, W)

    sequences = _iter_sequences(src_root)
    if not sequences:
        raise FileNotFoundError(f"No sequences found under {src_root}")
    print(f"[info] {len(sequences)} sequences -> {out_root}  (resize={resize}, save_frames={args.save_frames})")

    index: List[dict] = []
    n_new = n_skip = 0
    for vi, (vid_id, seq_dir) in enumerate(sequences):
        clip_path = out_root / vid_id / "clip_00000.npz"

        if clip_path.exists() and not args.overwrite:
            with np.load(clip_path) as d:
                num_frames, num_points = d["tracks"].shape[:2]
            index.append(dict(video=vid_id, clip_idx=0, path=f"{vid_id}/clip_00000.npz",
                              num_frames=int(num_frames), num_points=int(num_points)))
            n_skip += 1
            continue

        npz = _sequence_to_npz(seq_dir, args.save_frames)
        if npz is None:
            continue

        frames = npz.pop("_frames_for_resize", npz.get("frames"))
        frames, npz["tracks"] = _resize_clip(frames, npz["tracks"], resize)
        if args.save_frames:
            npz["frames"] = frames
        npz["queries"] = _compute_queries(npz["tracks"], npz["visibility"])  # final (resized) scale

        T, N = npz["tracks"].shape[:2]
        if (vi + 1) % 25 == 0 or vi == 0:
            print(f"  [{vi + 1}/{len(sequences)}] {vid_id}  T={T}  N={N}")
        _atomic_savez(clip_path, compress=args.compress, **npz)
        index.append(dict(video=vid_id, clip_idx=0, path=f"{vid_id}/clip_00000.npz",
                          num_frames=int(T), num_points=int(N)))
        n_new += 1

    index.sort(key=lambda e: e["video"])
    _atomic_write_json(out_root / "index.json", index)
    _atomic_write_json(out_root / "meta.json", dict(
        source="VLsurgPT (ground-truth vision-language keyframe tracks)",
        src_root=str(src_root),
        merged_subtrees=["export_tissue_new", "export_instrument_new"],
        video_id_format="<tissue|instrument>_<grp>_<seq>",
        resize=list(resize),
        save_frames=args.save_frames,
        compressed=args.compress,
    ))
    print(f"[done] {len(index)} clips in index ({n_new} written, {n_skip} skipped) -> {out_root / 'index.json'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_root", default="DATA/VLsurgPT",
                   help="dataset root containing export_tissue_new/ and export_instrument_new/")
    p.add_argument("--out_root", default="DATA/VLsurgPT/gt_tracks", help="output root folder")
    p.add_argument("--resize", type=int, nargs=2, default=[540, 960], metavar=("H", "W"),
                   help="uniform stored frame size; all clips resized to this, coords scaled to match")
    p.add_argument("--overwrite", action="store_true", help="recompute clips that already exist")
    p.add_argument("--no_save_frames", dest="save_frames", action="store_false",
                   help="store only tracks/visibility/queries (coords still scaled to --resize)")
    p.add_argument("--no_compress", dest="compress", action="store_false",
                   help="uncompressed .npz (faster to write, larger)")
    p.set_defaults(save_frames=True, compress=True)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    process(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
