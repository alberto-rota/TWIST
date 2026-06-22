"""Collation for tracking clips.

When every clip shares the same ``(T, N)`` -- the usual case for Kubric, where
``clip_len`` and ``max_points`` are fixed -- the default ``torch`` collate
stacks them directly and you do not need anything here.

:func:`pad_collate` is the fallback for *variable-length* clips (different ``T``
or ``N`` across a batch), which the surgical readers produce when a whole
sequence is one clip. It right-pads time and points to the batch maximum and
returns the masks needed to ignore the padding.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data._utils.collate import default_collate


def is_fixed_shape(dataset) -> bool:
    """True iff every clip in ``dataset`` (or its index) shares ``(T, N)``.

    Cheap heuristic from the index: same ``clip_len`` everywhere and a fixed
    ``max_points``. Falls back to ``True`` when it cannot tell (default collate
    will then raise loudly if shapes actually differ).
    """
    index = getattr(dataset, "index", None)
    if not index:
        return True
    lens = {e.get("clip_len") for e in index if isinstance(e, dict)}
    fixed_points = getattr(dataset, "max_points", None) is not None
    return len(lens) <= 1 and fixed_points


def pad_collate(batch: List[Dict]) -> Dict:
    """Right-pad ``tracks``/``visibility``/``frames``/``depths`` to batch max.

    Adds two masks:
        time_mask   (B, T)     True for real frames
        point_mask  (B, N)     True for real points
    Non-tensor fields (``video``) are gathered into lists; ``clip_idx`` and
    ``frame_size`` are stacked.
    """
    B = len(batch)
    T = max(b["tracks"].shape[0] for b in batch)
    N = max(b["tracks"].shape[1] for b in batch)

    out: Dict = {}
    time_mask = torch.zeros(B, T, dtype=torch.bool)
    point_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        t, n = b["tracks"].shape[:2]
        time_mask[i, :t] = True
        point_mask[i, :n] = True

    def _pad_tn(x, fill=0):  # (T, N, ...) -> (T_max, N_max, ...)
        t, n = x.shape[:2]
        pad = x.new_full((T, N, *x.shape[2:]), fill)
        pad[:t, :n] = x
        return pad

    out["tracks"] = torch.stack([_pad_tn(b["tracks"]) for b in batch])
    out["visibility"] = torch.stack([_pad_tn(b["visibility"], fill=False) for b in batch])

    Nq = N
    queries = []
    for b in batch:
        q = b["queries"]
        pad = q.new_zeros((Nq, q.shape[1]))
        pad[: q.shape[0]] = q
        queries.append(pad)
    out["queries"] = torch.stack(queries)

    if "frames" in batch[0]:
        C, H, W = batch[0]["frames"].shape[1:]
        frames = []
        for b in batch:
            f = b["frames"]
            pad = f.new_zeros((T, C, H, W))
            pad[: f.shape[0]] = f
            frames.append(pad)
        out["frames"] = torch.stack(frames)
    if "depths" in batch[0]:
        H, W = batch[0]["depths"].shape[1:]
        depths = []
        for b in batch:
            d = b["depths"]
            pad = d.new_zeros((T, H, W))
            pad[: d.shape[0]] = d
            depths.append(pad)
        out["depths"] = torch.stack(depths)

    out["time_mask"] = time_mask
    out["point_mask"] = point_mask
    out["video"] = [b["video"] for b in batch]
    out["clip_idx"] = default_collate([b["clip_idx"] for b in batch])
    if "frame_size" in batch[0]:
        out["frame_size"] = torch.stack([b["frame_size"] for b in batch])
    return out
