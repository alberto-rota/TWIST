"""Point-sampling strategies for dense point-tracking datasets.

A tracking clip has far more candidate points than we want to train on (e.g.
CT3Kubric stores 32768 trajectories per video, the bulk of which sit *outside*
the frame for the whole clip). :func:`select_point_indices` turns that raw pool
into exactly ``max_points`` indices that are actually trackable at the clip's
query frame, drawn with a configurable spatial strategy.

Keeping the count fixed at ``max_points`` (padding with leftovers when too few
qualify) is what lets clips batch with the default collate.

This module is dataset-agnostic: it only needs the raw ``(N, T, 2)`` tracks and
``(N, T)`` visibility plus the stored frame size, so the surgical readers added
later can reuse it unchanged.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

POINT_SAMPLE_MODES = ("even", "random", "grid", "first")


def candidate_mask(
    trajs: np.ndarray,
    vis: np.ndarray,
    query_raw_frame: int,
    frame_hw: Optional[Tuple[int, int]],
    require_visible_at_query: bool = True,
    min_visible_frames: int = 1,
) -> np.ndarray:
    """Boolean ``(N,)`` mask of points that are usable for this clip.

    A point qualifies when, at the query frame, it is inside the (stored) frame
    and -- if ``require_visible_at_query`` -- visible, and it is visible on at
    least ``min_visible_frames`` frames over the whole clip window.

    Args:
        trajs: ``(N, Tclip, 2)`` pixel tracks (x, y) for the clip window.
        vis:   ``(N, Tclip)`` per-point visibility for the clip window.
        query_raw_frame: index *into the clip window* used as the query frame.
        frame_hw: stored ``(H, W)``; if ``None`` the in-frame test is skipped.
        require_visible_at_query: also require visibility at the query frame.
        min_visible_frames: minimum number of visible frames over the clip.
    """
    n = trajs.shape[0]
    ok = np.ones(n, dtype=bool)
    if frame_hw is not None:
        h, w = frame_hw
        qx = trajs[:, query_raw_frame, 0]
        qy = trajs[:, query_raw_frame, 1]
        ok &= (qx >= 0) & (qx < w) & (qy >= 0) & (qy < h)
    if require_visible_at_query:
        ok &= vis[:, query_raw_frame].astype(bool)
    if min_visible_frames > 1:
        ok &= vis.astype(bool).sum(axis=1) >= int(min_visible_frames)
    return ok


def _even(idx: np.ndarray, k: int) -> np.ndarray:
    """``k`` evenly spaced picks from a sorted index array (good spatial spread)."""
    pick = np.linspace(0, idx.size - 1, k).round().astype(np.int64)
    return idx[pick]


def _grid(idx: np.ndarray, k: int, xy: np.ndarray) -> np.ndarray:
    """Pick the candidate nearest to each cell of a ~sqrt(k) x sqrt(k) grid.

    Spreads points uniformly across the *image plane* (rather than across the
    storage order), so a small ``max_points`` still covers the whole frame.
    ``xy`` is the candidate query-frame coordinates, shape ``(len(idx), 2)``.
    """
    side = int(np.ceil(np.sqrt(k)))
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    gx = np.linspace(x0, x1, side)
    gy = np.linspace(y0, y1, side)
    targets = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)  # (side*side, 2)
    chosen: list[int] = []
    used = np.zeros(idx.size, dtype=bool)
    for t in targets:
        if len(chosen) >= k:
            break
        d = ((xy - t) ** 2).sum(1)
        d[used] = np.inf
        j = int(np.argmin(d))
        used[j] = True
        chosen.append(j)
    return idx[np.array(chosen, dtype=np.int64)]


def select_point_indices(
    trajs: np.ndarray,
    vis: np.ndarray,
    query_raw_frame: int,
    frame_hw: Optional[Tuple[int, int]],
    max_points: Optional[int],
    mode: str = "even",
    require_visible_at_query: bool = True,
    min_visible_frames: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Pick ``max_points`` point indices for a clip.

    Candidates (see :func:`candidate_mask`) are sub-selected by ``mode``:

    * ``"even"``   evenly spaced over candidate order (deterministic spread)
    * ``"random"`` uniform random without replacement (seeded -> reproducible)
    * ``"grid"``   nearest-to-a-regular-grid on the query frame (spatial spread)
    * ``"first"``  first ``max_points`` candidates (fast; debugging)

    If fewer than ``max_points`` candidates qualify, the remainder is filled with
    non-candidate points so the returned count is always ``max_points`` (those
    fillers are typically off-frame and get masked out by visibility). Returns
    every point when ``max_points`` is ``None`` or exceeds the pool.
    """
    n_full = trajs.shape[0]
    if max_points is None or max_points >= n_full:
        return np.arange(n_full)
    if mode not in POINT_SAMPLE_MODES:
        raise ValueError(f"mode must be one of {POINT_SAMPLE_MODES}, got {mode!r}")

    ok = candidate_mask(
        trajs, vis, query_raw_frame, frame_hw,
        require_visible_at_query=require_visible_at_query,
        min_visible_frames=min_visible_frames,
    )
    cand = np.flatnonzero(ok)
    k = int(max_points)

    if cand.size >= k:
        if mode == "even":
            return _even(cand, k)
        if mode == "first":
            return cand[:k]
        if mode == "random":
            rng = np.random.default_rng(seed)
            return np.sort(rng.choice(cand, size=k, replace=False))
        if mode == "grid":
            xy = trajs[cand, query_raw_frame]  # (len(cand), 2)
            return np.sort(_grid(cand, k, xy))

    # Too few qualifying points: keep all candidates, pad with the rest.
    rest = np.flatnonzero(~ok)
    return np.concatenate([cand, rest])[:k]
