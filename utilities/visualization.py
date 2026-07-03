"""utilities.visualization.py

Drawing / animation utilities for the CoTracker tissue-point tracks produced by
``cholec80_data_prep.py`` and served by ``Cholec80TracksDataset``.

The main entry point is :func:`animate_tracks`, which overlays the predicted
point tracks (coloured by identity, with a fading motion trail and visibility
encoded by opacity) on top of the RGB frames and returns a Matplotlib
``FuncAnimation``. In a notebook display it with::

    from IPython.display import HTML
    HTML(animate_tracks(frames, tracks, visibility).to_jshtml())

:func:`draw_tracks_on_frame` renders a single timestep and is handy for quick
static inspection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # torch is optional for the pure-drawing path
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Array normalisation helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x) -> np.ndarray:
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _frames_to_hwc_uint8(frames) -> np.ndarray:
    """Accept (T,3,H,W) or (T,H,W,3), torch/np, float[0,1]/uint8 -> (T,H,W,3) uint8."""
    f = _to_numpy(frames)
    if f.ndim != 4:
        raise ValueError(f"frames must be 4D, got shape {f.shape}")
    if f.shape[1] == 3 and f.shape[-1] != 3:        # (T, 3, H, W)
        f = np.transpose(f, (0, 2, 3, 1))
    elif f.shape[-1] != 3 and f.shape[1] == 3:      # defensive
        f = np.transpose(f, (0, 2, 3, 1))
    if f.dtype != np.uint8:
        f = f.astype(np.float32)
        if f.max() <= 1.0 + 1e-6:                   # assume normalised [0,1]
            f = f * 255.0
        f = np.clip(f, 0, 255).astype(np.uint8)
    return f


def _present_mask(pos, vis, hw=None, eps: float = 1e-3,
                  occluded_coords_real: bool = True) -> np.ndarray:
    """``(T, N)`` bool: where each point has a *real* position to draw.

    A point queried/born mid-clip (or that has already left) stores a padding
    **sentinel** in those frames -- either the ``(0, 0)`` origin or, once a resize
    has shifted it, an off-frame coordinate -- always with ``visibility == False``.
    Drawing those snaps the marker/trail to the top-left corner. A slot counts as
    *present* when the point is **visible**, or when it is occluded **but** still
    carries a genuine in-frame position (Kubric / PointOdyssey keep real coords
    through occlusion, which we still want to show as hollow rings).

    ``occluded_coords_real`` gates that occluded-but-in-frame case. It is only true
    for datasets that keep genuine coordinates through occlusion (the
    ``HAS_OCCLUDED_GT`` readers -- Kubric / DynamicReplica / PointOdyssey). For every
    other dataset the coordinate stored while a point is invisible is a **placeholder,
    not a sentinel**: STIR repeats the start/query position on every unannotated
    interior frame (GT exists only at the first and last frame), so ``~sentinel`` is
    ``True`` there and the point would otherwise be drawn frozen at its start for the
    whole clip. Pass ``occluded_coords_real=False`` for those readers so occluded
    slots are dropped (drawn only where ``visibility`` is true), regardless of the
    stored coordinate.

    Shapes: ``pos (T, N, 2)``, ``vis (T, N)``; ``hw=(H, W)`` enables the off-frame
    sentinel test (skip it when frame size is unknown).
    """
    pos = _to_numpy(pos).astype(np.float32)                      # (T, N, 2)
    vis = _to_numpy(vis).astype(bool)                            # (T, N)
    if not occluded_coords_real:
        return vis                                               # occluded coords are placeholders -> draw only visible
    at_origin = np.abs(pos).sum(-1) < eps                        # (T, N) the (0,0) sentinel
    sentinel = at_origin
    if hw is not None:
        H, W = hw
        off = ((pos[..., 0] < 0) | (pos[..., 0] >= W)
               | (pos[..., 1] < 0) | (pos[..., 1] >= H))         # (T, N)
        sentinel = sentinel | off
    return vis | ~sentinel                                       # (T, N) drawable


def _point_colors(tracks: np.ndarray, cmap: str) -> np.ndarray:
    """One RGBA colour per point, by initial x-position -> horizontal rainbow.

    tracks: (T, N, 2) -> returns (N, 4) in [0, 1].
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    x0 = tracks[0, :, 0]                              # (N,)
    norm = Normalize(vmin=float(np.nanmin(x0)), vmax=float(np.nanmax(x0)) + 1e-6)
    return mpl.colormaps[cmap](norm(x0))             # (N, 4)


def _draw_markers(ax, points, colors, visibility=None, *, marker: str = "o",
                  size: float = 20.0, edge_lw: float = 0.5, show_occluded: bool = True):
    """Scatter one marker layer with visibility encoded by *fill*, not opacity.

    Visible points are **filled** with their identity colour; occluded points are
    drawn **edge-only** (transparent fill, coloured outline) so they read as a
    hollow ring of the same colour. ``show_occluded=False`` drops occluded points
    entirely. ``colors`` is an ``(N, 4)`` per-identity RGBA array.
    """
    pts = _to_numpy(points).astype(np.float32)        # (N, 2)
    n = pts.shape[0]
    rgb = _to_numpy(colors)[:, :3]                    # (N, 3) identity colour
    vis = (_to_numpy(visibility).astype(bool) if visibility is not None
           else np.ones(n, dtype=bool))
    face_a = vis.astype(np.float32)                   # 1 where visible, 0 (hollow) where occluded
    edge_a = np.ones(n, np.float32) if show_occluded else face_a
    face = np.concatenate([rgb, face_a[:, None]], axis=1)   # (N, 4)
    edge = np.concatenate([rgb, edge_a[:, None]], axis=1)   # (N, 4)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, marker=marker,
               facecolors=face, edgecolors=edge, linewidths=edge_lw)


# --------------------------------------------------------------------------- #
# Static single-frame rendering
# --------------------------------------------------------------------------- #
def draw_tracks_on_frame(
    frame,                                  # (3,H,W) or (H,W,3)
    points,                                 # (N, 2) x,y
    visibility=None,                        # (N,) bool
    ax=None,
    colors: Optional[np.ndarray] = None,
    cmap: str = "rainbow",
    point_size: float = 0.03,                # Percentage of image height (default 5%)
    marker: str = "o",
    show_occluded: bool = True,
):
    """Draw one frame with its points; returns the Matplotlib ``Axes``.

    Visible points are filled, occluded points are drawn edge-only (hollow ring)
    — see :func:`_draw_markers`.
    
    point_size: size of points as a fraction of image height (e.g., 0.05 = 5%).
    """
    import matplotlib.pyplot as plt

    img = _frames_to_hwc_uint8(frame[None])[0]       # (H, W, 3)
    pts = _to_numpy(points).astype(np.float32)       # (N, 2)
    if colors is None:
        colors = _point_colors(pts[None], cmap)      # (N, 4)

    if ax is None:
        _, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    # Compute marker size as a percentage of image height in points^2
    H = img.shape[0]
    # Set the marker diameter in pixels as percentage of H, then convert to points^2
    marker_diameter_pixels = point_size * H
    # Conversion to marker 'size' (points^2) for scatter:
    # 1 pixel ≈ 0.75 points (matplotlib), area in points^2
    # Use area = π*(radius_in_points)^2; marker_diameter_pixels in points ≈ marker_diameter_pixels*0.75
    marker_radius_points = 0.5 * marker_diameter_pixels * 0.75
    point_size = (marker_radius_points ** 2) * 3.14159  # π*radius^2

    ax.imshow(img)
    _draw_markers(ax, pts, colors, visibility, marker=marker,
                  size=point_size, show_occluded=show_occluded)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    return ax


# --------------------------------------------------------------------------- #
# Animation
# --------------------------------------------------------------------------- #
def animate_tracks(
    frames,                                 # (T,3,H,W) or (T,H,W,3)
    tracks,                                 # (T, N, 2) x,y in pixels
    visibility=None,                        # (T, N) bool
    *,
    tail: int = 10,
    point_size: float = 10.0,
    linewidth: float = 0.75,
    max_points: Optional[int] = None,
    cmap: str = "rainbow",
    fps: int = 10,
    figsize: Optional[tuple] = None,
    dpi: int = 80,
    show_occluded: bool = True,
    title: Optional[str] = None,
    occluded_coords_real: bool = True,
    save_path: Optional[str] = None,
):
    """Animate point tracks over a clip.

    Points are coloured by identity (initial x-position), trail a fading tail of
    length ``tail`` frames, and fade to low opacity while occluded
    (``visibility == False``). Returns a ``matplotlib.animation.FuncAnimation``;
    optionally also writes it to ``save_path`` (``.mp4`` via ffmpeg, ``.gif`` via
    Pillow).

    Shapes: frames (T,3,H,W)|(T,H,W,3), tracks (T,N,2), visibility (T,N).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection

    imgs = _frames_to_hwc_uint8(frames)              # (T, H, W, 3)
    pos = _to_numpy(tracks).astype(np.float32)       # (T, N, 2)
    T, N = pos.shape[0], pos.shape[1]
    vis = _to_numpy(visibility).astype(bool) if visibility is not None else np.ones((T, N), bool)

    # optional point subsampling for speed
    if max_points is not None and max_points < N:
        sel = np.linspace(0, N - 1, max_points).round().astype(int)  # (max_points,)
        pos, vis, N = pos[:, sel], vis[:, sel], max_points

    H, W = imgs.shape[1], imgs.shape[2]
    # NaN out padding/sentinel slots so they never anchor a marker or trail to (0,0)
    present = _present_mask(pos, vis, hw=(H, W),
                            occluded_coords_real=occluded_coords_real)  # (T, N) bool
    pos_draw = pos.copy()
    pos_draw[~present] = np.nan                                   # (T, N, 2)
    base_rgb = _point_colors(pos, cmap)[:, :3]       # (N, 3)

    if figsize is None:
        figsize = (W / 100.0, H / 100.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    im = ax.imshow(imgs[0])
    tails = LineCollection([], linewidths=linewidth, capstyle="round")
    ax.add_collection(tails)
    scat = ax.scatter(pos_draw[0, :, 0], pos_draw[0, :, 1], s=point_size, linewidths=0.5)
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                  color="white", fontsize=10,
                  bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    def update(t: int):
        im.set_data(imgs[t])

        # markers: colour by id; visible -> filled, occluded -> edge-only (hollow).
        # padding/sentinel slots are NaN in pos_draw, so scatter skips them entirely.
        face_a = vis[t].astype(np.float32)
        edge_a = np.ones(N, np.float32) if show_occluded else face_a
        face = np.concatenate([base_rgb, face_a[:, None]], axis=1)   # (N, 4)
        edge = np.concatenate([base_rgb, edge_a[:, None]], axis=1)   # (N, 4)
        scat.set_offsets(pos_draw[t])                                # (N, 2), NaN where absent
        scat.set_facecolors(face)
        scat.set_edgecolors(edge)

        # fading motion trails over the last `tail` frames; NaN sentinel vertices
        # break the polyline, so a trail never reaches back into padding (0, 0)
        s = max(0, t - tail)
        segs = [pos_draw[s:t + 1, i, :] for i in range(N)]           # list of (<=tail+1, 2)
        tail_rgba = np.concatenate([base_rgb, np.full((N, 1), 0.6)], axis=1)
        tails.set_segments(segs)
        tails.set_color(tail_rgba)

        label = f"frame {t + 1}/{T}"
        txt.set_text(f"{title} | {label}" if title else label)
        return im, scat, tails, txt

    anim = FuncAnimation(fig, update, frames=T, interval=1000.0 / fps, blit=False)
    plt.close(fig)  # prevent the static first frame from also rendering in notebooks

    if save_path is not None:
        if str(save_path).endswith(".gif"):
            from matplotlib.animation import PillowWriter

            anim.save(save_path, writer=PillowWriter(fps=fps))
        else:
            anim.save(save_path, fps=fps)
    return anim


def animate_sample(sample: dict, **kwargs):
    """Convenience wrapper for a single ``Cholec80TracksDataset`` item.

    Picks frames/tracks/visibility out of the dict and labels the clip.
    """
    title = kwargs.pop("title", None)
    if title is None and "video" in sample and "clip_idx" in sample:
        ci = sample["clip_idx"]
        ci = int(ci) if not hasattr(ci, "item") else int(ci)
        title = f"{sample['video']} clip {ci}"
    return animate_tracks(
        sample["frames"], sample["tracks"], sample.get("visibility"),
        title=title, **kwargs,
    )


# --------------------------------------------------------------------------- #
# Predicted vs. ground-truth overlay (for evaluating the world model)
# --------------------------------------------------------------------------- #
def overlay_tracks_on_frame(
    frame,                                   # (3,H,W) or (H,W,3)
    gt_points,                               # (N, 2) x,y  ground truth
    pred_points,                             # (N, 2) x,y  prediction
    gt_visibility=None,                      # (N,) bool
    pred_visibility=None,                    # (N,) bool
    ax=None,
    colors: Optional[np.ndarray] = None,
    cmap: str = "rainbow",
    point_size: float = 20.0,
    show_occluded: bool = True,
    draw_error: bool = True,
    occluded_coords_real: bool = True,
):
    """Overlay GT (triangles △) and predicted (circles ○) points on one frame.

    Points share a per-identity colour; the **prediction is a circle** and the
    **GT is a triangle**, and a thin line connects each GT→prediction (the
    endpoint error). For both layers visibility is shown by *fill*: visible points
    are filled, occluded points are edge-only (hollow).

    **GT-absent handling.** When a GT point is occluded and the reader stored it
    as the ``(0, 0)`` / off-frame *sentinel* (no real position this frame), there
    is nothing to compare against: we drop its triangle (so it never snaps to the
    top-left corner), drop its error line (so no streak runs into ``(0, 0)``), and
    instead ring the prediction with a thin hollow **white** circle — the bare
    ``O`` then reads as "tracked, but no GT here" rather than a confirmed hit.
    """
    import matplotlib.pyplot as plt

    img = _frames_to_hwc_uint8(frame[None])[0]           # (H, W, 3)
    gt = _to_numpy(gt_points).astype(np.float32)         # (N, 2)
    pr = _to_numpy(pred_points).astype(np.float32)       # (N, 2)
    if colors is None:
        colors = _point_colors(gt[None], cmap)           # (N, 4) shared identity colours
    if ax is None:
        _, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100))

    H, W = img.shape[0], img.shape[1]
    # Which GT points actually have a position to compare against this frame? A
    # point the reader pads with the (0,0)/off-frame sentinel while occluded has no
    # GT here — `_present_mask` flags the slots that carry a real position (visible,
    # or occluded-but-genuine as Kubric/PointOdyssey keep).
    gv = (_to_numpy(gt_visibility).astype(bool) if gt_visibility is not None
          else np.ones(gt.shape[0], bool))               # (N,)
    gt_present = _present_mask(gt[None], gv[None], hw=(H, W),
                               occluded_coords_real=occluded_coords_real)[0]   # (N,) bool

    # Draw the frame ONCE here, then both layers via `_draw_markers` so GT and
    # pred share the same raw scatter-area `point_size` semantics. (Routing the
    # GT layer through `draw_tracks_on_frame` reinterpreted `point_size` as a
    # fraction of image height, blowing the GT triangles up to cover the frame.)
    ax.imshow(img)
    # GT layer: triangles (filled where visible, hollow where occluded). Sentinel
    # slots are NaN'd so no triangle anchors to the (0,0) corner.
    gt_draw = gt.copy()
    gt_draw[~gt_present] = np.nan
    _draw_markers(ax, gt_draw, colors, gv, marker="^",
                  size=point_size, show_occluded=show_occluded)
    if draw_error:
        for i in range(gt.shape[0]):
            if not gt_present[i]:                         # no real GT -> no error line
                continue
            ax.plot([gt[i, 0], pr[i, 0]], [gt[i, 1], pr[i, 1]],
                    color=colors[i], linewidth=0.4, alpha=0.5)
    # prediction layer: circles (filled where visible, hollow where occluded)
    _draw_markers(ax, pr, colors, pred_visibility, marker="o",
                  size=point_size, show_occluded=show_occluded)
    # GT-absent predictions: a thin hollow white halo so the lone 'O' is unmistakably
    # "no GT to compare against" (a real position is missing for these this frame).
    missing = ~gt_present
    if missing.any():
        ax.scatter(pr[missing, 0], pr[missing, 1], s=point_size * 9.0, marker="o",
                   facecolors="none", edgecolors="white", linewidths=0.6)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.axis("off")
    return ax


def animate_comparison(
    frames,                                  # (T,3,H,W) or (T,H,W,3)
    gt_tracks,                               # (T, N, 2)
    pred_tracks,                             # (T, N, 2)
    gt_visibility=None,                      # (T, N)
    pred_visibility=None,                    # (T, N)
    *,
    point_size: float = 20.0,
    max_points: Optional[int] = None,
    cmap: str = "rainbow",
    fps: int = 10,
    figsize: Optional[tuple] = None,
    dpi: int = 80,
    title: Optional[str] = None,
    draw_error: bool = True,
    occluded_coords_real: bool = True,
    save_path: Optional[str] = None,
):
    """Animate predicted (circles ○) vs ground-truth (triangles △) tracks over a clip.

    Returns a ``matplotlib.animation.FuncAnimation`` (render with ``.to_jshtml()``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    imgs = _frames_to_hwc_uint8(frames)                  # (T,H,W,3)
    gt = _to_numpy(gt_tracks).astype(np.float32)
    pr = _to_numpy(pred_tracks).astype(np.float32)
    gv = _to_numpy(gt_visibility).astype(bool) if gt_visibility is not None else None
    pv = _to_numpy(pred_visibility).astype(bool) if pred_visibility is not None else None
    t, n = gt.shape[0], gt.shape[1]
    if max_points is not None and max_points < n:
        sel = np.linspace(0, n - 1, max_points).round().astype(int)
        gt, pr = gt[:, sel], pr[:, sel]
        gv = gv[:, sel] if gv is not None else None
        pv = pv[:, sel] if pv is not None else None
        n = max_points
    colors = _point_colors(gt, cmap)                     # shared identity colours

    h, w = imgs.shape[1], imgs.shape[2]
    if figsize is None:
        figsize = (w / 100.0, h / 100.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def update(ti: int):
        ax.clear()
        overlay_tracks_on_frame(
            imgs[ti], gt[ti], pr[ti],
            gt_visibility=gv[ti] if gv is not None else None,
            pred_visibility=pv[ti] if pv is not None else None,
            ax=ax, colors=colors, point_size=point_size, draw_error=draw_error,
            occluded_coords_real=occluded_coords_real,
        )
        label = f"frame {ti + 1}/{t}  (○ pred  △ GT)"
        ax.set_title(f"{title} | {label}" if title else label, fontsize=9)
        return ()

    anim = FuncAnimation(fig, update, frames=t, interval=1000.0 / fps, blit=False)
    plt.close(fig)
    if save_path is not None:
        if str(save_path).endswith(".gif"):
            from matplotlib.animation import PillowWriter
            anim.save(save_path, writer=PillowWriter(fps=fps))
        else:
            anim.save(save_path, fps=fps)
    return anim


def animate_sample_comparison(sample: dict, pred_tracks, pred_visibility=None, **kwargs):
    """Like :func:`animate_sample` but overlays predictions on the GT clip."""
    title = kwargs.pop("title", None)
    if title is None and "video" in sample:
        ci = sample.get("clip_idx", 0)
        ci = int(ci) if not hasattr(ci, "item") else int(ci)
        title = f"{sample['video']} clip {ci}"
    return animate_comparison(
        sample["frames"], sample["tracks"], pred_tracks,
        gt_visibility=sample.get("visibility"), pred_visibility=pred_visibility,
        title=title, **kwargs,
    )


def render_comparison_frames(
    frames,                                  # (T,3,H,W) or (T,H,W,3)
    gt_tracks,                               # (T, N, 2)
    pred_tracks,                             # (T, N, 2)
    gt_visibility=None,                      # (T, N)
    pred_visibility=None,                    # (T, N)
    *,
    max_points: Optional[int] = 64,
    point_size: float = 1.5,
    cmap: str = "rainbow",
    dpi: int = 80,
    out_size: Optional[int] = None,
    title: Optional[str] = None,
    draw_error: bool = True,
    occluded_coords_real: bool = True,
) -> np.ndarray:
    """Render a pred-vs-GT comparison clip to a ``(T, 3, H, W)`` uint8 array.

    ``occluded_coords_real`` (see :func:`_present_mask`) must be ``False`` for
    datasets whose GT is annotated only on some frames and pads the rest with a
    placeholder coordinate (STIR: GT only at first/last frame). Leaving it ``True``
    for those draws the GT triangle frozen at its start position for the whole clip.

    A non-interactive (Agg) sibling of :func:`animate_comparison` that returns the
    rendered RGB frames directly — the per-frame RGB the engine logs one frame at a
    time (paired with a ``viz/frame_number`` slider metric on W&B).

    ``out_size`` forces an exact square ``out_size x out_size`` pixel canvas
    (overrides ``dpi``); leave it ``None`` to keep the source aspect ratio at ``dpi``.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    imgs = _frames_to_hwc_uint8(frames)                  # (T,H,W,3)
    gt = _to_numpy(gt_tracks).astype(np.float32)
    pr = _to_numpy(pred_tracks).astype(np.float32)
    gv = _to_numpy(gt_visibility).astype(bool) if gt_visibility is not None else None
    pv = _to_numpy(pred_visibility).astype(bool) if pred_visibility is not None else None
    t, n = gt.shape[0], gt.shape[1]
    if max_points is not None and max_points < n:
        sel = np.linspace(0, n - 1, max_points).round().astype(int)
        gt, pr = gt[:, sel], pr[:, sel]
        gv = gv[:, sel] if gv is not None else None
        pv = pv[:, sel] if pv is not None else None
    colors = _point_colors(gt, cmap)                     # shared identity colours

    h, w = imgs.shape[1], imgs.shape[2]
    if out_size is not None:                             # exact out_size x out_size px
        fig = Figure(figsize=(1.0, 1.0), dpi=int(out_size))
    else:
        fig = Figure(figsize=(w / 100.0, h / 100.0), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])

    out = []
    for ti in range(t):
        ax.clear()
        overlay_tracks_on_frame(
            imgs[ti], gt[ti], pr[ti],
            gt_visibility=gv[ti] if gv is not None else None,
            pred_visibility=pv[ti] if pv is not None else None,
            ax=ax, colors=colors, point_size=point_size, draw_error=draw_error,
            occluded_coords_real=occluded_coords_real,
        )
        if title:
            ax.set_title(f"{title} | {ti + 1}/{t}", fontsize=8)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3]  # (Hp,Wp,3) uint8
        out.append(np.transpose(buf.copy(), (2, 0, 1)))  # (3,Hp,Wp)
    return np.stack(out, axis=0)                         # (T,3,Hp,Wp) uint8


def render_track_frames(
    frames,                                  # (T,3,H,W) or (T,H,W,3)
    tracks,                                  # (T, N, 2) x,y px
    visibility=None,                         # (T, N) bool
    *,
    max_points: Optional[int] = 64,
    point_size: float = 1.25,
    linewidth: float = 0.6,
    tail: int = 12,
    cmap: str = "rainbow",
    dpi: int = 80,
    out_size: Optional[int] = None,
    title: Optional[str] = None,
    show_occluded: bool = True,
    occluded_coords_real: bool = True,
) -> np.ndarray:
    """Render *tracks* (per-point fading motion trails) to ``(T, 3, H, W)`` uint8.

    The Agg/array sibling of :func:`animate_tracks` — used to log the **predicted
    tracks** (not just per-frame points): each point trails its last ``tail``
    positions so the motion path is visible. Frames are emitted one at a time for
    the W&B ``viz/frame_number`` slider. ``out_size`` forces an exact square canvas.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    imgs = _frames_to_hwc_uint8(frames)                  # (T,H,W,3)
    pos = _to_numpy(tracks).astype(np.float32)           # (T,N,2)
    t, n = pos.shape[0], pos.shape[1]
    vis = _to_numpy(visibility).astype(bool) if visibility is not None else np.ones((t, n), bool)
    if max_points is not None and max_points < n:
        sel = np.linspace(0, n - 1, max_points).round().astype(int)
        pos, vis, n = pos[:, sel], vis[:, sel], max_points
    base_rgb = _point_colors(pos, cmap)[:, :3]           # (N,3)
    colors = np.concatenate([base_rgb, np.ones((n, 1))], axis=1)  # (N,4) identity RGBA

    h, w = imgs.shape[1], imgs.shape[2]
    # NaN out padding/sentinel slots so markers/trails never anchor to (0,0)
    present = _present_mask(pos, vis, hw=(h, w),
                            occluded_coords_real=occluded_coords_real)  # (T, N) bool
    pos_draw = pos.copy()
    pos_draw[~present] = np.nan                          # (T, N, 2)
    if out_size is not None:
        fig = Figure(figsize=(1.0, 1.0), dpi=int(out_size))
    else:
        fig = Figure(figsize=(w / 100.0, h / 100.0), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])

    out = []
    for ti in range(t):
        ax.clear()
        ax.imshow(imgs[ti])
        ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis("off")
        # fading trails over the last `tail` frames; NaN sentinel vertices break
        # the polyline so a trail never reaches back into padding (0, 0)
        s = max(0, ti - tail)
        segs = [pos_draw[s:ti + 1, i, :] for i in range(n)]
        tails = LineCollection(segs, linewidths=linewidth, capstyle="round",
                               colors=np.concatenate([base_rgb, np.full((n, 1), 0.7)], axis=1))
        ax.add_collection(tails)
        # visible -> filled circle, occluded -> edge-only (hollow) ring;
        # sentinel/padding slots are NaN, so _draw_markers skips them entirely
        _draw_markers(ax, pos_draw[ti], colors, vis[ti], marker="o",
                      size=point_size, show_occluded=show_occluded)
        if title:
            ax.set_title(f"{title} | {ti + 1}/{t}", fontsize=8)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3]
        out.append(np.transpose(buf.copy(), (2, 0, 1)))
    return np.stack(out, axis=0)                         # (T,3,out,out) uint8
