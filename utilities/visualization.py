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


def _point_colors(tracks: np.ndarray, cmap: str) -> np.ndarray:
    """One RGBA colour per point, by initial x-position -> horizontal rainbow.

    tracks: (T, N, 2) -> returns (N, 4) in [0, 1].
    """
    import matplotlib as mpl
    from matplotlib.colors import Normalize

    x0 = tracks[0, :, 0]                              # (N,)
    norm = Normalize(vmin=float(np.nanmin(x0)), vmax=float(np.nanmax(x0)) + 1e-6)
    return mpl.colormaps[cmap](norm(x0))             # (N, 4)


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
    point_size: float = 40.0,
    show_occluded: bool = True,
):
    """Draw one frame with its points; returns the Matplotlib ``Axes``."""
    import matplotlib.pyplot as plt

    img = _frames_to_hwc_uint8(frame[None])[0]       # (H, W, 3)
    pts = _to_numpy(points).astype(np.float32)       # (N, 2)
    if colors is None:
        colors = _point_colors(pts[None], cmap)      # (N, 4)
    rgba = colors.copy()
    if visibility is not None:
        vis = _to_numpy(visibility).astype(bool)     # (N,)
        rgba[~vis, 3] = 0.25 if show_occluded else 0.0

    if ax is None:
        _, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100))
    ax.imshow(img)
    ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=rgba, edgecolors="white", linewidths=0.4)
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
    point_size: float = 20.0,
    linewidth: float = 1.5,
    max_points: Optional[int] = None,
    cmap: str = "rainbow",
    fps: int = 10,
    figsize: Optional[tuple] = None,
    dpi: int = 80,
    show_occluded: bool = True,
    title: Optional[str] = None,
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
    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=point_size,
                      edgecolors="white", linewidths=0.4)
    txt = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", ha="left",
                  color="white", fontsize=10,
                  bbox=dict(facecolor="black", alpha=0.4, pad=2, edgecolor="none"))

    def update(t: int):
        im.set_data(imgs[t])

        # markers: colour by id, alpha by visibility
        rgba = np.concatenate([base_rgb, np.ones((N, 1))], axis=1)   # (N, 4)
        rgba[~vis[t], 3] = 0.25 if show_occluded else 0.0
        scat.set_offsets(pos[t])                                     # (N, 2)
        scat.set_facecolors(rgba)

        # fading motion trails over the last `tail` frames
        s = max(0, t - tail)
        segs = [pos[s:t + 1, i, :] for i in range(N)]                # list of (<=tail+1, 2)
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
    point_size: float = 40.0,
    show_occluded: bool = True,
    draw_error: bool = True,
):
    """Overlay GT (circles) and predicted (×) points on one frame.

    Points share a per-identity colour; GT are circles, predictions are crosses,
    and a thin line connects each GT→prediction (the endpoint error). Reuses
    :func:`draw_tracks_on_frame` for the GT layer.
    """
    gt = _to_numpy(gt_points).astype(np.float32)
    pr = _to_numpy(pred_points).astype(np.float32)
    if colors is None:
        colors = _point_colors(gt[None], cmap)           # (N, 4) shared identity colours
    ax = draw_tracks_on_frame(
        frame, gt, visibility=gt_visibility, ax=ax, colors=colors,
        point_size=point_size, show_occluded=show_occluded,
    )
    if draw_error:
        for i in range(gt.shape[0]):
            ax.plot([gt[i, 0], pr[i, 0]], [gt[i, 1], pr[i, 1]],
                    color=colors[i], linewidth=0.5, alpha=0.6)
    rgba = colors.copy()
    if pred_visibility is not None:
        vis = _to_numpy(pred_visibility).astype(bool)
        rgba[~vis, 3] = 0.25 if show_occluded else 0.0
    ax.scatter(pr[:, 0], pr[:, 1], s=point_size, c=rgba, marker="x", linewidths=1.2)
    return ax


def animate_comparison(
    frames,                                  # (T,3,H,W) or (T,H,W,3)
    gt_tracks,                               # (T, N, 2)
    pred_tracks,                             # (T, N, 2)
    gt_visibility=None,                      # (T, N)
    pred_visibility=None,                    # (T, N)
    *,
    point_size: float = 40.0,
    max_points: Optional[int] = None,
    cmap: str = "rainbow",
    fps: int = 10,
    figsize: Optional[tuple] = None,
    dpi: int = 80,
    title: Optional[str] = None,
    draw_error: bool = True,
    save_path: Optional[str] = None,
):
    """Animate predicted (×) vs ground-truth (circles) tracks over a clip.

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
        )
        label = f"frame {ti + 1}/{t}  (○ GT  × pred)"
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
    point_size: float = 30.0,
    cmap: str = "rainbow",
    dpi: int = 80,
    title: Optional[str] = None,
    draw_error: bool = True,
) -> np.ndarray:
    """Render a pred-vs-GT comparison clip to a ``(T, 3, H, W)`` uint8 array.

    A non-interactive (Agg) sibling of :func:`animate_comparison` that returns the
    rendered RGB frames directly — the shape ``wandb.Video`` wants — instead of a
    ``FuncAnimation``. The engine logs the result with ``wandb.Video(arr, fps=...)``.
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
        )
        if title:
            ax.set_title(f"{title} | {ti + 1}/{t}", fontsize=8)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[..., :3]  # (Hp,Wp,3) uint8
        out.append(np.transpose(buf.copy(), (2, 0, 1)))  # (3,Hp,Wp)
    return np.stack(out, axis=0)                         # (T,3,Hp,Wp) uint8
