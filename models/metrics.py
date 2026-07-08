"""TAP-Vid point-tracking metrics (monitoring + evaluation).

All operate on the canonical tensors and return Python floats:
  coords     (B,T,N,2)  predicted pixel tracks
  gt_tracks  (B,T,N,2)  ground-truth pixel tracks
  vis_logits (B,T,N)    predicted visibility logits  (sigmoid > 0.5 -> visible)
  gt_vis     (B,T,N)    ground-truth visibility

Standard TAP-Vid definitions:
  * ``epe``                 mean L2 endpoint error over visible GT points (px)
  * ``delta_avg``           mean over thresholds {1,2,4,8,16}px of the fraction of
                            visible GT points predicted within that threshold
  * ``occlusion_accuracy``  fraction of points whose predicted visibility matches GT
  * ``average_jaccard``     mean over thresholds of Jaccard(correct ∩ / ∪) combining
                            position accuracy and visibility agreement
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

TAP_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)

# Occlusion-length strata for the post-occlusion-recovery curve: (lo, hi) inclusive
# frame counts, hi=None for the open-ended top bin. Short blink -> long hide.
RECOVERY_LENGTH_BINS: Tuple[Tuple[int, Optional[int]], ...] = (
    (1, 1), (2, 4), (5, 8), (9, 16), (17, None),
)


def _exist_mask(gt_vis, time_mask, point_mask):
    b, t, n = gt_vis.shape
    tm = time_mask.float() if time_mask is not None else gt_vis.new_ones(b, t)
    pm = point_mask.float() if point_mask is not None else gt_vis.new_ones(b, n)
    return tm[:, :, None] * pm[:, None, :]               # (B,T,N) real entries


@torch.no_grad()
def tracking_metrics(
    coords: torch.Tensor,
    gt_tracks: torch.Tensor,
    vis_logits: torch.Tensor,
    gt_vis: torch.Tensor,
    time_mask: Optional[torch.Tensor] = None,
    point_mask: Optional[torch.Tensor] = None,
    thresholds=TAP_THRESHOLDS,
    query_frame: int = 0,
    eval_mask: Optional[torch.Tensor] = None,
    visible_only: bool = False,
) -> Dict[str, float]:
    """``eval_mask`` (B,T,N) bool, when given, is the set of **evaluation points**
    (the TAP-Vid ``evaluation_points``) and *replaces* the ``time_mask``⊗``point_mask``
    outer product. Use it to express per-point eval regions the outer product can't —
    e.g. TAP-Vid "queried first" scores only frames *after* each point's own query
    frame (see :mod:`utilities.evaluation`). Visibility is still applied on top, so
    occluded GT frames inside the region count only toward OA / Jaccard-FP, never EPE.

    ``visible_only`` restricts the evaluation region to frames where GT is **visible**
    (``exist &= gt_vis``). Set it for datasets whose visibility GT is *sparse* — present
    only on annotated frames, with every other frame marked occluded merely because it
    is *unlabelled*, not genuinely hidden (STIR: GT only at the first/last frame). Left
    off (the default, for TAP-Vid / Kubric where "occluded" is a real label) an occluded
    evaluated frame the model predicts visible is a Jaccard false-positive and an OA
    miss — correct there, but for STIR it turns every unlabelled interior frame into a
    spurious FP/miss and collapses AJ/OA to ~0. It does **not** change EPE or δ (both
    already sum over ``gt_vis & exist`` only)."""
    coords = coords.float(); gt_tracks = gt_tracks.float()
    gt_vis_b = gt_vis.bool()
    exist = (eval_mask.bool() if eval_mask is not None
             else _exist_mask(gt_vis, time_mask, point_mask).bool())
    if visible_only:
        exist = exist & gt_vis_b               # sparse-GT datasets: score only annotated (visible) frames
    pred_vis = (torch.sigmoid(vis_logits) > 0.5)

    dist = torch.linalg.norm(coords - gt_tracks, dim=-1)       # (B,T,N) px
    vis_eval = gt_vis_b & exist                                # visible & real

    epe = dist[vis_eval].mean().item() if vis_eval.any() else float("nan")

    # --- motion diagnostics (the unambiguous static-collapse alarm) ---
    # epe ~= GT-displacement is consistent with BOTH random and frozen predictions;
    # motion_ratio (pred travel / GT travel from the query frame) and stuck_frac
    # (moving GT points the model leaves pinned) tell them apart at a glance.
    qf = max(0, min(int(query_frame), coords.shape[1] - 1))
    pred_disp = torch.linalg.norm(coords - coords[:, qf:qf + 1], dim=-1)      # (B,T,N)
    gt_disp = torch.linalg.norm(gt_tracks - gt_tracks[:, qf:qf + 1], dim=-1)  # (B,T,N)
    if vis_eval.any():
        pred_travel = pred_disp[vis_eval].mean().item()
        gt_travel = gt_disp[vis_eval].mean().item()
        motion_ratio = pred_travel / gt_travel if gt_travel > 1e-6 else float("nan")
        moving = vis_eval & (gt_disp > 5.0)
        stuck_frac = ((pred_disp < 2.0) & moving).float().sum().item() / moving.float().sum().clamp_min(1).item()
    else:
        motion_ratio = float("nan"); stuck_frac = float("nan")
        pred_travel = float("nan"); gt_travel = float("nan")

    deltas, jaccards = [], []
    per_threshold: Dict[str, float] = {}
    for thr in thresholds:
        within = dist < thr
        # <delta: fraction of visible GT points within threshold
        if vis_eval.any():
            d = (within & vis_eval).float().sum().item() / vis_eval.float().sum().item()
            deltas.append(d)
            per_threshold[f"delta_{thr:g}px"] = d
        # average jaccard: position + visibility agreement
        tp = (vis_eval & pred_vis & within).float().sum().item()
        fp = (exist & pred_vis & ~(vis_eval & within)).float().sum().item()
        fn = (vis_eval & ~(pred_vis & within)).float().sum().item()
        denom = tp + fp + fn
        jaccards.append(tp / denom if denom > 0 else float("nan"))

    occ_acc = (pred_vis.eq(gt_vis_b) & exist).float().sum().item() / exist.float().sum().clamp_min(1).item()

    def _nanmean(xs):
        xs = [x for x in xs if x == x]  # drop nan
        return sum(xs) / len(xs) if xs else float("nan")

    return {
        "epe": epe,
        "delta_avg": _nanmean(deltas),
        "occlusion_accuracy": occ_acc,
        "average_jaccard": _nanmean(jaccards),
        "motion_ratio": motion_ratio,          # per-batch ratio-of-means (unstable; see below)
        # pred/GT travel are returned separately so callers can POOL across batches
        # (Σpred/Σgt) instead of averaging the per-batch ratio — a single near-static
        # clip drives gt_travel→0 and blows the per-batch ratio into the hundreds,
        # which the arithmetic mean-of-ratios then inflates (the ~40 artifact).
        "pred_travel": pred_travel,
        "gt_travel": gt_travel,
        "stuck_frac": stuck_frac,
        **per_threshold,
    }


# =========================================================================== #
# Post-Occlusion Recovery (POR)
# =========================================================================== #
# How well does the model *re-acquire* a point once it reappears after being
# occluded?  Distinct from ``occlusion_accuracy`` (a visibility classifier) and
# from "delta on occluded frames" (tracking *while* hidden): POR scores the
# localization on the frames *after* an occlusion ends. Two reductions per
# event — snap-back (the first re-emergence frame) and a sustained window — each
# in an EPE form (px, lower better) and a delta form (fraction within
# {1,2,4,8,16}px, higher better), aggregated as a length-weighted mean with
# weight ``w(L)=L`` (longer occlusions count proportionally more) and always
# strata'd by occlusion length. On synthetic full-GT data (``has_occluded_gt``)
# it additionally scores the through-occlusion track + the drift-vs-time curve.


def _bin_label(lo: int, hi: Optional[int]) -> str:
    return f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}-{hi}")


def _length_bin(length: int, bins) -> str:
    for lo, hi in bins:
        if length >= lo and (hi is None or length <= hi):
            return _bin_label(lo, hi)
    return str(length)


@torch.no_grad()
def recovery_metrics(
    coords: torch.Tensor,
    gt_tracks: torch.Tensor,
    vis_logits: torch.Tensor,
    gt_vis: torch.Tensor,
    *,
    point_mask: Optional[torch.Tensor] = None,
    has_occluded_gt: bool = False,
    window: int = 8,
    thresholds=TAP_THRESHOLDS,
    length_bins=RECOVERY_LENGTH_BINS,
    drift_horizon: int = 16,
) -> Dict[str, float]:
    """Post-occlusion-recovery **sufficient statistics** for one batch.

    Inputs are the canonical eval tensors (same as :func:`tracking_metrics`):
    ``coords``/``gt_tracks`` ``(B,T,N,2)`` px, ``vis_logits``/``gt_vis``
    ``(B,T,N)`` (``vis_logits`` accepted for symmetry / future visibility-gated
    variants; the position metrics here do not use it). Predicted ``coords`` must
    be populated for **all** frames (they are, under the TAP-Vid "first" forward
    in :mod:`utilities.evaluation`).

    For every point it finds each **occlusion event** — a maximal run of
    GT-occluded frames ``[s,f]`` (length ``L=f-s+1``) that begins *after* the
    point's first-visible frame (so there was a live track to lose; the leading
    pre-query occlusion is ignored). If the event **re-emerges** (a visible frame
    ``r=f+1`` exists) it scores recovery:

      * **snap-back**: error at ``r`` (W=1),
      * **sustained**: mean error over the contiguous visible run from ``r``,
        capped at ``window`` frames,

    each as EPE (px) and delta (fraction within ``thresholds``), accumulated with
    weight ``w(L)=L``. With ``has_occluded_gt`` it also scores the **through-
    occlusion** span ``[s,f]`` (EPE/delta) and the **drift** curve (mean error vs
    frames-since-onset). Leave ``has_occluded_gt`` False on placeholder-GT
    datasets (occluded coords stored ``(0,0)``): recovery touches visible frames
    only, so the placeholder is never read.

    Returns a flat ``{key: float}`` of sums + weights — *not* finished means.
    Events are per-point and vary in count per clip, so accumulate across batches
    with :func:`merge_recovery_stats` and reduce once with
    :func:`finalize_recovery`.
    """
    coords = coords.float(); gt_tracks = gt_tracks.float()
    B, T, N = gt_vis.shape
    dist = torch.linalg.norm(coords - gt_tracks, dim=-1)                  # (B,T,N) px
    thr = dist.new_tensor(tuple(thresholds))
    dpf = (dist.unsqueeze(-1) < thr).float().mean(-1)                     # (B,T,N) per-frame delta

    vis_np = gt_vis.bool().cpu().numpy()
    dist_np = dist.cpu().numpy()
    dpf_np = dpf.cpu().numpy()
    pm_np = (point_mask.bool().cpu().numpy() if point_mask is not None
             else np.ones((B, N), dtype=bool))

    acc: Dict[str, float] = {}
    def add(k: str, v) -> None:
        acc[k] = acc.get(k, 0.0) + float(v)

    for b in range(B):
        for n in range(N):
            if not pm_np[b, n]:
                continue
            vis_bn = vis_np[b, :, n]
            if not vis_bn.any():
                continue
            qf = int(vis_bn.argmax())                 # first-visible (query) frame
            occ = (~vis_bn).astype(np.int8)
            occ[: qf + 1] = 0                         # ignore pre-query / query-frame occlusion
            if occ.sum() == 0:
                continue
            dd = np.diff(np.concatenate(([0], occ, [0])))
            starts = np.where(dd == 1)[0]
            ends = np.where(dd == -1)[0] - 1          # inclusive last occluded frame
            for s, f in zip(starts.tolist(), ends.tolist()):
                L = f - s + 1
                w = float(L)
                lab = _length_bin(L, length_bins)

                if has_occluded_gt:                   # Case B: track scored *through* occlusion
                    e_tho = float(dist_np[b, s:f + 1, n].mean())
                    d_tho = float(dpf_np[b, s:f + 1, n].mean())
                    add("w_tho", w); add("n_tho", 1)
                    add("wse_tho", w * e_tho); add("wsd_tho", w * d_tho)
                    add(f"thobin_w|{lab}", w); add(f"thobin_n|{lab}", 1)
                    add(f"thobin_wse|{lab}", w * e_tho); add(f"thobin_wsd|{lab}", w * d_tho)
                    for k in range(min(L, drift_horizon)):
                        add(f"drift_sum|{k}", dist_np[b, s + k, n]); add(f"drift_cnt|{k}", 1)

                if f >= T - 1:                        # runs to clip end -> never re-emerges
                    continue
                r = f + 1                             # re-emergence frame (visible by construction)
                end_win = r
                while (end_win + 1 <= T - 1) and (end_win + 1 < r + window) and vis_bn[end_win + 1]:
                    end_win += 1                      # extend over contiguous visible, capped at W
                e_snap = float(dist_np[b, r, n]); d_snap = float(dpf_np[b, r, n])
                e_w8 = float(dist_np[b, r:end_win + 1, n].mean())
                d_w8 = float(dpf_np[b, r:end_win + 1, n].mean())
                add("w_por", w); add("n_por", 1)
                add("wse_snap", w * e_snap); add("wsd_snap", w * d_snap)
                add("wse_w8", w * e_w8); add("wsd_w8", w * d_w8)
                add(f"bin_w|{lab}", w); add(f"bin_n|{lab}", 1)
                add(f"bin_wse_snap|{lab}", w * e_snap); add(f"bin_wsd_snap|{lab}", w * d_snap)
                add(f"bin_wse_w8|{lab}", w * e_w8); add(f"bin_wsd_w8|{lab}", w * d_w8)
    return acc


def merge_recovery_stats(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """Sum two :func:`recovery_metrics` stat dicts key-wise (batch accumulation)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def finalize_recovery(
    stats: Dict[str, float],
    *,
    length_bins=RECOVERY_LENGTH_BINS,
    drift_horizon: int = 16,
) -> Dict[str, object]:
    """Reduce accumulated :func:`recovery_metrics` stats to reported numbers.

    Headline scalars (length-weighted, ``w(L)=L``):

      * ``por_epe_snap`` / ``por_delta_snap``  — snap-back (first re-emergence frame)
      * ``por_epe_w8``   / ``por_delta_w8``    — sustained window (<= W frames)
      * ``n_recovery_events``

    Plus ``by_length`` (the recovery-vs-occlusion-length curve). When through-
    occlusion stats are present (synthetic full-GT eval): ``tho_epe`` /
    ``tho_delta``, ``tho_by_length``, and ``drift_epe`` (error vs frames-since-
    onset). Scalars are NaN when no events were seen.
    """
    def ratio(num: str, den: str) -> float:
        d = stats.get(den, 0.0)
        return stats.get(num, 0.0) / d if d > 0 else float("nan")

    out: Dict[str, object] = {
        "por_epe_snap":   ratio("wse_snap", "w_por"),
        "por_delta_snap": ratio("wsd_snap", "w_por"),
        "por_epe_w8":     ratio("wse_w8", "w_por"),
        "por_delta_w8":   ratio("wsd_w8", "w_por"),
        "n_recovery_events": int(stats.get("n_por", 0.0)),
    }

    by_len: Dict[str, Dict[str, float]] = {}
    for lo, hi in length_bins:
        lab = _bin_label(lo, hi)
        bw = stats.get(f"bin_w|{lab}", 0.0)
        if bw > 0:
            by_len[lab] = {
                "epe_snap":   stats.get(f"bin_wse_snap|{lab}", 0.0) / bw,
                "delta_snap": stats.get(f"bin_wsd_snap|{lab}", 0.0) / bw,
                "epe_w8":     stats.get(f"bin_wse_w8|{lab}", 0.0) / bw,
                "delta_w8":   stats.get(f"bin_wsd_w8|{lab}", 0.0) / bw,
                "n":          int(stats.get(f"bin_n|{lab}", 0.0)),
            }
    out["by_length"] = by_len

    if stats.get("w_tho", 0.0) > 0:                   # Case B present
        out["tho_epe"] = ratio("wse_tho", "w_tho")
        out["tho_delta"] = ratio("wsd_tho", "w_tho")
        out["n_through_occlusion_events"] = int(stats.get("n_tho", 0.0))
        tho_len: Dict[str, Dict[str, float]] = {}
        for lo, hi in length_bins:
            lab = _bin_label(lo, hi)
            bw = stats.get(f"thobin_w|{lab}", 0.0)
            if bw > 0:
                tho_len[lab] = {
                    "epe":   stats.get(f"thobin_wse|{lab}", 0.0) / bw,
                    "delta": stats.get(f"thobin_wsd|{lab}", 0.0) / bw,
                    "n":     int(stats.get(f"thobin_n|{lab}", 0.0)),
                }
        out["tho_by_length"] = tho_len
        drift: Dict[int, float] = {}
        for k in range(drift_horizon):
            c = stats.get(f"drift_cnt|{k}", 0.0)
            if c > 0:
                drift[k] = stats.get(f"drift_sum|{k}", 0.0) / c
        out["drift_epe"] = drift
    return out
