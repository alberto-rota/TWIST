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

from typing import Dict, Optional

import torch

TAP_THRESHOLDS = (1.0, 2.0, 4.0, 8.0, 16.0)


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
) -> Dict[str, float]:
    """``eval_mask`` (B,T,N) bool, when given, is the set of **evaluation points**
    (the TAP-Vid ``evaluation_points``) and *replaces* the ``time_mask``⊗``point_mask``
    outer product. Use it to express per-point eval regions the outer product can't —
    e.g. TAP-Vid "queried first" scores only frames *after* each point's own query
    frame (see :mod:`utilities.evaluation`). Visibility is still applied on top, so
    occluded GT frames inside the region count only toward OA / Jaccard-FP, never EPE."""
    coords = coords.float(); gt_tracks = gt_tracks.float()
    gt_vis_b = gt_vis.bool()
    exist = (eval_mask.bool() if eval_mask is not None
             else _exist_mask(gt_vis, time_mask, point_mask).bool())
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
        "motion_ratio": motion_ratio,
        "stuck_frac": stuck_frac,
        **per_threshold,
    }
