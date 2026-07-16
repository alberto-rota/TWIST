"""Training objective for the TWIST world model.

``TrackerLoss`` combines the terms below, all masked to valid (non-padded)
points and computed in **normalized** coordinate units (resolution-invariant);
endpoint error is reported in pixels for monitoring:

* **Position Huber** (dominant) on the corrected coords, the dynamics-prior
  mean, and the multi-step frame-free rollout. All three are masked with
  ``pos_valid`` — visible frames PLUS occluded-but-in-frame frames on datasets
  that store real GT through occlusion (``HAS_OCCLUDED_GT``: Kubric /
  PointOdyssey / DynamicReplica). That through-occlusion signal (~28% of Kubric
  frames) is what trains the coast-and-reacquire behaviour the project targets;
  ``use_occluded_gt=False`` falls back to visible-only (the historical mask,
  kept as an A/B switch).
* **Decoupled aleatoric uncertainty** (``unc_weight``) — heteroscedastic NLL
  for BOTH the posterior and the prior log-variance heads against a *detached*
  Huber error, so calibration training cannot perturb the position gradient
  (nor be gamed by over-confidence). These calibrated variances feed the
  observation's gate/search heads — they are load-bearing, not decorative.
* **Visibility BCE** on the visibility logits, masked to point-steps where the
  observation actually ran (``observed``): on simulated-occlusion (dropout)
  steps the window content contradicts the GT label, so they are excluded.
* **Correlation cross-entropy** (``ce_weight``) — supervises the local
  cost-volume (and, when the coarse re-acquisition stage is on, the global
  correlation map, ``global_ce_weight``) directly with the GT cell. This
  trains the matcher *independently of the Kalman gate* (a low gate attenuates
  the soft-argmax position gradient, but never the CE) and is the direct
  sub-cell precision signal. Visible frames only — appearance cannot localize
  an occluded point.
* **KL(posterior ‖ prior)** with Dreamer-style balancing + free bits. Known to
  pin at the free-bits floor when both heads are GT-supervised — kept for
  monitoring/back-compat (``kl_raw`` logs the pre-clamp value); new configs run
  ``KL_WEIGHT: 0``.

``forward(outputs, batch) -> (total, parts)`` where ``parts`` is a dict of
detached scalars for logging — matching ``main.py``'s ``(scalar, dict)``
contract.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import normalize_coords


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over entries where ``mask`` is truthy (safe when empty)."""
    mask = mask.to(x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def kl_diag_gauss(mu_q, lv_q, mu_p, lv_p) -> torch.Tensor:
    """KL( N(mu_q, e^lv_q) ‖ N(mu_p, e^lv_p) ) summed over the last (coord) axis."""
    term = (lv_p - lv_q) + (torch.exp(lv_q - lv_p) + (mu_q - mu_p) ** 2 * torch.exp(-lv_p)) - 1.0
    return 0.5 * term.sum(dim=-1)


class TrackerLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 10.0,    # position dominates (its per-point value is small)
        vis_weight: float = 0.5,
        kl_weight: float = 0.05,     # the engine anneals this up over training
        kl_free_bits: float = 0.5,
        kl_balance_alpha: float = 0.8,
        huber_delta: float = 0.2,    # normalized units. 0.2 (~51px @512) is ~pure-L2 in the
                                     # operating range; 0.01-0.02 (~2.5-5px) makes ordinary
                                     # errors L1-like — the tight-threshold precision lever.
        unc_weight: float = 0.0,     # decoupled heteroscedastic NLL (post + prior logvars)
        prior_weight: float = 0.5,   # direct GT supervision of the dynamics-prior mean
        rollout_weight: float = 0.0, # multi-step frame-free rollout vs GT (0 disables)
        ce_weight: float = 0.0,      # local cost-volume cross-entropy (0 disables)
        global_ce_weight: Optional[float] = None,  # global-map CE; None -> follow ce_weight
        use_occluded_gt: bool = True,  # position terms use pos_valid (occluded GT) when present
    ) -> None:
        super().__init__()
        self.pos_weight = pos_weight
        self.vis_weight = vis_weight
        self.kl_weight = kl_weight
        self.kl_free_bits = kl_free_bits
        self.kl_balance_alpha = kl_balance_alpha
        self.huber_delta = huber_delta
        self.unc_weight = unc_weight
        self.prior_weight = prior_weight
        self.rollout_weight = rollout_weight
        self.ce_weight = ce_weight
        self.global_ce_weight = ce_weight if global_ce_weight is None else global_ce_weight
        self.use_occluded_gt = use_occluded_gt

    # ------------------------------------------------------------------ #
    # Correlation-map cross-entropy helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _local_ce(outputs, gt_tracks, vis_valid) -> torch.Tensor:
        """CE of the local cost-volume vs the window cell nearest the GT offset.

        Masked to visible-and-real points whose GT lies inside the window and
        whose correlation was actually computed this step (``corr_valid``).
        """
        logits = outputs["corr_logits"]                       # (B,T,N,k*k)
        center = outputs["win_center"]                        # (B,T,N,2) px
        k, radius = outputs["corr_grid"]                      # ints/float
        d = gt_tracks - center                                # (B,T,N,2) px offsets
        inside = (d.abs() <= radius).all(dim=-1)              # (B,T,N)
        spacing = (2.0 * radius) / max(k - 1, 1)
        cell = torch.round((d + radius) / spacing).long().clamp_(0, k - 1)  # (B,T,N,2)
        target = cell[..., 1] * k + cell[..., 0]              # iy*k + ix
        mask = (vis_valid.bool() & inside & outputs["corr_valid"].bool()).reshape(-1)
        if not bool(mask.any()):
            return logits.new_zeros(())
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask].float(),
                               target.reshape(-1)[mask])

    @staticmethod
    def _global_ce(outputs, gt_tracks, vis_valid, hw) -> torch.Tensor:
        """CE of the global correlation map vs the feature cell holding the GT.

        Trains template re-detection anywhere in frame — the coarse
        re-acquisition stage's direct supervision. Visible frames only.
        """
        logits = outputs["gcorr_logits"]                      # (B,T,N,Hf*Wf)
        hf, wf = outputs["feat_hw"]
        h, w = hw
        x, y = gt_tracks[..., 0], gt_tracks[..., 1]
        inframe = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
        gx = torch.round(x * (wf - 1) / max(w - 1, 1)).long().clamp_(0, wf - 1)
        gy = torch.round(y * (hf - 1) / max(h - 1, 1)).long().clamp_(0, hf - 1)
        target = gy * wf + gx                                 # (B,T,N)
        mask = (vis_valid.bool() & inframe & outputs["corr_valid"].bool()).reshape(-1)
        if not bool(mask.any()):
            return logits.new_zeros(())
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1])[mask].float(),
                               target.reshape(-1)[mask])

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        coords = outputs["coords"]                      # (B,T,N,2) px
        prior_mean = outputs["prior_mean"]              # (B,T,N,2) px
        coord_logvar = outputs["coord_logvar"]          # (B,T,N,2)
        prior_logvar = outputs["prior_logvar"]          # (B,T,N,2)
        vis_logits = outputs["vis_logits"]              # (B,T,N)
        hw = outputs["frame_hw"]

        gt_tracks = batch["tracks"].to(coords.dtype)    # (B,T,N,2)
        gt_vis = batch["visibility"].to(coords.dtype)   # (B,T,N)
        b, t, n = gt_vis.shape
        time_mask = batch.get("time_mask")
        point_mask = batch.get("point_mask")
        tm = time_mask.to(coords.dtype) if time_mask is not None else coords.new_ones(b, t)
        pm = point_mask.to(coords.dtype) if point_mask is not None else coords.new_ones(b, n)
        exist = tm[:, :, None] * pm[:, None, :]         # (B,T,N) real (non-padded) entries

        # --- supervision masks ---
        # vis_valid: strictly visible (CE targets, comparable EPE monitoring).
        # valid: the POSITION mask — includes occluded-but-in-frame frames when the
        # dataset ships real occluded GT (pos_valid from the reader), so the coast
        # -through-occlusion behaviour is trained directly, not just at re-emergence.
        vis_valid = gt_vis * exist
        pos_valid_b = batch.get("pos_valid")
        if self.use_occluded_gt and pos_valid_b is not None:
            valid = pos_valid_b.to(coords.dtype) * exist
        else:
            valid = vis_valid
        # observed: 1 where the observation ran AND was applied for that point-step
        # (the model's own record). Simulated-occlusion (dropout) steps are 0: their
        # window content contradicts the GT visibility label, so the BCE skips them.
        observed = outputs.get("observed")
        bce_mask = exist * observed.to(coords.dtype) if observed is not None else exist

        # --- normalized coordinates ---
        mu_q = normalize_coords(coords, hw)
        mu_p = normalize_coords(prior_mean, hw)
        gt_n = normalize_coords(gt_tracks, hw)

        # --- position loss (direct Huber) + decoupled aleatoric uncertainty ---
        # A plain Huber drives the position gradient (and EPE) uniformly. The
        # heteroscedastic terms train the log-variances to calibrate to the error
        # WITHOUT a position gradient (huber detached) -- avoiding the
        # geometric-mean pathology where a raw NLL is gamed by over-confidence.
        huber = F.huber_loss(mu_q, gt_n, reduction="none", delta=self.huber_delta).sum(-1)  # (B,T,N)
        pos_reg = masked_mean(huber, valid)
        log_b = 0.5 * coord_logvar.mean(-1)             # (B,T,N) posterior log-scale
        unc = huber.detach() * torch.exp(-log_b) + log_b
        unc_reg = masked_mean(unc, valid)               # heteroscedastic NLL (can be < 0)
        # Direct GT supervision of the dynamics-prior mean (see git history: the
        # prior once got its position gradient only through a floored KL and was
        # free to saturate). With pos_valid this now anchors the prior THROUGH
        # occlusion — exactly the frames where it is the only position source.
        huber_prior = F.huber_loss(mu_p, gt_n, reduction="none", delta=self.huber_delta).sum(-1)
        prior_reg = masked_mean(huber_prior, valid)
        log_bp = 0.5 * prior_logvar.mean(-1)            # (B,T,N) prior log-scale
        unc_prior = huber_prior.detach() * torch.exp(-log_bp) + log_bp
        unc_prior_reg = masked_mean(unc_prior, valid)
        pos_loss = (pos_reg + self.prior_weight * prior_reg
                    + self.unc_weight * (unc_reg + unc_prior_reg))

        # --- visibility BCE (real points whose observation actually ran) ---
        bce = F.binary_cross_entropy_with_logits(vis_logits, gt_vis, reduction="none")
        vis_loss = masked_mean(bce, bce_mask)

        # --- KL(posterior ‖ prior), balanced + free bits (vestigial; see docstring) ---
        a = self.kl_balance_alpha
        kl_sgq_p = kl_diag_gauss(mu_q.detach(), coord_logvar.detach(), mu_p, prior_logvar)
        kl_q_sgp = kl_diag_gauss(mu_q, coord_logvar, mu_p.detach(), prior_logvar.detach())
        kl = a * kl_sgq_p + (1.0 - a) * kl_q_sgp        # (B,T,N)
        kl_raw = masked_mean(kl.detach(), valid)        # PRE-clamp (the observable one)
        if self.kl_free_bits > 0:
            kl = kl.clamp_min(self.kl_free_bits)
        kl_loss = masked_mean(kl, valid)

        # --- correlation cross-entropy (matcher supervision, gate-independent) ---
        if self.ce_weight > 0 and "corr_logits" in outputs:
            ce_reg = self._local_ce(outputs, gt_tracks, vis_valid)
        else:
            ce_reg = coords.new_zeros(())
        if self.global_ce_weight > 0 and "gcorr_logits" in outputs:
            gce_reg = self._global_ce(outputs, gt_tracks, vis_valid, hw)
        else:
            gce_reg = coords.new_zeros(())

        # --- multi-step frame-free rollout loss (the dynamics-prior fix) ---
        # ``rollout_coords`` (B,H,N,2) are the prior rolled out from its OWN state
        # for H steps with no observation. With pos_valid the forecast is now also
        # supervised THROUGH occluded spans (full-GT sets) — the exact protocol
        # val/epoch/rollout/* measures. ``rollout_epe`` is computed whenever the
        # block is present (independent of the weight) so it stays a valid monitor.
        if "rollout_coords" in outputs:
            s = int(outputs["rollout_start"])
            rc = outputs["rollout_coords"]                          # (B,H,N,2) px
            h_roll = rc.shape[1]
            mu_roll = normalize_coords(rc, hw)
            hub_roll = F.huber_loss(mu_roll, gt_n[:, s:s + h_roll], reduction="none",
                                    delta=self.huber_delta).sum(-1)
            valid_roll = valid[:, s:s + h_roll]
            rollout_reg = masked_mean(hub_roll, valid_roll)
            with torch.no_grad():
                rollout_epe = masked_mean(
                    torch.linalg.norm(rc - gt_tracks[:, s:s + h_roll], dim=-1), valid_roll)
        else:
            rollout_reg = coords.new_zeros(())
            rollout_epe = coords.new_zeros(())

        total = (self.pos_weight * pos_loss + self.vis_weight * vis_loss
                 + self.kl_weight * kl_loss + self.rollout_weight * rollout_reg
                 + self.ce_weight * ce_reg + self.global_ce_weight * gce_reg)

        with torch.no_grad():
            err_px = torch.linalg.norm(coords - gt_tracks, dim=-1)   # (B,T,N)
            epe = masked_mean(err_px, vis_valid)         # visible-only: comparable across runs
            occ_mask = valid * (1.0 - gt_vis)            # occluded-but-supervised frames
            epe_occ = (masked_mean(err_px, occ_mask) if bool((occ_mask > 0).any())
                       else err_px.new_tensor(float("nan")))
        # Report both the raw terms AND their weighted contributions to ``total``,
        # so a (legitimately) negative total is decomposable: the only terms that
        # can be < 0 are the heteroscedastic uncertainty NLLs (``unc*`` / ``w_unc``).
        parts = {
            "pos": float(pos_reg.detach()),
            "prior": float(prior_reg.detach()),
            "unc": float(unc_reg.detach()),
            "unc_prior": float(unc_prior_reg.detach()),
            "vis": float(vis_loss.detach()),
            "kl": float(kl_loss.detach()),
            "kl_raw": float(kl_raw),
            "ce": float(ce_reg.detach()),
            "gce": float(gce_reg.detach()),
            "rollout": float(rollout_reg.detach()),
            "rollout_epe": float(rollout_epe),
            "epe": float(epe),
            "epe_occ": float(epe_occ),
            "w_pos": float((self.pos_weight * pos_reg).detach()),
            "w_prior": float((self.pos_weight * self.prior_weight * prior_reg).detach()),
            "w_unc": float((self.pos_weight * self.unc_weight
                            * (unc_reg + unc_prior_reg)).detach()),
            "w_vis": float((self.vis_weight * vis_loss).detach()),
            "w_kl": float((self.kl_weight * kl_loss).detach()),
            "w_ce": float((self.ce_weight * ce_reg).detach()),
            "w_gce": float((self.global_ce_weight * gce_reg).detach()),
            "w_rollout": float((self.rollout_weight * rollout_reg).detach()),
        }
        return total, parts
