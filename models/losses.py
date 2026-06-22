"""Training objective for the TWIST world model.

``TrackerLoss`` combines three terms, all masked to valid (visible, non-padded)
points and computed in **normalized** coordinate units (resolution-invariant);
endpoint error is reported in pixels for monitoring:

* **Position NLL** — heteroscedastic Laplace over a Huber position error
  (predicted ``coord_logvar`` down-weights ambiguous matches).
* **Visibility BCE** — on the visibility logits (over all real points, so the
  occluded class is learned).
* **KL(posterior ‖ prior)** — forces the frame-free dynamics prior to predict
  where the observation lands. Uses Dreamer-style **KL balancing** (pull the
  prior toward the posterior harder than vice-versa) and a **free-bits** floor
  so the posterior may still use the observation cheaply.

``forward(outputs, batch) -> (total, parts)`` where ``parts`` is a dict of
detached scalars (``pos, vis, kl, epe``) for logging — matching ``main.py``'s
``(scalar, dict)`` contract.
"""

from __future__ import annotations

from typing import Dict, Tuple

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
        huber_delta: float = 0.2,    # normalized units; large enough to stay ~quadratic
        unc_weight: float = 0.0,     # heteroscedastic NLL (0 disables; it drove total<0 + gamed confidence)
        prior_weight: float = 0.5,   # direct GT supervision of the dynamics-prior mean
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
        valid = gt_vis * exist                          # supervise positions only where visible

        # --- normalized coordinates ---
        mu_q = normalize_coords(coords, hw)
        mu_p = normalize_coords(prior_mean, hw)
        gt_n = normalize_coords(gt_tracks, hw)

        # --- position loss (direct Huber) + decoupled aleatoric uncertainty ---
        # A plain Huber drives the position gradient (and EPE) uniformly. The
        # heteroscedastic term trains log_b to calibrate to the error WITHOUT a
        # position gradient (huber detached) -- avoiding the geometric-mean
        # pathology where a raw NLL is gamed by over-confidence on easy points.
        huber = F.huber_loss(mu_q, gt_n, reduction="none", delta=self.huber_delta).sum(-1)  # (B,T,N)
        pos_reg = masked_mean(huber, valid)
        log_b = 0.5 * coord_logvar.mean(-1)             # (B,T,N) log-scale
        unc = huber.detach() * torch.exp(-log_b) + log_b
        unc_reg = masked_mean(unc, valid)               # heteroscedastic NLL (can be < 0)
        # Direct GT supervision of the dynamics-prior mean. The prior previously got
        # a position gradient ONLY through KL(post||prior) -- killed by free bits and
        # pointing at the (possibly degenerate) posterior, not GT -- so it was free to
        # saturate to ~+87px/step. Anchoring it to GT motion (where visible) makes it a
        # competent constant-velocity smoother and removes the +prior/-obs cancellation.
        huber_prior = F.huber_loss(mu_p, gt_n, reduction="none", delta=self.huber_delta).sum(-1)
        prior_reg = masked_mean(huber_prior, valid)
        pos_loss = pos_reg + self.unc_weight * unc_reg + self.prior_weight * prior_reg

        # --- visibility BCE (over all real points) ---
        bce = F.binary_cross_entropy_with_logits(vis_logits, gt_vis, reduction="none")
        vis_loss = masked_mean(bce, exist)

        # --- KL(posterior ‖ prior), balanced + free bits ---
        a = self.kl_balance_alpha
        kl_sgq_p = kl_diag_gauss(mu_q.detach(), coord_logvar.detach(), mu_p, prior_logvar)
        kl_q_sgp = kl_diag_gauss(mu_q, coord_logvar, mu_p.detach(), prior_logvar.detach())
        kl = a * kl_sgq_p + (1.0 - a) * kl_q_sgp        # (B,T,N)
        if self.kl_free_bits > 0:
            kl = kl.clamp_min(self.kl_free_bits)
        kl_loss = masked_mean(kl, valid)

        total = self.pos_weight * pos_loss + self.vis_weight * vis_loss + self.kl_weight * kl_loss

        with torch.no_grad():
            epe = masked_mean(torch.linalg.norm(coords - gt_tracks, dim=-1), valid)
        # Report both the raw terms AND their weighted contributions to ``total``,
        # so a (legitimately) negative total is decomposable: the only term that
        # can be < 0 is the heteroscedastic uncertainty NLL (``unc`` / ``w_unc``).
        parts = {
            "pos": float(pos_reg.detach()),
            "prior": float(prior_reg.detach()),
            "unc": float(unc_reg.detach()),
            "vis": float(vis_loss.detach()),
            "kl": float(kl_loss.detach()),
            "epe": float(epe),
            "w_pos": float((self.pos_weight * pos_reg).detach()),
            "w_prior": float((self.pos_weight * self.prior_weight * prior_reg).detach()),
            "w_unc": float((self.pos_weight * self.unc_weight * unc_reg).detach()),
            "w_vis": float((self.vis_weight * vis_loss).detach()),
            "w_kl": float((self.kl_weight * kl_loss).detach()),
        }
        return total, parts
