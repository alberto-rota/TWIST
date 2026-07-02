"""The TWIST world model: an observation-corrected state-space point tracker.

State = explicit point coordinates ``s_t (B,N,2)``. Each step:
  1. **Transition** (dynamics prior, frame-free): predicts ``p(s_t|s_{t-1})`` from
     state alone — a per-point GRU coupled across points by self-attention
     (rigidity). Runnable without any frame → occlusion rollout / forecasting.
  2. **Observation** (correction): an optional **coarse re-acquisition** stage
     (template vs the FULL feature grid; confidence-gated re-centering, so a
     point whose prior drifted beyond the local window during occlusion can
     still snap back) followed by a local cost-volume around the (re-centered)
     position in the next frame's features.
  3. **Fusion**: a per-point Kalman gate blends observation vs prior. Its input
     includes detached correlation-surface statistics (peak / entropy / margin)
     and the prior's own calibrated log-variance, so gain follows measurement
     quality rather than being an unconstrained function of appearance.

Both transition and observation emit diagonal Gaussians over the (normalized)
coordinates; their log-variances are trained by a decoupled NLL in
:mod:`models.losses` and are load-bearing (gate features). The model works
internally in normalized ``[-1,1]`` coords and returns pixels.

Forward returns a dict (the loss/metrics/viz contract)::

    coords        (B,T,N,2)  fused prediction = tracks (pixels)
    vis_logits    (B,T,N)    visibility logits (BCE on observed point-steps)
    gate_logits   (B,T,N)    Kalman-gate logits (obs-trust blend; pos-loss only)
    coord_logvar  (B,T,N,2)  posterior log-variance (normalized units)
    prior_mean    (B,T,N,2)  dynamics-only prior mean (pixels)
    prior_logvar  (B,T,N,2)  prior log-variance (normalized units)
    observed      (B,T,N)    1 where the observation ran AND was applied
    corr_valid    (B,T,N)    1 where the local cost-volume was computed
    corr_logits   (B,T,N,k*k)  local cost-volume logits (CE supervision)
    win_center    (B,T,N,2)  window center the cost-volume was sampled at (px)
    corr_grid     (k, radius_px)
    gcorr_logits  (B,T,N,Hf*Wf)  global map logits (coarse stage; train only)
    feat_hw       (Hf, Wf)       (coarse stage; train only)
    coarse_gate   (B,T,N)        coarse re-centering gate (coarse stage only)
    frame_hw      (H,W)      frame size the coords live in
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import (
    FrozenFrameEncoder,
    denormalize_coords,
    normalize_coords,
    sample_features,
    sample_window,
)

LOGVAR_MIN, LOGVAR_MAX = -10.0, 2.0


class ParticleState(TypedDict):
    pos: torch.Tensor        # (B, N, 2) pixel coordinates (x, y)
    vel: torch.Tensor        # (B, N, 2) last realized velocity (normalized units / frame)
    feat: torch.Tensor       # (B, N, C) frozen appearance template (from query frame)
    hidden: torch.Tensor     # (B, N, Dh) GRU dynamics state
    vis_logit: torch.Tensor  # (B, N, 1) current visibility logit


# Velocity (normalized units / frame) is clamped before it feeds the prior, so a
# noisy observation can't bootstrap a runaway constant-velocity extrapolation.
VEL_CLAMP = 0.2


# --------------------------------------------------------------------------- #
# Transition model (dynamics prior, frame-free)
# --------------------------------------------------------------------------- #
class TransitionModel(nn.Module):
    """Predict the next position distribution from state alone.

    Builds a particle token from ``[appearance, hidden, pos, velocity]`` (plus,
    with ``vis_input``, the point's current visibility belief — so the dynamics
    can condition on "am I currently blind" and lean on neighbours accordingly),
    couples tokens across the N points with ``depth`` self-attention blocks (the
    inter-point rigidity prior that carries occluded points on their neighbours'
    motion), advances a per-point GRU, and reads out a normalized displacement
    mean + log-variance.

    **Constant-velocity prior.** The mean is ``pos + last_velocity + bounded_delta``
    rather than ``pos + bounded_delta``: "predict nothing learned" then defaults to
    extrapolating the last realized velocity (the correct low-variance prior for
    the smooth tissue/scene motion in our data), and the GRU only has to learn the
    *residual* (acceleration). This is the smoother the Kalman filter leans on when
    the observation is too noisy (per-frame motion << frozen-feature match noise).
    """

    def __init__(self, c_dim: int, ds: int, dh: int, n_heads: int = 4, depth: int = 2,
                 max_step: float = 0.12, vis_input: bool = False) -> None:
        super().__init__()
        self.max_step = max_step          # max learned residual displacement (normalized units)
        self.vis_input = bool(vis_input)
        in_dim = c_dim + dh + 4 + (1 if self.vis_input else 0)  # +2 pos, +2 velocity, +1 vis belief
        self.to_token = nn.Linear(in_dim, ds)
        self.blocks = nn.ModuleList()
        for _ in range(max(depth, 1)):
            self.blocks.append(nn.ModuleDict({
                "norm1": nn.LayerNorm(ds),
                "attn": nn.MultiheadAttention(ds, n_heads, batch_first=True),
                "norm2": nn.LayerNorm(ds),
                "ffn": nn.Sequential(nn.Linear(ds, 2 * ds), nn.GELU(), nn.Linear(2 * ds, ds)),
            }))
        self.gru = nn.GRUCell(ds, dh)
        self.mean_head = nn.Linear(dh, 2)
        self.logvar_head = nn.Linear(dh, 2)
        nn.init.zeros_(self.mean_head.weight); nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.logvar_head.bias, -2.0)  # start moderately confident

    def forward(
        self, state: ParticleState, hw: Tuple[int, int], point_mask: Optional[torch.Tensor] = None
    ):
        b, n, _ = state["pos"].shape
        pos_n = normalize_coords(state["pos"], hw)                      # (B,N,2) in [-1,1]
        vel = state["vel"]                                             # (B,N,2) normalized / frame
        parts = [state["feat"], state["hidden"], pos_n, vel]
        if self.vis_input:
            parts.append(torch.sigmoid(state["vis_logit"]))            # (B,N,1) visibility belief
        tok = self.to_token(torch.cat(parts, dim=-1))                  # (B,N,Ds)
        kpm = (~point_mask) if point_mask is not None else None         # True == ignore
        for blk in self.blocks:
            x = blk["norm1"](tok)
            attn, _ = blk["attn"](x, x, x, key_padding_mask=kpm, need_weights=False)
            tok = tok + attn
            tok = tok + blk["ffn"](blk["norm2"](tok))
        new_hidden = self.gru(tok.reshape(b * n, -1), state["hidden"].reshape(b * n, -1)).reshape(b, n, -1)
        # constant-velocity baseline + bounded learned residual (tanh keeps rollouts stable)
        delta = self.max_step * torch.tanh(self.mean_head(new_hidden))  # (B,N,2) normalized residual
        prior_mean = denormalize_coords(pos_n + vel + delta, hw)        # (B,N,2) pixels
        prior_logvar = self.logvar_head(new_hidden).clamp(LOGVAR_MIN, LOGVAR_MAX)
        return prior_mean, prior_logvar, new_hidden, tok


# --------------------------------------------------------------------------- #
# Observation model (correction from the next frame — coarse-to-fine cost volume)
# --------------------------------------------------------------------------- #
class ObservationModel(nn.Module):
    """Localize each point in the next frame by matching its (frozen, query-frame)
    appearance template against the frame's features.

    **Cost-volume soft-argmax (the position correction).** A learnable matching
    projection (init identity) maps template + window features into a matching
    space, cosine similarity gives a per-position score, and the correction is
    the score-softmax-weighted **average window offset** — structurally bounded
    to the search window. A small, zero-init learned residual (``max_corr``,
    default 0) can refine sub-cell. The raw logits are exported
    (``corr_logits``) so the loss can supervise the map directly with a
    cross-entropy — a matcher gradient that does NOT pass through (and so is
    never attenuated by) the Kalman gate.

    **Coarse re-acquisition stage (``coarse=True``).** The fine window is
    ±``radius_px`` around the prior — if the prior drifted further than that
    during an occlusion, re-acquisition used to be structurally impossible. The
    coarse stage correlates the template against the FULL feature grid, takes a
    global soft-argmax, and re-centers the fine window toward it by a
    confidence gate (zero-init, bias -2 → starts nearly closed, i.e. the old
    prior-centered behaviour; it must *earn* the right to move the window).
    Global map logits are exported for CE supervision.

    **Head statistics (``stats=True``).** The visibility / gate / uncertainty
    heads additionally read detached correlation-surface statistics (peak,
    normalized entropy, top-1/2 margin, the prior's log-scale, and the coarse
    stats when enabled) — the measurement-quality signals a Kalman gain should
    be a function of, delivered explicitly instead of hoping the cross-attention
    trunk learns to compute them.

    The cross-attention trunk is kept to read the window for the visibility /
    Kalman-gate / uncertainty heads (those still want a learned summary)."""

    #: number of detached scalar stats appended per point when ``stats`` is on
    N_STATS_LOCAL = 4   # local peak, local entropy, local margin, prior log-scale
    N_STATS_COARSE = 3  # global peak, global entropy, prior->coarse distance

    def __init__(self, c_dim: int, ds: int, n_heads: int = 4, k: int = 7, radius_px: float = 24.0,
                 max_corr: float = 0.0, stats: bool = False, coarse: bool = False) -> None:
        super().__init__()
        self.k = k
        self.radius_px = radius_px
        self.max_corr = max_corr          # bound on the OPTIONAL learned residual (normalized); 0 disables
        self.stats = bool(stats)
        self.coarse = bool(coarse)
        # learnable matching metric on top of the frozen features (identity init =
        # raw cosine correspondence, a sane starting point for DINOv3 features).
        self.match_proj = nn.Linear(c_dim, c_dim, bias=False)
        nn.init.eye_(self.match_proj.weight)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))   # exp() ~= 10: softmax temperature
        self.q_proj = nn.Linear(ds + c_dim, c_dim)
        self.pos_enc = nn.Sequential(nn.Linear(2, c_dim), nn.GELU(), nn.Linear(c_dim, c_dim))
        self.cross = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(c_dim)
        if self.coarse:
            # its own (sharper) temperature: the global map must peak hard for the
            # soft-argmax over the whole frame to mean anything.
            self.coarse_logit_scale = nn.Parameter(torch.tensor(3.0))   # exp() ~= 20
            # re-centering gate off detached coarse stats; zero-init weight + bias -2
            # -> sigmoid ~0.12: starts (almost) prior-centered, opens with evidence.
            self.coarse_gate_head = nn.Linear(self.N_STATS_COARSE + 1, 1)
            nn.init.zeros_(self.coarse_gate_head.weight)
            nn.init.constant_(self.coarse_gate_head.bias, -2.0)
        # Separate heads off a shared trunk so the visibility gradient doesn't
        # corrupt the position features; the uncertainty head reads a DETACHED
        # trunk (calibration only -- it must not perturb the matching features).
        n_stats = (self.N_STATS_LOCAL + (self.N_STATS_COARSE if self.coarse else 0)) if self.stats else 0
        self.trunk = nn.Sequential(nn.Linear(c_dim + n_stats, c_dim), nn.GELU())
        self.corr_head = nn.Linear(c_dim, 2)
        self.vis_head = nn.Linear(c_dim, 1)
        # The Kalman gate (how much to trust the observation) is a SEPARATE head
        # from the visibility logit (tying them let the visibility BCE drag the
        # gate toward 0). Initialised NEUTRAL (bias 0 -> sigmoid 0.5): the prior is
        # a competent constant-velocity smoother and the observation is a bounded
        # localizer, so the gate must be free to learn a LOW gain where the match
        # is noisier than the per-frame motion and a high gain where it is
        # informative. With ``stats`` it sees the correlation quality directly.
        self.gate_head = nn.Linear(c_dim, 1)
        self.logvar_head = nn.Linear(c_dim, 2)
        nn.init.zeros_(self.corr_head.weight); nn.init.zeros_(self.corr_head.bias)
        nn.init.zeros_(self.gate_head.weight); nn.init.zeros_(self.gate_head.bias)
        nn.init.constant_(self.logvar_head.bias, -2.0)

    def _offset_grid(self, device, dtype) -> torch.Tensor:
        lin = torch.linspace(-self.radius_px, self.radius_px, self.k, device=device, dtype=dtype)
        oy, ox = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([ox, oy], dim=-1).reshape(self.k * self.k, 2)   # (k*k, 2) pixels

    def _offsets_normalized(self, hw, device, dtype) -> torch.Tensor:
        """``k*k`` window offsets in normalize_coords units (``2*d_px/(dim-1)``),
        consistent with ``normalize_coords`` so they compose with the prior."""
        h, w = hw
        off_px = self._offset_grid(device, dtype)                          # (k*k,2) pixels
        scale = off_px.new_tensor([2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1)])
        return off_px * scale                                              # (k*k,2) normalized

    @staticmethod
    def _grid_centers_px(hf: int, wf: int, hw, device, dtype) -> torch.Tensor:
        """Feature-cell centers in pixel coords, matching the align_corners=True
        convention of :func:`normalize_coords` (cell (0,0) -> pixel (0,0), cell
        (hf-1,wf-1) -> pixel (h-1,w-1))."""
        h, w = hw
        xs = torch.linspace(0, w - 1, wf, device=device, dtype=dtype)
        ys = torch.linspace(0, h - 1, hf, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], dim=-1).reshape(hf * wf, 2)          # (Hf*Wf, 2) px

    @staticmethod
    def _softmax_stats(p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(peak, normalized entropy, top1-top2 margin) of a prob map ``p (...,M)``,
        each ``(...,1)`` and DETACHED (head inputs must not shape the matcher)."""
        m = p.shape[-1]
        peak = p.max(dim=-1, keepdim=True).values
        ent = -(p * (p + 1e-9).log()).sum(-1, keepdim=True) / torch.log(
            torch.tensor(float(m), device=p.device, dtype=p.dtype))
        top2 = p.topk(2, dim=-1).values
        margin = top2[..., :1] - top2[..., 1:2]
        return peak.detach(), ent.detach(), margin.detach()

    def forward(self, prior_mean, tokens, feat_template, feats_next, hw,
                prior_logvar: Optional[torch.Tensor] = None):
        b, n, _ = prior_mean.shape
        c = feat_template.shape[-1]
        kk = self.k * self.k
        tpl = F.normalize(self.match_proj(feat_template), dim=-1)                # (B,N,C)
        # prior log-scale: the calibrated "how lost am I" signal (grows over a
        # coasted occlusion) — detached: heads read it, they don't train it.
        if prior_logvar is not None:
            pls = 0.5 * prior_logvar.mean(-1, keepdim=True).detach()             # (B,N,1)
        else:
            pls = prior_mean.new_zeros(b, n, 1)
        extras: Dict[str, torch.Tensor] = {}

        # --- coarse re-acquisition: template vs the FULL grid, gated re-centering ---
        center = prior_mean
        coarse_stats = None
        if self.coarse:
            hf, wf = int(feats_next.shape[-2]), int(feats_next.shape[-1])
            featm = self.match_proj(feats_next.permute(0, 2, 3, 1).reshape(b, hf * wf, c))
            featm = F.normalize(featm, dim=-1)                                   # (B,Hf*Wf,C)
            glog = torch.bmm(tpl, featm.transpose(1, 2)) * self.coarse_logit_scale.exp()  # (B,N,Hf*Wf)
            gp = glog.softmax(dim=-1)
            centers_px = self._grid_centers_px(hf, wf, hw, glog.device, glog.dtype)
            coarse_xy = gp @ centers_px                                          # (B,N,2) px
            gpeak, gent, _ = self._softmax_stats(gp)
            dist = (normalize_coords(coarse_xy, hw)
                    - normalize_coords(prior_mean, hw)).norm(dim=-1, keepdim=True).detach()
            cg = torch.sigmoid(self.coarse_gate_head(
                torch.cat([gpeak, gent, dist, pls], dim=-1)))                     # (B,N,1)
            center = prior_mean + cg * (coarse_xy - prior_mean)
            coarse_stats = [gpeak, gent, dist]
            extras["gcorr_logits"] = glog
            extras["feat_hw"] = (hf, wf)
            extras["coarse_gate"] = cg

        # --- local cost-volume soft-argmax localization (bounded to the window) ---
        win = sample_window(feats_next, center, hw, self.k, self.radius_px)      # (B,N,k*k,C)
        off_n = self._offsets_normalized(hw, prior_mean.device, prior_mean.dtype)  # (k*k,2) normalized
        winm = F.normalize(self.match_proj(win), dim=-1)                         # (B,N,k*k,C)
        corr_vol = (winm * tpl.unsqueeze(2)).sum(-1) * self.logit_scale.exp()    # (B,N,k*k)
        wsm = corr_vol.softmax(dim=-1)                                           # (B,N,k*k)
        corr = (wsm.unsqueeze(-1) * off_n.view(1, 1, kk, 2)).sum(2)              # (B,N,2) normalized
        extras["corr_logits"] = corr_vol
        extras["win_center"] = center

        # --- learned read of the window for the vis / gate / uncertainty heads ---
        kv = win + self.pos_enc(off_n).reshape(1, 1, kk, c)                      # inject offset (normalized units)
        q = self.q_proj(torch.cat([tokens, feat_template], dim=-1)).reshape(b * n, 1, c)
        ev, _ = self.cross(q, kv.reshape(b * n, kk, c), kv.reshape(b * n, kk, c), need_weights=False)
        ev = self.norm(ev.reshape(b, n, c))
        if self.stats:
            lpeak, lent, lmarg = self._softmax_stats(wsm)
            feats_in = [ev, lpeak, lent, lmarg, pls]
            if coarse_stats is not None:
                feats_in += coarse_stats
            h = self.trunk(torch.cat(feats_in, dim=-1))
        else:
            h = self.trunk(ev)
        if self.max_corr > 0:                                                    # optional zero-init sub-cell refine
            corr = corr + self.max_corr * torch.tanh(self.corr_head(h))
        post_mean = denormalize_coords(normalize_coords(center, hw) + corr, hw)
        vis_logit = self.vis_head(h)
        gate_logit = self.gate_head(h)                                           # Kalman gate (decoupled from vis)
        post_logvar = self.logvar_head(h.detach()).clamp(LOGVAR_MIN, LOGVAR_MAX)  # calibration only
        return post_mean, post_logvar, vis_logit, gate_logit, extras


# --------------------------------------------------------------------------- #
# Full world model
# --------------------------------------------------------------------------- #
class TrackerWorldModel(nn.Module):
    def __init__(
        self,
        encoder: FrozenFrameEncoder,
        hidden_dim: int = 256,
        token_dim: int = 256,
        obs_k: int = 7,
        obs_radius_px: float = 24.0,
        obs_heads: int = 4,
        obs_max_corr: float = 0.0,
        obs_stats: bool = False,
        obs_coarse: bool = False,
        trans_heads: int = 4,
        trans_depth: int = 2,
        trans_max_step: float = 0.12,
        trans_vis_input: bool = False,
        uncertainty: bool = True,
        encode_chunk: int = 32,
        rollout_vel_decay: float = 1.0,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        c = encoder.feature_dim
        self.transition = TransitionModel(c, token_dim, hidden_dim, trans_heads, trans_depth,
                                          max_step=trans_max_step, vis_input=trans_vis_input)
        self.observation = ObservationModel(c, token_dim, obs_heads, obs_k, obs_radius_px,
                                            max_corr=obs_max_corr, stats=obs_stats,
                                            coarse=obs_coarse)
        self.dh = hidden_dim
        self.uncertainty = uncertainty
        self.encode_chunk = encode_chunk
        # friction on the constant-velocity prior during FRAME-FREE steps only: the
        # ``pos + vel + delta`` skip compounds a biased velocity into the ~4x over-move
        # seen in long rollouts. <1 decays carried momentum toward 0 each forecast step
        # (no effect on observed steps, where the observation re-anchors velocity). 1=off.
        self.rollout_vel_decay = float(rollout_vel_decay)

    # -- state ---------------------------------------------------------------- #
    def init_state(self, query_xy: torch.Tensor, feats_q: torch.Tensor, hw) -> ParticleState:
        b, n, _ = query_xy.shape
        return ParticleState(
            pos=query_xy,
            vel=query_xy.new_zeros(b, n, 2),                  # zero velocity at the query frame
            feat=sample_features(feats_q, query_xy, hw),
            hidden=query_xy.new_zeros(b, n, self.dh),
            vis_logit=query_xy.new_zeros(b, n, 1),
        )

    def _advance(self, prev_pos, new_pos, hw) -> torch.Tensor:
        """Realized velocity (normalized units/frame) from prev->new position, clamped
        so a noisy observation can't bootstrap a runaway constant-velocity prior.

        Detached: velocity is a *measured momentum state feature* (Kalman-style), not a
        learnable quantity — the model influences future motion only through positions
        (which are supervised) and the GRU hidden (the intended differentiable recurrence).
        Detaching keeps the constant-velocity skip ``pos + vel + delta`` useful in the
        forward pass while truncating the extra position-space BPTT path it would add over
        long rollouts (T up to 48), so gradients stay bounded."""
        vel = normalize_coords(new_pos, hw) - normalize_coords(prev_pos, hw)
        return vel.clamp(-VEL_CLAMP, VEL_CLAMP).detach()

    def step(self, state, feats_next, hw, point_mask=None, use_observation=True,
             obs_point_mask: Optional[torch.Tensor] = None):
        """One filter step.

        ``use_observation=False`` -> pure frame-free rollout for every point.
        ``obs_point_mask`` (B or 1, N, 1) float in {0,1}: PER-POINT observation
        gating — the training-time simulation of real occlusion (some points
        blinded while their neighbours keep observing and, through the
        transition's self-attention, carry them). Masked points take the prior
        and keep their previous visibility belief; the observation branch is
        computed for all points (it must be — neighbours need the frame) but is
        *applied* only where the mask is 1.
        """
        prior_mean, prior_logvar, new_hidden, tokens = self.transition(state, hw, point_mask)
        extras: Dict[str, torch.Tensor] = {}
        if use_observation:
            post_mean, post_logvar, vis_logit_o, gate_logit_o, extras = self.observation(
                prior_mean, tokens, state["feat"], feats_next, hw, prior_logvar)
            g = torch.sigmoid(gate_logit_o)                    # Kalman gain (trained by position loss)
            pos_obs = g * post_mean + (1.0 - g) * prior_mean
            if obs_point_mask is None:
                pos = pos_obs
                vis_logit, gate_logit = vis_logit_o, gate_logit_o
                obs_applied = prior_mean.new_ones(prior_mean.shape[:-1] + (1,))
            else:                                              # per-point simulated occlusion
                m = obs_point_mask.to(pos_obs.dtype)
                pos = m * pos_obs + (1.0 - m) * prior_mean
                vis_logit = m * vis_logit_o + (1.0 - m) * state["vis_logit"]
                gate_logit = m * gate_logit_o + (1.0 - m) * gate_logit_o.new_full(
                    gate_logit_o.shape, -10.0)
                post_logvar = m * post_logvar + (1.0 - m) * prior_logvar
                obs_applied = m.expand(prior_mean.shape[:-1] + (1,)).to(prior_mean.dtype)
        else:  # frame-free rollout: no observation -> trust the prior (gate closed)
            post_mean, post_logvar = prior_mean, prior_logvar
            pos = prior_mean
            vis_logit = state["vis_logit"]
            gate_logit = prior_mean.new_zeros(prior_mean.shape[:-1] + (1,)) - 10.0
            obs_applied = prior_mean.new_zeros(prior_mean.shape[:-1] + (1,))
        new_vel = self._advance(state["pos"], pos, hw)
        if not use_observation and self.rollout_vel_decay != 1.0:
            new_vel = new_vel * self.rollout_vel_decay          # friction: damp the frame-free runaway
        new_state = ParticleState(pos=pos, vel=new_vel,
                                  feat=state["feat"], hidden=new_hidden, vis_logit=vis_logit)
        out = dict(prior_mean=prior_mean, prior_logvar=prior_logvar,
                   post_mean=post_mean, post_logvar=post_logvar,
                   vis_logit=vis_logit, gate_logit=gate_logit,
                   obs_applied=obs_applied, extras=extras)
        return new_state, out

    # -- encoding ------------------------------------------------------------- #
    def _encode(self, frames: torch.Tensor) -> torch.Tensor:
        """``frames (B,T,3,H,W)`` -> ``feats (B,T,C,Hf,Wf)`` (chunked)."""
        b, t = frames.shape[:2]
        flat = frames.reshape(b * t, *frames.shape[2:])
        outs: List[torch.Tensor] = []
        for i in range(0, flat.shape[0], self.encode_chunk):
            outs.append(self.encoder(flat[i:i + self.encode_chunk]))
        feats = torch.cat(outs, dim=0)
        return feats.reshape(b, t, *feats.shape[1:])

    # -- forward -------------------------------------------------------------- #
    def forward(
        self,
        frames: torch.Tensor,                    # (B,T,3,H,W)
        queries: torch.Tensor,                   # (B,N,3) = (t,x,y)
        point_mask: Optional[torch.Tensor] = None,
        observe_steps: Optional[int] = None,
        observe_mask: Optional[torch.Tensor] = None,
        tf_prob: float = 0.0,
        gt_tracks: Optional[torch.Tensor] = None,
        rollout_observe: Optional[int] = None,
        rollout_horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        b, t = frames.shape[:2]
        h, w = int(frames.shape[-2]), int(frames.shape[-1])
        hw = (h, w)
        n = queries.shape[1]
        feats = self._encode(frames)                              # (B,T,C,Hf,Wf)

        qf = int(queries[0, 0, 0].item())                         # query frame (constant per clip)
        # Every real query in the batch must share that frame — the recurrence
        # starts once per forward. Padded query rows are all-zero and pass the
        # check (qf==0 batches trivially satisfy it).
        qtimes = queries[..., 0]
        if not bool(((qtimes == qf) | (qtimes == 0)).all()):
            raise ValueError("all queries in a batch must share one query frame "
                             f"(got frames {torch.unique(qtimes).tolist()})")
        qf = max(0, min(qf, t - 1))
        query_xy = queries[..., 1:]                               # (B,N,2)
        state = self.init_state(query_xy, feats[:, qf], hw)

        kk = self.observation.k ** 2
        coarse = self.observation.coarse
        record_gcorr = coarse and self.training                   # big map: train-time only
        hf, wf = int(feats.shape[-2]), int(feats.shape[-1])

        zlv = query_xy.new_full((b, n, 2), LOGVAR_MIN)
        vis_obs = query_xy.new_full((b, n, 1), 4.0)   # query frame is observed
        gate_obs = query_xy.new_full((b, n, 1), 4.0)  # query frame fully observed
        keys = ["coords", "prior_mean", "prior_logvar", "post_logvar", "vis_logits",
                "gate_logits", "observed", "corr_valid", "corr_logits", "win_center"]
        if record_gcorr:
            keys.append("gcorr_logits")
        if coarse:
            keys.append("coarse_gate")
        rec: Dict[str, list] = {k: [None] * t for k in keys}

        def zeros_for(key):
            if key == "corr_logits":
                return query_xy.new_zeros(b, n, kk)
            if key == "gcorr_logits":
                return query_xy.new_zeros(b, n, hf * wf)
            return query_xy.new_zeros(b, n, 1)        # observed / corr_valid / coarse_gate

        def put(ti, *, coords, pm, plv, polv, vl, gl, obs_applied=None, corr_valid=0.0,
                extras=None):
            rec["coords"][ti] = coords; rec["prior_mean"][ti] = pm
            rec["prior_logvar"][ti] = plv; rec["post_logvar"][ti] = polv
            rec["vis_logits"][ti] = vl; rec["gate_logits"][ti] = gl
            rec["observed"][ti] = (obs_applied if obs_applied is not None
                                   else query_xy.new_ones(b, n, 1))
            rec["corr_valid"][ti] = query_xy.new_full((b, n, 1), float(corr_valid))
            ex = extras or {}
            rec["corr_logits"][ti] = ex.get("corr_logits", zeros_for("corr_logits"))
            rec["win_center"][ti] = ex.get("win_center", coords)
            if record_gcorr:
                rec["gcorr_logits"][ti] = ex.get("gcorr_logits", zeros_for("gcorr_logits"))
            if coarse:
                rec["coarse_gate"][ti] = ex.get("coarse_gate", zeros_for("coarse_gate"))

        # frames up to and including the query frame: emit the (frozen) query
        for ti in range(qf + 1):
            put(ti, coords=query_xy, pm=query_xy, plv=zlv, polv=zlv, vl=vis_obs, gl=gate_obs)

        for ti in range(qf + 1, t):
            # ``observe_mask`` takes precedence: training-time observation dropout.
            #   (T,) bool  -> whole-frame drop (all points coast together);
            #   (T,N) bool -> PER-POINT drop (simulated occlusion: masked points coast
            #                 on the transition + neighbours while the rest observe).
            # Falls back to the contiguous ``observe_steps`` horizon of rollout eval.
            opm = None
            if observe_mask is not None:
                om = observe_mask[ti]
                if om.dim() == 0:
                    observe = bool(om)
                else:                                          # (N,) per-point row
                    observe = bool(om.any())
                    opm = None if bool(om.all()) else om.reshape(1, n, 1)
            else:
                observe = observe_steps is None or (ti - qf) <= observe_steps
            prev_pos = state["pos"]
            state, o = self.step(state, feats[:, ti] if observe else None, hw, point_mask,
                                 use_observation=observe, obs_point_mask=opm)
            put(ti, coords=state["pos"], pm=o["prior_mean"], plv=o["prior_logvar"],
                polv=o["post_logvar"], vl=o["vis_logit"], gl=o["gate_logit"],
                obs_applied=o["obs_applied"], corr_valid=1.0 if observe else 0.0,
                extras=o["extras"])
            # Scheduled sampling: with per-point probability ``tf_prob`` feed the GT
            # position into the NEXT step (the recorded ``coords`` above is always the
            # model's own prediction, so the loss still penalises it). Annealing
            # ``tf_prob`` 1->0 over training removes the hard teacher-forcing cliff and
            # teaches the filter to recover from its own errors. The velocity is
            # recomputed from the (possibly GT-substituted) position so the
            # constant-velocity prior stays consistent with the fed trajectory.
            if tf_prob > 0.0 and gt_tracks is not None:
                take_gt = torch.rand(b, state["pos"].shape[1], 1, device=state["pos"].device) < tf_prob
                mixed = torch.where(take_gt, gt_tracks[:, ti], state["pos"])
                state = ParticleState(pos=mixed, vel=self._advance(prev_pos, mixed, hw),
                                      feat=state["feat"], hidden=state["hidden"],
                                      vis_logit=state["vis_logit"])

        out = {
            "coords": torch.stack(rec["coords"], dim=1),              # (B,T,N,2)
            "prior_mean": torch.stack(rec["prior_mean"], dim=1),      # (B,T,N,2)
            "prior_logvar": torch.stack(rec["prior_logvar"], dim=1),  # (B,T,N,2)
            "coord_logvar": torch.stack(rec["post_logvar"], dim=1),   # (B,T,N,2)
            "vis_logits": torch.stack(rec["vis_logits"], dim=1).squeeze(-1),    # (B,T,N)
            "gate_logits": torch.stack(rec["gate_logits"], dim=1).squeeze(-1),  # (B,T,N)
            "observed": torch.stack(rec["observed"], dim=1).squeeze(-1),        # (B,T,N)
            "corr_valid": torch.stack(rec["corr_valid"], dim=1).squeeze(-1),    # (B,T,N)
            "corr_logits": torch.stack(rec["corr_logits"], dim=1),    # (B,T,N,k*k)
            "win_center": torch.stack(rec["win_center"], dim=1),      # (B,T,N,2)
            "corr_grid": (self.observation.k, self.observation.radius_px),
            "frame_hw": hw,
        }
        if record_gcorr:
            out["gcorr_logits"] = torch.stack(rec["gcorr_logits"], dim=1)  # (B,T,N,Hf*Wf)
            out["feat_hw"] = (hf, wf)
        if coarse:
            out["coarse_gate"] = torch.stack(rec["coarse_gate"], dim=1).squeeze(-1)  # (B,T,N)

        # --- optional differentiable multi-step rollout supervision -------------
        # Mirrors the rollout-eval protocol (observe ``rollout_observe`` real frames
        # from the query, then forecast frame-free) but WITH gradients, so the loss
        # can train the prior to predict from its OWN propagated state -- the cure
        # for the exposure bias that lets a 1-step-supervised prior diverge in
        # free-running rollout. Reuses ``feats`` (no re-encode) from a fresh state,
        # observed CLEANLY (no teacher forcing / dropout) so the forecast starts from
        # the model's own observed estimate exactly as eval does. ``rollout_horizon``
        # defaults to the rest of the clip.
        if rollout_observe is not None:
            obs_end = min(qf + int(rollout_observe), t - 1)          # last observed step
            start = obs_end + 1                                      # first forecast step
            horizon = (t - start) if rollout_horizon is None else min(int(rollout_horizon), t - start)
            if horizon > 0:
                rs = self.init_state(query_xy, feats[:, qf], hw)
                for ti in range(qf + 1, obs_end + 1):               # clean observe
                    rs, _ = self.step(rs, feats[:, ti], hw, point_mask, use_observation=True)
                roll = []
                for _ in range(horizon):                            # frame-free forecast (grad on)
                    rs, _ = self.step(rs, feats_next=None, hw=hw, point_mask=point_mask,
                                      use_observation=False)
                    roll.append(rs["pos"])
                out["rollout_coords"] = torch.stack(roll, dim=1)    # (B,horizon,N,2)
                out["rollout_start"] = start
        return out

    # -- frame-free rollout (occlusion / forecast / counterfactual) ----------- #
    @torch.no_grad()
    def rollout(self, state: ParticleState, n_steps: int, hw) -> Dict[str, torch.Tensor]:
        """Advance the dynamics prior ``n_steps`` with NO observations."""
        coords, pmean, plv, vis = [], [], [], []
        for _ in range(n_steps):
            state, o = self.step(state, feats_next=None, hw=hw, use_observation=False)
            coords.append(state["pos"]); pmean.append(o["prior_mean"])
            plv.append(o["prior_logvar"]); vis.append(o["vis_logit"])
        return {
            "coords": torch.stack(coords, dim=1),
            "prior_mean": torch.stack(pmean, dim=1),
            "prior_logvar": torch.stack(plv, dim=1),
            "vis_logits": torch.stack(vis, dim=1).squeeze(-1),
            "frame_hw": hw,
        }

    def rollout_from_queries(self, frames, queries, observe_steps: int, n_future: int):
        """Observe ``observe_steps`` real frames from the query, then forecast
        ``n_future`` steps frame-free. Returns the rollout dict."""
        b, t = frames.shape[:2]
        h, w = int(frames.shape[-2]), int(frames.shape[-1])
        hw = (h, w)
        feats = self._encode(frames)
        qf = max(0, min(int(queries[0, 0, 0].item()), t - 1))
        state = self.init_state(queries[..., 1:], feats[:, qf], hw)
        end = min(qf + observe_steps, t - 1)
        for ti in range(qf + 1, end + 1):
            state, _ = self.step(state, feats[:, ti], hw, use_observation=True)
        return self.rollout(state, n_future, hw)
