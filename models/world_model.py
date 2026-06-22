"""The TWIST world model: an observation-corrected state-space point tracker.

State = explicit point coordinates ``s_t (B,N,2)``. Each step:
  1. **Transition** (dynamics prior, frame-free): predicts ``p(s_t|s_{t-1})`` from
     state alone — a per-point GRU coupled across points by self-attention
     (rigidity). Runnable without any frame → occlusion rollout / forecasting.
  2. **Observation** (correction): a local cost-volume around the prior position
     in the next frame's features yields ``q(s_t|s_{t-1},I_t)``.

Both emit diagonal Gaussians over the (normalized) coordinates; the KL between
them (in :mod:`models.losses`) is what forces a useful dynamics prior. The model
works internally in normalized ``[-1,1]`` coordinates and returns tracks in
pixels.

Forward returns a dict (the loss/metrics/viz contract)::

    coords        (B,T,N,2)  posterior mean = predicted tracks (pixels)
    vis_logits    (B,T,N)     visibility logits (supervised by BCE only)
    gate_logits   (B,T,N)     Kalman-gate logits (obs-trust blend; pos-loss only)
    coord_logvar  (B,T,N,2)   posterior log-variance (normalized units)
    prior_mean    (B,T,N,2)   dynamics-only prior mean (pixels)
    prior_logvar  (B,T,N,2)   prior log-variance (normalized units)
    frame_hw      (H,W)       frame size the coords live in
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

    Builds a particle token from ``[appearance, hidden, pos, velocity]``, couples
    tokens across the N points with ``depth`` self-attention blocks (the
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
                 max_step: float = 0.12) -> None:
        super().__init__()
        self.max_step = max_step          # max learned residual displacement (normalized units)
        self.to_token = nn.Linear(c_dim + dh + 4, ds)   # +2 pos, +2 velocity
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
        tok = self.to_token(torch.cat([state["feat"], state["hidden"], pos_n, vel], dim=-1))  # (B,N,Ds)
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
# Observation model (correction from the next frame — local cost-volume)
# --------------------------------------------------------------------------- #
class ObservationModel(nn.Module):
    """Localize each point in the next frame by matching its (frozen, query-frame)
    appearance template against a local window of the next frame's features.

    **Cost-volume soft-argmax (the position correction).** The previous version
    regressed the correction with a free ``max_corr * tanh(linear)`` head off a
    pooled cross-attention vector — nothing tied the emitted offset to *where* the
    template actually matched, so training sculpted it into a saturated near-constant
    that merely cancelled the (also-degenerate) prior, freezing the output at the
    query. We replace it with an explicit correlation soft-argmax: a learnable
    matching projection (init identity) maps template + window features into a
    matching space, cosine similarity gives a per-position score, and the
    correction is the score-softmax-weighted **average window offset**. This is
    *structurally bounded to the search window* — a constant offset is no longer
    expressible unless the correlation surface is genuinely flat. A small,
    zero-init learned residual (``max_corr``, default 0) can refine sub-cell.

    The cross-attention trunk is kept only to read the window for the visibility /
    Kalman-gate / uncertainty heads (those still want a learned summary)."""

    def __init__(self, c_dim: int, ds: int, n_heads: int = 4, k: int = 7, radius_px: float = 24.0,
                 max_corr: float = 0.0) -> None:
        super().__init__()
        self.k = k
        self.radius_px = radius_px
        self.max_corr = max_corr          # bound on the OPTIONAL learned residual (normalized); 0 disables
        # learnable matching metric on top of the frozen features (identity init =
        # raw cosine correspondence, a sane starting point for DINOv3 features).
        self.match_proj = nn.Linear(c_dim, c_dim, bias=False)
        nn.init.eye_(self.match_proj.weight)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))   # exp() ~= 10: softmax temperature
        self.q_proj = nn.Linear(ds + c_dim, c_dim)
        self.pos_enc = nn.Sequential(nn.Linear(2, c_dim), nn.GELU(), nn.Linear(c_dim, c_dim))
        self.cross = nn.MultiheadAttention(c_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(c_dim)
        # Separate heads off a shared trunk so the visibility gradient doesn't
        # corrupt the position features; the uncertainty head reads a DETACHED
        # trunk (calibration only -- it must not perturb the matching features).
        self.trunk = nn.Sequential(nn.Linear(c_dim, c_dim), nn.GELU())
        self.corr_head = nn.Linear(c_dim, 2)
        self.vis_head = nn.Linear(c_dim, 1)
        # The Kalman gate (how much to trust the observation) is a SEPARATE head
        # from the visibility logit. Tying them — as an earlier version did, using
        # sigmoid(vis_logit) as the blend weight — let the visibility BCE (many
        # points are genuinely occluded) drag the gate toward 0. It is now
        # initialised NEUTRAL (bias 0 -> sigmoid 0.5): the prior is a competent
        # constant-velocity smoother and the observation is a bounded localizer, so
        # the gate must be free to learn a LOW gain where the frozen-feature match
        # is noisier than the per-frame motion (slow scenes) and a high gain where
        # it is informative (fast motion) — the previous bias 2.0 pinned it open.
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

    def forward(self, prior_mean, tokens, feat_template, feats_next, hw):
        b, n, _ = prior_mean.shape
        c = feat_template.shape[-1]
        kk = self.k * self.k
        win = sample_window(feats_next, prior_mean, hw, self.k, self.radius_px)  # (B,N,k*k,C)
        off_n = self._offsets_normalized(hw, prior_mean.device, prior_mean.dtype)  # (k*k,2) normalized

        # --- cost-volume soft-argmax localization (bounded to the window) ---
        tpl = F.normalize(self.match_proj(feat_template), dim=-1)                # (B,N,C)
        winm = F.normalize(self.match_proj(win), dim=-1)                         # (B,N,k*k,C)
        corr_vol = (winm * tpl.unsqueeze(2)).sum(-1) * self.logit_scale.exp()    # (B,N,k*k)
        wsm = corr_vol.softmax(dim=-1)                                           # (B,N,k*k)
        corr = (wsm.unsqueeze(-1) * off_n.view(1, 1, kk, 2)).sum(2)              # (B,N,2) normalized

        # --- learned read of the window for the vis / gate / uncertainty heads ---
        kv = win + self.pos_enc(off_n).reshape(1, 1, kk, c)                      # inject offset (normalized units)
        q = self.q_proj(torch.cat([tokens, feat_template], dim=-1)).reshape(b * n, 1, c)
        ev, _ = self.cross(q, kv.reshape(b * n, kk, c), kv.reshape(b * n, kk, c), need_weights=False)
        ev = self.norm(ev.reshape(b, n, c))
        h = self.trunk(ev)
        if self.max_corr > 0:                                                    # optional zero-init sub-cell refine
            corr = corr + self.max_corr * torch.tanh(self.corr_head(h))
        post_mean = denormalize_coords(normalize_coords(prior_mean, hw) + corr, hw)
        vis_logit = self.vis_head(h)
        gate_logit = self.gate_head(h)                                           # Kalman gate (decoupled from vis)
        post_logvar = self.logvar_head(h.detach()).clamp(LOGVAR_MIN, LOGVAR_MAX)  # calibration only
        return post_mean, post_logvar, vis_logit, gate_logit


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
        trans_heads: int = 4,
        trans_depth: int = 2,
        trans_max_step: float = 0.12,
        uncertainty: bool = True,
        encode_chunk: int = 32,
        verbose: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        c = encoder.feature_dim
        self.transition = TransitionModel(c, token_dim, hidden_dim, trans_heads, trans_depth,
                                          max_step=trans_max_step)
        self.observation = ObservationModel(c, token_dim, obs_heads, obs_k, obs_radius_px,
                                            max_corr=obs_max_corr)
        self.dh = hidden_dim
        self.uncertainty = uncertainty
        self.encode_chunk = encode_chunk

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

    def step(self, state, feats_next, hw, point_mask=None, use_observation=True):
        prior_mean, prior_logvar, new_hidden, tokens = self.transition(state, hw, point_mask)
        if use_observation:
            post_mean, post_logvar, vis_logit, gate_logit = self.observation(
                prior_mean, tokens, state["feat"], feats_next, hw)
            g = torch.sigmoid(gate_logit)                      # Kalman gain (trained by position loss)
            pos = g * post_mean + (1.0 - g) * prior_mean
        else:  # frame-free rollout: no observation -> trust the prior (gate closed)
            post_mean, post_logvar = prior_mean, prior_logvar
            vis_logit, gate_logit = state["vis_logit"], prior_mean.new_zeros(prior_mean.shape[:-1] + (1,)) - 10.0
            pos = prior_mean
        new_state = ParticleState(pos=pos, vel=self._advance(state["pos"], pos, hw),
                                  feat=state["feat"], hidden=new_hidden, vis_logit=vis_logit)
        out = dict(prior_mean=prior_mean, prior_logvar=prior_logvar,
                   post_mean=post_mean, post_logvar=post_logvar,
                   vis_logit=vis_logit, gate_logit=gate_logit)
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
        tf_prob: float = 0.0,
        gt_tracks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b, t = frames.shape[:2]
        h, w = int(frames.shape[-2]), int(frames.shape[-1])
        hw = (h, w)
        feats = self._encode(frames)                              # (B,T,C,Hf,Wf)

        qf = int(queries[0, 0, 0].item())                         # query frame (constant per clip)
        qf = max(0, min(qf, t - 1))
        query_xy = queries[..., 1:]                               # (B,N,2)
        state = self.init_state(query_xy, feats[:, qf], hw)

        zlv = query_xy.new_full((b, query_xy.shape[1], 2), LOGVAR_MIN)
        vis_obs = query_xy.new_full((b, query_xy.shape[1], 1), 4.0)   # query frame is observed
        gate_obs = query_xy.new_full((b, query_xy.shape[1], 1), 4.0)  # query frame fully observed
        rec = {k: [None] * t for k in
               ("coords", "prior_mean", "prior_logvar", "post_logvar", "vis_logits", "gate_logits")}

        def put(ti, coords, pm, plv, polv, vl, gl):
            rec["coords"][ti] = coords; rec["prior_mean"][ti] = pm
            rec["prior_logvar"][ti] = plv; rec["post_logvar"][ti] = polv
            rec["vis_logits"][ti] = vl; rec["gate_logits"][ti] = gl

        # frames before the query frame: emit the (frozen) query as a placeholder
        for ti in range(qf):
            put(ti, query_xy, query_xy, zlv, zlv, vis_obs, gate_obs)
        put(qf, query_xy, query_xy, zlv, zlv, vis_obs, gate_obs)  # query frame itself

        for ti in range(qf + 1, t):
            observe = observe_steps is None or (ti - qf) <= observe_steps
            prev_pos = state["pos"]
            state, o = self.step(state, feats[:, ti], hw, point_mask, use_observation=observe)
            put(ti, state["pos"], o["prior_mean"], o["prior_logvar"],
                o["post_logvar"], o["vis_logit"], o["gate_logit"])
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
            "frame_hw": hw,
        }
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
