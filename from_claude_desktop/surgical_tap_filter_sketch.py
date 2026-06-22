"""
Closed-loop dense point tracking as a recurrent particle filter (surgical domain).

Pure tracking. The next frame is ALWAYS observed, so there is no generative /
forecasting branch and no mode switch. Each step runs:
    (1) a transition (motion prior) that carries every point forward from state
        alone, including points currently occluded by instruments, smoke or blood;
    (2) an observation update that corrects the prediction using the new frame.
The visibility head learns how much the observation should override the prior.

Shape conventions (batch dimension is always first, dtype float32):
    B  = batch            T  = clip length         N  = tracked points
    H, W = image size     p  = patch size          Hf, Wf = H // p, W // p
    C  = frame feature dim   Ds = particle token dim   Dh = recurrent hidden dim

This is a SKETCH: submodule internals are minimal stand-ins so the tensor shapes
flow end to end. The blocks marked `sketch:` are where the real architecture goes.
"""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# State container
# --------------------------------------------------------------------------- #
class ParticleState(TypedDict):
    pos: torch.Tensor        # (B, N, 2) tracked pixel coordinates (x, y)
    feat: torch.Tensor       # (B, N, C) appearance descriptor sampled at query time
    hidden: torch.Tensor     # (B, N, Dh) recurrent dynamics hidden state
    vis_logit: torch.Tensor  # (B, N, 1) visibility logit at current frame


def sample_features(feats: torch.Tensor, xy: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly sample dense features at sub-pixel point locations.

    Args:
        feats: (B, C, Hf, Wf) dense frame features.
        xy:    (B, N, 2) point coordinates in ORIGINAL-image pixels, order (x, y).
        hw:    (H, W) original image size used to normalise coordinates.
    Returns:
        (B, N, C) sampled feature vectors.
    """
    H, W = hw
    grid = torch.empty_like(xy)
    grid[..., 0] = 2.0 * xy[..., 0] / (W - 1) - 1.0          # x -> [-1, 1]
    grid[..., 1] = 2.0 * xy[..., 1] / (H - 1) - 1.0          # y -> [-1, 1]
    grid = grid.unsqueeze(1)                                 # (B, 1, N, 2)
    sampled = F.grid_sample(feats, grid, mode="bilinear", align_corners=True)  # (B, C, 1, N)
    return sampled.squeeze(2).transpose(1, 2).contiguous()   # (B, N, C)


# --------------------------------------------------------------------------- #
# Frozen frame encoder
# --------------------------------------------------------------------------- #
class FrozenFrameEncoder(nn.Module):
    """Frozen dense feature extractor.

    In practice: DINOv2 ViT-S/14, or a surgical foundation model (EndoFM,
    Surgical-DINO). Here a single patch-conv stands in so shapes flow.
    """

    def __init__(self, out_dim: int = 384, patch: int = 14) -> None:
        super().__init__()
        self.patch = patch
        # sketch: replace with a frozen pretrained backbone.
        self.stem = nn.Conv2d(3, out_dim, kernel_size=patch, stride=patch)
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, 3, H, W)  ->  feats: (B, C, Hf, Wf)."""
        assert frames.dim() == 4 and frames.shape[1] == 3, "expected (B, 3, H, W)"
        return self.stem(frames)                              # (B, C, H // p, W // p)


# --------------------------------------------------------------------------- #
# Transition model (motion prior, frame-free)
# --------------------------------------------------------------------------- #
class TransitionModel(nn.Module):
    """Predict next position from state alone, with an inter-point rigidity prior.

    This is what keeps occluded points alive: when the observation is unreliable
    (low visibility), the corrected position stays close to this prior.
    """

    def __init__(self, c_dim: int, ds: int, dh: int) -> None:
        super().__init__()
        self.to_token = nn.Linear(c_dim + dh + 2, ds)                       # build particle token
        self.spatial = nn.MultiheadAttention(ds, num_heads=4, batch_first=True)  # coupling across N
        self.gru = nn.GRUCell(ds, dh)                                       # temporal recurrence
        self.delta = nn.Linear(dh, 2)                                       # predicted displacement

    def forward(self, state: ParticleState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (pred_pos (B, N, 2), new_hidden (B, N, Dh), tokens (B, N, Ds))."""
        b, n, _ = state["pos"].shape
        tok = self.to_token(torch.cat([state["feat"], state["hidden"], state["pos"]], dim=-1))  # (B, N, Ds)
        tok, _ = self.spatial(tok, tok, tok)                                # (B, N, Ds) rigidity coupling
        new_hidden = self.gru(
            tok.reshape(b * n, -1), state["hidden"].reshape(b * n, -1)
        ).reshape(b, n, -1)                                                 # (B, N, Dh)
        pred_pos = state["pos"] + self.delta(new_hidden)                    # (B, N, 2) prior position
        return pred_pos, new_hidden, tok


# --------------------------------------------------------------------------- #
# Observation model (correction from the new frame)
# --------------------------------------------------------------------------- #
class ObservationModel(nn.Module):
    """Correct the predicted position using the newly observed frame.

    sketch: cross-attention to all patch tokens stands in for a local cost-volume
    lookup around `pred_pos`. Reads out position correction, visibility logit and
    an aleatoric log-uncertainty used to down-weight ambiguous (homogeneous-tissue)
    matches in the loss.
    """

    def __init__(self, c_dim: int, ds: int) -> None:
        super().__init__()
        self.q = nn.Linear(ds, c_dim)                                       # query from state token
        self.cross = nn.MultiheadAttention(c_dim, num_heads=4, batch_first=True)
        self.head = nn.Linear(c_dim, 2 + 1 + 1)                             # (dx, dy, vis_logit, log_sigma)

    def forward(
        self, pred_pos: torch.Tensor, tokens: torch.Tensor, feats_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """pred_pos (B, N, 2), tokens (B, N, Ds), feats_next (B, C, Hf, Wf)."""
        b, c, hf, wf = feats_next.shape
        kv = feats_next.flatten(2).transpose(1, 2)                          # (B, Hf*Wf, C) frame tokens
        q = self.q(tokens)                                                  # (B, N, C)
        evidence, _ = self.cross(q, kv, kv)                                 # (B, N, C) matched evidence
        out = self.head(evidence)                                           # (B, N, 4)
        corr_pos = pred_pos + out[..., :2]                                  # (B, N, 2) corrected position
        vis_logit = out[..., 2:3]                                           # (B, N, 1)
        log_sigma = out[..., 3:4]                                           # (B, N, 1)
        return corr_pos, vis_logit, log_sigma


# --------------------------------------------------------------------------- #
# Full tracker (predict -> update recurrence)
# --------------------------------------------------------------------------- #
class ParticleTracker(nn.Module):
    def __init__(self, c_dim: int = 384, ds: int = 256, dh: int = 256, patch: int = 14) -> None:
        super().__init__()
        self.encoder = FrozenFrameEncoder(c_dim, patch)
        self.transition = TransitionModel(c_dim, ds, dh)
        self.observation = ObservationModel(c_dim, ds)
        self.dh = dh

    def init_state(self, query_xy: torch.Tensor, feats0: torch.Tensor, hw: tuple[int, int]) -> ParticleState:
        """query_xy (B, N, 2), feats0 (B, C, Hf, Wf) -> ParticleState at the query frame."""
        b, n, _ = query_xy.shape
        return ParticleState(
            pos=query_xy,                                                   # (B, N, 2)
            feat=sample_features(feats0, query_xy, hw),                     # (B, N, C) appearance memory
            hidden=torch.zeros(b, n, self.dh, device=query_xy.device),      # (B, N, Dh)
            vis_logit=torch.zeros(b, n, 1, device=query_xy.device),         # (B, N, 1)
        )

    def step(
        self, state: ParticleState, feats_next: torch.Tensor
    ) -> tuple[ParticleState, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One predict-update step -> (new_state, pos (B,N,2), vis_logit (B,N,1), log_sigma (B,N,1))."""
        pred_pos, new_hidden, tokens = self.transition(state)                      # prior
        corr_pos, vis_logit, log_sigma = self.observation(pred_pos, tokens, feats_next)  # posterior
        new_state = ParticleState(pos=corr_pos, feat=state["feat"], hidden=new_hidden, vis_logit=vis_logit)
        return new_state, corr_pos, vis_logit, log_sigma

    def forward(
        self, frames: torch.Tensor, query_xy: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """frames (B, T, 3, H, W), query_xy (B, N, 2) at t=0.

        Returns tracks (B, T, N, 2), vis_logit (B, T, N, 1), log_sigma (B, T, N, 1).
        """
        b, t, _, h, w = frames.shape
        feats = self.encoder(frames.flatten(0, 1)).unflatten(0, (b, t))     # (B, T, C, Hf, Wf)
        state = self.init_state(query_xy, feats[:, 0], (h, w))
        tracks = [state["pos"]]                                             # list of (B, N, 2)
        viss = [state["vis_logit"]]
        sigmas = [torch.zeros_like(state["vis_logit"])]
        # The scan over time is the intrinsic online recurrence; everything inside
        # step() is vectorised over (B, N). No per-point or per-batch python loop.
        for ti in range(1, t):
            state, pos, vis, log_sigma = self.step(state, feats[:, ti])
            tracks.append(pos)
            viss.append(vis)
            sigmas.append(log_sigma)
        return torch.stack(tracks, 1), torch.stack(viss, 1), torch.stack(sigmas, 1)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def tracking_loss(
    tracks: torch.Tensor,
    vis_logit: torch.Tensor,
    log_sigma: torch.Tensor,
    gt_tracks: torch.Tensor,
    gt_vis: torch.Tensor,
    huber_beta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Uncertainty-weighted position loss on visible points + visibility BCE.

    tracks (B, T, N, 2), vis_logit (B, T, N, 1), log_sigma (B, T, N, 1),
    gt_tracks (B, T, N, 2), gt_vis (B, T, N).  Returns (scalar loss, components).
    """
    assert tracks.shape == gt_tracks.shape, "track / target shape mismatch"
    vis_mask = gt_vis.unsqueeze(-1)                                         # (B, T, N, 1)
    huber = F.huber_loss(tracks, gt_tracks, reduction="none", delta=huber_beta).sum(-1, keepdim=True)  # (B,T,N,1)
    # Laplace negative log-likelihood: scale the position error by predicted uncertainty.
    pos_nll = (huber * torch.exp(-log_sigma) + log_sigma) * vis_mask        # (B, T, N, 1)
    pos_loss = pos_nll.sum() / vis_mask.sum().clamp_min(1.0)
    vis_loss = F.binary_cross_entropy_with_logits(vis_logit.squeeze(-1), gt_vis)
    total = pos_loss + vis_loss
    return total, {"pos": pos_loss.detach(), "vis": vis_loss.detach()}


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class SurgicalTAPDataset(Dataset):
    """Clip-level loader for surgical point tracking.

    A real item loads: decoded RGB frames, query points at the query frame, and
    per-frame target tracks + visibility. Targets come from STIR infrared markers
    (evaluation) or from teacher pseudo-labels (CoTracker3 / MFT) run on unlabeled
    surgical video (training). Random tensors stand in here so the shapes are explicit.
    """

    def __init__(self, num_clips: int, t: int = 24, n: int = 256, hw: tuple[int, int] = (224, 224)) -> None:
        self.num_clips = num_clips
        self.t = t
        self.n = n
        self.hw = hw

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h, w = self.hw
        scale = torch.tensor([w - 1, h - 1], dtype=torch.float32)
        # sketch: replace with real frame decode + label loading.
        return {
            "frames": torch.rand(self.t, 3, h, w),                          # (T, 3, H, W)
            "query_xy": torch.rand(self.n, 2) * scale,                      # (N, 2) at t=0
            "gt_tracks": torch.rand(self.t, self.n, 2) * scale,             # (T, N, 2)
            "gt_vis": (torch.rand(self.t, self.n) > 0.2).float(),           # (T, N)
        }


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(num_epochs: int = 1, batch_size: int = 2) -> ParticleTracker:
    model = ParticleTracker().to(DEVICE).float()
    params = [p for p in model.parameters() if p.requires_grad]            # excludes frozen encoder
    opt = torch.optim.AdamW(params, lr=2e-4)
    loader = DataLoader(SurgicalTAPDataset(num_clips=64), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(num_epochs):
        for batch in loader:                                               # default collate -> batch first
            frames = batch["frames"].to(DEVICE)                            # (B, T, 3, H, W)
            query_xy = batch["query_xy"].to(DEVICE)                        # (B, N, 2)
            gt_tracks = batch["gt_tracks"].to(DEVICE)                      # (B, T, N, 2)
            gt_vis = batch["gt_vis"].to(DEVICE)                            # (B, T, N)

            tracks, vis_logit, log_sigma = model(frames, query_xy)         # (B, T, N, 2), (B,T,N,1), (B,T,N,1)
            loss, _ = tracking_loss(tracks, vis_logit, log_sigma, gt_tracks, gt_vis)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


if __name__ == "__main__":
    train()
