"""TWIST world model.

The observation-corrected state-space point tracker (``TrackerWorldModel``), its
frozen encoders, loss, and TAP metrics. Re-exported here so
``utilities.config.create_model_from_config`` can resolve any class named in a
config via ``getattr(models, NAME)`` (with ``MODEL_MODULE: "models"``).

Normally built from a config, not instantiated directly:
``utilities.config.create_model_from_config`` / ``create_loss_from_config``.
"""

from .encoder import (
    DINOv3,
    FrozenFrameEncoder,
    denormalize_coords,
    normalize_coords,
    sample_features,
    sample_window,
)
from .losses import TrackerLoss, kl_diag_gauss, masked_mean
from .metrics import (
    finalize_recovery,
    merge_recovery_stats,
    recovery_metrics,
    tracking_metrics,
)
from .world_model import (
    ObservationModel,
    ParticleState,
    TrackerWorldModel,
    TransitionModel,
)

__all__ = [
    # encoders + sampling
    "FrozenFrameEncoder",
    "DINOv3",
    "normalize_coords",
    "denormalize_coords",
    "sample_features",
    "sample_window",
    # world model
    "TrackerWorldModel",
    "TransitionModel",
    "ObservationModel",
    "ParticleState",
    # loss + metrics
    "TrackerLoss",
    "kl_diag_gauss",
    "masked_mean",
    "tracking_metrics",
    "recovery_metrics",
    "merge_recovery_stats",
    "finalize_recovery",
]
