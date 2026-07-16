"""Differentiable point-cloud geometry, vendored from GateTracker.

This package holds the geometry primitives used by the pseudo-GT dense
point-tracking pipeline (``dataset.pseudo_gt``):

* :mod:`geometry.transforms`   — SE(3) pose construction (``euler2mat`` / ``mat2euler``).
* :mod:`geometry.projections`  — ``BackProject`` (RGB+depth+invK → 3-D cloud) and
  ``Project`` (warp cloud to a new pose → novel-view RGB + projected tracks).
* :mod:`geometry.pipeline`     — ``GeometryPipeline`` wrapping MoGe monocular
  depth/normal/intrinsics estimation.

``GeometryPipeline`` imports ``moge`` lazily (only when a MoGe model is actually
constructed), so ``geometry.projections`` / ``geometry.transforms`` — and the
novel-view warping path built on them — import fine without ``moge`` installed.
"""

from .transforms import euler2mat, mat2euler, Tdist
from .projections import BackProject, Project, Warp

__all__ = [
    "euler2mat",
    "mat2euler",
    "Tdist",
    "BackProject",
    "Project",
    "Warp",
]
