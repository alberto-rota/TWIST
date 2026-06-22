"""TWIST datasets.

The canonical tracking item dict (shared by every reader so they are mutually
drop-in)::

    frames      (T, 3, H, W)  uint8 | float[0,1]   (optional)
    tracks      (T, N, 2)     float32  pixel coords (x, y)
    visibility  (T, N)        bool
    queries     (N, 3)        float32  (t, x, y) at the query frame
    frame_size  (2,)          long     final (H, W)
    video       str
    clip_idx    int
    depths      (T, H, W)     float32  (optional)

Datasets are normally built from a config via
:func:`utilities.config.create_datasets_from_config`, not instantiated directly.
"""

from .base import BaseTracksDataset
from .cotracker import (
    CoTrackerTracksDataset,
    list_sequences,
    list_sequences as list_cotracker_sequences,
)
from .sampling import select_point_indices, candidate_mask, POINT_SAMPLE_MODES
from .splits import split_sequences
from .collate import pad_collate, is_fixed_shape
from .wrappers import DATASET_DEFAULTS, ALL_DATASETS_KEY, reader_class_for

__all__ = [
    "BaseTracksDataset",
    "CoTrackerTracksDataset",
    "list_sequences",
    "list_cotracker_sequences",
    "select_point_indices",
    "candidate_mask",
    "POINT_SAMPLE_MODES",
    "split_sequences",
    "pad_collate",
    "is_fixed_shape",
    "DATASET_DEFAULTS",
    "ALL_DATASETS_KEY",
    "reader_class_for",
]
