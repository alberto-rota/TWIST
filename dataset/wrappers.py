"""Dataset identity defaults and reader resolution.

``DATASET_DEFAULTS`` is the single source of truth for *where* each dataset
lives and *which reader* serves it. The config YAML then only needs to override
what varies for an experiment (sampling density, clip length, split, ...).

Add a dataset: drop an entry here (``ROOT_DIR`` + ``READER`` + any default
sampling), then list its name under ``DATASETS:`` in the config. The surgical
CoTracker datasets (Cholec80 / EndoTAPP / SurgT, served by the shared
``index.json`` reader) and STIR slot in the same way in a later step.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict

# Reserved key under DATASETS: overrides applied to *every* dataset (per-dataset
# keys take precedence). Mirrors the unreflectanything convention.
ALL_DATASETS_KEY = "ALL_DATASETS"
# Reserved key under DATASETS: forced overrides applied to *every* dataset that
# win OVER per-dataset keys (the opposite precedence to ALL_DATASETS). Use it to
# pin e.g. TARGET_SIZE / MAX_CLIPS across all datasets in one place.
OVERRIDE_ALL_DATASETS_KEY = "OVERRIDE_ALL_DATASETS"


# All readers expose the identical config-driven API (see
# dataset.base.BaseTracksDataset), so a dataset's only mandatory identity here is
# ROOT_DIR + READER; the listed sampling keys are per-dataset defaults that the
# config YAML overrides per experiment.
_COTRACKER_READER = "dataset.cotracker.CoTrackerTracksDataset"

DATASET_DEFAULTS: Dict[str, Dict[str, Any]] = {
    # Phase 1 — synthetic out-of-domain pretraining (dense GT tracks). CT3Kubric
    # is converted to the shared CoTracker layout by ``ct3kubric_data_prep.py``
    # and read through the same reader as every other dataset (no bespoke reader).
    "KUBRIC": {
        "ROOT_DIR": "/anvme/workspace/v120bb18-kubric/gt_tracks",
        "READER": _COTRACKER_READER,
        # Sensible training defaults; override per-experiment in the YAML.
        "CLIP_LEN": 24,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_FRACTION": 0.1,
        "SPLIT_SEED": 42,
    },
    # Long-horizon synthetic phase. Folder on disk is spelled "PointOdissey" and
    # ships ground-truth tracks under gt_tracks/ (same index.json + .npz layout).
    "POINTODYSSEY": {
        "ROOT_DIR": "$DATASET_DIR/PointOdissey/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": 48,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_FRACTION": 0.1,
        "SPLIT_SEED": 42,
    },
    # Long-horizon synthetic phase (sibling of PointOdyssey). Dynamic Replica
    # ships ground-truth tracks, converted to gt_tracks/ by
    # dynamicreplica_data_prep.py (same index.json + .npz layout).
    "DYNAMICREPLICA": {
        "ROOT_DIR": "$DATASET_DIR/DynamicReplica/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": 48,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_FRACTION": 0.1,
        "SPLIT_SEED": 42,
    },
    # Surgical adaptation datasets (pre-tracked offline, shared CoTracker layout).
    "CHOLEC80": {
        "ROOT_DIR": "$DATASET_DIR/cholec80/cotracker_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": 24,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_FRACTION": 0.1,
        "SPLIT_SEED": 42,
    },
    "ENDOTAPP": {
        "ROOT_DIR": "$DATASET_DIR/EndoTAPP/cotracker_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": 24,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_FRACTION": 0.1,
        "SPLIT_SEED": 42,
    },
    "SURGT": {
        "ROOT_DIR": "$DATASET_DIR/SurgT/cotracker_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": 24,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_FRACTION": 0.1,
        "SPLIT_SEED": 42,
    },

    "SURGICALMOTION": {
        "ROOT_DIR": "$DATASET_DIR/SurgicalMotion/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": 24,
        "FRAME_STRIDE": 1,
        "MAX_POINTS": 256,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "VAL_SEQUENCES": [
            "val_case2_1", "val_case2_2", "val_case2_4",
            "val_case3_1", "val_case3_2",
        ],
        "SPLIT_SEED": 42,
    },
    # --- Evaluation-only benchmarks (ground-truth point tracks) -----------
    # Converted from TAP-Vid pickles by tapvid_data_prep.py.
    # VAL_FRACTION=1.0 -> all sequences are validation, none are training.
    # IS_EVAL_DATASET=True marks them for the standalone benchmark evaluation
    # (utilities.evaluation): these are the datasets the headline TAP metrics are
    # reported on. The flag defaults False everywhere else, so training datasets
    # are never benchmarked unless a config explicitly opts them in.
    "TAPVID_DAVIS": {
        "ROOT_DIR": "$DATASET_DIR/tapvid_davis/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": None,          # whole sequence as one clip
        "MAX_POINTS": None,        # keep all GT points (~5 per video)
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "REQUIRE_VISIBLE_AT_QUERY": False,  # some GT points start occluded
        "VAL_FRACTION": 1.0,       # eval-only: all sequences -> val
        "SPLIT_SEED": 42,
        "IS_EVAL_DATASET": True,
    },
    "TAPVID_RGB_STACKING": {
        "ROOT_DIR": "$DATASET_DIR/tapvid_rgb_stacking/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": None,
        "MAX_POINTS": None,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "REQUIRE_VISIBLE_AT_QUERY": False,
        "VAL_FRACTION": 1.0,
        "SPLIT_SEED": 42,
        "IS_EVAL_DATASET": True,
    },
    # TAP-Vid-Kinetics: converted from the sharded generate_tapvid.py pickles by
    # tapvid_kinetics_data_prep.py (see assets/dataprep/TAPVID_KINETICS.md for the
    # download + processing runbook). Same eval-only conventions as DAVIS.
    "TAPVID_KINETICS": {
        "ROOT_DIR": "$DATASET_DIR/tapvid_kinetics/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": None,          # whole sequence as one clip
        "MAX_POINTS": None,        # keep all GT points
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "REQUIRE_VISIBLE_AT_QUERY": False,  # some GT points start occluded
        "VAL_FRACTION": 1.0,       # eval-only: all sequences -> val
        "SPLIT_SEED": 42,
        "IS_EVAL_DATASET": True,
    },
    "ROBOTAP": {
        "ROOT_DIR": "$DATASET_DIR/robotap/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": None,
        "MAX_POINTS": None,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "REQUIRE_VISIBLE_AT_QUERY": False,
        "VAL_FRACTION": 1.0,
        "SPLIT_SEED": 42,
        "IS_EVAL_DATASET": True,
    },
    "ENDOTAPP_GT": {
        "ROOT_DIR": "$DATASET_DIR/EndoTAPP/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": None,
        "MAX_POINTS": None,
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "REQUIRE_VISIBLE_AT_QUERY": False,
        "VAL_FRACTION": 1.0,
        "SPLIT_SEED": 42,
        "IS_EVAL_DATASET": True,
    },
  
    "VLSURGPT": {
        "ROOT_DIR": "$DATASET_DIR/VLsurgPT/gt_tracks",
        "READER": _COTRACKER_READER,
        "CLIP_LEN": None,          # whole (keyframe) sequence as one clip
        "MAX_POINTS": None,        # keep all GT points
        "POINT_SAMPLE_MODE": "even",
        "QUERY_FRAME": 0,
        "REQUIRE_VISIBLE_AT_QUERY": False,  # some GT points start occluded
        "VAL_FRACTION": 1.0,       # eval-only: all sequences -> val
        "SPLIT_SEED": 42,
        "IS_EVAL_DATASET": True,
    },
}


def reader_class_for(dataset_name: str, dataset_config: Dict[str, Any]):
    """Import and return the reader class for ``dataset_name``.

    ``READER`` in the per-dataset config wins; otherwise the registry default;
    otherwise the shared CoTracker reader. Accepts a dotted path
    (``"dataset.cotracker.CoTrackerTracksDataset"``).
    """
    spec = dataset_config.get("READER") or DATASET_DEFAULTS.get(dataset_name, {}).get("READER")
    if not spec:
        from dataset.cotracker import CoTrackerTracksDataset
        return CoTrackerTracksDataset
    module_path, _, class_name = spec.rpartition(".")
    return getattr(importlib.import_module(module_path), class_name)
