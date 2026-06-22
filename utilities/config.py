"""Config loading and dataset/dataloader construction.

Mirrors the unreflectanything workflow so the same YAML drives both a normal
``python train.py`` run and a W&B sweep:

* configs are written in **W&B-sweep format** -- ``parameters: {KEY: {value: ...}}``
  -- and :func:`load_and_process_config` flattens that to ``{KEY: value}``.
  A sweep agent instead passes ``wandb.config`` (already ``{KEY: value}``)
  straight in as ``config=``.
* CLI ``--KEY=value`` (and dotted ``--DATASETS.KUBRIC.MAX_POINTS=64``) override
  any field, type-coerced to the existing value's type.
* ``boot_mode`` shrinks everything for a fast smoke test.

:func:`create_datasets_from_config` turns the ``DATASETS`` block into train /
val / test datasets (sequence-level split + sampling), and
:func:`build_dataloaders` wraps them in ``DataLoader``s.
"""

from __future__ import annotations

import ast
import copy
import importlib
import inspect
from typing import Any, Dict, List, Optional

import yaml
from torch.utils.data import ConcatDataset, DataLoader

from dataset.collate import is_fixed_shape, pad_collate
from dataset.splits import split_sequences
from dataset.wrappers import ALL_DATASETS_KEY, DATASET_DEFAULTS, reader_class_for
from utilities.env import expand_path
from utilities.log import get_logger

logger = get_logger(__name__).set_context("CONFIG")

try:
    from dotmap import DotMap
except Exception:  # pragma: no cover - dotmap is a declared dep
    class DotMap(dict):  # minimal fallback
        def __getattr__(self, k):
            v = self.get(k)
            return DotMap(v) if isinstance(v, dict) else v

        def __setattr__(self, k, v):
            self[k] = v

        def toDict(self):
            return dict(self)


# Config keys that configure the train/val *split* or dataset identity, and so
# are consumed here rather than forwarded to the reader constructor.
_SPLIT_KEYS = {
    "ROOT_DIR", "READER", "VAL_FRACTION", "SPLIT_SEED",
    "VAL_SEQUENCES", "TRAIN_SEQUENCES", "MAX_SEQUENCES",
}


def _as_dict(x: Any) -> dict:
    """Plain-dict view of a DotMap / dict (recursively for DotMap)."""
    if x is None:
        return {}
    if hasattr(x, "toDict") and callable(x.toDict):
        return x.toDict()
    return dict(x)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def _coerce(value: str, like: Any) -> Any:
    """Coerce a CLI string to the type of an existing value (best effort)."""
    if isinstance(like, bool):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot parse bool from {value!r}")
    if isinstance(like, (list, tuple, dict)):
        return ast.literal_eval(value)
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if like is None:
        try:
            return ast.literal_eval(value)
        except Exception:  # noqa: BLE001
            return value
    return type(like)(value)


def _apply_dotted_override(config_dict: dict, dotted_key: str, value: str) -> None:
    """Apply ``A.B.C=value`` into a nested dict, coercing to the existing type."""
    parts = dotted_key.split(".")
    node = config_dict
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    leaf = parts[-1]
    existing = node.get(leaf)
    try:
        node[leaf] = _coerce(value, existing)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not coerce {dotted_key}={value!r}: {e}")
        node[leaf] = value


def load_and_process_config(
    config_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    unknown_args: Optional[List[str]] = None,
    boot_mode: bool = False,
) -> "DotMap":
    """Load a YAML config (or accept a dict), apply CLI overrides + boot mode.

    Args:
        config_path: path to a ``parameters: {KEY: {value: ...}}`` YAML.
        config: a flat ``{KEY: value}`` dict (e.g. ``wandb.config``); if given,
            ``config_path`` is ignored.
        unknown_args: leftover ``--KEY=value`` / ``--KEY value`` CLI tokens.
        boot_mode: shrink batch/epochs/datasets for a quick smoke test.
    """
    if config is not None:
        config_dict = dict(config)
    else:
        if config_path is None:
            raise ValueError("provide either config_path or config")
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        params = raw.get("parameters", raw) if isinstance(raw, dict) else {}
        config_dict = {
            k: (v.get("value") if isinstance(v, dict) and "value" in v else v)
            for k, v in params.items()
        }

    # CLI overrides: --KEY=value | --KEY value | --A.B.C=value
    if unknown_args:
        i = 0
        while i < len(unknown_args):
            arg = unknown_args[i]
            if arg.startswith("--"):
                body = arg[2:]
                if "=" in body:
                    key, val = body.split("=", 1)
                elif i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                    key, val = body, unknown_args[i + 1]
                    i += 1
                else:
                    key, val = body, "true"
                if "." in key:
                    _apply_dotted_override(config_dict, key, val)
                else:
                    key = key.upper()
                    if key in config_dict:
                        try:
                            config_dict[key] = _coerce(val, config_dict[key])
                        except Exception as e:  # noqa: BLE001
                            logger.warning(f"could not coerce {key}={val!r}: {e}")
                    else:
                        try:
                            config_dict[key] = ast.literal_eval(val)
                        except Exception:  # noqa: BLE001
                            config_dict[key] = val
            i += 1

    # Fold any dotted keys into the nested structure. W&B sweeps deliver nested
    # overrides as flat ``DATASETS.KUBRIC.MAX_POINTS`` keys; this makes sampling
    # params sweepable without flattening the whole config.
    for key in [k for k in config_dict if isinstance(k, str) and "." in k]:
        _apply_dotted_override(config_dict, key, str(config_dict.pop(key)))

    if boot_mode:
        config_dict["BATCH_SIZE"] = 1
        config_dict["EPOCHS"] = 1
        config_dict["NO_WANDB"] = True
        for name, dcfg in (config_dict.get("DATASETS") or {}).items():
            if isinstance(dcfg, dict):
                # force caps even when the key is present-but-null in the YAML
                if dcfg.get("MAX_SEQUENCES") is None:
                    dcfg["MAX_SEQUENCES"] = 4
                if dcfg.get("MAX_CLIPS_PER_VIDEO") is None:
                    dcfg["MAX_CLIPS_PER_VIDEO"] = 2
                dcfg["MAX_POINTS"] = min(dcfg.get("MAX_POINTS") or 64, 64)
                dcfg["CLIP_LEN"] = min(dcfg.get("CLIP_LEN") or 8, 8)
        # Force a tiny, no-download CNN encoder so boot never hits the network /
        # instantiates DINOv3 (the login node is CPU-only).
        mc = config_dict.setdefault("MODEL", {})
        if isinstance(mc, dict):
            enc = mc.setdefault("RGB_ENCODER", {})
            enc.update({"ENCODER": "cnn", "FEATURE_DIM": 32, "PATCH_SIZE": 8,
                        "FREEZE_BACKBONE": False, "RGB_ENCODER_LR": 1.0e-3})
            mc["HIDDEN_DIM"] = min(mc.get("HIDDEN_DIM") or 64, 64)
            mc["TOKEN_DIM"] = min(mc.get("TOKEN_DIM") or 64, 64)
            mc.setdefault("OBSERVATION", {}).update({"K": 7, "HEADS": 2})
            mc.setdefault("TRANSITION", {}).update({"DEPTH": 1, "HEADS": 2})
        logger.info("boot mode: minimal batch/epochs/datasets + cnn encoder for a quick smoke test")

    return DotMap(config_dict)


# --------------------------------------------------------------------------- #
# Training schedule (phases / stages)
# --------------------------------------------------------------------------- #
# Training runs in ordered **stages** (Phase 1 Kubric pretrain -> Phase 2
# PointOdyssey -> surgical adaptation -> ...). A stage is expressed as a
# top-level override block under ``STAGES``; resolving a stage shallow-merges
# that block onto the base config, so each phase can swap ``DATASETS`` (and
# later its model/loss/freeze settings) while sharing everything else. A run
# that finishes one stage is resumed and continued at the next (see
# utilities.runs). Configs without ``STAGES`` are a single implicit stage.
STAGE_INDEX_KEY = "STAGE_INDEX"
STAGE_NAME_KEY = "STAGE_NAME"


def get_stages(config: "DotMap") -> List[dict]:
    """Ordered list of stage-override dicts. One implicit stage if no ``STAGES``."""
    stages = config.get("STAGES")
    if not stages:
        return [{"NAME": config.get("EXPERIMENT_NAME", "stage0")}]
    return [_as_dict(s) for s in stages]


def resolve_stage_config(config: "DotMap", stage_idx: int) -> "DotMap":
    """Config for one stage: base config with ``STAGES[stage_idx]`` overlaid.

    Top-level keys in the stage block replace the base value (so a stage's
    ``DATASETS`` fully defines that phase's data). Adds ``STAGE_INDEX`` /
    ``STAGE_NAME`` and carries ``EXPERIMENT_NAME`` (the run identity) through.
    """
    stages = get_stages(config)
    if not (0 <= stage_idx < len(stages)):
        raise IndexError(f"stage {stage_idx} out of range (have {len(stages)})")
    base = _as_dict(config)
    base.pop("STAGES", None)
    merged = {**copy.deepcopy(base), **copy.deepcopy(stages[stage_idx])}
    merged[STAGE_INDEX_KEY] = stage_idx
    merged[STAGE_NAME_KEY] = stages[stage_idx].get("NAME", f"stage{stage_idx}")
    merged.setdefault("EXPERIMENT_NAME", config.get("EXPERIMENT_NAME", "run"))
    return DotMap(merged)


# --------------------------------------------------------------------------- #
# Dataset creation
# --------------------------------------------------------------------------- #
def _reader_kwargs(cfg: dict, reader_cls) -> dict:
    """Map UPPER_CASE config keys to the reader's snake_case constructor kwargs.

    Only keys the reader actually accepts are forwarded, so the same config can
    target readers with different signatures. ``CLIP_LEN: 0`` means "one clip
    per sequence" (-> ``clip_len=None``).
    """
    def norm(v):
        return tuple(v) if isinstance(v, list) else v

    candidate = {k.lower(): norm(v) for k, v in cfg.items() if k not in _SPLIT_KEYS}
    if candidate.get("clip_len") in (0, None):
        candidate["clip_len"] = None
    accepted = set(inspect.signature(reader_cls.__init__).parameters)
    return {k: v for k, v in candidate.items() if k in accepted}


def _list_sequences(reader_cls, root: str) -> List[str]:
    """Find the reader module's ``list_sequences`` helper, else scan subdirs."""
    mod = importlib.import_module(reader_cls.__module__)
    if hasattr(mod, "list_sequences"):
        return list(mod.list_sequences(root))
    from pathlib import Path
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir())


def create_datasets_from_config(
    config: "DotMap", dataset_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Build train / val / test datasets from ``config.DATASETS``.

    For each dataset: merge ``DATASET_DEFAULTS`` < ``ALL_DATASETS`` overrides <
    per-dataset config; resolve the reader + root; split sequences (explicit
    lists or fractional+seed); instantiate the train and val readers. Validation
    forces deterministic point sampling for reproducible metrics.

    Returns ``{"Training", "Validation", "Test", "workers"}`` where each split
    is a ``ConcatDataset`` (or ``None``).
    """
    datasets_cfg = _as_dict(config.get("DATASETS"))
    if not datasets_cfg:
        raise ValueError("config has no DATASETS section")

    all_overrides = _as_dict(datasets_cfg.get(ALL_DATASETS_KEY))
    if all_overrides:
        logger.info(f"applying ALL_DATASETS overrides: {all_overrides}")

    names = dataset_names or [
        n for n, c in datasets_cfg.items()
        if n != ALL_DATASETS_KEY and isinstance(_as_dict(c), dict) and _as_dict(c)
    ]
    if not names:
        raise ValueError("no datasets listed under DATASETS")

    train_sets, val_sets = [], []
    logger.info(f"building {len(names)} dataset(s): {names}")

    for name in names:
        # merge: registry defaults < ALL_DATASETS < per-dataset
        merged = dict(DATASET_DEFAULTS.get(name, {}))
        merged.update(all_overrides)
        merged.update(_as_dict(datasets_cfg.get(name)))

        reader_cls = reader_class_for(name, merged)
        root = expand_path(merged.get("ROOT_DIR", f"$DATASET_DIR/{name}"))

        all_seqs = _list_sequences(reader_cls, root)
        if not all_seqs:
            logger.warning(f"  {name}: no sequences found at {root} -- skipping")
            continue
        max_seqs = merged.get("MAX_SEQUENCES")
        if max_seqs is not None:
            all_seqs = all_seqs[: int(max_seqs)]

        train_seqs, val_seqs = split_sequences(
            all_seqs,
            val_fraction=merged.get("VAL_FRACTION", 0.1),
            seed=merged.get("SPLIT_SEED", 42),
            train_sequences=merged.get("TRAIN_SEQUENCES"),
            val_sequences=merged.get("VAL_SEQUENCES"),
        )

        kw = _reader_kwargs(merged, reader_cls)
        if train_seqs:
            train_sets.append(reader_cls(root=root, include=train_seqs, **kw))
            logger.info(
                f"  ✓ {name} train: {len(train_seqs)} seqs -> {len(train_sets[-1])} clips"
                f"  (N={kw.get('max_points')}, clip_len={kw.get('clip_len')})"
            )
        if val_seqs:
            val_kw = dict(kw)
            if val_kw.get("point_sample_mode") == "random":
                val_kw["point_sample_mode"] = "even"  # reproducible val metrics
            val_sets.append(reader_cls(root=root, include=val_seqs, **val_kw))
            logger.info(
                f"  ✓ {name} val:   {len(val_seqs)} seqs -> {len(val_sets[-1])} clips"
            )

    training = ConcatDataset(train_sets) if train_sets else None
    validation = ConcatDataset(val_sets) if val_sets else None
    logger.info(
        f"=== totals: train={len(training) if training else 0} clips, "
        f"val={len(validation) if validation else 0} clips ==="
    )
    return {
        "Training": training,
        "Validation": validation,
        "Test": validation,
        "workers": int(config.get("WORKERS", 4)),
    }


# --------------------------------------------------------------------------- #
# DataLoaders
# --------------------------------------------------------------------------- #
def _collate_for(dataset):
    """Default collate when clips are fixed-shape; padding collate otherwise."""
    sets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    return None if all(is_fixed_shape(s) for s in sets) else pad_collate


def build_dataloaders(config: "DotMap", datasets: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap the datasets dict in ``DataLoader``s.

    Train shuffles and drops the last partial batch; val/test do neither.
    Picks the padding collate automatically for variable-length clips.
    """
    batch_size = int(config.get("BATCH_SIZE", 4))
    workers = int(datasets.get("workers", config.get("WORKERS", 4)))
    loaders: Dict[str, Any] = {}

    specs = [("train", "Training", True), ("val", "Validation", False), ("test", "Test", False)]
    for key, dkey, is_train in specs:
        ds = datasets.get(dkey)
        if ds is None or len(ds) == 0:
            loaders[key] = None
            continue
        loaders[key] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=workers,
            drop_last=is_train and len(ds) >= batch_size,
            pin_memory=config.get("PIN_MEMORY", False),
            persistent_workers=workers > 0,
            collate_fn=_collate_for(ds),
        )
    return loaders


# --------------------------------------------------------------------------- #
# Model + loss construction (mirrors unreflectanything's create_model_from_config)
# --------------------------------------------------------------------------- #
def _target_size_from_config(config: "DotMap"):
    """First dataset's TARGET_SIZE (square side), default 256."""
    for dcfg in _as_dict(config.get("DATASETS")).values():
        ds = _as_dict(dcfg)
        if ds.get("TARGET_SIZE"):
            ts = ds["TARGET_SIZE"]
            return tuple(ts) if isinstance(ts, (list, tuple)) else (int(ts), int(ts))
    return (256, 256)


def create_model_from_config(config: "DotMap", device, verbose: bool = True):
    """Build the world model from ``config.MODEL`` (mirrors unreflectanything).

    Dynamically imports ``MODEL_MODULE`` (default ``"models"``), builds the frozen
    encoder from the ``RGB_ENCODER`` block (``ENCODER`` is a DINOv3 HF id, or
    ``"cnn"`` for the no-download CPU fallback; frozen when ``FREEZE_BACKBONE`` or
    ``RGB_ENCODER_LR == 0``), then instantiates ``MODEL_CLASS`` (default
    ``TrackerWorldModel``) via ``getattr``.
    """
    import torch  # noqa: F401

    mc = _as_dict(config.get("MODEL"))
    models_module = importlib.import_module(mc.get("MODEL_MODULE", "models"))
    target_size = _target_size_from_config(config)

    enc = _as_dict(mc.get("RGB_ENCODER"))
    encoder_name = enc.get("ENCODER", "facebook/dinov3-vitl16-pretrain-lvd1689m")
    encoder_lr = enc.get("RGB_ENCODER_LR", 0.0)
    variant = "cnn" if str(encoder_name).lower() == "cnn" else "dino"
    encoder_cfg = {
        "variant": variant,
        "model_name": encoder_name,
        "image_size": int(enc.get("IMAGE_SIZE", min(target_size))),
        "feature_dim": int(enc.get("FEATURE_DIM", 64)),       # used by cnn; dino reads its own
        "patch_size": int(enc.get("PATCH_SIZE", 8)),          # used by cnn
        "freeze_backbone": bool(enc.get("FREEZE_BACKBONE", True)),
        "encoder_lr": encoder_lr,
    }
    encoder = models_module.FrozenFrameEncoder(encoder_cfg).to(device)

    obs = _as_dict(mc.get("OBSERVATION"))
    trans = _as_dict(mc.get("TRANSITION"))
    heads = _as_dict(mc.get("HEADS"))
    model_class = getattr(models_module, mc.get("MODEL_CLASS", "TrackerWorldModel"))
    model = model_class(
        encoder=encoder,
        hidden_dim=int(mc.get("HIDDEN_DIM", 256)),
        token_dim=int(mc.get("TOKEN_DIM", 256)),
        obs_k=int(obs.get("K", 7)),
        obs_radius_px=float(obs.get("RADIUS_PX", 24.0)),
        obs_heads=int(obs.get("HEADS", 4)),
        obs_max_corr=float(obs.get("MAX_CORR", 0.0)),
        trans_heads=int(trans.get("HEADS", 4)),
        trans_depth=int(trans.get("DEPTH", 2)),
        trans_max_step=float(trans.get("MAX_STEP", 0.12)),
        uncertainty=bool(heads.get("UNCERTAINTY", True)),
        encode_chunk=int(mc.get("ENCODE_CHUNK", 32)),
        verbose=verbose,
    ).to(device)

    if verbose:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        frozen = encoder_cfg["freeze_backbone"] or encoder_lr in (0, 0.0, None)
        logger.info(
            f"model {model.__class__.__name__}: {n_train:,} trainable / {n_total:,} total params "
            f"(encoder={variant}, dim={encoder.feature_dim}, frozen={frozen})"
        )
    return model


def create_loss_from_config(config: "DotMap", device=None):
    """Build the training loss from ``config.MODEL.LOSS``."""
    mc = _as_dict(config.get("MODEL"))
    lc = _as_dict(mc.get("LOSS"))
    models_module = importlib.import_module(mc.get("MODEL_MODULE", "models"))
    loss = models_module.TrackerLoss(
        pos_weight=float(lc.get("POS_WEIGHT", 10.0)),
        vis_weight=float(lc.get("VIS_WEIGHT", 0.5)),
        kl_weight=float(lc.get("KL_WEIGHT", 0.05)),
        kl_free_bits=float(lc.get("KL_FREE_BITS", 0.5)),
        kl_balance_alpha=float(lc.get("KL_BALANCE_ALPHA", 0.8)),
        huber_delta=float(lc.get("HUBER_DELTA", 0.2)),
        unc_weight=float(lc.get("UNC_WEIGHT", 0.0)),
        prior_weight=float(lc.get("PRIOR_WEIGHT", 0.5)),
    )
    return loss.to(device) if device is not None else loss
