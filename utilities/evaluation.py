"""Standalone benchmark evaluation for the TWIST tracker.

Computes the headline TAP metrics on the **evaluation datasets** -- the ones a
config flags with ``IS_EVAL_DATASET: True`` (default ``False``) -- and reports
their per-dataset mean. The reported metrics are:

  * ``EPE``                 (``epe``)                 mean L2 endpoint error (px), lower better
  * ``Delta AVG``           (``delta_avg``)           position accuracy, higher better
  * ``Average Jaccard``     (``average_jaccard``)     position+visibility, higher better
  * ``Occlusion Accuracy``  (``occlusion_accuracy``)  visibility match, higher better
  * ``Time (ms/frame)``     (``ms_per_frame``)        wall-clock inference cost / frame

The first four reuse :func:`models.metrics.tracking_metrics` (the *same*
definitions the engine monitors with), so eval numbers and the training
``val/epe`` family are directly comparable. ``ms_per_frame`` is measured around
the model forward (CUDA-synchronised), warm-up batch excluded.

Two ways in -- both go through the same scoring code:

  * **Standalone**   :func:`evaluate_checkpoint` loads a trained run's ``.pt``,
    rebuilds the model from the checkpoint's config, evaluates, writes the CSV
    (and optionally a W&B table). Driven by ``evaluate.py``.
  * **From training** the engine calls :func:`evaluate_and_report` with the
    live model at the end of a stage (``EVAL_AT_END``) or every few validation
    epochs (``EVAL_EVERY``) for monitoring.

Outputs: a CSV under the run dir (rows = datasets + a ``MEAN`` row, cols =
metrics) and, when a W&B run is active, a ``wandb.Table`` (same shape) plus
per-dataset/mean scalars for time-series tracking.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from dataset.wrappers import (
    ALL_DATASETS_KEY,
    DATASET_DEFAULTS,
    OVERRIDE_ALL_DATASETS_KEY,
    reader_class_for,
)
from models import tracking_metrics
from utilities.config import (
    _as_dict,
    _collate_for,
    _list_sequences,
    _reader_kwargs,
    create_model_from_config,
    load_and_process_config,
)
from utilities.env import expand_path
from utilities.log import get_logger

logger = get_logger(__name__).set_context("EVAL")

# Per-dataset config flag (default False) that opts a dataset into evaluation.
IS_EVAL_DATASET_KEY = "IS_EVAL_DATASET"

# Reported metrics: machine key -> human header (CSV / W&B table columns).
# Order here is the column order everywhere.
METRIC_KEYS = ["epe", "delta_avg", "average_jaccard", "occlusion_accuracy", "ms_per_frame"]
METRIC_HEADERS = {
    "epe": "EPE (px)",
    "delta_avg": "Delta AVG",
    "average_jaccard": "Average Jaccard",
    "occlusion_accuracy": "Occlusion Accuracy",
    "ms_per_frame": "Time (ms/frame)",
}
# The quality metrics come from tracking_metrics; ms_per_frame is timed here.
_QUALITY_KEYS = ["epe", "delta_avg", "average_jaccard", "occlusion_accuracy"]


def _nanmean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and x == x]  # drop None / NaN
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# Dataset selection + construction
# --------------------------------------------------------------------------- #
def _merged_dataset_cfg(name: str, datasets_cfg: dict) -> dict:
    """Merge registry defaults < ALL_DATASETS < per-dataset < OVERRIDE_ALL_DATASETS
    for one dataset (the same precedence as create_datasets_from_config)."""
    merged = dict(DATASET_DEFAULTS.get(name, {}))
    merged.update(_as_dict(datasets_cfg.get(ALL_DATASETS_KEY)))
    merged.update(_as_dict(datasets_cfg.get(name)))
    merged.update(_as_dict(datasets_cfg.get(OVERRIDE_ALL_DATASETS_KEY)))
    return merged


def select_eval_datasets(config: Any) -> List[str]:
    """Names of datasets to evaluate: any whose merged config has
    ``IS_EVAL_DATASET`` truthy.

    Candidates are the registry datasets (``DATASET_DEFAULTS``) plus any extra
    datasets named under ``config.DATASETS`` -- so the eval-only benchmarks are
    picked up by default (they carry the flag in the registry), and a config can
    additionally flag a training dataset, or turn one off with
    ``IS_EVAL_DATASET: False``.
    """
    datasets_cfg = _as_dict(config.get("DATASETS"))
    reserved = {ALL_DATASETS_KEY, OVERRIDE_ALL_DATASETS_KEY}
    candidates = set(DATASET_DEFAULTS) | {n for n in datasets_cfg if n not in reserved}
    selected = [n for n in sorted(candidates)
                if bool(_merged_dataset_cfg(n, datasets_cfg).get(IS_EVAL_DATASET_KEY, False))]
    return selected


def build_eval_dataset(name: str, config: Any, max_clips: Optional[int] = None):
    """Build a reader over **all** sequences of ``name`` (no train/val split:
    evaluation scores the whole dataset). Forces deterministic ``even`` point
    sampling for reproducible metrics. Returns ``None`` if the data is absent.
    """
    datasets_cfg = _as_dict(config.get("DATASETS"))
    merged = _merged_dataset_cfg(name, datasets_cfg)
    reader_cls = reader_class_for(name, merged)
    root = expand_path(merged.get("ROOT_DIR", f"$DATASET_DIR/{name}"))

    seqs = _list_sequences(reader_cls, root)
    if not seqs:
        logger.warning(f"  {name}: no sequences found at {root} -- skipping")
        return None
    max_seqs = merged.get("MAX_SEQUENCES")
    if max_seqs is not None:
        seqs = seqs[: int(max_seqs)]

    kw = _reader_kwargs(merged, reader_cls)
    if kw.get("point_sample_mode") == "random":
        kw["point_sample_mode"] = "even"          # reproducible eval metrics
    if max_clips is not None:
        kw["max_clips"] = int(max_clips)

    ds = reader_cls(root=root, include=seqs, **kw)
    ds.dataset_name = name
    return ds


# --------------------------------------------------------------------------- #
# Per-dataset scoring
# --------------------------------------------------------------------------- #
def _amp_settings(device: torch.device, amp: Optional[bool], amp_dtype):
    """Mirror the engine's AMP policy: bf16 on bf16-capable CUDA, else fp16, off on CPU."""
    use_amp = (device.type == "cuda") if amp is None else (amp and device.type == "cuda")
    if amp_dtype is None:
        bf16 = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16 else torch.float16
    return use_amp, amp_dtype


# Benchmark query protocols (how each GT point's query frame is chosen).
QUERY_MODES = ("first", "frame0")


@torch.no_grad()
def _first_visible_eval(model, frames, gt_tracks, gt_vis, point_mask, *, autocast_ctx):
    """TAP-Vid **"queried first"** forward for a batch of clips.

    Each GT point is queried at its **first visible frame** and scored only on the
    frames **strictly after** that frame (the standard TAP-Vid ``evaluation_points``
    for ``first`` mode). The TWIST model takes a single query frame per forward, so
    points that share a first-visible frame are grouped and run together (one forward
    per distinct first-visible frame), then scattered back into per-point arrays.
    Points never visible (or padded out by ``point_mask``) are left out of the eval
    mask entirely. This is the protocol that makes our numbers comparable to
    CoTracker's — and it sidesteps the ``(0,0)`` occluded-coordinate placeholder ever
    being used as a *query* (the real harm of those placeholders; see CLAUDE.md).

    Returns ``(coords, vis_logits, eval_mask)`` shaped ``(B,T,N,2)/(B,T,N)/(B,T,N)``.
    """
    B, T = frames.shape[:2]
    N = gt_vis.shape[2]
    device = frames.device
    gt_vis_b = gt_vis.bool()
    pm = point_mask.bool() if point_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)

    coords = gt_tracks.clone().float()                          # overwritten per group
    vis_logits = torch.full((B, T, N), -10.0, device=device, dtype=torch.float32)
    eval_mask = torch.zeros((B, T, N), dtype=torch.bool, device=device)

    for b in range(B):
        vis = gt_vis_b[b]                                       # (T,N)
        usable = vis.any(0) & pm[b]                             # (N,) visible somewhere & real
        if not usable.any():
            continue
        first_vis = vis.float().argmax(0)                      # (N,) first visible frame (0 if none)
        for f in torch.unique(first_vis[usable]).tolist():
            f = int(f)
            idx = ((first_vis == f) & usable).nonzero(as_tuple=True)[0]   # points queried at f
            q_xy = gt_tracks[b, f, idx].float()                # (n_g,2) the (visible) query coord
            t_col = torch.full((idx.numel(), 1), float(f), device=device)
            queries = torch.cat([t_col, q_xy], dim=-1).unsqueeze(0).float()   # (1,n_g,3)
            with autocast_ctx():
                out = model(frames[b:b + 1], queries)
            coords[b, :, idx] = out["coords"][0].float()
            vis_logits[b, :, idx] = out["vis_logits"][0].float()
            if f + 1 < T:
                eval_mask[b, f + 1:, idx] = True               # score strictly after the query frame
    return coords, vis_logits, eval_mask


@torch.no_grad()
def evaluate_model_on_dataset(
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    amp: Optional[bool] = None,
    amp_dtype=None,
    max_steps: int = 0,
    query_frame: int = 0,
    query_mode: str = "first",
) -> Dict[str, float]:
    """Run ``model`` over ``dataset`` and return the reported metric means.

    Quality metrics are averaged per batch (NaN-safe), matching the engine's
    ``validate``. ``ms_per_frame`` = total forward wall-clock / real frames
    processed; on CUDA a warm-up forward runs first (outside timing) so cudnn
    autotune isn't charged to it. Returns ``delta_avg``, ``average_jaccard``,
    ``occlusion_accuracy``, ``ms_per_frame`` plus ``n_clips`` / ``n_frames``.

    ``query_mode``: ``"first"`` (TAP-Vid "queried first" — each point queried at its
    first visible frame, scored only after it; the comparable-to-CoTracker default)
    or ``"frame0"`` (legacy — all points queried at frame 0, every frame scored).
    Timing (``ms_per_frame``) always uses the single all-points-at-frame-0 forward,
    so it reflects realistic one-pass inference cost regardless of ``query_mode``.
    """
    model.eval()
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {query_mode!r}")
    use_amp, amp_dtype = _amp_settings(device, amp, amp_dtype)

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False, collate_fn=_collate_for(dataset),
    )

    agg: Dict[str, float] = {k: 0.0 for k in _QUALITY_KEYS}
    cnt: Dict[str, int] = {k: 0 for k in _QUALITY_KEYS}
    compute_s = 0.0
    timed_frames = 0
    n_clips = 0
    n_frames = 0

    # CUDA warm-up: run one forward *outside* the timed loop so cudnn autotune /
    # lazy init isn't charged to the first batch's ms/frame. Every batch in the
    # loop below is then timed, so ms_per_frame is always populated.
    if device.type == "cuda":
        try:
            wb = next(iter(loader))
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                model(wb["frames"].to(device), wb["queries"].float().to(device),
                      point_mask=(wb["point_mask"].to(device) if wb.get("point_mask") is not None else None))
            torch.cuda.synchronize(device)
        except Exception:  # noqa: BLE001 -- warm-up is best-effort
            pass

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        frames = batch["frames"].to(device, non_blocking=True)
        queries = batch["queries"].float().to(device, non_blocking=True)
        gt_tracks = batch["tracks"].float().to(device, non_blocking=True)
        gt_vis = batch["visibility"].to(device, non_blocking=True)
        time_mask = batch.get("time_mask")
        point_mask = batch.get("point_mask")
        if time_mask is not None:
            time_mask = time_mask.to(device, non_blocking=True)
        if point_mask is not None:
            point_mask = point_mask.to(device, non_blocking=True)

        # real (unpadded) frames in this batch -> the per-frame timing denominator
        nf = int(time_mask.sum().item()) if time_mask is not None else int(frames.shape[0] * frames.shape[1])

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with autocast_ctx():
            out = model(frames, queries, point_mask=point_mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        compute_s += time.perf_counter() - t0
        timed_frames += nf

        # Quality metrics: TAP-Vid "first" (query each point at its first visible
        # frame, score only after it) or legacy "frame0" (reuse the timed forward).
        if query_mode == "first":
            e_coords, e_vislog, e_mask = _first_visible_eval(
                model, frames, gt_tracks, gt_vis, point_mask, autocast_ctx=autocast_ctx)
            m = tracking_metrics(e_coords, gt_tracks, e_vislog, gt_vis,
                                 eval_mask=e_mask, query_frame=query_frame)
        else:
            m = tracking_metrics(out["coords"], gt_tracks, out["vis_logits"], gt_vis,
                                 time_mask, point_mask, query_frame=query_frame)
        for k in _QUALITY_KEYS:
            v = m.get(k, float("nan"))
            if v == v:                                  # NaN-safe
                agg[k] += v
                cnt[k] += 1
        n_clips += int(frames.shape[0])
        n_frames += nf

    result = {k: (agg[k] / cnt[k] if cnt[k] else float("nan")) for k in _QUALITY_KEYS}
    result["ms_per_frame"] = (1e3 * compute_s / timed_frames) if timed_frames else float("nan")
    result["n_clips"] = n_clips
    result["n_frames"] = n_frames
    return result


# --------------------------------------------------------------------------- #
# Multi-dataset evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    config: Any,
    device: torch.device,
    *,
    dataset_names: Optional[List[str]] = None,
    max_clips: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    amp: Optional[bool] = None,
    amp_dtype=None,
    max_steps: int = 0,
    query_frame: int = 0,
    query_mode: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate ``model`` on every selected eval dataset.

    Returns ``{dataset_name: metrics, ..., "MEAN": metrics}`` where ``MEAN`` is
    the across-dataset mean of each metric (NaN-safe). Datasets whose data is not
    present on disk are skipped with a warning. ``query_mode`` defaults to the
    config's ``EVAL_QUERY_MODE`` (``"first"`` if unset) — see
    :func:`evaluate_model_on_dataset`.
    """
    names = dataset_names if dataset_names is not None else select_eval_datasets(config)
    if not names:
        logger.warning("no datasets flagged IS_EVAL_DATASET -- nothing to evaluate")
        return {}
    qm = (query_mode or str(config.get("EVAL_QUERY_MODE", "first"))).lower()
    logger.info(f"evaluating on {len(names)} dataset(s) [query_mode={qm}]: {names}")

    results: Dict[str, Dict[str, float]] = {}
    for name in names:
        ds = build_eval_dataset(name, config, max_clips=max_clips)
        if ds is None or len(ds) == 0:
            continue
        logger.info(f"  {name}: {len(ds)} clips ...")
        m = evaluate_model_on_dataset(
            model, ds, device, batch_size=batch_size, num_workers=num_workers,
            amp=amp, amp_dtype=amp_dtype, max_steps=max_steps, query_frame=query_frame,
            query_mode=qm,
        )
        results[name] = m
        logger.info(
            f"  {name}: EPE={m['epe']:.2f}px δ_avg={m['delta_avg']:.3f} "
            f"AJ={m['average_jaccard']:.3f} OA={m['occlusion_accuracy']:.3f} "
            f"{m['ms_per_frame']:.2f} ms/frame ({m['n_clips']} clips)"
        )

    if results:
        results["MEAN"] = {k: _nanmean([r[k] for r in results.values()]) for k in METRIC_KEYS}
    return results


# --------------------------------------------------------------------------- #
# Reporting: CSV, console, W&B
# --------------------------------------------------------------------------- #
def write_csv(results: Dict[str, Dict[str, float]], path: Path) -> Path:
    """Write ``results`` to a CSV (rows = datasets then ``MEAN``, cols = metrics)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["dataset"] + [METRIC_HEADERS[k] for k in METRIC_KEYS] + ["n_clips", "n_frames"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for name, m in results.items():
            row = [name]
            for k in METRIC_KEYS:
                v = m.get(k, float("nan"))
                row.append(f"{v:.4f}" if v == v else "")
            row += [m.get("n_clips", ""), m.get("n_frames", "")]
            w.writerow(row)
    return path


def format_table(results: Dict[str, Dict[str, float]]) -> str:
    """A monospace console table of the results (datasets x metrics)."""
    cols = [METRIC_HEADERS[k] for k in METRIC_KEYS]
    name_w = max([len("dataset")] + [len(n) for n in results]) if results else len("dataset")
    col_w = max(16, *[len(c) for c in cols])
    head = f"{'dataset':<{name_w}}  " + "  ".join(f"{c:>{col_w}}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in results.items():
        cells = []
        for k in METRIC_KEYS:
            v = m.get(k, float("nan"))
            cells.append(f"{v:>{col_w}.4f}" if v == v else f"{'-':>{col_w}}")
        lines.append(f"{name:<{name_w}}  " + "  ".join(cells))
    return "\n".join(lines)


def log_wandb_table(
    results: Dict[str, Dict[str, float]],
    wandb_run: Any,
    *,
    key: str = "eval/metrics",
    epoch: Optional[int] = None,
) -> None:
    """Log a ``wandb.Table`` (rows = datasets, cols = metrics) plus per-dataset
    and mean scalars (``eval/<dataset>/<metric>``) for time-series tracking.

    No-op when ``wandb_run`` is None. Logged without an explicit step (W&B
    auto-increments), with an ``eval/epoch`` field to align with training.
    """
    if wandb_run is None or not results:
        return
    try:
        import wandb
        columns = ["dataset"] + [METRIC_HEADERS[k] for k in METRIC_KEYS]
        table = wandb.Table(columns=columns)
        row: Dict[str, Any] = {}
        for name, m in results.items():
            cells = [m.get(k, float("nan")) for k in METRIC_KEYS]
            table.add_data(name, *[round(c, 5) if c == c else None for c in cells])
            for k in METRIC_KEYS:
                v = m.get(k, float("nan"))
                if v == v:
                    row[f"eval/{name}/{k}"] = v
        row[key] = table
        if epoch is not None:
            row["eval/epoch"] = epoch
        wandb_run.log(row)
    except Exception as e:  # noqa: BLE001 -- logging must never crash a run
        logger.warning(f"W&B eval table log skipped ({e})")


# --------------------------------------------------------------------------- #
# Orchestrator (used by the engine + the standalone CLI)
# --------------------------------------------------------------------------- #
def evaluate_and_report(
    model: torch.nn.Module,
    config: Any,
    device: torch.device,
    run_dir: Any,
    *,
    wandb_run: Any = None,
    tag: str = "",
    epoch: Optional[int] = None,
    dataset_names: Optional[List[str]] = None,
    max_clips: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    amp: Optional[bool] = None,
    amp_dtype=None,
    max_steps: int = 0,
    query_frame: int = 0,
    query_mode: Optional[str] = None,
    csv_name: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate then emit all reports: CSV under ``run_dir``, a console table, and
    (if active) a W&B table + scalars. Returns the results dict.

    ``tag`` differentiates monitoring snapshots: the CSV is ``evaluation.csv`` by
    default, ``evaluation_<tag>.csv`` when a tag is given.
    """
    results = evaluate(
        model, config, device, dataset_names=dataset_names, max_clips=max_clips,
        batch_size=batch_size, num_workers=num_workers, amp=amp, amp_dtype=amp_dtype,
        max_steps=max_steps, query_frame=query_frame, query_mode=query_mode,
    )
    if not results:
        logger.warning("evaluation produced no results (no datasets available)")
        return results

    fname = csv_name or (f"evaluation_{tag}.csv" if tag else "evaluation.csv")
    csv_path = Path(run_dir) / fname
    write_csv(results, csv_path)
    logger.info(f"evaluation results ({len(results) - 1} datasets + MEAN):\n{format_table(results)}")
    logger.info(f"evaluation CSV -> {csv_path}")
    log_wandb_table(results, wandb_run, epoch=epoch)
    return results


# --------------------------------------------------------------------------- #
# Standalone: load a checkpoint and evaluate it
# --------------------------------------------------------------------------- #
def load_model_from_checkpoint(
    ckpt_path: Any,
    device: Optional[torch.device] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    unknown_args: Optional[List[str]] = None,
    verbose: bool = True,
):
    """Rebuild the model from a checkpoint's embedded config and load its weights.

    Returns ``(model, config, checkpoint_dict)``. ``config_overrides`` (flat
    ``{KEY: value}``, may use dotted keys) are applied on top of the saved config
    before the model is built -- e.g. to point at different eval datasets or swap
    the encoder to ``cnn`` for a CPU run. ``unknown_args`` are leftover CLI
    ``--KEY=value`` tokens, type-coerced against the saved config's values.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(ckpt_path)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = dict(ck.get("config") or {})
    if config_overrides:
        cfg_dict.update(config_overrides)
    config = load_and_process_config(config=cfg_dict, unknown_args=unknown_args)

    model = create_model_from_config(config, device, verbose=verbose)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        logger.warning(f"loaded {ckpt_path.name} non-strictly "
                       f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    logger.info(f"loaded checkpoint {ckpt_path} (epoch {ck.get('epoch', '?')})")
    return model, config, ck


def evaluate_checkpoint(
    ckpt_path: Any,
    *,
    device: Optional[torch.device] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    unknown_args: Optional[List[str]] = None,
    out_dir: Optional[Any] = None,
    use_wandb: bool = False,
    tag: str = "",
    dataset_names: Optional[List[str]] = None,
    max_clips: Optional[int] = None,
    batch_size: int = 1,
    num_workers: int = 0,
    max_steps: int = 0,
    query_mode: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """Load a trained checkpoint and evaluate it end to end.

    The CSV is written next to the checkpoint's run dir (``<run>/evaluation.csv``)
    unless ``out_dir`` is given. With ``use_wandb`` a fresh W&B run is opened for
    the table. Returns the results dict.
    """
    model, config, _ = load_model_from_checkpoint(
        ckpt_path, device, config_overrides=config_overrides, unknown_args=unknown_args)
    device = next(model.parameters()).device
    # checkpoints live at <run_dir>/stage{idx}_{name}/{best,last}.pt -> run dir is 2 up
    run_dir = Path(out_dir) if out_dir is not None else Path(ckpt_path).resolve().parent.parent

    wandb_run, owns = None, False
    if use_wandb:
        from utilities.engine import finish_wandb, init_wandb
        wandb_run, owns = init_wandb(config, run_dir)
    try:
        results = evaluate_and_report(
            model, config, device, run_dir, wandb_run=wandb_run, tag=tag,
            dataset_names=dataset_names, max_clips=max_clips, batch_size=batch_size,
            num_workers=num_workers, max_steps=max_steps, query_mode=query_mode,
        )
    finally:
        if owns and wandb_run is not None:
            from utilities.engine import finish_wandb
            finish_wandb(wandb_run, owns)
    return results
