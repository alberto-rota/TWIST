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
per-dataset/mean scalars for time-series tracking. Each dataset's scalars are
logged to W&B (and the CSV rewritten) **as soon as that dataset finishes**, not
just once at the very end. Unless ``EVAL_SKIP_COMPLETED`` is set False, a
dataset already recorded as finished for this run dir + tag (``eval_state*.json``)
is skipped and its cached metrics are reused rather than re-run -- so a
benchmark job that gets pre-empted or crashes partway through resumes without
redoing the datasets it already finished.

Alongside, unless ``EVAL_RECOVERY`` is off, the **Post-Occlusion Recovery** (POR)
metric is reported in its *own* artifacts (so the table above stays
CoTracker-comparable): ``recovery.csv`` (headline POR/THO scalars),
``recovery_by_length.csv`` (the recovery-vs-occlusion-length curve), and
``recovery_drift.csv`` (Case-B through-occlusion drift). Datasets that store
valid GT on occluded frames are flagged ``HAS_OCCLUDED_GT: True`` in the registry
to enable the through-occlusion (Case B) measures. See :mod:`models.metrics`.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from dataset.wrappers import (
    ALL_DATASETS_KEY,
    DATASET_DEFAULTS,
    OVERRIDE_ALL_DATASETS_KEY,
    reader_class_for,
)
from models import (
    finalize_recovery,
    merge_recovery_stats,
    recovery_metrics,
    tracking_metrics,
)
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
# Per-dataset config flag (default False): the dataset stores valid GT coords even
# on occluded frames (synthetic full GT, not the (0,0) placeholder). Enables the
# Post-Occlusion-Recovery through-occlusion + drift measures (Case B) for it.
HAS_OCCLUDED_GT_KEY = "HAS_OCCLUDED_GT"
# Per-dataset config key (default None -> TAP-Vid {1,2,4,8,16}): override the
# delta/Jaccard pixel thresholds for this dataset. STIR sets [4,8,16,32,64] to
# match its official 2D accuracy metric.
EVAL_THRESHOLDS_KEY = "EVAL_THRESHOLDS"
# Per-dataset config flag (default False): the dataset's visibility GT is *sparse* —
# present only on annotated frames, with every other frame marked occluded merely
# because it is unlabelled (STIR: GT only at first/last frame). When set, the metric
# scores only the visible (annotated) frames (tracking_metrics(visible_only=True)), so
# AJ/OA are not collapsed to ~0 by treating unlabelled interior frames as occluded and
# counting the model's (probably-correct) visible predictions there as false positives.
# EPE/δ are unaffected (already visible-only).
EVAL_VISIBLE_ONLY_KEY = "EVAL_VISIBLE_ONLY"
# Top-level config flag (default True): skip datasets whose metrics were already
# finished (and reported) in a previous call over this run dir + tag -- see
# _load_eval_state / _save_eval_state. Lets a benchmark job that got pre-empted
# or crashed resume without re-running (and re-timing) datasets it already
# finished. Set False to force a clean re-evaluation of everything.
EVAL_SKIP_COMPLETED_KEY = "EVAL_SKIP_COMPLETED"
# Per-dataset config key (default []): metric keys this dataset must NOT
# contribute to the MEAN row. STIR excludes average_jaccard/occlusion_accuracy:
# its GT visibility is only defined on the annotated endpoint frames, so AJ/OA
# degenerate (~0.005 for EVERY method — a scoring artifact, not signal) and were
# silently deflating everyone's mean. The excluded values still appear on the
# dataset's own row.
EVAL_EXCLUDE_FROM_MEAN_KEY = "EVAL_EXCLUDE_FROM_MEAN"

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

# Post-Occlusion Recovery (POR) — TWIST-specific occlusion-recovery metric
# (models.recovery_metrics), reported in its *own* artifacts so the TAP table
# above stays CoTracker-comparable. Quality keys are length-weighted means; count
# keys sum across datasets. ``tho_*`` (through-occlusion) appear only for
# full-GT (Case B) datasets, NaN elsewhere.
RECOVERY_QUALITY_KEYS = ["por_epe_snap", "por_epe_w8", "por_delta_snap", "por_delta_w8",
                         "tho_epe", "tho_delta"]
RECOVERY_COUNT_KEYS = ["n_recovery_events", "n_through_occlusion_events"]
RECOVERY_HEADERS = {
    "por_epe_snap": "POR-EPE snap (px)",
    "por_epe_w8": "POR-EPE w8 (px)",
    "por_delta_snap": "POR-delta snap",
    "por_delta_w8": "POR-delta w8",
    "tho_epe": "THO-EPE (px)",
    "tho_delta": "THO-delta",
    "n_recovery_events": "n recovery events",
    "n_through_occlusion_events": "n occlusion events",
}
# Nested per-dataset structured recovery output (by_length / tho_by_length / drift_epe).
RECOVERY_DETAIL_KEY = "_recovery_detail"


def _nanmean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and x == x]  # drop None / NaN
    return sum(xs) / len(xs) if xs else float("nan")


# --------------------------------------------------------------------------- #
# Per-run eval state — {dataset_name: metrics} already finished (and reported)
# for this run dir + tag, so a restarted evaluation can skip re-computing (and
# re-timing) them. Mirrors utilities.runs' run_state.json (same atomic-write
# pattern), scoped to eval instead of training stages.
# --------------------------------------------------------------------------- #
def _eval_state_path(run_dir: Path, tag: str) -> Path:
    return Path(run_dir) / f"eval_state{f'_{tag}' if tag else ''}.json"


def _load_eval_state(run_dir: Path, tag: str) -> Dict[str, Dict[str, Any]]:
    p = _eval_state_path(run_dir, tag)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001 -- corrupt/partial state -> start clean
            logger.warning(f"could not read {p}, ignoring cached eval state")
    return {}


def _save_eval_state(run_dir: Path, tag: str, state: Dict[str, Dict[str, Any]]) -> None:
    """Atomically write the eval state (temp file + replace)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(run_dir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, _eval_state_path(run_dir, tag))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _mark_dataset_done(run_dir: Path, tag: str, name: str, metrics: Dict[str, Any]) -> None:
    """Record ``name``'s finished metrics into this run+tag's eval state."""
    state = _load_eval_state(run_dir, tag)
    state[name] = metrics
    _save_eval_state(run_dir, tag, state)


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
    # Benchmark eval scores WHOLE sequences (TAP-Vid queried-first runs over the
    # full video), so the training-time temporal-sampling knobs that
    # OVERRIDE_ALL_DATASETS pins for the train mix (CLIP_LEN=24, strides,
    # per-video caps) must NOT chop the eval clips -- doing so changes which
    # points are queried and, for STIR, drops the end-frame GT out of the query
    # window entirely. Force the whole-sequence reading every eval dataset
    # expects (each registry entry already declares CLIP_LEN: None).
    kw["clip_len"] = None
    kw["clip_stride"] = None
    kw["frame_stride"] = 1
    kw["max_clips_per_video"] = None
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
    for ``first`` mode). Points never visible (or padded out by ``point_mask``) are
    left out of the eval mask entirely. This is the protocol that makes our numbers
    comparable to CoTracker's — and it sidesteps the ``(0,0)`` occluded-coordinate
    placeholder ever being used as a *query* (the real harm of those placeholders;
    see CLAUDE.md).

    How many forwards this costs per clip depends on the model:

    * The TWIST model takes a **single** query frame per forward, so points that share
      a first-visible frame are grouped and run together — **one forward per distinct
      first-visible frame**.
    * A model that sets ``supports_query_times = True`` (e.g. the CoTracker / offline
      TAPIR adapters, whose predictors natively accept queries at arbitrary per-point
      ``t``) instead gets **one forward per clip**, with every point queried at its own
      first-visible frame. This is not just faster (multi-group clips drop from ~8
      forwards to 1 on Kinetics / RoboTAP) but more faithful — it tracks all points
      jointly, exactly as CoTracker's own eval does, rather than in smaller attention
      subsets.

    Both paths scatter into the same per-point arrays and mark each point evaluated
    strictly after *its own* query frame, so they are metric-equivalent.

    Frozen-encoder features don't depend on the query, so on a model exposing
    ``.encode()`` (the TWIST world model) they are computed **once per clip** here
    and reused across every group/forward -- otherwise the single-query-per-forward
    path would re-run the (expensive) backbone once per distinct first-visible
    frame. This matters most on datasets like RoboTAP, which combine long clips
    with many distinct query groups (~9 on average, up to 28), so the naive
    per-group re-encode multiplied total backbone work ~8x.

    Returns ``(coords, vis_logits, eval_mask)`` shaped ``(B,T,N,2)/(B,T,N)/(B,T,N)``.
    """
    B, T = frames.shape[:2]
    N = gt_vis.shape[2]
    device = frames.device
    gt_vis_b = gt_vis.bool()
    pm = point_mask.bool() if point_mask is not None else torch.ones(B, N, dtype=torch.bool, device=device)
    multiquery = bool(getattr(model, "supports_query_times", False))
    t_ar = torch.arange(T, device=device).unsqueeze(1)          # (T,1), for the eval mask

    coords = gt_tracks.clone().float()                          # overwritten per group
    vis_logits = torch.full((B, T, N), -10.0, device=device, dtype=torch.float32)
    eval_mask = torch.zeros((B, T, N), dtype=torch.bool, device=device)

    can_cache_feats = hasattr(model, "encode")
    with autocast_ctx():
        feats = model.encode(frames) if can_cache_feats else None  # (B,T,C,Hf,Wf) once for the whole batch

    def _run_group(b, idx, first_vis):
        """One forward for points ``idx`` (each queried at its own first-visible frame),
        scattering coords / vis_logits back and marking each point evaluated strictly
        after its own query frame. When all of ``idx`` share a frame this is a group; in
        the multiquery path ``idx`` is every usable point at once."""
        fv = first_vis[idx]                                     # (n,) per-point query frame
        q_xy = gt_tracks[b, fv, idx].float()                    # (n,2) query coord at that frame
        queries = torch.cat([fv.float().unsqueeze(-1), q_xy], dim=-1).unsqueeze(0)   # (1,n,3)
        with autocast_ctx():
            if can_cache_feats:
                out = model(frames[b:b + 1], queries, feats=feats[b:b + 1])
            else:
                out = model(frames[b:b + 1], queries)
        coords[b, :, idx] = out["coords"][0].float()
        vis_logits[b, :, idx] = out["vis_logits"][0].float()
        eval_mask[b, :, idx] = t_ar > fv.unsqueeze(0)           # (T,n): strictly after each query frame

    for b in range(B):
        vis = gt_vis_b[b]                                       # (T,N)
        usable = vis.any(0) & pm[b]                             # (N,) visible somewhere & real
        if not usable.any():
            continue
        first_vis = vis.float().argmax(0)                      # (N,) first visible frame (0 if none)
        if multiquery:                                         # one forward, all points at their own query frame
            _run_group(b, usable.nonzero(as_tuple=True)[0], first_vis)
        else:                                                  # one forward per distinct query frame (TWIST)
            for f in torch.unique(first_vis[usable]).tolist():
                idx = ((first_vis == int(f)) & usable).nonzero(as_tuple=True)[0]
                _run_group(b, idx, first_vis)
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
    compute_recovery: bool = True,
    has_occluded_gt: bool = False,
    recovery_window: int = 8,
    eval_thresholds: Optional[Sequence[float]] = None,
    visible_only: bool = False,
    name: Optional[str] = None,
    log_every: int = 50,
    timing_batches: int = 10,
) -> Dict[str, float]:
    """Run ``model`` over ``dataset`` and return the reported metric means.

    Quality metrics are averaged per batch (NaN-safe), matching the engine's
    ``validate``. ``ms_per_frame`` = total forward wall-clock / real frames
    processed; on CUDA a warm-up forward runs first (outside timing) so cudnn
    autotune isn't charged to it. Returns ``delta_avg``, ``average_jaccard``,
    ``occlusion_accuracy``, ``ms_per_frame`` plus ``n_clips`` / ``n_frames``.

    With ``compute_recovery`` (default on) it also pools Post-Occlusion-Recovery
    stats over the clips and adds the finalized POR scalars (``por_epe_snap`` /
    ``por_epe_w8`` / ``por_delta_*``, plus ``tho_*`` when ``has_occluded_gt``) and
    a nested ``_recovery_detail`` (per-length curve + drift) to the result.

    ``query_mode``: ``"first"`` (TAP-Vid "queried first" — each point queried at its
    first visible frame, scored only after it; the comparable-to-CoTracker default)
    or ``"frame0"`` (legacy — all points queried at frame 0, every frame scored).
    Timing (``ms_per_frame``) uses the single all-points-at-frame-0 forward, so it
    reflects realistic one-pass inference cost regardless of ``query_mode``. In
    ``"frame0"`` that forward also produces the scored predictions (so it runs every
    batch); in ``"first"`` the scoring is done separately, so the timing forward runs
    only on the first ``timing_batches`` batches — an unbiased per-frame sample that
    avoids ~doubling the work — and ``timing_batches=0`` skips it entirely.

    ``name`` (optional) tags the progress lines; ``log_every`` controls how often the
    ``processed X/Y clips`` line is emitted (every N clips; ``0`` disables it). Each
    progress line includes the latest batch ``frames`` / ``queries`` tensor shapes.
    """
    model.eval()
    if query_mode not in QUERY_MODES:
        raise ValueError(f"query_mode must be one of {QUERY_MODES}, got {query_mode!r}")
    use_amp, amp_dtype = _amp_settings(device, amp, amp_dtype)

    def autocast_ctx():
        return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp)

    # Per-dataset delta/Jaccard pixel thresholds (e.g. STIR's official
    # [4,8,16,32,64]); None -> tracking_metrics/recovery_metrics defaults (TAP-Vid
    # {1,2,4,8,16}). Threaded into every metric call so the table stays consistent.
    thr_kw = {} if eval_thresholds is None else {"thresholds": tuple(eval_thresholds)}

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        drop_last=False, collate_fn=_collate_for(dataset),
    )

    agg: Dict[str, float] = {k: 0.0 for k in _QUALITY_KEYS}
    cnt: Dict[str, int] = {k: 0 for k in _QUALITY_KEYS}
    rec_stats: Dict[str, float] = {}                     # POR sufficient stats, pooled over clips
    compute_s = 0.0
    timed_frames = 0
    n_clips = 0
    n_frames = 0
    tag = f"[{name}] " if name else ""
    try:
        total_clips = len(dataset)                       # for the "X/Y clips" progress line
    except TypeError:
        total_clips = None
    last_logged = 0
    last_shapes = ""

    def _shape_str(frames_t, queries_t):
        return f"frames={tuple(frames_t.shape)} queries={tuple(queries_t.shape)}"

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

        last_shapes = _shape_str(frames, queries)
        if step == 0:
            logger.info(f"    {tag}batch tensor shapes: {last_shapes}")

        # real (unpadded) frames in this batch -> the per-frame timing denominator
        nf = int(time_mask.sum().item()) if time_mask is not None else int(frames.shape[0] * frames.shape[1])

        # The dedicated single-pass (all points at frame 0) forward that measures
        # ms/frame. In "frame0" mode this *is* the scored forward, so it always runs.
        # In "first" mode the scoring happens in _first_visible_eval and this ``out``
        # is used only for timing — so we run it on just the first ``timing_batches``
        # batches (a representative per-frame sample) rather than every batch, which
        # would otherwise ~double the work on datasets where most points share a query
        # frame. ms_per_frame is normalised per frame, so a sample is unbiased.
        out = None
        if query_mode == "frame0" or step < timing_batches:
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
                                 eval_mask=e_mask, query_frame=query_frame,
                                 visible_only=visible_only, **thr_kw)
            rec_coords, rec_vislog = e_coords, e_vislog
        else:
            m = tracking_metrics(out["coords"], gt_tracks, out["vis_logits"], gt_vis,
                                 time_mask, point_mask, query_frame=query_frame,
                                 visible_only=visible_only, **thr_kw)
            rec_coords, rec_vislog = out["coords"], out["vis_logits"]
        for k in _QUALITY_KEYS:
            v = m.get(k, float("nan"))
            if v == v:                                  # NaN-safe
                agg[k] += v
                cnt[k] += 1
        # Post-occlusion recovery: pool sufficient stats over clips (events are
        # per-point, so they must accumulate globally, not be averaged per clip).
        if compute_recovery:
            rec_stats = merge_recovery_stats(rec_stats, recovery_metrics(
                rec_coords, gt_tracks, rec_vislog, gt_vis, point_mask=point_mask,
                has_occluded_gt=has_occluded_gt, window=recovery_window, **thr_kw))
        n_clips += int(frames.shape[0])
        n_frames += nf

        if log_every and n_clips - last_logged >= log_every:
            total_str = f"/{total_clips}" if total_clips is not None else ""
            logger.info(
                f"    {tag}processed {n_clips}{total_str} clips "
                f"({last_shapes}) ..."
            )
            last_logged = n_clips

    if log_every and n_clips != last_logged:             # final tally (avoid a dup line)
        total_str = f"/{total_clips}" if total_clips is not None else ""
        logger.info(
            f"    {tag}processed {n_clips}{total_str} clips "
            f"({last_shapes}) (done)"
        )

    result = {k: (agg[k] / cnt[k] if cnt[k] else float("nan")) for k in _QUALITY_KEYS}
    result["ms_per_frame"] = (1e3 * compute_s / timed_frames) if timed_frames else float("nan")
    result["n_clips"] = n_clips
    result["n_frames"] = n_frames
    if compute_recovery:
        rec = finalize_recovery(rec_stats)
        for k in RECOVERY_QUALITY_KEYS + RECOVERY_COUNT_KEYS:
            if k in rec:
                result[k] = rec[k]
        result[RECOVERY_DETAIL_KEY] = {
            kk: rec[kk] for kk in ("by_length", "tho_by_length", "drift_epe") if kk in rec
        }
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
    cached_results: Optional[Dict[str, Dict[str, Any]]] = None,
    on_dataset_done: Optional[Callable[[str, Dict[str, Any], bool], None]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate ``model`` on every selected eval dataset.

    Returns ``{dataset_name: metrics, ..., "MEAN": metrics}`` where ``MEAN`` is
    the across-dataset mean of each metric (NaN-safe). Datasets whose data is not
    present on disk are skipped with a warning. ``query_mode`` defaults to the
    config's ``EVAL_QUERY_MODE`` (``"first"`` if unset) — see
    :func:`evaluate_model_on_dataset`.

    ``cached_results`` (optional ``{dataset_name: metrics}``) short-circuits any
    matching dataset -- its cached metrics are used verbatim instead of
    re-running the model, so a resumed evaluation doesn't redo already-finished
    work. ``on_dataset_done(name, metrics, from_cache)`` (optional), when given,
    fires right after each dataset's metrics are available (fresh or cached) --
    this is how :func:`evaluate_and_report` logs each dataset to W&B / persists
    the eval state as soon as it finishes, instead of waiting for every dataset.
    """
    names = dataset_names if dataset_names is not None else select_eval_datasets(config)
    if not names:
        logger.warning("no datasets flagged IS_EVAL_DATASET -- nothing to evaluate")
        return {}
    qm = (query_mode or str(config.get("EVAL_QUERY_MODE", "first"))).lower()
    compute_recovery = bool(config.get("EVAL_RECOVERY", True))
    recovery_window = int(config.get("EVAL_RECOVERY_WINDOW", 8))
    datasets_cfg = _as_dict(config.get("DATASETS"))
    cached_results = cached_results or {}
    logger.info(f"evaluating on {len(names)} dataset(s) [query_mode={qm}, "
                f"recovery={'on' if compute_recovery else 'off'}]: {names}")
    if cached_results:
        already = [n for n in names if n in cached_results]
        if already:
            logger.info(f"  {len(already)} already evaluated this run, skipping: {already}")

    results: Dict[str, Dict[str, float]] = {}
    for name in names:
        if name in cached_results:
            m = cached_results[name]
            results[name] = m
            logger.info(
                f"  {name}: skipped (already evaluated) -- "
                f"EPE={m.get('epe', float('nan')):.2f}px δ_avg={m.get('delta_avg', float('nan')):.3f}"
            )
            if on_dataset_done is not None:
                on_dataset_done(name, m, True)
            continue
        ds = build_eval_dataset(name, config, max_clips=max_clips)
        if ds is None or len(ds) == 0:
            continue
        merged = _merged_dataset_cfg(name, datasets_cfg)
        has_occ = bool(merged.get(HAS_OCCLUDED_GT_KEY, False))
        eval_thr = merged.get(EVAL_THRESHOLDS_KEY)        # e.g. STIR [4,8,16,32,64]; None -> TAP default
        vis_only = bool(merged.get(EVAL_VISIBLE_ONLY_KEY, False))  # sparse-GT (STIR): score annotated frames only
        logger.info(f"  {name}: {len(ds)} clips ..."
                    + (f" [thresholds={list(eval_thr)}]" if eval_thr else "")
                    + (" [visible-only]" if vis_only else ""))
        m = evaluate_model_on_dataset(
            model, ds, device, batch_size=batch_size, num_workers=num_workers,
            amp=amp, amp_dtype=amp_dtype, max_steps=max_steps, query_frame=query_frame,
            query_mode=qm, compute_recovery=compute_recovery, has_occluded_gt=has_occ,
            recovery_window=recovery_window, eval_thresholds=eval_thr,
            visible_only=vis_only, name=name,
            timing_batches=int(config.get("EVAL_TIMING_BATCHES", 10)),
        )
        results[name] = m
        rec_suffix = ""
        if compute_recovery and m.get("n_recovery_events"):
            rec_suffix = (f" | POR-EPE w8={m.get('por_epe_w8', float('nan')):.1f}px "
                          f"snap={m.get('por_epe_snap', float('nan')):.1f}px "
                          f"({int(m['n_recovery_events'])} ev"
                          + (f", THO={m['tho_epe']:.1f}px" if m.get("tho_epe") == m.get("tho_epe")
                             and "tho_epe" in m else "") + ")")
        logger.info(
            f"  {name}: EPE={m['epe']:.2f}px δ_avg={m['delta_avg']:.3f} "
            f"AJ={m['average_jaccard']:.3f} OA={m['occlusion_accuracy']:.3f} "
            f"{m['ms_per_frame']:.2f} ms/frame ({m['n_clips']} clips){rec_suffix}"
        )
        if on_dataset_done is not None:
            on_dataset_done(name, m, False)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if results:
        results["MEAN"] = compute_mean_row(results, datasets_cfg, compute_recovery)
    return results


def compute_mean_row(results: Dict[str, Dict[str, float]], datasets_cfg: dict,
                     compute_recovery: bool = True) -> Dict[str, float]:
    """The cross-dataset ``MEAN`` row, honouring per-dataset exclusions.

    A dataset listing a metric key under ``EVAL_EXCLUDE_FROM_MEAN`` (registry or
    config) keeps the value on its own row but does not contribute it to the mean
    — e.g. STIR's degenerate AJ/OA (visibility GT only exists on its endpoint
    frames, so those two are a scoring artifact for every method). Quality keys
    are nan-means; recovery counts sum.
    """
    excl = {n: set(_merged_dataset_cfg(n, datasets_cfg).get(EVAL_EXCLUDE_FROM_MEAN_KEY) or ())
            for n in results}

    def _mean_over(k: str) -> float:
        return _nanmean([r.get(k, float("nan")) for n, r in results.items()
                         if k not in excl[n]])

    mean = {k: _mean_over(k) for k in METRIC_KEYS}
    if compute_recovery:
        for k in RECOVERY_QUALITY_KEYS:
            mean[k] = _mean_over(k)
        for k in RECOVERY_COUNT_KEYS:                    # counts sum, not mean
            mean[k] = sum(int(r.get(k, 0) or 0) for r in results.values())
    return mean


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


def log_dataset_wandb(
    name: str,
    metrics: Dict[str, Any],
    wandb_run: Any,
    *,
    epoch: Optional[int] = None,
) -> None:
    """Log one dataset's scalars (``eval/<name>/<metric>``) right when it finishes,
    instead of waiting for the whole evaluation to log a single end-of-run table.

    Covers the same keys ``log_wandb_table`` / ``log_recovery_wandb`` put in their
    per-dataset rows (TAP quality metrics + POR quality metrics, not the count
    keys or the ``_recovery_detail`` curve, which stay table-only). No-op when
    ``wandb_run`` is None.
    """
    if wandb_run is None:
        return
    try:
        row: Dict[str, Any] = {}
        for k in METRIC_KEYS + RECOVERY_QUALITY_KEYS:
            v = metrics.get(k, float("nan"))
            if v == v:
                row[f"eval/{name}/{k}"] = v
        if not row:
            return
        if epoch is not None:
            row["eval/epoch"] = epoch
        wandb_run.log(row)
    except Exception as e:  # noqa: BLE001 -- logging must never crash a run
        logger.warning(f"W&B per-dataset log skipped for {name} ({e})")


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
# Reporting: Post-Occlusion Recovery (its own artifacts, separate from the TAP table)
# --------------------------------------------------------------------------- #
def _has_recovery(results: Dict[str, Dict[str, float]]) -> bool:
    return any(RECOVERY_DETAIL_KEY in m or "n_recovery_events" in m for m in results.values())


def _fmt(m: Dict[str, Any], k: str) -> str:
    v = m.get(k, float("nan"))
    if v != v:                                          # NaN
        return ""
    return str(int(v)) if k in RECOVERY_COUNT_KEYS else f"{v:.4f}"


def write_recovery_csv(results: Dict[str, Dict[str, float]], path: Path) -> Path:
    """Recovery headline scalars (rows = datasets then ``MEAN``, cols = POR/THO)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = RECOVERY_QUALITY_KEYS + RECOVERY_COUNT_KEYS
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset"] + [RECOVERY_HEADERS[k] for k in keys])
        for name, m in results.items():
            w.writerow([name] + [_fmt(m, k) for k in keys])
    return path


def write_recovery_curve_csv(results: Dict[str, Dict[str, float]], path: Path) -> Path:
    """The recovery-vs-occlusion-length curve (one row per dataset x length bin)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "occlusion_length", "por_epe_snap", "por_epe_w8",
                    "por_delta_snap", "por_delta_w8", "por_n", "tho_epe", "tho_delta", "tho_n"])
        for name, m in results.items():
            detail = m.get(RECOVERY_DETAIL_KEY) or {}
            por = detail.get("by_length") or {}
            tho = detail.get("tho_by_length") or {}
            bins = list(por.keys()) + [b for b in tho if b not in por]
            for b in bins:
                p, t = por.get(b, {}), tho.get(b, {})
                cell = lambda d, k: (f"{d[k]:.4f}" if k in d else "")
                w.writerow([name, b, cell(p, "epe_snap"), cell(p, "epe_w8"),
                            cell(p, "delta_snap"), cell(p, "delta_w8"), p.get("n", ""),
                            cell(t, "epe"), cell(t, "delta"), t.get("n", "")])
    return path


def write_recovery_drift_csv(results: Dict[str, Dict[str, float]], path: Path) -> bool:
    """Case-B through-occlusion drift (EPE vs frames-since-onset). Returns whether
    anything was written (only full-GT datasets have a drift curve)."""
    rows = []
    for name, m in results.items():
        drift = (m.get(RECOVERY_DETAIL_KEY) or {}).get("drift_epe") or {}
        # keys are ints fresh from finalize_recovery, but strings when round-tripped
        # through the JSON eval-state cache -- sort numerically either way.
        rows += [[name, int(k), f"{drift[k]:.4f}"] for k in sorted(drift, key=lambda k: int(k))]
    if not rows:
        return False
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "frames_since_onset", "epe_px"])
        w.writerows(rows)
    return True


def format_recovery_table(results: Dict[str, Dict[str, float]]) -> str:
    """A monospace console table of the recovery headline scalars."""
    keys = RECOVERY_QUALITY_KEYS + RECOVERY_COUNT_KEYS
    cols = [RECOVERY_HEADERS[k] for k in keys]
    name_w = max([len("dataset")] + [len(n) for n in results]) if results else len("dataset")
    col_w = max(16, *[len(c) for c in cols])
    head = f"{'dataset':<{name_w}}  " + "  ".join(f"{c:>{col_w}}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in results.items():
        cells = [(f"{s:>{col_w}}" if (s := _fmt(m, k)) else f"{'-':>{col_w}}") for k in keys]
        lines.append(f"{name:<{name_w}}  " + "  ".join(cells))
    return "\n".join(lines)


def log_recovery_wandb(
    results: Dict[str, Dict[str, float]],
    wandb_run: Any,
    *,
    key: str = "eval/recovery",
    epoch: Optional[int] = None,
) -> None:
    """Log a recovery ``wandb.Table`` plus per-dataset/mean scalars
    (``eval/<dataset>/por_epe_w8`` ...). No-op when ``wandb_run`` is None."""
    if wandb_run is None or not results:
        return
    try:
        import wandb
        keys = RECOVERY_QUALITY_KEYS + RECOVERY_COUNT_KEYS
        table = wandb.Table(columns=["dataset"] + [RECOVERY_HEADERS[k] for k in keys])
        row: Dict[str, Any] = {}
        for name, m in results.items():
            cells = []
            for k in keys:
                v = m.get(k, float("nan"))
                cells.append((int(v) if k in RECOVERY_COUNT_KEYS else round(v, 5)) if v == v else None)
            table.add_data(name, *cells)
            for k in RECOVERY_QUALITY_KEYS:
                v = m.get(k, float("nan"))
                if v == v:
                    row[f"eval/{name}/{k}"] = v
        row[key] = table
        if epoch is not None:
            row["eval/epoch"] = epoch
        wandb_run.log(row)
    except Exception as e:  # noqa: BLE001 -- logging must never crash a run
        logger.warning(f"W&B recovery table log skipped ({e})")


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
    default, ``evaluation_<tag>.csv`` when a tag is given; the per-run eval state
    (see below) is likewise scoped to ``tag``.

    Each dataset is logged to W&B (``eval/<dataset>/<metric>`` scalars) and
    written into the CSV **as soon as it finishes**, rather than all at once
    after the last dataset -- so progress is visible on W&B and on disk while a
    long evaluation is still running. Unless ``EVAL_SKIP_COMPLETED`` is set False
    in ``config``, a dataset already recorded as finished for this ``run_dir`` +
    ``tag`` (``eval_state[_<tag>].json``) is skipped and its cached metrics are
    reused (and re-logged) instead of re-running the model -- so a benchmark job
    that got pre-empted or crashed partway through resumes without redoing (and
    re-timing) the datasets it already finished.
    """
    run_dir = Path(run_dir)
    skip_completed = bool(config.get(EVAL_SKIP_COMPLETED_KEY, True))
    cached = _load_eval_state(run_dir, tag) if skip_completed else {}

    fname = csv_name or (f"evaluation_{tag}.csv" if tag else "evaluation.csv")
    csv_path = run_dir / fname
    partial: Dict[str, Dict[str, Any]] = {}

    def _on_dataset_done(name: str, metrics: Dict[str, Any], from_cache: bool) -> None:
        partial[name] = metrics
        log_dataset_wandb(name, metrics, wandb_run, epoch=epoch)
        if not from_cache:
            _mark_dataset_done(run_dir, tag, name, metrics)
        write_csv(partial, csv_path)          # keep the on-disk CSV current as datasets finish

    results = evaluate(
        model, config, device, dataset_names=dataset_names, max_clips=max_clips,
        batch_size=batch_size, num_workers=num_workers, amp=amp, amp_dtype=amp_dtype,
        max_steps=max_steps, query_frame=query_frame, query_mode=query_mode,
        cached_results=cached, on_dataset_done=_on_dataset_done,
    )
    if not results:
        logger.warning("evaluation produced no results (no datasets available)")
        return results

    write_csv(results, csv_path)              # final rewrite: adds the MEAN row
    logger.info(f"evaluation results ({len(results) - 1} datasets + MEAN):\n{format_table(results)}")
    logger.info(f"evaluation CSV -> {csv_path}")
    log_wandb_table(results, wandb_run, epoch=epoch)

    # Post-occlusion recovery: its own CSVs + console table + W&B table, so the
    # TAP table above stays exactly the CoTracker-comparable headline.
    if _has_recovery(results):
        sfx = f"_{tag}" if tag else ""
        rec_csv = Path(run_dir) / f"recovery{sfx}.csv"
        curve_csv = Path(run_dir) / f"recovery_by_length{sfx}.csv"
        drift_csv = Path(run_dir) / f"recovery_drift{sfx}.csv"
        write_recovery_csv(results, rec_csv)
        write_recovery_curve_csv(results, curve_csv)
        wrote_drift = write_recovery_drift_csv(results, drift_csv)
        logger.info(f"post-occlusion recovery:\n{format_recovery_table(results)}")
        logger.info(f"recovery CSV -> {rec_csv} (by-length -> {curve_csv}"
                    + (f", drift -> {drift_csv}" if wrote_drift else "") + ")")
        log_recovery_wandb(results, wandb_run, epoch=epoch)
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
    unless ``out_dir`` is given. With ``use_wandb``, the checkpoint's saved
    ``wandb_run_id`` (see ``Engine._ckpt``) is used to **resume the original
    training run** -- so eval metrics land on the same run/chart instead of a
    same-named duplicate -- falling back to a fresh run if that id is missing or
    can't be resumed (older checkpoint, deleted run, offline). Returns the
    results dict.
    """
    model, config, ck = load_model_from_checkpoint(
        ckpt_path, device, config_overrides=config_overrides, unknown_args=unknown_args)
    device = next(model.parameters()).device
    # checkpoints live at <run_dir>/stage{idx}_{name}/{best,last}.pt -> run dir is 2 up
    run_dir = Path(out_dir) if out_dir is not None else Path(ckpt_path).resolve().parent.parent

    wandb_run, owns = None, False
    if use_wandb:
        from utilities.engine import finish_wandb, init_wandb
        wandb_run, owns = init_wandb(config, run_dir, run_id=ck.get("wandb_run_id"))
    try:
        results = evaluate_and_report(
            model, config, device, run_dir, wandb_run=wandb_run, tag=tag,
            dataset_names=dataset_names, max_clips=max_clips, batch_size=batch_size,
            num_workers=num_workers, max_steps=max_steps, query_mode=query_mode,
            epoch=ck.get("epoch"),
        )
    finally:
        if owns and wandb_run is not None:
            from utilities.engine import finish_wandb
            finish_wandb(wandb_run, owns)
    return results
