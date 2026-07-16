"""Run directories and cross-stage resume bookkeeping.

A run is identified by ``EXPERIMENT_NAME`` and lives at ``$RESULTS_DIR/<name>/``.
``run_state.json`` records which schedule **stages** have finished, so a run that
completed pretraining can be resumed and continued at the next stage:

    python train.py config/schedule.yaml --resume-run <name>   # -> first unfinished stage

The training engine (next step) calls :func:`mark_stage_complete` when a stage's
training actually finishes and writes its checkpoint here; the resume logic then
skips finished stages. Until then the helpers are model-independent and fully
usable on their own.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict

from utilities.env import expand_path, load_env

_BUILTIN_RESUME = frozenset({"scratch", "last", "best"})
_SOURCE_PLACEHOLDER = re.compile(r"\{source\}|\{resume\}", re.IGNORECASE)


def _usable_name(raw: Any) -> str:
    name = str(raw).strip() if isinstance(raw, str) else ""
    return "" if name in ("", "None") else name


def parse_resume_value(raw: Any) -> tuple[str, str | None]:
    """Parse ``RESUME`` into ``(token, checkpoint_hint)``.

    Built-ins: ``scratch`` | ``last`` | ``best``.
    Cross-run: ``<source-run>`` or ``<source-run>:last`` / ``<source-run>:best``.
    """
    s = str(raw or "last").strip()
    if ":" in s:
        run, ckpt = s.rsplit(":", 1)
        if run.strip().lower() not in _BUILTIN_RESUME and ckpt.strip().lower() in ("last", "best"):
            return run.strip(), ckpt.strip().lower()
    return s, None


def is_cross_run_resume(raw: Any) -> bool:
    token, _ = parse_resume_value(raw)
    return token.lower() not in _BUILTIN_RESUME


def apply_resume_name_template(template: str, source: str) -> str:
    """Substitute ``{source}`` / ``{resume}`` in a resume-name template."""
    return _SOURCE_PLACEHOLDER.sub(source, str(template))


def resolve_experiment_name(config: Any) -> str | None:
    """Resolve the output run name from ``EXPERIMENT_NAME`` and ``RESUME_NAME``.

    Cross-run ``RESUME`` (a source run name, not scratch/last/best):

    * ``RESUME_NAME`` template is applied (``{source}`` -> source run name).
    * An explicit ``EXPERIMENT_NAME`` wins only when it differs from that source
      (setting ``EXPERIMENT_NAME`` to the source name is treated as a mistaken
      placeholder and ``RESUME_NAME`` is used instead).

    Otherwise ``EXPERIMENT_NAME`` is returned when set; if unset, ``None`` (W&B auto-name).
    """
    resume_raw = config.get("RESUME", "last")
    token, _ = parse_resume_value(resume_raw)
    cross_run = token.lower() not in _BUILTIN_RESUME

    explicit = _usable_name(config.get("EXPERIMENT_NAME", None))
    if cross_run:
        template = _usable_name(config.get("RESUME_NAME", None)) or "{source}"
        derived = apply_resume_name_template(template, token)
        if explicit and explicit != token:
            return explicit
        return derived

    if explicit:
        return explicit
    return None


def resolve_source_run_dir(source_name: str, create: bool = False) -> Path:
    """``$RESULTS_DIR/<source_name>/`` for a cross-run ``RESUME`` token."""
    load_env()
    results = os.environ.get("RESULTS_DIR") or str(Path.cwd() / "results")
    return Path(expand_path(results)) / str(source_name).strip()


def _checkpoint_in_stage_dir(stage_path: Path, which: str) -> Path | None:
    primary, fallback = ("best.pt", "last.pt") if which == "best" else ("last.pt", "best.pt")
    for fn in (primary, fallback):
        p = stage_path / fn
        if p.exists():
            return p
    return None


def _prev_stage_checkpoint_in_run(run_dir: Path, idx: int) -> Path | None:
    """``best.pt`` (else ``last.pt``) of the highest stage ``< idx`` that has one."""
    run_dir = Path(run_dir)
    for j in range(idx - 1, -1, -1):
        for d in sorted(run_dir.glob(f"stage{j}_*")):
            ckpt = _checkpoint_in_stage_dir(d, "best")
            if ckpt is not None:
                return ckpt
    bare = run_dir / f"stage{idx - 1}" if idx > 0 else None
    if bare is not None and bare.is_dir():
        ckpt = _checkpoint_in_stage_dir(bare, "best")
        if ckpt is not None:
            return ckpt
    return None


def find_cross_run_checkpoint(
    source_run_dir: Path,
    stage_idx: int,
    which: str = "best",
) -> Path | None:
    """Locate a warm-start checkpoint inside another run directory."""
    source_run_dir = Path(source_run_dir)
    if not source_run_dir.is_dir():
        return None

    which = str(which or "best").strip().lower()
    if which not in ("last", "best"):
        which = "best"

    for d in sorted(source_run_dir.glob(f"stage{stage_idx}_*")):
        ckpt = _checkpoint_in_stage_dir(d, which)
        if ckpt is not None:
            return ckpt

    bare = source_run_dir / f"stage{stage_idx}"
    if bare.is_dir():
        ckpt = _checkpoint_in_stage_dir(bare, which)
        if ckpt is not None:
            return ckpt

    return _prev_stage_checkpoint_in_run(source_run_dir, stage_idx)


def resolve_run_dir(config: Any, create: bool = True) -> Path:
    """``$RESULTS_DIR/<EXPERIMENT_NAME>/`` (``./results`` if RESULTS_DIR unset)."""
    load_env()
    results = os.environ.get("RESULTS_DIR") or str(Path.cwd() / "results")
    name = str(config.get("EXPERIMENT_NAME", "run"))
    run_dir = Path(expand_path(results)) / name
    if create:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _state_path(run_dir: Path) -> Path:
    return Path(run_dir) / "run_state.json"


def load_run_state(run_dir: Path) -> Dict[str, Any]:
    """Return the run's recorded state, or a fresh one (no stages completed)."""
    p = _state_path(run_dir)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {"completed": [], "stage_names": {}}


def save_run_state(run_dir: Path, state: Dict[str, Any]) -> None:
    """Atomically write ``run_state.json`` (temp file + replace)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(run_dir), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, _state_path(run_dir))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def mark_stage_complete(run_dir: Path, stage_idx: int, stage_name: str | None = None) -> None:
    """Record stage ``stage_idx`` as finished for this run."""
    state = load_run_state(run_dir)
    completed = set(state.get("completed", []))
    completed.add(int(stage_idx))
    state["completed"] = sorted(completed)
    if stage_name is not None:
        state.setdefault("stage_names", {})[str(stage_idx)] = stage_name
    save_run_state(run_dir, state)


def first_incomplete_stage(run_dir: Path, n_stages: int) -> int:
    """Lowest stage index not yet completed (``n_stages`` if all are done)."""
    completed = set(load_run_state(run_dir).get("completed", []))
    for i in range(n_stages):
        if i not in completed:
            return i
    return n_stages
