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
import tempfile
from pathlib import Path
from typing import Any, Dict

from utilities.env import expand_path, load_env


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
