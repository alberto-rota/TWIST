"""Environment / path resolution.

Config files refer to dataset roots with ``$DATASET_DIR/...`` placeholders so
the same YAML works on any machine; the actual root comes from ``.env`` (loaded
via :func:`load_env`). :func:`expand_path` resolves those placeholders.
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_LOADED = False


def load_env() -> None:
    """Load ``.env`` from the repo root once (idempotent)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # noqa: BLE001
        pass
    _ENV_LOADED = True


def expand_path(path: str) -> str:
    """Expand ``$VARS`` and ``~`` in a path string (loads ``.env`` first).

    ``$DATASET_DIR`` defaults to ``./DATA`` if unset, so the repo is usable
    out-of-the-box without a ``.env``.
    """
    load_env()
    os.environ.setdefault("DATASET_DIR", str(Path.cwd() / "DATA"))
    return os.path.expanduser(os.path.expandvars(str(path)))
