"""Rich-colored, DDP-aware logger for TWIST.

Console output is limited to rank 0 when running under DDP (checked at emit
time, so the logger can be created before dist.init_process_group).
Based on the unreflectanything logger pattern.
"""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# --------------------------------------------------------------------------- #
# DDP helpers (safe before dist is initialized)
# --------------------------------------------------------------------------- #
def _get_rank() -> Optional[int]:
    try:
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return None
        return dist.get_rank()
    except Exception:
        return None


def _is_main_process() -> bool:
    rank = _get_rank()
    return rank is None or rank == 0


# --------------------------------------------------------------------------- #
# Rich theme
# --------------------------------------------------------------------------- #
CUSTOM_THEME = Theme({
    "main":       "white",
    "engine":     "yellow",
    "training":   "orange1",
    "validation": "green",
    "test":       "cyan",
    "dataset":    "green",
    "config":     "cyan",
    "checkpoint": "magenta",
    "resume":     "magenta",
    "wandb":      "bright_yellow",
    "warning":    "yellow",
    "error":      "bold red",
    "info":       "white",
    "ddp":        "blue",
    "debug":      "dim cyan",
})


def _strip_markup(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\](.*?)\[/\1\]", r"\2", text)
    text = re.sub(r"\[([^\]]+)\]", "", text)
    return text


def _align(s: str, width: int = 10) -> str:
    return s[:width].ljust(width)


# --------------------------------------------------------------------------- #
# File formatter (strips Rich markup)
# --------------------------------------------------------------------------- #
class _PlainFormatter(logging.Formatter):
    def format(self, record):
        record.msg = _strip_markup(record.msg)
        return super().format(record)


# --------------------------------------------------------------------------- #
# Logger class
# --------------------------------------------------------------------------- #
_QUIET = False


def set_quiet(quiet: bool = True) -> None:
    global _QUIET
    _QUIET = quiet


class _Logger:
    def __init__(self, name: str, context: str = "INFO", log_file: Optional[str] = None):
        self.name = name
        self.context = context
        self._console = Console(theme=CUSTOM_THEME)
        self._py_logger = logging.getLogger(name)
        self._py_logger.setLevel(logging.DEBUG)

        if self._py_logger.hasHandlers():
            self._py_logger.handlers.clear()

        # Console handler: emits only on rank 0
        _parent = self

        class _RankFilteredHandler(RichHandler):
            def emit(self, record):
                if not _is_main_process():
                    return
                record.message = record.getMessage()
                _parent._console.print(record.message)

        rich_h = _RankFilteredHandler(
            console=self._console,
            rich_tracebacks=True,
            show_time=False,
            show_path=False,
            show_level=False,
            markup=True,
        )
        rich_h.setLevel(logging.DEBUG)
        self._py_logger.addHandler(rich_h)

        # File handler
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(_PlainFormatter("%(asctime)s [%(context)s] %(message)s"))
            self._py_logger.addHandler(fh)

    def set_context(self, context: str) -> "_Logger":
        self.context = context
        return self

    def _emit(self, msg: str, ctx: Optional[str], level: int) -> None:
        if _QUIET and level < logging.WARNING:
            return
        context = (ctx or self.context).upper()
        style = context.lower()
        if style not in CUSTOM_THEME.styles:
            style = "info"
        time_str = datetime.now().strftime("%H:%M:%S")
        rich_msg = f"[{style}]{_align(context)}[/{style}] [{time_str}] {msg}"
        self._py_logger.log(level, rich_msg, extra={"context": context})

    def info(self, msg: str, context: Optional[str] = None, **_) -> None:
        self._emit(msg, context, logging.INFO)

    def warning(self, msg: str, context: Optional[str] = None, **_) -> None:
        self._emit(msg, context, logging.WARNING)

    def error(self, msg: str, context: Optional[str] = None, **_) -> None:
        self._emit(msg, context, logging.ERROR)

    def debug(self, msg: str, context: Optional[str] = None, **_) -> None:
        self._emit(msg, context, logging.DEBUG)


# --------------------------------------------------------------------------- #
# Module-level cache + factory
# --------------------------------------------------------------------------- #
_loggers: dict[str, _Logger] = {}


def get_logger(name: str = "twist") -> _Logger:
    if name in _loggers:
        return _loggers[name]
    log_dir = os.path.expandvars(
        os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "logs"))
    )
    log_file = os.path.join(log_dir, f"{name.split('.')[-1]}.log")
    logger = _Logger(name, log_file=log_file)
    _loggers[name] = logger
    return logger
