"""Minimal logger. Uses ``rich`` when available, falls back to plain prints.

Kept dependency-light on purpose; swap for a fuller logger later without
touching call sites.
"""

from __future__ import annotations

try:
    from rich.console import Console

    _console = Console()

    def _emit(msg: str, style: str = "") -> None:
        # markup=False so bracketed labels like "[train]" aren't parsed as
        # rich style tags (and stripped); style still colours the whole line.
        _console.print(msg, style=style, markup=False, highlight=False)
except Exception:  # noqa: BLE001
    def _emit(msg: str, style: str = "") -> None:
        print(msg)


_QUIET = False


def set_quiet(quiet: bool = True) -> None:
    """Mute ``info`` logging (warnings/errors still print). Handy in notebooks
    when calling the dataset builders in a loop."""
    global _QUIET
    _QUIET = quiet


class _Logger:
    def __init__(self, name: str, context: str = ""):
        self.name = name
        self.context = context

    def set_context(self, context: str) -> "_Logger":
        self.context = context
        return self

    def _fmt(self, level: str, msg: str) -> str:
        ctx = f"[{self.context}] " if self.context else ""
        return f"{level:7s} {ctx}{msg}"

    def info(self, msg: str, context: str | None = None) -> None:
        if _QUIET:
            return
        _emit(self._fmt("INFO", msg))

    def warning(self, msg: str, context: str | None = None) -> None:
        _emit(self._fmt("WARN", msg), style="yellow")

    def error(self, msg: str, context: str | None = None) -> None:
        _emit(self._fmt("ERROR", msg), style="bold red")


def get_logger(name: str = "twist") -> _Logger:
    return _Logger(name)
