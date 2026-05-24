"""Single source of truth for the project logger."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def get_logger(name: str = "ctd", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers:
        return log
    log.setLevel(level)
    handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.propagate = False
    return log
