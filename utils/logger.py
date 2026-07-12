"""
Centralized logging configuration using Loguru.

Every module imports `get_logger(__name__)` to obtain a bound logger
that writes structured logs to both console and rotating log files.
"""
from __future__ import annotations

import sys

from loguru import logger as _logger

from config.settings import get_settings

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    _logger.remove()

    _logger.add(
        sys.stderr,
        level=settings.log_level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[component]}</cyan> | {message}"
        ),
        filter=lambda record: record["extra"].setdefault("component", "app"),
    )

    _logger.add(
        settings.logs_dir / "investiq.log",
        level="DEBUG",
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
            "{extra[component]} | {message}"
        ),
        filter=lambda record: record["extra"].setdefault("component", "app"),
    )

    _logger.add(
        settings.logs_dir / "errors.log",
        level="ERROR",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
    )

    _CONFIGURED = True


def get_logger(component: str):
    """Return a Loguru logger bound to a component name (module/agent)."""
    _configure()
    return _logger.bind(component=component)
