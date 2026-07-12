"""
Disk-backed caching layer to avoid duplicate/expensive API calls.

Usage:
    from utils.cache import cached

    @cached(ttl=3600)
    def fetch_something(ticker: str) -> dict: ...
"""
from __future__ import annotations

import functools
import hashlib
import json
from typing import Any, Callable, TypeVar

import diskcache

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("cache")
T = TypeVar("T")

_settings = get_settings()
_cache = diskcache.Cache(str(_settings.cache_dir))


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    raw = json.dumps({"f": func_name, "a": args, "k": kwargs}, default=str, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def cached(ttl: int | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that caches a function's return value on disk keyed by its arguments."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = _make_key(func.__qualname__, args, kwargs)
            hit = _cache.get(key, default=None)
            if hit is not None:
                logger.debug(f"Cache hit for {func.__qualname__}")
                return hit  # type: ignore[return-value]

            result = func(*args, **kwargs)
            expire = ttl if ttl is not None else _settings.cache_ttl_seconds
            _cache.set(key, result, expire=expire)
            logger.debug(f"Cache stored for {func.__qualname__} (ttl={expire}s)")
            return result

        return wrapper

    return decorator


def clear_cache() -> None:
    """Clear the entire disk cache. Used by maintenance scripts/tests."""
    _cache.clear()
    logger.info("Cache cleared")
