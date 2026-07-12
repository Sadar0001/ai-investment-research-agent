"""
Finnhub tool for supplemental fundamentals, peer lists, and
analyst sentiment. Degrades gracefully when FINNHUB_API_KEY is
not configured.
"""
from __future__ import annotations

from typing import Any

import requests
from langchain.tools import tool

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("tools.finnhub")

FINNHUB_BASE = "https://finnhub.io/api/v1"


@with_retry(max_attempts=3)
def _get(path: str, params: dict[str, Any]) -> Any:
    settings = get_settings()
    if not settings.has_key("finnhub_api_key"):
        logger.warning("FINNHUB_API_KEY not set; skipping Finnhub call")
        return None

    params = {**params, "token": settings.finnhub_api_key}
    resp = requests.get(f"{FINNHUB_BASE}/{path}", params=params, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.json()


@cached(ttl=1800)
def fetch_peers(ticker: str) -> list[str]:
    """Return a list of peer ticker symbols for the given company."""
    try:
        data = _get("stock/peers", {"symbol": ticker})
        if not data:
            return []
        return [p for p in data if p != ticker][:6]
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Finnhub peers fetch failed for {ticker}: {exc}")
        return []


@cached(ttl=1800)
def fetch_recommendation_trends(ticker: str) -> list[dict[str, Any]]:
    """Return analyst buy/hold/sell recommendation trend data."""
    try:
        data = _get("stock/recommendation", {"symbol": ticker})
        return data or []
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Finnhub recommendation fetch failed for {ticker}: {exc}")
        return []


@cached(ttl=900)
def fetch_company_news_sentiment(ticker: str) -> dict[str, Any]:
    """Return Finnhub's aggregate news sentiment score for the company."""
    try:
        data = _get("news-sentiment", {"symbol": ticker})
        return data or {}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Finnhub sentiment fetch failed for {ticker}: {exc}")
        return {}


@tool
def finnhub_lookup(ticker: str) -> dict[str, Any]:
    """LangChain tool: fetch peer companies, analyst recommendation
    trends, and aggregate news sentiment for a ticker from Finnhub."""
    return {
        "peers": fetch_peers(ticker),
        "recommendations": fetch_recommendation_trends(ticker),
        "sentiment": fetch_company_news_sentiment(ticker),
    }
