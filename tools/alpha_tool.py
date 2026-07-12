"""
Alpha Vantage tool for supplemental fundamental overview data.
Degrades gracefully when ALPHA_VANTAGE_API_KEY is not configured.
"""
from __future__ import annotations

from typing import Any

import requests
from langchain.tools import tool

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("tools.alpha_vantage")

ALPHA_BASE = "https://www.alphavantage.co/query"


@with_retry(max_attempts=3)
def _call_alpha(function: str, symbol: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.has_key("alpha_vantage_api_key"):
        logger.warning("ALPHA_VANTAGE_API_KEY not set; skipping Alpha Vantage call")
        return {}

    params = {
        "function": function,
        "symbol": symbol,
        "apikey": settings.alpha_vantage_api_key,
    }
    resp = requests.get(ALPHA_BASE, params=params, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.json()


@cached(ttl=3600)
def fetch_company_overview_av(ticker: str) -> dict[str, Any]:
    """Fetch Alpha Vantage's fundamental OVERVIEW endpoint for a ticker."""
    try:
        data = _call_alpha("OVERVIEW", ticker)
        if not data or "Symbol" not in data:
            return {}
        return {
            "pe_ratio": _safe_float(data.get("PERatio")),
            "peg_ratio": _safe_float(data.get("PEGRatio")),
            "eps": _safe_float(data.get("EPS")),
            "roe": _safe_float(data.get("ReturnOnEquityTTM")),
            "profit_margin": _safe_float(data.get("ProfitMargin")),
            "market_cap": _safe_float(data.get("MarketCapitalization")),
            "beta": _safe_float(data.get("Beta")),
            "analyst_target_price": _safe_float(data.get("AnalystTargetPrice")),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Alpha Vantage overview fetch failed for {ticker}: {exc}")
        return {}


def _safe_float(value: Any) -> float | None:
    try:
        if value in (None, "None", "-", ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@tool
def alpha_vantage_lookup(ticker: str) -> dict[str, Any]:
    """LangChain tool: fetch supplemental fundamental ratios (PE, PEG,
    EPS, ROE, beta, analyst target price) from Alpha Vantage."""
    return fetch_company_overview_av(ticker)
