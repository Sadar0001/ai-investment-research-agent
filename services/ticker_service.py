"""
Resolves a free-text company name (e.g. "Apple") to a stock ticker
(e.g. "AAPL") using Yahoo Finance's public search endpoint, with a
small static fallback map for common companies to keep the app usable
even if the search endpoint is rate-limited.
"""
from __future__ import annotations

import requests

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("services.ticker")

YAHOO_SEARCH_ENDPOINT = "https://query2.finance.yahoo.com/v1/finance/search"

_FALLBACK_MAP = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "nvidia": "NVDA",
    "netflix": "NFLX",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "tata motors": "TATAMOTORS.NS",
    "hdfc bank": "HDFCBANK.NS",
}


@with_retry(max_attempts=2)
def _search_yahoo(query: str) -> dict:
    settings = get_settings()
    headers = {"User-Agent": "Mozilla/5.0 (InvestIQ-AI)"}
    resp = requests.get(
        YAHOO_SEARCH_ENDPOINT,
        params={"q": query, "quotesCount": 5, "newsCount": 0},
        headers=headers,
        timeout=settings.request_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json()


@cached(ttl=86400)
def resolve_ticker(company_name: str) -> str | None:
    """Best-effort resolution of a company name to its primary ticker."""
    normalized = company_name.strip().lower()

    # If it already looks like a ticker (short, uppercase-ish), trust it.
    if company_name.isupper() and 1 <= len(company_name) <= 6 and " " not in company_name:
        return company_name

    if normalized in _FALLBACK_MAP:
        return _FALLBACK_MAP[normalized]

    try:
        data = _search_yahoo(company_name)
        quotes = data.get("quotes", [])
        for q in quotes:
            if q.get("quoteType") == "EQUITY" and q.get("symbol"):
                return q["symbol"]
        if quotes:
            return quotes[0].get("symbol")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Ticker search failed for '{company_name}': {exc}")

    return None
