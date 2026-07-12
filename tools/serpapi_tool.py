"""
SerpAPI (Google Search) tool. Degrades gracefully when
SERPAPI_API_KEY is not configured.
"""
from __future__ import annotations

from typing import Any

import requests
from langchain.tools import tool

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("tools.serpapi")

SERPAPI_ENDPOINT = "https://serpapi.com/search"


@with_retry(max_attempts=3)
def _call_serpapi(query: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.has_key("serpapi_api_key"):
        logger.warning("SERPAPI_API_KEY not set; skipping SerpAPI search")
        return {"organic_results": []}

    params = {
        "q": query,
        "api_key": settings.serpapi_api_key,
        "engine": "google",
        "num": 10,
    }
    resp = requests.get(SERPAPI_ENDPOINT, params=params, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.json()


@cached(ttl=1800)
def serpapi_search(query: str) -> list[dict[str, Any]]:
    """Run a Google search via SerpAPI and return normalized organic results."""
    try:
        data = _call_serpapi(query)
        return [
            {
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "snippet": r.get("snippet", ""),
            }
            for r in data.get("organic_results", [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error(f"SerpAPI search failed for '{query}': {exc}")
        return []


@tool
def serpapi_web_search(query: str) -> list[dict[str, Any]]:
    """LangChain tool: search Google via SerpAPI, useful for finding
    named competitors, recent announcements, and general web presence."""
    return serpapi_search(query)
