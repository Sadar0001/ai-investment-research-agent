"""
Tavily web search tool. Degrades gracefully (returns empty results)
when TAVILY_API_KEY is not configured.
"""
from __future__ import annotations

from typing import Any

import requests
from langchain.tools import tool

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("tools.tavily")

TAVILY_ENDPOINT = "https://api.tavily.com/search"


@with_retry(max_attempts=3)
def _call_tavily(query: str, max_results: int) -> dict[str, Any]:
    settings = get_settings()
    if not settings.has_key("tavily_api_key"):
        logger.warning("TAVILY_API_KEY not set; skipping Tavily search")
        return {"results": []}

    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
    }
    resp = requests.post(TAVILY_ENDPOINT, json=payload, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.json()


@cached(ttl=1800)
def tavily_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a Tavily web search and return a normalized list of results."""
    try:
        data = _call_tavily(query, max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score"),
            }
            for r in data.get("results", [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Tavily search failed for '{query}': {exc}")
        return []


@tool
def tavily_web_search(query: str) -> list[dict[str, Any]]:
    """LangChain tool: search the web via Tavily for a given query and
    return titles, URLs, and short content snippets."""
    return tavily_search(query)
