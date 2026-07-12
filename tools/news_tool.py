"""
NewsAPI tool for fetching recent company news (last 30 days).
Degrades gracefully when NEWS_API_KEY is not configured.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import requests
from langchain.tools import tool

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("tools.news")

NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"


@with_retry(max_attempts=3)
def _call_newsapi(query: str, from_date: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.has_key("news_api_key"):
        logger.warning("NEWS_API_KEY not set; skipping NewsAPI fetch")
        return {"articles": []}

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 25,
        "apiKey": settings.news_api_key,
    }
    resp = requests.get(NEWSAPI_ENDPOINT, params=params, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.json()


@cached(ttl=1800)
def fetch_recent_news(company_name: str, days: int = 30) -> list[dict[str, Any]]:
    """Fetch news articles about a company from the last `days` days."""
    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        data = _call_newsapi(company_name, from_date)
        return [
            {
                "title": a.get("title", ""),
                "source": (a.get("source") or {}).get("name", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
                "description": a.get("description", "") or "",
            }
            for a in data.get("articles", [])
        ]
    except Exception as exc:  # noqa: BLE001
        logger.error(f"NewsAPI fetch failed for '{company_name}': {exc}")
        return []


@tool
def news_search(company_name: str) -> list[dict[str, Any]]:
    """LangChain tool: fetch the latest 30 days of news articles about
    a company, including title, source, URL, and short description."""
    return fetch_recent_news(company_name)
