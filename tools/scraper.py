"""
Lightweight, polite web scraper used as a last-resort fallback when
structured APIs don't return enough context (e.g. an official "About"
page). Respects timeouts and never crashes the caller on failure.
"""
from __future__ import annotations

from typing import Optional

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool

from config.settings import get_settings
from utils.cache import cached
from utils.logger import get_logger
from utils.retry import with_retry

logger = get_logger("tools.scraper")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; InvestIQ-AI-Research-Bot/1.0; "
        "+https://github.com/investiq-ai)"
    )
}


@with_retry(max_attempts=2)
def _fetch_html(url: str) -> str:
    settings = get_settings()
    resp = requests.get(url, headers=HEADERS, timeout=settings.request_timeout_seconds)
    resp.raise_for_status()
    return resp.text


@cached(ttl=3600)
def scrape_page_text(url: str, max_chars: int = 4000) -> Optional[str]:
    """Fetch a URL and return cleaned, visible text content (truncated)."""
    try:
        html = _fetch_html(url)
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars] if text else None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Scrape failed for {url}: {exc}")
        return None


@tool
def scrape_official_page(url: str) -> Optional[str]:
    """LangChain tool: scrape and clean the visible text of a company's
    official web page (e.g. About/Investor Relations) for context."""
    return scrape_page_text(url)
