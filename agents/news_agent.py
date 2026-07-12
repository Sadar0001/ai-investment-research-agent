from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, NewsItem, NewsOutput
from prompts.agent_prompts import NEWS_AGENT_PROMPT
from tools.news_tool import fetch_recent_news
from tools.tavily_tool import tavily_search


class NewsAgent(BaseAgent[NewsOutput]):
    name = "news_agent"
    system_prompt = NEWS_AGENT_PROMPT
    timeout_seconds = 40.0
    temperature = 0.2

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        company_name = context["company_name"]
        articles = fetch_recent_news(company_name, days=30)

        if not articles:
            # Fall back to Tavily's recency-aware search for headline-like snippets
            web = tavily_search(f"{company_name} news", max_results=8)
            articles = [
                {
                    "title": r.get("title", ""),
                    "source": "web",
                    "url": r.get("url", ""),
                    "published_at": "",
                    "description": r.get("content", "")[:300],
                }
                for r in web
            ]

        context["_articles"] = articles

        schema_hint = (
            '{"items": [{"title": str, "source": str, "url": str, '
            '"published_at": str, "sentiment": "positive"|"negative"|"neutral", '
            '"summary": str}], "positive_count": int, "negative_count": int, '
            '"neutral_count": int, "overall_summary": str}'
        )
        return (
            f"Company: {company_name}\n\nRecent articles (last 30 days):\n{articles}\n\n"
            f"Classify each article's sentiment and summarize. Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> NewsOutput:
        raw_items = raw_json.get("items") or []
        items = []
        for it in raw_items:
            try:
                items.append(
                    NewsItem(
                        title=it.get("title", ""),
                        source=it.get("source", ""),
                        url=it.get("url", ""),
                        published_at=it.get("published_at"),
                        sentiment=it.get("sentiment", "neutral"),
                        summary=it.get("summary", ""),
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        pos = raw_json.get("positive_count", sum(1 for i in items if i.sentiment == "positive"))
        neg = raw_json.get("negative_count", sum(1 for i in items if i.sentiment == "negative"))
        neu = raw_json.get("neutral_count", sum(1 for i in items if i.sentiment == "neutral"))

        return NewsOutput(
            agent_name=self.name,
            items=items,
            positive_count=pos,
            negative_count=neg,
            neutral_count=neu,
            overall_summary=raw_json.get("overall_summary", ""),
        )

    def empty_output(self, error_message: str) -> NewsOutput:
        return NewsOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
