from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, SentimentOutput
from prompts.agent_prompts import SENTIMENT_AGENT_PROMPT
from tools.finnhub_tool import fetch_company_news_sentiment, fetch_recommendation_trends


class SentimentAgent(BaseAgent[SentimentOutput]):
    name = "sentiment_agent"
    system_prompt = SENTIMENT_AGENT_PROMPT
    timeout_seconds = 35.0
    temperature = 0.2

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        ticker = context["ticker"]
        news_output = context.get("news_output")
        finnhub_sentiment = fetch_company_news_sentiment(ticker)
        recommendations = fetch_recommendation_trends(ticker)

        news_counts = {
            "positive": getattr(news_output, "positive_count", 0),
            "negative": getattr(news_output, "negative_count", 0),
            "neutral": getattr(news_output, "neutral_count", 0),
        }

        schema_hint = (
            '{"news_sentiment": str, "market_sentiment": str, '
            '"analyst_sentiment": str, "social_sentiment": str|null, '
            '"sentiment_score": number (-100 to 100), "rationale": str}'
        )
        return (
            f"Ticker: {ticker}\n\nNews sentiment counts:\n{news_counts}\n\n"
            f"Finnhub aggregate sentiment:\n{finnhub_sentiment}\n\n"
            f"Analyst recommendation trend (latest first):\n{recommendations[:3]}\n\n"
            f"Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> SentimentOutput:
        score = raw_json.get("sentiment_score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        return SentimentOutput(
            agent_name=self.name,
            news_sentiment=raw_json.get("news_sentiment", "neutral"),
            market_sentiment=raw_json.get("market_sentiment", "neutral"),
            analyst_sentiment=raw_json.get("analyst_sentiment", "neutral"),
            social_sentiment=raw_json.get("social_sentiment"),
            sentiment_score=max(-100.0, min(100.0, score)),
            rationale=raw_json.get("rationale", ""),
        )

    def empty_output(self, error_message: str) -> SentimentOutput:
        return SentimentOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
