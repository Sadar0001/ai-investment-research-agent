from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, CompetitorOutput, CompetitorProfile
from prompts.agent_prompts import COMPETITOR_AGENT_PROMPT
from tools.finnhub_tool import fetch_peers
from tools.tavily_tool import tavily_search
from tools.yahoo_tool import fetch_financial_metrics


class CompetitorAgent(BaseAgent[CompetitorOutput]):
    name = "competitor_agent"
    system_prompt = COMPETITOR_AGENT_PROMPT
    timeout_seconds = 45.0
    temperature = 0.2

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        ticker = context["ticker"]
        company_name = context["company_name"]

        peer_tickers = fetch_peers(ticker)
        peer_data = {}
        for peer in peer_tickers:
            metrics = fetch_financial_metrics(peer)
            if metrics:
                peer_data[peer] = {
                    "market_cap": metrics.get("market_cap"),
                    "pe_ratio": metrics.get("pe_ratio"),
                    "revenue": metrics.get("revenue"),
                }

        if not peer_data:
            web = tavily_search(f"{company_name} main competitors", max_results=5)
            context["_web_competitors"] = web
        else:
            context["_web_competitors"] = []

        context["_peer_data"] = peer_data

        schema_hint = (
            '{"competitors": [{"name": str, "ticker": str|null, '
            '"revenue": number|null, "market_cap": number|null, '
            '"pe_ratio": number|null, "growth_rate": number|null, '
            '"strengths": [str], "weaknesses": [str]}], '
            '"competitor_score": number (0-100), "commentary": str}'
        )
        return (
            f"Subject company: {company_name} ({ticker})\n\n"
            f"Peer financial data (from Finnhub/Yahoo):\n{peer_data}\n\n"
            f"Web search results (if peer data was unavailable):\n{context['_web_competitors']}\n\n"
            f"Identify real named competitors and compare them. Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> CompetitorOutput:
        raw_competitors = raw_json.get("competitors") or []
        competitors = []
        for c in raw_competitors:
            try:
                competitors.append(
                    CompetitorProfile(
                        name=c.get("name", "Unknown"),
                        ticker=c.get("ticker"),
                        revenue=c.get("revenue"),
                        market_cap=c.get("market_cap"),
                        pe_ratio=c.get("pe_ratio"),
                        growth_rate=c.get("growth_rate"),
                        strengths=c.get("strengths") or [],
                        weaknesses=c.get("weaknesses") or [],
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        score = raw_json.get("competitor_score", 50.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 50.0

        return CompetitorOutput(
            agent_name=self.name,
            competitors=competitors,
            competitor_score=max(0.0, min(100.0, score)),
            commentary=raw_json.get("commentary", ""),
        )

    def empty_output(self, error_message: str) -> CompetitorOutput:
        return CompetitorOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
