from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, CompanyOverview, ResearchOutput
from prompts.agent_prompts import RESEARCH_AGENT_PROMPT
from tools.scraper import scrape_page_text
from tools.tavily_tool import tavily_search
from tools.yahoo_tool import fetch_company_overview


class ResearchAgent(BaseAgent[ResearchOutput]):
    name = "research_agent"
    system_prompt = RESEARCH_AGENT_PROMPT
    timeout_seconds = 40.0
    temperature = 0.2

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        ticker = context["ticker"]
        company_name = context["company_name"]

        overview = fetch_company_overview(ticker)
        web_results = tavily_search(f"{company_name} company overview business model", max_results=4)
        page_text = None
        website = overview.get("website")
        if website:
            page_text = scrape_page_text(website)

        context["_yahoo_overview"] = overview

        schema_hint = (
            '{"company_name": str, "ticker": str, "sector": str|null, '
            '"industry": str|null, "ceo": str|null, "headquarters": str|null, '
            '"employees": int|null, "market_cap": number|null, '
            '"business_model": str|null, "products": [str], '
            '"mission": str|null, "vision": str|null, "summary": str}'
        )

        return (
            f"Company: {company_name}\nTicker: {ticker}\n\n"
            f"Yahoo Finance data:\n{overview}\n\n"
            f"Web search snippets:\n{web_results}\n\n"
            f"Official site excerpt:\n{page_text or 'N/A'}\n\n"
            f"Return JSON matching this schema:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> ResearchOutput:
        yahoo = context.get("_yahoo_overview", {})
        overview = CompanyOverview(
            company_name=raw_json.get("company_name") or yahoo.get("company_name") or context["company_name"],
            ticker=raw_json.get("ticker") or context["ticker"],
            sector=raw_json.get("sector") or yahoo.get("sector"),
            industry=raw_json.get("industry") or yahoo.get("industry"),
            ceo=raw_json.get("ceo") or yahoo.get("ceo"),
            headquarters=raw_json.get("headquarters") or yahoo.get("headquarters"),
            employees=raw_json.get("employees") or yahoo.get("employees"),
            market_cap=raw_json.get("market_cap") or yahoo.get("market_cap"),
            business_model=raw_json.get("business_model") or yahoo.get("business_model"),
            products=raw_json.get("products") or [],
            mission=raw_json.get("mission"),
            vision=raw_json.get("vision"),
            summary=raw_json.get("summary", ""),
        )
        return ResearchOutput(agent_name=self.name, overview=overview)

    def empty_output(self, error_message: str) -> ResearchOutput:
        return ResearchOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
