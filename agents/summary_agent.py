from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, SummaryOutput
from prompts.agent_prompts import SUMMARY_AGENT_PROMPT


class SummaryAgent(BaseAgent[SummaryOutput]):
    name = "summary_agent"
    system_prompt = SUMMARY_AGENT_PROMPT
    timeout_seconds = 35.0
    temperature = 0.3

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        parts = {
            "research": getattr(context.get("research_output"), "overview", None),
            "finance_score": getattr(context.get("finance_output"), "financial_score", None),
            "finance_commentary": getattr(context.get("finance_output"), "commentary", ""),
            "news_summary": getattr(context.get("news_output"), "overall_summary", ""),
            "sentiment": getattr(context.get("sentiment_output"), "rationale", ""),
            "competitors": getattr(context.get("competitor_output"), "commentary", ""),
            "risk": getattr(context.get("risk_output"), "commentary", ""),
            "valuation": getattr(context.get("valuation_output"), "commentary", ""),
            "swot": context.get("swot_output"),
        }

        schema_hint = '{"executive_summary": str}'
        return f"Agent outputs:\n{parts}\n\nWrite the executive summary. Return JSON matching:\n{schema_hint}"

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> SummaryOutput:
        return SummaryOutput(agent_name=self.name, executive_summary=raw_json.get("executive_summary", ""))

    def empty_output(self, error_message: str) -> SummaryOutput:
        return SummaryOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
