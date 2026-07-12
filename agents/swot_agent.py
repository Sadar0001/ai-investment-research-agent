from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, SwotOutput
from prompts.agent_prompts import SWOT_AGENT_PROMPT


class SwotAgent(BaseAgent[SwotOutput]):
    name = "swot_agent"
    system_prompt = SWOT_AGENT_PROMPT
    timeout_seconds = 35.0
    temperature = 0.3

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        research = context.get("research_output")
        finance = context.get("finance_output")
        competitors = context.get("competitor_output")
        risk = context.get("risk_output")

        schema_hint = (
            '{"strengths": [str], "weaknesses": [str], '
            '"opportunities": [str], "threats": [str]}'
        )
        return (
            f"Company overview: {getattr(research, 'overview', None)}\n\n"
            f"Financial summary: score={getattr(finance, 'financial_score', None)}, "
            f"commentary={getattr(finance, 'commentary', '')}\n\n"
            f"Competitor landscape: {getattr(competitors, 'commentary', '')}\n\n"
            f"Risk factors: {[f.category + ': ' + f.level for f in getattr(risk, 'factors', [])]}\n\n"
            f"Produce a grounded SWOT analysis. Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> SwotOutput:
        return SwotOutput(
            agent_name=self.name,
            strengths=raw_json.get("strengths") or [],
            weaknesses=raw_json.get("weaknesses") or [],
            opportunities=raw_json.get("opportunities") or [],
            threats=raw_json.get("threats") or [],
        )

    def empty_output(self, error_message: str) -> SwotOutput:
        return SwotOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
