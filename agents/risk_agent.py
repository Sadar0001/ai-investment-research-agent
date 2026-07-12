from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, RiskFactor, RiskOutput
from prompts.agent_prompts import RISK_AGENT_PROMPT
from tools.tavily_tool import tavily_search


class RiskAgent(BaseAgent[RiskOutput]):
    name = "risk_agent"
    system_prompt = RISK_AGENT_PROMPT
    timeout_seconds = 45.0
    temperature = 0.2

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        company_name = context["company_name"]
        finance_output = context.get("finance_output")
        research_output = context.get("research_output")

        risk_search = tavily_search(f"{company_name} lawsuit regulatory risk controversy", max_results=5)

        metrics = getattr(finance_output, "metrics", None)
        overview = getattr(research_output, "overview", None)

        schema_hint = (
            '{"factors": [{"category": str, "level": "Low"|"Medium"|"High", '
            '"description": str}], "risk_score": number (0-100), "commentary": str}'
        )
        return (
            f"Company: {company_name}\n\n"
            f"Sector/Industry: {getattr(overview, 'sector', None)} / {getattr(overview, 'industry', None)}\n"
            f"Debt-to-Equity: {getattr(metrics, 'debt_to_equity', None)}\n"
            f"Current Ratio: {getattr(metrics, 'current_ratio', None)}\n\n"
            f"Recent risk-related web search results:\n{risk_search}\n\n"
            f"Assess risk across Legal, Debt, Political, Economic, Regulatory, "
            f"Management, Competition, Technology, Cybersecurity, Supply Chain. "
            f"Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> RiskOutput:
        raw_factors = raw_json.get("factors") or []
        factors = []
        for f in raw_factors:
            try:
                factors.append(
                    RiskFactor(
                        category=f.get("category", "General"),
                        level=f.get("level", "Medium"),
                        description=f.get("description", ""),
                    )
                )
            except Exception:  # noqa: BLE001
                continue

        score = raw_json.get("risk_score", 50.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 50.0

        return RiskOutput(
            agent_name=self.name,
            factors=factors,
            risk_score=max(0.0, min(100.0, score)),
            commentary=raw_json.get("commentary", ""),
        )

    def empty_output(self, error_message: str) -> RiskOutput:
        return RiskOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
