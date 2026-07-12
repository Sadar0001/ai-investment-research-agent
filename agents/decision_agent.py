from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, DecisionOutput
from prompts.agent_prompts import DECISION_AGENT_PROMPT


class DecisionAgent(BaseAgent[DecisionOutput]):
    name = "decision_agent"
    system_prompt = DECISION_AGENT_PROMPT
    timeout_seconds = 40.0
    temperature = 0.1

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        finance = context.get("finance_output")
        risk = context.get("risk_output")
        valuation = context.get("valuation_output")
        competitors = context.get("competitor_output")
        sentiment = context.get("sentiment_output")
        summary = context.get("summary_output")

        scores = {
            "financial_score": getattr(finance, "financial_score", None),
            "risk_score": getattr(risk, "risk_score", None),
            "valuation_score": getattr(valuation, "valuation_score", None),
            "competitor_score": getattr(competitors, "competitor_score", None),
            "sentiment_score": getattr(sentiment, "sentiment_score", None),
        }

        # Flag data completeness so the LLM can calibrate confidence honestly
        missing = [k for k, v in scores.items() if v is None]
        context["_missing_scores"] = missing

        schema_hint = (
            '{"decision": "INVEST"|"DO NOT INVEST", "overall_score": number (0-100), '
            '"confidence": number (0-100), "investment_horizon": str, '
            '"risk_level": "Low"|"Medium"|"High", "pros": [str], "cons": [str], '
            '"reasons": [str]}'
        )
        return (
            f"Upstream scores: {scores}\n"
            f"Missing scores (reduce confidence accordingly): {missing}\n\n"
            f"Executive summary: {getattr(summary, 'executive_summary', '')}\n\n"
            f"Make the final call. Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> DecisionOutput:
        def _num(val: Any, default: float = 0.0) -> float:
            try:
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        return DecisionOutput(
            agent_name=self.name,
            decision=raw_json.get("decision", "DO NOT INVEST"),
            overall_score=max(0.0, min(100.0, _num(raw_json.get("overall_score")))),
            confidence=max(0.0, min(100.0, _num(raw_json.get("confidence")))),
            investment_horizon=raw_json.get("investment_horizon", "Medium-Term"),
            risk_level=raw_json.get("risk_level", "Medium"),
            pros=raw_json.get("pros") or [],
            cons=raw_json.get("cons") or [],
            reasons=raw_json.get("reasons") or [],
        )

    def empty_output(self, error_message: str) -> DecisionOutput:
        return DecisionOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
