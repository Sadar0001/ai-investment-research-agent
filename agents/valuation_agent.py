from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, ValuationOutput
from prompts.agent_prompts import VALUATION_AGENT_PROMPT
from tools.yahoo_tool import fetch_financial_metrics


class ValuationAgent(BaseAgent[ValuationOutput]):
    name = "valuation_agent"
    system_prompt = VALUATION_AGENT_PROMPT
    timeout_seconds = 40.0
    temperature = 0.1

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        ticker = context["ticker"]
        metrics = fetch_financial_metrics(ticker)
        context["_metrics"] = metrics

        schema_hint = (
            '{"current_price": number|null, "intrinsic_value": number|null, '
            '"pe_ratio": number|null, "peg_ratio": number|null, '
            '"margin_of_safety": number|null, '
            '"verdict": "Overvalued"|"Fairly Valued"|"Undervalued", '
            '"valuation_score": number (0-100), "commentary": str}'
        )
        return (
            f"Ticker: {ticker}\n\n"
            f"Current Price: {metrics.get('current_price')}\n"
            f"PE Ratio: {metrics.get('pe_ratio')}\n"
            f"PEG Ratio: {metrics.get('peg_ratio')}\n"
            f"EPS: {metrics.get('eps')}\n"
            f"5Y Revenue CAGR: {metrics.get('revenue_cagr_5y')}\n"
            f"52-Week High/Low: {metrics.get('52w_high')}/{metrics.get('52w_low')}\n\n"
            f"Estimate valuation using a clearly-stated heuristic. Return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> ValuationOutput:
        metrics = context.get("_metrics", {})

        def _num(val: Any) -> float | None:
            try:
                return float(val) if val is not None else None
            except (TypeError, ValueError):
                return None

        score = raw_json.get("valuation_score", 50.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 50.0

        return ValuationOutput(
            agent_name=self.name,
            current_price=_num(raw_json.get("current_price")) or metrics.get("current_price"),
            intrinsic_value=_num(raw_json.get("intrinsic_value")),
            pe_ratio=_num(raw_json.get("pe_ratio")) or metrics.get("pe_ratio"),
            peg_ratio=_num(raw_json.get("peg_ratio")) or metrics.get("peg_ratio"),
            margin_of_safety=_num(raw_json.get("margin_of_safety")),
            verdict=raw_json.get("verdict", "Fairly Valued"),
            valuation_score=max(0.0, min(100.0, score)),
            commentary=raw_json.get("commentary", ""),
        )

    def empty_output(self, error_message: str) -> ValuationOutput:
        return ValuationOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
