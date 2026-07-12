from __future__ import annotations

from typing import Any

from agents.base_agent import BaseAgent
from models.schemas import AgentStatus, FinanceOutput, FinancialMetrics
from prompts.agent_prompts import FINANCE_AGENT_PROMPT
from tools.alpha_tool import fetch_company_overview_av
from tools.yahoo_tool import fetch_financial_metrics


class FinanceAgent(BaseAgent[FinanceOutput]):
    name = "finance_agent"
    system_prompt = FINANCE_AGENT_PROMPT
    timeout_seconds = 40.0
    temperature = 0.1

    def build_user_prompt(self, context: dict[str, Any]) -> str:
        ticker = context["ticker"]
        metrics = fetch_financial_metrics(ticker)
        av_metrics = fetch_company_overview_av(ticker)

        # Alpha Vantage fills gaps Yahoo left empty
        for key in ("pe_ratio", "peg_ratio", "eps", "roe", "profit_margin"):
            if metrics.get(key) is None and av_metrics.get(key) is not None:
                metrics[key] = av_metrics[key]

        context["_metrics"] = metrics

        schema_hint = (
            '{"financial_score": number (0-100), "commentary": str}'
        )
        return (
            f"Ticker: {ticker}\n\nRaw financial metrics:\n{metrics}\n\n"
            f"Assess financial health and return JSON matching:\n{schema_hint}"
        )

    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> FinanceOutput:
        raw_metrics = context.get("_metrics", {})
        metrics = FinancialMetrics(
            revenue=raw_metrics.get("revenue"),
            net_income=raw_metrics.get("net_income"),
            operating_cash_flow=raw_metrics.get("operating_cash_flow"),
            free_cash_flow=raw_metrics.get("free_cash_flow"),
            total_debt=raw_metrics.get("total_debt"),
            profit_margin=raw_metrics.get("profit_margin"),
            eps=raw_metrics.get("eps"),
            pe_ratio=raw_metrics.get("pe_ratio"),
            peg_ratio=raw_metrics.get("peg_ratio"),
            roe=raw_metrics.get("roe"),
            roce=raw_metrics.get("roce"),
            current_ratio=raw_metrics.get("current_ratio"),
            quick_ratio=raw_metrics.get("quick_ratio"),
            debt_to_equity=raw_metrics.get("debt_to_equity"),
            revenue_cagr_5y=raw_metrics.get("revenue_cagr_5y"),
        )
        score = raw_json.get("financial_score")
        score = float(score) if isinstance(score, (int, float)) else self._heuristic_score(metrics)

        return FinanceOutput(
            agent_name=self.name,
            metrics=metrics,
            financial_score=max(0.0, min(100.0, score)),
            commentary=raw_json.get("commentary", ""),
        )

    @staticmethod
    def _heuristic_score(metrics: FinancialMetrics) -> float:
        """Fallback deterministic score if the LLM output is unusable."""
        score = 50.0
        if metrics.profit_margin:
            score += min(max(metrics.profit_margin, -20), 20)
        if metrics.debt_to_equity is not None:
            score -= min(metrics.debt_to_equity / 10, 20)
        if metrics.revenue_cagr_5y:
            score += min(max(metrics.revenue_cagr_5y, -10), 15)
        return max(0.0, min(100.0, score))

    def empty_output(self, error_message: str) -> FinanceOutput:
        return FinanceOutput(agent_name=self.name, status=AgentStatus.FAILED, error_message=error_message)
