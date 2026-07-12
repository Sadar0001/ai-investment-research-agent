"""Unit tests for agent output parsing and fallback logic.
These do not call the real OpenAI API - they test parse_output/
empty_output/heuristic behavior directly."""
from __future__ import annotations

from models.schemas import AgentStatus, FinancialMetrics
from agents.finance_agent import FinanceAgent
from agents.decision_agent import DecisionAgent
from agents.swot_agent import SwotAgent


class TestFinanceAgent:
    def test_parse_output_builds_metrics(self):
        agent = FinanceAgent.__new__(FinanceAgent)  # bypass __init__ (no LLM needed)
        agent.name = "finance_agent"
        context = {
            "_metrics": {
                "revenue": 1000.0,
                "net_income": 100.0,
                "pe_ratio": 20.0,
                "debt_to_equity": 50.0,
                "revenue_cagr_5y": 8.0,
            }
        }
        raw_json = {"financial_score": 72.5, "commentary": "Solid margins."}
        output = agent.parse_output(raw_json, context)

        assert output.financial_score == 72.5
        assert output.metrics.revenue == 1000.0
        assert output.commentary == "Solid margins."

    def test_heuristic_score_bounded(self):
        metrics = FinancialMetrics(profit_margin=50, debt_to_equity=500, revenue_cagr_5y=100)
        score = FinanceAgent._heuristic_score(metrics)
        assert 0.0 <= score <= 100.0

    def test_empty_output_marks_failed(self):
        agent = FinanceAgent.__new__(FinanceAgent)
        agent.name = "finance_agent"
        output = agent.empty_output("boom")
        assert output.status == AgentStatus.FAILED
        assert output.error_message == "boom"


class TestDecisionAgent:
    def test_parse_output_clamps_scores(self):
        agent = DecisionAgent.__new__(DecisionAgent)
        agent.name = "decision_agent"
        raw_json = {
            "decision": "INVEST",
            "overall_score": 150,  # out of range, should clamp to 100
            "confidence": -10,  # out of range, should clamp to 0
            "pros": ["Strong margins"],
            "cons": ["High valuation"],
            "reasons": ["Consistent growth"],
        }
        output = agent.parse_output(raw_json, {})
        assert output.overall_score == 100.0
        assert output.confidence == 0.0
        assert output.decision == "INVEST"


class TestSwotAgent:
    def test_parse_output_defaults_to_empty_lists(self):
        agent = SwotAgent.__new__(SwotAgent)
        agent.name = "swot_agent"
        output = agent.parse_output({}, {})
        assert output.strengths == []
        assert output.threats == []
