from __future__ import annotations

import orjson

from models.schemas import DecisionOutput, FullReport
from services.report_service import to_json, to_markdown


def _sample_report() -> FullReport:
    return FullReport(
        company_name="Apple Inc.",
        ticker="AAPL",
        decision=DecisionOutput(
            agent_name="decision_agent",
            decision="INVEST",
            overall_score=82.0,
            confidence=75.0,
            pros=["Strong brand", "Healthy margins"],
            cons=["Rich valuation"],
            reasons=["Consistent growth"],
        ),
    )


def test_to_markdown_includes_key_sections():
    report = _sample_report()
    md = to_markdown(report)
    assert "Apple Inc." in md
    assert "INVEST" in md
    assert "Strong brand" in md


def test_to_json_round_trips():
    report = _sample_report()
    data = orjson.loads(to_json(report))
    assert data["company_name"] == "Apple Inc."
    assert data["decision"]["decision"] == "INVEST"
