"""
Supervisor: manually orchestrates all ten agents (no LangGraph/CrewAI).

Execution is organized into dependency-respecting stages so independent
agents run concurrently via asyncio.gather, while agents that need
upstream outputs (Sentiment needs News; SWOT/Summary/Decision need
almost everything) run after their dependencies resolve.

Stage 1 (parallel): Research, Finance, News, Competitor, Risk, Valuation
Stage 2 (parallel): Sentiment (needs News)
Stage 3 (sequential): SWOT -> Summary -> Decision
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

from agents.competitor_agent import CompetitorAgent
from agents.decision_agent import DecisionAgent
from agents.finance_agent import FinanceAgent
from agents.news_agent import NewsAgent
from agents.research_agent import ResearchAgent
from agents.risk_agent import RiskAgent
from agents.sentiment_agent import SentimentAgent
from agents.summary_agent import SummaryAgent
from agents.swot_agent import SwotAgent
from agents.valuation_agent import ValuationAgent
from models.schemas import FullReport
from utils.logger import get_logger

logger = get_logger("supervisor")

ProgressCallback = Optional[Callable[[str, str], None]]


class Supervisor:
    """Creates, runs, and merges outputs from all specialized agents."""

    def __init__(self) -> None:
        self.research_agent = ResearchAgent()
        self.finance_agent = FinanceAgent()
        self.news_agent = NewsAgent()
        self.competitor_agent = CompetitorAgent()
        self.risk_agent = RiskAgent()
        self.valuation_agent = ValuationAgent()
        self.sentiment_agent = SentimentAgent()
        self.swot_agent = SwotAgent()
        self.summary_agent = SummaryAgent()
        self.decision_agent = DecisionAgent()

    async def run_full_analysis(
        self,
        company_name: str,
        ticker: str,
        on_progress: ProgressCallback = None,
    ) -> FullReport:
        """Run the full 10-agent pipeline and return an aggregated report."""
        start = time.monotonic()
        base_context: dict[str, Any] = {"company_name": company_name, "ticker": ticker}

        def notify(agent: str, status: str) -> None:
            if on_progress:
                on_progress(agent, status)
            logger.info(f"[{agent}] {status}")

        # ---- Stage 1: independent agents run in parallel ----
        notify("stage_1", "started")
        stage1_names = ["research_agent", "finance_agent", "news_agent", "competitor_agent", "risk_agent", "valuation_agent"]
        for n in stage1_names:
            notify(n, "running")

        research_task = self.research_agent.run(dict(base_context))
        finance_task = self.finance_agent.run(dict(base_context))
        news_task = self.news_agent.run(dict(base_context))
        competitor_task = self.competitor_agent.run(dict(base_context))
        risk_task = self.risk_agent.run(dict(base_context))
        valuation_task = self.valuation_agent.run(dict(base_context))

        (
            research_output,
            finance_output,
            news_output,
            competitor_output,
            risk_output,
            valuation_output,
        ) = await asyncio.gather(
            research_task, finance_task, news_task, competitor_task, risk_task, valuation_task
        )

        for n in stage1_names:
            notify(n, "done")

        # Risk agent benefits from finance/research context; re-run cheaply enriched
        # (kept single-pass here for latency; context is passed for downstream stages)

        # ---- Stage 2: dependent on stage 1 outputs ----
        notify("sentiment_agent", "running")
        sentiment_context = {
            **base_context,
            "news_output": news_output,
        }
        sentiment_output = await self.sentiment_agent.run(sentiment_context)
        notify("sentiment_agent", "done")

        # ---- Stage 3: sequential synthesis ----
        shared_context = {
            **base_context,
            "research_output": research_output,
            "finance_output": finance_output,
            "news_output": news_output,
            "competitor_output": competitor_output,
            "risk_output": risk_output,
            "valuation_output": valuation_output,
            "sentiment_output": sentiment_output,
        }

        notify("swot_agent", "running")
        swot_output = await self.swot_agent.run(dict(shared_context))
        notify("swot_agent", "done")

        notify("summary_agent", "running")
        shared_context["swot_output"] = swot_output
        summary_output = await self.summary_agent.run(dict(shared_context))
        notify("summary_agent", "done")

        notify("decision_agent", "running")
        shared_context["summary_output"] = summary_output
        decision_output = await self.decision_agent.run(dict(shared_context))
        notify("decision_agent", "done")

        elapsed = time.monotonic() - start
        logger.info(f"Full analysis for {company_name} ({ticker}) completed in {elapsed:.1f}s")

        return FullReport(
            company_name=company_name,
            ticker=ticker,
            research=research_output,
            finance=finance_output,
            news=news_output,
            sentiment=sentiment_output,
            competitors=competitor_output,
            risk=risk_output,
            valuation=valuation_output,
            swot=swot_output,
            summary=summary_output,
            decision=decision_output,
        )

    def run_full_analysis_sync(
        self, company_name: str, ticker: str, on_progress: ProgressCallback = None
    ) -> FullReport:
        """Synchronous convenience wrapper for callers outside an event loop
        (e.g. Streamlit's script-execution model)."""
        return asyncio.run(self.run_full_analysis(company_name, ticker, on_progress))
