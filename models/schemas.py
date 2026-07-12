"""
Pydantic models defining the structured output contract for every agent.

Each agent returns one of these models (never raw strings) so the
Supervisor and Streamlit UI can rely on a stable schema.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class BaseAgentOutput(BaseModel):
    """Common envelope every agent output inherits."""

    agent_name: str
    status: AgentStatus = AgentStatus.SUCCESS
    error_message: Optional[str] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CompanyOverview(BaseModel):
    company_name: str
    ticker: str
    sector: Optional[str] = None
    industry: Optional[str] = None
    ceo: Optional[str] = None
    headquarters: Optional[str] = None
    employees: Optional[int] = None
    market_cap: Optional[float] = None
    business_model: Optional[str] = None
    products: list[str] = Field(default_factory=list)
    mission: Optional[str] = None
    vision: Optional[str] = None
    summary: str = ""


class ResearchOutput(BaseAgentOutput):
    overview: Optional[CompanyOverview] = None


class FinancialMetrics(BaseModel):
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    free_cash_flow: Optional[float] = None
    total_debt: Optional[float] = None
    profit_margin: Optional[float] = None
    eps: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    roe: Optional[float] = None
    roce: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_cagr_5y: Optional[float] = None


class FinanceOutput(BaseAgentOutput):
    metrics: Optional[FinancialMetrics] = None
    financial_score: float = 0.0
    commentary: str = ""


class NewsItem(BaseModel):
    title: str
    source: str = ""
    url: str = ""
    published_at: Optional[str] = None
    sentiment: str = "neutral"
    summary: str = ""


class NewsOutput(BaseAgentOutput):
    items: list[NewsItem] = Field(default_factory=list)
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    overall_summary: str = ""


class SentimentOutput(BaseAgentOutput):
    news_sentiment: str = "neutral"
    market_sentiment: str = "neutral"
    analyst_sentiment: str = "neutral"
    social_sentiment: Optional[str] = None
    sentiment_score: float = 0.0
    rationale: str = ""


class CompetitorProfile(BaseModel):
    name: str
    ticker: Optional[str] = None
    revenue: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    growth_rate: Optional[float] = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class CompetitorOutput(BaseAgentOutput):
    competitors: list[CompetitorProfile] = Field(default_factory=list)
    competitor_score: float = 0.0
    commentary: str = ""


class RiskFactor(BaseModel):
    category: str
    level: str  # Low / Medium / High
    description: str


class RiskOutput(BaseAgentOutput):
    factors: list[RiskFactor] = Field(default_factory=list)
    risk_score: float = 0.0  # 0 (low risk) - 100 (high risk)
    commentary: str = ""


class ValuationOutput(BaseAgentOutput):
    current_price: Optional[float] = None
    intrinsic_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    margin_of_safety: Optional[float] = None
    verdict: str = "Fairly Valued"  # Overvalued / Fairly Valued / Undervalued
    valuation_score: float = 0.0
    commentary: str = ""


class SwotOutput(BaseAgentOutput):
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    threats: list[str] = Field(default_factory=list)


class SummaryOutput(BaseAgentOutput):
    executive_summary: str = ""


class DecisionOutput(BaseAgentOutput):
    decision: str = "DO NOT INVEST"  # INVEST / DO NOT INVEST
    overall_score: float = 0.0
    confidence: float = 0.0
    investment_horizon: str = "Medium-Term"
    risk_level: str = "Medium"
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class FullReport(BaseModel):
    """Aggregated output of the entire multi-agent pipeline."""

    company_name: str
    ticker: str
    research: Optional[ResearchOutput] = None
    finance: Optional[FinanceOutput] = None
    news: Optional[NewsOutput] = None
    sentiment: Optional[SentimentOutput] = None
    competitors: Optional[CompetitorOutput] = None
    risk: Optional[RiskOutput] = None
    valuation: Optional[ValuationOutput] = None
    swot: Optional[SwotOutput] = None
    summary: Optional[SummaryOutput] = None
    decision: Optional[DecisionOutput] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
