"""
Report export service: renders a FullReport into Markdown, JSON, and
PDF formats and writes them to the `reports/` directory.
"""
from __future__ import annotations

from pathlib import Path

import orjson
from fpdf import FPDF

from config.settings import get_settings
from models.schemas import FullReport
from utils.logger import get_logger

logger = get_logger("services.report")


def _slug(company_name: str, ticker: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in f"{company_name}_{ticker}")
    return safe.strip("_").lower()


def to_markdown(report: FullReport) -> str:
    r = report
    lines: list[str] = []
    lines.append(f"# Investment Research Report: {r.company_name} ({r.ticker})")
    lines.append(f"_Generated: {r.generated_at.isoformat()}_\n")

    if r.decision:
        lines.append("## Final Decision")
        lines.append(f"- **Decision:** {r.decision.decision}")
        lines.append(f"- **Overall Score:** {r.decision.overall_score:.1f}/100")
        lines.append(f"- **Confidence:** {r.decision.confidence:.1f}/100")
        lines.append(f"- **Risk Level:** {r.decision.risk_level}")
        lines.append(f"- **Investment Horizon:** {r.decision.investment_horizon}\n")
        if r.decision.pros:
            lines.append("**Pros:**")
            lines += [f"- {p}" for p in r.decision.pros]
        if r.decision.cons:
            lines.append("\n**Cons:**")
            lines += [f"- {c}" for c in r.decision.cons]
        if r.decision.reasons:
            lines.append("\n**Reasons:**")
            lines += [f"- {x}" for x in r.decision.reasons]
        lines.append("")

    if r.summary:
        lines.append("## Executive Summary")
        lines.append(r.summary.executive_summary + "\n")

    if r.research and r.research.overview:
        o = r.research.overview
        lines.append("## Company Overview")
        lines.append(f"- **Sector:** {o.sector or 'N/A'}")
        lines.append(f"- **Industry:** {o.industry or 'N/A'}")
        lines.append(f"- **CEO:** {o.ceo or 'N/A'}")
        lines.append(f"- **Headquarters:** {o.headquarters or 'N/A'}")
        lines.append(f"- **Market Cap:** {o.market_cap or 'N/A'}")
        lines.append(f"\n{o.summary}\n")

    if r.finance and r.finance.metrics:
        m = r.finance.metrics
        lines.append("## Financial Health")
        lines.append(f"- **Financial Score:** {r.finance.financial_score:.1f}/100")
        lines.append(f"- Revenue: {m.revenue}")
        lines.append(f"- Net Income: {m.net_income}")
        lines.append(f"- PE Ratio: {m.pe_ratio}")
        lines.append(f"- PEG Ratio: {m.peg_ratio}")
        lines.append(f"- ROE: {m.roe}")
        lines.append(f"- Debt/Equity: {m.debt_to_equity}")
        lines.append(f"- 5Y Revenue CAGR: {m.revenue_cagr_5y}")
        lines.append(f"\n{r.finance.commentary}\n")

    if r.valuation:
        v = r.valuation
        lines.append("## Valuation")
        lines.append(f"- **Verdict:** {v.verdict}")
        lines.append(f"- **Valuation Score:** {v.valuation_score:.1f}/100")
        lines.append(f"- Current Price: {v.current_price}")
        lines.append(f"- Intrinsic Value (heuristic): {v.intrinsic_value}")
        lines.append(f"- Margin of Safety: {v.margin_of_safety}")
        lines.append(f"\n{v.commentary}\n")

    if r.risk:
        lines.append("## Risk Analysis")
        lines.append(f"- **Risk Score:** {r.risk.risk_score:.1f}/100")
        lines.append("\n| Category | Level | Description |")
        lines.append("|---|---|---|")
        for f in r.risk.factors:
            lines.append(f"| {f.category} | {f.level} | {f.description} |")
        lines.append(f"\n{r.risk.commentary}\n")

    if r.competitors:
        lines.append("## Competitor Analysis")
        lines.append(f"- **Competitor Score:** {r.competitors.competitor_score:.1f}/100")
        lines.append("\n| Name | Ticker | Market Cap | PE Ratio | Growth |")
        lines.append("|---|---|---|---|---|")
        for c in r.competitors.competitors:
            lines.append(f"| {c.name} | {c.ticker or '-'} | {c.market_cap or '-'} | {c.pe_ratio or '-'} | {c.growth_rate or '-'} |")
        lines.append(f"\n{r.competitors.commentary}\n")

    if r.news:
        lines.append("## News (Last 30 Days)")
        lines.append(f"Positive: {r.news.positive_count} | Negative: {r.news.negative_count} | Neutral: {r.news.neutral_count}\n")
        lines.append(r.news.overall_summary + "\n")

    if r.sentiment:
        s = r.sentiment
        lines.append("## Sentiment")
        lines.append(f"- News Sentiment: {s.news_sentiment}")
        lines.append(f"- Market Sentiment: {s.market_sentiment}")
        lines.append(f"- Analyst Sentiment: {s.analyst_sentiment}")
        lines.append(f"- Sentiment Score: {s.sentiment_score:.1f}")
        lines.append(f"\n{s.rationale}\n")

    if r.swot:
        sw = r.swot
        lines.append("## SWOT Analysis")
        lines.append("**Strengths:**")
        lines += [f"- {x}" for x in sw.strengths]
        lines.append("\n**Weaknesses:**")
        lines += [f"- {x}" for x in sw.weaknesses]
        lines.append("\n**Opportunities:**")
        lines += [f"- {x}" for x in sw.opportunities]
        lines.append("\n**Threats:**")
        lines += [f"- {x}" for x in sw.threats]

    lines.append("\n---\n*Generated by InvestIQ AI. This report is for informational and educational purposes only and does not constitute financial advice.*")
    return "\n".join(lines)


def to_json(report: FullReport) -> bytes:
    return orjson.dumps(report.model_dump(), option=orjson.OPT_INDENT_2 | orjson.OPT_NAIVE_UTC)


def to_pdf(report: FullReport) -> bytes:
    md_text = to_markdown(report)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    for raw_line in md_text.split("\n"):
        line = raw_line.replace("**", "").replace("#", "").strip()
        if not line:
            pdf.ln(3)
            continue
        try:
            pdf.multi_cell(0, 6, line)
        except Exception:  # noqa: BLE001
            # Skip characters FPDF's core font can't encode
            pdf.multi_cell(0, 6, line.encode("latin-1", "ignore").decode("latin-1"))

    return bytes(pdf.output())


def save_all_formats(report: FullReport) -> dict[str, Path]:
    """Persist markdown, JSON, and PDF versions of the report to disk."""
    settings = get_settings()
    slug = _slug(report.company_name, report.ticker)
    paths = {
        "markdown": settings.reports_dir / f"{slug}_report.md",
        "json": settings.reports_dir / f"{slug}_report.json",
        "pdf": settings.reports_dir / f"{slug}_report.pdf",
    }

    try:
        paths["markdown"].write_text(to_markdown(report), encoding="utf-8")
        paths["json"].write_bytes(to_json(report))
        paths["pdf"].write_bytes(to_pdf(report))
        logger.info(f"Saved report exports for {slug}")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed saving report exports: {exc}")

    return paths
