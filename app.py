"""
InvestIQ AI - Professional AI Investment Research Platform

Streamlit entrypoint. Run with:
    streamlit run app.py
"""
from __future__ import annotations

import time
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from agents.supervisor import Supervisor
from config.settings import get_settings
from models.schemas import FullReport
from services.report_service import to_json, to_markdown, to_pdf
from services.ticker_service import resolve_ticker
from utils.logger import get_logger

logger = get_logger("app")
settings = get_settings()

st.set_page_config(
    page_title="InvestIQ AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Dark, professional theme
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; color: #e6e6e6; }
    .metric-card {
        background: linear-gradient(145deg, #161b22, #1c2128);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .metric-label { font-size: 13px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 28px; font-weight: 700; color: #ffffff; }
    .invest-badge {
        display: inline-block; padding: 6px 16px; border-radius: 20px;
        font-weight: 700; font-size: 15px;
    }
    .invest-yes { background-color: #1f6f43; color: #d3f9d8; }
    .invest-no { background-color: #7a2020; color: #ffd6d6; }
    .section-header {
        font-size: 20px; font-weight: 700; margin-top: 28px; margin-bottom: 10px;
        border-left: 4px solid #4f8cff; padding-left: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "report" not in st.session_state:
    st.session_state.report = None
if "company_display" not in st.session_state:
    st.session_state.company_display = ""

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("📊 InvestIQ AI")
    st.caption("Professional AI Investment Research Platform")
    st.divider()

    company_input = st.text_input("Company name or ticker", placeholder="e.g. Apple, TCS, Reliance")
    analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.divider()
    st.caption("Data sources")
    st.markdown(
        "- Yahoo Finance (always on)\n"
        "- Finnhub, Alpha Vantage, NewsAPI, Tavily, SerpAPI (if keys configured)"
    )

    if not settings.has_key("openai_api_key"):
        st.warning("OPENAI_API_KEY is not set. Add it to your .env file before analyzing.")

    st.divider()
    st.caption("⚠️ Educational tool only. Not financial advice.")

# ----------------------------------------------------------------------
# Analysis trigger
# ----------------------------------------------------------------------
if analyze_clicked and company_input.strip():
    ticker = resolve_ticker(company_input.strip())
    if not ticker:
        st.error(f"Could not resolve a ticker for '{company_input}'. Try the exact ticker symbol instead.")
    else:
        st.session_state.company_display = company_input.strip()
        progress_area = st.container()
        progress_bar = progress_area.progress(0, text="Starting multi-agent analysis...")
        status_lines: list[str] = []
        status_box = progress_area.empty()

        agent_order = [
            "research_agent", "finance_agent", "news_agent", "competitor_agent",
            "risk_agent", "valuation_agent", "sentiment_agent", "swot_agent",
            "summary_agent", "decision_agent",
        ]
        total_steps = len(agent_order)
        completed: set[str] = set()

        def on_progress(agent: str, status: str) -> None:
            if status == "done" and agent in agent_order:
                completed.add(agent)
                pct = int(len(completed) / total_steps * 100)
                progress_bar.progress(min(pct, 100), text=f"Completed {agent.replace('_', ' ').title()} ({len(completed)}/{total_steps})")
            if agent not in ("stage_1",):
                status_lines.append(f"`{agent}` → {status}")
                status_box.markdown("\n\n".join(status_lines[-6:]))

        supervisor = Supervisor()
        with st.spinner("Running specialized AI agents..."):
            try:
                report = supervisor.run_full_analysis_sync(
                    company_name=company_input.strip(), ticker=ticker, on_progress=on_progress
                )
                st.session_state.report = report
                progress_bar.progress(100, text="Analysis complete")
            except Exception as exc:  # noqa: BLE001
                logger.exception(f"Analysis failed: {exc}")
                st.error(f"Analysis failed: {exc}")

elif analyze_clicked:
    st.warning("Please enter a company name or ticker.")


# ----------------------------------------------------------------------
# Dashboard rendering
# ----------------------------------------------------------------------
def render_metric_card(col, label: str, value: Any, suffix: str = "") -> None:
    col.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}{suffix}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_radar_chart(report: FullReport) -> go.Figure:
    categories = ["Financial", "Valuation", "Risk (inverted)", "Competitor", "Sentiment"]
    finance = report.finance.financial_score if report.finance else 0
    valuation = report.valuation.valuation_score if report.valuation else 0
    risk_inv = 100 - (report.risk.risk_score if report.risk else 50)
    competitor = report.competitors.competitor_score if report.competitors else 0
    sentiment = ((report.sentiment.sentiment_score if report.sentiment else 0) + 100) / 2

    values = [finance, valuation, risk_inv, competitor, sentiment]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself", line_color="#4f8cff"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e6e6e6",
        margin=dict(l=40, r=40, t=20, b=20),
    )
    return fig


def render_news_pie(report: FullReport) -> go.Figure:
    n = report.news
    labels = ["Positive", "Negative", "Neutral"]
    values = [n.positive_count, n.negative_count, n.neutral_count] if n else [0, 0, 0]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.5, marker_colors=["#2ea043", "#da3633", "#8b949e"])])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6e6e6", margin=dict(l=10, r=10, t=10, b=10))
    return fig


def render_competitor_bar(report: FullReport) -> go.Figure:
    c = report.competitors
    names = [comp.name for comp in c.competitors] if c else []
    market_caps = [comp.market_cap or 0 for comp in c.competitors] if c else []
    fig = go.Figure(data=[go.Bar(x=names, y=market_caps, marker_color="#4f8cff")])
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e6e6e6",
        yaxis_title="Market Cap", margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def render_price_timeline(report: FullReport) -> go.Figure | None:
    metrics = report.finance.metrics if report.finance else None
    return None  # price history rendering handled via separate fetch if desired


report: FullReport | None = st.session_state.report

if report is None:
    st.markdown("## Welcome to InvestIQ AI")
    st.write(
        "Enter a company name or ticker in the sidebar and click **Analyze** to run a "
        "full multi-agent investment research pipeline: research, financials, news, "
        "sentiment, competitors, risk, valuation, SWOT, and a final investment decision."
    )
else:
    st.markdown(f"# {report.company_name} ({report.ticker})")

    if report.decision:
        badge_class = "invest-yes" if report.decision.decision.upper() == "INVEST" else "invest-no"
        st.markdown(
            f'<span class="invest-badge {badge_class}">{report.decision.decision}</span>',
            unsafe_allow_html=True,
        )

        cols = st.columns(5)
        render_metric_card(cols[0], "Overall Score", f"{report.decision.overall_score:.0f}", "/100")
        render_metric_card(cols[1], "Confidence", f"{report.decision.confidence:.0f}", "/100")
        render_metric_card(cols[2], "Financial Score", f"{report.finance.financial_score:.0f}" if report.finance else "N/A", "/100")
        render_metric_card(cols[3], "Risk Score", f"{report.risk.risk_score:.0f}" if report.risk else "N/A", "/100")
        render_metric_card(cols[4], "Valuation", report.valuation.verdict if report.valuation else "N/A")

    if report.summary:
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        st.write(report.summary.executive_summary)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown('<div class="section-header">Score Breakdown</div>', unsafe_allow_html=True)
        st.plotly_chart(render_radar_chart(report), use_container_width=True)
    with chart_col2:
        st.markdown('<div class="section-header">News Sentiment (30d)</div>', unsafe_allow_html=True)
        st.plotly_chart(render_news_pie(report), use_container_width=True)

    if report.decision:
        pros_col, cons_col = st.columns(2)
        with pros_col:
            st.markdown('<div class="section-header">✅ Pros</div>', unsafe_allow_html=True)
            for p in report.decision.pros:
                st.markdown(f"- {p}")
        with cons_col:
            st.markdown('<div class="section-header">⚠️ Cons</div>', unsafe_allow_html=True)
            for c in report.decision.cons:
                st.markdown(f"- {c}")

        st.markdown('<div class="section-header">Reasoning</div>', unsafe_allow_html=True)
        for r_ in report.decision.reasons:
            st.markdown(f"- {r_}")

    if report.research and report.research.overview:
        st.markdown('<div class="section-header">Company Overview</div>', unsafe_allow_html=True)
        o = report.research.overview
        st.write(o.summary)
        info_cols = st.columns(4)
        info_cols[0].metric("Sector", o.sector or "N/A")
        info_cols[1].metric("Industry", o.industry or "N/A")
        info_cols[2].metric("CEO", o.ceo or "N/A")
        info_cols[3].metric("HQ", o.headquarters or "N/A")

    if report.finance and report.finance.metrics:
        st.markdown('<div class="section-header">Financial Metrics</div>', unsafe_allow_html=True)
        m = report.finance.metrics
        st.dataframe(
            {
                "Metric": ["Revenue", "Net Income", "PE Ratio", "PEG Ratio", "EPS", "ROE %", "Debt/Equity", "Current Ratio", "5Y Rev CAGR %"],
                "Value": [m.revenue, m.net_income, m.pe_ratio, m.peg_ratio, m.eps, m.roe, m.debt_to_equity, m.current_ratio, m.revenue_cagr_5y],
            },
            use_container_width=True,
            hide_index=True,
        )
        st.caption(report.finance.commentary)

    if report.competitors and report.competitors.competitors:
        st.markdown('<div class="section-header">Competitor Comparison</div>', unsafe_allow_html=True)
        st.plotly_chart(render_competitor_bar(report), use_container_width=True)
        st.dataframe(
            [
                {
                    "Name": c.name, "Ticker": c.ticker, "Market Cap": c.market_cap,
                    "PE Ratio": c.pe_ratio, "Growth %": c.growth_rate,
                    "Strengths": ", ".join(c.strengths), "Weaknesses": ", ".join(c.weaknesses),
                }
                for c in report.competitors.competitors
            ],
            use_container_width=True,
            hide_index=True,
        )

    if report.risk and report.risk.factors:
        st.markdown('<div class="section-header">Risk Analysis</div>', unsafe_allow_html=True)
        st.dataframe(
            [{"Category": f.category, "Level": f.level, "Description": f.description} for f in report.risk.factors],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(report.risk.commentary)

    if report.valuation:
        st.markdown('<div class="section-header">Valuation</div>', unsafe_allow_html=True)
        v = report.valuation
        val_cols = st.columns(4)
        val_cols[0].metric("Current Price", v.current_price or "N/A")
        val_cols[1].metric("Intrinsic Value (heuristic)", v.intrinsic_value or "N/A")
        val_cols[2].metric("Margin of Safety %", v.margin_of_safety or "N/A")
        val_cols[3].metric("Verdict", v.verdict)
        st.caption(v.commentary)

    if report.swot:
        st.markdown('<div class="section-header">SWOT Analysis</div>', unsafe_allow_html=True)
        swot_cols = st.columns(4)
        swot_cols[0].markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in report.swot.strengths))
        swot_cols[1].markdown("**Weaknesses**\n" + "\n".join(f"- {x}" for x in report.swot.weaknesses))
        swot_cols[2].markdown("**Opportunities**\n" + "\n".join(f"- {x}" for x in report.swot.opportunities))
        swot_cols[3].markdown("**Threats**\n" + "\n".join(f"- {x}" for x in report.swot.threats))

    if report.news and report.news.items:
        st.markdown('<div class="section-header">Recent News</div>', unsafe_allow_html=True)
        for item in report.news.items[:10]:
            emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(item.sentiment, "⚪")
            st.markdown(f"{emoji} **[{item.title}]({item.url})** — {item.source}")
            st.caption(item.summary)

    # ---- Export section ----
    st.markdown('<div class="section-header">📥 Export Report</div>', unsafe_allow_html=True)
    exp_cols = st.columns(3)
    exp_cols[0].download_button(
        "Download Markdown", data=to_markdown(report), file_name=f"{report.ticker}_report.md", mime="text/markdown", use_container_width=True,
    )
    exp_cols[1].download_button(
        "Download JSON", data=to_json(report), file_name=f"{report.ticker}_report.json", mime="application/json", use_container_width=True,
    )
    try:
        pdf_bytes = to_pdf(report)
        exp_cols[2].download_button(
            "Download PDF", data=pdf_bytes, file_name=f"{report.ticker}_report.pdf", mime="application/pdf", use_container_width=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"PDF generation failed: {exc}")
        exp_cols[2].button("PDF unavailable", disabled=True, use_container_width=True)
