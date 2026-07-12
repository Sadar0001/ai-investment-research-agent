"""
Yahoo Finance data tool (via yfinance). Requires no API key, so this is
the primary/reliable source for price and fundamental data.
"""
from __future__ import annotations

from typing import Any, Optional

import yfinance as yf
from langchain.tools import tool

from utils.cache import cached
from utils.logger import get_logger

logger = get_logger("tools.yahoo")


@cached(ttl=1800)
def _get_ticker_info(ticker: str) -> dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return info
    except Exception as exc:  # noqa: BLE001
        logger.error(f"yfinance info fetch failed for {ticker}: {exc}")
        return {}


@cached(ttl=1800)
def _get_financials(ticker: str) -> dict[str, Any]:
    """Pull income statement, balance sheet, and cash flow summaries."""
    try:
        t = yf.Ticker(ticker)
        income = t.income_stmt
        balance = t.balance_sheet
        cashflow = t.cashflow

        def _safe_first(df, row_names: list[str]) -> Optional[float]:
            if df is None or df.empty:
                return None
            for name in row_names:
                if name in df.index:
                    series = df.loc[name].dropna()
                    if not series.empty:
                        return float(series.iloc[0])
            return None

        data = {
            "revenue": _safe_first(income, ["Total Revenue"]),
            "net_income": _safe_first(income, ["Net Income"]),
            "operating_cash_flow": _safe_first(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"]),
            "free_cash_flow": _safe_first(cashflow, ["Free Cash Flow"]),
            "total_debt": _safe_first(balance, ["Total Debt"]),
        }

        # 5-year revenue CAGR if enough history
        try:
            if income is not None and not income.empty and "Total Revenue" in income.index:
                rev_row = income.loc["Total Revenue"].dropna()
                if len(rev_row) >= 2:
                    latest = float(rev_row.iloc[0])
                    earliest = float(rev_row.iloc[-1])
                    years = len(rev_row) - 1
                    if earliest > 0 and years > 0:
                        data["revenue_cagr_5y"] = ((latest / earliest) ** (1 / years) - 1) * 100
        except Exception:  # noqa: BLE001
            data["revenue_cagr_5y"] = None

        return data
    except Exception as exc:  # noqa: BLE001
        logger.error(f"yfinance financials fetch failed for {ticker}: {exc}")
        return {}


@cached(ttl=900)
def _get_history_summary(ticker: str, period: str = "1y") -> dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period)
        if hist.empty:
            return {}
        return {
            "current_price": float(hist["Close"].iloc[-1]),
            "52w_high": float(hist["Close"].max()),
            "52w_low": float(hist["Close"].min()),
            "price_history": hist["Close"].round(2).to_dict(),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error(f"yfinance history fetch failed for {ticker}: {exc}")
        return {}


def fetch_company_overview(ticker: str) -> dict[str, Any]:
    """Fetch descriptive company overview data for the Research Agent."""
    info = _get_ticker_info(ticker)
    return {
        "company_name": info.get("longName") or info.get("shortName") or ticker,
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "ceo": _extract_ceo(info),
        "headquarters": _extract_hq(info),
        "employees": info.get("fullTimeEmployees"),
        "market_cap": info.get("marketCap"),
        "business_model": info.get("longBusinessSummary"),
        "website": info.get("website"),
    }


def _extract_ceo(info: dict[str, Any]) -> Optional[str]:
    officers = info.get("companyOfficers") or []
    for officer in officers:
        title = (officer.get("title") or "").lower()
        if "chief executive" in title or title == "ceo":
            return officer.get("name")
    return None


def _extract_hq(info: dict[str, Any]) -> Optional[str]:
    parts = [info.get("city"), info.get("state"), info.get("country")]
    parts = [p for p in parts if p]
    return ", ".join(parts) if parts else None


def fetch_financial_metrics(ticker: str) -> dict[str, Any]:
    """Fetch quantitative financial metrics for the Finance/Valuation agents."""
    info = _get_ticker_info(ticker)
    fin = _get_financials(ticker)
    hist = _get_history_summary(ticker)

    revenue = fin.get("revenue")
    net_income = fin.get("net_income")
    profit_margin = (net_income / revenue * 100) if revenue and net_income else info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None

    return {
        **fin,
        "profit_margin": profit_margin,
        "eps": info.get("trailingEps"),
        "pe_ratio": info.get("trailingPE"),
        "peg_ratio": info.get("pegRatio"),
        "roe": (info.get("returnOnEquity") or 0) * 100 if info.get("returnOnEquity") else None,
        "roce": None,  # not directly available from yfinance; computed downstream if possible
        "current_ratio": info.get("currentRatio"),
        "quick_ratio": info.get("quickRatio"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_price": hist.get("current_price") or info.get("currentPrice"),
        "52w_high": hist.get("52w_high") or info.get("fiftyTwoWeekHigh"),
        "52w_low": hist.get("52w_low") or info.get("fiftyTwoWeekLow"),
        "market_cap": info.get("marketCap"),
        "price_history": hist.get("price_history", {}),
    }


def fetch_competitor_tickers(ticker: str, sector: Optional[str] = None) -> list[str]:
    """Best-effort peer discovery. yfinance does not expose peers directly,
    so this returns an empty list; the Competitor Agent falls back to
    LLM knowledge + web search tools to identify named peers."""
    return []


@tool
def yahoo_finance_lookup(ticker: str) -> dict[str, Any]:
    """LangChain tool: given a stock ticker, return company overview,
    financial metrics, and price history from Yahoo Finance."""
    return {
        "overview": fetch_company_overview(ticker),
        "financials": fetch_financial_metrics(ticker),
    }
