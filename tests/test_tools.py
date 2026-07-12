"""Unit tests for data-source tools. External APIs are mocked so tests
run offline and deterministically."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from tools import alpha_tool, finnhub_tool, news_tool, tavily_tool, yahoo_tool


class TestYahooTool:
    @patch("tools.yahoo_tool.yf.Ticker")
    def test_fetch_company_overview_maps_fields(self, mock_ticker):
        mock_instance = MagicMock()
        mock_instance.info = {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "city": "Cupertino",
            "state": "CA",
            "country": "United States",
            "fullTimeEmployees": 161000,
            "marketCap": 3_000_000_000_000,
            "longBusinessSummary": "Apple designs consumer electronics.",
            "companyOfficers": [{"title": "Chief Executive Officer", "name": "Tim Cook"}],
        }
        mock_ticker.return_value = mock_instance

        yahoo_tool._get_ticker_info.__wrapped__.cache_clear = lambda: None  # no-op guard
        overview = yahoo_tool.fetch_company_overview("AAPL")

        assert overview["company_name"] == "Apple Inc."
        assert overview["sector"] == "Technology"
        assert overview["ceo"] == "Tim Cook"
        assert overview["headquarters"] == "Cupertino, CA, United States"

    def test_extract_ceo_returns_none_when_missing(self):
        assert yahoo_tool._extract_ceo({}) is None

    def test_extract_hq_handles_partial_data(self):
        assert yahoo_tool._extract_hq({"country": "India"}) == "India"


class TestTavilyTool:
    @patch("tools.tavily_tool.get_settings")
    def test_returns_empty_without_api_key(self, mock_settings):
        mock_settings.return_value.has_key.return_value = False
        result = tavily_tool._call_tavily("Apple Inc overview", 5)
        assert result == {"results": []}


class TestNewsTool:
    @patch("tools.news_tool.get_settings")
    def test_returns_empty_without_api_key(self, mock_settings):
        mock_settings.return_value.has_key.return_value = False
        result = news_tool._call_newsapi("Apple", "2026-06-01")
        assert result == {"articles": []}


class TestFinnhubTool:
    @patch("tools.finnhub_tool.get_settings")
    def test_returns_none_without_api_key(self, mock_settings):
        mock_settings.return_value.has_key.return_value = False
        result = finnhub_tool._get("stock/peers", {"symbol": "AAPL"})
        assert result is None


class TestAlphaVantageTool:
    @patch("tools.alpha_tool.get_settings")
    def test_returns_empty_without_api_key(self, mock_settings):
        mock_settings.return_value.has_key.return_value = False
        result = alpha_tool._call_alpha("OVERVIEW", "AAPL")
        assert result == {}

    def test_safe_float_handles_invalid_values(self):
        assert alpha_tool._safe_float("None") is None
        assert alpha_tool._safe_float("-") is None
        assert alpha_tool._safe_float("12.5") == 12.5
