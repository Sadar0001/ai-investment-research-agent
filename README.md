# InvestIQ AI 📊

**Professional AI Investment Research Platform** — a multi-agent LangChain system that researches a public company across ten specialized dimensions and synthesizes a final Invest / Do Not Invest recommendation.

![Python](https://img.shields.io/badge/python-3.12-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> ⚠️ **Educational tool only.** InvestIQ AI generates AI-assisted research summaries, not licensed financial advice. Always do your own due diligence.

---

## Overview

Enter a company name or ticker. Ten independent AI agents research it in parallel and in sequence, then a Decision Agent synthesizes everything into a single investment recommendation with a transparent score breakdown.

Built entirely on **LangChain + Streamlit** — no LangGraph, CrewAI, AutoGen, or agent frameworks. Orchestration (parallel execution, dependency ordering, error isolation) is implemented by hand in `agents/supervisor.py`.

## Architecture

```
                     ┌─────────────────┐
                     │   Streamlit UI  │
                     │     app.py      │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │   Supervisor    │  manual orchestration
                     └────────┬────────┘
          ┌──────────┬────────┼────────┬──────────┬──────────┐
          ▼          ▼        ▼        ▼          ▼          ▼
      Research   Finance    News   Competitor   Risk    Valuation      Stage 1 (parallel)
          │          │        │        │          │          │
          └──────────┴────────┼────────┴──────────┴──────────┘
                              ▼
                          Sentiment                                    Stage 2 (needs News)
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                  SWOT  →  Summary  →  Decision                        Stage 3 (sequential synthesis)
```

Each agent (`agents/*.py`) is a self-contained `BaseAgent` subclass with:
- its own system prompt (`prompts/agent_prompts.py`)
- its own tool calls (`tools/*.py`)
- its own Pydantic structured output (`models/schemas.py`)
- its own async execution, timeout, retry-with-backoff, and safe JSON parsing
- its own graceful fallback output if the LLM or upstream data fails

## Features

- **10 specialized agents**: Research, Finance, News, Sentiment, Competitor, Risk, Valuation, SWOT, Summary, Decision
- **Manual async orchestration** — dependency-aware staged execution via `asyncio.gather`, no agent framework
- **Multi-source data**: Yahoo Finance (no key required), Finnhub, Alpha Vantage, NewsAPI, Tavily, SerpAPI — every tool degrades gracefully when a key is absent
- **Resilience**: every agent has its own timeout, retry/backoff, structured-output validation, and non-crashing fallback
- **Disk caching** (DiskCache) to avoid duplicate API calls
- **Structured logging** (Loguru) to console + rotating log files
- **Professional dark-themed Streamlit dashboard**: score cards, radar chart, sentiment pie chart, competitor bar chart, data tables, live agent progress
- **Report export**: Markdown, JSON, and PDF, generated from a single shared renderer
- **Unit tests** for tool fallback behavior, agent parsing/scoring, and report rendering

## Data Sources

| Source | Purpose | Key required |
|---|---|---|
| Yahoo Finance (`yfinance`) | Company overview, fundamentals, price history | No |
| Finnhub | Peer list, analyst recommendations, aggregate sentiment | Yes (optional) |
| Alpha Vantage | Supplemental ratios (PE, PEG, ROE, beta) | Yes (optional) |
| NewsAPI | Last-30-days news articles | Yes (optional) |
| Tavily | General web search / fallback research | Yes (optional) |
| SerpAPI | Google search fallback | Yes (optional) |

All tools are implemented with `tenacity`-based retry and return safe empty defaults instead of raising when a key is missing or a request fails — the pipeline never crashes due to a missing integration.

## Folder Structure

```
InvestIQ-AI/
├── app.py                  # Streamlit entrypoint
├── requirements.txt
├── .env.example
├── config/                 # Settings (pydantic-settings)
├── agents/                 # Supervisor + 10 specialized agents + BaseAgent
├── prompts/                # System prompts per agent
├── tools/                  # yahoo, tavily, serpapi, news, finnhub, alpha, scraper
├── models/                 # Pydantic schemas (structured agent I/O)
├── services/                # Ticker resolution, report export (md/json/pdf)
├── chains/                 # LLM factory (ChatOpenAI construction)
├── utils/                  # Logger, disk cache, retry policy
├── tests/                  # Pytest unit tests
├── logs/, cache/, reports/ # Runtime artifacts (gitignored contents)
└── assets/
```

## Installation

```bash
git clone https://github.com/<your-username>/InvestIQ-AI.git
cd InvestIQ-AI

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# then edit .env and add at minimum OPENAI_API_KEY

streamlit run app.py
```

## Environment Variables

See `.env.example`. Only `OPENAI_API_KEY` is required to run the LLM agents; all data-source keys are optional and each tool degrades gracefully without one.

## Running Tests

```bash
pytest tests/ -v
```

## Deployment

- **Streamlit Community Cloud**: point it at `app.py`, add secrets matching `.env.example` under app settings.
- **Docker**: build a standard Python 3.12-slim image, `pip install -r requirements.txt`, `CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]`.
- **GCP Cloud Run / AWS App Runner**: containerize as above; mount environment variables as secrets rather than baking them into the image.

## Future Improvements

- Add a discounted cash flow (DCF) model to the Valuation Agent alongside the current heuristic
- Add a vector-store-backed memory layer for multi-turn follow-up questions per company
- Add historical backtesting of past InvestIQ recommendations against actual price performance
- Add SEC EDGAR filings ingestion for US equities

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

Built with [LangChain](https://www.langchain.com/), [Streamlit](https://streamlit.io/), [yfinance](https://github.com/ranaroussi/yfinance), and Plotly.
