"""
Centralized application configuration.

All environment-driven settings live here so that no other module
reads `os.environ` directly. This keeps configuration auditable and
makes testing (via monkeypatched Settings) straightforward.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """Strongly typed application settings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")

    # LangSmith (optional observability)
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="InvestIQ", alias="LANGCHAIN_PROJECT")

    # Data source keys (all optional -> tools degrade gracefully)
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    finnhub_api_key: str = Field(default="", alias="FINNHUB_API_KEY")
    alpha_vantage_api_key: str = Field(default="", alias="ALPHA_VANTAGE_API_KEY")
    serpapi_api_key: str = Field(default="", alias="SERPAPI_API_KEY")

    # App behavior
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cache_ttl_seconds: int = Field(default=3600, alias="CACHE_TTL_SECONDS")
    request_timeout_seconds: int = Field(default=20, alias="REQUEST_TIMEOUT_SECONDS")

    # Paths
    base_dir: Path = BASE_DIR
    logs_dir: Path = BASE_DIR / "logs"
    cache_dir: Path = BASE_DIR / "cache"
    reports_dir: Path = BASE_DIR / "reports"

    def has_key(self, name: str) -> bool:
        """Check whether a named API key is configured (non-empty)."""
        return bool(getattr(self, name, "") or "")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    settings = Settings()
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    if settings.langchain_tracing_v2 and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    return settings
