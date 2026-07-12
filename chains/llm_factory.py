"""
Factory for LangChain ChatOpenAI instances. Centralizing construction
means every agent shares consistent model/temperature/timeout config
and makes it trivial to swap providers later.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from config.settings import get_settings


@lru_cache
def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a cached ChatOpenAI client. Pass a temperature override
    for agents that need more/less deterministic output."""
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.openai_api_key or "sk-placeholder",
        model=settings.llm_model,
        temperature=temperature if temperature is not None else settings.llm_temperature,
        timeout=settings.request_timeout_seconds,
        max_retries=2,
    )
