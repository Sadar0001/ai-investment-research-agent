"""
BaseAgent: shared scaffolding every specialized agent inherits.

Provides:
  - its own LangChain LLMChain (prompt + LLM)
  - its own async execution wrapper with timeout
  - its own retry policy
  - its own structured-output parsing/validation via Pydantic
  - its own logging and graceful-failure handling

Each subclass only needs to implement `build_user_prompt` (how to turn
raw inputs into the human message) and `parse_output` (how to map the
LLM's JSON into its specific Pydantic output model).
"""
from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from chains.llm_factory import get_llm
from models.schemas import AgentStatus, BaseAgentOutput
from utils.logger import get_logger

OutputT = TypeVar("OutputT", bound=BaseAgentOutput)


class AgentTimeoutError(RuntimeError):
    """Raised when an agent exceeds its configured execution timeout."""


class BaseAgent(ABC, Generic[OutputT]):
    """Abstract base class for all InvestIQ agents."""

    name: str = "base_agent"
    system_prompt: str = ""
    timeout_seconds: float = 45.0
    temperature: float = 0.2
    max_retries: int = 2

    def __init__(self) -> None:
        self.logger = get_logger(f"agent.{self.name}")
        self.llm = get_llm(temperature=self.temperature)

    # -- Hooks subclasses must implement -----------------------------
    @abstractmethod
    def build_user_prompt(self, context: dict[str, Any]) -> str:
        """Turn raw tool/context data into the human message content."""

    @abstractmethod
    def parse_output(self, raw_json: dict[str, Any], context: dict[str, Any]) -> OutputT:
        """Map validated JSON into this agent's specific output model."""

    @abstractmethod
    def empty_output(self, error_message: str) -> OutputT:
        """Return a safe fallback output when the agent fails entirely."""

    # -- Shared execution machinery -----------------------------------
    async def run(self, context: dict[str, Any]) -> OutputT:
        """Execute this agent end-to-end with timeout + retry + fallback."""
        try:
            return await asyncio.wait_for(self._run_with_retry(context), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            msg = f"{self.name} timed out after {self.timeout_seconds}s"
            self.logger.error(msg)
            return self.empty_output(msg)
        except Exception as exc:  # noqa: BLE001
            msg = f"{self.name} failed: {exc}"
            self.logger.exception(msg)
            return self.empty_output(msg)

    async def _run_with_retry(self, context: dict[str, Any]) -> OutputT:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                return await self._execute_once(context)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self.logger.warning(f"{self.name} attempt {attempt} failed: {exc}")
                if attempt <= self.max_retries:
                    await asyncio.sleep(min(2 ** attempt, 8))
        assert last_exc is not None
        raise last_exc

    async def _execute_once(self, context: dict[str, Any]) -> OutputT:
        user_prompt = self.build_user_prompt(context)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        self.logger.info(f"{self.name} invoking LLM")
        response = await self.llm.ainvoke(messages)
        raw_json = self._safe_json_parse(response.content)
        output = self.parse_output(raw_json, context)
        output.status = AgentStatus.SUCCESS
        self.logger.info(f"{self.name} completed successfully")
        return output

    def _safe_json_parse(self, content: Any) -> dict[str, Any]:
        """Parse the LLM's response as JSON, tolerating stray code fences."""
        text = content if isinstance(content, str) else str(content)
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Attempt to salvage a JSON object from within surrounding text
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass
            self.logger.warning(f"{self.name} could not parse JSON output; returning empty dict")
            return {}
