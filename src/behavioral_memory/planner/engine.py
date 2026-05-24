"""Executive Layer — PlanEngine.

Orchestrates the three layers to produce a structured execution plan:
  1. Behavioral Layer: retrieve relevant traces from memory
  2. Tool Layer: gather available tool schemas
  3. Executive: assemble prompt, call LLM, parse response

This is the primary interface for external agents integrating the framework.
The engine is model-agnostic — it accepts any LangChain BaseChatModel.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from behavioral_memory.core.config import Settings
from behavioral_memory.core.exceptions import PlanGenerationError
from behavioral_memory.core.schemas import ExecutionTrace, Plan, ToolSchema
from behavioral_memory.memory.token_budget import select_traces_within_budget
from behavioral_memory.planner.postprocess import postprocess_plan
from behavioral_memory.planner.prompt import SYSTEM_PROMPT, build_prompt
from behavioral_memory.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def _extract_text(response: Any) -> str:
    """Extract plain text from an LLM response.

    Handles both plain-string content (older providers) and list-of-blocks
    content (e.g. langchain-google-genai v4+ returns
    ``[{"type": "text", "text": "..."}]``).
    """
    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        if parts:
            return "\n".join(parts)
    return str(content)


class PlanEngine:
    """Core planning engine — the heart of the executive layer.

    Accepts any LangChain-compatible chat model, making the framework
    entirely model-agnostic.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        store: Any,
        registry: ToolRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._registry = registry or ToolRegistry()
        self._settings = settings or Settings()

    def generate(
        self,
        query: str,
        tool_schemas: list[ToolSchema] | None = None,
        traces: list[ExecutionTrace] | None = None,
    ) -> Plan:
        """Generate an execution plan for the given query.

        If tool_schemas are provided, they're used directly. Otherwise
        the engine uses whatever is in the registry.
        If traces are provided, they're used directly. Otherwise
        the engine retrieves from the store.
        """
        schemas = tool_schemas or self._registry.list_tools()
        if not schemas:
            raise PlanGenerationError("No tool schemas available for planning")

        if traces is None:
            traces = select_traces_within_budget(
                store=self._store,
                query=query,
                tool_schemas=schemas,
                settings=self._settings,
            )

        prompt_content = build_prompt(
            query=query,
            traces=traces,
            tool_schemas=schemas,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_content),
        ]

        try:
            response = self._llm.invoke(messages)
            raw_output = _extract_text(response)
        except Exception as e:
            raise PlanGenerationError(f"LLM invocation failed: {e}") from e

        steps = postprocess_plan(raw_output)

        from behavioral_memory.memory.token_budget import count_tokens

        token_budget = count_tokens(SYSTEM_PROMPT) + count_tokens(prompt_content)

        return Plan(
            query=query,
            steps=steps,
            retrieved_traces=traces,
            schemas_used=schemas,
            token_budget_used=token_budget,
            raw_llm_output=raw_output,
        )

    def generate_zero_shot(self, query: str, tool_schemas: list[ToolSchema]) -> Plan:
        """Generate a plan with no trace retrieval (baseline)."""
        return self.generate(query=query, tool_schemas=tool_schemas, traces=[])

    def generate_static_few_shot(
        self,
        query: str,
        tool_schemas: list[ToolSchema],
        static_traces: list[ExecutionTrace],
    ) -> Plan:
        """Generate a plan with fixed static examples (baseline)."""
        return self.generate(query=query, tool_schemas=tool_schemas, traces=static_traces)
