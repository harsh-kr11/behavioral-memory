"""Agent state definition for the LangGraph StateGraph."""

from __future__ import annotations

from typing import Any, TypedDict

from behavioral_memory.core.schemas import ExecutionTrace, Plan, ToolSchema


class AgentState(TypedDict, total=False):
    """State that flows through the LangGraph agent nodes."""

    query: str
    retrieved_traces: list[ExecutionTrace]
    tool_schemas: list[ToolSchema]
    plan: Plan | None
    execution_results: list[dict[str, Any]]
    langfuse_trace_id: str | None
    error: str | None
