"""Evaluation strategies: zero-shot, static few-shot, dynamic retrieval.

These correspond to the three conditions compared in the paper (Section IV.B).
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace, Plan, ToolSchema
from behavioral_memory.planner.engine import PlanEngine


class ZeroShotStrategy:
    """Baseline: model receives only the task and tool schemas, no examples."""

    def __init__(self, engine: PlanEngine) -> None:
        self._engine = engine

    def generate(self, query: str, tool_schemas: list[ToolSchema]) -> Plan:
        return self._engine.generate_zero_shot(query, tool_schemas)


class StaticFewShotStrategy:
    """Baseline: a fixed set of 3 examples is included for all tasks."""

    def __init__(
        self, engine: PlanEngine, static_traces: list[ExecutionTrace]
    ) -> None:
        self._engine = engine
        self._static = static_traces[:3]

    def generate(self, query: str, tool_schemas: list[ToolSchema]) -> Plan:
        return self._engine.generate_static_few_shot(
            query, tool_schemas, self._static
        )


class DynamicRetrievalStrategy:
    """Proposed approach: retrieve semantically similar traces dynamically."""

    def __init__(self, engine: PlanEngine) -> None:
        self._engine = engine

    def generate(self, query: str, tool_schemas: list[ToolSchema]) -> Plan:
        return self._engine.generate(query=query, tool_schemas=tool_schemas)
