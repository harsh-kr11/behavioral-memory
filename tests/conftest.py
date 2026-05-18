"""Shared fixtures for behavioral-memory tests."""

from __future__ import annotations

import pytest

from behavioral_memory.core.schemas import ExecutionTrace, ToolCall, ToolSchema
from behavioral_memory.tools.mock_tools import get_tool_schemas
from behavioral_memory.tools.registry import ToolRegistry


@pytest.fixture
def sample_tool_call() -> ToolCall:
    return ToolCall(
        step_id="step_1",
        tool_name="query_database",
        parameters={"query": "SELECT COUNT(*) FROM customers;"},
    )


@pytest.fixture
def sample_trace() -> ExecutionTrace:
    return ExecutionTrace(
        task_description="Get the total number of customers",
        tool_chain=[
            ToolCall(
                step_id="step_1",
                tool_name="query_database",
                parameters={"query": "SELECT COUNT(*) FROM customers;"},
            ),
        ],
        validated=True,
        source="seed",
    )


@pytest.fixture
def multi_step_trace() -> ExecutionTrace:
    return ExecutionTrace(
        task_description="Get revenue and generate a report",
        tool_chain=[
            ToolCall(
                step_id="step_1",
                tool_name="query_database",
                parameters={"query": "SELECT SUM(quantity * unit_price) FROM order_items;"},
            ),
            ToolCall(
                step_id="step_2",
                tool_name="generate_report",
                parameters={"source_step": "step_1", "format": "markdown_table", "title": "Revenue"},
            ),
        ],
        validated=True,
        source="seed",
    )


@pytest.fixture
def benchmark_schemas() -> list[ToolSchema]:
    return get_tool_schemas()


@pytest.fixture
def benchmark_registry(benchmark_schemas: list[ToolSchema]) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_many(benchmark_schemas)
    return reg
