"""Tests for core Pydantic models."""

from __future__ import annotations

import pytest

from behavioral_memory.core.schemas import (
    ExecutionTrace,
    GatekeeperResult,
    ToolCall,
)


class TestToolCall:
    def test_basic_creation(self):
        tc = ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"})
        assert tc.step_id == "s1"
        assert tc.tool_name == "query_database"

    def test_empty_step_id_rejected(self):
        with pytest.raises(ValueError, match="step_id must not be empty"):
            ToolCall(step_id="  ", tool_name="query_database")

    def test_empty_tool_name_rejected(self):
        with pytest.raises(ValueError, match="tool_name must not be empty"):
            ToolCall(step_id="s1", tool_name="")

    def test_defaults(self):
        tc = ToolCall(step_id="s1", tool_name="t1")
        assert tc.parameters == {}
        assert tc.depends_on == []


class TestExecutionTrace:
    def test_basic_creation(self, sample_trace):
        assert sample_trace.task_description == "Get the total number of customers"
        assert len(sample_trace.tool_chain) == 1
        assert sample_trace.validated is True

    def test_empty_task_rejected(self):
        with pytest.raises(ValueError, match="task_description must not be empty"):
            ExecutionTrace(
                task_description="",
                tool_chain=[ToolCall(step_id="s1", tool_name="t1")],
            )

    def test_empty_chain_rejected(self):
        with pytest.raises(ValueError, match="tool_chain must contain at least one"):
            ExecutionTrace(task_description="test", tool_chain=[])

    def test_step_ids_property(self, multi_step_trace):
        assert multi_step_trace.step_ids == ["step_1", "step_2"]

    def test_tool_names_property(self, multi_step_trace):
        assert multi_step_trace.tool_names == ["query_database", "generate_report"]

    def test_to_prompt_str(self, sample_trace):
        prompt = sample_trace.to_prompt_str()
        assert "Get the total number of customers" in prompt
        assert "query_database" in prompt


class TestGatekeeperResult:
    def test_accepted(self):
        r = GatekeeperResult(accepted=True, schema_valid=True, sandbox_passed=True)
        assert r.accepted is True

    def test_rejected_with_failures(self):
        r = GatekeeperResult(
            accepted=False,
            schema_valid=False,
            rejection_reason="Bad structure",
            failures=["Unknown tool 'foo'"],
        )
        assert not r.accepted
        assert len(r.failures) == 1
