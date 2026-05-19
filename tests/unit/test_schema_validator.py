"""Tests for the gatekeeper's schema validator."""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace, ToolCall
from behavioral_memory.gatekeeper.schema_validator import SchemaValidator


class TestSchemaValidator:
    def test_valid_trace(self, benchmark_registry, sample_trace):
        validator = SchemaValidator(benchmark_registry)
        is_valid, failures = validator.validate(sample_trace)
        assert is_valid
        assert failures == []

    def test_unknown_tool_rejected(self, benchmark_registry):
        validator = SchemaValidator(benchmark_registry)
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[ToolCall(step_id="s1", tool_name="nonexistent_tool")],
        )
        is_valid, failures = validator.validate(trace)
        assert not is_valid
        assert any("Unknown tool" in f for f in failures)

    def test_duplicate_step_ids(self, benchmark_registry):
        validator = SchemaValidator(benchmark_registry)
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"}),
                ToolCall(
                    step_id="s1",
                    tool_name="generate_report",
                    parameters={"source_step": "s1", "format": "csv", "title": "t"},
                ),
            ],
        )
        is_valid, failures = validator.validate(trace)
        assert not is_valid
        assert any("Duplicate" in f for f in failures)

    def test_missing_required_params(self, benchmark_registry):
        validator = SchemaValidator(benchmark_registry)
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="query_database", parameters={}),
            ],
        )
        is_valid, failures = validator.validate(trace)
        assert not is_valid
        assert any("missing required param" in f for f in failures)

    def test_broken_dependency(self, benchmark_registry):
        validator = SchemaValidator(benchmark_registry)
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[
                ToolCall(
                    step_id="s1",
                    tool_name="query_database",
                    parameters={"query": "SELECT 1"},
                    depends_on=["nonexistent"],
                ),
            ],
        )
        is_valid, failures = validator.validate(trace)
        assert not is_valid
        assert any("non-existent step" in f for f in failures)

    def test_valid_multi_step(self, benchmark_registry, multi_step_trace):
        validator = SchemaValidator(benchmark_registry)
        is_valid, _failures = validator.validate(multi_step_trace)
        assert is_valid
