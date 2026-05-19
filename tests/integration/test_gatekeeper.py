"""Integration test: gatekeeper pipeline with schema validator + sandbox.

These tests use in-memory objects (no pgvector needed).
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace, ToolCall
from behavioral_memory.gatekeeper.sandbox import SandboxExecutor
from behavioral_memory.gatekeeper.schema_validator import SchemaValidator


class TestSchemaValidatorIntegration:
    def test_validates_complete_pipeline(self, benchmark_registry):
        trace = ExecutionTrace(
            task_description="Revenue pipeline",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"}),
                ToolCall(
                    step_id="s2",
                    tool_name="transform_data",
                    parameters={
                        "source_step": "s1",
                        "operation": "compute",
                        "params": {"new_column": "x", "expression": "1+1"},
                    },
                ),
                ToolCall(
                    step_id="s3",
                    tool_name="generate_report",
                    parameters={"source_step": "s2", "format": "markdown_table", "title": "Report"},
                ),
                ToolCall(
                    step_id="s4",
                    tool_name="send_notification",
                    parameters={
                        "channel": "email",
                        "recipient": "a@b.com",
                        "subject": "Report",
                        "body": "Done",
                        "attach_step": "s3",
                    },
                ),
                ToolCall(
                    step_id="s5",
                    tool_name="store_results",
                    parameters={"source_step": "s2", "target": "csv_file", "target_name": "out.csv"},
                ),
            ],
            source="execution",
        )
        validator = SchemaValidator(benchmark_registry)
        is_valid, failures = validator.validate(trace)
        assert is_valid, f"Failures: {failures}"


class TestSandboxIntegration:
    def test_valid_data_flow(self):
        sandbox = SandboxExecutor()
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"}),
                ToolCall(
                    step_id="s2",
                    tool_name="generate_report",
                    parameters={"source_step": "s1", "format": "csv", "title": "t"},
                ),
            ],
        )
        passed, _msg = sandbox.execute(trace)
        assert passed

    def test_invalid_source_ref(self):
        sandbox = SandboxExecutor()
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"}),
                ToolCall(
                    step_id="s2",
                    tool_name="generate_report",
                    parameters={"source_step": "nonexistent", "format": "csv", "title": "t"},
                ),
            ],
        )
        passed, msg = sandbox.execute(trace)
        assert not passed
        assert "nonexistent" in msg
