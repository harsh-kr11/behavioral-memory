"""Tests for prompt assembly."""

from __future__ import annotations

from behavioral_memory.core.schemas import ToolSchema
from behavioral_memory.planner.prompt import SYSTEM_PROMPT, build_prompt


class TestBuildPrompt:
    def test_includes_query(self):
        prompt = build_prompt(query="Find customers", traces=[], tool_schemas=[])
        assert "Find customers" in prompt

    def test_includes_traces_section(self, sample_trace):
        prompt = build_prompt(query="test", traces=[sample_trace])
        assert "REFERENCE EXAMPLES" in prompt
        assert sample_trace.task_description in prompt

    def test_no_traces_section_when_empty(self):
        prompt = build_prompt(query="test", traces=[])
        assert "REFERENCE EXAMPLES" not in prompt

    def test_includes_tools_section(self):
        schema = ToolSchema(
            name="test_tool",
            description="A test",
            parameters_schema={"properties": {"x": {"type": "string"}}},
            required_params=["x"],
        )
        prompt = build_prompt(query="test", traces=[], tool_schemas=[schema])
        assert "test_tool" in prompt
        assert "AVAILABLE TOOLS" in prompt

    def test_system_prompt_exists(self):
        assert "JSON array" in SYSTEM_PROMPT
        assert "step_id" in SYSTEM_PROMPT
