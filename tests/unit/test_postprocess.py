"""Tests for LLM output post-processing."""

from __future__ import annotations

import pytest

from behavioral_memory.core.exceptions import PlanGenerationError
from behavioral_memory.planner.postprocess import (
    extract_json_array,
    parse_tool_calls,
    postprocess_plan,
)


class TestExtractJsonArray:
    def test_plain_json(self):
        result = extract_json_array('[{"step_id": "s1", "tool_name": "t1"}]')
        assert len(result) == 1

    def test_markdown_fenced(self):
        raw = '```json\n[{"step_id": "s1", "tool_name": "t1"}]\n```'
        result = extract_json_array(raw)
        assert len(result) == 1

    def test_trailing_comma_handled(self):
        raw = '[{"step_id": "s1", "tool_name": "t1",}]'
        result = extract_json_array(raw)
        assert len(result) == 1

    def test_invalid_json_raises(self):
        with pytest.raises(PlanGenerationError):
            extract_json_array("not json at all")

    def test_non_array_raises(self):
        with pytest.raises(PlanGenerationError, match="Expected JSON array"):
            extract_json_array('{"key": "value"}')


class TestParseToolCalls:
    def test_standard_format(self):
        raw = [{"step_id": "s1", "tool_name": "query_database", "parameters": {"query": "SELECT 1"}}]
        calls = parse_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].tool_name == "query_database"

    def test_alternative_keys(self):
        raw = [{"step_id": "s1", "tool": "query_database", "params": {"query": "SELECT 1"}}]
        calls = parse_tool_calls(raw)
        assert calls[0].tool_name == "query_database"
        assert "query" in calls[0].parameters


class TestPostprocessPlan:
    def test_end_to_end(self):
        raw = '```json\n[{"step_id": "s1", "tool_name": "query_database", "parameters": {"query": "SELECT 1"}, "depends_on": []}]\n```'
        calls = postprocess_plan(raw)
        assert len(calls) == 1
        assert calls[0].step_id == "s1"
