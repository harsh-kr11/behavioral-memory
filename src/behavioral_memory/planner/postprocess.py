"""Post-processing for LLM-generated execution plans.

Extracts and validates the JSON tool chain from raw LLM output,
handling common formatting quirks (markdown fences, trailing commas, etc.).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from behavioral_memory.core.exceptions import PlanGenerationError
from behavioral_memory.core.schemas import ToolCall

logger = logging.getLogger(__name__)


def extract_json_array(raw: str) -> list[dict[str, Any]]:
    """Extract a JSON array from raw LLM output."""
    text = raw.strip()

    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
    match = fence_pattern.search(text)
    if match:
        text = match.group(1).strip()

    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        text = bracket_match.group(0)

    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanGenerationError(f"Failed to parse LLM output as JSON: {e}") from e

    if not isinstance(parsed, list):
        raise PlanGenerationError(f"Expected JSON array, got {type(parsed).__name__}")

    return parsed


def parse_tool_calls(raw_steps: list[dict[str, Any]]) -> list[ToolCall]:
    """Convert raw JSON steps into validated ToolCall models."""
    calls: list[ToolCall] = []
    for i, step in enumerate(raw_steps):
        try:
            call = ToolCall(
                step_id=step.get("step_id", f"step_{i + 1}"),
                tool_name=step.get("tool_name", step.get("tool", "")),
                parameters=step.get("parameters", step.get("params", {})),
                depends_on=step.get("depends_on", []),
            )
            calls.append(call)
        except Exception as e:
            logger.warning("Skipping malformed step %d: %s", i, e)
    return calls


def postprocess_plan(raw_output: str) -> list[ToolCall]:
    """Full post-processing pipeline: extract JSON, parse into ToolCalls."""
    raw_steps = extract_json_array(raw_output)
    return parse_tool_calls(raw_steps)
