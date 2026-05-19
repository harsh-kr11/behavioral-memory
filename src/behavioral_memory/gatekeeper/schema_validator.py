"""Structural validation of execution traces.

Implements the first gate of the gatekeeper pipeline (Section III.E.1):
  - Tools exist in the registry
  - Required parameters are present
  - Step IDs are unique
  - Dependencies reference valid step IDs
  - No circular dependencies
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace, ToolCall, ToolSchema
from behavioral_memory.tools.registry import ToolRegistry


class SchemaValidator:
    """Validates trace structure against registered tool schemas."""

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def validate(self, trace: ExecutionTrace) -> tuple[bool, list[str]]:
        """Validate a trace structurally.

        Returns (is_valid, list_of_failures).
        """
        failures: list[str] = []

        failures.extend(self._check_tool_existence(trace))
        failures.extend(self._check_unique_step_ids(trace))
        failures.extend(self._check_dependencies(trace))
        failures.extend(self._check_required_params(trace))

        return len(failures) == 0, failures

    def _check_tool_existence(self, trace: ExecutionTrace) -> list[str]:
        failures: list[str] = []
        for step in trace.tool_chain:
            if not self._registry.has_tool(step.tool_name):
                failures.append(f"Unknown tool '{step.tool_name}' in step '{step.step_id}'")
        return failures

    def _check_unique_step_ids(self, trace: ExecutionTrace) -> list[str]:
        seen: set[str] = set()
        failures: list[str] = []
        for step in trace.tool_chain:
            if step.step_id in seen:
                failures.append(f"Duplicate step_id '{step.step_id}'")
            seen.add(step.step_id)
        return failures

    def _check_dependencies(self, trace: ExecutionTrace) -> list[str]:
        failures: list[str] = []
        all_ids = {step.step_id for step in trace.tool_chain}
        id_order = {step.step_id: i for i, step in enumerate(trace.tool_chain)}

        for step in trace.tool_chain:
            for dep in step.depends_on:
                if dep not in all_ids:
                    failures.append(f"Step '{step.step_id}' depends on non-existent step '{dep}'")
                elif id_order.get(dep, -1) >= id_order.get(step.step_id, 0):
                    failures.append(f"Step '{step.step_id}' depends on '{dep}' which comes after it")
        return failures

    def _check_required_params(self, trace: ExecutionTrace) -> list[str]:
        failures: list[str] = []
        for step in trace.tool_chain:
            schema = self._registry.get(step.tool_name)
            if schema is None:
                continue
            self._validate_step_params(step, schema, failures)
        return failures

    @staticmethod
    def _validate_step_params(step: ToolCall, schema: ToolSchema, failures: list[str]) -> None:
        for req_param in schema.required_params:
            if req_param not in step.parameters:
                failures.append(
                    f"Step '{step.step_id}': missing required param '{req_param}' for tool '{step.tool_name}'"
                )
