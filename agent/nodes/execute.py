"""Node: execute_tools — runs the planned tool chain (stub for benchmarks)."""

from __future__ import annotations

from typing import Any

from agent.state import AgentState


def execute_tools(state: AgentState) -> dict:
    """Execute the tool chain from the plan.

    In benchmark mode, this is a stub that records what would be executed.
    In production, this would dispatch to real MCP tool servers.
    """
    plan = state.get("plan")
    if plan is None:
        return {"execution_results": []}

    results: list[dict[str, Any]] = []
    for step in plan.steps:
        results.append(
            {
                "step_id": step.step_id,
                "tool_name": step.tool_name,
                "status": "executed_stub",
                "parameters": step.parameters,
            }
        )

    return {"execution_results": results}
