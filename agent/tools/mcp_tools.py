"""MCP tool wrappers for LangGraph's ToolNode.

These wrap the mock tools for use in the reference agent's
execution node. In production, these would dispatch to real
MCP servers.
"""

from __future__ import annotations

from typing import Any


def create_stub_executor() -> dict[str, Any]:
    """Create stub tool executors for benchmark mode.

    Returns a dict mapping tool_name -> executor function.
    Each executor returns a synthetic result.
    """
    from behavioral_memory.tools.mock_tools import get_tool_names

    executors: dict[str, Any] = {}
    for name in get_tool_names():
        executors[name] = _make_stub(name)
    return executors


def _make_stub(tool_name: str):
    """Create a stub function for a tool."""

    def stub(**kwargs: Any) -> dict[str, Any]:
        return {
            "tool": tool_name,
            "status": "success",
            "result": f"Stub execution of {tool_name}",
            "params_received": kwargs,
        }

    stub.__name__ = tool_name
    return stub
