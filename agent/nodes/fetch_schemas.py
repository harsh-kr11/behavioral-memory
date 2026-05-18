"""Node: fetch_schemas — loads tool schemas from the registry."""

from __future__ import annotations

from agent.state import AgentState
from behavioral_memory.tools.registry import ToolRegistry


def make_fetch_schemas_node(registry: ToolRegistry):
    """Factory that creates a fetch_schemas node bound to a ToolRegistry."""

    def fetch_schemas(state: AgentState) -> dict:
        schemas = registry.list_tools()
        return {"tool_schemas": schemas}

    return fetch_schemas
