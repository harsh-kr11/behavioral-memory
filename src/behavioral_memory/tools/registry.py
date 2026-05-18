"""Tool registry — runtime cache for MCP tool schemas.

Maintains a lookup of available tools and their schemas, populated
either from live MCP servers or from static definitions (for benchmarks).
"""

from __future__ import annotations

import logging

from behavioral_memory.core.schemas import ToolSchema

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Registry of available tools, used by the executive layer for prompt assembly."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSchema] = {}

    def register(self, schema: ToolSchema) -> None:
        self._tools[schema.name] = schema
        logger.debug("Registered tool: %s", schema.name)

    def register_many(self, schemas: list[ToolSchema]) -> None:
        for s in schemas:
            self.register(s)

    def get(self, name: str) -> ToolSchema | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        return list(self._tools.values())

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def clear(self) -> None:
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def format_for_prompt(self) -> str:
        """Format all registered tools as a structured string for LLM prompts."""
        lines: list[str] = []
        for tool in self._tools.values():
            lines.append(f"Tool: {tool.name}")
            lines.append(f"  Description: {tool.description}")
            props = tool.parameters_schema.get("properties", {})
            required = tool.required_params
            if props:
                lines.append("  Parameters:")
                for pname, pdef in props.items():
                    req = " (REQUIRED)" if pname in required else ""
                    ptype = pdef.get("type", "any")
                    pdesc = pdef.get("description", "")
                    enum_vals = pdef.get("enum")
                    enum_str = f" [{', '.join(str(e) for e in enum_vals)}]" if enum_vals else ""
                    lines.append(f"    - {pname}: {ptype}{enum_str}{req} — {pdesc}")
            lines.append("")
        return "\n".join(lines)
