"""MCP client — dynamic tool schema fetcher.

Connects to an MCP server at runtime to discover available tools and
their JSON schemas, implementing the Tool Layer described in Section III.C
of the paper.
"""

from __future__ import annotations

import logging
from typing import Any

from behavioral_memory.core.exceptions import MCPConnectionError
from behavioral_memory.core.schemas import ToolSchema
from behavioral_memory.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


async def fetch_mcp_schemas(server_url: str) -> list[ToolSchema]:
    """Fetch tool schemas from a live MCP server.

    Uses the MCP SDK to connect via stdio or SSE transport and
    enumerate available tools.
    """
    try:
        from mcp import ClientSession
        from mcp.client.sse import sse_client
    except ImportError as e:
        raise MCPConnectionError(
            "MCP SDK not installed. Install with: pip install mcp"
        ) from e

    schemas: list[ToolSchema] = []
    try:
        async with sse_client(server_url) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                input_schema: dict[str, Any] = {}
                if hasattr(tool, "inputSchema") and tool.inputSchema:
                    input_schema = dict(tool.inputSchema)
                elif hasattr(tool, "input_schema") and tool.input_schema:
                    input_schema = dict(tool.input_schema)

                required = input_schema.get("required", [])
                schemas.append(
                    ToolSchema(
                        name=tool.name,
                        description=tool.description or "",
                        parameters_schema=input_schema,
                        required_params=required,
                    )
                )
            logger.info("Fetched %d tool schemas from %s", len(schemas), server_url)
    except Exception as e:
        raise MCPConnectionError(f"Failed to fetch schemas from {server_url}: {e}") from e

    return schemas


def load_schemas_into_registry(
    schemas: list[ToolSchema], registry: ToolRegistry | None = None
) -> ToolRegistry:
    """Load a list of ToolSchemas into a ToolRegistry."""
    reg = registry or ToolRegistry()
    reg.register_many(schemas)
    return reg
