"""Three-layer prompt assembly for the executive layer.

Implements the prompt structure from Section III.D of the paper:
  Layer 1 (Behavioral):  Retrieved execution traces as reference examples
  Layer 2 (Tool):        Available MCP tool schemas
  Layer 3 (Executive):   User query + generation instructions
"""

from __future__ import annotations

from behavioral_memory.core.schemas import ExecutionTrace, ToolSchema
from behavioral_memory.tools.registry import ToolRegistry

SYSTEM_PROMPT = """\
You are an expert tool orchestration agent. Your job is to produce a structured
execution plan — an ordered JSON array of tool invocations — that accomplishes
the user's task.

RULES:
1. Output ONLY a valid JSON array. No explanation, no markdown fences.
2. Each element must have: step_id, tool_name, parameters, depends_on.
3. step_id must be unique (e.g. "step_1", "step_2", ...).
4. depends_on lists step_ids that must complete before this step runs.
5. Only use tools listed in the AVAILABLE TOOLS section.
6. Use REFERENCE EXAMPLES to learn domain conventions — do NOT copy verbatim.
7. If a task requires data from a previous step, reference its step_id in parameters.
"""


def _format_traces_section(traces: list[ExecutionTrace]) -> str:
    if not traces:
        return ""
    lines = [
        "── REFERENCE EXAMPLES ──────────────────────────────────────",
        "Below are validated task→tool-chain pairs. Use them to learn",
        "domain conventions, tool sequencing, and parameter patterns.",
        "",
    ]
    for i, trace in enumerate(traces, 1):
        lines.append(f"Example {i}:")
        lines.append(trace.to_prompt_str())
        lines.append("")
    return "\n".join(lines)


def _format_tools_section(
    schemas: list[ToolSchema] | None = None,
    registry: ToolRegistry | None = None,
) -> str:
    lines = ["── AVAILABLE TOOLS ─────────────────────────────────────────"]
    if registry:
        lines.append(registry.format_for_prompt())
    elif schemas:
        temp_reg = ToolRegistry()
        temp_reg.register_many(schemas)
        lines.append(temp_reg.format_for_prompt())
    return "\n".join(lines)


def _format_query_section(query: str) -> str:
    return (
        "── USER TASK ───────────────────────────────────────────────\n"
        f"{query}\n\n"
        "Generate the execution plan as a JSON array now."
    )


def build_prompt(
    query: str,
    traces: list[ExecutionTrace],
    tool_schemas: list[ToolSchema] | None = None,
    registry: ToolRegistry | None = None,
) -> str:
    """Assemble the full three-layer prompt.

    Returns the combined user message content (system prompt is separate).
    """
    sections = [
        _format_traces_section(traces),
        _format_tools_section(schemas=tool_schemas, registry=registry),
        _format_query_section(query),
    ]
    return "\n\n".join(s for s in sections if s)
