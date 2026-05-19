"""Core data models matching the paper's terminology.

Paper: "Behavioral Memory for Tool Orchestration: Semantic Retrieval of
Validated Execution Traces in MCP-Based Agent Systems"

These models represent the structured execution traces, tool definitions,
and generated plans that flow through the three-layer architecture.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ToolCall(BaseModel):
    """A single step in an execution trace — one tool invocation.

    Maps to the paper's concept of a step within a tool chain, including
    explicit dependency tracking between steps.
    """

    step_id: str = Field(description="Unique identifier for this step within the trace")
    tool_name: str = Field(description="Name of the MCP tool to invoke")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Tool invocation parameters")
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs that must complete before this step executes",
    )

    @field_validator("step_id")
    @classmethod
    def step_id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("step_id must not be empty")
        return v.strip()

    @field_validator("tool_name")
    @classmethod
    def tool_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tool_name must not be empty")
        return v.strip()


class ToolSchema(BaseModel):
    """Schema for an MCP-compatible tool, fetched at runtime via the Tool Layer.

    Represents the information the executive layer needs to know about
    each available tool when assembling the prompt.
    """

    name: str = Field(description="Tool name as registered in MCP")
    description: str = Field(default="", description="Human-readable tool description")
    parameters_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON Schema for the tool's input parameters"
    )
    required_params: list[str] = Field(default_factory=list, description="List of required parameter names")


class ExecutionTrace(BaseModel):
    """A validated execution trace: a task description paired with its tool chain.

    This is the core unit stored in the behavioral memory. Each trace is a
    natural-language task description paired with the ordered sequence of
    tool calls that accomplish it, as described in Section III.B of the paper.
    """

    task_description: str = Field(description="Natural language description of the task")
    tool_chain: list[ToolCall] = Field(description="Ordered sequence of tool invocations")
    validated: bool = Field(default=False, description="Whether this trace passed the gatekeeper")
    source: Literal["seed", "execution", "feedback"] = Field(
        default="seed",
        description="How this trace entered memory: seeded, from execution, or from feedback",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_description")
    @classmethod
    def task_description_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("task_description must not be empty")
        return v.strip()

    @field_validator("tool_chain")
    @classmethod
    def tool_chain_not_empty(cls, v: list[ToolCall]) -> list[ToolCall]:
        if not v:
            raise ValueError("tool_chain must contain at least one step")
        return v

    @property
    def step_ids(self) -> list[str]:
        return [step.step_id for step in self.tool_chain]

    @property
    def tool_names(self) -> list[str]:
        return [step.tool_name for step in self.tool_chain]

    def to_prompt_str(self) -> str:
        """Serialize this trace into a string suitable for prompt injection."""
        chain_str = json.dumps([step.model_dump() for step in self.tool_chain], indent=2)
        return f"Task: {self.task_description}\nTool Chain:\n{chain_str}"


class Plan(BaseModel):
    """A generated execution plan — the output of the Executive Layer.

    Combines the user query, the LLM-generated steps, and the context
    that was used to produce them (retrieved traces and tool schemas).
    """

    query: str = Field(description="Original user query")
    steps: list[ToolCall] = Field(description="Ordered tool invocations in the plan")
    retrieved_traces: list[ExecutionTrace] = Field(
        default_factory=list, description="Traces retrieved from behavioral memory"
    )
    schemas_used: list[ToolSchema] = Field(default_factory=list, description="Tool schemas available during planning")
    token_budget_used: int = Field(default=0, description="Tokens consumed by the prompt")
    raw_llm_output: str = Field(default="", description="Raw LLM response before parsing")


class GatekeeperResult(BaseModel):
    """Outcome of running a trace through the gatekeeper pipeline.

    Records which checks passed and which failed, enabling debugging
    and the ablation study described in Section IV.D.5 of the paper.
    """

    accepted: bool = Field(description="Whether the trace was accepted into memory")
    schema_valid: bool = Field(default=False, description="Passed structural validation")
    sandbox_passed: bool = Field(default=False, description="Passed sandboxed execution")
    is_duplicate: bool = Field(default=False, description="Flagged as semantic duplicate")
    rejection_reason: str = Field(default="", description="Human-readable rejection reason")
    failures: list[str] = Field(default_factory=list, description="List of specific validation failures")
