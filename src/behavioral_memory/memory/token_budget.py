"""Token-aware trace selection.

Ensures that retrieved traces + tool schemas fit within the prompt's
token budget (default 3500 tokens per the paper, Section III.B).
Traces are greedily packed in order of descending similarity score.
"""

from __future__ import annotations

import tiktoken

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace, ToolSchema
from behavioral_memory.memory.store import TraceStore

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using the cl100k_base encoding (proxy for most LLMs)."""
    return len(_ENCODER.encode(text))


def estimate_trace_tokens(trace: ExecutionTrace) -> int:
    """Estimate token cost of including a trace in the prompt."""
    return count_tokens(trace.to_prompt_str())


def _schema_tokens(schemas: list[ToolSchema]) -> int:
    total = 0
    for s in schemas:
        text = f"Tool: {s.name}\n{s.description}\nParameters: {s.parameters_schema}"
        total += count_tokens(text)
    return total


def select_traces_within_budget(
    store: TraceStore,
    query: str,
    tool_schemas: list[ToolSchema],
    *,
    max_tokens: int | None = None,
    base_prompt_tokens: int = 500,
    settings: Settings | None = None,
) -> list[ExecutionTrace]:
    """Select as many traces as fit within the token budget.

    Priority:
      1. Tool schemas (non-negotiable — the planner needs schema context)
      2. Retrieved traces (highest similarity first, greedy packing)

    Returns traces ordered by similarity (most similar first).
    """
    _settings = settings or Settings()
    budget = max_tokens or _settings.max_prompt_tokens
    used = base_prompt_tokens + _schema_tokens(tool_schemas) + count_tokens(query)
    remaining = budget - used

    if remaining <= 0:
        return []

    k_candidates = _settings.few_shot_k * 2
    candidates = store.search(query, k=k_candidates)

    selected: list[ExecutionTrace] = []
    for trace, _score in candidates:
        cost = estimate_trace_tokens(trace)
        if cost > remaining:
            break
        selected.append(trace)
        remaining -= cost
        if len(selected) >= _settings.few_shot_k:
            break

    return selected
