"""Node: retrieve_traces — fetches relevant traces from behavioral memory."""

from __future__ import annotations

from agent.state import AgentState
from behavioral_memory.memory.store import TraceStore
from behavioral_memory.memory.token_budget import select_traces_within_budget


def make_retrieve_node(store: TraceStore):
    """Factory that creates a retrieve_traces node bound to a TraceStore."""

    def retrieve_traces(state: AgentState) -> dict:
        query = state["query"]
        tool_schemas = state.get("tool_schemas", [])

        traces = select_traces_within_budget(
            store=store,
            query=query,
            tool_schemas=tool_schemas,
        )
        return {"retrieved_traces": traces}

    return retrieve_traces
