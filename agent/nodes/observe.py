"""Node: log_to_langfuse — logs the plan and execution to Langfuse."""

from __future__ import annotations

from agent.state import AgentState
from behavioral_memory.observability.tracer import LangfuseTracer


def make_observe_node(tracer: LangfuseTracer):
    """Factory that creates a log_to_langfuse node bound to a tracer."""

    def log_to_langfuse(state: AgentState) -> dict:
        plan = state.get("plan")
        if plan is None or not tracer.enabled:
            return {"langfuse_trace_id": None}

        trace_id = tracer.log_plan(plan)
        return {"langfuse_trace_id": trace_id}

    return log_to_langfuse
