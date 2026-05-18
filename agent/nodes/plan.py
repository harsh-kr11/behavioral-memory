"""Node: generate_plan — calls the PlanEngine to produce a tool chain."""

from __future__ import annotations

from agent.state import AgentState
from behavioral_memory.planner.engine import PlanEngine


def make_plan_node(engine: PlanEngine):
    """Factory that creates a generate_plan node bound to a PlanEngine."""

    def generate_plan(state: AgentState) -> dict:
        query = state["query"]
        traces = state.get("retrieved_traces", [])
        schemas = state.get("tool_schemas", [])

        try:
            plan = engine.generate(
                query=query,
                tool_schemas=schemas,
                traces=traces,
            )
            return {"plan": plan}
        except Exception as e:
            return {"plan": None, "error": str(e)}

    return generate_plan
