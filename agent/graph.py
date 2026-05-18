"""LangGraph 1.x reference agent — StateGraph definition.

Wires up the five nodes into a graph that implements the full
three-layer architecture from the paper:

    START -> fetch_schemas -> retrieve_traces -> generate_plan -> execute_tools -> log_trace -> END
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from agent.nodes.execute import execute_tools
from agent.nodes.fetch_schemas import make_fetch_schemas_node
from agent.nodes.observe import make_observe_node
from agent.nodes.plan import make_plan_node
from agent.nodes.retrieve import make_retrieve_node
from agent.state import AgentState
from behavioral_memory.core.config import Settings
from behavioral_memory.observability.tracer import LangfuseTracer
from behavioral_memory.planner.engine import PlanEngine
from behavioral_memory.tools.registry import ToolRegistry


def build_agent_graph(
    llm: BaseChatModel,
    store: Any,
    registry: ToolRegistry,
    settings: Settings | None = None,
) -> StateGraph:
    """Build and compile the reference agent graph.

    All components are injected — the graph itself is model-agnostic.
    """
    _settings = settings or Settings()
    engine = PlanEngine(llm=llm, store=store, registry=registry, settings=_settings)
    tracer = LangfuseTracer(settings=_settings)

    graph = StateGraph(AgentState)

    graph.add_node("fetch_schemas", make_fetch_schemas_node(registry))
    graph.add_node("retrieve_traces", make_retrieve_node(store))
    graph.add_node("generate_plan", make_plan_node(engine))
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("log_trace", make_observe_node(tracer))

    graph.add_edge(START, "fetch_schemas")
    graph.add_edge("fetch_schemas", "retrieve_traces")
    graph.add_edge("retrieve_traces", "generate_plan")
    graph.add_edge("generate_plan", "execute_tools")
    graph.add_edge("execute_tools", "log_trace")
    graph.add_edge("log_trace", END)

    return graph
