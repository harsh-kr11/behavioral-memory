"""LangGraph server entry point for the behavioral memory agent.

Exposes the agent as a LangGraph-compatible graph that works with:
  - `langgraph dev` (development server with hot reload)
  - Agent Chat UI (https://github.com/langchain-ai/agent-chat-ui)
  - Any LangGraph Platform-compatible client

Usage:
    pip install "langgraph-cli[inmem]"
    langgraph dev

    # Or directly:
    python server.py
"""

from __future__ import annotations

import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState

from behavioral_memory.core.config import Settings
from behavioral_memory.evaluation.seed_traces import get_seed_traces
from behavioral_memory.memory.in_memory_store import InMemoryTraceStore
from behavioral_memory.observability.tracer import LangfuseTracer
from behavioral_memory.planner.engine import PlanEngine
from behavioral_memory.tools.mock_tools import get_tool_schemas
from behavioral_memory.tools.registry import ToolRegistry

settings = Settings()

model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

use_postgres = os.getenv("USE_POSTGRES", "false").lower() == "true"
if use_postgres:
    from behavioral_memory.memory.store import TraceStore

    store = TraceStore(embeddings=embeddings, settings=settings)
else:
    store = InMemoryTraceStore(embeddings=embeddings, settings=settings)

registry = ToolRegistry()
registry.register_many(get_tool_schemas())
store.add_bulk(get_seed_traces())

engine = PlanEngine(llm=llm, store=store, registry=registry, settings=settings)
tracer = LangfuseTracer(settings=settings)

schemas = get_tool_schemas()


def plan_agent(state: MessagesState) -> dict:
    """Core agent node: takes the last user message, generates a plan."""
    messages = state["messages"]
    last_msg = messages[-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    try:
        plan = engine.generate(query=query, tool_schemas=schemas)

        steps_text = []
        for step in plan.steps:
            params_str = ", ".join(f"{k}={v!r}" for k, v in step.parameters.items())
            deps = f" (depends on: {', '.join(step.depends_on)})" if step.depends_on else ""
            steps_text.append(f"**{step.step_id}**: `{step.tool_name}({params_str})`{deps}")

        retrieved_info = ""
        if plan.retrieved_traces:
            examples = [f"  - {t.task_description[:80]}" for t in plan.retrieved_traces]
            retrieved_info = (
                f"\n\n**Retrieved {len(plan.retrieved_traces)} reference traces from behavioral memory:**\n"
                + "\n".join(examples)
            )

        response = (
            f"## Execution Plan ({len(plan.steps)} steps)\n\n"
            + "\n".join(steps_text)
            + retrieved_info
            + f"\n\n*Token budget used: {plan.token_budget_used} tokens*"
        )

        if tracer.enabled:
            trace_id = tracer.log_plan(plan, tags=["agent-chat"])
            if trace_id:
                response += f"\n\n*Logged to Langfuse: `{trace_id}`*"
            tracer.flush()

        return {"messages": [AIMessage(content=response)]}

    except Exception as e:
        return {"messages": [AIMessage(content=f"Planning failed: {e}")]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("plan", plan_agent)
graph_builder.add_edge(START, "plan")
graph_builder.add_edge("plan", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    print("Testing the graph locally...")
    result = graph.invoke({"messages": [HumanMessage(content="Build a revenue analysis pipeline")]})
    for msg in result["messages"]:
        print(f"\n{msg.content}")
