"""Reference agent entry point.

Demonstrates the full behavioral memory system end-to-end using
LangGraph 1.x with Gemini as the default LLM.
"""

from __future__ import annotations

import json
import sys

from rich.console import Console

console = Console()


def run_agent(query: str, verbose: bool = False) -> dict:
    """Run the reference agent on a single query."""
    from langchain_core.embeddings import Embeddings
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    from agent.graph import build_agent_graph
    from behavioral_memory.core.config import Settings
    from behavioral_memory.memory.store import TraceStore
    from behavioral_memory.tools.mock_tools import get_tool_schemas
    from behavioral_memory.tools.registry import ToolRegistry

    settings = Settings()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
    )

    embeddings: Embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
    )

    store = TraceStore(embeddings=embeddings, settings=settings)
    registry = ToolRegistry()
    registry.register_many(get_tool_schemas())

    graph = build_agent_graph(
        llm=llm,
        store=store,
        registry=registry,
        settings=settings,
    )

    compiled = graph.compile()
    result = compiled.invoke({"query": query})

    if verbose:
        plan = result.get("plan")
        if plan:
            console.print(f"\n[cyan]Plan ({len(plan.steps)} steps):[/cyan]")
            for step in plan.steps:
                console.print(f"  {step.step_id}: {step.tool_name}")
                console.print(f"    params: {json.dumps(step.parameters, indent=4)}")

    return result


def main() -> None:
    """CLI entry point for the reference agent."""
    if len(sys.argv) < 2:
        console.print("[red]Usage: python -m agent.app 'your query here'[/red]")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    console.print(f"[dim]Query:[/dim] {query}")

    result = run_agent(query, verbose=True)

    plan = result.get("plan")
    if plan:
        console.print(f"\n[green]Plan generated with {len(plan.steps)} steps[/green]")
    else:
        console.print(f"\n[red]Planning failed: {result.get('error', 'unknown')}[/red]")


if __name__ == "__main__":
    main()
