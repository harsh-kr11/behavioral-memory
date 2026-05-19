"""Reference agent entry point — works with real LLM + in-memory store.

Run modes:
    python -m agent.app "Build a revenue analysis pipeline"   # single query
    python -m agent.app --interactive                          # REPL mode
    python -m agent.app --benchmark --limit 5                  # quick benchmark
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

console = Console()


def create_agent(model: str = "gemini-2.5-pro", use_postgres: bool = False):
    """Create the agent with all components wired up.

    Returns (graph, store, registry, tracer, settings).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    from agent.graph import build_agent_graph
    from behavioral_memory.core.config import Settings
    from behavioral_memory.evaluation.seed_traces import get_seed_traces
    from behavioral_memory.observability.tracer import LangfuseTracer
    from behavioral_memory.tools.mock_tools import get_tool_schemas
    from behavioral_memory.tools.registry import ToolRegistry

    settings = Settings()

    llm = ChatGoogleGenerativeAI(model=model, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if use_postgres:
        from behavioral_memory.memory.store import TraceStore

        store = TraceStore(embeddings=embeddings, settings=settings)
    else:
        from behavioral_memory.memory.in_memory_store import InMemoryTraceStore

        store = InMemoryTraceStore(embeddings=embeddings, settings=settings)

    registry = ToolRegistry()
    registry.register_many(get_tool_schemas())
    tracer = LangfuseTracer(settings=settings)

    seed_traces = get_seed_traces()
    store.add_bulk(seed_traces)

    graph = build_agent_graph(
        llm=llm,
        store=store,
        registry=registry,
        settings=settings,
    )

    return graph, store, registry, tracer, settings


def run_single(query: str, model: str = "gemini-2.5-pro", verbose: bool = True) -> dict:
    """Run the agent on a single query and display results."""
    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Query: {query}[/dim]\n")

    graph, store, _registry, tracer, _settings = create_agent(model=model)

    console.print(f"[dim]Memory: {store.count()} seed traces loaded[/dim]")
    if tracer.enabled:
        console.print("[green]Langfuse tracing: enabled[/green]")

    compiled = graph.compile()
    result = compiled.invoke({"query": query})

    plan = result.get("plan")
    if plan:
        console.print(f"\n[bold green]Plan generated — {len(plan.steps)} steps:[/bold green]")
        for step in plan.steps:
            console.print(f"  [cyan]{step.step_id}[/cyan]: {step.tool_name}")
            if verbose:
                for k, v in step.parameters.items():
                    val_str = str(v)[:100]
                    console.print(f"    {k}: {val_str}")
            if step.depends_on:
                console.print(f"    [dim]depends_on: {step.depends_on}[/dim]")

        if plan.retrieved_traces:
            console.print(f"\n[dim]Retrieved {len(plan.retrieved_traces)} traces from memory:[/dim]")
            for t in plan.retrieved_traces:
                console.print(f"  [dim]• {t.task_description[:70]}[/dim]")

        console.print(f"\n[dim]Token budget used: {plan.token_budget_used}[/dim]")

        if tracer.enabled:
            trace_id = tracer.log_plan(plan, tags=["agent-run"])
            if trace_id:
                console.print(f"[green]Logged to Langfuse: {trace_id}[/green]")
            tracer.flush()
    else:
        console.print(f"\n[red]Planning failed: {result.get('error', 'unknown')}[/red]")

    return result


def run_interactive(model: str = "gemini-2.5-pro") -> None:
    """Interactive REPL — type queries, see plans, compare with/without memory."""
    console.print(
        Panel.fit(
            "[bold]Behavioral Memory Agent — Interactive Mode[/bold]\n\n"
            f"Model: {model}\n"
            "Type a query to generate a plan. The agent retrieves relevant\n"
            "traces from behavioral memory to guide its planning.\n\n"
            "Special commands:\n"
            "  /compare <query>  — run with AND without memory, show difference\n"
            "  /memory            — show what's in behavioral memory\n"
            "  /quit              — exit",
            title="Interactive Agent",
        )
    )

    graph, store, registry, tracer, settings = create_agent(model=model)
    compiled = graph.compile()

    console.print(f"[green]Ready. Memory: {store.count()} traces loaded.[/green]\n")

    while True:
        try:
            query = console.input("[bold]Query>[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("/quit", "/exit", "quit", "exit"):
            break

        if query.startswith("/memory"):
            from behavioral_memory.evaluation.seed_traces import get_seed_traces

            for trace in get_seed_traces():
                tools = " → ".join(trace.tool_names)
                console.print(f"  [cyan]{trace.task_description[:60]}[/cyan]")
                console.print(f"    [dim]{tools}[/dim]")
            console.print(f"\n  [dim]Total: {store.count()} traces[/dim]\n")
            continue

        if query.startswith("/compare "):
            actual_query = query[9:].strip()
            _run_comparison(compiled, actual_query, store, registry, settings, tracer)
            continue

        result = compiled.invoke({"query": query})
        plan = result.get("plan")
        if plan:
            console.print(f"\n[green]Plan ({len(plan.steps)} steps):[/green]")
            for step in plan.steps:
                console.print(f"  [cyan]{step.step_id}[/cyan]: {step.tool_name}")
                for k, v in step.parameters.items():
                    console.print(f"    {k}: {str(v)[:80]}")
            if plan.retrieved_traces:
                console.print(f"\n  [dim]Retrieved {len(plan.retrieved_traces)} traces from memory[/dim]")
            if tracer.enabled:
                tracer.log_plan(plan, tags=["interactive"])
                tracer.flush()
        else:
            console.print(f"[red]Failed: {result.get('error', 'unknown')}[/red]")
        console.print()


def _run_comparison(compiled, query, store, registry, settings, tracer):
    """Run with and without memory, show the difference."""
    from behavioral_memory.planner.prompt import build_prompt
    from behavioral_memory.tools.mock_tools import get_tool_schemas

    console.print(f'\n[bold]Comparing: "{query}"[/bold]\n')

    result_with = compiled.invoke({"query": query})
    plan_with = result_with.get("plan")

    schemas = get_tool_schemas()

    console.print("[yellow]WITHOUT memory (zero-shot):[/yellow]")
    zs_prompt = build_prompt(query=query, traces=[], tool_schemas=schemas)
    console.print(f"  Prompt: {len(zs_prompt)} chars, 0 reference examples")

    console.print("\n[green]WITH memory (dynamic retrieval):[/green]")
    if plan_with:
        console.print(f"  Retrieved: {len(plan_with.retrieved_traces)} traces")
        for t in plan_with.retrieved_traces:
            console.print(f"    [dim]• {t.task_description[:60]}[/dim]")
        console.print(f"\n  Plan ({len(plan_with.steps)} steps):")
        for step in plan_with.steps:
            console.print(f"    [cyan]{step.step_id}[/cyan]: {step.tool_name}")
    else:
        console.print(f"  [red]Failed: {result_with.get('error')}[/red]")
    console.print()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Behavioral Memory Reference Agent")
    parser.add_argument("query", nargs="*", help="Query to process")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model to use")
    parser.add_argument("--postgres", action="store_true", help="Use PostgreSQL instead of in-memory store")
    args = parser.parse_args()

    if args.interactive:
        run_interactive(model=args.model)
    elif args.query:
        query = " ".join(args.query)
        run_single(query, model=args.model)
    else:
        console.print("[red]Usage:[/red]")
        console.print("  python -m agent.app 'your query here'")
        console.print("  python -m agent.app --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
