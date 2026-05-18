"""CLI for the behavioral-memory framework.

Provides commands for running benchmarks, managing memory, and
operating the reference agent.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="behavioral-memory",
    help="Behavioral Memory for Tool Orchestration — CLI",
    no_args_is_help=True,
)
console = Console()

# --- Sub-commands ---
memory_app = typer.Typer(help="Manage the behavioral memory store")
benchmark_app = typer.Typer(help="Run evaluation benchmarks")
app.add_typer(memory_app, name="memory")
app.add_typer(benchmark_app, name="benchmark")


@app.command()
def version() -> None:
    """Show the installed version."""
    from behavioral_memory import __version__

    console.print(f"behavioral-memory v{__version__}")


@memory_app.command("count")
def memory_count() -> None:
    """Count traces in the behavioral memory store."""
    from behavioral_memory.core.config import Settings

    settings = Settings()
    console.print(f"[dim]Store:[/dim] {settings.vector_store_url}")
    console.print("[yellow]Note: requires a running PostgreSQL with pgvector[/yellow]")


@memory_app.command("seed")
def memory_seed() -> None:
    """Load the 12 seed traces into memory (for benchmarking)."""
    from behavioral_memory.evaluation.seed_traces import get_seed_traces

    traces = get_seed_traces()
    console.print(f"[green]Loaded {len(traces)} seed traces[/green]")
    console.print("[dim]To store them, connect to pgvector and use the Python API[/dim]")


@benchmark_app.command("info")
def benchmark_info() -> None:
    """Show benchmark dataset information."""
    from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS
    from behavioral_memory.evaluation.seed_traces import get_seed_traces

    table = Table(title="Benchmark Dataset")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    difficulties = {}
    for task in EVALUATION_TASKS:
        d = task["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1

    for diff, count in sorted(difficulties.items()):
        table.add_row(f"Tasks ({diff})", str(count))
    table.add_row("Total tasks", str(len(EVALUATION_TASKS)), style="bold")
    table.add_row("Seed traces", str(len(get_seed_traces())))
    table.add_row("Tools", "7")

    console.print(table)


@benchmark_app.command("ground-truth")
def benchmark_ground_truth(limit: int = typer.Option(0, help="Limit output")) -> None:
    """Display the ground truth evaluation tasks."""
    from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS

    tasks = EVALUATION_TASKS[:limit] if limit > 0 else EVALUATION_TASKS

    for task in tasks:
        console.print(f"\n[cyan]Task {task['task_id']}[/cyan] ({task['difficulty']})")
        console.print(f"  {task['task']}")
        tools = [s["tool"] for s in task["gold_tool_chain"]]
        console.print(f"  [dim]Tools: {' → '.join(tools)}[/dim]")


@benchmark_app.command("seed-traces")
def benchmark_seed_traces() -> None:
    """Display the 12 seed traces."""
    from behavioral_memory.evaluation.seed_traces import get_seed_traces

    for trace in get_seed_traces():
        console.print(f"\n[cyan]{trace.task_description}[/cyan]")
        tools = " → ".join(trace.tool_names)
        console.print(f"  [dim]Chain: {tools}[/dim]")
        if "explanation" in trace.metadata:
            console.print(f"  [italic]{trace.metadata['explanation']}[/italic]")


@benchmark_app.command("tools")
def benchmark_tools() -> None:
    """Display the 7 benchmark tool definitions."""
    from behavioral_memory.tools.mock_tools import TOOL_DEFINITIONS

    for tool in TOOL_DEFINITIONS:
        required = tool["input_schema"].get("required", [])
        console.print(f"\n[cyan]{tool['name']}[/cyan]")
        console.print(f"  {tool['description'][:100]}...")
        console.print(f"  [dim]Required: {', '.join(required)}[/dim]")
