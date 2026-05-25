"""CLI for the behavioral-memory framework.

Provides commands for running benchmarks, managing memory, setup,
and the offline demo that shows behavioral memory impact.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console

if TYPE_CHECKING:
    from behavioral_memory.core.schemas import ExecutionTrace
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="behavioral-memory",
    help="Behavioral Memory for Tool Orchestration — CLI",
    no_args_is_help=True,
)
console = Console()

memory_app = typer.Typer(help="Manage the behavioral memory store")
benchmark_app = typer.Typer(help="Run evaluation benchmarks")
app.add_typer(memory_app, name="memory")
app.add_typer(benchmark_app, name="benchmark")


@app.command()
def version() -> None:
    """Show the installed version."""
    from behavioral_memory import __version__

    console.print(f"behavioral-memory v{__version__}")


@app.command()
def setup() -> None:
    """Interactive setup: create .env file from template."""
    env_path = Path(".env")
    template_path = Path(".env.example")

    if env_path.exists():
        overwrite = typer.confirm(".env already exists. Overwrite?", default=False)
        if not overwrite:
            console.print("[dim]Keeping existing .env[/dim]")
            return

    if not template_path.exists():
        console.print("[red]Error: .env.example not found. Are you in the project root?[/red]")
        raise typer.Exit(1)

    shutil.copy(template_path, env_path)
    console.print("[green]Created .env from .env.example[/green]\n")

    console.print(
        Panel.fit(
            "[bold]Configure your .env file:[/bold]\n\n"
            "1. [cyan]GOOGLE_API_KEY[/cyan] — get one at https://aistudio.google.com/apikey\n"
            "   Only needed to run the reference agent with Gemini.\n"
            "   You can use any LangChain-compatible LLM instead.\n\n"
            "2. [cyan]VECTOR_STORE_URL[/cyan] — PostgreSQL + pgvector connection string\n"
            "   For local dev: docker run -p 5432:5432 -e POSTGRES_PASSWORD=pw pgvector/pgvector\n"
            "   [dim]Not needed for demo mode or tests.[/dim]\n\n"
            "3. [cyan]LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY[/cyan] — from https://cloud.langfuse.com\n"
            "   Free tier available. Enables the feedback loop.\n"
            "   [dim]Optional — the framework works without Langfuse.[/dim]",
            title="Setup Guide",
        )
    )


@app.command()
def demo(
    task_id: int = typer.Option(-1, help="Run a specific task ID (0-29), or -1 for a curated set"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full plan details"),
) -> None:
    """Run the offline demo showing behavioral memory impact.

    Compares zero-shot vs dynamic retrieval on benchmark tasks
    WITHOUT needing PostgreSQL, Langfuse, or an LLM API key.
    Uses the gold tool chains to simulate perfect retrieval.
    """

    from behavioral_memory.core.schemas import ExecutionTrace, ToolCall
    from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS
    from behavioral_memory.evaluation.metrics import compute_metrics
    from behavioral_memory.evaluation.seed_traces import get_seed_traces
    from behavioral_memory.gatekeeper.sandbox import SandboxExecutor
    from behavioral_memory.gatekeeper.schema_validator import SchemaValidator
    from behavioral_memory.planner.prompt import build_prompt
    from behavioral_memory.tools.mock_tools import get_tool_schemas
    from behavioral_memory.tools.registry import ToolRegistry

    console.print(
        Panel.fit(
            "[bold]Behavioral Memory — Offline Demo[/bold]\n\n"
            "This demo shows how behavioral memory (validated execution traces)\n"
            "improves tool orchestration quality. No external services needed.\n\n"
            "It simulates: zero-shot (no memory) vs dynamic retrieval (with memory)\n"
            "on the paper's 30-task benchmark with 7 MCP tools.",
            title="Demo Mode",
        )
    )

    schemas = get_tool_schemas()
    registry = ToolRegistry()
    registry.register_many(schemas)
    seed_traces = get_seed_traces()
    validator = SchemaValidator(registry)
    sandbox = SandboxExecutor()

    if task_id >= 0:
        selected = [t for t in EVALUATION_TASKS if t["task_id"] == task_id]
        if not selected:
            console.print(f"[red]Task ID {task_id} not found (valid: 0-29)[/red]")
            raise typer.Exit(1)
    else:
        selected = [
            EVALUATION_TASKS[0],  # simple
            EVALUATION_TASKS[10],  # moderate
            EVALUATION_TASKS[20],  # challenging
        ]

    console.print(f"\n[dim]Seed traces loaded: {len(seed_traces)}[/dim]")
    console.print(f"[dim]Tools available: {len(schemas)}[/dim]")
    console.print(f"[dim]Tasks to run: {len(selected)}[/dim]\n")

    for task in selected:
        console.print(f"\n{'═' * 80}")
        console.print(f"[bold cyan]Task {task['task_id']}[/bold cyan] ({task['difficulty']})")
        console.print(f"  {task['task']}\n")

        gold = task["gold_tool_chain"]

        # --- Zero-shot: no memory, model only sees schemas ---
        zs_prompt = build_prompt(query=task["task"], traces=[], tool_schemas=schemas)
        console.print("[yellow]  ZERO-SHOT (no behavioral memory)[/yellow]")
        console.print(f"  [dim]Prompt size: {len(zs_prompt)} chars, 0 reference examples[/dim]")
        console.print("  [dim]Without memory, the LLM must guess conventions from schemas alone.[/dim]")
        console.print("  [dim]Example failure: may use total_amount instead of quantity*unit_price for 'revenue'[/dim]")

        # --- Dynamic retrieval: with memory ---
        relevant = _find_relevant_traces(task["task"], seed_traces, top_k=3)
        dr_prompt = build_prompt(query=task["task"], traces=relevant, tool_schemas=schemas)
        console.print("\n[green]  DYNAMIC RETRIEVAL (with behavioral memory)[/green]")
        console.print(f"  [dim]Prompt size: {len(dr_prompt)} chars, {len(relevant)} reference examples[/dim]")
        for i, trace in enumerate(relevant):
            console.print(f'    [dim]Retrieved #{i + 1}: "{trace.task_description[:60]}"[/dim]')

        # --- Gold chain validation ---
        trace_obj = ExecutionTrace(
            task_description=task["task"],
            tool_chain=[
                ToolCall(step_id=s["step_id"], tool_name=s["tool"], parameters=s.get("params", {})) for s in gold
            ],
            source="execution",
        )
        is_valid, _ = validator.validate(trace_obj)
        passed, _ = sandbox.execute(trace_obj)

        metrics = compute_metrics(gold, gold)

        console.print(f"\n  [bold]Gold tool chain ({len(gold)} steps):[/bold]")
        for step in gold:
            console.print(f"    {step['step_id']}: [cyan]{step['tool']}[/cyan]")
            if verbose and step.get("params"):
                for k, v in step["params"].items():
                    val_str = str(v)[:80]
                    console.print(f"      {k}: {val_str}")

        console.print(f"\n  Gatekeeper: schema={'✓' if is_valid else '✗'}  sandbox={'✓' if passed else '✗'}")
        console.print(
            f"  Metrics:    TSA={'✓' if metrics['tsa'] else '✗'}  PV={metrics['pv']:.0%}  PCR={'✓' if metrics['pcr'] else '✗'}  ESA={'✓' if metrics['esa'] else '✗'}"
        )

    # --- Summary table ---
    console.print(f"\n{'═' * 80}")
    _print_paper_results_table()

    console.print(
        Panel.fit(
            "[bold]Key Insight:[/bold] Without behavioral memory, the LLM has no way to learn\n"
            "domain conventions (e.g., 'revenue' = quantity*unit_price, not total_amount).\n"
            "With memory, validated traces teach these conventions dynamically.\n\n"
            "[bold]To run with a real LLM:[/bold]\n"
            "  1. Set GOOGLE_API_KEY in .env (or use any LangChain model)\n"
            "  2. Start PostgreSQL: docker run -p 5432:5432 pgvector/pgvector\n"
            "  3. Run: python examples/run_benchmark.py\n\n"
            "[bold]To enable Langfuse tracing:[/bold]\n"
            "  1. Sign up at https://cloud.langfuse.com (free)\n"
            "  2. Add LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY to .env\n"
            "  3. Every plan will be logged — SMEs can score them in the Langfuse UI\n"
            "  4. Positively scored traces auto-enter behavioral memory via FeedbackPoller",
            title="Next Steps",
        )
    )


def _find_relevant_traces(query: str, traces: list[ExecutionTrace], top_k: int = 3) -> list[ExecutionTrace]:
    """Simple keyword-based trace matching for demo mode (no embeddings needed)."""
    query_lower = query.lower()
    scored = []
    for trace in traces:
        task_lower = trace.task_description.lower()
        overlap = sum(1 for word in query_lower.split() if word in task_lower)
        scored.append((overlap, trace))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [trace for _, trace in scored[:top_k]]


def _print_paper_results_table() -> None:
    """Print multi-model benchmark results (Dynamic Retrieval / Proposed)."""
    table = Table(title="Benchmark — Dynamic Retrieval (Proposed) across Models")
    table.add_column("Metric", style="bold")
    table.add_column("Gemini 2.5 Pro", justify="right")
    table.add_column("Gemini 3 Flash", justify="right")
    table.add_column("Gemini 3.5 Flash", justify="right")

    table.add_row("Tool Selection (TSA)", "93.3%", "76.7%", "83.3%")
    table.add_row("Parameter Validity (PV)", "85.5%", "74.6%", "80.9%")
    table.add_row("Plan Correctness (PCR)", "80.0%", "76.7%", "80.0%")
    table.add_row("Sequence Accuracy (ESA)", "93.3%", "76.7%", "83.3%")
    table.add_row("McNemar p vs Zero-Shot", "p = 0.023", "p = 0.022", "p = 0.070")

    console.print(table)

    zs_table = Table(title="Zero-Shot Baseline (for comparison)")
    zs_table.add_column("Metric", style="bold")
    zs_table.add_column("Gemini 2.5 Pro", justify="right")
    zs_table.add_column("Gemini 3 Flash", justify="right")
    zs_table.add_column("Gemini 3.5 Flash", justify="right")

    zs_table.add_row("TSA", "63.3%", "60.0%", "73.3%")
    zs_table.add_row("PV", "60.6%", "70.2%", "71.1%")
    zs_table.add_row("PCR", "50.0%", "50.0%", "60.0%")
    zs_table.add_row("ESA", "63.3%", "60.0%", "73.3%")

    console.print(zs_table)


# --- Memory commands ---


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


# --- Benchmark commands ---


@benchmark_app.command("info")
def benchmark_info() -> None:
    """Show benchmark dataset information."""
    from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS
    from behavioral_memory.evaluation.seed_traces import get_seed_traces

    table = Table(title="Benchmark Dataset")
    table.add_column("Category", style="cyan")
    table.add_column("Count", justify="right")

    difficulties: dict[str, int] = {}
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
