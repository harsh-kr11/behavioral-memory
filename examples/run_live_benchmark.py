"""Run the REAL benchmark — calls the LLM and produces actual numbers.

This script:
  1. Seeds 12 traces into an in-memory vector store (no PostgreSQL needed)
  2. Runs all 30 tasks through 3 strategies (zero-shot, static, dynamic)
  3. Scores every plan against gold tool chains
  4. Prints real TSA/PV/PCR/ESA numbers with bootstrap confidence intervals
  5. Optionally logs every plan to Langfuse

Prerequisites:
    pip install behavioral-memory[agent,eval]
    export GOOGLE_API_KEY=your-key-here

    Optional (for Langfuse tracing):
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_PUBLIC_KEY=pk-lf-...

Usage:
    python examples/run_live_benchmark.py
    python examples/run_live_benchmark.py --limit 5    # quick test with 5 tasks
    python examples/run_live_benchmark.py --model gemini-2.0-flash  # cheaper model
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the behavioral memory benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Limit to N tasks (0 = all 30)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro", help="Gemini model name")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output file")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold]Behavioral Memory — Live Benchmark[/bold]\n\n"
            "This runs the REAL benchmark from the paper.\n"
            "It calls the LLM for every task and scores plans against gold chains.\n"
            f"Model: {args.model}  |  Tasks: {'all 30' if args.limit == 0 else args.limit}",
            title="Live Benchmark",
        )
    )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError:
        console.print("[red]Missing dependency: pip install langchain-google-genai[/red]")
        sys.exit(1)

    from behavioral_memory.core.config import Settings
    from behavioral_memory.evaluation.benchmark import BenchmarkRunner
    from behavioral_memory.evaluation.seed_traces import get_seed_traces
    from behavioral_memory.evaluation.strategies import (
        DynamicRetrievalStrategy,
        StaticFewShotStrategy,
        ZeroShotStrategy,
    )
    from behavioral_memory.memory.in_memory_store import InMemoryTraceStore
    from behavioral_memory.observability.tracer import LangfuseTracer
    from behavioral_memory.planner.engine import PlanEngine
    from behavioral_memory.tools.mock_tools import get_tool_schemas
    from behavioral_memory.tools.registry import ToolRegistry

    settings = Settings()

    console.print("\n[dim]Initializing LLM and embeddings...[/dim]")
    llm = ChatGoogleGenerativeAI(model=args.model, temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    console.print("[dim]Creating in-memory vector store (no PostgreSQL needed)...[/dim]")
    store = InMemoryTraceStore(embeddings=embeddings, settings=settings)

    registry = ToolRegistry()
    schemas = get_tool_schemas()
    registry.register_many(schemas)

    seed_traces = get_seed_traces()
    store.add_bulk(seed_traces)
    console.print(f"[green]Seeded {store.count()} traces into in-memory store[/green]")

    engine = PlanEngine(llm=llm, store=store, registry=registry, settings=settings)
    runner = BenchmarkRunner(tool_schemas=schemas)

    tracer = LangfuseTracer(settings=settings)
    if tracer.enabled:
        console.print("[green]Langfuse tracing enabled — plans will be logged[/green]")
    else:
        console.print("[dim]Langfuse not configured — set LANGFUSE_SECRET_KEY to enable[/dim]")

    limit = args.limit if args.limit > 0 else None

    # --- Zero-shot ---
    console.print("\n[cyan]Running zero-shot baseline...[/cyan]")
    t0 = time.time()
    zero_shot = runner.run(ZeroShotStrategy(engine), limit=limit)
    zs_time = time.time() - t0
    console.print(f"  [dim]Completed in {zs_time:.1f}s[/dim]")
    _log_results_to_langfuse(tracer, zero_shot, "zero-shot")

    # --- Static few-shot ---
    console.print("[cyan]Running static few-shot baseline...[/cyan]")
    t0 = time.time()
    static = runner.run(StaticFewShotStrategy(engine, seed_traces[:3]), limit=limit)
    sf_time = time.time() - t0
    console.print(f"  [dim]Completed in {sf_time:.1f}s[/dim]")
    _log_results_to_langfuse(tracer, static, "static-few-shot")

    # --- Dynamic retrieval (proposed) ---
    console.print("[cyan]Running dynamic retrieval (proposed)...[/cyan]")
    t0 = time.time()
    dynamic = runner.run(DynamicRetrievalStrategy(engine), limit=limit)
    dr_time = time.time() - t0
    console.print(f"  [dim]Completed in {dr_time:.1f}s[/dim]")
    _log_results_to_langfuse(tracer, dynamic, "dynamic-retrieval")

    # --- Results table ---
    n = zero_shot["n_tasks"]
    table = Table(title=f"Benchmark Results (N={n}, model={args.model})")
    table.add_column("Metric", style="bold")
    table.add_column("Zero-Shot", justify="right")
    table.add_column("Static Few-Shot", justify="right")
    table.add_column("Dynamic (Proposed)", justify="right", style="bold green")

    for metric in ["tsa", "pv", "pcr", "esa"]:
        zs = zero_shot["aggregate"][metric]
        sf = static["aggregate"][metric]
        dy = dynamic["aggregate"][metric]

        zs_str = _fmt_metric(zs)
        sf_str = _fmt_metric(sf)
        dy_str = _fmt_metric(dy)

        table.add_row(metric.upper(), zs_str, sf_str, dy_str)

    console.print("\n")
    console.print(table)

    comparison = runner.compare(zero_shot, dynamic, "Zero-Shot", "Proposed")
    p_val = comparison["mcnemar_pcr"]["p_value"]
    console.print(f"\nMcNemar's test (zero-shot vs proposed): p = {p_val:.4f}")
    if p_val < 0.05:
        console.print("[green]  → Statistically significant (p < 0.05)[/green]")
    else:
        console.print("[yellow]  → Not statistically significant (p >= 0.05)[/yellow]")

    # --- Per-difficulty breakdown ---
    diff_table = Table(title="Plan Correctness by Difficulty")
    diff_table.add_column("Difficulty", style="bold")
    diff_table.add_column("n", justify="right")
    diff_table.add_column("Zero-Shot PCR", justify="right")
    diff_table.add_column("Static PCR", justify="right")
    diff_table.add_column("Dynamic PCR", justify="right", style="bold green")

    for diff in ["simple", "moderate", "challenging"]:
        zs_diff = runner.results_by_difficulty(zero_shot).get(diff, {})
        sf_diff = runner.results_by_difficulty(static).get(diff, {})
        dy_diff = runner.results_by_difficulty(dynamic).get(diff, {})
        diff_table.add_row(
            diff,
            str(zs_diff.get("n", 0)),
            f"{zs_diff.get('pcr', 0):.0%}",
            f"{sf_diff.get('pcr', 0):.0%}",
            f"{dy_diff.get('pcr', 0):.0%}",
        )

    console.print(diff_table)

    # --- Per-task details ---
    console.print("\n[bold]Per-task breakdown (dynamic retrieval):[/bold]")
    for task_result in dynamic["per_task"]:
        m = task_result["metrics"]
        status = "✓" if m["pcr"] else "✗"
        style = "green" if m["pcr"] else "red"
        console.print(
            f"  [{style}]{status}[/{style}] Task {task_result['task_id']} ({task_result['difficulty']}): "
            f"TSA={'✓' if m['tsa'] else '✗'} PV={m['pv']:.0%} ESA={'✓' if m['esa'] else '✗'}"
        )

    # --- Save results ---
    all_results = {
        "model": args.model,
        "n_tasks": n,
        "zero_shot": zero_shot,
        "static_few_shot": static,
        "dynamic_retrieval": dynamic,
        "comparison": comparison,
        "timing": {"zero_shot_s": zs_time, "static_s": sf_time, "dynamic_s": dr_time},
    }
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    console.print(f"\n[dim]Full results saved to {args.output}[/dim]")

    if tracer.enabled:
        tracer.flush()
        console.print("[green]All results logged to Langfuse[/green]")


def _fmt_metric(m: dict) -> str:
    mean = m["mean"]
    if isinstance(mean, bool):
        return "✓" if mean else "✗"
    ci = m.get("ci_95")
    if ci:
        return f"{mean:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"
    return f"{mean:.1%}"


def _log_results_to_langfuse(tracer, results: dict, strategy_name: str) -> None:
    """Log each plan to Langfuse for observability."""
    if not tracer.enabled:
        return
    for task_result in results.get("per_task", []):
        if "predicted_steps" in task_result:
            from behavioral_memory.core.schemas import Plan, ToolCall

            steps = [ToolCall(**s) for s in task_result["predicted_steps"]]
            plan = Plan(
                query=task_result["task"],
                steps=steps,
                raw_llm_output=json.dumps(task_result["predicted_steps"]),
            )
            tracer.log_plan(
                plan,
                tags=["benchmark", strategy_name, task_result["difficulty"]],
            )


if __name__ == "__main__":
    main()
