"""Reproduce the paper's benchmark results.

Runs the 30-task evaluation with zero-shot, static few-shot, and
dynamic retrieval strategies, then prints the comparison table.

Prerequisites:
    pip install behavioral-memory[agent,eval]
    # PostgreSQL with pgvector running
"""

import json

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from rich.console import Console
from rich.table import Table

from behavioral_memory import PlanEngine, ToolRegistry, TraceStore
from behavioral_memory.core.config import Settings
from behavioral_memory.evaluation.benchmark import BenchmarkRunner
from behavioral_memory.evaluation.seed_traces import get_seed_traces
from behavioral_memory.evaluation.strategies import (
    DynamicRetrievalStrategy,
    StaticFewShotStrategy,
    ZeroShotStrategy,
)
from behavioral_memory.tools.mock_tools import get_tool_schemas

console = Console()


def main():
    settings = Settings()
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    store = TraceStore(embeddings=embeddings, settings=settings)
    registry = ToolRegistry()
    schemas = get_tool_schemas()
    registry.register_many(schemas)
    engine = PlanEngine(llm=llm, store=store, registry=registry, settings=settings)

    # Seed the store with 12 traces
    seed_traces = get_seed_traces()
    store.add_bulk(seed_traces)
    console.print(f"[green]Seeded {len(seed_traces)} traces into memory[/green]")

    runner = BenchmarkRunner(tool_schemas=schemas)

    # Run all three strategies
    console.print("\n[cyan]Running zero-shot baseline...[/cyan]")
    zero_shot = runner.run(ZeroShotStrategy(engine))

    console.print("[cyan]Running static few-shot baseline...[/cyan]")
    static = runner.run(StaticFewShotStrategy(engine, seed_traces[:3]))

    console.print("[cyan]Running dynamic retrieval (proposed)...[/cyan]")
    dynamic = runner.run(DynamicRetrievalStrategy(engine))

    # Print results table
    table = Table(title="Benchmark Results (N=30)")
    table.add_column("Metric")
    table.add_column("Zero-Shot", justify="right")
    table.add_column("Static Few-Shot", justify="right")
    table.add_column("Dynamic (Proposed)", justify="right", style="bold")

    for metric in ["tsa", "pv", "pcr", "esa"]:
        zs = zero_shot["aggregate"][metric]
        sf = static["aggregate"][metric]
        dy = dynamic["aggregate"][metric]

        zs_str = f"{zs['mean']:.1%}" if isinstance(zs["mean"], float) else f"{zs['mean']}"
        sf_str = f"{sf['mean']:.1%}" if isinstance(sf["mean"], float) else f"{sf['mean']}"
        dy_str = f"{dy['mean']:.1%}" if isinstance(dy["mean"], float) else f"{dy['mean']}"

        table.add_row(metric.upper(), zs_str, sf_str, dy_str)

    console.print(table)

    # McNemar comparison
    comparison = runner.compare(zero_shot, dynamic, "Zero-Shot", "Proposed")
    p_val = comparison["mcnemar_pcr"]["p_value"]
    console.print(f"\nMcNemar's test (zero-shot vs proposed): p = {p_val:.4f}")

    # Save full results
    with open("benchmark_results.json", "w") as f:
        json.dump({"zero_shot": zero_shot, "static": static, "dynamic": dynamic}, f, indent=2, default=str)
    console.print("[dim]Full results saved to benchmark_results.json[/dim]")


if __name__ == "__main__":
    main()
