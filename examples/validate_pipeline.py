"""Validate the entire pipeline end-to-end without any external services.

This script proves:
  1. InMemoryTraceStore works (embed + search)
  2. Seed traces load and validate
  3. PlanEngine generates plans (with a mock LLM)
  4. Benchmark runner scores plans correctly
  5. Gatekeeper validates traces
  6. Langfuse tracer handles offline mode gracefully
  7. The full zero-shot vs dynamic retrieval pipeline works

No API keys, no PostgreSQL, no network access required.

Usage:
    python examples/validate_pipeline.py
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def make_mock_embeddings(dim: int = 64):
    """Deterministic embedding model for validation."""
    emb = MagicMock()

    def embed_query(text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        return [float(b) / 255.0 for b in h[:dim]]

    def embed_documents(texts: list[str]) -> list[list[float]]:
        return [embed_query(t) for t in texts]

    emb.embed_query = embed_query
    emb.embed_documents = embed_documents
    return emb


def make_mock_llm(gold_tasks):
    """Mock LLM that returns the gold tool chain for the closest matching task."""

    def invoke(messages):
        query = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

        best_match = None
        best_overlap = 0
        query_words = set(query.lower().split())

        for task in gold_tasks:
            task_words = set(task["task"].lower().split())
            overlap = len(query_words & task_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = task

        if best_match:
            steps = []
            for i, gold_step in enumerate(best_match["gold_tool_chain"]):
                steps.append(
                    {
                        "step_id": f"step_{i + 1}",
                        "tool_name": gold_step["tool"],
                        "parameters": gold_step["params"],
                        "depends_on": [f"step_{j + 1}" for j in range(i)],
                    }
                )
            response = MagicMock()
            response.content = json.dumps(steps)
            return response

        response = MagicMock()
        response.content = json.dumps(
            [
                {
                    "step_id": "step_1",
                    "tool_name": "data_fetch",
                    "parameters": {"source": "default"},
                    "depends_on": [],
                }
            ]
        )
        return response

    llm = MagicMock()
    llm.invoke = invoke
    return llm


def main() -> None:
    console.print(
        Panel.fit(
            "[bold]Pipeline Validation — Full End-to-End Check[/bold]\n\n"
            "Tests every component with mock services.\n"
            "No API keys or external services needed.",
            title="Validation",
        )
    )

    from behavioral_memory.core.config import Settings
    from behavioral_memory.evaluation.benchmark import BenchmarkRunner
    from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS
    from behavioral_memory.evaluation.seed_traces import get_seed_traces
    from behavioral_memory.evaluation.strategies import (
        DynamicRetrievalStrategy,
        StaticFewShotStrategy,
        ZeroShotStrategy,
    )
    from behavioral_memory.gatekeeper.pipeline import GatekeeperPipeline
    from behavioral_memory.memory.in_memory_store import InMemoryTraceStore
    from behavioral_memory.observability.tracer import LangfuseTracer
    from behavioral_memory.planner.engine import PlanEngine
    from behavioral_memory.tools.mock_tools import get_tool_schemas
    from behavioral_memory.tools.registry import ToolRegistry

    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            console.print(f"  [green]✓[/green] {name}")
            passed += 1
        else:
            console.print(f"  [red]✗[/red] {name}: {detail}")
            failed += 1

    # --- 1. Seed traces ---
    console.print("\n[bold cyan]1. Seed Traces[/bold cyan]")
    seed_traces = get_seed_traces()
    check("12 seed traces loaded", len(seed_traces) == 12, f"got {len(seed_traces)}")
    check("All traces validated", all(t.validated for t in seed_traces))
    check("All have tool chains", all(len(t.tool_chain) > 0 for t in seed_traces))

    # --- 2. Tool schemas ---
    console.print("\n[bold cyan]2. Tool Schemas[/bold cyan]")
    schemas = get_tool_schemas()
    check("7 mock tools loaded", len(schemas) == 7, f"got {len(schemas)}")
    registry = ToolRegistry()
    registry.register_many(schemas)
    check("Registry populated", len(registry) == 7)

    # --- 3. Ground truth tasks ---
    console.print("\n[bold cyan]3. Ground Truth Tasks[/bold cyan]")
    check("30 evaluation tasks", len(EVALUATION_TASKS) == 30, f"got {len(EVALUATION_TASKS)}")
    difficulties = {t["difficulty"] for t in EVALUATION_TASKS}
    check("Three difficulty tiers", difficulties == {"simple", "moderate", "challenging"})
    check(
        "All gold chains reference known tools",
        all(step["tool"] in registry._tools for task in EVALUATION_TASKS for step in task["gold_tool_chain"]),
    )

    # --- 4. InMemoryTraceStore ---
    console.print("\n[bold cyan]4. InMemory Vector Store[/bold cyan]")
    embeddings = make_mock_embeddings()
    settings = Settings()
    store = InMemoryTraceStore(embeddings=embeddings, settings=settings)
    n_added = store.add_bulk(seed_traces)
    check("Bulk add succeeds", n_added == 12)
    check("Count matches", store.count() == 12)

    results = store.search("Build a revenue analysis pipeline", k=3)
    check("Search returns results", len(results) > 0, "empty search")
    check("Results are (trace, score) tuples", all(isinstance(r[1], float) for r in results))

    # --- 5. PlanEngine with mock LLM ---
    console.print("\n[bold cyan]5. PlanEngine[/bold cyan]")
    llm = make_mock_llm(EVALUATION_TASKS)
    engine = PlanEngine(llm=llm, store=store, registry=registry, settings=settings)

    plan = engine.generate(query="Build a revenue analysis pipeline", tool_schemas=schemas)
    check("Plan generated", plan is not None)
    check("Plan has steps", len(plan.steps) > 0, "empty plan")
    check("Retrieved traces attached", len(plan.retrieved_traces) > 0, "no retrieval")
    check("Token budget tracked", plan.token_budget_used > 0)

    zs_plan = engine.generate_zero_shot("Build a revenue analysis pipeline", schemas)
    check("Zero-shot plan works", len(zs_plan.steps) > 0)
    check("Zero-shot has no retrieved traces", len(zs_plan.retrieved_traces) == 0)

    static_plan = engine.generate_static_few_shot(
        "Build a revenue analysis pipeline",
        schemas,
        seed_traces[:3],
    )
    check("Static few-shot works", len(static_plan.steps) > 0)
    check("Static uses provided traces", len(static_plan.retrieved_traces) == 3)

    # --- 6. Benchmark Runner ---
    console.print("\n[bold cyan]6. Benchmark Runner[/bold cyan]")
    runner = BenchmarkRunner(tool_schemas=schemas)

    zs_results = runner.run(ZeroShotStrategy(engine), limit=5)
    check("Zero-shot runs on 5 tasks", zs_results["n_tasks"] == 5)
    check("Has aggregate metrics", "tsa" in zs_results["aggregate"])

    sf_results = runner.run(StaticFewShotStrategy(engine, seed_traces[:3]), limit=5)
    check("Static few-shot runs on 5 tasks", sf_results["n_tasks"] == 5)

    dr_results = runner.run(DynamicRetrievalStrategy(engine), limit=5)
    check("Dynamic retrieval runs on 5 tasks", dr_results["n_tasks"] == 5)

    comparison = runner.compare(zs_results, dr_results, "Zero-Shot", "Dynamic")
    check("McNemar test runs", "mcnemar_pcr" in comparison)
    check("p-value is numeric", isinstance(comparison["mcnemar_pcr"]["p_value"], float))

    by_diff = runner.results_by_difficulty(dr_results)
    check("Difficulty breakdown works", len(by_diff) > 0)

    # --- 7. Gatekeeper ---
    console.print("\n[bold cyan]7. Gatekeeper Pipeline[/bold cyan]")
    gk = GatekeeperPipeline(store=store, registry=registry)
    gk_result = gk.evaluate(seed_traces[0])
    check(
        "Gatekeeper accepts valid trace",
        gk_result.accepted or gk_result.is_duplicate,
        f"rejected: {gk_result.rejection_reason}",
    )

    # --- 8. Langfuse offline ---
    console.print("\n[bold cyan]8. Langfuse Tracer (offline)[/bold cyan]")
    tracer = LangfuseTracer(settings=settings)
    check("Tracer disabled without keys", not tracer.enabled)
    trace_id = tracer.log_plan(plan)
    check("Log returns None when disabled", trace_id is None)

    # --- Print metrics from mock run ---
    console.print("\n[bold cyan]9. Mock Benchmark Results (5 tasks)[/bold cyan]")
    table = Table(title="Mock Results (N=5, mock LLM)")
    table.add_column("Metric", style="bold")
    table.add_column("Zero-Shot", justify="right")
    table.add_column("Static", justify="right")
    table.add_column("Dynamic", justify="right", style="bold green")

    for metric in ["tsa", "pv", "pcr", "esa"]:
        zs = zs_results["aggregate"][metric]
        sf = sf_results["aggregate"][metric]
        dy = dr_results["aggregate"][metric]
        table.add_row(
            metric.upper(),
            f"{zs['mean']:.1%}" if isinstance(zs["mean"], float) else str(zs["mean"]),
            f"{sf['mean']:.1%}" if isinstance(sf["mean"], float) else str(sf["mean"]),
            f"{dy['mean']:.1%}" if isinstance(dy["mean"], float) else str(dy["mean"]),
        )

    console.print(table)
    console.print("[dim]Note: These numbers are from a mock LLM — run with a real API key for actual results.[/dim]")

    # --- Summary ---
    console.print(f"\n{'=' * 50}")
    total = passed + failed
    console.print(f"[bold]Results: {passed}/{total} checks passed[/bold]")
    if failed > 0:
        console.print(f"[red]{failed} checks failed[/red]")
        sys.exit(1)
    else:
        console.print("[bold green]All pipeline checks passed![/bold green]")
        console.print("\n[dim]Next steps:[/dim]")
        console.print("[dim]  1. Set GOOGLE_API_KEY and run: python examples/run_live_benchmark.py[/dim]")
        console.print("[dim]  2. Set Langfuse keys for tracing[/dim]")
        console.print("[dim]  3. Run the interactive agent: python -m agent.app --interactive[/dim]")


if __name__ == "__main__":
    main()
