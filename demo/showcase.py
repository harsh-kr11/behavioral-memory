"""Unified demo runner for conference talks and presentations.

Initializes the full behavioral-memory pipeline once, then runs
interactive demo "acts" that showcase each layer of the system:

  Act 1 — Memory Inspector:   Browse the trace store
  Act 2 — Side-by-Side Compare: Zero-shot vs static vs dynamic retrieval
  Act 3 — Gatekeeper Challenge: Feed poisoned traces through validation
  Act 4 — Custom Query:        Interactive REPL for audience questions

Usage:
    python demo/showcase.py                      # run all acts sequentially
    python demo/showcase.py --act 2              # jump to a specific act
    python demo/showcase.py --act 2 --query "…"  # custom query for compare
    python demo/showcase.py --model gemini-2.5-flash

Requires: GOOGLE_API_KEY (set in .env or environment)
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace, GatekeeperResult, Plan, ToolCall, ToolSchema
from behavioral_memory.evaluation.seed_traces import get_seed_traces
from behavioral_memory.gatekeeper.pipeline import GatekeeperPipeline
from behavioral_memory.memory.in_memory_store import InMemoryTraceStore
from behavioral_memory.memory.token_budget import count_tokens, select_traces_within_budget
from behavioral_memory.planner.engine import PlanEngine
from behavioral_memory.planner.prompt import SYSTEM_PROMPT, build_prompt
from behavioral_memory.tools.mock_tools import get_tool_schemas
from behavioral_memory.tools.registry import ToolRegistry

console = Console()

DEFAULT_QUERY = "Calculate average basket size per customer segment and alert the data team"

POISONED_TRACES_RAW: list[dict[str, Any]] = [
    {
        "task": "Get product data and generate report",
        "chain": [
            {"step_id": "s1", "tool": "generate_report", "params": {"source_step": "s0_nonexistent", "format": "markdown_table", "title": "Products"}},
        ],
        "poison_type": "broken_dependency",
        "explanation": "References s0_nonexistent which doesn't exist; skips the query step entirely",
    },
    {
        "task": "Get quarterly revenue and send report",
        "chain": [
            {"step_id": "s1", "tool": "query_database", "params": {"query": "SELECT total_amount FROM orders;"}},
            {"step_id": "s2", "tool": "generate_report", "params": {"source_step": "s1", "format": "csv", "title": "Revenue"}},
        ],
        "poison_type": "wrong_convention",
        "explanation": "Uses total_amount instead of quantity*unit_price for revenue; uses csv instead of markdown_table",
    },
    {
        "task": "Get completed orders and cache for dashboard",
        "chain": [
            {"step_id": "s1", "tool": "query_database", "params": {"query": "SELECT * FROM orders WHERE status = 'completed';"}},
            {"step_id": "s2", "tool": "store_results", "params": {"source_step": "s1", "target": "database_table", "target_name": "dashboard_data"}},
        ],
        "poison_type": "wrong_convention",
        "explanation": "Uses status='completed' instead of IN('shipped','delivered'); stores to database_table instead of cache",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_poisoned_trace(raw: dict[str, Any]) -> ExecutionTrace:
    chain = [ToolCall(step_id=s["step_id"], tool_name=s["tool"], parameters=s["params"]) for s in raw["chain"]]
    return ExecutionTrace(
        task_description=raw["task"],
        tool_chain=chain,
        validated=False,
        source="execution",
        metadata={"poison_type": raw["poison_type"], "explanation": raw["explanation"]},
    )


def _tool_chain_str(steps: list[ToolCall]) -> str:
    return " -> ".join(s.tool_name for s in steps)


def _pause(message: str = "Press Enter to continue...") -> None:
    console.print(f"\n[dim]{message}[/dim]")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass


class PipelineTimer:
    """Lightweight wrapper that prints timestamped pipeline events."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        self._t0 = time.perf_counter()

    def _elapsed(self) -> float:
        return time.perf_counter() - self._t0

    def log(self, msg: str) -> None:
        console.print(f"  [dim][{self._elapsed():5.1f}s][/dim] {msg}")


def _timed_generate(
    timer: PipelineTimer,
    engine: PlanEngine,
    store: InMemoryTraceStore,
    query: str,
    schemas: list[ToolSchema],
    strategy: str,
    static_traces: list[ExecutionTrace] | None = None,
) -> tuple[Plan, list[tuple[ExecutionTrace, float]]]:
    """Run plan generation with pipeline event logging.

    Returns (plan, retrieved_traces_with_scores).
    """
    timer.reset()
    retrieved_with_scores: list[tuple[ExecutionTrace, float]] = []

    if strategy == "zero_shot":
        timer.log("Strategy: [yellow]zero-shot[/yellow] (no memory)")
        prompt = build_prompt(query=query, traces=[], tool_schemas=schemas)
        token_count = count_tokens(SYSTEM_PROMPT) + count_tokens(prompt)
        timer.log(f"Assembling prompt ({token_count:,} tokens, 0 examples)")
        timer.log("Calling model...")
        plan = engine.generate_zero_shot(query, schemas)
        timer.log(f"Plan parsed ({len(plan.steps)} steps)")
        return plan, []

    if strategy == "static":
        traces = static_traces or []
        timer.log(f"Strategy: [blue]static few-shot[/blue] ({len(traces)} fixed examples)")
        prompt = build_prompt(query=query, traces=traces, tool_schemas=schemas)
        token_count = count_tokens(SYSTEM_PROMPT) + count_tokens(prompt)
        timer.log(f"Assembling prompt ({token_count:,} tokens)")
        timer.log("Calling model...")
        plan = engine.generate_static_few_shot(query, schemas, traces)
        timer.log(f"Plan parsed ({len(plan.steps)} steps)")
        return plan, []

    # Dynamic retrieval
    timer.log("Strategy: [green]dynamic retrieval[/green] (behavioral memory)")
    timer.log("Embedding query...")
    retrieved_with_scores = store.search(query, k=6)
    scores_str = ", ".join(f"{s:.2f}" for _, s in retrieved_with_scores[:5])
    timer.log(f"Retrieved {len(retrieved_with_scores)} candidates (scores: {scores_str})")

    settings = engine._settings
    selected = select_traces_within_budget(
        store=store, query=query, tool_schemas=schemas, settings=settings,
    )
    timer.log(f"Selected {len(selected)} traces within token budget")

    prompt = build_prompt(query=query, traces=selected, tool_schemas=schemas)
    token_count = count_tokens(SYSTEM_PROMPT) + count_tokens(prompt)
    timer.log(f"Assembling prompt ({token_count:,} / {settings.max_prompt_tokens:,} tokens)")
    timer.log("Calling model...")
    plan = engine.generate(query=query, tool_schemas=schemas, traces=selected)
    timer.log(f"Plan parsed ({len(plan.steps)} steps)")
    return plan, retrieved_with_scores


# ---------------------------------------------------------------------------
# Act 1 — Memory Inspector
# ---------------------------------------------------------------------------

def act_memory_inspector(store: InMemoryTraceStore) -> None:
    console.print(Panel.fit(
        "[bold]Act 1: Memory Inspector[/bold]\n\n"
        "What's inside behavioral memory? Every trace the agent can\n"
        "retrieve at query time — task descriptions, validated tool\n"
        "chains, provenance, and similarity neighborhoods.",
        title="ACT 1",
    ))

    traces = store._traces
    table = Table(title=f"Behavioral Memory Store ({len(traces)} traces)")
    table.add_column("#", style="dim", width=3)
    table.add_column("Task Description", max_width=52)
    table.add_column("Steps", justify="center", width=5)
    table.add_column("Tool Chain", style="cyan")
    table.add_column("Source", justify="center", width=8)

    for i, trace in enumerate(traces, 1):
        table.add_row(
            str(i),
            trace.task_description[:50] + ("..." if len(trace.task_description) > 50 else ""),
            str(len(trace.tool_chain)),
            _tool_chain_str(trace.tool_chain),
            f"[green]{trace.source}[/green]" if trace.validated else trace.source,
        )

    console.print(table)

    # Show embedding neighborhood for one trace
    if len(traces) >= 2:
        probe = traces[0]
        console.print(f"\n[bold]Embedding neighborhood[/bold] for: [italic]\"{probe.task_description[:60]}\"[/italic]")
        neighbors = store.search(probe.task_description, k=4)
        for rank, (t, score) in enumerate(neighbors):
            if t.task_description == probe.task_description:
                continue
            bar_len = int(score * 30)
            bar = "[green]" + "█" * bar_len + "[/green]" + "░" * (30 - bar_len)
            console.print(f"  {bar}  {score:.3f}  {t.task_description[:55]}")


# ---------------------------------------------------------------------------
# Act 2 — Side-by-Side Compare
# ---------------------------------------------------------------------------

def _print_plan_steps(steps: list[ToolCall], color: str) -> None:
    for step in steps:
        params_short = ", ".join(f"{k}={repr(v)[:30]}" for k, v in list(step.parameters.items())[:3])
        console.print(f"  [{color}]{step.step_id}[/{color}]: {step.tool_name}({params_short})")


def _print_diff(zs_steps: list[ToolCall], dyn_steps: list[ToolCall]) -> None:
    """Print a structural diff of tool chains: zero-shot vs dynamic."""
    zs_tools = [s.tool_name for s in zs_steps]
    dyn_tools = [s.tool_name for s in dyn_steps]

    diff = list(difflib.unified_diff(
        zs_tools, dyn_tools,
        fromfile="zero-shot", tofile="dynamic",
        lineterm="",
    ))

    if not diff:
        console.print("  [dim]Plans are identical — no diff to show.[/dim]")
        return

    console.print()
    for line in diff:
        if line.startswith("---"):
            console.print(f"  [bold red]{line}[/bold red]")
        elif line.startswith("+++"):
            console.print(f"  [bold green]{line}[/bold green]")
        elif line.startswith("@@"):
            console.print(f"  [dim]{line}[/dim]")
        elif line.startswith("-"):
            console.print(f"  [red]{line}  ← removed/changed[/red]")
        elif line.startswith("+"):
            console.print(f"  [green]{line}  ← added/changed[/green]")
        else:
            console.print(f"  [dim]{line}[/dim]")


def act_compare(
    engine: PlanEngine,
    store: InMemoryTraceStore,
    schemas: list[ToolSchema],
    seed_traces: list[ExecutionTrace],
    query: str,
) -> None:
    console.print(Panel.fit(
        "[bold]Act 2: Side-by-Side Strategy Comparison[/bold]\n\n"
        "The same query is sent through three strategies:\n"
        "  [yellow]Zero-shot[/yellow]      — model sees only tool schemas, no examples\n"
        "  [blue]Static few-shot[/blue] — 3 fixed examples regardless of query\n"
        "  [green]Dynamic retrieval[/green] — semantically similar traces from memory\n\n"
        f"Query: [italic]{query}[/italic]",
        title="ACT 2",
    ))

    timer = PipelineTimer()

    # --- Zero-shot ---
    console.print("\n[bold yellow]━━━ ZERO-SHOT (no memory) ━━━[/bold yellow]")
    zs_plan, _ = _timed_generate(timer, engine, store, query, schemas, "zero_shot")
    _print_plan_steps(zs_plan.steps, "yellow")

    # --- Static few-shot ---
    console.print("\n[bold blue]━━━ STATIC FEW-SHOT (3 fixed examples) ━━━[/bold blue]")
    static_traces = seed_traces[:3]
    sf_plan, _ = _timed_generate(timer, engine, store, query, schemas, "static", static_traces)
    _print_plan_steps(sf_plan.steps, "blue")

    # --- Dynamic retrieval ---
    console.print("\n[bold green]━━━ DYNAMIC RETRIEVAL (behavioral memory) ━━━[/bold green]")
    dyn_plan, retrieved = _timed_generate(timer, engine, store, query, schemas, "dynamic")
    _print_plan_steps(dyn_plan.steps, "green")

    # --- Retrieved traces panel ---
    if retrieved:
        console.print("\n[bold]Retrieved traces (ranked by cosine similarity):[/bold]")
        for rank, (trace, score) in enumerate(retrieved[:5], 1):
            bar_len = int(score * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            console.print(f"  [green]{bar}[/green]  {score:.3f}  #{rank}")
            console.print(f"    [italic]{trace.task_description[:70]}[/italic]")
            console.print(f"    [dim]{_tool_chain_str(trace.tool_chain)}[/dim]")

    # --- Comparison table ---
    console.print()
    max_steps = max(len(zs_plan.steps), len(sf_plan.steps), len(dyn_plan.steps))
    comp_table = Table(title="Plan Comparison (tool sequence)")
    comp_table.add_column("Step", style="dim", width=6)
    comp_table.add_column("Zero-Shot", style="yellow")
    comp_table.add_column("Static Few-Shot", style="blue")
    comp_table.add_column("Dynamic (Proposed)", style="bold green")

    for i in range(max_steps):
        zs_tool = zs_plan.steps[i].tool_name if i < len(zs_plan.steps) else "—"
        sf_tool = sf_plan.steps[i].tool_name if i < len(sf_plan.steps) else "—"
        dy_tool = dyn_plan.steps[i].tool_name if i < len(dyn_plan.steps) else "—"
        comp_table.add_row(f"#{i+1}", zs_tool, sf_tool, dy_tool)

    console.print(comp_table)

    # --- Structural diff ---
    console.print("\n[bold]Structural diff (zero-shot vs dynamic):[/bold]")
    _print_diff(zs_plan.steps, dyn_plan.steps)


# ---------------------------------------------------------------------------
# Act 3 — Gatekeeper Challenge
# ---------------------------------------------------------------------------

def _run_gate_by_gate(
    trace: ExecutionTrace,
    gatekeeper: GatekeeperPipeline,
) -> GatekeeperResult:
    """Run a trace through the gatekeeper, printing each gate's result."""
    # Gate 1: Schema
    schema_valid, schema_failures = gatekeeper._schema_validator.validate(trace)
    if schema_valid:
        console.print("  Gate 1 (Schema Validation):   [green]PASS[/green]")
    else:
        console.print("  Gate 1 (Schema Validation):   [red]FAIL[/red]")
        for f in schema_failures:
            console.print(f"    [red]> {f}[/red]")
        console.print("  Gate 2 (Sandbox Execution):   [dim]SKIPPED[/dim]")
        console.print("  Gate 3 (Semantic Dedup):      [dim]SKIPPED[/dim]")
        console.print("  [bold red]VERDICT: REJECTED[/bold red]")
        return GatekeeperResult(
            accepted=False, schema_valid=False,
            rejection_reason="Schema validation failed", failures=schema_failures,
        )

    # Gate 2: Sandbox
    sandbox_passed, sandbox_detail = gatekeeper._sandbox.execute(trace)
    if sandbox_passed:
        console.print("  Gate 2 (Sandbox Execution):   [green]PASS[/green]")
    else:
        console.print("  Gate 2 (Sandbox Execution):   [red]FAIL[/red]")
        console.print(f"    [red]> {sandbox_detail}[/red]")
        console.print("  Gate 3 (Semantic Dedup):      [dim]SKIPPED[/dim]")
        console.print("  [bold red]VERDICT: REJECTED[/bold red]")
        return GatekeeperResult(
            accepted=False, schema_valid=True, sandbox_passed=False,
            rejection_reason=f"Sandbox check failed: {sandbox_detail}",
            failures=[sandbox_detail],
        )

    # Gate 3: Dedup
    is_unique, score = gatekeeper._dedup_gate.check(trace)
    if is_unique:
        console.print(f"  Gate 3 (Semantic Dedup):      [green]PASS[/green] (nearest: {score:.3f}, threshold: 0.95)")
    else:
        console.print(f"  Gate 3 (Semantic Dedup):      [red]FAIL[/red] (score: {score:.3f} >= 0.95)")
        console.print("  [bold red]VERDICT: REJECTED (duplicate)[/bold red]")
        return GatekeeperResult(
            accepted=False, schema_valid=True, sandbox_passed=True, is_duplicate=True,
            rejection_reason=f"Semantic duplicate (score={score:.3f})",
            failures=[f"Duplicate score {score:.3f} >= threshold"],
        )

    console.print("  [bold green]VERDICT: ADMITTED[/bold green]")
    return GatekeeperResult(
        accepted=True, schema_valid=True, sandbox_passed=True, is_duplicate=False,
    )


def act_gatekeeper(
    gatekeeper: GatekeeperPipeline,
) -> None:
    console.print(Panel.fit(
        "[bold]Act 3: Gatekeeper Challenge[/bold]\n\n"
        "The gatekeeper pipeline runs three validation gates before\n"
        "any trace enters memory:\n"
        "  1. Schema Validation  — tools exist, params valid, deps logical\n"
        "  2. Sandbox Execution  — dry-run data-flow check\n"
        "  3. Semantic Dedup     — cosine similarity < 0.95 to existing traces\n\n"
        "We feed it deliberately poisoned traces and watch what happens.",
        title="ACT 3",
    ))

    for i, raw in enumerate(POISONED_TRACES_RAW, 1):
        trace = _build_poisoned_trace(raw)
        poison_label = raw["poison_type"].replace("_", " ")

        console.print(f"\n[bold]Candidate #{i}[/bold] [dim]({poison_label})[/dim]")
        console.print(f"  Task:    [italic]{trace.task_description}[/italic]")
        console.print(f"  Chain:   [cyan]{_tool_chain_str(trace.tool_chain)}[/cyan]")
        console.print(f"  Defect:  [dim]{raw['explanation']}[/dim]")
        console.print()

        _run_gate_by_gate(trace, gatekeeper)

    # --- The honest moment ---
    console.print(Panel.fit(
        "[bold]The honest limitation (Section IV-F)[/bold]\n\n"
        "Notice that the 'wrong convention' traces PASS all three gates.\n"
        "They use real tools, valid parameters, and correct data flow.\n"
        "The gatekeeper checks [italic]structure[/italic], not [italic]domain semantics[/italic].\n\n"
        "A trace that computes revenue from total_amount instead of\n"
        "quantity*unit_price is structurally valid but semantically wrong.\n"
        "This is exactly the paper's honest finding — the gatekeeper\n"
        "catches broken traces, not subtly incorrect ones.\n\n"
        "That's why the [green]seed traces[/green] matter: they teach the\n"
        "correct conventions that the LLM learns to follow.",
        title="Limitation",
    ))


# ---------------------------------------------------------------------------
# Act 4 — Custom Query REPL
# ---------------------------------------------------------------------------

def act_custom_query(
    engine: PlanEngine,
    store: InMemoryTraceStore,
    schemas: list[ToolSchema],
    seed_traces: list[ExecutionTrace],
) -> None:
    console.print(Panel.fit(
        "[bold]Act 4: Custom Query (interactive)[/bold]\n\n"
        "Type any task and see how all three strategies handle it.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to end.",
        title="ACT 4",
    ))

    timer = PipelineTimer()
    static_traces = seed_traces[:3]

    while True:
        console.print()
        try:
            query = console.input("[bold]Query > [/bold]").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        console.print(f"\n[dim]Running 3 strategies for: \"{query}\"[/dim]\n")

        # Zero-shot
        console.print("[bold yellow]━━━ ZERO-SHOT ━━━[/bold yellow]")
        try:
            zs_plan, _ = _timed_generate(timer, engine, store, query, schemas, "zero_shot")
            _print_plan_steps(zs_plan.steps, "yellow")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            zs_plan = None

        # Static
        console.print("\n[bold blue]━━━ STATIC FEW-SHOT ━━━[/bold blue]")
        try:
            sf_plan, _ = _timed_generate(timer, engine, store, query, schemas, "static", static_traces)
            _print_plan_steps(sf_plan.steps, "blue")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            sf_plan = None

        # Dynamic
        console.print("\n[bold green]━━━ DYNAMIC RETRIEVAL ━━━[/bold green]")
        try:
            dyn_plan, retrieved = _timed_generate(timer, engine, store, query, schemas, "dynamic")
            _print_plan_steps(dyn_plan.steps, "green")
            if retrieved:
                console.print("\n  [dim]Top retrieved traces:[/dim]")
                for rank, (t, score) in enumerate(retrieved[:3], 1):
                    console.print(f"    {score:.3f}  {t.task_description[:60]}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            dyn_plan = None

        # Quick comparison table
        if zs_plan and sf_plan and dyn_plan:
            max_steps = max(len(zs_plan.steps), len(sf_plan.steps), len(dyn_plan.steps))
            console.print()
            t = Table(title="Comparison")
            t.add_column("Step", style="dim", width=5)
            t.add_column("Zero-Shot", style="yellow")
            t.add_column("Static", style="blue")
            t.add_column("Dynamic", style="bold green")
            for j in range(max_steps):
                zs = zs_plan.steps[j].tool_name if j < len(zs_plan.steps) else "—"
                sf = sf_plan.steps[j].tool_name if j < len(sf_plan.steps) else "—"
                dy = dyn_plan.steps[j].tool_name if j < len(dyn_plan.steps) else "—"
                t.add_row(f"#{j+1}", zs, sf, dy)
            console.print(t)

            console.print("\n[bold]Diff (zero-shot vs dynamic):[/bold]")
            _print_diff(zs_plan.steps, dyn_plan.steps)

    console.print("[dim]Exiting interactive mode.[/dim]")


# ---------------------------------------------------------------------------
# Main — initialization & menu
# ---------------------------------------------------------------------------

def _init_runtime(model_name: str | None = None) -> dict[str, Any]:
    """One-time initialization of LLM, embeddings, store, and tools."""
    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

    load_dotenv()

    model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    console.print(f"[dim]Initializing LLM: {model}[/dim]")
    llm = ChatGoogleGenerativeAI(model=model, temperature=0)

    console.print("[dim]Initializing embeddings: gemini-embedding-001[/dim]")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    settings = Settings()
    store = InMemoryTraceStore(embeddings=embeddings, settings=settings)

    console.print("[dim]Loading 7 tool schemas...[/dim]")
    schemas = get_tool_schemas()
    registry = ToolRegistry()
    registry.register_many(schemas)

    console.print("[dim]Seeding 12 validated traces...[/dim]")
    seed_traces = get_seed_traces()
    store.add_bulk(seed_traces)
    console.print(f"[dim]Store ready: {store.count()} traces embedded[/dim]")

    engine = PlanEngine(llm=llm, store=store, registry=registry, settings=settings)
    gatekeeper = GatekeeperPipeline(store=store, registry=registry, settings=settings)

    return {
        "llm": llm,
        "store": store,
        "schemas": schemas,
        "registry": registry,
        "seed_traces": seed_traces,
        "engine": engine,
        "gatekeeper": gatekeeper,
        "settings": settings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Behavioral Memory — Unified Demo Showcase",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python demo/showcase.py                         # all acts\n"
            "  python demo/showcase.py --act 2                 # just compare\n"
            '  python demo/showcase.py --act 2 --query "..."   # custom query\n'
            "  python demo/showcase.py --act 1 --act 3         # memory + gatekeeper\n"
        ),
    )
    parser.add_argument(
        "--act", type=int, action="append", choices=[1, 2, 3, 4],
        help="Run specific act(s). Omit to run all sequentially.",
    )
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY, help="Query for Act 2 compare")
    parser.add_argument("--model", type=str, default=None, help="LLM model name override")
    parser.add_argument("--no-pause", action="store_true", help="Skip pauses between acts")
    args = parser.parse_args()

    acts_to_run = sorted(args.act) if args.act else [1, 2, 3, 4]

    console.print(Panel.fit(
        "[bold]Behavioral Memory — Demo Showcase[/bold]\n\n"
        "\"Semantic Retrieval of Validated Execution Traces\n"
        " in MCP-Based Agent Systems\" (IEEE, 2025)\n\n"
        f"Acts: {', '.join(str(a) for a in acts_to_run)}",
        title="DEMO",
    ))

    # --- One-time initialization ---
    t0 = time.perf_counter()
    try:
        runtime = _init_runtime(args.model)
    except Exception as e:
        console.print(f"\n[bold red]Initialization failed:[/bold red] {e}")
        console.print("[dim]Make sure GOOGLE_API_KEY is set in your environment or .env file.[/dim]")
        sys.exit(1)
    init_time = time.perf_counter() - t0
    console.print(f"[dim]Initialization complete in {init_time:.1f}s[/dim]\n")

    # --- Run acts ---
    for act_num in acts_to_run:
        if act_num == 1:
            act_memory_inspector(runtime["store"])
        elif act_num == 2:
            act_compare(
                runtime["engine"], runtime["store"], runtime["schemas"],
                runtime["seed_traces"], args.query,
            )
        elif act_num == 3:
            act_gatekeeper(runtime["gatekeeper"])
        elif act_num == 4:
            act_custom_query(
                runtime["engine"], runtime["store"], runtime["schemas"],
                runtime["seed_traces"],
            )

        if not args.no_pause and act_num != acts_to_run[-1] and act_num != 4:
            _pause()

    console.print(Panel.fit("[bold]Demo complete.[/bold]", title="END"))


if __name__ == "__main__":
    main()
