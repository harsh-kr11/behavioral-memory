"""Compare benchmark results across multiple LLM models.

Loads JSON results files from run_live_benchmark.py and produces:
  1. A Rich side-by-side table in the terminal
  2. A markdown-formatted table (copy-paste into README)
  3. Cross-model McNemar's tests on PCR
  4. A merged multi_model_results.json for archival

Usage:
    python examples/compare_models.py \\
        --results results_25pro.json results_3flash.json results_35flash.json

    python examples/compare_models.py \\
        --results results_25pro.json results_3flash.json \\
        --output multi_model_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from behavioral_memory.evaluation.statistics import mcnemar_test

console = Console()

METRIC_LABELS = {
    "tsa": "Tool Selection (TSA)",
    "pv": "Parameter Validity (PV)",
    "pcr": "Plan Correctness (PCR)",
    "esa": "Sequence Accuracy (ESA)",
}

STRATEGY_KEYS = [
    ("zero_shot", "Zero-Shot"),
    ("static_few_shot", "Static Few-Shot"),
    ("dynamic_retrieval", "Dynamic (Proposed)"),
]


def load_results(paths: list[str]) -> list[dict]:
    results = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            console.print(f"[red]File not found: {p}[/red]")
            sys.exit(1)
        with open(path) as f:
            data = json.load(f)
        if "model" not in data:
            data["model"] = path.stem
        results.append(data)
    return results


def fmt_pct(value: float) -> str:
    return f"{value:.1%}"


def print_rich_table(all_results: list[dict]) -> None:
    """Print a Rich table comparing all models side-by-side."""
    table = Table(
        title=f"Multi-Model Benchmark Comparison (N={all_results[0]['n_tasks']})",
        show_lines=True,
    )
    table.add_column("Metric", style="bold")
    table.add_column("Strategy")
    for r in all_results:
        table.add_column(r["model"], justify="right")

    for metric_key, metric_label in METRIC_LABELS.items():
        for strat_key, strat_label in STRATEGY_KEYS:
            style = "bold green" if strat_key == "dynamic_retrieval" else ""
            row = [metric_label if strat_key == STRATEGY_KEYS[0][0] else "", strat_label]
            for r in all_results:
                agg = r[strat_key]["aggregate"][metric_key]
                mean = agg["mean"] if isinstance(agg, dict) else agg
                row.append(fmt_pct(mean))
            table.add_row(*row, style=style)

    console.print()
    console.print(table)


def print_mcnemar_comparisons(all_results: list[dict]) -> None:
    """Print McNemar's test results for each model and cross-model."""
    console.print("\n[bold]McNemar's Test — Zero-Shot vs Dynamic (per model):[/bold]")

    mcnemar_table = Table(title="Statistical Significance")
    mcnemar_table.add_column("Model", style="bold")
    mcnemar_table.add_column("p-value", justify="right")
    mcnemar_table.add_column("Significant (p<0.05)?", justify="center")

    for r in all_results:
        zs_pcr = [t["metrics"]["pcr"] for t in r["zero_shot"]["per_task"]]
        dyn_pcr = [t["metrics"]["pcr"] for t in r["dynamic_retrieval"]["per_task"]]
        result = mcnemar_test(zs_pcr, dyn_pcr)
        p = result["p_value"]
        sig = "[green]Yes[/green]" if p < 0.05 else "[yellow]No[/yellow]"
        mcnemar_table.add_row(r["model"], f"{p:.4f}", sig)

    console.print(mcnemar_table)


def print_difficulty_breakdown(all_results: list[dict]) -> None:
    """Print PCR breakdown by difficulty across models."""
    from behavioral_memory.evaluation.benchmark import BenchmarkRunner

    diff_table = Table(title="Plan Correctness (PCR) by Difficulty — Dynamic Retrieval", show_lines=True)
    diff_table.add_column("Difficulty", style="bold")
    diff_table.add_column("n", justify="right")
    for r in all_results:
        diff_table.add_column(r["model"], justify="right", style="bold green")

    for diff in ["simple", "moderate", "challenging"]:
        row: list[str] = [diff]
        n_str = ""
        for r in all_results:
            by_diff = BenchmarkRunner.results_by_difficulty(r["dynamic_retrieval"])
            d = by_diff.get(diff, {})
            n_str = str(d.get("n", 0))
            row.append(fmt_pct(d.get("pcr", 0)))
        row.insert(1, n_str)
        diff_table.add_row(*row)

    console.print()
    console.print(diff_table)


def generate_markdown(all_results: list[dict]) -> str:
    """Generate a markdown table for the README."""
    lines = []
    models = [r["model"] for r in all_results]
    n = all_results[0]["n_tasks"]

    lines.append(f"On a {n}-task benchmark with 7 MCP tools (temperature 0, embeddings: `gemini-embedding-001`):")
    lines.append("")

    header = "| Metric | Strategy | " + " | ".join(f"**{m}**" for m in models) + " |"
    sep = "|--------|----------|" + "|".join("-" * (len(m) + 6) for m in models) + "|"
    lines.append(header)
    lines.append(sep)

    for metric_key, metric_label in METRIC_LABELS.items():
        for i, (strat_key, strat_label) in enumerate(STRATEGY_KEYS):
            label = metric_label if i == 0 else ""
            is_dynamic = strat_key == "dynamic_retrieval"
            cells = []
            for r in all_results:
                agg = r[strat_key]["aggregate"][metric_key]
                mean = agg["mean"] if isinstance(agg, dict) else agg
                val = fmt_pct(mean)
                cells.append(f"**{val}**" if is_dynamic else val)

            strat_display = f"**{strat_label}**" if is_dynamic else strat_label
            row = f"| {label} | {strat_display} | " + " | ".join(cells) + " |"
            lines.append(row)

    lines.append("")
    lines.append("**McNemar's test (Zero-Shot vs Dynamic):**")
    lines.append("")
    lines.append("| Model | p-value | Significant? |")
    lines.append("|-------|---------|-------------|")

    for r in all_results:
        zs_pcr = [t["metrics"]["pcr"] for t in r["zero_shot"]["per_task"]]
        dyn_pcr = [t["metrics"]["pcr"] for t in r["dynamic_retrieval"]["per_task"]]
        result = mcnemar_test(zs_pcr, dyn_pcr)
        p = result["p_value"]
        sig = "Yes" if p < 0.05 else "No"
        lines.append(f"| {r['model']} | p = {p:.4f} | {sig} |")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare benchmark results across models")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="Paths to benchmark result JSON files (from run_live_benchmark.py)",
    )
    parser.add_argument(
        "--output",
        default="multi_model_results.json",
        help="Output path for merged comparison JSON",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown-formatted table for README",
    )
    args = parser.parse_args()

    all_results = load_results(args.results)

    console.print(
        Panel.fit(
            "[bold]Multi-Model Benchmark Comparison[/bold]\n\n"
            f"Models: {', '.join(r['model'] for r in all_results)}\n"
            f"Tasks: {all_results[0]['n_tasks']}",
            title="Comparison",
        )
    )

    print_rich_table(all_results)
    print_mcnemar_comparisons(all_results)
    print_difficulty_breakdown(all_results)

    md = generate_markdown(all_results)

    if args.markdown:
        console.print("\n[bold]Markdown for README:[/bold]\n")
        console.print(md)

    merged = {
        "models": [r["model"] for r in all_results],
        "n_tasks": all_results[0]["n_tasks"],
        "per_model": {r["model"]: r for r in all_results},
        "markdown_table": md,
    }

    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, default=str)
    console.print(f"\n[dim]Merged results saved to {args.output}[/dim]")

    console.print("\n[bold]Copy the markdown table into README.md:[/bold]")
    console.print(f"  python examples/compare_models.py --results {' '.join(args.results)} --markdown")


if __name__ == "__main__":
    main()
