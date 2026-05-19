"""Benchmark runner for reproducing paper results.

Runs the 30-task evaluation across zero-shot, static few-shot, and
dynamic retrieval strategies, computing all four metrics per task.
"""

from __future__ import annotations

import logging
from typing import Any

from behavioral_memory.core.schemas import Plan, ToolSchema
from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS
from behavioral_memory.evaluation.metrics import compute_metrics
from behavioral_memory.evaluation.statistics import bootstrap_ci, mcnemar_test

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs the paper's 30-task benchmark and computes metrics."""

    def __init__(self, tool_schemas: list[ToolSchema]) -> None:
        self._schemas = tool_schemas

    def evaluate_plan(self, plan: Plan, gold_chain: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate a single plan against its gold tool chain."""
        predicted_chain = [
            {
                "tool": step.tool_name,
                "params": step.parameters,
            }
            for step in plan.steps
        ]
        return compute_metrics(predicted_chain, gold_chain)

    def run(
        self,
        strategy: Any,
        tasks: list[dict[str, Any]] | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Run a strategy across all (or a subset of) evaluation tasks.

        Returns aggregate metrics and per-task breakdown.
        """
        eval_tasks = tasks or EVALUATION_TASKS
        if limit:
            eval_tasks = eval_tasks[:limit]

        per_task: list[dict[str, Any]] = []
        tsa_results: list[bool] = []
        pcr_results: list[bool] = []
        esa_results: list[bool] = []
        pv_values: list[float] = []

        for task in eval_tasks:
            try:
                plan = strategy.generate(task["task"], self._schemas)
                metrics = self.evaluate_plan(plan, task["gold_tool_chain"])

                tsa_results.append(bool(metrics["tsa"]))
                pv_values.append(float(metrics["pv"]))
                pcr_results.append(bool(metrics["pcr"]))
                esa_results.append(bool(metrics["esa"]))

                per_task.append(
                    {
                        "task_id": task["task_id"],
                        "task": task["task"],
                        "difficulty": task["difficulty"],
                        "predicted_steps": [s.model_dump() for s in plan.steps],
                        "metrics": metrics,
                    }
                )
            except Exception as e:
                logger.warning("Task %d failed: %s", task["task_id"], e)
                tsa_results.append(False)
                pv_values.append(0.0)
                pcr_results.append(False)
                esa_results.append(False)
                per_task.append(
                    {
                        "task_id": task["task_id"],
                        "task": task["task"],
                        "difficulty": task["difficulty"],
                        "error": str(e),
                        "metrics": {"tsa": False, "pv": 0.0, "pcr": False, "esa": False},
                    }
                )

        n = len(eval_tasks)
        tsa_mean, tsa_lo, tsa_hi = bootstrap_ci(tsa_results)
        pcr_mean, pcr_lo, pcr_hi = bootstrap_ci(pcr_results)
        esa_mean, esa_lo, esa_hi = bootstrap_ci(esa_results)

        return {
            "n_tasks": n,
            "aggregate": {
                "tsa": {"mean": tsa_mean, "ci_95": [tsa_lo, tsa_hi]},
                "pv": {"mean": sum(pv_values) / n if n > 0 else 0.0},
                "pcr": {"mean": pcr_mean, "ci_95": [pcr_lo, pcr_hi]},
                "esa": {"mean": esa_mean, "ci_95": [esa_lo, esa_hi]},
            },
            "per_task": per_task,
        }

    @staticmethod
    def compare(
        results_a: dict[str, Any],
        results_b: dict[str, Any],
        label_a: str = "Method A",
        label_b: str = "Method B",
    ) -> dict[str, Any]:
        """Compare two benchmark runs using McNemar's test on PCR."""
        pcr_a = [t["metrics"]["pcr"] for t in results_a["per_task"]]
        pcr_b = [t["metrics"]["pcr"] for t in results_b["per_task"]]
        test_result = mcnemar_test(pcr_a, pcr_b)

        return {
            "comparison": f"{label_a} vs {label_b}",
            "mcnemar_pcr": test_result,
            label_a: results_a["aggregate"],
            label_b: results_b["aggregate"],
        }

    @staticmethod
    def results_by_difficulty(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Break down results by difficulty tier."""
        by_diff: dict[str, list[dict[str, Any]]] = {}
        for task in results["per_task"]:
            diff = task["difficulty"]
            by_diff.setdefault(diff, []).append(task)

        summary: dict[str, dict[str, Any]] = {}
        for diff, tasks in by_diff.items():
            pcr_list = [bool(t["metrics"]["pcr"]) for t in tasks]
            esa_list = [bool(t["metrics"]["esa"]) for t in tasks]
            summary[diff] = {
                "n": len(tasks),
                "pcr": sum(pcr_list) / len(pcr_list) if pcr_list else 0.0,
                "esa": sum(esa_list) / len(esa_list) if esa_list else 0.0,
            }
        return summary
