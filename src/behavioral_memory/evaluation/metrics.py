"""Evaluation metrics from the paper (Section IV.C).

  TSA — Tool Selection Accuracy: correct identification of required tools
  PV  — Parameter Validity: correct specification of key parameters
  PCR — Plan Correctness Rate: correct tools AND >=80% parameter accuracy
  ESA — Execution Sequence Accuracy: correct ordering of tool calls
"""

from __future__ import annotations

from collections import Counter


def tool_selection_accuracy(predicted_tools: list[str], gold_tools: list[str]) -> bool:
    """TSA: do the predicted and gold tool multisets match?"""
    return Counter(predicted_tools) == Counter(gold_tools)


def parameter_validity(
    predicted_params: list[dict], gold_params: list[dict]
) -> float:
    """PV: fraction of key parameters correctly specified.

    Compares parameter keys step-by-step. If step counts differ, missing
    steps count as 0% accuracy for those steps.
    """
    if not gold_params:
        return 1.0

    total_params = 0
    correct_params = 0

    for i, gold_step in enumerate(gold_params):
        gold_p = gold_step.get("params", {})
        pred_p = predicted_params[i].get("params", {}) if i < len(predicted_params) else {}

        for key, gold_val in gold_p.items():
            total_params += 1
            pred_val = pred_p.get(key)
            if _param_matches(pred_val, gold_val):
                correct_params += 1

    return correct_params / total_params if total_params > 0 else 1.0


def plan_correctness(
    predicted_tools: list[str],
    gold_tools: list[str],
    predicted_params: list[dict],
    gold_params: list[dict],
    pv_threshold: float = 0.8,
) -> bool:
    """PCR: correct tools AND parameter validity >= threshold."""
    tsa = tool_selection_accuracy(predicted_tools, gold_tools)
    pv = parameter_validity(predicted_params, gold_params)
    return tsa and pv >= pv_threshold


def execution_sequence_accuracy(
    predicted_tools: list[str], gold_tools: list[str]
) -> bool:
    """ESA: are tools in the correct order?"""
    return predicted_tools == gold_tools


def compute_metrics(
    predicted_chain: list[dict], gold_chain: list[dict]
) -> dict[str, float | bool]:
    """Compute all four metrics for a single task."""
    pred_tools = [s.get("tool", s.get("tool_name", "")) for s in predicted_chain]
    gold_tools = [s.get("tool", s.get("tool_name", "")) for s in gold_chain]

    tsa = tool_selection_accuracy(pred_tools, gold_tools)
    pv = parameter_validity(predicted_chain, gold_chain)
    pcr = plan_correctness(pred_tools, gold_tools, predicted_chain, gold_chain)
    esa = execution_sequence_accuracy(pred_tools, gold_tools)

    return {"tsa": tsa, "pv": pv, "pcr": pcr, "esa": esa}


def _param_matches(predicted: object, gold: object) -> bool:
    """Flexible parameter comparison."""
    if predicted is None:
        return False
    if isinstance(gold, str) and isinstance(predicted, str):
        return _normalize_sql(predicted) == _normalize_sql(gold)
    return predicted == gold


def _normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison (collapse whitespace, lowercase)."""
    import re

    normalized = re.sub(r"\s+", " ", sql.strip().lower())
    normalized = re.sub(r"\s*;\s*$", "", normalized)
    return normalized
