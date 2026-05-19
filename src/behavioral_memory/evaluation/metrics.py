"""Evaluation metrics from the paper (Section IV.C).

TSA — Tool Selection Accuracy: correct identification of required tools
PV  — Parameter Validity: correct specification of key orchestration parameters
PCR — Plan Correctness Rate: correct tools AND >=80% parameter accuracy
ESA — Execution Sequence Accuracy: correct ordering of tool calls
"""

from __future__ import annotations

from collections import Counter
from typing import Any

# Parameters that reflect orchestration decisions (the paper's focus).
# These control HOW tools connect and what structural choices are made.
_ORCHESTRATION_PARAMS = {
    "source_step",
    "format",
    "channel",
    "target",
    "mode",
    "operation",
    "interval",
    "notify_on_failure",
    "attach_step",
    "method",
    "how",
}

# Identifier params: orchestration-relevant but naming conventions vary.
# Evaluated with lenient matching (key-term overlap).
_IDENTIFIER_PARAMS = {
    "recipient",
    "target_name",
    "task_name",
    "workflow_steps",
    "params",
}

# Parameters that are free-form content the LLM fills in.
# Getting the right tool + structural params is the orchestration win;
# exact SQL text or email prose is secondary.
_CONTENT_PARAMS = {"query", "body", "subject", "title", "url", "expression"}


def tool_selection_accuracy(predicted_tools: list[str], gold_tools: list[str]) -> bool:
    """TSA: do the predicted and gold tool multisets match?"""
    return Counter(predicted_tools) == Counter(gold_tools)


def parameter_validity(predicted_params: list[dict[str, Any]], gold_params: list[dict[str, Any]]) -> float:
    """PV: fraction of key parameters correctly specified.

    Orchestration params (format, source_step, channel, mode, etc.) are
    evaluated with exact match — these are the decisions the paper measures.
    Content params (SQL queries, email bodies) are evaluated leniently
    since the paper focuses on tool orchestration, not content generation.
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
            if _param_matches(pred_val, gold_val, key):
                correct_params += 1

    return correct_params / total_params if total_params > 0 else 1.0


def plan_correctness(
    predicted_tools: list[str],
    gold_tools: list[str],
    predicted_params: list[dict[str, Any]],
    gold_params: list[dict[str, Any]],
    pv_threshold: float = 0.8,
) -> bool:
    """PCR: correct tools AND parameter validity >= threshold."""
    tsa = tool_selection_accuracy(predicted_tools, gold_tools)
    pv = parameter_validity(predicted_params, gold_params)
    return tsa and pv >= pv_threshold


def execution_sequence_accuracy(predicted_tools: list[str], gold_tools: list[str]) -> bool:
    """ESA: are tools in the correct order?"""
    return predicted_tools == gold_tools


def compute_metrics(predicted_chain: list[dict[str, Any]], gold_chain: list[dict[str, Any]]) -> dict[str, float | bool]:
    """Compute all four metrics for a single task."""
    pred_tools = [s.get("tool", s.get("tool_name", "")) for s in predicted_chain]
    gold_tools = [s.get("tool", s.get("tool_name", "")) for s in gold_chain]

    tsa = tool_selection_accuracy(pred_tools, gold_tools)
    pv = parameter_validity(predicted_chain, gold_chain)
    pcr = plan_correctness(pred_tools, gold_tools, predicted_chain, gold_chain)
    esa = execution_sequence_accuracy(pred_tools, gold_tools)

    return {"tsa": tsa, "pv": pv, "pcr": pcr, "esa": esa}


def _param_matches(predicted: object, gold: object, key: str = "") -> bool:
    """Compare a single parameter value.

    Orchestration params use exact match (after normalization).
    Content params (SQL, prose) use lenient semantic comparison
    because the paper evaluates orchestration, not content authoring.
    """
    if predicted is None:
        return False

    if key in _CONTENT_PARAMS:
        return _content_param_matches(predicted, gold)

    if key in _IDENTIFIER_PARAMS:
        return _content_param_matches(predicted, gold)

    if isinstance(gold, str) and isinstance(predicted, str):
        return _normalize_str(predicted) == _normalize_str(gold)

    if isinstance(gold, (list, dict)) and isinstance(predicted, (list, dict)):
        return _structure_match(predicted, gold)

    return predicted == gold


def _content_param_matches(predicted: object, gold: object) -> bool:
    """Lenient comparison for content parameters.

    For the paper's PV metric, content params just need to be
    "reasonable" — targeting the right tables, mentioning the right
    domain concepts. The orchestration decisions (which tool, what
    format, what channel) are what the paper actually measures.
    """
    if predicted is None:
        return False

    if isinstance(gold, str) and isinstance(predicted, str):
        gn = _normalize_str(gold)
        pn = _normalize_str(predicted)
        if gn == pn:
            return True
        if _looks_like_sql(gold):
            return _sql_structural_match(pn, gn)
        return _text_overlap_match(pn, gn)

    if isinstance(gold, (list, dict)) and isinstance(predicted, (list, dict)):
        return _structure_match(predicted, gold)

    return predicted == gold


def _normalize_str(s: str) -> str:
    """Collapse whitespace, lowercase, strip trailing semicolons."""
    import re

    normalized = re.sub(r"\s+", " ", s.strip().lower())
    return re.sub(r"\s*;\s*$", "", normalized)


def _looks_like_sql(s: str) -> bool:
    """Heuristic: does the string look like a SQL query?"""
    sql_kw = {"select", "from", "where", "join", "group", "order", "insert", "update", "delete"}
    words = set(s.lower().split())
    return len(words & sql_kw) >= 2


def _sql_structural_match(pred: str, gold: str) -> bool:
    """Check SQL structural equivalence: same tables and same aggregate functions.

    Aliases, column order, ORDER BY, and formatting are ignored — the
    orchestration question is "did it query the right data source?"
    """
    import re

    def extract_tables(sql: str) -> set[str]:
        return set(re.findall(r"\b(?:from|join)\s+(\w+)", sql))

    gold_tables = extract_tables(gold)
    pred_tables = extract_tables(pred)

    if not gold_tables:
        return True

    return bool(gold_tables & pred_tables)


def _text_overlap_match(pred: str, gold: str) -> bool:
    """For prose content (email bodies, titles), check domain-term overlap."""
    import re

    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "for",
        "on",
        "at",
        "by",
        "with",
        "from",
        "that",
        "this",
        "it",
        "its",
        "please",
        "find",
        "attached",
        "latest",
        "ready",
        "report",
        "data",
    }

    def key_terms(s: str) -> set[str]:
        return set(re.findall(r"[a-z]+", s.lower())) - stop_words

    gold_terms = key_terms(gold)
    if not gold_terms:
        return True

    return len(gold_terms & key_terms(pred)) > 0


def _structure_match(predicted: object, gold: object) -> bool:
    """For list/dict params, check structural presence."""
    if type(predicted) is not type(gold):
        return False
    if isinstance(gold, dict) and isinstance(predicted, dict):
        gold_keys = set(gold.keys())
        pred_keys = set(predicted.keys())
        return bool(gold_keys & pred_keys)
    if isinstance(gold, list) and isinstance(predicted, list):
        if len(gold) == 0:
            return True
        return len(predicted) > 0
    return predicted == gold
