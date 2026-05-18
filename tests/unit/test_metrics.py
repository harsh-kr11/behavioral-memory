"""Tests for evaluation metrics."""

from __future__ import annotations

from behavioral_memory.evaluation.metrics import (
    compute_metrics,
    execution_sequence_accuracy,
    parameter_validity,
    plan_correctness,
    tool_selection_accuracy,
)


class TestToolSelectionAccuracy:
    def test_exact_match(self):
        assert tool_selection_accuracy(["a", "b"], ["a", "b"])

    def test_different_order_still_matches(self):
        assert tool_selection_accuracy(["b", "a"], ["a", "b"])

    def test_missing_tool(self):
        assert not tool_selection_accuracy(["a"], ["a", "b"])

    def test_extra_tool(self):
        assert not tool_selection_accuracy(["a", "b", "c"], ["a", "b"])


class TestParameterValidity:
    def test_perfect_match(self):
        gold = [{"params": {"query": "SELECT 1"}}]
        pred = [{"params": {"query": "SELECT 1"}}]
        assert parameter_validity(pred, gold) == 1.0

    def test_partial_match(self):
        gold = [{"params": {"query": "SELECT 1", "timeout": 30}}]
        pred = [{"params": {"query": "SELECT 1"}}]
        assert parameter_validity(pred, gold) == 0.5

    def test_empty_gold(self):
        assert parameter_validity([], []) == 1.0


class TestExecutionSequence:
    def test_correct_order(self):
        assert execution_sequence_accuracy(["a", "b", "c"], ["a", "b", "c"])

    def test_wrong_order(self):
        assert not execution_sequence_accuracy(["b", "a", "c"], ["a", "b", "c"])


class TestPlanCorrectness:
    def test_correct_plan(self):
        assert plan_correctness(["a", "b"], ["a", "b"], [{"params": {"x": 1}}], [{"params": {"x": 1}}])

    def test_wrong_tools(self):
        assert not plan_correctness(["a"], ["a", "b"], [], [])


class TestComputeMetrics:
    def test_full_computation(self):
        gold = [{"tool": "query_database", "params": {"query": "SELECT 1"}}]
        pred = [{"tool": "query_database", "params": {"query": "SELECT 1"}}]
        result = compute_metrics(pred, gold)
        assert result["tsa"] is True
        assert result["pcr"] is True
        assert result["esa"] is True
        assert result["pv"] == 1.0
