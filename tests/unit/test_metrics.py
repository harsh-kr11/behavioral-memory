"""Tests for evaluation metrics."""

from __future__ import annotations

from behavioral_memory.evaluation.metrics import (
    _content_param_matches,
    _looks_like_sql,
    _normalize_str,
    _param_matches,
    _sql_structural_match,
    _structure_match,
    _text_overlap_match,
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


# ---------- Lenient PV matching (content / identifier / orchestration) ----------


class TestParamMatchesRouting:
    """Verify _param_matches routes to the right comparison by key."""

    def test_content_param_uses_lenient_match(self):
        assert _param_matches("SELECT * FROM orders", "SELECT * FROM orders WHERE id=1", key="query")

    def test_orchestration_param_uses_exact_match(self):
        assert _param_matches("csv", "csv", key="format")
        assert not _param_matches("json", "csv", key="format")

    def test_identifier_param_uses_lenient_match(self):
        assert _param_matches("#data-alerts", "#data-alerts", key="recipient")
        assert _param_matches("ops-team", "ops team alert", key="recipient")

    def test_none_predicted_always_false(self):
        assert not _param_matches(None, "any", key="format")
        assert not _param_matches(None, "SELECT 1", key="query")

    def test_unlisted_param_uses_exact_match(self):
        assert _param_matches("30", "30", key="timeout")
        assert not _param_matches("60", "30", key="timeout")


class TestSqlStructuralMatch:
    """SQL matching should check table overlap, not exact string equality."""

    def test_same_tables_different_aliases(self):
        gold = "select sum(quantity * unit_price) as revenue from order_items"
        pred = "select sum(oi.quantity * oi.unit_price) as rev from order_items oi"
        assert _sql_structural_match(_normalize_str(pred), _normalize_str(gold))

    def test_shared_table_with_extra_join(self):
        gold = "select * from order_items"
        pred = "select * from order_items join orders on order_items.order_id = orders.id"
        assert _sql_structural_match(_normalize_str(pred), _normalize_str(gold))

    def test_completely_different_tables_fails(self):
        gold = "select * from order_items"
        pred = "select * from customers"
        assert not _sql_structural_match(_normalize_str(pred), _normalize_str(gold))

    def test_empty_gold_tables_returns_true(self):
        assert _sql_structural_match("select 1", "select count(*)")

    def test_gold_table_not_in_pred_fails(self):
        gold = "select * from order_items join products on true"
        pred = "select * from orders"
        assert not _sql_structural_match(_normalize_str(pred), _normalize_str(gold))


class TestTextOverlapMatch:
    """Text matching should check domain-term overlap, ignoring stop words."""

    def test_same_domain_terms(self):
        assert _text_overlap_match("deployment completed successfully", "deployment update")

    def test_no_overlapping_terms(self):
        assert not _text_overlap_match("hello world", "quarterly revenue")

    def test_stop_words_only_gold_is_vacuously_true(self):
        # Gold with only stop words has no key terms → vacuously true
        assert _text_overlap_match("hello world", "the is a an")

    def test_stop_words_filtered_from_matching(self):
        # "quarterly" is a key term in gold; "monthly" doesn't match it
        assert not _text_overlap_match("monthly the is", "quarterly revenue")

    def test_empty_gold_returns_true(self):
        assert _text_overlap_match("anything", "the a is")


class TestStructureMatch:
    """Dict/list structural matching."""

    def test_dict_with_shared_keys(self):
        assert _structure_match({"a": 1, "b": 2}, {"a": 10, "c": 3})

    def test_dict_no_shared_keys(self):
        assert not _structure_match({"x": 1}, {"y": 2})

    def test_list_nonempty_if_gold_nonempty(self):
        assert _structure_match([1], [10, 20])

    def test_empty_pred_list_fails(self):
        assert not _structure_match([], [1, 2])

    def test_empty_gold_list_passes(self):
        assert _structure_match([], [])

    def test_type_mismatch_fails(self):
        assert not _structure_match({"a": 1}, [1, 2])


class TestLooksLikeSql:
    def test_sql_detected(self):
        assert _looks_like_sql("SELECT * FROM customers WHERE id = 1")

    def test_non_sql_not_detected(self):
        assert not _looks_like_sql("Deployment completed successfully")

    def test_needs_two_keywords(self):
        assert not _looks_like_sql("select all items")


class TestContentParamMatches:
    """End-to-end content param matching."""

    def test_exact_match_short_circuits(self):
        assert _content_param_matches("hello", "hello")

    def test_sql_uses_structural_match(self):
        assert _content_param_matches(
            "SELECT SUM(qty) FROM order_items GROUP BY product_id",
            "SELECT SUM(quantity * unit_price) FROM order_items",
        )

    def test_prose_uses_text_overlap(self):
        assert _content_param_matches("Revenue report for Q1", "Quarterly revenue summary")

    def test_none_returns_false(self):
        assert not _content_param_matches(None, "anything")

    def test_dict_structural(self):
        assert _content_param_matches({"new_column": "x"}, {"new_column": "y", "extra": "z"})
