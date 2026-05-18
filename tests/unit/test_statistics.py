"""Tests for statistical utilities."""

from __future__ import annotations

from behavioral_memory.evaluation.statistics import bootstrap_ci, mcnemar_test


class TestBootstrapCI:
    def test_all_true(self):
        mean, lo, _hi = bootstrap_ci([True] * 10)
        assert mean == 1.0
        assert lo >= 0.9

    def test_all_false(self):
        mean, _lo, _hi = bootstrap_ci([False] * 10)
        assert mean == 0.0

    def test_mixed(self):
        mean, lo, hi = bootstrap_ci([True, False, True, False, True])
        assert 0.0 < mean < 1.0
        assert lo <= mean <= hi

    def test_empty(self):
        mean, _lo, _hi = bootstrap_ci([])
        assert mean == 0.0


class TestMcNemarTest:
    def test_identical_results(self):
        result = mcnemar_test([True, True], [True, True])
        assert result["p_value"] == 1.0

    def test_different_results(self):
        a = [True, False, False, False, False, False, False, False, False, True]
        b = [True, True, True, True, True, True, True, True, True, True]
        result = mcnemar_test(a, b)
        assert result["p_value"] < 0.05
