"""Tests for token counting utilities."""

from __future__ import annotations

from behavioral_memory.memory.token_budget import count_tokens, estimate_trace_tokens


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_short_string(self):
        tokens = count_tokens("Hello world")
        assert 1 < tokens < 10

    def test_longer_string(self):
        tokens = count_tokens("SELECT * FROM customers WHERE id = 1;")
        assert tokens > 5


class TestEstimateTraceTokens:
    def test_returns_positive(self, sample_trace):
        tokens = estimate_trace_tokens(sample_trace)
        assert tokens > 0
