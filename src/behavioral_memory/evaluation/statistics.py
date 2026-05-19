"""Statistical analysis utilities for evaluation results.

Bootstrap confidence intervals and McNemar's exact test,
as used in the paper (Section IV.D).
"""

from __future__ import annotations

import random
from typing import Any


def bootstrap_ci(
    results: list[bool],
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a binary metric.

    Returns (mean, lower_bound, upper_bound).
    """
    rng = random.Random(seed)
    n = len(results)
    if n == 0:
        return 0.0, 0.0, 0.0

    means: list[float] = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(results) for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    point_estimate = sum(results) / n
    return point_estimate, means[lower_idx], means[upper_idx]


def mcnemar_test(results_a: list[bool], results_b: list[bool]) -> dict[str, Any]:
    """McNemar's exact test for paired binary outcomes.

    Compares two methods on the same set of tasks.
    Returns test statistic details and p-value.
    """
    assert len(results_a) == len(results_b), "Results must be same length"

    b_count = 0  # A wrong, B right
    c_count = 0  # A right, B wrong

    for a, b in zip(results_a, results_b, strict=True):
        if not a and b:
            b_count += 1
        elif a and not b:
            c_count += 1

    n = b_count + c_count
    if n == 0:
        return {"b": b_count, "c": c_count, "n": n, "p_value": 1.0, "note": "No discordant pairs"}

    try:
        from scipy.stats import binomtest

        result = binomtest(b_count, n, 0.5)
        p_value = result.pvalue
    except ImportError:
        p_value = _exact_binomial_p(b_count, n)

    return {
        "b": b_count,
        "c": c_count,
        "n": n,
        "p_value": p_value,
    }


def _exact_binomial_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value (fallback without scipy)."""
    from math import comb

    p = 0.0
    for i in range(n + 1):
        prob = comb(n, i) * (0.5**n)
        if comb(n, i) <= comb(n, k):
            p += prob
    return min(p, 1.0)
