"""Annotation handler — bridges the feedback loop to the gatekeeper.

Processes positively scored traces from Langfuse, runs them through
the gatekeeper pipeline, and stores accepted traces in memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from behavioral_memory.core.schemas import ExecutionTrace
from behavioral_memory.gatekeeper.pipeline import GatekeeperPipeline
from behavioral_memory.observability.feedback import FeedbackPoller

logger = logging.getLogger(__name__)


@dataclass
class FeedbackStats:
    """Statistics from a feedback processing run."""

    traces_found: int = 0
    accepted: int = 0
    rejected_validation: int = 0
    rejected_sandbox: int = 0
    rejected_duplicate: int = 0
    errors: int = 0
    details: list[str] = field(default_factory=list)


class AnnotationHandler:
    """Processes feedback from Langfuse into behavioral memory via the gatekeeper."""

    def __init__(
        self,
        poller: FeedbackPoller,
        gatekeeper: GatekeeperPipeline,
    ) -> None:
        self._poller = poller
        self._gatekeeper = gatekeeper

    def process_feedback(self, traces: list[ExecutionTrace] | None = None) -> FeedbackStats:
        """Process a batch of traces through the gatekeeper.

        If traces are not provided, polls Langfuse for positive ones.
        """
        if traces is None:
            traces = self._poller.fetch_positive_traces()

        stats = FeedbackStats(traces_found=len(traces))

        for trace in traces:
            try:
                result = self._gatekeeper.submit(trace)
                if result.accepted:
                    stats.accepted += 1
                    stats.details.append(f"Accepted: {trace.task_description[:60]}")
                elif result.is_duplicate:
                    stats.rejected_duplicate += 1
                    stats.details.append(f"Duplicate: {trace.task_description[:60]}")
                elif not result.schema_valid:
                    stats.rejected_validation += 1
                    stats.details.append(f"Invalid: {trace.task_description[:60]} — {', '.join(result.failures)}")
                elif not result.sandbox_passed:
                    stats.rejected_sandbox += 1
                    stats.details.append(f"Sandbox fail: {trace.task_description[:60]}")
            except Exception as e:
                stats.errors += 1
                logger.warning("Error processing trace: %s", e)

        logger.info(
            "Feedback processed: %d found, %d accepted, %d rejected",
            stats.traces_found,
            stats.accepted,
            stats.traces_found - stats.accepted,
        )
        return stats

    def run_once(self) -> FeedbackStats:
        """Single feedback processing pass (poll + process)."""
        return self.process_feedback()

    def run_loop(self, max_iterations: int | None = None) -> None:
        """Continuous feedback loop."""
        self._poller.poll_loop(
            callback=lambda trace: self._gatekeeper.submit(trace),
            max_iterations=max_iterations,
        )
