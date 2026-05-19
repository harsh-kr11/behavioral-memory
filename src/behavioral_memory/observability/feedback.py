"""Feedback poller — polls Langfuse for positively scored traces.

Implements the inbound side of the Langfuse feedback loop (Section III.F):
periodically checks for traces that SMEs have scored positively, and
feeds them through the gatekeeper into behavioral memory.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace, ToolCall

logger = logging.getLogger(__name__)


class FeedbackPoller:
    """Polls Langfuse for traces with positive SME scores."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    secret_key=self._settings.langfuse_secret_key,
                    public_key=self._settings.langfuse_public_key,
                    host=self._settings.langfuse_host,
                )
            except Exception as e:
                logger.warning("Failed to initialize Langfuse client: %s", e)
        return self._client

    def fetch_positive_traces(self) -> list[ExecutionTrace]:
        """Fetch traces with positive SME scores from Langfuse.

        Returns ExecutionTrace objects ready for gatekeeper evaluation.
        Compatible with Langfuse SDK v4+ (client.api.trace/scores).
        """
        if self.client is None:
            return []

        try:
            traces_response = self.client.api.trace.list(tags="behavioral-memory")
            traces = traces_response.data if hasattr(traces_response, "data") else []
        except Exception as e:
            logger.warning("Failed to fetch traces from Langfuse: %s", e)
            return []

        positive_traces: list[ExecutionTrace] = []
        for trace in traces:
            if self._has_positive_score(trace):
                execution_trace = self._trace_to_execution_trace(trace)
                if execution_trace:
                    positive_traces.append(execution_trace)

        logger.info("Found %d positive traces", len(positive_traces))
        return positive_traces

    def _has_positive_score(self, trace: Any) -> bool:
        """Check if a trace has a positive score meeting the threshold."""
        try:
            scores_response = self.client.api.scores.list(
                trace_id=trace.id,
                config_id=None,
            )
            scores = scores_response.data if hasattr(scores_response, "data") else []
            for score in scores:
                if (
                    score.name == self._settings.feedback_score_name
                    and score.value >= self._settings.feedback_positive_threshold
                ):
                    return True
        except Exception as e:
            logger.debug("Failed to fetch scores for trace %s: %s", trace.id, e)
        return False

    def _trace_to_execution_trace(self, trace: Any) -> ExecutionTrace | None:
        """Convert a Langfuse trace to an ExecutionTrace."""
        try:
            query = trace.input if isinstance(trace.input, str) else str(trace.input)
            output = trace.output or ""

            if isinstance(output, str):
                try:
                    steps_data = json.loads(output)
                except json.JSONDecodeError:
                    return None
            elif isinstance(output, list):
                steps_data = output
            else:
                return None

            tool_chain = [
                ToolCall(
                    step_id=s.get("step_id", f"step_{i + 1}"),
                    tool_name=str(s.get("tool_name", s.get("tool", ""))),
                    parameters=dict(s.get("parameters", s.get("params", {}))),  # type: ignore[arg-type]
                    depends_on=s.get("depends_on", []),
                )
                for i, s in enumerate(steps_data)
                if isinstance(s, dict)
            ]

            if not tool_chain:
                return None

            return ExecutionTrace(
                task_description=query,
                tool_chain=tool_chain,
                source="feedback",
                metadata={"langfuse_trace_id": trace.id},
            )
        except Exception as e:
            logger.debug("Failed to convert trace %s: %s", getattr(trace, "id", "?"), e)
            return None

    def poll_once(self) -> list[ExecutionTrace]:
        """Single poll cycle."""
        return self.fetch_positive_traces()

    def poll_loop(self, callback: Any = None, max_iterations: int | None = None) -> None:
        """Continuous polling loop.

        Calls callback(trace) for each positive trace found.
        """
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            traces = self.poll_once()
            if callback:
                for trace in traces:
                    callback(trace)
            iteration += 1
            time.sleep(self._settings.feedback_poll_interval)
