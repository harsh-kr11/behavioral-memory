"""Langfuse trace logging for the behavioral memory framework.

Logs execution traces, retrieved examples, and generated plans to
Langfuse for observability and later SME review (Section III.F).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import Plan

logger = logging.getLogger(__name__)


class LangfuseTracer:
    """Logs execution data to Langfuse for observability and feedback collection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._client: Any = None

    @property
    def enabled(self) -> bool:
        return self._settings.langfuse_enabled

    @property
    def client(self) -> Any:
        if self._client is None and self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    secret_key=self._settings.langfuse_secret_key,
                    public_key=self._settings.langfuse_public_key,
                    host=self._settings.langfuse_host,
                )
            except ImportError:
                logger.warning("langfuse not installed, tracing disabled")
            except Exception as e:
                logger.warning("Failed to initialize Langfuse: %s", e)
        return self._client

    def log_plan(
        self,
        plan: Plan,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str | None:
        """Log a generated plan to Langfuse.

        Returns the trace ID if successful, None otherwise.
        """
        if not self.enabled or self.client is None:
            return None

        try:
            trace_tags = ["behavioral-memory"]
            if tags:
                trace_tags.extend(tags)

            trace = self.client.trace(
                name="plan_generation",
                input=plan.query,
                output=json.dumps([s.model_dump() for s in plan.steps], indent=2),
                user_id=user_id,
                session_id=session_id,
                tags=trace_tags,
                metadata={
                    "retrieved_traces_count": len(plan.retrieved_traces),
                    "schemas_used": [s.name for s in plan.schemas_used],
                    "token_budget_used": plan.token_budget_used,
                    "steps_count": len(plan.steps),
                },
            )

            trace.generation(
                name="llm_plan_generation",
                input=plan.query,
                output=plan.raw_llm_output,
                metadata={
                    "retrieved_examples": [
                        t.task_description for t in plan.retrieved_traces
                    ],
                },
            )

            self.client.flush()
            trace_id: str = trace.id
            logger.info("Logged plan to Langfuse: trace_id=%s", trace_id)
            return trace_id

        except Exception as e:
            logger.warning("Failed to log to Langfuse: %s", e)
            return None

    def flush(self) -> None:
        if self.client:
            self.client.flush()
