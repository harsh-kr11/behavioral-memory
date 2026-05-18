"""Gatekeeper Pipeline — chains all three validation checks.

Implements the full gatekeeper described in Section III.E:
  1. Schema validation (structural)
  2. Sandboxed execution (runtime)
  3. Semantic deduplication (memory hygiene)

A trace must pass all three gates to be accepted into behavioral memory.
"""

from __future__ import annotations

import logging

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace, GatekeeperResult
from behavioral_memory.gatekeeper.dedup_gate import DeduplicationGate
from behavioral_memory.gatekeeper.sandbox import SandboxExecutor
from behavioral_memory.gatekeeper.schema_validator import SchemaValidator
from behavioral_memory.memory.dedup import Deduplicator
from behavioral_memory.memory.store import TraceStore
from behavioral_memory.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class GatekeeperPipeline:
    """Full gatekeeper pipeline: validate -> sandbox -> dedup -> store."""

    def __init__(
        self,
        store: TraceStore,
        registry: ToolRegistry,
        settings: Settings | None = None,
    ) -> None:
        self._store = store
        self._settings = settings or Settings()
        self._schema_validator = SchemaValidator(registry)
        self._sandbox = SandboxExecutor(settings=self._settings)
        self._dedup_gate = DeduplicationGate(
            Deduplicator(store=store, settings=self._settings)
        )

    def evaluate(self, trace: ExecutionTrace) -> GatekeeperResult:
        """Run a trace through all three gates and return the result.

        Does NOT store the trace — call submit() for that.
        """
        schema_valid, schema_failures = self._schema_validator.validate(trace)
        if not schema_valid:
            return GatekeeperResult(
                accepted=False,
                schema_valid=False,
                rejection_reason="Schema validation failed",
                failures=schema_failures,
            )

        sandbox_passed, sandbox_detail = self._sandbox.execute(trace)
        if not sandbox_passed:
            return GatekeeperResult(
                accepted=False,
                schema_valid=True,
                sandbox_passed=False,
                rejection_reason=f"Sandbox check failed: {sandbox_detail}",
                failures=[sandbox_detail],
            )

        is_unique, score = self._dedup_gate.check(trace)
        if not is_unique:
            return GatekeeperResult(
                accepted=False,
                schema_valid=True,
                sandbox_passed=True,
                is_duplicate=True,
                rejection_reason=f"Semantic duplicate (score={score:.3f})",
                failures=[f"Duplicate score {score:.3f} >= threshold"],
            )

        return GatekeeperResult(
            accepted=True,
            schema_valid=True,
            sandbox_passed=True,
            is_duplicate=False,
        )

    def submit(self, trace: ExecutionTrace) -> GatekeeperResult:
        """Evaluate and, if accepted, store the trace in memory."""
        result = self.evaluate(trace)
        if result.accepted:
            trace.validated = True
            self._store.add(trace)
            logger.info("Trace accepted and stored: %s", trace.task_description[:60])
        else:
            logger.info(
                "Trace rejected (%s): %s",
                result.rejection_reason,
                trace.task_description[:60],
            )
        return result
