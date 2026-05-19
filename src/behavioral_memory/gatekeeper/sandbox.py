"""Sandboxed execution check for traces.

Implements gate 2 of the gatekeeper pipeline (Section III.E.2):
runs the trace in a controlled environment with timeouts to catch
runtime errors before storing in memory.

For the benchmark, this runs stub executions. In production, this
would connect to actual MCP tool servers.
"""

from __future__ import annotations

import logging
import signal
from typing import Any

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import ExecutionTrace

logger = logging.getLogger(__name__)


class _TimeoutError(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _TimeoutError("Sandbox execution timed out")


class SandboxExecutor:
    """Execute a trace in a sandboxed environment to catch runtime errors."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._timeout = self._settings.sandbox_timeout_seconds

    def execute(self, trace: ExecutionTrace) -> tuple[bool, str]:
        """Run a trace through sandbox checks.

        Returns (passed, detail_message).
        In benchmark mode, this performs structural dry-run checks.
        """
        try:
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self._timeout)

            result = self._dry_run(trace)

            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            return result
        except _TimeoutError:
            return False, f"Execution timed out after {self._timeout}s"
        except Exception as e:
            return False, f"Sandbox execution error: {e}"

    def _dry_run(self, trace: ExecutionTrace) -> tuple[bool, str]:
        """Structural dry-run: verify data flow and parameter references."""
        available_outputs: set[str] = set()

        for step in trace.tool_chain:
            source_step = step.parameters.get("source_step")
            if source_step and source_step != "previous_step" and source_step not in available_outputs:
                return (
                    False,
                    f"Step '{step.step_id}' references '{source_step}' which has no output yet",
                )

            right_step = None
            if isinstance(step.parameters.get("params"), dict):
                right_step = step.parameters["params"].get("right_step")
            if right_step and right_step not in available_outputs:
                return (
                    False,
                    f"Step '{step.step_id}' join references '{right_step}' which has no output yet",
                )

            available_outputs.add(step.step_id)

        return True, "Dry-run passed"
