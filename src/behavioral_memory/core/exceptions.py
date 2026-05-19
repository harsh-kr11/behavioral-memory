"""Custom exception hierarchy for behavioral-memory."""

from __future__ import annotations

from typing import Any


class BehavioralMemoryError(Exception):
    """Base exception for all behavioral-memory errors."""


class MemoryStoreError(BehavioralMemoryError):
    """Raised when the trace store encounters an error (connection, query, etc.)."""


class TraceValidationError(BehavioralMemoryError):
    """Raised when a trace fails structural or semantic validation."""

    def __init__(self, message: str, failures: list[str] | None = None) -> None:
        super().__init__(message)
        self.failures = failures or []


class GatekeeperRejectionError(BehavioralMemoryError):
    """Raised when the gatekeeper pipeline rejects a trace."""

    def __init__(self, message: str, stage: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details or {}


class PlanGenerationError(BehavioralMemoryError):
    """Raised when the executive layer fails to produce a valid plan."""


class MCPConnectionError(BehavioralMemoryError):
    """Raised when the MCP client cannot connect to or fetch from an MCP server."""


class FeedbackLoopError(BehavioralMemoryError):
    """Raised when the Langfuse feedback polling encounters an error."""
