"""behavioral-memory: Validated execution traces as memory for MCP-based agents.

This library provides a retrieval-based architecture that uses a memory bank
of validated execution traces to guide LLM tool orchestration during inference.

Quick start (no PostgreSQL needed):
    from behavioral_memory import InMemoryTraceStore, PlanEngine

With PostgreSQL (pip install behavioral-memory[postgres]):
    from behavioral_memory import TraceStore, PlanEngine

See https://github.com/harsh-kr11/behavioral-memory for full documentation.
"""

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import (
    ExecutionTrace,
    GatekeeperResult,
    Plan,
    ToolCall,
    ToolSchema,
)
from behavioral_memory.gatekeeper.pipeline import GatekeeperPipeline
from behavioral_memory.memory.dedup import Deduplicator
from behavioral_memory.memory.in_memory_store import InMemoryTraceStore
from behavioral_memory.observability.annotation import AnnotationHandler
from behavioral_memory.observability.feedback import FeedbackPoller
from behavioral_memory.observability.tracer import LangfuseTracer
from behavioral_memory.planner.engine import PlanEngine
from behavioral_memory.tools.registry import ToolRegistry


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import for TraceStore to avoid requiring PostgreSQL deps at import time."""
    if name == "TraceStore":
        from behavioral_memory.memory.store import TraceStore

        return TraceStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnnotationHandler",
    "Deduplicator",
    "ExecutionTrace",
    "FeedbackPoller",
    "GatekeeperPipeline",
    "GatekeeperResult",
    "InMemoryTraceStore",
    "LangfuseTracer",
    "Plan",
    "PlanEngine",
    "Settings",
    "ToolCall",
    "ToolRegistry",
    "ToolSchema",
    "TraceStore",
]

__version__ = "0.1.1"
