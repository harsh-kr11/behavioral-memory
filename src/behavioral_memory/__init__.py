"""behavioral-memory: Validated execution traces as memory for MCP-based agents.

This library provides a retrieval-based architecture that uses a memory bank
of validated execution traces to guide LLM tool orchestration during inference.

Quick start:
    from behavioral_memory import TraceStore, PlanEngine, GatekeeperPipeline

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
from behavioral_memory.memory.store import TraceStore
from behavioral_memory.observability.annotation import AnnotationHandler
from behavioral_memory.observability.feedback import FeedbackPoller
from behavioral_memory.observability.tracer import LangfuseTracer
from behavioral_memory.planner.engine import PlanEngine
from behavioral_memory.tools.registry import ToolRegistry

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

__version__ = "0.1.0"
