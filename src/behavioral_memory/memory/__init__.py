from behavioral_memory.memory.dedup import Deduplicator
from behavioral_memory.memory.in_memory_store import InMemoryTraceStore
from behavioral_memory.memory.token_budget import select_traces_within_budget


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "TraceStore":
        from behavioral_memory.memory.store import TraceStore

        return TraceStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Deduplicator", "InMemoryTraceStore", "TraceStore", "select_traces_within_budget"]
