"""Behavioral Layer — TraceStore backed by PostgreSQL + pgvector.

Stores validated execution traces as vector-embedded documents and
retrieves the most semantically similar traces for a given query.
Corresponds to Section III.B of the paper.

Requires: pip install behavioral-memory[postgres]
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from behavioral_memory.core.config import Settings
from behavioral_memory.core.exceptions import MemoryStoreError
from behavioral_memory.core.schemas import ExecutionTrace, ToolCall

logger = logging.getLogger(__name__)

_HAS_PGVECTOR = False
try:
    import langchain_postgres  # noqa: F401

    _HAS_PGVECTOR = True
except ImportError:
    pass

if TYPE_CHECKING:
    from langchain_postgres import PGVector


def _check_postgres_deps() -> None:
    if not _HAS_PGVECTOR:
        raise ImportError(
            "TraceStore requires PostgreSQL dependencies. Install them with: pip install behavioral-memory[postgres]"
        )


def _create_pgvector(
    collection_name: str,
    connection: str,
    embeddings: Embeddings,
) -> PGVector:
    """Deferred import + construction so the module loads without postgres deps."""
    from langchain_postgres import PGVector

    return PGVector(
        collection_name=collection_name,
        connection=connection,
        embeddings=embeddings,
        use_jsonb=True,
    )


class TraceStore:
    """Vector store for validated execution traces.

    Accepts any LangChain-compatible Embeddings model, making the
    framework model-agnostic. The caller decides which embedding
    provider to use (Gemini, OpenAI, local, etc.).

    Requires: pip install behavioral-memory[postgres]
    """

    def __init__(
        self,
        embeddings: Embeddings,
        connection_url: str | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        _check_postgres_deps()
        self._settings = settings or Settings()
        self._connection_url = connection_url or self._settings.vector_store_url
        self._collection_name = collection_name or self._settings.vector_store_collection
        self._embeddings = embeddings
        self._vectorstore: Any = None

    @property
    def vectorstore(self) -> Any:
        if self._vectorstore is None:
            try:
                self._vectorstore = _create_pgvector(
                    collection_name=self._collection_name,
                    connection=self._connection_url,
                    embeddings=self._embeddings,
                )
            except Exception as e:
                raise MemoryStoreError(f"Failed to connect to vector store: {e}") from e
        return self._vectorstore

    def search(self, query: str, k: int | None = None) -> list[tuple[ExecutionTrace, float]]:
        """Retrieve the top-k most similar traces for a query."""
        k = k or self._settings.few_shot_k
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as e:
            raise MemoryStoreError(f"Similarity search failed: {e}") from e

        traces: list[tuple[ExecutionTrace, float]] = []
        for doc, score in results:
            trace = self._doc_to_trace(doc)
            if trace is not None:
                traces.append((trace, float(score)))
        return traces

    def add(self, trace: ExecutionTrace) -> None:
        """Add a single validated trace to the store."""
        doc = self._trace_to_doc(trace)
        try:
            self.vectorstore.add_documents([doc])
            logger.info("Stored trace: %s", trace.task_description[:80])
        except Exception as e:
            raise MemoryStoreError(f"Failed to add trace: {e}") from e

    def add_bulk(self, traces: list[ExecutionTrace]) -> int:
        """Bulk-add traces. Returns the number successfully added."""
        docs = [self._trace_to_doc(t) for t in traces]
        try:
            self.vectorstore.add_documents(docs)
            logger.info("Bulk-added %d traces", len(docs))
            return len(docs)
        except Exception as e:
            raise MemoryStoreError(f"Bulk add failed: {e}") from e

    def similarity_score(self, query: str) -> float:
        """Return the highest similarity score for a query against the store."""
        results = self.vectorstore.similarity_search_with_score(query, k=1)
        if not results:
            return 0.0
        _, score = results[0]
        return float(score)

    def count(self) -> int:
        """Approximate count of traces in the store."""
        try:
            docs = self.vectorstore.similarity_search("*", k=10_000)
            return len(docs)
        except Exception:
            return 0

    @staticmethod
    def _trace_to_doc(trace: ExecutionTrace) -> Document:
        """Serialize an ExecutionTrace into a LangChain Document."""
        tool_chain_data = [step.model_dump() for step in trace.tool_chain]
        return Document(
            page_content=trace.task_description,
            metadata={
                "tool_chain": json.dumps(tool_chain_data),
                "validated": trace.validated,
                "source": trace.source,
                "created_at": trace.created_at.isoformat(),
                "extra": json.dumps(trace.metadata),
            },
        )

    @staticmethod
    def _doc_to_trace(doc: Document) -> ExecutionTrace | None:
        """Deserialize a Document back into an ExecutionTrace."""
        try:
            meta = doc.metadata
            chain_raw = json.loads(meta.get("tool_chain", "[]"))
            tool_chain = [ToolCall(**step) for step in chain_raw]
            return ExecutionTrace(
                task_description=doc.page_content,
                tool_chain=tool_chain,
                validated=meta.get("validated", False),
                source=meta.get("source", "seed"),
                metadata=json.loads(meta.get("extra", "{}")),
            )
        except Exception:
            logger.warning("Failed to deserialize document: %s", doc.page_content[:60])
            return None
