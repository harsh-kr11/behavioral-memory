#!/usr/bin/env python3
"""Verify that behavioral-memory is installed and working correctly.

Run:  python .cursor/skills/behavioral-memory/scripts/verify_setup.py

Checks:
  1. Import works
  2. InMemoryTraceStore initializes (needs an embeddings model)
  3. Add / search round-trip works
  4. GatekeeperPipeline accepts a valid trace
  5. (Optional) pgvector TraceStore connects
  6. (Optional) Langfuse connectivity
"""
from __future__ import annotations

import os
import sys

PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
SKIP = "\033[93m SKIP \033[0m"


def check(label: str, fn, skip_if=None):
    if skip_if:
        print(f"  [{SKIP}] {label} — {skip_if}")
        return True
    try:
        fn()
        print(f"  [{PASS}] {label}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {label}: {e}")
        return False


def main():
    print("\n=== behavioral-memory setup verification ===\n")
    all_ok = True

    # 1. Import
    def _import():
        import behavioral_memory  # noqa: F401
        assert hasattr(behavioral_memory, "__version__")

    all_ok &= check("Import behavioral_memory", _import)

    # 2. Check embeddings model availability
    embeddings = None
    llm = None

    def _init_models():
        nonlocal embeddings, llm
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")

        if api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        elif openai_key:
            from langchain_openai import OpenAIEmbeddings, ChatOpenAI
            embeddings = OpenAIEmbeddings()
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        else:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or OPENAI_API_KEY to run full checks"
            )

    all_ok &= check("Initialize embeddings + LLM", _init_models)

    if embeddings is None:
        print("\n  Skipping remaining checks (no embeddings model).")
        sys.exit(1)

    # 3. InMemoryTraceStore round-trip
    store = None

    def _store_roundtrip():
        nonlocal store
        from behavioral_memory import InMemoryTraceStore, ExecutionTrace, ToolCall
        store = InMemoryTraceStore(embeddings=embeddings)
        store.add(ExecutionTrace(
            task_description="fetch customer records from CRM",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="crm_search", parameters={"query": "customer"}),
            ],
            source="seed",
        ))
        assert store.count() == 1
        results = store.search("find customers", k=1)
        assert len(results) == 1
        assert results[0][1] > 0.5  # similarity should be high

    all_ok &= check("InMemoryTraceStore add + search", _store_roundtrip)

    # 4. GatekeeperPipeline validation
    def _gatekeeper():
        from behavioral_memory import (
            GatekeeperPipeline, ToolRegistry, ToolSchema,
            ExecutionTrace, ToolCall,
        )
        reg = ToolRegistry()
        reg.register(ToolSchema(
            name="crm_search",
            description="Search CRM",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            required_params=["query"],
        ))
        gk = GatekeeperPipeline(store=store, registry=reg)
        trace = ExecutionTrace(
            task_description="look up a customer in CRM",
            tool_chain=[ToolCall(step_id="s1", tool_name="crm_search", parameters={"query": "test"})],
            source="seed",
        )
        result = gk.submit(trace)
        assert result.accepted or result.is_duplicate, f"Unexpected rejection: {result.rejection_reason}"

    all_ok &= check("GatekeeperPipeline submit", _gatekeeper)

    # 5. PlanEngine generate
    def _plan_engine():
        from behavioral_memory import PlanEngine, ToolRegistry, ToolSchema
        reg = ToolRegistry()
        reg.register(ToolSchema(
            name="crm_search", description="Search CRM",
            parameters_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            required_params=["query"],
        ))
        reg.register(ToolSchema(
            name="send_email", description="Send email",
            parameters_schema={"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}},
            required_params=["to", "body"],
        ))
        engine = PlanEngine(llm=llm, store=store, registry=reg)
        plan = engine.generate("send a follow-up email to the customer")
        assert len(plan.steps) > 0, "Plan has no steps"

    all_ok &= check("PlanEngine generate", _plan_engine)

    # 6. pgvector TraceStore (optional)
    pg_url = os.environ.get("VECTOR_STORE_URL", "")

    def _pgvector():
        from behavioral_memory import TraceStore
        pg_store = TraceStore(
            embeddings=embeddings,
            connection_url=pg_url,
            collection_name="verify_test",
        )
        _ = pg_store.count()

    all_ok &= check(
        "pgvector TraceStore connect",
        _pgvector,
        skip_if=None if pg_url else "VECTOR_STORE_URL not set",
    )

    # 7. Langfuse (optional)
    lf_secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    lf_public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")

    def _langfuse():
        from behavioral_memory import FeedbackPoller, Settings
        settings = Settings(
            langfuse_secret_key=lf_secret,
            langfuse_public_key=lf_public,
        )
        poller = FeedbackPoller(settings=settings)
        assert poller.client is not None, "Langfuse client failed to initialize"

    all_ok &= check(
        "Langfuse connectivity",
        _langfuse,
        skip_if=None if (lf_secret and lf_public) else "LANGFUSE_SECRET_KEY/PUBLIC_KEY not set",
    )

    # Summary
    print()
    if all_ok:
        print("  All checks passed. behavioral-memory is ready to use.")
    else:
        print("  Some checks failed. Review errors above.")
    print()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
