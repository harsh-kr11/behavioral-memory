---
name: behavioral-memory
description: >-
  Integrate behavioral-memory into any LangChain/LangGraph agent. Use when the
  user asks to add behavioral memory, execution trace retrieval, validated
  memory, learning from feedback, or tool orchestration memory to their agent.
  Also use when wiring thumbs-up/down feedback into a trace store, connecting
  Langfuse feedback loops, or setting up pgvector persistence for agent traces.
disable-model-invocation: true
---

# Behavioral Memory Integration

Add validated execution trace retrieval to any LLM agent in under 50 lines.
The library stores task-to-tool-chain mappings and retrieves semantically
similar ones at query time so the agent learns from past successes.

## Install

```bash
pip install behavioral-memory                    # in-memory store (no DB)
pip install "behavioral-memory[postgres]"        # pgvector persistence
```

## Prerequisites (check before starting)

| Need | Why |
|------|-----|
| A LangChain-compatible LLM (`BaseChatModel`) | PlanEngine calls it |
| A LangChain-compatible `Embeddings` model | Vector similarity search |
| Python 3.11+ | Library requirement |
| PostgreSQL + pgvector (optional) | Persistent `TraceStore` |
| Langfuse account (optional) | Feedback loop |

## Quick Integration (3 steps)

### Step 1 — Create store + engine

```python
from behavioral_memory import InMemoryTraceStore, PlanEngine, ToolRegistry, ToolSchema

store = InMemoryTraceStore(embeddings=your_embeddings)
registry = ToolRegistry()

# Register every tool your agent has
for tool in your_agent_tools:
    registry.register(ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters_schema=tool.args_schema.model_json_schema() if tool.args_schema else {},
    ))

engine = PlanEngine(llm=your_llm, store=store, registry=registry)
```

### Step 2 — Seed domain knowledge

```python
from behavioral_memory import ExecutionTrace, ToolCall

store.add(ExecutionTrace(
    task_description="your natural language task here",
    tool_chain=[
        ToolCall(step_id="s1", tool_name="tool_a", parameters={"key": "val"}),
        ToolCall(step_id="s2", tool_name="tool_b", parameters={"source_step": "s1"}),
    ],
    source="seed",
))
```

### Step 3 — Generate plans with memory

```python
plan = engine.generate(query="user's task description")
for step in plan.steps:
    print(f"{step.step_id}: {step.tool_name}({step.parameters})")
```

## Persistence — pgvector (production)

`TraceStore` persists traces across restarts. The collection name is stable
— it does NOT recreate on every deployment. Deduplication (cosine >= 0.95)
prevents the same trace from being stored twice.

```python
from behavioral_memory import TraceStore  # requires [postgres] extra

store = TraceStore(
    embeddings=your_embeddings,
    connection_url="postgresql+psycopg://user:pass@host:5432/dbname",
    collection_name="validated_traces",  # stable across deploys
)
```

**Critical**: both `InMemoryTraceStore` and `TraceStore` expose the same API
(`search`, `add`, `add_bulk`, `similarity_score`, `count`). Swap freely.

## Gatekeeper — validate before storing

Never store unvalidated traces. The gatekeeper runs three gates:

1. **Schema validation** — tools exist, required params present, deps valid
2. **Sandbox execution** — dry-run data-flow check with timeout
3. **Semantic dedup** — cosine similarity >= 0.95 → rejected

```python
from behavioral_memory import GatekeeperPipeline

gatekeeper = GatekeeperPipeline(store=store, registry=registry)
result = gatekeeper.submit(trace)  # validates AND stores if accepted
# result.accepted, result.schema_valid, result.is_duplicate
```

## Feedback Loop — learn from thumbs-up (Langfuse v4+)

Wire your existing feedback endpoint to behavioral memory. When a user
gives thumbs-up, capture the trace and feed it through the gatekeeper.

### Option A: Direct capture in your feedback handler

```python
from behavioral_memory import ExecutionTrace, ToolCall, GatekeeperPipeline

async def on_positive_feedback(run_id: str, user_query: str, tool_calls: list):
    trace = ExecutionTrace(
        task_description=user_query,
        tool_chain=[
            ToolCall(
                step_id=f"step_{i+1}",
                tool_name=tc["name"],
                parameters=tc.get("args", {}),
            )
            for i, tc in enumerate(tool_calls)
        ],
        source="feedback",
        metadata={"run_id": run_id},
    )
    result = gatekeeper.submit(trace)
    return result.accepted
```

### Option B: Poll Langfuse for positively scored traces

```python
from behavioral_memory import FeedbackPoller, AnnotationHandler, Settings

settings = Settings(
    langfuse_secret_key="sk-lf-...",
    langfuse_public_key="pk-lf-...",
    langfuse_host="https://us.cloud.langfuse.com",
    feedback_score_name="user_feedback",     # must match your Langfuse score name
    feedback_positive_threshold=1.0,         # score >= this = positive
)

poller = FeedbackPoller(settings=settings)
handler = AnnotationHandler(poller=poller, gatekeeper=gatekeeper)
handler.run_once()  # single poll cycle
# handler.run_loop()  # continuous background polling
```

### Langfuse v4+ compatibility

The library uses `client.api.trace.list()` and `client.api.scores.list()`
(Langfuse SDK v4 API). If your agent logs scores via `client.create_score()`
or `client.score()`, the poller reads them correctly. The key mapping:

| Your agent sends | Poller reads |
|---|---|
| `client.create_score(trace_id=run_id, name="user_feedback", value=1)` | `score.name == settings.feedback_score_name` |
| `client.score(trace_id=run_id, name="quality", value=1)` | Same — both APIs write to the same store |

## Injecting traces into a ReAct agent prompt

If you don't use `PlanEngine` and want to inject traces into your own prompt:

```python
from behavioral_memory.planner.prompt import build_prompt, SYSTEM_PROMPT
from behavioral_memory.memory.token_budget import select_traces_within_budget

traces = select_traces_within_budget(store=store, query=user_query, tool_schemas=schemas)
prompt = build_prompt(query=user_query, traces=traces, tool_schemas=schemas)
# Send SYSTEM_PROMPT as system message, prompt as user message to your LLM
```

## Configuration (env vars / .env)

| Variable | Default | Purpose |
|---|---|---|
| `FEW_SHOT_K` | `3` | Traces to retrieve per query |
| `MAX_PROMPT_TOKENS` | `3500` | Token budget for prompt |
| `SIMILARITY_DEDUP_THRESHOLD` | `0.95` | Reject traces above this cosine similarity |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | Gatekeeper sandbox timeout |
| `VECTOR_STORE_URL` | — | PostgreSQL connection string |
| `VECTOR_STORE_COLLECTION` | `validated_traces` | pgvector collection name |
| `LANGFUSE_SECRET_KEY` | — | For feedback polling |
| `LANGFUSE_PUBLIC_KEY` | — | For feedback polling |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Langfuse instance URL |
| `FEEDBACK_SCORE_NAME` | `quality` | Langfuse score name to watch |
| `FEEDBACK_POSITIVE_THRESHOLD` | `1.0` | Minimum score to accept |

## Common mistakes

1. **Forgetting to register tools** — `GatekeeperPipeline` rejects traces
   referencing unknown tools. Register all tools BEFORE submitting traces.
2. **Using TraceStore without `[postgres]`** — causes `ImportError`. Use
   `InMemoryTraceStore` for dev or install the extra.
3. **Langfuse score name mismatch** — if your agent sends
   `name="user_feedback_positive"` but settings say `feedback_score_name="quality"`,
   the poller finds nothing. These must match.
4. **pgvector distance vs similarity** — `TraceStore.similarity_score()` already
   converts cosine distance to similarity (1 - distance). The 0.95 dedup threshold
   works correctly out of the box.

## For detailed integration examples

See [integration-examples.md](integration-examples.md) for:
- LangGraph ReAct agent integration
- FastAPI feedback endpoint wiring
- Multi-agent system setup
- Bootstrap script usage

## Bootstrap script

Run the bootstrap script to validate your setup:

```bash
python .cursor/skills/behavioral-memory/scripts/verify_setup.py
```

This checks: import works, store initializes, add/search cycle passes,
gatekeeper accepts valid traces, and optionally tests Langfuse connectivity.
