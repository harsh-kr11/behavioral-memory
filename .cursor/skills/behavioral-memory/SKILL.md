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

Add validated execution trace retrieval to any LLM agent. The library stores
task-to-tool-chain mappings and retrieves semantically similar ones at query
time so the agent learns from past successes.

## Step 0 — Analyse the target agent

Before writing any code, scan the agent's codebase and classify its setup.
Check for these four signals:

```
Checklist:
- [ ] PostgreSQL: look for connection strings, SQLAlchemy, asyncpg, psycopg,
      PGVector, AsyncPostgresSaver, or DATABASE_URL in env/config
- [ ] Langfuse: look for langfuse imports, CallbackHandler, create_score,
      client.score, LANGFUSE_SECRET_KEY in env/config
- [ ] Feedback endpoint: look for thumbs-up/down routes, /feedback, score
      submission, user rating handlers
- [ ] Tools: list all @tool functions, ToolNode, MCP servers, tool arrays
```

Based on what you find, follow the matching path:

**Path A → Production agent (has PostgreSQL + Langfuse + feedback)**
Examples: enterprise agents with RAG, MCP tools, thumbs-up/down UI.
Follow: Install → Path A setup → Seed → Wire feedback → Inject into prompt

**Path B → Agent with PostgreSQL but no Langfuse**
Examples: agents using pgvector for RAG but no observability.
Follow: Install → Path B setup → Seed → Inject into prompt

**Path C → Lightweight agent (no PostgreSQL, no Langfuse)**
Examples: simple ReAct scripts, demos, prototypes, CLI agents.
Follow: Install → Path C setup → Seed → Inject into prompt

---

## Step 1 — Install

```bash
# Path A or B (agent has PostgreSQL):
pip install "behavioral-memory[postgres]"

# Path C (no PostgreSQL):
pip install behavioral-memory
```

---

## Step 2 — Set up store and registry (pick your path)

### Path A: Production (PostgreSQL + Langfuse + Feedback)

The agent already has a PostgreSQL connection. Reuse it for behavioral memory.
Find the existing connection string (e.g. `DATABASE_URL`, `rag_pg_conn`,
`POSTGRES_CONNECTION_STRING`) and use it.

```python
from behavioral_memory import TraceStore, ToolRegistry, ToolSchema, Settings

# Reuse the agent's existing PostgreSQL connection
store = TraceStore(
    embeddings=your_embeddings,  # reuse agent's existing embeddings model
    connection_url=existing_pg_connection_string,  # from agent's config
    collection_name="behavioral_traces",  # new collection, won't conflict
)

registry = ToolRegistry()

# Register ALL tools the agent has (iterate over the agent's tool list)
for tool in agent_tools:
    registry.register(ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters_schema=tool.args_schema.model_json_schema() if tool.args_schema else {},
    ))
```

**Persistence**: the `behavioral_traces` collection is created once and persists
across deployments. It does NOT recreate on restart. Dedup (cosine >= 0.95)
prevents storing the same trace twice.

### Path B: PostgreSQL, no Langfuse

Same as Path A for store setup. Skip the feedback wiring in Step 4.

```python
from behavioral_memory import TraceStore, ToolRegistry, ToolSchema

store = TraceStore(
    embeddings=your_embeddings,
    connection_url=existing_pg_connection_string,
    collection_name="behavioral_traces",
)

registry = ToolRegistry()
for tool in agent_tools:
    registry.register(ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters_schema=tool.args_schema.model_json_schema() if tool.args_schema else {},
    ))
```

### Path C: Lightweight (no PostgreSQL)

Uses in-memory store. Traces are lost on restart — seed on every startup.

```python
from behavioral_memory import InMemoryTraceStore, ToolRegistry, ToolSchema

store = InMemoryTraceStore(embeddings=your_embeddings)

registry = ToolRegistry()
for tool in agent_tools:
    registry.register(ToolSchema(
        name=tool.name,
        description=tool.description,
        parameters_schema=tool.args_schema.model_json_schema() if tool.args_schema else {},
    ))
```

---

## Step 3 — Seed domain knowledge

Create traces that encode your agent's best practices. Look at the agent's
tools and think: "what are the ideal multi-step workflows?"

```python
from behavioral_memory import ExecutionTrace, ToolCall, GatekeeperPipeline

gatekeeper = GatekeeperPipeline(store=store, registry=registry)

# For Path A/B: only seed if collection is empty (first deploy)
if store.count() == 0:
    seed_traces = [
        ExecutionTrace(
            task_description="describe the task in natural language",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="actual_tool_name",
                         parameters={"param": "value"}),
                ToolCall(step_id="s2", tool_name="another_tool",
                         parameters={"source_step": "s1"}),
            ],
            source="seed",
        ),
        # Add 3-5 traces covering the agent's core workflows
    ]
    for trace in seed_traces:
        result = gatekeeper.submit(trace)  # validates before storing

# For Path C: always seed (in-memory resets on restart)
```

**Tip**: study the agent's tools and create traces that show the correct
tool ordering, parameter patterns, and dependencies for common tasks.

---

## Step 4 — Wire feedback loop (Path A only)

Path A agents have Langfuse + a feedback endpoint. Wire positive feedback
into behavioral memory so the agent learns from real user approvals.

### Find the feedback endpoint

Search for the existing feedback handler (e.g. `/feedback`, `/v1/feedback`,
`score`, `thumbs`). It typically calls `langfuse_client.create_score()` or
`langfuse_client.score()`.

### Add behavioral memory capture

In that handler, after the existing Langfuse score call, add:

```python
from behavioral_memory import ExecutionTrace, ToolCall

# When score indicates positive feedback (thumbs up):
if score_value >= 1.0:
    # Extract tool calls from the Langfuse trace or from your agent's state
    trace = ExecutionTrace(
        task_description=user_query,
        tool_chain=[
            ToolCall(
                step_id=f"step_{i+1}",
                tool_name=tc["name"],
                parameters=tc.get("args", {}),
            )
            for i, tc in enumerate(tool_calls_from_run)
        ],
        source="feedback",
        metadata={"run_id": run_id},
    )
    gatekeeper.submit(trace)
```

### Langfuse v4+ API compatibility

The library uses `client.api.trace.list()` and `client.api.scores.list()`
(Langfuse SDK v4). Both `client.create_score()` and `client.score()` write
to the same store — the poller reads them correctly. **Critical**: the
`feedback_score_name` setting MUST match exactly what your agent sends as
the score `name` parameter.

### Alternative: Background Langfuse poller

Instead of modifying the feedback handler, poll Langfuse periodically:

```python
from behavioral_memory import FeedbackPoller, AnnotationHandler, Settings

settings = Settings(
    langfuse_secret_key="sk-lf-...",
    langfuse_public_key="pk-lf-...",
    langfuse_host="https://us.cloud.langfuse.com",
    feedback_score_name="user_feedback",  # MUST match your score name
    feedback_positive_threshold=1.0,
)

poller = FeedbackPoller(settings=settings)
handler = AnnotationHandler(poller=poller, gatekeeper=gatekeeper)
handler.run_once()  # or handler.run_loop() as background task
```

---

## Step 5 — Inject traces into the agent's prompt

This is where behavioral memory actually helps the agent. Pick the method
that fits how the agent is built:

### Method 1: Add a graph node (for LangGraph StateGraph agents)

Add a `retrieve_memory` node BEFORE the LLM node:

```python
from behavioral_memory.memory.token_budget import select_traces_within_budget
from langchain_core.messages import SystemMessage

def retrieve_memory(state):
    user_msg = state["messages"][-1].content
    traces = select_traces_within_budget(
        store=store, query=user_msg, tool_schemas=registry.list_tools()
    )
    if traces:
        context = "Validated patterns from past successful executions:\n"
        for t in traces:
            context += f"- {t.task_description}: {' → '.join(t.tool_names)}\n"
        return {"messages": [SystemMessage(content=context)]}
    return {"messages": []}

# Add to existing graph:
# graph.add_node("retrieve_memory", retrieve_memory)
# graph.add_edge(START, "retrieve_memory")
# graph.add_edge("retrieve_memory", "existing_first_node")
```

### Method 2: Enhance system prompt (for any agent)

If the agent builds a system prompt string, enhance it before the LLM call:

```python
from behavioral_memory.memory.token_budget import select_traces_within_budget

def enhance_prompt(base_prompt: str, user_query: str) -> str:
    traces = select_traces_within_budget(
        store=store, query=user_query, tool_schemas=registry.list_tools()
    )
    if not traces:
        return base_prompt
    section = "\n".join(
        f"- {t.task_description}: {' → '.join(t.tool_names)}"
        for t in traces
    )
    return f"{base_prompt}\n\n## Validated Patterns:\n{section}"
```

### Method 3: Use PlanEngine directly (for plan-first architectures)

```python
from behavioral_memory import PlanEngine

engine = PlanEngine(llm=your_llm, store=store, registry=registry)
plan = engine.generate(query="user's task")
# plan.steps contains the recommended tool chain
```

---

## Configuration (env vars / .env)

| Variable | Default | Path |
|---|---|---|
| `VECTOR_STORE_URL` | — | A, B |
| `VECTOR_STORE_COLLECTION` | `validated_traces` | A, B |
| `LANGFUSE_SECRET_KEY` | — | A |
| `LANGFUSE_PUBLIC_KEY` | — | A |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | A |
| `FEEDBACK_SCORE_NAME` | `quality` | A |
| `FEEDBACK_POSITIVE_THRESHOLD` | `1.0` | A |
| `FEW_SHOT_K` | `3` | All |
| `MAX_PROMPT_TOKENS` | `3500` | All |
| `SIMILARITY_DEDUP_THRESHOLD` | `0.95` | All |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | All |

---

## Common mistakes

1. **Forgetting to register tools** — `GatekeeperPipeline` rejects traces
   referencing unknown tools. Register ALL tools BEFORE submitting traces.
2. **Using TraceStore without `[postgres]`** — causes `ImportError`. Use
   `InMemoryTraceStore` for Path C or install the extra.
3. **Langfuse score name mismatch** — if your agent sends
   `name="user_feedback"` but settings say `feedback_score_name="quality"`,
   the poller finds nothing. These MUST match exactly.
4. **Re-seeding on every deploy** — for Path A/B, check `store.count() == 0`
   first. The collection persists. Dedup catches accidental duplicates but
   checking count is cleaner.

## Verification

```bash
python .cursor/skills/behavioral-memory/scripts/verify_setup.py
```

## Detailed examples

See [integration-examples.md](integration-examples.md) for full code samples.
