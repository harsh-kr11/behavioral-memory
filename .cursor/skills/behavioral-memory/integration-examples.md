# Integration Examples

Concrete patterns for wiring behavioral-memory into real agents.

## 1. LangGraph ReAct Agent (create_react_agent / custom StateGraph)

Add a **pre-planning step** that retrieves traces and injects them into
the system prompt before the ReAct loop starts.

```python
from behavioral_memory import InMemoryTraceStore, PlanEngine, ToolRegistry, ToolSchema
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import create_react_agent

# --- 1. Your existing agent setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
tools = [your_tool_a, your_tool_b, your_tool_c]

# --- 2. Add behavioral memory (new code) ---
store = InMemoryTraceStore(embeddings=embeddings)
registry = ToolRegistry()
for t in tools:
    registry.register(ToolSchema(
        name=t.name,
        description=t.description,
        parameters_schema=t.args_schema.model_json_schema() if t.args_schema else {},
    ))

# Seed domain knowledge
from behavioral_memory import ExecutionTrace, ToolCall
store.add(ExecutionTrace(
    task_description="Example task that worked well",
    tool_chain=[
        ToolCall(step_id="s1", tool_name="your_tool_a", parameters={"query": "example"}),
        ToolCall(step_id="s2", tool_name="your_tool_b", parameters={"source_step": "s1"}),
    ],
    source="seed",
))

# --- 3. Build enhanced system prompt ---
from behavioral_memory.memory.token_budget import select_traces_within_budget
from behavioral_memory.planner.prompt import build_prompt

def get_enhanced_prompt(user_query: str, base_prompt: str) -> str:
    traces = select_traces_within_budget(
        store=store, query=user_query, tool_schemas=registry.list_tools()
    )
    if not traces:
        return base_prompt
    trace_section = "\n".join(
        f"Reference: {t.task_description}\n→ {' → '.join(t.tool_names)}"
        for t in traces
    )
    return f"{base_prompt}\n\n## Validated Patterns:\n{trace_section}"

# --- 4. Use with create_react_agent ---
enhanced_prompt = get_enhanced_prompt("user query", "You are a helpful assistant.")
agent = create_react_agent(model=llm, tools=tools, state_modifier=enhanced_prompt)
result = agent.invoke({"messages": [("user", "user query")]})
```

## 2. FastAPI Feedback Endpoint (thumbs-up → behavioral memory)

Wire into an existing `/feedback` endpoint that sends scores to Langfuse.

```python
from fastapi import APIRouter
from behavioral_memory import (
    ExecutionTrace, ToolCall, GatekeeperPipeline, ToolRegistry,
    InMemoryTraceStore, Settings,
)

router = APIRouter()

# Initialize once at startup (reuse across requests)
# In production, use TraceStore with your existing PostgreSQL
store = InMemoryTraceStore(embeddings=your_embeddings)
registry = ToolRegistry()
# ... register your tools ...
gatekeeper = GatekeeperPipeline(store=store, registry=registry)


@router.post("/v1/feedback")
async def feedback(run_id: str, score: float, user_query: str, tool_calls: list):
    # Your existing Langfuse scoring (keep as-is)
    langfuse_client.create_score(trace_id=run_id, name="user_feedback", value=score)

    # NEW: On positive feedback, capture into behavioral memory
    if score >= 1.0 and tool_calls:
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
        logger.info(f"Behavioral memory: {'accepted' if result.accepted else result.rejection_reason}")

    return {"status": "ok"}
```

## 3. LangGraph Custom StateGraph with Behavioral Memory Node

Add a dedicated `retrieve_traces` node to your existing graph.

```python
from langgraph.graph import StateGraph, START, END, MessagesState
from behavioral_memory.memory.token_budget import select_traces_within_budget

# Your existing nodes
def call_model(state): ...
def should_continue(state): ...

# NEW: behavioral memory retrieval node
def retrieve_traces(state: MessagesState):
    user_msg = state["messages"][-1].content
    traces = select_traces_within_budget(
        store=store, query=user_msg, tool_schemas=registry.list_tools()
    )
    if traces:
        context = "Validated patterns from past successful executions:\n"
        for t in traces:
            context += f"- {t.task_description}: {' → '.join(t.tool_names)}\n"
        from langchain_core.messages import SystemMessage
        return {"messages": [SystemMessage(content=context)]}
    return {"messages": []}

# Wire into graph
wf = StateGraph(MessagesState)
wf.add_node("retrieve_memory", retrieve_traces)  # NEW
wf.add_node("LLM", call_model)
wf.add_node("tools", tool_node)
wf.add_edge(START, "retrieve_memory")             # NEW: memory first
wf.add_edge("retrieve_memory", "LLM")             # then LLM
wf.add_conditional_edges("LLM", should_continue)
wf.add_edge("tools", "LLM")
agent = wf.compile()
```

## 4. Production Setup with pgvector (persistent across deploys)

```python
from behavioral_memory import TraceStore, Settings

settings = Settings(
    vector_store_url="postgresql+psycopg://user:pass@db-host:5432/mydb",
    vector_store_collection="validated_traces",  # stable name
    similarity_dedup_threshold=0.95,
    few_shot_k=3,
    max_prompt_tokens=3500,
)

store = TraceStore(
    embeddings=your_embeddings,
    connection_url=settings.vector_store_url,
    collection_name=settings.vector_store_collection,
    settings=settings,
)

# On first deploy: seed traces
if store.count() == 0:
    store.add_bulk(your_seed_traces)

# On subsequent deploys: traces persist, no re-seeding needed
# Dedup gate (cosine >= 0.95) prevents accidental duplicates
```

## 5. Background Feedback Poller (Langfuse v4+)

Run as a background task or separate service.

```python
import asyncio
from behavioral_memory import FeedbackPoller, AnnotationHandler, Settings

settings = Settings(
    langfuse_secret_key="sk-lf-...",
    langfuse_public_key="pk-lf-...",
    langfuse_host="https://us.cloud.langfuse.com",
    feedback_score_name="user_feedback",       # MUST match what your agent sends
    feedback_positive_threshold=1.0,
    feedback_poll_interval=300,                # poll every 5 minutes
)

poller = FeedbackPoller(settings=settings)
handler = AnnotationHandler(poller=poller, gatekeeper=gatekeeper)

# As a background task in FastAPI
@app.on_event("startup")
async def start_feedback_loop():
    asyncio.create_task(asyncio.to_thread(handler.run_loop))
```

## 6. Using PlanEngine Directly (full planning, not just retrieval)

When you want behavioral memory to generate a complete structured plan:

```python
from behavioral_memory import PlanEngine

engine = PlanEngine(llm=your_llm, store=store, registry=registry)

# With memory (retrieves similar traces automatically)
plan = engine.generate(query="Build a revenue dashboard")

# Without memory (zero-shot baseline)
plan_zs = engine.generate_zero_shot(query="Build a revenue dashboard", tool_schemas=schemas)

# Compare
print(f"With memory: {len(plan.steps)} steps, {len(plan.retrieved_traces)} traces used")
print(f"Zero-shot:   {len(plan_zs.steps)} steps")
```
