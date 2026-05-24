# behavioral-memory

**Give your agent institutional memory. Drop-in retrieval of validated execution traces for any LLM agent framework.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![CI](https://github.com/harsh-kr11/behavioral-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/harsh-kr11/behavioral-memory/actions/workflows/ci.yml)

Your agent makes the same mistakes repeatedly because it has no memory of what worked before. **behavioral-memory** fixes this — it stores validated execution traces (task → tool chain mappings) and retrieves semantically similar ones at query time, so your agent learns from past successes instead of starting from scratch every time.

Based on: *"Behavioral Memory for Tool Orchestration: Semantic Retrieval of Validated Execution Traces in MCP-Based Agent Systems"* (IEEE, 2025)

---

## Install

```bash
pip install behavioral-memory
```

---

## Plug Into Your Agent (3 lines)

The library is **framework-agnostic**. You bring your own LLM, your own agent — behavioral-memory handles the memory layer.

### Core API

```python
from behavioral_memory import PlanEngine, ToolRegistry, InMemoryTraceStore

# 1. Choose your LLM (any LangChain-compatible model)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 2. Create a memory store (no database needed)
store = InMemoryTraceStore(embeddings=embeddings)

# 3. Generate plans with behavioral memory
engine = PlanEngine(llm=llm, store=store)
plan = engine.generate(query="Get revenue data and email a report")
```

That's it. Your agent now has memory.

### With OpenAI

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o", temperature=0)
store = InMemoryTraceStore(embeddings=OpenAIEmbeddings())
engine = PlanEngine(llm=llm, store=store)
```

### With Ollama (fully local)

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings

llm = ChatOllama(model="llama3")
store = InMemoryTraceStore(embeddings=OllamaEmbeddings(model="nomic-embed-text"))
engine = PlanEngine(llm=llm, store=store)
```

### Production: PostgreSQL + pgvector

```python
from behavioral_memory import TraceStore  # pip install behavioral-memory[postgres]

store = TraceStore(
    embeddings=embeddings,
    connection_url="postgresql+psycopg://user:pass@localhost/behavioral_memory",
)
```

---

## How It Helps Your Agent

Before behavioral memory, your agent sees only the task and tool schemas — it has to figure out orchestration from scratch every time. With behavioral memory, it retrieves validated examples of similar tasks that worked before.

```
Your Agent's Query: "Build a revenue analysis pipeline"
                │
   ┌────────────┴────────────┐
   │   BEHAVIORAL MEMORY     │
   │                         │
   │  1. Retrieve top-k      │  ← finds 3 similar validated traces
   │     similar traces      │     from past successful executions
   │                         │
   │  2. Merge with tool     │  ← current MCP tool schemas
   │     schemas             │
   │                         │
   │  3. Generate plan       │  ← LLM sees examples + schemas + query
   └────────────┬────────────┘
                │
                ▼
         Better execution plan
         (right tools, right params, right order)
```

### Seed your memory with domain knowledge

```python
from behavioral_memory import ExecutionTrace, ToolCall

trace = ExecutionTrace(
    task_description="Calculate quarterly revenue",
    tool_chain=[
        ToolCall(step_id="s1", tool_name="query_database",
                 parameters={"query": "SELECT SUM(quantity * unit_price) FROM order_items"}),
        ToolCall(step_id="s2", tool_name="generate_report",
                 parameters={"source_step": "s1", "format": "markdown_table"}),
    ],
    source="seed",
)
store.add(trace)
```

### Register your own tool schemas

The `PlanEngine` needs to know what tools your agent has:

```python
from behavioral_memory import ToolSchema, ToolRegistry

schema = ToolSchema(
    name="search_docs",
    description="Search internal documentation",
    parameters_schema={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)

registry = ToolRegistry()
registry.register(schema)
engine = PlanEngine(llm=llm, store=store, registry=registry)
```

Or load schemas dynamically from an MCP server:

```python
from behavioral_memory.tools.mcp_client import fetch_mcp_schemas

schemas = await fetch_mcp_schemas("http://localhost:3000/sse")
registry.register_many(schemas)
```

### Validate before storing (Gatekeeper Pipeline)

Don't let bad traces into memory. The gatekeeper runs three checks before accepting a trace:

```python
from behavioral_memory import GatekeeperPipeline

gatekeeper = GatekeeperPipeline(store=store, registry=registry)
result = gatekeeper.submit(trace)  # schema check → sandbox → dedup → store
print(result.accepted)  # True if all gates passed
```

### Learn from production (Langfuse Feedback Loop)

Traces logged to Langfuse can be reviewed by domain experts. Positively scored traces automatically flow back into memory through the gatekeeper:

```python
from behavioral_memory import FeedbackPoller, AnnotationHandler

poller = FeedbackPoller(settings=settings)
handler = AnnotationHandler(poller=poller, gatekeeper=gatekeeper)
handler.run_loop()  # continuously polls → validates → stores
```

### Without LangChain (plain Python)

If you don't use LangChain, you can use the lower-level primitives directly:

```python
from behavioral_memory.planner.prompt import SYSTEM_PROMPT, build_prompt
from behavioral_memory.planner.postprocess import postprocess_plan

# Build the prompt yourself
prompt = build_prompt(query="Get revenue data", traces=my_traces, tool_schemas=my_schemas)

# Call your own LLM
raw_output = your_llm.chat(system=SYSTEM_PROMPT, user=prompt)

# Parse the JSON plan
steps = postprocess_plan(raw_output)  # returns list[ToolCall]
```

---

## Persistence and Limitations

| Store | Persistence | Multi-user | Best for |
|-------|------------|------------|----------|
| `InMemoryTraceStore` | Process memory only | No | Dev, CI, demos |
| `TraceStore` (pgvector) | PostgreSQL, survives restarts | Shared DB, single collection | Production |

**Current limitations:**
- All traces share one collection (default: `validated_traces`). No per-user or per-session isolation.
- Langfuse is **optional** — the core framework (planning, retrieval, gatekeeper) works without it.
- The reference agent at `agent/` is a planning demo with stub tool execution — bring your own tool runtime.

---

## Key Results

### Multi-Model Comparison

30-task benchmark, 7 MCP tools, temperature 0, embeddings: `gemini-embedding-001`. Behavioral memory improves plan correctness across all three models tested — the gain is consistent regardless of model family or size.

| Metric | Strategy | **Gemini 2.5 Pro** | **Gemini 3 Flash Preview** | **Gemini 3.5 Flash** |
|--------|----------|----------------|------------------------|------------------|
| Tool Selection (TSA) | Zero-Shot | 63.3% | 60.0% | 73.3% |
| | Static Few-Shot | 70.0% | 80.0% | 76.7% |
| | **Dynamic (Proposed)** | **93.3%** | **76.7%** | **83.3%** |
| Parameter Validity (PV) | Zero-Shot | 60.6% | 70.2% | 71.1% |
| | Static Few-Shot | 68.9% | 77.2% | 74.7% |
| | **Dynamic (Proposed)** | **85.5%** | **74.6%** | **80.9%** |
| Plan Correctness (PCR) | Zero-Shot | 50.0% | 50.0% | 60.0% |
| | Static Few-Shot | 63.3% | 73.3% | 70.0% |
| | **Dynamic (Proposed)** | **80.0%** | **76.7%** | **80.0%** |
| Sequence Accuracy (ESA) | Zero-Shot | 63.3% | 60.0% | 73.3% |
| | Static Few-Shot | 70.0% | 80.0% | 76.7% |
| | **Dynamic (Proposed)** | **93.3%** | **76.7%** | **83.3%** |

**McNemar's test (Zero-Shot vs Dynamic):**

| Model | p-value | Significant? |
|-------|---------|-------------|
| Gemini 2.5 Pro | p = 0.0225 | Yes |
| Gemini 3 Flash Preview | p = 0.0215 | Yes |
| Gemini 3.5 Flash | p = 0.0703 | No |

Dynamic retrieval reaches **76.7–80.0% PCR** across all models. The improvement is statistically significant (p < 0.05) for two of three models; Gemini 3.5 Flash starts with a higher zero-shot baseline (60.0%), narrowing the gap.

<details>
<summary>Original paper results (Gemini 2.5 Pro, single-model)</summary>

| Metric | Zero-Shot | Static Few-Shot | With Behavioral Memory |
|--------|-----------|----------------|------------------------|
| TSA | 63.3% | 70.0% | 83.3% |
| PV | 72.2% | 79.6% | 84.0% |
| PCR | 33.3% | 50.0% | 63.3% |
| ESA | 63.3% | 70.0% | 83.3% |

McNemar's test: p = 0.004 vs zero-shot.

</details>

---

## Architecture

Three layers (from the paper):

| Layer | What it does | Key class |
|-------|-------------|-----------|
| **Behavioral** | Store and retrieve validated execution traces via cosine similarity | `InMemoryTraceStore` / `TraceStore` |
| **Tool** | Load tool schemas dynamically via MCP | `ToolRegistry` / `MCPClient` |
| **Executive** | Assemble prompt (traces + schemas + query), call LLM, parse plan | `PlanEngine` |

**Gatekeeper Pipeline** guards memory quality with three gates:
1. **Schema validation** — tools exist, params valid, deps logical
2. **Sandboxed execution** — runtime check with timeout
3. **Semantic deduplication** — cosine > 0.95 rejected

---

## Reproduce the Paper

```bash
git clone https://github.com/harsh-kr11/behavioral-memory.git
cd behavioral-memory
pip install -e ".[agent,eval]"
export GOOGLE_API_KEY=your-key

# Run the 30-task benchmark (default: gemini-2.5-pro)
python examples/run_live_benchmark.py

# Quick test (5 tasks)
python examples/run_live_benchmark.py --limit 5

# Multi-model comparison
python examples/run_live_benchmark.py --model gemini-2.5-pro --output results_25pro.json
python examples/run_live_benchmark.py --model gemini-3-flash-preview --output results_3flash.json
python examples/run_live_benchmark.py --model gemini-3.5-flash --output results_35flash.json
python examples/compare_models.py --results results_25pro.json results_3flash.json results_35flash.json --markdown

# Exact paper reproduction (with pgvector)
pip install -e ".[postgres]"
docker compose up -d  # or: podman-compose up -d
python examples/run_live_benchmark.py --postgres

# Gatekeeper ablation study (Section IV.D.5)
python examples/gatekeeper_ablation.py --verbose

# Validate pipeline offline (no API keys)
python examples/validate_pipeline.py
```

A reference LangGraph agent is included at `agent/` for demo purposes.

---

## Cursor Agent Skill

This repo ships a [Cursor Agent Skill](.cursor/skills/behavioral-memory/) for guided integration. Open this repo in Cursor and type `/behavioral-memory` in the Agent chat to invoke the skill — it walks through store setup, seed traces, feedback loops, Langfuse v4 wiring, and pgvector persistence.

```bash
# Verify your setup after following the skill
python .cursor/skills/behavioral-memory/scripts/verify_setup.py
```

See [integration-examples.md](.cursor/skills/behavioral-memory/integration-examples.md) for LangGraph, FastAPI, and production patterns.

---

## Development

```bash
pip install -e ".[dev,eval]"
make test         # 104 tests
make lint         # ruff check
make typecheck    # mypy (strict)
make ci           # all checks
```

---

## Configuration

All via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `FEW_SHOT_K` | `3` | Traces to retrieve per query |
| `MAX_PROMPT_TOKENS` | `3500` | Token budget for prompt |
| `SIMILARITY_DEDUP_THRESHOLD` | `0.95` | Dedup cosine threshold |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | Gatekeeper sandbox timeout |
| `VECTOR_STORE_URL` | — | PostgreSQL connection (only for `TraceStore`) |
| `LANGFUSE_SECRET_KEY` | — | Langfuse secret (optional) |
| `LANGFUSE_PUBLIC_KEY` | — | Langfuse public key (optional) |

---

## Citation

```bibtex
@inproceedings{khan2025behavioral,
  title={Behavioral Memory for Tool Orchestration: Semantic Retrieval of
         Validated Execution Traces in MCP-Based Agent Systems},
  author={Khan, Mehvash and Kumar, Harsh and Jangir, Rahul},
  booktitle={IEEE Conference Proceedings},
  year={2025}
}
```

## License

Apache 2.0 — See [LICENSE](LICENSE).
