# behavioral-memory

**Validated execution traces as memory for MCP-based agent tool orchestration.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![CI](https://github.com/harsh-kr11/behavioral-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/harsh-kr11/behavioral-memory/actions/workflows/ci.yml)

A retrieval-based framework that uses a memory bank of validated execution traces to guide LLM tool orchestration during inference. Instead of relying on static few-shot examples, the system dynamically retrieves semantically similar, validated traces from past successful executions — giving your agent **institutional memory** that improves with every interaction.

Based on the IEEE paper: *"Behavioral Memory for Tool Orchestration: Semantic Retrieval of Validated Execution Traces in MCP-Based Agent Systems"*

---

## Key Results (from the paper)

On a 30-task benchmark with 7 MCP tools, using Gemini 2.5 Pro:

| Metric | Zero-Shot | Static Few-Shot | **Proposed** |
|--------|-----------|----------------|-------------|
| Tool Selection (TSA) | 63.3% | 70.0% | **83.3%** |
| Parameter Validity (PV) | 72.2% | 79.6% | **84.0%** |
| Plan Correctness (PCR) | 33.3% | 50.0% | **63.3%** |
| Sequence Accuracy (ESA) | 63.3% | 70.0% | **83.3%** |

McNemar's test: **p = 0.004** vs zero-shot.

> **Note:** These numbers are from the published paper. To reproduce them yourself, see [Running the Real Benchmark](#running-the-real-benchmark) below.

---

## Quick Start

### Option A: No API keys needed (validation + demo)

```bash
git clone https://github.com/harsh-kr11/behavioral-memory.git
cd behavioral-memory
pip install -e ".[agent,eval,dev]"

# Validate the entire pipeline (30/30 checks, no external services)
python examples/validate_pipeline.py

# Quick demo showing behavioral memory impact
behavioral-memory demo
```

### Option B: With a Google API key (real benchmark)

```bash
export GOOGLE_API_KEY=your-key-here
python examples/run_live_benchmark.py               # all 30 tasks
python examples/run_live_benchmark.py --limit 5      # quick test with 5 tasks
python examples/run_live_benchmark.py --model gemini-2.0-flash  # cheaper model
```

### Option C: Interactive agent

```bash
export GOOGLE_API_KEY=your-key-here
python -m agent.app --interactive

# Or single query:
python -m agent.app "Build a revenue analysis pipeline"
```

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  1. BEHAVIORAL LAYER                                │
│     Retrieve top-k similar traces from memory       │
│     (pgvector or in-memory — your choice)           │
│                                                     │
│  2. TOOL LAYER                                      │
│     Fetch available tool schemas via MCP             │
│                                                     │
│  3. EXECUTIVE LAYER                                 │
│     Assemble 3-layer prompt → LLM → JSON plan       │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
                    Execution Plan
                    (ordered tool calls)
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  GATEKEEPER PIPELINE                                 │
│  ┌──────────────┬──────────────┬──────────────────┐  │
│  │ Schema       │ Sandboxed    │ Semantic         │  │
│  │ Validation   │ Execution    │ Deduplication    │  │
│  └──────────────┴──────────────┴──────────────────┘  │
│           Only validated traces enter memory          │
└──────────────────────────────────────────────────────┘
                           │
                           ▼
               ┌────────────────────┐
               │ Langfuse           │
               │ (trace + feedback) │
               └────────────────────┘
```

---

## Two Ways to Use

### 1. As a Library (Bring Your Own Agent)

Install and plug into your existing agent:

```bash
pip install behavioral-memory
```

```python
from behavioral_memory import PlanEngine, ToolRegistry, InMemoryTraceStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

# No PostgreSQL needed — InMemoryTraceStore works anywhere
store = InMemoryTraceStore(embeddings=embeddings)
registry = ToolRegistry()
engine = PlanEngine(llm=llm, store=store, registry=registry)

plan = engine.generate(query="Get revenue data and email a report")
```

For production with PostgreSQL + pgvector:

```python
from behavioral_memory import TraceStore

store = TraceStore(embeddings=embeddings, connection_url="postgresql+psycopg://...")
```

### 2. Run the Reference Agent (LangGraph 1.x)

```bash
git clone https://github.com/harsh-kr11/behavioral-memory.git
cd behavioral-memory
pip install -e ".[agent]"

export GOOGLE_API_KEY=your-key

# Interactive mode
python -m agent.app --interactive

# Single query
python -m agent.app "Build a revenue analysis pipeline"
```

The interactive agent supports:
- `/compare <query>` — run with AND without memory, see the difference
- `/memory` — inspect what's in behavioral memory
- `/quit` — exit

---

## Running the Real Benchmark

The benchmark sends 30 tasks through 3 strategies (zero-shot, static few-shot, dynamic retrieval), scoring each plan against gold tool chains.

### Prerequisites

Only a Google API key. No PostgreSQL required — the benchmark uses `InMemoryTraceStore`.

```bash
pip install -e ".[agent,eval]"
export GOOGLE_API_KEY=your-key-here
```

### Run

```bash
# Full benchmark (30 tasks × 3 strategies = 90 LLM calls)
python examples/run_live_benchmark.py

# Quick test (5 tasks × 3 strategies = 15 LLM calls)
python examples/run_live_benchmark.py --limit 5

# Use a cheaper/faster model
python examples/run_live_benchmark.py --model gemini-2.0-flash

# With Langfuse logging
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_PUBLIC_KEY=pk-lf-...
python examples/run_live_benchmark.py
```

### What you'll see

```
Benchmark Results (N=30, model=gemini-2.5-pro)
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric ┃ Zero-Shot        ┃ Static Few-Shot     ┃ Dynamic (Proposed)       ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ TSA    │ 63.3% [53%, 73%] │ 70.0% [56%, 83%]    │ 83.3% [70%, 93%]         │
│ PV     │ 72.2%            │ 79.6%               │ 84.0%                    │
│ PCR    │ 33.3% [16%, 50%] │ 50.0% [33%, 66%]    │ 63.3% [46%, 80%]         │
│ ESA    │ 63.3% [46%, 80%] │ 70.0% [53%, 86%]    │ 83.3% [70%, 93%]         │
└────────┴──────────────────┴─────────────────────┴──────────────────────────┘
```

Results include per-task breakdowns, difficulty-tier analysis, and McNemar's test.

---

## Pipeline Validation (No API Keys)

Validates every component works correctly using mock services:

```bash
python examples/validate_pipeline.py
```

This verifies:
- 12 seed traces load and pass schema validation
- 30 ground truth tasks have correct structure
- InMemoryTraceStore embeds, stores, and retrieves traces
- PlanEngine generates plans (zero-shot, static, dynamic)
- BenchmarkRunner scores and compares strategies
- Gatekeeper pipeline accepts/rejects traces
- Langfuse tracer handles offline mode gracefully

All **30 checks** pass with zero external dependencies.

---

## Installation

### Prerequisites

- Python 3.11+
- (Optional) PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) for production deployments

### Install with uv (recommended)

```bash
uv add behavioral-memory              # core framework (no PostgreSQL needed)
uv add behavioral-memory[agent]       # + reference LangGraph agent
uv add behavioral-memory[eval]        # + evaluation/statistics (scipy)
uv add behavioral-memory[postgres]    # + PostgreSQL/pgvector store
uv add behavioral-memory[all]         # everything
```

### Install with pip

```bash
pip install behavioral-memory
pip install behavioral-memory[agent,eval]
pip install behavioral-memory[postgres]  # only if using PostgreSQL
```

### Environment Setup

```bash
# Interactive setup (guides you through each variable)
behavioral-memory setup

# Or manual
cp .env.example .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | For LLM calls | Gemini API key (or use any LangChain-compatible LLM) |
| `VECTOR_STORE_URL` | For PostgreSQL mode | `postgresql+psycopg://localhost/behavioral_memory` |
| `LANGFUSE_SECRET_KEY` | For observability | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | For observability | Langfuse public key |

---

## Architecture

### Project Structure

```
behavioral-memory/
├── src/behavioral_memory/     # The pip-installable library
│   ├── core/                  #   Schemas, config, exceptions
│   ├── memory/                #   Behavioral Layer (TraceStore, InMemoryTraceStore, dedup)
│   ├── tools/                 #   Tool Layer (MCP client, registry, mock tools)
│   ├── planner/               #   Executive Layer (PlanEngine, prompt, postprocess)
│   ├── gatekeeper/            #   Gatekeeper (schema validator, sandbox, dedup gate)
│   ├── observability/         #   Langfuse (tracer, feedback poller, annotation)
│   └── evaluation/            #   Benchmark (30 tasks, metrics, statistics)
├── agent/                     # Reference LangGraph 1.x agent
│   ├── graph.py               #   StateGraph definition
│   ├── state.py               #   Agent state
│   └── nodes/                 #   Graph nodes (retrieve, plan, execute, observe)
├── tests/                     # 104 tests (unit + integration + e2e)
│   ├── unit/                  #   61 unit tests
│   ├── integration/           #   3 integration tests
│   └── e2e/                   #   40 end-to-end tests
├── examples/
│   ├── validate_pipeline.py       # Full pipeline validation (no API keys)
│   ├── run_live_benchmark.py      # Real benchmark (needs API key)
│   ├── gatekeeper_ablation.py     # Gatekeeper ablation study (Section IV.D.5)
│   └── run_benchmark.py           # Benchmark with PostgreSQL
├── Makefile                       # Common dev tasks (make lint, make test, etc.)
└── .github/workflows/             # CI/CD (lint, typecheck, test on 3.11/3.12/3.13)
```

### Store Options

| Store | When to Use | Requires |
|-------|------------|----------|
| `InMemoryTraceStore` | Development, demos, CI, benchmarks | Nothing (numpy only) |
| `TraceStore` | Production with persistent memory | PostgreSQL + pgvector |

### The Framework is Model-Agnostic

| Provider | LLM | Embeddings |
|----------|-----|------------|
| Google | `ChatGoogleGenerativeAI` | `GoogleGenerativeAIEmbeddings` |
| OpenAI | `ChatOpenAI` | `OpenAIEmbeddings` |
| Anthropic | `ChatAnthropic` | (use OpenAI or Voyage) |
| Local | `ChatOllama` | `OllamaEmbeddings` |

---

## Feedback Loop (Langfuse)

The system learns from human feedback via Langfuse:

1. Agent generates a plan → logged to Langfuse
2. SME reviews and scores the trace in Langfuse
3. FeedbackPoller detects positive scores
4. Gatekeeper validates the trace (schema + sandbox + dedup)
5. Validated trace enters behavioral memory
6. Future queries retrieve this trace as a reference example

```python
from behavioral_memory import FeedbackPoller, GatekeeperPipeline

poller = FeedbackPoller(settings=settings)
gatekeeper = GatekeeperPipeline(store=store, registry=registry)

poller.poll_loop(callback=lambda trace: gatekeeper.submit(trace))
```

---

## Testing

### Run all tests (104 tests, no external services needed)

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

### Test breakdown

| Suite | Tests | What it covers |
|-------|-------|---------------|
| `tests/unit/` | 61 | Schemas, metrics, postprocessing, prompt assembly, token budget, in-memory store |
| `tests/integration/` | 3 | Schema validator + sandbox with real traces |
| `tests/e2e/` | 40 | Full pipeline: seed traces → prompt → mock LLM → metrics → gatekeeper |

### Pipeline validation

```bash
python examples/validate_pipeline.py   # 30 checks, 0 external deps
```

### Linting and type checking

```bash
make lint         # or: ruff check src/ tests/ agent/ examples/ server.py
make format       # or: ruff format src/ tests/ agent/ examples/ server.py
make typecheck    # or: mypy src/
```

---

## Gatekeeper Ablation Study (Section IV.D.5)

Tests the critical role of the gatekeeper by injecting 8 deliberately poisoned traces
(wrong conventions, broken dependencies, incorrect tools) into memory:

```bash
python examples/gatekeeper_ablation.py --verbose
```

Three conditions are compared:
1. **Baseline** — only valid seed traces (gatekeeper ON)
2. **Poisoned** — bad traces injected (gatekeeper OFF)
3. **Recovered** — gatekeeper re-enabled, bad traces filtered out

The script shows how poisoned traces degrade plan quality (PCR drops) and how the
gatekeeper catches and rejects them. Recovery restores baseline performance.

---

## Development

```bash
# Using the Makefile (recommended)
make dev          # Install all dev dependencies + pre-commit hooks
make lint         # Run ruff linter
make format       # Auto-format code
make typecheck    # Run mypy
make test         # Run all 104 tests
make ci           # Run all CI checks locally
make ablation     # Run gatekeeper ablation study
make validate     # Pipeline validation (no API keys)
make demo         # Offline demo
```

---

## Evaluation Metrics (Section IV.C)

| Metric | Description |
|--------|-------------|
| **TSA** | Tool Selection Accuracy — correct tool multiset |
| **PV** | Parameter Validity — fraction of key params correct |
| **PCR** | Plan Correctness Rate — correct tools AND >= 80% PV |
| **ESA** | Execution Sequence Accuracy — correct tool ordering |

---

## CLI Tools

```bash
behavioral-memory setup                    # Interactive .env setup
behavioral-memory demo                     # Offline demo of behavioral memory
behavioral-memory benchmark info           # Dataset summary
behavioral-memory benchmark ground-truth   # View all 30 tasks
behavioral-memory benchmark seed-traces    # View 12 seed traces
behavioral-memory benchmark tools          # View 7 tool definitions
```

---

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_STORE_URL` | `postgresql+psycopg://localhost/behavioral_memory` | pgvector connection |
| `VECTOR_STORE_COLLECTION` | `validated_traces` | Collection name |
| `FEW_SHOT_K` | `3` | Traces to retrieve per query |
| `MAX_PROMPT_TOKENS` | `3500` | Token budget for prompt |
| `SIMILARITY_DEDUP_THRESHOLD` | `0.95` | Cosine threshold for dedup |
| `SANDBOX_TIMEOUT_SECONDS` | `30` | Sandbox execution timeout |
| `FEEDBACK_SCORE_NAME` | `quality` | Langfuse score name |
| `FEEDBACK_POSITIVE_THRESHOLD` | `1.0` | Min score for positive |
| `FEEDBACK_POLL_INTERVAL` | `60` | Seconds between polls |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Store | PostgreSQL + pgvector (production) / In-memory (development) |
| Embeddings | Any LangChain Embeddings (default: Gemini) |
| LLM | Any LangChain ChatModel (default: Gemini 2.5 Pro) |
| Agent Framework | LangGraph 1.x (reference agent) |
| Observability | Langfuse |
| Config | Pydantic Settings |
| Tokenization | tiktoken |
| CLI | Typer + Rich |
| Testing | pytest (104 tests) |
| Linting | ruff + pre-commit hooks |
| Type Checking | mypy (strict) |
| Package Management | uv |

---

## Citation

If you use this software in your research, please cite our paper:

```bibtex
@inproceedings{khan2025behavioral,
  title={Behavioral Memory for Tool Orchestration: Semantic Retrieval of Validated Execution Traces in MCP-Based Agent Systems},
  author={Khan, Mehvash and Kumar, Harsh and Jangir, Rahul},
  booktitle={IEEE Conference Proceedings},
  year={2025}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
