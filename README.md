# behavioral-memory

**Validated execution traces as memory for MCP-based agent tool orchestration.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![CI](https://github.com/SteveGates11/behavioral-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/SteveGates11/behavioral-memory/actions/workflows/ci.yml)

A retrieval-based framework that uses a memory bank of validated execution traces to guide LLM tool orchestration during inference. Instead of relying on static few-shot examples, the system dynamically retrieves semantically similar, validated traces from past successful executions — giving your agent **institutional memory** that improves with every interaction.

Based on the IEEE paper: *"Behavioral Memory for Tool Orchestration: Semantic Retrieval of Validated Execution Traces in MCP-Based Agent Systems"*

---

## Key Results

On a 30-task benchmark with 7 MCP tools:

| Metric | Zero-Shot | Static Few-Shot | **Proposed** |
|--------|-----------|----------------|-------------|
| Plan Correctness (PCR) | 33.3% | 50.0% | **63.3%** |
| Tool Selection (TSA) | 63.3% | 70.0% | **83.3%** |
| Sequence Accuracy (ESA) | 63.3% | 70.0% | **83.3%** |

McNemar's test: **p = 0.004** vs zero-shot.

---

## How It Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  1. BEHAVIORAL LAYER                                │
│     Retrieve top-k similar traces from pgvector     │
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

### 1. Bring Your Own Agent (library)

Install the framework and plug it into your existing agent:

```bash
pip install behavioral-memory
```

```python
from behavioral_memory import TraceStore, PlanEngine, ToolRegistry
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # or any provider

llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings()

store = TraceStore(embeddings=embeddings, connection_url="postgresql+psycopg://...")
registry = ToolRegistry()
engine = PlanEngine(llm=llm, store=store, registry=registry)

plan = engine.generate(query="Get revenue data and email a report")
```

### 2. Run the Reference Agent (LangGraph 1.x)

Clone the repo and run the complete system:

```bash
git clone https://github.com/SteveGates11/behavioral-memory.git
cd behavioral-memory
pip install -e ".[agent]"

python -m agent.app "Build a revenue analysis pipeline"
```

---

## Installation

### Prerequisites

- Python 3.11+
- PostgreSQL with [pgvector](https://github.com/pgvector/pgvector) extension

### Install with uv (recommended)

```bash
uv add behavioral-memory           # core framework
uv add behavioral-memory[agent]    # + reference LangGraph agent
uv add behavioral-memory[eval]     # + evaluation/statistics
uv add behavioral-memory[all]      # everything
```

### Install with pip

```bash
pip install behavioral-memory
pip install behavioral-memory[agent,eval]
```

### Configure

```bash
cp .env.example .env
# Edit .env with your credentials
```

| Variable | Required | Description |
|----------|----------|-------------|
| `VECTOR_STORE_URL` | Yes | PostgreSQL+pgvector connection string |
| `GOOGLE_API_KEY` | For reference agent | Gemini API key |
| `LANGFUSE_SECRET_KEY` | For feedback loop | Langfuse secret key |
| `LANGFUSE_PUBLIC_KEY` | For feedback loop | Langfuse public key |

---

## Architecture

### Project Structure

```
behavioral-memory/
├── src/behavioral_memory/     # The pip-installable library
│   ├── core/                  #   Schemas, config, exceptions
│   ├── memory/                #   Behavioral Layer (TraceStore, dedup, token budget)
│   ├── tools/                 #   Tool Layer (MCP client, registry, mock tools)
│   ├── planner/               #   Executive Layer (PlanEngine, prompt, postprocess)
│   ├── gatekeeper/            #   Gatekeeper (schema validator, sandbox, dedup gate)
│   ├── observability/         #   Langfuse (tracer, feedback poller, annotation)
│   └── evaluation/            #   Benchmark (30 tasks, metrics, statistics)
├── agent/                     # Reference LangGraph 1.x agent
│   ├── graph.py               #   StateGraph definition
│   ├── state.py               #   Agent state
│   └── nodes/                 #   Graph nodes (retrieve, plan, execute, observe)
├── tests/                     # Unit + integration tests
└── examples/                  # Usage examples
```

### The Framework is Model-Agnostic

The library accepts any LangChain-compatible model:

| Provider | LLM | Embeddings |
|----------|-----|------------|
| Google | `ChatGoogleGenerativeAI` | `GoogleGenerativeAIEmbeddings` |
| OpenAI | `ChatOpenAI` | `OpenAIEmbeddings` |
| Anthropic | `ChatAnthropic` | (use OpenAI or Voyage) |
| Local | `ChatOllama` | `OllamaEmbeddings` |

---

## Feedback Loop

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

# Auto-learn in the background
poller.poll_loop(callback=lambda trace: gatekeeper.submit(trace))
```

---

## Evaluation

### Reproduce Paper Results

```bash
pip install behavioral-memory[agent,eval]
python examples/run_benchmark.py
```

### CLI Tools

```bash
behavioral-memory benchmark info          # Dataset summary
behavioral-memory benchmark ground-truth  # View all 30 tasks
behavioral-memory benchmark seed-traces   # View 12 seed traces
behavioral-memory benchmark tools         # View 7 tool definitions
```

### Metrics (Section IV.C)

| Metric | Description |
|--------|-------------|
| **TSA** | Tool Selection Accuracy — correct tool multiset |
| **PV** | Parameter Validity — fraction of key params correct |
| **PCR** | Plan Correctness Rate — correct tools AND >= 80% PV |
| **ESA** | Execution Sequence Accuracy — correct tool ordering |

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
| Vector Store | PostgreSQL + pgvector |
| Embeddings | Any LangChain Embeddings (default: Gemini) |
| LLM | Any LangChain ChatModel (default: Gemini 2.5 Pro) |
| Agent Framework | LangGraph 1.x (reference agent) |
| Observability | Langfuse |
| Config | Pydantic Settings |
| Tokenization | tiktoken |
| CLI | Typer + Rich |
| Testing | pytest |
| Linting | ruff |
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
