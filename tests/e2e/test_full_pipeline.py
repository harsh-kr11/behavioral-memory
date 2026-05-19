"""End-to-end tests: full pipeline without external services.

Tests the complete behavioral memory pipeline using mocks for:
  - PostgreSQL/pgvector (replaced by in-memory storage)
  - LLM calls (replaced by a deterministic fake model)
  - Langfuse (replaced by a no-op tracer)

These tests validate that all layers compose correctly:
  Seed traces -> Registry -> Prompt assembly -> LLM (mocked) ->
  Postprocess -> Plan -> Gatekeeper -> Metrics
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from behavioral_memory.core.config import Settings
from behavioral_memory.core.schemas import (
    ExecutionTrace,
    Plan,
    ToolCall,
)
from behavioral_memory.evaluation.ground_truth import EVALUATION_TASKS
from behavioral_memory.evaluation.metrics import compute_metrics
from behavioral_memory.evaluation.seed_traces import get_seed_traces
from behavioral_memory.gatekeeper.sandbox import SandboxExecutor
from behavioral_memory.gatekeeper.schema_validator import SchemaValidator
from behavioral_memory.memory.token_budget import count_tokens, estimate_trace_tokens
from behavioral_memory.planner.postprocess import postprocess_plan
from behavioral_memory.planner.prompt import SYSTEM_PROMPT, build_prompt
from behavioral_memory.tools.mock_tools import get_tool_schemas
from behavioral_memory.tools.registry import ToolRegistry

# ── Helpers ───────────────────────────────────────────────────────


def _build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register_many(get_tool_schemas())
    return reg


def _make_fake_llm_response(steps: list[dict]) -> str:
    return json.dumps(steps)


# ── Test: seed traces are all valid ──────────────────────────────


class TestSeedTraceValidation:
    """Every seed trace must pass both schema validation and sandbox."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return _build_registry()

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def test_all_12_seed_traces_load(self):
        traces = get_seed_traces()
        assert len(traces) == 12

    def test_all_seed_traces_are_validated(self):
        for trace in get_seed_traces():
            assert trace.validated is True
            assert trace.source == "seed"

    def test_all_seed_traces_pass_schema_validation(self, registry):
        validator = SchemaValidator(registry)
        for trace in get_seed_traces():
            is_valid, failures = validator.validate(trace)
            assert is_valid, f"Seed trace '{trace.task_description[:50]}' failed: {failures}"

    def test_all_seed_traces_pass_sandbox(self, settings):
        sandbox = SandboxExecutor(settings=settings)
        for trace in get_seed_traces():
            passed, msg = sandbox.execute(trace)
            assert passed, f"Seed trace '{trace.task_description[:50]}' sandbox failed: {msg}"


# ── Test: ground truth tasks are structurally valid ──────────────


class TestGroundTruthIntegrity:
    """The 30 ground truth tasks and their gold chains are valid."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return _build_registry()

    def test_30_tasks_exist(self):
        assert len(EVALUATION_TASKS) == 30

    def test_difficulty_distribution(self):
        by_diff = {}
        for task in EVALUATION_TASKS:
            d = task["difficulty"]
            by_diff[d] = by_diff.get(d, 0) + 1
        assert by_diff == {"simple": 10, "moderate": 10, "challenging": 10}

    def test_all_gold_chains_use_known_tools(self, registry):
        for task in EVALUATION_TASKS:
            for step in task["gold_tool_chain"]:
                assert registry.has_tool(step["tool"]), f"Task {task['task_id']}: unknown tool '{step['tool']}'"

    def test_gold_chain_step_ids_are_unique(self):
        for task in EVALUATION_TASKS:
            ids = [s["step_id"] for s in task["gold_tool_chain"]]
            assert len(ids) == len(set(ids)), f"Task {task['task_id']}: duplicate step_ids"

    def test_gold_chains_pass_schema_validation(self, registry):
        validator = SchemaValidator(registry)
        for task in EVALUATION_TASKS:
            trace = ExecutionTrace(
                task_description=task["task"],
                tool_chain=[
                    ToolCall(
                        step_id=s["step_id"],
                        tool_name=s["tool"],
                        parameters=s.get("params", {}),
                    )
                    for s in task["gold_tool_chain"]
                ],
            )
            is_valid, failures = validator.validate(trace)
            assert is_valid, f"Task {task['task_id']} gold chain failed validation: {failures}"


# ── Test: prompt assembly pipeline ───────────────────────────────


class TestPromptAssemblyPipeline:
    """From query + traces + schemas -> correctly formed prompt."""

    def test_prompt_with_traces_and_schemas(self):
        traces = get_seed_traces()[:3]
        schemas = get_tool_schemas()
        prompt = build_prompt(
            query="Get total revenue and send a report",
            traces=traces,
            tool_schemas=schemas,
        )
        assert "REFERENCE EXAMPLES" in prompt
        assert "AVAILABLE TOOLS" in prompt
        assert "USER TASK" in prompt
        assert "Get total revenue" in prompt
        assert "query_database" in prompt
        for trace in traces:
            assert trace.task_description in prompt

    def test_prompt_zero_shot(self):
        schemas = get_tool_schemas()
        prompt = build_prompt(query="Test query", traces=[], tool_schemas=schemas)
        assert "REFERENCE EXAMPLES" not in prompt
        assert "AVAILABLE TOOLS" in prompt

    def test_system_prompt_has_instructions(self):
        assert "JSON array" in SYSTEM_PROMPT
        assert "step_id" in SYSTEM_PROMPT
        assert "tool_name" in SYSTEM_PROMPT

    def test_prompt_fits_in_token_budget(self):
        traces = get_seed_traces()
        schemas = get_tool_schemas()
        prompt = build_prompt(
            query="Full revenue pipeline with all traces",
            traces=traces,
            tool_schemas=schemas,
        )
        tokens = count_tokens(SYSTEM_PROMPT) + count_tokens(prompt)
        assert tokens < 15000, f"Prompt is {tokens} tokens — unreasonably large"


# ── Test: postprocessor ──────────────────────────────────────────


class TestPostprocessorEndToEnd:
    """LLM response parsing handles realistic outputs."""

    def test_parses_perfect_response(self):
        raw = json.dumps(
            [
                {
                    "step_id": "step_1",
                    "tool_name": "query_database",
                    "parameters": {"query": "SELECT 1"},
                    "depends_on": [],
                },
                {
                    "step_id": "step_2",
                    "tool_name": "generate_report",
                    "parameters": {"source_step": "step_1", "format": "csv", "title": "t"},
                    "depends_on": ["step_1"],
                },
            ]
        )
        steps = postprocess_plan(raw)
        assert len(steps) == 2
        assert steps[0].tool_name == "query_database"
        assert steps[1].depends_on == ["step_1"]

    def test_parses_markdown_fenced_response(self):
        raw = """```json
[
    {"step_id": "step_1", "tool_name": "query_database", "parameters": {"query": "SELECT 1"}, "depends_on": []}
]
```"""
        steps = postprocess_plan(raw)
        assert len(steps) == 1
        assert steps[0].tool_name == "query_database"

    def test_parses_response_with_trailing_comma(self):
        raw = """[
            {"step_id": "step_1", "tool_name": "fetch_api_data", "parameters": {"url": "http://example.com"}, "depends_on": []},
        ]"""
        steps = postprocess_plan(raw)
        assert len(steps) == 1

    def test_parses_alternative_key_names(self):
        raw = json.dumps(
            [
                {"step_id": "step_1", "tool": "query_database", "params": {"query": "SELECT 1"}},
            ]
        )
        steps = postprocess_plan(raw)
        assert steps[0].tool_name == "query_database"


# ── Test: metrics computation ────────────────────────────────────


class TestMetricsEndToEnd:
    """Metrics score correctly on gold vs predicted chains."""

    def test_perfect_prediction_scores_all_metrics(self):
        for task in EVALUATION_TASKS[:5]:
            gold = task["gold_tool_chain"]
            metrics = compute_metrics(gold, gold)
            assert metrics["tsa"] is True
            assert metrics["pv"] == 1.0
            assert metrics["pcr"] is True
            assert metrics["esa"] is True

    def test_wrong_tool_gets_zero_tsa(self):
        gold = [{"tool": "query_database", "params": {"query": "SELECT 1"}}]
        pred = [{"tool": "fetch_api_data", "params": {"url": "http://x.com"}}]
        metrics = compute_metrics(pred, gold)
        assert metrics["tsa"] is False

    def test_wrong_order_fails_esa_but_may_pass_tsa(self):
        gold = [
            {"tool": "query_database", "params": {"query": "SELECT 1"}},
            {"tool": "generate_report", "params": {"source_step": "s1", "format": "csv", "title": "t"}},
        ]
        pred = [
            {"tool": "generate_report", "params": {"source_step": "s1", "format": "csv", "title": "t"}},
            {"tool": "query_database", "params": {"query": "SELECT 1"}},
        ]
        metrics = compute_metrics(pred, gold)
        assert metrics["tsa"] is True
        assert metrics["esa"] is False


# ── Test: gatekeeper pipeline (mocked store) ────────────────────


class TestGatekeeperEndToEnd:
    """Full gatekeeper: schema -> sandbox -> dedup, with mocked vector store."""

    @pytest.fixture
    def registry(self) -> ToolRegistry:
        return _build_registry()

    def test_valid_trace_accepted(self, registry):
        validator = SchemaValidator(registry)
        sandbox = SandboxExecutor()
        trace = ExecutionTrace(
            task_description="Get revenue and generate a report",
            tool_chain=[
                ToolCall(
                    step_id="s1", tool_name="query_database", parameters={"query": "SELECT SUM(amount) FROM orders"}
                ),
                ToolCall(
                    step_id="s2",
                    tool_name="generate_report",
                    parameters={"source_step": "s1", "format": "markdown_table", "title": "Revenue"},
                ),
            ],
            source="execution",
        )
        is_valid, failures = validator.validate(trace)
        assert is_valid, f"Failures: {failures}"
        passed, msg = sandbox.execute(trace)
        assert passed, msg

    def test_unknown_tool_rejected_at_gate1(self, registry):
        validator = SchemaValidator(registry)
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[ToolCall(step_id="s1", tool_name="nonexistent_tool")],
        )
        is_valid, failures = validator.validate(trace)
        assert not is_valid
        assert any("Unknown tool" in f for f in failures)

    def test_broken_data_flow_rejected_at_gate2(self):
        sandbox = SandboxExecutor()
        trace = ExecutionTrace(
            task_description="test",
            tool_chain=[
                ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"}),
                ToolCall(
                    step_id="s2",
                    tool_name="generate_report",
                    parameters={"source_step": "s99", "format": "csv", "title": "t"},
                ),
            ],
        )
        passed, msg = sandbox.execute(trace)
        assert not passed
        assert "s99" in msg


# ── Test: PlanEngine with mocked LLM ────────────────────────────


class TestPlanEngineEndToEnd:
    """PlanEngine orchestration with a mocked LLM and mocked store."""

    def test_generate_produces_plan(self):
        expected_output = json.dumps(
            [
                {
                    "step_id": "step_1",
                    "tool_name": "query_database",
                    "parameters": {"query": "SELECT COUNT(*) FROM customers"},
                    "depends_on": [],
                },
            ]
        )

        mock_response = MagicMock()
        mock_response.content = expected_output

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        mock_store = MagicMock()

        from behavioral_memory.planner.engine import PlanEngine

        engine = PlanEngine(
            llm=mock_llm,
            store=mock_store,
            registry=_build_registry(),
        )

        plan = engine.generate(
            query="Get customer count",
            traces=[],
            tool_schemas=get_tool_schemas(),
        )

        assert isinstance(plan, Plan)
        assert plan.query == "Get customer count"
        assert len(plan.steps) == 1
        assert plan.steps[0].tool_name == "query_database"
        assert plan.token_budget_used > 0
        mock_llm.invoke.assert_called_once()

    def test_generate_with_traces_passes_them_to_prompt(self):
        seed_traces = get_seed_traces()[:2]
        expected_output = json.dumps(
            [
                {
                    "step_id": "step_1",
                    "tool_name": "query_database",
                    "parameters": {"query": "SELECT 1"},
                    "depends_on": [],
                },
            ]
        )

        mock_response = MagicMock()
        mock_response.content = expected_output
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_store = MagicMock()

        from behavioral_memory.planner.engine import PlanEngine

        engine = PlanEngine(llm=mock_llm, store=mock_store, registry=_build_registry())
        plan = engine.generate(
            query="Revenue analysis",
            traces=seed_traces,
            tool_schemas=get_tool_schemas(),
        )

        assert len(plan.retrieved_traces) == 2
        call_args = mock_llm.invoke.call_args[0][0]
        prompt_text = call_args[1].content
        assert "REFERENCE EXAMPLES" in prompt_text

    def test_generate_zero_shot(self):
        expected_output = json.dumps(
            [
                {
                    "step_id": "step_1",
                    "tool_name": "query_database",
                    "parameters": {"query": "SELECT 1"},
                    "depends_on": [],
                },
            ]
        )
        mock_response = MagicMock()
        mock_response.content = expected_output
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response
        mock_store = MagicMock()

        from behavioral_memory.planner.engine import PlanEngine

        engine = PlanEngine(llm=mock_llm, store=mock_store)
        plan = engine.generate_zero_shot(
            query="Simple query",
            tool_schemas=get_tool_schemas(),
        )

        assert len(plan.retrieved_traces) == 0


# ── Test: Langfuse tracer (offline) ─────────────────────────────


class TestLangfuseTracerOffline:
    """Verify the tracer is graceful when Langfuse is not configured."""

    def test_disabled_when_no_keys(self):
        from behavioral_memory.observability.tracer import LangfuseTracer

        settings = Settings(langfuse_secret_key="", langfuse_public_key="")
        tracer = LangfuseTracer(settings=settings)
        assert not tracer.enabled
        plan = Plan(
            query="test",
            steps=[ToolCall(step_id="s1", tool_name="query_database", parameters={"query": "SELECT 1"})],
        )
        result = tracer.log_plan(plan)
        assert result is None

    def test_flush_safe_when_disabled(self):
        from behavioral_memory.observability.tracer import LangfuseTracer

        tracer = LangfuseTracer(settings=Settings(langfuse_secret_key="", langfuse_public_key=""))
        tracer.flush()


# ── Test: FeedbackPoller (offline) ───────────────────────────────


class TestFeedbackPollerOffline:
    """Verify the poller handles missing Langfuse gracefully."""

    def test_returns_empty_when_no_client(self):
        from behavioral_memory.observability.feedback import FeedbackPoller

        settings = Settings(langfuse_secret_key="", langfuse_public_key="")
        poller = FeedbackPoller(settings=settings)
        traces = poller.fetch_positive_traces()
        assert traces == []


# ── Test: token budget utilities ─────────────────────────────────


class TestTokenBudget:
    def test_trace_token_estimate_is_positive(self):
        for trace in get_seed_traces():
            tokens = estimate_trace_tokens(trace)
            assert tokens > 0

    def test_total_seed_traces_under_budget(self):
        total = sum(estimate_trace_tokens(t) for t in get_seed_traces())
        assert total < 20000, f"All 12 seed traces use {total} tokens"


# ── Test: full round-trip (trace -> validate -> plan -> score) ───


class TestFullRoundTrip:
    """Simulate the complete flow for a single evaluation task."""

    def test_task_0_round_trip(self):
        task = EVALUATION_TASKS[0]
        registry = _build_registry()
        schemas = get_tool_schemas()
        seed_traces = get_seed_traces()[:3]

        prompt = build_prompt(
            query=task["task"],
            traces=seed_traces,
            tool_schemas=schemas,
        )
        assert "customer" in prompt.lower()

        gold_chain = task["gold_tool_chain"]
        fake_llm_output = json.dumps(
            [
                {
                    "step_id": s["step_id"],
                    "tool_name": s["tool"],
                    "parameters": s.get("params", {}),
                    "depends_on": [],
                }
                for s in gold_chain
            ]
        )

        steps = postprocess_plan(fake_llm_output)
        assert len(steps) == len(gold_chain)

        trace = ExecutionTrace(
            task_description=task["task"],
            tool_chain=steps,
            source="execution",
        )
        validator = SchemaValidator(registry)
        is_valid, failures = validator.validate(trace)
        assert is_valid, f"Failures: {failures}"

        sandbox = SandboxExecutor()
        passed, msg = sandbox.execute(trace)
        assert passed, msg

        predicted_chain = [{"tool": s.tool_name, "params": s.parameters} for s in steps]
        metrics = compute_metrics(predicted_chain, gold_chain)
        assert metrics["tsa"] is True
        assert metrics["esa"] is True
        assert metrics["pcr"] is True

    def test_all_simple_tasks_round_trip(self):
        """Simulate perfect prediction for all simple tasks."""
        registry = _build_registry()
        validator = SchemaValidator(registry)
        sandbox = SandboxExecutor()

        simple_tasks = [t for t in EVALUATION_TASKS if t["difficulty"] == "simple"]
        for task in simple_tasks:
            gold = task["gold_tool_chain"]
            fake_output = json.dumps(
                [
                    {
                        "step_id": s["step_id"],
                        "tool_name": s["tool"],
                        "parameters": s.get("params", {}),
                        "depends_on": [],
                    }
                    for s in gold
                ]
            )
            steps = postprocess_plan(fake_output)
            trace = ExecutionTrace(
                task_description=task["task"],
                tool_chain=steps,
                source="execution",
            )
            is_valid, _ = validator.validate(trace)
            assert is_valid, f"Task {task['task_id']} failed validation"
            passed, _ = sandbox.execute(trace)
            assert passed, f"Task {task['task_id']} failed sandbox"

    def test_all_challenging_tasks_round_trip(self):
        """Simulate perfect prediction for all challenging tasks."""
        registry = _build_registry()
        validator = SchemaValidator(registry)
        sandbox = SandboxExecutor()

        challenging = [t for t in EVALUATION_TASKS if t["difficulty"] == "challenging"]
        for task in challenging:
            gold = task["gold_tool_chain"]
            fake_output = json.dumps(
                [
                    {
                        "step_id": s["step_id"],
                        "tool_name": s["tool"],
                        "parameters": s.get("params", {}),
                        "depends_on": [],
                    }
                    for s in gold
                ]
            )
            steps = postprocess_plan(fake_output)
            trace = ExecutionTrace(
                task_description=task["task"],
                tool_chain=steps,
                source="execution",
            )
            is_valid, failures = validator.validate(trace)
            assert is_valid, f"Task {task['task_id']} gold chain failed validation: {failures}"
            passed, msg = sandbox.execute(trace)
            assert passed, f"Task {task['task_id']} failed sandbox: {msg}"


# ── Test: registry completeness ──────────────────────────────────


class TestRegistryCompleteness:
    def test_7_tools_registered(self):
        reg = _build_registry()
        assert len(reg) == 7

    def test_all_tools_have_descriptions(self):
        for schema in get_tool_schemas():
            assert len(schema.description) > 10

    def test_all_tools_have_required_params(self):
        for schema in get_tool_schemas():
            assert len(schema.required_params) >= 1

    def test_prompt_formatting(self):
        reg = _build_registry()
        formatted = reg.format_for_prompt()
        assert "query_database" in formatted
        assert "REQUIRED" in formatted
        assert len(formatted) > 500


# ── Test: serialization round-trip ───────────────────────────────


class TestSerializationRoundTrip:
    """TraceStore._trace_to_doc and _doc_to_trace are inverses."""

    def test_round_trip_preserves_data(self):
        from behavioral_memory.memory.store import TraceStore

        trace = ExecutionTrace(
            task_description="Get revenue by category",
            tool_chain=[
                ToolCall(
                    step_id="s1",
                    tool_name="query_database",
                    parameters={"query": "SELECT category, SUM(revenue) FROM products GROUP BY category"},
                ),
                ToolCall(
                    step_id="s2",
                    tool_name="generate_report",
                    parameters={"source_step": "s1", "format": "markdown_table", "title": "Revenue Report"},
                ),
            ],
            validated=True,
            source="seed",
            metadata={"explanation": "Revenue uses order_items"},
        )

        doc = TraceStore._trace_to_doc(trace)
        recovered = TraceStore._doc_to_trace(doc)
        assert recovered is not None
        assert recovered.task_description == trace.task_description
        assert len(recovered.tool_chain) == 2
        assert recovered.tool_chain[0].tool_name == "query_database"
        assert recovered.tool_chain[1].parameters["format"] == "markdown_table"
        assert recovered.validated is True
        assert recovered.source == "seed"
        assert recovered.metadata["explanation"] == "Revenue uses order_items"

    def test_all_seed_traces_serialize_round_trip(self):
        from behavioral_memory.memory.store import TraceStore

        for trace in get_seed_traces():
            doc = TraceStore._trace_to_doc(trace)
            recovered = TraceStore._doc_to_trace(doc)
            assert recovered is not None
            assert recovered.task_description == trace.task_description
            assert len(recovered.tool_chain) == len(trace.tool_chain)
