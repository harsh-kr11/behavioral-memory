# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-17

### Added

- Initial release of behavioral-memory framework
- Three-layer architecture: Behavioral Layer, Tool Layer, Executive Layer
- TraceStore (PostgreSQL + pgvector) and InMemoryTraceStore (zero-infra) for semantic retrieval
- GatekeeperPipeline: schema validation, sandboxed execution, semantic dedup (cosine 0.95 threshold)
- Langfuse integration: trace logging, feedback polling, annotation handler
- PlanEngine: model-agnostic plan generation with any LangChain-compatible chat model
- Reference LangGraph 1.x agent with interactive mode and LangGraph server support
- 30-task benchmark with 7 MCP tools and 12 seed traces
- Evaluation metrics: TSA, PV, PCR, ESA with bootstrap CI and McNemar's test
- Gatekeeper ablation study script (Section IV.D.5 — memory poisoning experiment)
- CLI for benchmarks, demos, memory management, and dataset inspection
- Full CI/CD with GitHub Actions (lint, typecheck, test on Python 3.11/3.12/3.13)
- Makefile for common development tasks
- PEP 561 `py.typed` marker for downstream type checking
- Apache 2.0 license
