# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2025-XX-XX

### Added

- Initial release of behavioral-memory framework
- Three-layer architecture: Behavioral Layer, Tool Layer, Executive Layer
- TraceStore with pgvector for semantic retrieval of execution traces
- GatekeeperPipeline: schema validation, sandboxed execution, semantic dedup
- Langfuse integration: trace logging, feedback polling, annotation handler
- PlanEngine: model-agnostic plan generation with any LangChain chat model
- Reference LangGraph 1.x agent with full paper implementation
- 30-task benchmark with 7 tools and 12 seed traces
- Evaluation metrics: TSA, PV, PCR, ESA with bootstrap CI and McNemar's test
- CLI for benchmarks, memory management, and dataset inspection
- Apache 2.0 license
