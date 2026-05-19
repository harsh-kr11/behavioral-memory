.PHONY: help install dev lint format typecheck test test-unit test-e2e demo benchmark validate clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install core package
	uv sync

dev:  ## Install with all dev dependencies
	uv sync --extra dev --extra eval --extra agent --extra server
	uv run pre-commit install

lint:  ## Run linter
	uv run ruff check src/ tests/ agent/ examples/ server.py

format:  ## Format code
	uv run ruff format src/ tests/ agent/ examples/ server.py

typecheck:  ## Run mypy type checker
	uv run mypy src/

test:  ## Run all tests
	uv run pytest tests/ -v --tb=short

test-unit:  ## Run unit tests only
	uv run pytest tests/unit/ -v

test-e2e:  ## Run end-to-end tests only
	uv run pytest tests/e2e/ -v

demo:  ## Run offline demo (no API keys needed)
	uv run behavioral-memory demo

benchmark:  ## Run live benchmark (requires GOOGLE_API_KEY)
	uv run python examples/run_live_benchmark.py

ablation:  ## Run gatekeeper ablation study
	uv run python examples/gatekeeper_ablation.py --verbose

validate:  ## Validate full pipeline (no API keys needed)
	uv run python examples/validate_pipeline.py

clean:  ## Clean build artifacts
	rm -rf dist/ build/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ci: lint format typecheck test  ## Run all CI checks locally
