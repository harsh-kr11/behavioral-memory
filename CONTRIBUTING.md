# Contributing to behavioral-memory

Thank you for your interest in contributing! This document provides guidelines
for contributing to the project.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/SteveGates11/behavioral-memory.git
cd behavioral-memory

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (including dev)
uv sync --all-extras

# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/

# Run type checker
uv run mypy src/
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass: `uv run pytest`
4. Ensure code passes linting: `uv run ruff check --fix src/ tests/`
5. Update documentation if needed.
6. Submit a pull request with a clear description.

## Code Style

- We use `ruff` for linting and formatting.
- We use `mypy` in strict mode for type checking.
- All public functions and classes must have docstrings.
- Follow existing code patterns for consistency.

## Commit Messages

Use clear, descriptive commit messages. Prefix with the area of change:

- `core:` for changes to core schemas/config
- `memory:` for behavioral layer changes
- `planner:` for executive layer changes
- `gatekeeper:` for gatekeeper pipeline changes
- `eval:` for evaluation framework changes
- `agent:` for reference agent changes
- `docs:` for documentation
- `ci:` for CI/CD changes

## License

By contributing, you agree that your contributions will be licensed under the
Apache License 2.0.
