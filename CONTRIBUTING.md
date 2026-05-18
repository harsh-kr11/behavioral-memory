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

# Set up pre-commit hooks (runs ruff automatically on each commit)
uv run pre-commit install

# Verify hooks are working
uv run pre-commit run --all-files
```

### Alternative setup with pip

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
pip install -e ".[dev,eval]"
pre-commit install
```

### Environment setup

```bash
# Create your .env from the template
cp .env.example .env
# Or use the interactive setup command:
behavioral-memory setup
```

## Pre-commit Hooks

We use [pre-commit](https://pre-commit.com) with `ruff` to enforce code quality on every commit:

- **ruff check** — lints for errors, import sorting, style issues
- **ruff format** — auto-formats code

If a commit is rejected by the hook, ruff will auto-fix most issues. Just re-stage the
fixed files and commit again:

```bash
git add -u
git commit -m "your message"
```

To run checks manually at any time:

```bash
uv run ruff check src/ tests/ agent/      # Lint only
uv run ruff check --fix src/ tests/ agent/ # Lint + auto-fix
uv run ruff format src/ tests/ agent/      # Format
uv run mypy src/                           # Type check
uv run pytest                              # Run all tests
```

## Running Tests

```bash
# All 96 tests (no external services needed)
uv run pytest

# With verbose output
uv run pytest -v

# Just e2e tests
uv run pytest tests/e2e/ -v
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass: `uv run pytest`
4. Ensure code passes linting: `uv run ruff check --fix src/ tests/`
5. Ensure pre-commit hooks pass: `uv run pre-commit run --all-files`
6. Update documentation if needed.
7. Submit a pull request with a clear description.

## Code Style

- We use `ruff` for linting and formatting (see `pyproject.toml` for config).
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
