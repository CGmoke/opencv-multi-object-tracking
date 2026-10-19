# Lightweight development entry points. `make help` lists targets.
PY ?= python3
VENV ?= .venv

.PHONY: help venv install install-dev lint fmt test test-cov build clean smoke

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv: ## Create .venv if missing
	test -d $(VENV) || $(PY) -m venv $(VENV)

install: ## Install the package (runtime deps)
	$(PY) -m pip install -e .

install-dev: ## Install with dev + optional extras
	$(PY) -m pip install -e ".[dev,hungarian]"

lint: ## Static checks
	$(PY) -m ruff check src tests

fmt: ## Format (writes files)
	$(PY) -m ruff format src tests

test: ## Run the test suite
	$(PY) -m pytest

test-cov: ## Run tests with coverage (needs pytest-cov)
	$(PY) -m pytest --cov=mot --cov-report=term-missing

build: ## Build sdist + wheel and validate metadata
	$(PY) -m build
	$(PY) -m twine check dist/*

smoke: ## Headless CLI smoke test: config dump + tracker listing
	PYTHONPATH=src $(PY) -m mot --dump-config >/dev/null
	PYTHONPATH=src $(PY) -m mot --list-trackers

clean: ## Remove build/test artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
