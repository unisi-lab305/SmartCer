# Makefile for SmartCer
#
# Prerequisites: pip install -e .[dev]
#
# Run `make help` to list the available targets.

.PHONY: test test-unit test-integration test-coverage bench version clean help

# default target
.DEFAULT_GOAL := help

# test targets
test: ## Run all tests (unit + integration)
	@echo "Running all tests..."
	python -m pytest test/ -v

test-unit: ## Run only unit tests
	@echo "Running unit tests..."
	python -m pytest test/unit/ -v

test-integration: ## Run only integration tests
	@echo "Running integration tests..."
	python -m pytest test/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	python -m pytest test/ --cov=smartcer --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "Coverage report generated in htmlcov/index.html"

# benchmark
bench: ## Run the step benchmark (one year of the local Leather scenario, 3 episodes)
	python benchmarks/bench_step.py

# utility targets
version: ## Verify version consistency between pyproject.toml and smartcer/__init__.py
	@PYPROJECT_VER=$$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/'); \
	INIT_VER=$$(grep "^__version__ = " smartcer/__init__.py | sed "s/__version__ = '\\(.*\\)'/\\1/"); \
	if [ -n "$$PYPROJECT_VER" ] && [ "$$PYPROJECT_VER" = "$$INIT_VER" ]; then \
		echo "Version OK: $$PYPROJECT_VER"; \
	else \
		echo "ERROR: version mismatch - pyproject.toml=$$PYPROJECT_VER, smartcer/__init__.py=$$INIT_VER"; \
		exit 1; \
	fi

clean: ## Remove test cache, coverage and build files
	@echo "Cleaning test and build artifacts..."
	rm -rf .pytest_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete."

help: ## Show this help message
	@echo "SmartCer - development commands"
	@echo "==============================="
	@echo ""
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
