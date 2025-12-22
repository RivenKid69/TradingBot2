# Makefile for AI-Powered Quantitative Research Platform
# Cross-platform build system for Cython/C++ extensions + development tools
#
# Usage:
#   make build          # Build all extensions in-place
#   make clean          # Remove build artifacts
#   make rebuild        # Clean + build
#   make test           # Run tests after build
#   make verify-hash    # Verify build reproducibility
#   make format         # Format code (isort + black)
#   make lint           # Lint code with ruff and enforce black style
#
# Supported Platforms: Windows (MSVC), Linux (GCC), macOS (Clang)

# Platform detection
ifeq ($(OS),Windows_NT)
    PLATFORM := Windows
    PYTHON := python
    RM := del /Q
    RMDIR := rmdir /S /Q
    MKDIR := mkdir
    SEP := \\
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        PLATFORM := macOS
    else
        PLATFORM := Linux
    endif
    PYTHON := python3
    RM := rm -f
    RMDIR := rm -rf
    MKDIR := mkdir -p
    SEP := /
endif

# Build directories
BUILD_DIR := build
DIST_DIR := dist
HASH_REPORT := build_hash_report.json
CLEAN_SCRIPT := tools/clean_artifacts.py
VERIFY_SCRIPT := tools/verify_hash_report.py
CCEA_VERSION := $(shell $(PYTHON) -c "import tomllib, pathlib; print(tomllib.loads(pathlib.Path('pyproject.toml').read_text(encoding='utf-8'))['project']['version'])")

# Colors (for Unix-like systems)
ifndef NO_COLOR
    GREEN := \033[0;32m
    YELLOW := \033[0;33m
    RED := \033[0;31m
    NC := \033[0m
else
    GREEN :=
    YELLOW :=
    RED :=
    NC :=
endif

.PHONY: all build clean check-clean rebuild test verify-hash install-build-deps help
.PHONY: format lint no-trade-mask-sample check
.PHONY: lock-cpu lock-gpu lockfiles
.PHONY: dist-cloud dist-agent dist-all artifact-check-cloud

# Default target
all: build

help:
	@echo "$(GREEN)AI-Powered Quantitative Research Platform - Build System$(NC)"
	@echo ""
	@echo "$(YELLOW)Build targets:$(NC)"
	@echo "  make build             Build all Cython/C++ extensions in-place"
	@echo "  make clean             Remove build artifacts and compiled extensions"
	@echo "  make check-clean       Validate that no generated artifacts remain"
	@echo "  make rebuild           Clean then build"
	@echo "  make test              Run tests after building"
	@echo "  make verify-hash       Verify build hash report exists"
	@echo "  make lock-cpu          Regenerate requirements-cpu.lock.txt (Python 3.12)"
	@echo "  make lock-gpu          Regenerate requirements-gpu.lock.txt (Python 3.12)"
	@echo "  make install-build-deps Install build dependencies"
	@echo ""
	@echo "$(YELLOW)Development targets:$(NC)"
	@echo "  make format            Format Python code (isort + black)"
	@echo "  make lint              Lint code with ruff and check black style"
	@echo "  make check             Quick syntax check (no build)"
	@echo "  make no-trade-mask-sample Run no-trade mask sample"
	@echo ""
	@echo "$(YELLOW)Platform:$(NC) $(PLATFORM)"
	@echo "$(YELLOW)Python:$(NC) $(shell $(PYTHON) --version)"

# Install build dependencies
install-build-deps:
	@echo "$(GREEN)Installing build dependencies...$(NC)"
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-build.txt
	@echo "$(GREEN)Build dependencies installed.$(NC)"

# Build all extensions
build:
	@echo "$(GREEN)Building native extensions for $(PLATFORM)...$(NC)"
	@echo "$(YELLOW)This may take a few minutes on first build.$(NC)"
	$(PYTHON) setup.py build_ext --inplace
	@echo ""
	@echo "$(GREEN)[OK] Build complete!$(NC)"
	@echo "$(YELLOW)Hash report:$(NC) $(HASH_REPORT)"

# Clean build artifacts
clean:
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	$(PYTHON) $(CLEAN_SCRIPT)
	@echo "$(GREEN)[OK] Clean complete.$(NC)"

check-clean:
	@echo "$(YELLOW)Checking for generated artifacts...$(NC)"
	$(PYTHON) $(CLEAN_SCRIPT) --check-only
	@echo "$(GREEN)[OK] Worktree free of generated artifacts.$(NC)"

# Rebuild from scratch
rebuild: clean build

# Run tests (requires pytest)
test: build
	@echo "$(GREEN)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "$(GREEN)[OK] Tests complete.$(NC)"

# Run tests with coverage tracking (PM-005 support)
# Generates coverage.xml for CI artifact upload and threshold enforcement
test-coverage: build
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(PYTHON) -m pytest tests/ -v --tb=short \
		--cov=. \
		--cov-report=xml:coverage.xml \
		--cov-report=html:htmlcov \
		--cov-report=term-missing \
		--cov-fail-under=0
	@echo "$(GREEN)[OK] Tests with coverage complete.$(NC)"
	@echo "$(GREEN)Coverage report: coverage.xml, htmlcov/index.html$(NC)"

# ============================================================================
# Phase 9 (P1) - Zone-separated distributions + artifact content checks
# ============================================================================

dist-cloud:
	@echo "$(GREEN)Building Cloud distribution...$(NC)"
	$(PYTHON) tools/build_zone_distributions.py --cloud --out $(DIST_DIR)

dist-agent:
	@echo "$(GREEN)Building Agent distribution...$(NC)"
	$(PYTHON) tools/build_zone_distributions.py --agent --out $(DIST_DIR)

dist-all:
	@echo "$(GREEN)Building Cloud + Agent distributions...$(NC)"
	$(PYTHON) tools/build_zone_distributions.py --cloud --agent --out $(DIST_DIR)

artifact-check-cloud: dist-cloud
	@echo "$(GREEN)Scanning Cloud artifact contents...$(NC)"
	$(PYTHON) -m ccea.guardrails.build_artifact_check --artifact $(DIST_DIR)/ccea_cloud-$(CCEA_VERSION)-py3-none-any.whl --fail-on-violation

# Verify hash report exists
verify-hash:
	@echo "$(YELLOW)Verifying build hash report...$(NC)"
	$(PYTHON) $(VERIFY_SCRIPT) --report $(HASH_REPORT) --root . --require-all-artifacts
	@echo "$(GREEN)[OK] Hash report verified.$(NC)"
# Quick syntax check (no build)
check:
	@echo "$(YELLOW)Running syntax checks...$(NC)"
	$(PYTHON) -m py_compile setup.py
	@echo "$(GREEN)[OK] Syntax OK.$(NC)"

# ============================================================================
# Development Targets (existing)
# ============================================================================

format:
	@echo "$(GREEN)Formatting imports with isort...$(NC)"
	$(PYTHON) -m isort .
	@echo "$(GREEN)Formatting code with black...$(NC)"
	$(PYTHON) -m black .
	@echo "$(GREEN)[OK] Format complete.$(NC)"

lint:
	@echo "$(GREEN)Linting code with ruff...$(NC)"
	$(PYTHON) -m ruff check .
	@echo "$(GREEN)Checking Black formatting...$(NC)"
	$(PYTHON) -m black --check .
	@echo "$(GREEN)[OK] Lint complete.$(NC)"

no-trade-mask-sample:
	@echo "$(GREEN)Running no-trade mask sample...$(NC)"
	$(PYTHON) tests/run_no_trade_mask_sample.py

lock-cpu:
	@echo "$(GREEN)Regenerating CPU lockfile (Python 3.12.x)...$(NC)"
	$(PYTHON) -m piptools compile --extra cpu --resolver=backtracking --output-file=requirements-cpu.lock.txt pyproject.toml
	@echo "$(GREEN)[OK] requirements-cpu.lock.txt updated.$(NC)"

lock-gpu:
	@echo "$(GREEN)Regenerating GPU lockfile (Python 3.12.x)...$(NC)"
	$(PYTHON) -m piptools compile --extra gpu --resolver=backtracking --output-file=requirements-gpu.lock.txt pyproject.toml
	@echo "$(GREEN)[OK] requirements-gpu.lock.txt updated.$(NC)"

lockfiles: lock-cpu lock-gpu
