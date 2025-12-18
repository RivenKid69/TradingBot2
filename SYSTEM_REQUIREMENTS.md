# System Requirements

Minimum system requirements and recommended configurations for the CustodiaCloud codebase (this repository).

## Table of Contents

- [Python & Runtime](#python--runtime)
- [Dependency Locking](#dependency-locking)
- [Hardware Requirements](#hardware-requirements)
- [Operating System Support](#operating-system-support)
- [C++ Toolchain](#c-toolchain)
- [GPU Requirements (Optional)](#gpu-requirements-optional)
- [Dependency Installation](#dependency-installation)
- [Build & CI Workflow](#build--ci-workflow)
- [Verification](#verification)

---

## Python & Runtime

### Required

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.12.x | Required. 3.12.0+ for full feature support |
| **pip** | 23.0+ | For installing dependencies |
| **venv** | Built-in | Recommended for isolation |

### Why Python 3.12?

- **Performance**: 5-10% faster than 3.11 due to interpreter optimizations
- **Typing**: Full support for `TypedDict`, `ParamSpec`, `TypeVarTuple`
- **Error messages**: Improved exception messages for debugging
- **Stable ABI**: Required for Cython 3.0.10 compatibility

### Verification

```bash
python --version
# Python 3.12.x

pip --version
# pip 23.x or higher
```

## Dependency Locking

- Lock files: `requirements-cpu.lock.txt` and `requirements-gpu.lock.txt` (generated with Python 3.12.x + `pip-tools>=7.4`).
- Regenerate with `make lock-cpu` / `make lock-gpu` after updating dependencies; commit updated files with the refreshed **Generated** date.
- CI and developers install from the lock files first, then layer `requirements-dev.txt` for tooling.
- `make verify-hash` enforces that compiled artifacts match the recorded hashes from the build.

---

## Hardware Requirements

### Minimum (Development/Backtesting)

| Resource | Requirement | Notes |
|----------|-------------|-------|
| **CPU** | 4 cores, 2.5GHz | Intel i5 / AMD Ryzen 5 or better |
| **RAM** | 16 GB | 32 GB recommended for large datasets |
| **Storage** | 50 GB SSD | For data cache and model checkpoints |
| **Network** | 10 Mbps | For live data streaming |

### Recommended (Training)

| Resource | Requirement | Notes |
|----------|-------------|-------|
| **CPU** | 8+ cores, 3.5GHz | Intel i7/i9 / AMD Ryzen 7/9 |
| **RAM** | 32-64 GB | For parallel training (PBT) |
| **Storage** | 200+ GB NVMe SSD | Fast I/O for data loading |
| **GPU** | NVIDIA RTX 3080+ | 10+ GB VRAM (optional but 5-10x faster) |
| **Network** | 100 Mbps | Low-latency connection for live trading |

### Production (Live Trading)

| Resource | Requirement | Notes |
|----------|-------------|-------|
| **CPU** | 8+ cores | Low-latency inference |
| **RAM** | 32+ GB | For model + data in memory |
| **Storage** | 100+ GB SSD | Logs, state persistence |
| **Network** | Dedicated | Co-location recommended for HFT |
| **UPS** | Required | Uninterrupted power supply |

---

## Operating System Support

### Fully Supported

| OS | Version | Architecture | Notes |
|----|---------|--------------|-------|
| **Windows** | 10/11 | x64 | Primary development platform |
| **Ubuntu** | 22.04 LTS | x64 | Production recommended |
| **macOS** | 13+ (Ventura) | x64, ARM64 | Apple Silicon native support |

### Partially Supported

| OS | Version | Notes |
|----|---------|-------|
| **Windows** | Server 2019+ | No GUI components |
| **Debian** | 12 (Bookworm) | Requires manual Python 3.12 install |
| **RHEL/Rocky** | 9+ | Enterprise support |
| **WSL2** | Ubuntu 22.04 | Works but with I/O overhead |

### Not Supported

- Windows 7/8
- macOS 12 (Monterey) and earlier
- 32-bit systems
- ARM64 Linux (Raspberry Pi, etc.)

---

## C++ Toolchain

Required for building Cython/C++ extensions. See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for details.

### Windows

| Tool | Version | Notes |
|------|---------|-------|
| **Visual Studio** | 2022 (v17.x) | Community Edition sufficient |
| **MSVC** | v143 (19.38+) | C++17 support required |
| **Windows SDK** | 10.0.19041+ | Latest recommended |

**Installation**:
```
Visual Studio Installer -> Workloads -> "Desktop development with C++"
```

### Linux

| Tool | Version | Notes |
|------|---------|-------|
| **GCC** | 11+ | C++17 support required |
| **Clang** | 14+ | Alternative to GCC |
| **make** | 4.0+ | Build automation |
| **python3-dev** | 3.12 | Python headers |

**Installation (Ubuntu)**:
```bash
sudo apt-get update
sudo apt-get install build-essential python3-dev python3.12-dev
```

### macOS

| Tool | Version | Notes |
|------|---------|-------|
| **Xcode CLT** | 14+ | Command Line Tools |
| **Clang** | 14+ | Bundled with Xcode |

**Installation**:
```bash
xcode-select --install
```

---

## GPU Requirements (Optional)

GPU acceleration provides 5-10x speedup for training. CPU-only operation is fully supported.

### NVIDIA GPUs (Recommended)

| GPU | VRAM | Compute Capability | Notes |
|-----|------|-------------------|-------|
| **RTX 4090** | 24 GB | 8.9 | Best performance |
| **RTX 4080** | 16 GB | 8.9 | Excellent |
| **RTX 3090** | 24 GB | 8.6 | Great value |
| **RTX 3080** | 10 GB | 8.6 | Minimum recommended |
| **RTX 3070** | 8 GB | 8.6 | Adequate for small models |
| **A100** | 40/80 GB | 8.0 | Data center (PBT training) |

### CUDA Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| **NVIDIA Driver** | 525.60+ | For CUDA 12.1 |
| **CUDA Toolkit** | 12.1 | Optional (bundled with PyTorch) |
| **cuDNN** | 8.9.x | Bundled with PyTorch |

**Verification**:
```bash
nvidia-smi
# Should show driver version >= 525

python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

### AMD GPUs (Experimental)

ROCm support is experimental. Use Linux only.

| GPU | VRAM | Notes |
|-----|------|-------|
| **RX 7900 XTX** | 24 GB | ROCm 5.7+ |
| **RX 6900 XT** | 16 GB | ROCm 5.4+ |

---

## Dependency Installation

### Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate (Windows)
.venv\Scripts\activate
# Or (Linux/macOS)
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
# CPU-only:
pip install -r requirements-cpu.lock.txt

# GPU (CUDA 12.1):
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu.lock.txt

# Development:
pip install -r requirements-dev.txt

# 5. Build native extensions (preferred)
pip install -r requirements-build.txt
make build         # builds in-place and writes build_hash_report.json
make verify-hash   # validates recorded hashes
make test          # runs pytest with compiled extensions
# (fallback) python setup.py build_ext --inplace
```

### Using pyproject.toml

```bash
# Core dependencies only
pip install -e .

# With CPU PyTorch
pip install -e ".[cpu]"

# With GPU PyTorch
pip install -e ".[gpu]"

# Full development environment (CPU)
pip install -e ".[full-cpu]"

# Full development environment (GPU)
pip install -e ".[full-gpu]"
```

### Lock Files

For reproducible builds, use lock files:

| File | Purpose |
|------|---------|
| `requirements-cpu.lock.txt` | Exact versions for CPU-only |
| `requirements-gpu.lock.txt` | Exact versions for GPU (CUDA 12.1) |
| `requirements-build.txt` | Build dependencies (Cython, numpy) |
| `requirements-dev.txt` | Development tools (pytest, mypy, etc.) |

---

## Build & CI Workflow

- Canonical commands: `make build`, `make verify-hash`, `make test`, and `make clean && make check-clean` (enforces a clean tree with no generated artifacts).
- `make verify-hash` re-hashes compiled extensions against `build_hash_report.json` and fails on mismatches.
- GitHub Actions workflow `.github/workflows/build-and-test.yml` runs on Linux (GCC) and Windows (MSVC) with Python 3.12.x using the CPU lockfile plus `requirements-dev.txt`.
- Lock files must remain current and include their generation date (see [Dependency Locking](#dependency-locking)).
- Release gating: follow `RELEASE_CHECKLIST.md` for version bumps, changelog updates, test-matrix coverage, hash verification, and dependency audits before tagging.
- Local checks: install pre-commit hooks (`pre-commit install`) to enforce ruff, isort, black, trailing-whitespace, and end-of-file fixes before commits.

### Known Good Toolchains (CI baselines)

| Platform | Python | Compiler | Make | Notes |
|----------|--------|----------|------|-------|
| Ubuntu 22.04 LTS (CI) | 3.12.1 | GCC 12.3 | GNU Make 4.3 | `ubuntu-latest` runner; validates build, hash report, and pytest |
| Windows 11/Server 2022 (CI) | 3.12.1 | MSVC v143 (19.38+) | GNU Make 4.4 (Chocolatey) | `windows-latest` runner; builds extensions with Visual Studio 2022 toolchain |

---

## Verification

### 1. Python Environment

```bash
python --version        # Python 3.12.x
pip --version           # pip 23.x+
python -c "import sys; print(sys.executable)"  # Should be in .venv
```

### 2. Core Dependencies

```bash
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gymnasium; print(f'gymnasium: {gymnasium.__version__}')"
python -c "import stable_baselines3; print(f'SB3: {stable_baselines3.__version__}')"
```

### 3. GPU (if applicable)

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 4. Native Extensions

```bash
python -c "import obs_builder; print('[OK] obs_builder')"
python -c "import reward; print('[OK] reward')"
python -c "import fast_lob; print('[OK] fast_lob')"
```

### 5. Run Tests

```bash
# Quick smoke test
pytest tests/test_core_models.py -v

# Full test suite
pytest tests/ -v --tb=short

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### 6. Cleanliness (no generated files)

```bash
make clean
make check-clean  # fails if generated artifacts remain
```

---

## Environment Variables & Secret Handling

- Secrets **must** be injected via environment variables or a secret manager; never commit credentials to the repo.
- Use `.env.example` as the template and keep your real `.env` file untracked (gitignored).
- CI enforces gitleaks + TruffleHog scans on every PR and weekly schedule.

### Required for Live Trading

| Variable | Description | Source |
|----------|-------------|--------|
| `BINANCE_API_KEY` | Binance API key | Secret manager / `.env` |
| `BINANCE_API_SECRET` | Binance API secret | Secret manager / `.env` |

### Optional for US Equities

| Variable | Description | Source |
|----------|-------------|--------|
| `ALPACA_API_KEY` | Alpaca API key | Secret manager / `.env` |
| `ALPACA_API_SECRET` | Alpaca API secret | Secret manager / `.env` |
| `POLYGON_API_KEY` | Polygon.io API key | Secret manager / `.env` |

### Optional for Forex

| Variable | Description | Source |
|----------|-------------|--------|
| `OANDA_API_KEY` | OANDA API token | Secret manager / `.env` |
| `OANDA_ACCOUNT_ID` | OANDA account ID | Secret manager / `.env` |

### Development

| Variable | Description | Default |
|----------|-------------|---------|
| `TB_FAIL_ON_STALE_FILTERS` | Fail if exchange filters outdated | `0` |
| `BINANCE_PUBLIC_FEES_DISABLE_AUTO` | Disable auto fee refresh | `0` |
| `PYTHONDONTWRITEBYTECODE` | Don't create `.pyc` files | `1` |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'obs_builder'"

Native extensions not built. Run:
```bash
pip install -r requirements-build.txt
python setup.py build_ext --inplace
```

### "torch.cuda.is_available() returns False"

1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### "ImportError: numpy.core.multiarray failed to import"

NumPy version mismatch. Reinstall:
```bash
pip install --force-reinstall numpy==1.26.4
python setup.py build_ext --inplace  # Rebuild extensions
```

### "Python 3.12 not found"

Install Python 3.12:
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: `sudo apt install python3.12 python3.12-dev python3.12-venv`
- **macOS**: `brew install python@3.12`

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-03 | 1.1.0 | Added CI build matrix, locking policy, and hash verification requirements |
| 2025-12-03 | 1.0.0 | Initial comprehensive requirements document |

---

**Last Updated**: 2025-12-03
**Maintainer**: CustodiaCloud Team
