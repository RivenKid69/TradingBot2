# Build Instructions

Comprehensive guide for building native Cython/C++ extensions for the CustodiaCloud codebase (this repository).

## Table of Contents

- [Quick Start](#quick-start)
- [Requirements](#requirements)
- [Platform-Specific Setup](#platform-specific-setup)
- [Building](#building)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

---

## Quick Start

For most users, the build process is simple:

```bash
# 1. Install build dependencies
pip install -r requirements-build.txt

# 2. Build extensions
make build

# 3. Verify build
make verify-hash
```

That's it! The native extensions are now compiled and ready to use.

---

## Requirements

### Python & Core Tools

- **Python**: 3.12+ (tested on 3.12.0)
- **pip**: Latest version (automatically upgraded by Makefile)
- **setuptools**: 69.0.3
- **wheel**: 0.42.0

### Build Tools

- **Cython**: 3.0.10 (pinned for reproducibility)
- **numpy**: 1.26.4 (pinned for C API compatibility)

### Compilers

| Platform | Compiler | Version | Notes |
|----------|----------|---------|-------|
| **Windows** | MSVC (Visual Studio) | 19.x (2022/2019) | Requires "Desktop development with C++" workload |
| **Linux** | GCC or Clang | GCC 11+, Clang 14+ | Usually pre-installed |
| **macOS** | Clang (Xcode) | Xcode 14+ | Install via: `xcode-select --install` |

---

## Platform-Specific Setup

### Windows (MSVC)

1. **Install Visual Studio**:
   - Download [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
   - Select workload: **"Desktop development with C++"**
   - Minimum components:
     - MSVC v143 - VS 2022 C++ x64/x86 build tools
     - Windows 10 SDK (latest)

2. **Verify installation**:
   ```cmd
   cl.exe
   # Should output MSVC version
   ```

3. **Build**:
   ```cmd
   pip install -r requirements-build.txt
   make build
   ```

### Linux (GCC)

1. **Install build tools** (Ubuntu/Debian):
   ```bash
   sudo apt-get update
   sudo apt-get install build-essential python3-dev
   ```

   Or (Fedora/RHEL):
   ```bash
   sudo dnf groupinstall "Development Tools"
   sudo dnf install python3-devel
   ```

2. **Verify GCC**:
   ```bash
   gcc --version
   # Should be >= 11.0
   ```

3. **Build**:
   ```bash
   pip install -r requirements-build.txt
   make build
   ```

### macOS (Clang)

1. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Verify Clang**:
   ```bash
   clang --version
   # Should be >= 14.0
   ```

3. **Build**:
   ```bash
   pip install -r requirements-build.txt
   make build
   ```

---

## Building

### Using Makefile (Recommended)

The Makefile provides convenient targets for all platforms:

```bash
# Build all extensions in-place (default)
make build

# Clean build artifacts
make clean

# Clean + rebuild from scratch
make rebuild

# Run tests after building
make test

# Verify build hash report
make verify-hash

# Show all available targets
make help
```

### Using setup.py Directly

For fine-grained control:

```bash
# Build in-place (for development)
python setup.py build_ext --inplace

# Build in build/ directory
python setup.py build_ext

# Build with verbose output
python setup.py build_ext --inplace --verbose

# Build single extension (faster)
python setup.py build_ext --inplace obs_builder
```

### Build Output

After successful build, you'll see:

```
Building native extensions for Windows...
This may take a few minutes on first build.
running build_ext
building 'obs_builder' extension
...
✓ Build complete!
Hash report: build_hash_report.json
```

**Generated files**:
- `*.pyd` (Windows) or `*.so` (Linux/macOS) - compiled extensions
- `*.c` - generated C code from Cython (can be deleted)
- `build_hash_report.json` - SHA256 hashes for reproducibility

---

## Verification

### 1. Hash Report

The build system automatically generates a SHA256 hash report:

```bash
make verify-hash
```

**Example output**:
```json
{
  "build_info": {
    "timestamp": "2025-12-03T10:30:45.123456+00:00",
    "python_version": "3.12.0",
    "cython_version": "3.0.10",
    "platform": "Windows-10-10.0.19045-SP0",
    "compiler": "MSVC 19.38.33134"
  },
  "extensions": {
    "obs_builder": {
      "path": "obs_builder.cp312-win_amd64.pyd",
      "sha256": "a1b2c3d4e5f6...",
      "size_bytes": 123456
    }
  }
}
```

**Use cases**:
- **Reproducible builds**: Compare hashes across machines
- **Deployment**: Verify compiled extensions match expected builds
- **Debugging**: Detect accidental recompilation

### 2. Import Test

Verify extensions can be imported:

```python
# Test all extensions
python -c "import obs_builder; print('✓ obs_builder OK')"
python -c "import reward; print('✓ reward OK')"
python -c "import fast_lob; print('✓ fast_lob OK')"
```

### 3. Run Tests

The test suite verifies correctness:

```bash
make test
# Or manually:
pytest tests/ -v
```

---

## Troubleshooting

### Common Issues

#### 1. "Cython is required to build extensions"

**Solution**: Install build dependencies first:
```bash
pip install -r requirements-build.txt
```

#### 2. "Unable to find vcvarsall.bat" (Windows)

**Problem**: Visual Studio not installed or not in PATH.

**Solution**:
- Install Visual Studio 2022 with "Desktop development with C++" workload
- Or use Visual Studio Build Tools (minimal installation)

#### 3. "numpy/arrayobject.h: No such file" (Linux/macOS)

**Problem**: numpy not installed or headers missing.

**Solution**:
```bash
pip install numpy==1.26.4
# Or on Linux:
sudo apt-get install python3-numpy
```

#### 4. Warnings about version mismatches

**Example**:
```
WARNING: numpy 1.26.3 != 1.26.4 (pinned)
WARNING: Cython 3.0.9 != 3.0.10 (pinned)
```

**Impact**: Build will succeed, but reproducibility may not be guaranteed.

**Why this matters**: Different dependency versions can produce different compiled artifacts,
making it harder to audit and verify builds. For production deployments, use exact pinned versions.

**Solution**: Install exact pinned versions from lockfiles:
```bash
# For CPU-only builds
pip install -r requirements-cpu.lock.txt

# For GPU builds
pip install -r requirements-gpu.lock.txt

# Or force specific versions
pip install --force-reinstall numpy==1.26.4 Cython==3.0.10
```

**Verification**: After installing, run `make verify-hash` to confirm build hash matches expected value.

---

## Reproducibility

### Deterministic Build Requirements

For reproducible builds, ensure:

1. **Exact dependency versions**: Use lockfiles (`requirements-cpu.lock.txt` or `requirements-gpu.lock.txt`)
2. **Same Python version**: Python 3.12.x
3. **Same compiler toolchain**: MSVC 19.x on Windows, GCC 11+ on Linux
4. **Clean build environment**: Run `make clean` before building

### Lockfile-Based Installation

```bash
# Clean environment
make clean

# Install exact versions from lockfile
pip install -r requirements-cpu.lock.txt
pip install -r requirements-build.txt

# Build
make build

# Verify hash matches expected
make verify-hash
```

### CI/CD Verification

The build system includes automated verification:

- **CI workflow**: [.github/workflows/build-and-test.yml](.github/workflows/build-and-test.yml) runs on every PR
- **Hash verification**: `make verify-hash` compares build output hashes (step: "Verify build hash report")
- **SBOM generation**: [.github/workflows/security-sast.yml](.github/workflows/security-sast.yml) generates SBOM (CycloneDX) for dependency auditing

### Hash Report

After building, check the hash report:
```bash
make verify-hash
# Outputs: BUILD_HASH_REPORT.txt with SHA-256 hashes of all compiled artifacts
```

The hash report contains:
- SHA-256 hash of each `.so`/`.pyd` compiled extension
- Build timestamp and environment info
- Dependency versions used

For audit/compliance, this report provides evidence of deterministic builds.

### Debug Build

For debugging extension crashes:

**Windows**:
```bash
python setup.py build_ext --inplace --debug
```

**Linux/macOS**:
```bash
CFLAGS="-O0 -g" python setup.py build_ext --inplace
```

### Clean Rebuild

If you encounter strange errors, try a clean rebuild:

```bash
make clean
make build
```

---

## Advanced Topics

### Compiler Flags

Compiler flags are platform-specific and optimized for performance:

**Windows (MSVC)**:
- `/O2` - Maximum optimization
- `/GL` - Whole program optimization
- `/LTCG` - Link-time code generation
- `/std:c++17` - C++17 standard

**Linux/macOS (GCC/Clang)**:
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimizations
- `-ffast-math` - Fast floating-point math
- `-std=gnu++17` - C++17 with GNU extensions

### Parallel Build

Speed up compilation on multi-core systems:

```bash
# Windows/Linux/macOS
python setup.py build_ext --inplace --parallel 8
```

### Cross-Compilation

For advanced users deploying to different architectures:

**Linux → Linux ARM**:
```bash
CC=aarch64-linux-gnu-gcc python setup.py build_ext
```

**macOS Universal Binary** (x86_64 + arm64):
```bash
ARCHFLAGS="-arch x86_64 -arch arm64" python setup.py build_ext
```

---

## File Structure

After build, project structure:

```
AI-Powered-Quantitative-Research-Platform/
├── setup.py                    # Build configuration
├── Makefile                    # Build automation
├── requirements-build.txt      # Pinned build dependencies
├── build_hash_report.json      # SHA256 hashes (generated)
├── *.pyx                       # Cython source files
├── *.pyd / *.so               # Compiled extensions (generated)
├── *.c                        # Generated C code (can delete)
├── OrderBook.cpp              # Hand-written C++ (kept)
└── build/                     # Intermediate build files (ignored)
```

**Important**:
- `.pyd`/`.so` files are **platform-specific** - don't commit to git
- `.c` files are **generated** from `.pyx` - don't commit to git
- `.cpp` files are **hand-written** - commit to git
- `build_hash_report.json` is for verification - optionally commit

---

## References

- [Cython Documentation](https://cython.readthedocs.io/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Setuptools Documentation](https://setuptools.pypa.io/)
- [NumPy C API](https://numpy.org/doc/stable/reference/c-api/)

---

**Last Updated**: 2025-12-03
**Version**: 1.0.0
**Maintainer**: CustodiaCloud Team
