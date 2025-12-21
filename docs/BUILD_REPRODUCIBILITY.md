# Build Reproducibility

**Version**: 1.0
**Date**: 2025-12-21
**Status**: Active
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Purpose

This document describes the build reproducibility controls for CustodiaCloud native extensions and their verification scope.

---

## Scope of Verification

The build reproducibility system verifies:

| Component | Verification Method | Status |
|-----------|---------------------|--------|
| Cython extensions (.so/.pyd) | SHA256 hash comparison | Active |
| C++ extensions (lob/*.so) | SHA256 hash comparison | Active |
| Python version | Must be 3.12.x | Enforced |
| Build environment | Recorded in report | Documented |

### What IS Verified

1. **Native Extensions**
   - All compiled `.so` (Linux/macOS) and `.pyd` (Windows) files
   - SHA256 hashes recorded during build
   - Verified by `make verify-hash` and CI

2. **Python Version**
   - Build must use Python 3.12.x
   - Version recorded in `build_hash_report.json`
   - Enforced by `tools/verify_hash_report.py`

3. **Extension Coverage**
   - All built extensions must be in hash report
   - `--require-all-artifacts` flag enforces this

### What is NOT Verified

1. **Python Dependencies**
   - Not hash-locked (use lockfiles separately)
   - See `requirements-cpu.lock.txt`, `requirements-gpu.lock.txt`

2. **Transitive Dependencies**
   - SBOM generated with SHA256 hash verification
   - See CI artifacts: `sbom.json` and `sbom-verification.json`
   - Hash recorded per CI run for audit trail

3. **Cross-Build Reproducibility**
   - Hashes may differ across platforms
   - Platform recorded in report

---

## Control Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| `build_hash_report.json` | Repository root | SHA256 hashes of all extensions |
| `requirements-cpu.lock.txt` | Repository root | Pinned CPU dependencies |
| `requirements-gpu.lock.txt` | Repository root | Pinned GPU dependencies |
| `sbom.json` | CI artifact | CycloneDX SBOM |
| `sbom-verification.json` | CI artifact | SBOM hash + git SHA + timestamp for audit |

---

## Verification Commands

```bash
# Build extensions (generates hash report)
make build

# Verify hash report matches built artifacts
make verify-hash

# Verify all artifacts are covered
python tools/verify_hash_report.py --require-all-artifacts

# Regenerate lockfiles
make lock-cpu
make lock-gpu
```

---

## CI Integration

The following checks are performed in CI:

1. **Build** (`make build`)
   - Generates `build_hash_report.json`
   - Records extension hashes and build info

2. **Verify** (`make verify-hash`)
   - Validates report exists
   - Validates all hashes match
   - Validates Python version is 3.12.x

3. **Artifact Upload**
   - `build_hash_report.json` uploaded as CI artifact
   - Available for audit and comparison

---

## Lockfile Management

### Purpose

Lockfiles pin exact versions of Python dependencies to ensure reproducible installations.

### Files

| File | Scope | Platform |
|------|-------|----------|
| `requirements-cpu.lock.txt` | Production (CPU) | Any |
| `requirements-gpu.lock.txt` | Production (GPU) | CUDA-capable |
| `requirements-dev.txt` | Development | Any |
| `requirements-build.txt` | Build tools | Any |

### Regeneration

Lockfiles should be regenerated when:
- Adding new dependencies
- Updating existing dependencies
- Security vulnerabilities require updates

```bash
# Regenerate CPU lockfile
make lock-cpu

# Regenerate GPU lockfile
make lock-gpu
```

---

## Known Limitations

1. **Platform-Specific Builds**
   - Native extensions are platform-specific
   - Hashes will differ between Linux/macOS/Windows
   - Each platform has separate verification

2. **Compiler Variations**
   - Different compiler versions may produce different binaries
   - Compiler info recorded but not enforced

3. **Non-Deterministic Compilation**
   - Some compilers may produce non-deterministic output
   - Timestamp-based variations possible

---

## Tech Debt Reference

| ID | Registry Reference | Status |
|----|-------------------|--------|
| reproducibility-hash-scope | `docs/reports/TECH_DEBT_REGISTRY.md#reproducibility-hash-scope` | Controlled |

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-21 | Initial version documenting hash verification scope |

**Review Frequency**: Upon build system changes
**Owner**: Engineering
**Classification**: Internal

---

*This document follows the Documentation Canon - honest disclosure of verification scope and limitations.*
