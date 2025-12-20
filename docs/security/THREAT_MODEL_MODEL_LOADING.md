# Threat Model: Model Artifact Loading

**Document Status**: Active
**Last Updated**: 2025-12-20
**Owner**: Security/Engineering
**Review Cycle**: Quarterly or on significant change

---

## 1. Overview

This threat model covers the security risks associated with loading machine learning model artifacts (PyTorch `.pt`/`.pth`, scikit-learn `.pkl`/`.joblib`) in the CustodiaCloud platform.

### Scope

- **In Scope**: Model loading in `infer_signals.py`, training pipelines, any code using `torch.load()` or `joblib.load()`
- **Out of Scope**: Model training process, data pipeline security (covered separately)

### Architecture Context (CCEA)

Per CCEA architecture:
- Models are built and signed in **Cloud** (Artifact Builder)
- Models are deployed to **Agent** with signature verification
- Model loading occurs in both Cloud (backtest/sim) and Agent (live inference)
- Agent validates artifact digest and signature before loading

---

## 2. Threat Identification

### T1: Arbitrary Code Execution via Pickle Deserialization

**Description**: PyTorch's `torch.load()` and joblib's `joblib.load()` use Python's pickle module by default. Malicious pickle payloads can execute arbitrary code during deserialization.

**Attack Vector**:
1. Attacker creates malicious model file with embedded code
2. Model is loaded via `torch.load(path, weights_only=False)`
3. Pickle deserializes the payload, executing attacker's code

**Impact**:
- **Confidentiality**: Exfiltration of secrets, API keys, trading data
- **Integrity**: Modification of trading logic, strategy parameters
- **Availability**: System compromise, denial of service

**Likelihood**: Medium (requires attacker to inject malicious model into pipeline)

**Severity**: Critical

### T2: Model Substitution Attack

**Description**: Attacker replaces legitimate model with malicious or degraded model.

**Attack Vector**:
1. Attacker gains write access to models directory or artifact registry
2. Replaces legitimate model with attacker-controlled model
3. System loads substituted model, producing incorrect predictions

**Impact**:
- Trading decisions based on attacker-controlled model
- Financial losses from degraded model performance

**Likelihood**: Low (requires privileged access)

**Severity**: High

### T3: Legacy Model Accumulation

**Description**: Over time, legacy models in insecure format accumulate, creating ongoing risk.

**Attack Vector**:
1. Organization retains legacy models for reproducibility
2. Legacy models require `weights_only=False` to load
3. Relaxed security controls persist indefinitely

**Impact**: Persistent attack surface for T1

**Likelihood**: High (natural technical debt)

**Severity**: Medium

**Control Status**: CONTROLLED

**Control Artifact**: `docs/security/LEGACY_MODEL_REGISTRY.md` (legacy model registry with monthly audit)

**Metrics Tracked**:
- Legacy model count (current: 0)
- Monthly conversion rate
- `ALLOW_UNSAFE_MODEL_LOAD` production usage (current: 0)

**Tech Debt Tracking**: `docs/reports/TECH_DEBT_REGISTRY.md#security-legacy-models`

---

## 3. Security Controls

### C1: Fail-Closed Model Loading (IMPLEMENTED)

**Control**: `infer_signals.py` and `adversarial/pbt_scheduler.py` reject models that cannot be loaded with `weights_only=True` by default.

**Implementation**:
```python
try:
    model = torch.load(path, map_location="cpu", weights_only=True)
except (pickle.UnpicklingError, RuntimeError, AttributeError) as e:
    # FAIL-CLOSED: Reject unless explicitly allowed
    if not os.environ.get("ALLOW_UNSAFE_MODEL_LOAD"):
        raise RuntimeError("Model cannot be loaded securely...")
```

**Effectiveness**: High - blocks T1 by default

**Residual Risk**: Operators may enable unsafe loading for convenience

### C2: Explicit Opt-In for Unsafe Loading (IMPLEMENTED)

**Control**: Unsafe loading requires explicit environment variable `ALLOW_UNSAFE_MODEL_LOAD=1`.

**Implementation**: See C1 code block.

**Effectiveness**: Medium - creates audit trail and prevents accidental exposure

**Residual Risk**: Misconfiguration, permanent enablement in production

### C3: Model Conversion Utility (IMPLEMENTED)

**Control**: `tools/convert_legacy_models.py` converts legacy models to secure format.

**Implementation**:
- Loads legacy model (controlled context)
- Extracts state_dict
- Re-saves in secure format
- Verifies secure loading works

**Effectiveness**: High - eliminates T3 over time

**Residual Risk**: Unconverted models in backups/archives

### C4: Artifact Signing and Verification (CCEA)

**Control**: All model artifacts are signed by Cloud and verified by Agent before loading.

**Implementation**:
- Artifact Builder signs artifacts with platform key
- Agent verifies signature and digest before accepting artifact
- Allowlist of trusted registries

**Effectiveness**: High - mitigates T2

**Residual Risk**: Key compromise, registry compromise

### C5: Static Analysis (CI/CD)

**Control**: Security SAST scans detect unsafe pickle/torch.load usage.

**Implementation**:
- Bandit security scanner (MEDIUM+ severity enforcement)
- Semgrep with custom rules
- CI fails on new unsafe patterns

**Effectiveness**: Medium - catches new unsafe code

**Residual Risk**: False negatives, novel patterns

---

## 4. Control Verification

### Automated Verification

| Control | Verification Method | Frequency | Location |
|---------|---------------------|-----------|----------|
| C1 | Unit test: secure loading rejection | Every CI run | `tests/test_model_loading_security.py` |
| C2 | Unit test: env var behavior | Every CI run | `tests/test_model_loading_security.py` |
| C3 | Integration test: conversion utility | Weekly | `tests/integration/test_model_conversion.py` |
| C4 | CCEA artifact verification tests | Every CI run | `tests/phase3/test_phase3_supply_chain.py` |
| C5 | Bandit/Semgrep scans | Every CI run | `.github/workflows/security-sast.yml` |

### Manual Verification

| Control | Verification Method | Frequency | Owner |
|---------|---------------------|-----------|-------|
| C1-C3 | Security review of model loading code | Quarterly | Security Team |
| C4 | Artifact signing key audit | Annually | Security Team |
| All | Penetration testing (model injection) | Annually | External Auditor |

---

## 5. Incident Response

### Detection Indicators

- `ALLOW_UNSAFE_MODEL_LOAD` set in production environment
- SecurityWarning in logs mentioning "weights_only=False"
- Unexpected model files in models/ directory
- Signature verification failures in Agent logs

### Response Procedure

1. **Isolate**: Disable affected model loading pipeline
2. **Investigate**: Review model provenance, check for unauthorized changes
3. **Remediate**: Replace compromised model with verified artifact
4. **Report**: Document incident per security incident procedure

---

## 6. Recommendations

### Immediate (Completed)

- [x] Implement fail-closed model loading (C1)
- [x] Create model conversion utility (C3)
- [x] Document threat model

### Short-Term (Next Quarter)

- [ ] Add bandit rule for `weights_only=False` usage
- [ ] Implement model hash verification in Agent
- [ ] Create unit tests for secure loading behavior

### Long-Term

- [ ] Migrate to ONNX format (inherently more secure)
- [ ] Implement model versioning with cryptographic provenance
- [ ] Hardware security module (HSM) for artifact signing keys

---

## 7. References

- [PyTorch Security Advisory: Untrusted Models](https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models)
- [CVE-2022-3885: PyTorch Arbitrary Code Execution](https://nvd.nist.gov/vuln/detail/CVE-2022-3885)
- CCEA Architecture: `archive/root_files/Design Doc CCEA Cloud.txt` Section 15 (Supply Chain)
- Model Conversion Utility: `tools/convert_legacy_models.py`

---

*This document follows the Documentation Canon (docs/DOCUMENTATION_CANON_DESIGN.md) - avoiding absolute claims and focusing on design intent.*
