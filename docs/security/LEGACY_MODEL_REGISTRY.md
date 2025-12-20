# Legacy Model Registry

**Document Status**: Active
**Last Updated**: 2025-12-20
**Owner**: Security/Engineering
**Review Cycle**: Monthly

---

## Purpose

This registry tracks all legacy model artifacts that require `weights_only=False` for loading.
Per Threat Model T3 (Legacy Model Accumulation), this registry enables:
- Visibility into remaining insecure-format models
- Prioritized conversion planning
- Audit trail for compliance

---

## Registry Format

| Model ID | Path | Format | Created | Last Used | Conversion Status | Owner | Notes |
|----------|------|--------|---------|-----------|-------------------|-------|-------|
| *No legacy models currently registered* | - | - | - | - | - | - | - |

---

## Audit Metrics

### Current State (2025-12-20)

| Metric | Value |
|--------|-------|
| Total legacy models | 0 |
| Pending conversion | 0 |
| Converted this month | 0 |
| ALLOW_UNSAFE_MODEL_LOAD production usage | 0 |

### Audit History

| Date | Auditor | Legacy Count | Conversions | Notes |
|------|---------|--------------|-------------|-------|
| 2025-12-20 | Automated | 0 | 0 | Initial baseline - no legacy models in production |

---

## Conversion Procedure

1. Identify model requiring unsafe loading (detected via logging)
2. Add entry to this registry with owner and justification
3. Schedule conversion using `tools/convert_legacy_models.py`
4. Verify secure loading works post-conversion
5. Update registry entry with CONVERTED status
6. Remove from registry after 30-day validation period

---

## Monitoring

### Automated Detection

- All `weights_only=False` calls are logged with WARNING level
- CI/CD scans (Bandit) flag new unsafe loading code
- Production alerts on ALLOW_UNSAFE_MODEL_LOAD environment variable

### Monthly Audit

The security team performs monthly audits:
1. Query model storage for pickle-format files
2. Verify all models load with `weights_only=True`
3. Update this registry with any new legacy models
4. Report conversion progress to Engineering

---

## Control Artifacts

| Artifact | Location | Purpose |
|----------|----------|---------|
| Threat Model | `docs/security/THREAT_MODEL_MODEL_LOADING.md` | T3 description and controls |
| Conversion Utility | `tools/convert_legacy_models.py` | Convert legacy to secure format |
| SAST Rules | `.github/workflows/security-sast.yml` | Detect unsafe patterns |
| Tech Debt Registry | `docs/reports/TECH_DEBT_REGISTRY.md#security-legacy-models` | Overall tracking |

---

## Compliance Notes

Per Documentation Canon:
- This registry provides visibility, not guarantee of zero legacy models
- New models entering the pipeline should be created in secure format by default
- Legacy model usage requires documented business justification

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-20 | Initial registry creation for T3 tech debt closure |

---

*This document follows the Documentation Canon - honest disclosure of current state.*
