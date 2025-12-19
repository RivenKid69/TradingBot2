# Security Program Roadmap

**Version**: 1.0
**Date**: 2025-12-19
**Status**: Active
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## Purpose

This document tracks the security program roadmap for CustodiaCloud, including planned certifications, audits, and security initiatives. Per Documentation Canon, all items are disclosed honestly with their actual status.

---

## Current State

CustodiaCloud is a pre-seed stage company. The security program is in early development with the following active components:

| Component | Status | Description |
|-----------|--------|-------------|
| Internal Code Review | Active | Security review per major release |
| Dependency Scanning | Active | Automated via CI |
| Secret Scanning | Active | Pre-commit and CI hooks |
| Static Analysis | Active | SAST tools in CI pipeline |
| Vulnerability Tracking | Active | Internal tracking of known issues |

---

## Roadmap Items

### External Penetration Testing

| Field | Value |
|-------|-------|
| **Target** | Annual penetration test by external vendor |
| **Status** | Roadmap (no vendor contract) |
| **Dependency** | Funding (seed round) |
| **Earliest Timeline** | 2026 if funded |
| **Deliverable** | External pentest report |

### SOC 2 Type I

| Field | Value |
|-------|-------|
| **Target** | Initial SOC 2 Type I audit |
| **Status** | Roadmap (no auditor engagement) |
| **Dependency** | Funding, operational maturity |
| **Earliest Timeline** | 2026 if funded |
| **Deliverable** | SOC 2 Type I report |

### SOC 2 Type II

| Field | Value |
|-------|-------|
| **Target** | SOC 2 Type II certification |
| **Status** | Roadmap (no auditor engagement) |
| **Dependency** | SOC 2 Type I completion, funding |
| **Earliest Timeline** | 2027 if funded |
| **Deliverable** | SOC 2 Type II report |

### ISO 27001 (Optional)

| Field | Value |
|-------|-------|
| **Target** | ISO 27001 certification |
| **Status** | Evaluation phase |
| **Dependency** | Business need assessment, funding |
| **Earliest Timeline** | Post-2027 if pursued |
| **Deliverable** | ISO 27001 certificate |

---

## Internal Security Practices

These practices are currently active and do not depend on external certification:

1. **Code Review**: All code changes require peer review with security consideration
2. **Dependency Management**: Automated scanning for vulnerable dependencies
3. **Secret Scanning**: Pre-commit hooks and CI scanning prevent credential leaks
4. **Access Control**: RBAC for internal systems with audit logging
5. **Incident Response**: Documented procedures (see `docs/runbooks/INCIDENT_RESPONSE.md`)

---

## Tech Debt Reference

This roadmap addresses the following tech debt items:

| ID | Registry Reference | Status |
|----|-------------------|--------|
| security-external-audits | `docs/reports/TECH_DEBT_REGISTRY.md#security-external-audits` | Controlled |

---

## Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-19 | Initial roadmap creation as tech debt control artifact |

**Review Frequency**: Quarterly
**Owner**: Security Team
**Classification**: Internal

---

*This document follows the Documentation Canon - no absolute claims, honest disclosure of roadmap status.*
