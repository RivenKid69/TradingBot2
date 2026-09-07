# EU AI Act Phase 3 — Tooling Status (Draft)

## Quality Management System & Testing Framework (CustodiaCloud / CCEA)

**Date**: 2025-12-08  
**Phase**: 3 of 5  
**Status**: DRAFT (tooling description; not a compliance claim)

---

## Important Notice (Non-Legal)

This document summarizes optional engineering tooling intended to support governance and evidence exports (e.g., QMS scaffolding and testing workflows). It is not a claim of EU AI Act compliance, certification, or completed conformity assessment.

EU AI Act applicability and any required procedures are **deployment-dependent** and should be validated with qualified counsel (and, where applicable, a notified body).

---

## Scope (What this phase is about)

Phase 3 focuses on:

1. **Quality Management System (QMS) scaffolding** (process/evidence templates, internal controls)  
2. **Testing framework scaffolding** (pre-deployment testing workflows, exportable test evidence)  
3. **Cybersecurity posture documentation** (privacy-by-design and security-by-design evidence)  
4. **Post-market monitoring scaffolding** (governance signals and reporting workflows)

All tooling must remain consistent with CCEA boundaries:

- Secrets remain local (Agent)
- Cloud does not send live trading instructions (orders/targets/signals)
- Telemetry is redacted by design

Technical boundary reference: `archive/root_files/Design Doc CCEA Cloud.txt`.

---

## Key Artifacts (References)

- `docs/compliance/EU_AI_ACT_PHASE1_COMPLETION_REPORT.md` (Phase 1 tooling status)
- `docs/compliance/EU_AI_ACT_PHASE2_COMPLETION_REPORT.md` (Phase 2 tooling status)
- `docs/compliance/conformity_assessment/checklist.md` (checklist template, if applicable)

---

## Notes on Configuration

Configuration for QMS/testing/cybersecurity/post-market tooling is **deployment-specific**. Do not treat missing configuration paths as commitments; maintain placeholders until the relevant configuration exists and is validated for the target deployment.
