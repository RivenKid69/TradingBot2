# EU AI Act Phase 2 — Tooling Status (Draft)
## Technical Documentation & Logging (CustodiaCloud / CCEA)

**Date**: 2025-12-08  
**Phase**: 2 of 5  
**Status**: DRAFT (tooling description; not a compliance claim)

---

## Important Notice (Non-Legal)

This document summarizes **engineering tooling** that may support EU AI Act governance workflows (e.g., documentation and logging) when applicable. It is not a statement of EU AI Act compliance, certification, or completed conformity assessment.

Applicability is deployment-dependent:
- Roles (provider/deployer) and any required procedures depend on how CustodiaCloud is deployed and used.
- CustodiaCloud does not self-classify as “high-risk AI” in documentation without legal review.

---

## Scope (What this phase is about)

Phase 2 focuses on:

1. **Technical documentation scaffolding** (for structured evidence exports where required)  
2. **Record-keeping / logging scaffolding** (governance-grade logs, integrity, exports)

Both must remain consistent with CCEA boundaries:
- Cloud does not store secrets and does not send live trading instructions (orders/targets/signals)
- Orders are created and sent only in the customer-controlled Agent
- Telemetry is redacted by design; raw order events are opt-in and sensitive

Technical boundary reference: `archive/root_files/Design Doc CCEA Cloud.txt`.

---

## Key Artifacts (References)

- `docs/compliance/technical_documentation/README.md` (Annex IV reference structure)
- `docs/compliance/INSTRUCTIONS_FOR_USE.md` (Article 13 template)
- `docs/compliance/EU_AI_ACT_PHASE1_COMPLETION_REPORT.md` (Phase 1 tooling status)

---

## Notes on Configuration

Configuration files for EU AI Act-related tooling (logging retention, export formats, governance settings) are **deployment-specific** and may not be committed in this repository. Treat any configuration paths in historical documents as placeholders unless present in-tree.

