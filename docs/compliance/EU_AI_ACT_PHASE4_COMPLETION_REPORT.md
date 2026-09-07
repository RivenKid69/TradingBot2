# EU AI Act Phase 4 — Optional Conformity & Deployment Readiness (Draft)

## CustodiaCloud (CCEA)

**Date**: 2025-12-08  
**Phase**: 4 of 5  
**Status**: DRAFT (deployment-dependent; not a compliance claim)

---

## Important Notice (Non-Legal)

This document describes **optional tooling and templates** intended to support governance, evidence exports, and (where applicable) EU AI Act conformity-assessment workstreams.

- CustodiaCloud does **not** self-classify as a “high-risk AI system” in documentation.
- EU AI Act applicability, roles (provider/deployer), and any required procedures are **deployment- and jurisdiction-dependent** and must be validated with qualified counsel (and, where applicable, a notified body).
- CustodiaCloud is positioned as a **B2B software/ICT provider** with **CCEA** separation: the Cloud does not store customer secrets and does not send live trading instructions (orders/targets/signals); live execution (if used) occurs only via the customer-controlled Agent.

---

## Executive Summary

Phase 4 focuses on preparing **evidence exports and documentation templates** that can help customers and internal teams structure conformity-assessment activities **if** a given deployment falls under EU AI Act obligations that require them (e.g., where a system is classified as high-risk under Annex III).

This phase is intentionally written in a conservative, committee-friendly posture:

- “Designed to support” rather than “certified/compliant”
- “Evidence exports” rather than regulatory guarantees
- “Deployment-dependent” classification (no self-classification)

---

## Phase 4 Deliverables (Templates / Tooling)

### 4.1 Conformity Assessment Support (If Applicable)

**Document**: `docs/compliance/conformity_assessment/checklist.md`

Purpose:

- Provide a structured checklist mapped to relevant EU AI Act Articles **when applicable**
- Track gaps and remediation items as engineering work artifacts
- Export evidence packages for customer/vendor due diligence

### 4.2 EU Declaration of Conformity (Template)

**Document**: `docs/compliance/EU_DECLARATION_OF_CONFORMITY.md`

Purpose:

- Provide a template that can be completed **only after** legal review and (if applicable) notified-body involvement
- Keep placeholders for:
  - provider identity
  - system identification
  - applicable standards (if any)
  - procedure (if any)

### 4.3 Instructions for Use (Transparency Template)

**Document**: `docs/compliance/INSTRUCTIONS_FOR_USE.md`

Purpose:

- Provide an Article 13-aligned transparency/instructions template
- Emphasize CCEA boundary, human oversight, and deployment-defined performance disclosures (no performance promises)

### 4.4 EU Database Registration (Preparation Only)

Purpose:

- Provide a placeholder data structure for registration **if required** by a given deployment
- Keep all classification fields as **[TBD]** pending legal determination

---

## Implementation Notes (CCEA Constraints)

Any Phase 4 tooling must remain consistent with CCEA boundaries:

- **No secrets in Cloud**: broker/exchange credentials remain only in the Agent environment
- **No Cloud live trading instructions**: Cloud may send lifecycle commands and signed artifacts; execution remains customer-controlled via Agent
- **Telemetry redaction**: aggregated by default; raw order events are opt-in and must be treated as sensitive

Technical boundary reference: `archive/root_files/Design Doc CCEA Cloud.txt`.

---

## Next Steps

- Align remaining Phase 5 documentation to the same deployment-dependent posture
- Ensure all templates preserve placeholders rather than compliance assertions
- Keep public-facing narratives consistent with the canon: `docs/DOCUMENTATION_CANON_DESIGN.md`
