# GDPR Implementation Plan (CCEA / EU-only)

**Project**: AI-Powered Quantitative Research Platform (CCEA)  
**Scope**: GDPR controls required for a software provider operating the CCEA model (Cloud-controlled execution; Agent is customer-operated).  
**Deployment**: **EU-only** (no non-EU regions).  
**Enterprise option**: on-prem/VPC deployment is supported **within EU-only posture** (customer-controlled infrastructure located in EU; no vendor-operated non-EU processing).  
**Primary design source**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt` (Sections 13–16; Privacy/GDPR & Data Residency).

## 0) Why this plan (and what it is not)

This plan implements **only** the GDPR elements needed for this project’s architecture and positioning as a **software/platform provider**, aligned with CCEA constraints:

- **Telemetry has three sensitivity levels**: `AGGREGATED` (default), `DETAILED_NON_SENSITIVE` (opt-in), `RAW_ORDER_EVENTS` (enterprise-only, explicit opt-in) — with **mandatory redaction** before leaving the Agent.
- **No secrets** (API keys, tokens), **no env vars**, and **no order-like payloads in Cloud→Agent commands** are allowed (enforced by schema/CI); telemetry is separately governed by sensitivity level.
- **EU data residency by default** (and in this project: EU-only).
- **Retention per tenant + auto-purge + DSAR export/delete**.
- **RBAC + access audit + break-glass** for incident-only exceptional access.

This plan is not a substitute for legal advice. It is an engineering/compliance implementation plan that should be reviewed by counsel for the final determination of roles (Controller/Processor) and policy wording.

## 1) Design Doc requirements (CCEA-specific GDPR subset)

The Design Doc explicitly requires:

- **Data minimization**: collect only what’s necessary for monitoring, billing (if needed), support (with consent).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L873` (14.1), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1736` (6.1).
- **Telemetry sensitivity levels** with **AGGREGATED default**, an opt-in technical level (`DETAILED_NON_SENSITIVE`), and **optional `RAW_ORDER_EVENTS`** (dangerous; enterprise-only + explicit opt-in in this product posture).
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846` (13.1–13.2), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L742`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1739`.
- **Mandatory redaction** before telemetry transmission: remove secrets, mask account identifiers, forbid env var logging.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L863` (13.3), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1728` (5.4).
- **Retention per tenant + auto-purge + export/delete (DSAR)**.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L884` (14.2), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1759` (6.4).
- **EU residency by default** (and here EU-only).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L892` (14.3), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1745` (6.2).
- **Access control**: RBAC in workspace, audit log of access, break-glass with reason and auditable event.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L898` (14.4), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1751` (6.3).
- **Protocol constraints**: commands are enumerated/versioned; **order-like payloads are prohibited at schema + CI level**.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1693`.
- **CI guardrails as “hard constraints”** (build-time enforcement, not optional tests):  
  - Cloud builds **must not** include broker trading client libraries  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1029`.  
  - Redaction middleware **must be mandatory and not disableable by feature flag**  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1051`.  
  - Signed artifacts required; unsigned artifacts rejected  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1045`.
- **Evidence pack / auditability exports** (enterprise due diligence): digests, signatures, SBOM, approvals/change journal, incident logs, telemetry export by sensitivity.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L945`.
- **Change management**: TRADING_IMPACTING always requires local approval + auditable diff trail.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.

## 2) GDPR articles in scope (provider-focused)

The minimal GDPR subset for this project typically includes:

- **Art. 5** (principles: minimization, storage limitation), **Art. 6** (lawful basis)
- **Art. 13–14** (transparency obligations: what is processed, why, and boundaries)
- **Art. 12–23** (data subject rights: access, portability, erasure; response timing)
- **Art. 25** (privacy by design/default)
- **Art. 28** (processor obligations / DPAs where applicable)
- **Art. 30** (records of processing activities – RoPA)
- **Art. 32** (security of processing)
- **Art. 33–34** (personal data breach notification)
- **Art. 44+** (international transfers) — **out of scope by design** (EU-only), but must be continuously verified.

## 3) Non-negotiable product/architecture constraints (GDPR-by-design)

These constraints are required for the platform’s compliance posture and must remain invariant:

1. **Cloud never receives** broker credentials, API keys/tokens, or env vars (redaction + validation + CI guardrails).
2. **Cloud→Agent protocol commands** never carry **order-like payloads** (side/qty/price/order id/fill details); forbidden at schema + CI (“no order commands”).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L750`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L754`.
3. **Cloud telemetry sensitivity levels are fixed and named**: `AGGREGATED` (default), `DETAILED_NON_SENSITIVE` (opt-in), `RAW_ORDER_EVENTS` (enterprise-only, explicit opt-in).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`.
4. **Order/fill-like fields are forbidden in Cloud telemetry** unless `telemetry_level == RAW_ORDER_EVENTS` and the workspace is explicitly enabled for enterprise RAW (gated, audited, retention-minimized, access-restricted).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L851`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1739`.
5. Default telemetry is **AGGREGATED** (retail/pro); any increase in sensitivity is explicit, audited, and controlled; enterprise may select “telemetry stays local” (local-only mode) instead of Cloud ingestion.
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L855`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L861`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1749`.
6. **Telemetry redaction is always on** and cannot be disabled by configuration/feature flag; env var logging is forbidden.  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L871`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1051`.
7. EU-only residency: all storage, backups, logs, observability, and support tooling remain in EU; **EU-only drift checks are mandatory** (fail closed).  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L892`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1745`.
8. Break-glass access is **incident-only**, time-bound, scope-limited, reason-required, and fully audited.
9. Cloud builds **must not** contain broker trading client libraries (import/dependency boundary enforced in CI).  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1029`.
10. **Order-like payloads are prohibited at schema + CI** (hard constraint, not “best effort”).  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1039`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1697`.
11. **Artifacts/config blobs are referenced only by digest** (no “latest”); **unsigned artifacts are rejected**; **registry allowlist is enforced**.  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L911`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L913`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1045`.
12. **New protocol command types require security review and auditable approval** (recorded in the change journal).  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1043`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.
13. **Remote shell into Agent is prohibited** in the EU-only posture (no feature); any enterprise exception must be contractually scoped + break-glass + auditable.  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L943`.

### 3.1 Canonical stance: `RAW_ORDER_EVENTS`

The Design Doc allows `RAW_ORDER_EVENTS` as an opt-in sensitivity level and flags it as dangerous for privacy/IP.
Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1739`.

**Decision for this EU-only product posture**:
- `RAW_ORDER_EVENTS` is supported as a telemetry sensitivity level **only for enterprise** and only with **explicit per-workspace opt-in**, with strict validation (allowlist fields), minimal retention, and access controls.
- Default remains `AGGREGATED`; `DETAILED_NON_SENSITIVE` is opt-in for ops/debugging; enterprise may choose “telemetry stays local” instead of Cloud ingestion.
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L855`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L861`.

**Required alignment work**:
- Ensure protocol/schema, documentation, and runtime enforcement are consistent:
  - Cloud telemetry schema contains `AGGREGATED`, `DETAILED_NON_SENSITIVE`, `RAW_ORDER_EVENTS` per Design Doc.
  - Non-RAW levels must reject order/fill-like fields; RAW must be blocked unless enterprise + explicit opt-in is proven and audited.
  - Evidence pack must support telemetry export “by sensitivity level” including RAW when (and only when) enabled.
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L958`.

## 4) Phased execution plan

Each phase includes deliverables and “Definition of Done” (DoD). Phases are ordered to minimize risk and avoid rework.

### Phase 0 — Scope, data map, and role model (CCEA-first)

**Goal**: create a defensible processing map aligned with CCEA boundaries.

Key work:
- Build a **system data inventory**: where personal data can exist (accounts, billing, access logs, audit trails, telemetry identifiers).
- Produce a **data flow diagram**: Cloud ↔ Agent, including telemetry levels.
- Decide **Controller vs Processor** per data category (document assumptions; validate with counsel).
- Create a **RoPA-lite** record (Art. 30) sufficient for audits and due diligence.

Deliverables:
- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md` (roles, boundaries, data categories)
- RoPA-lite table (system → data → purpose → lawful basis → retention → access)

DoD:
- A RoPA-lite table exists with columns: system, data category, purpose, lawful basis, retention, residency, access roles, subprocessors.
- A Cloud↔Agent data flow diagram exists and labels Cloud telemetry levels (`AGGREGATED`/`DETAILED_NON_SENSITIVE`/`RAW_ORDER_EVENTS`), and documents RAW gating (enterprise-only, explicit opt-in) and “telemetry stays local” option.
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846`.
- Every listed data store/log stream has: owner, retention, lawful basis, and residency=EU (no blanks).

### Phase 1 — Transparency + legal artifacts aligned to CCEA [COMPLETED - 2025-12-16]

**Status**: ✅ **COMPLETED**

**Goal**: external commitments match the actual CCEA design (no overpromising, no mismatches).

Key work:
- Update Privacy Policy / ToS language to reflect: CCEA zones, what Cloud receives/never receives, telemetry redaction, EU-only, retention and DSAR.
- Finalize DPA templates and “support-with-consent” rules (how consent is captured, logged, and revoked; what data may be shared).
- Establish DSAR intake and identity verification approach proportional to risk.

Deliverables:
- Updated: `docs/legal/PRIVACY_POLICY.md`, `docs/legal/TERMS_OF_SERVICE.md`, `docs/legal/DPA_TEMPLATE.md`
- DSAR SOP (intake, verify, process, respond) + templates

DoD:
- Public/legal docs and engineering reality are consistent (no contradictions about credentials, order data, telemetry levels, EU-only residency, DSAR boundaries).
- “CCEA privacy guarantees” checklist is explicitly stated:
  - Cloud never receives secrets/credentials/env vars
  - No order-like payloads exist in Cloud→Agent protocol commands
  - Telemetry is redacted; defaults to `AGGREGATED`; `DETAILED_NON_SENSITIVE` is opt-in; `RAW_ORDER_EVENTS` is enterprise-only + explicit opt-in (and may be “telemetry stays local” instead)
  - EU-only residency for all data systems and subprocessors
  - DSAR scope is Cloud-only; Agent data remains customer-controlled
- A subprocessors/services list exists that includes region evidence (EU-only) and review timestamps (for inclusion in the evidence pack).
- Support-with-consent is enforceable:
  - A consent record exists (who/what/when/scope/expiry) and is auditable
  - Consent can be revoked and revocation is enforced (support export blocked without active consent)

**Implementation Summary (2025-12-16):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| Updated Privacy Policy | ✅ Done | `docs/legal/PRIVACY_POLICY.md` (v3.0.0) |
| Updated Terms of Service | ✅ Done | `docs/legal/TERMS_OF_SERVICE.md` (v3.0.0) |
| Updated DPA Template | ✅ Done | `docs/legal/DPA_TEMPLATE.md` (v2.0.0) |
| DSAR SOP | ✅ Done | `docs/compliance/DSAR_SOP.md` |
| DSAR Response Templates | ✅ Done | Included in DSAR_SOP.md |
| Subprocessors Register | ✅ Done | `docs/compliance/SUBPROCESSORS_REGISTER.md` |
| Support Consent Policy | ✅ Done | `docs/compliance/SUPPORT_CONSENT_POLICY.md` |
| CCEA Privacy Guarantees Checklist | ✅ Done | `docs/compliance/CCEA_PRIVACY_GUARANTEES_CHECKLIST.md` |
| Support Consent Service (Code) | ✅ Done | `packages/cloud/governance/consent.py` |
| DSAR CCEA Boundary Updates | ✅ Done | `packages/cloud/governance/dsar.py` |
| Tests | ✅ Done | `packages/cloud/governance/tests/test_consent.py`, `test_dsar_phase1.py` |

**Key Additions:**
- Privacy Policy Section 7A: CCEA Privacy Guarantees Checklist
- Privacy Policy Section 7B: Support-with-Consent Policy
- Privacy Policy Section 5.4.3: Telemetry Sensitivity Levels (AGGREGATED/DETAILED_NON_SENSITIVE/RAW_ORDER_EVENTS)
- Terms of Service Section 2.0.1: CCEA Privacy Guarantees (Binding Commitments)
- DPA Section 3.1: Telemetry Sensitivity Levels
- DPA Section 5.9: Support Access with Consent
- Full DSAR SOP with CCEA boundary notice
- Subprocessors Register with EU-only evidence and review timestamps
- SupportConsentService with auditable consent workflow

**Test Results:**
- 96 tests passing for governance module
- 84 tests passing for Phase 2 governance (existing tests - no regression)
- DSAR exports now include CCEA boundary notice

### Phase 2 — Data minimization enforcement (schema/CI + telemetry contracts) [COMPLETED - 2025-12-16]

**Status**: ✅ **COMPLETED**

**Goal**: make violations mechanically hard (or impossible).

Key work:
- Enforce “no order-like payloads” at protocol schema level and CI (explicit prohibited fields like side/qty/price).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1039`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1697`.
- Telemetry contract (Cloud ingestion): `AGGREGATED` default; `DETAILED_NON_SENSITIVE` is opt-in; `RAW_ORDER_EVENTS` is enterprise-only + explicit opt-in; any increase in sensitivity requires explicit config + audit event.
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L855`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L861`.
- Enforce RAW safety boundaries:
  - Non-RAW telemetry rejects order/fill-like fields and intent-like structures (“no order payloads unless RAW”)
  - RAW telemetry is blocked unless enterprise + explicit opt-in is proven and audited
  - RAW telemetry retention is minimized and access is restricted (RBAC + break-glass for support access)
- Mandatory redaction rules (secrets, identifiers, env vars) validated with tests, including “cannot be disabled by feature flag”.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1051`.
- Implement CI guardrails as non-bypassable build constraints (see `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md`):
  - No trading libs in cloud (dependency/import boundary)  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1029`.
  - Artifact signature required (pipeline publish gate; agent rejects unsigned)  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1045`.
- Enforce “new protocol command types require security review” (fail closed without approval/journal entry).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1043`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.
- Enforce digest pinning + registry allowlist invariants (“no latest” and no unknown registries).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L911`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L913`.
- Ensure protocol schema includes `RAW_ORDER_EVENTS` as a telemetry level per Design Doc, while guardrails enforce enterprise-only explicit opt-in and strict validation.
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L742`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`.
- Add regression tests for redaction + schema guardrails.

Deliverables:
- “Telemetry data dictionary (Cloud ingestion)” (allowed/forbidden fields per Cloud telemetry level, using the canonical IDs: `AGGREGATED`/`DETAILED_NON_SENSITIVE`/`RAW_ORDER_EVENTS`)
- “RAW_ORDER_EVENTS handling spec (enterprise-only)”:
  - Cloud ingestion gates (enterprise + explicit opt-in + audit events)
  - Optional Agent-local export path (customer-controlled storage) as “telemetry stays local” mode
  - Retention + encryption + access controls + break-glass expectations
- CI checks/tests proving:
  - order-like payloads rejected
  - secrets/env vars never shipped
  - redaction always enabled
- A protocol change review checklist and journal format for new command types (recorded in the evidence pack change journal).

DoD:
- A PR that attempts to introduce order-like payloads in Cloud→Agent protocol commands, or in non-`RAW_ORDER_EVENTS` telemetry, or introduces secrets/env vars in any telemetry, fails CI.
- A PR that attempts to disable redaction (even via feature flag/config) fails CI and/or tests.
- `RAW_ORDER_EVENTS` exists as a telemetry level, but:
  - any non-enterprise attempt to send RAW is rejected
  - any RAW attempt without explicit opt-in is rejected
  - any order-like fields in non-RAW telemetry are rejected
- A PR that introduces a new command type without a recorded security review approval fails CI.
- A PR that attempts to reference an artifact/config by anything other than digest (or uses "latest") fails CI.

**Implementation Summary (2025-12-16):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| Telemetry Data Dictionary | ✅ Done | `docs/compliance/TELEMETRY_DATA_DICTIONARY.md` |
| RAW_ORDER_EVENTS Handling Spec | ✅ Done | `docs/compliance/RAW_ORDER_EVENTS_HANDLING_SPEC.md` |
| Protocol Change Review Checklist | ✅ Done | `docs/compliance/PROTOCOL_CHANGE_REVIEW.md` |
| Protocol Change Journal | ✅ Done | `docs/compliance/protocol_change_journal.json` |
| TelemetryLevelContract (Code) | ✅ Done | `packages/cloud/governance/telemetry_contract.py` |
| RawOrderEventsGate (Code) | ✅ Done | `packages/cloud/governance/telemetry_contract.py` |
| Protocol Review Check (CI) | ✅ Done | `ccea/guardrails/protocol_review_check.py` |
| Artifact Digest Check (CI) | ✅ Done | `ccea/guardrails/artifact_digest_check.py` |
| Redaction Enforcement Check (CI) | ✅ Done | `ccea/guardrails/redaction_enforcement_check.py` |
| Telemetry Contract Tests | ✅ Done | `packages/cloud/governance/tests/test_telemetry_contract.py` |
| Phase 2 Guardrails Tests | ✅ Done | `ccea/guardrails/tests/test_phase2_guardrails.py` |

**Key Additions:**
- Telemetry Level Contracts with field validation per level (AGGREGATED/DETAILED_NON_SENSITIVE/RAW_ORDER_EVENTS)
- RAW_ORDER_EVENTS enterprise gating with opt-in workflow
- Protocol command security review enforcement with journal format
- Artifact digest pinning and registry allowlist enforcement
- Redaction non-bypassable enforcement (feature flag protection)
- Comprehensive field sets: AGGREGATED_ALLOWED_FIELDS, DETAILED_ALLOWED_FIELDS, RAW_ORDER_ALLOWED_FIELDS, ALWAYS_FORBIDDEN_FIELDS, ORDER_LIKE_FIELDS, PII_FIELDS

**Test Results:**
- 48 tests passing for telemetry contract module
- 47 tests passing for Phase 2 guardrails
- 144 tests passing for governance module (no regression)
- 40 tests passing for Phase 2 governance tests (no regression)

### Phase 3 — EU-only data residency enforcement (tenant/workspace)

**Goal**: residency is a runtime guarantee, not a claim.

Key work:
- Enforce EU region selection and prevent cross-region storage/processing.
- Validate all dependencies are EU-resident, including:
  - primary DBs and replicas
  - object storage + backups/snapshots
  - observability (logs, metrics, traces) + alert routing
  - email delivery and ticketing/support tooling (if used)
  - artifact registry/storage and SBOM storage
- Implement residency policy checks and evidence exports.

Deliverables:
- Residency policy enforcement code + config defaults (EU-only)
- Automated “EU-only drift check” (CI or deployment validation) that fails if any endpoint/bucket/region is not in EU.
- Drift check produces a machine-readable report (e.g., JSON) listing every configured endpoint/bucket/region/subprocessor used at runtime (for evidence pack storage).
- Evidence pack: list of EU services/subprocessors and regions

DoD:
- Automated drift check fails closed if any configured endpoint/storage/support tool is outside EU, and produces a stored report artifact.

### Phase 4 — Retention per tenant + auto-purge + legal hold

**Goal**: storage limitation and lifecycle control (Art. 5(1)(e)).

Key work:
- Define retention schedule per data type (telemetry, audit, access logs, backtest artifacts, support records).
- Implement scheduled purge jobs with auditing (who/what/when; counts deleted).
- Implement legal hold (if needed) with strict access control and audit.

Deliverables:
- Retention policy registry + API/admin flows
- Auto-purge scheduler + audit events for purge runs
- Tests: purge correctness; legal hold prevents deletion for scoped datasets

DoD:
- A scheduled purge run produces an auditable purge event including counts (deleted/archived/aggregated/anonymized) and timestamps per workspace.
- Integration tests seed data older than cutoff and prove it is deleted/changed according to policy; legal hold (if enabled) prevents deletion for the scoped dataset.

### Phase 5 — DSAR: access/export/delete (Cloud data) with CCEA boundary clarity

**Goal**: fulfill rights requests for data you actually control in Cloud (Art. 12–23).

Key work:
- Implement DSAR workflows for:
  - Access (Art. 15)
  - Portability/export (Art. 20)
  - Erasure (Art. 17) for Cloud-controlled datasets
- Clearly define boundary: Agent-zone data is customer-controlled; Cloud cannot export/delete what it never has (must be reflected in process and responses).
- Add DSAR tracking (status, deadlines, extensions) and audit log.

Deliverables:
- DSAR endpoints + service
- Export package format + checksum
- DSAR audit records and metrics

DoD:
- End-to-end tests: create DSAR → (identity verify where required) → export/delete → immutable audit record exists for each step.
- DSAR deadline rules are explicit: standard 30 days, one extension to 60 days when justified; tests prove deadline computation and state transitions.

### Phase 6 — Access control, access audit, and break-glass

**Goal**: least privilege with provable accountability for access to sensitive data.

Key work:
- RBAC inside workspace (read vs admin vs support scopes).
- Access audit log: who accessed what/when, especially for sensitive datasets and DSAR exports.
- Break-glass: **incident-only**, reason required, scope limited, time bounded, fully audited (and included in the evidence pack).
- Change management (Design Doc “trading-impacting” protections):
  - TRADING_IMPACTING changes always require local approval (no silent updates)
  - Approval records include diff/evidence hashes and are exportable
  - Audit trail includes who requested, who approved, and what changed  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.

Deliverables:
- RBAC policy definitions + enforcement points
- Access-audit event schema + storage + export
- Break-glass workflow and logs
- Explicit permission model for deploy/upgrade/change management (who can request vs approve)
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L966`.

DoD:
- Every sensitive access is attributable to a principal and request_id; break-glass additionally has a reason, scope, and expiry time; all are exportable.

### Phase 7 — Security controls (Art. 32) + breach workflow (Art. 33–34)

**Goal**: “appropriate measures” + repeatable incident handling for personal data breaches.

Key work:
- Security baseline: encryption at rest/in transit, key management, MFA for privileged roles, secrets management, logging/monitoring.
- Supply chain (Design Doc 15.1): signed artifacts, pinned digests, allowlist registries, SBOM stored and retrievable by digest.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L909`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L911`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L913`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L915`.
- Agent updates (Design Doc 15.2): signed agent updates, staged rollout, rollback, enterprise version pinning + change windows.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L917`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L921`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L923`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L925`.
- Cloud research execution isolation (Design Doc 15.3): sandboxing, CPU/RAM quotas, egress allowlist, abuse detection.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L927`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L931`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L933`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L935`.
- Personal data breach decision tree and notification workflow (72h to supervisory authority where required).
- Tabletop exercises and evidence retention (runbooks, timelines, outputs).

Deliverables:
- Breach SOP + templates (authority notification, user notification if applicable)
- Tabletop report + evidence artifacts
- Security control checklist mapped to Art. 32 (explicitly including the Design Doc 15.1/15.2/15.3 measures above)

DoD:
- A simulated breach produces a complete notification decision package and evidence trail within defined targets (72h external deadline; internal tabletop produces draft package + timeline within 24h).
- Evidence pack can export: signed artifact inventory + SBOM + change journal + staged rollout/rollback records + research sandbox policy/violations.

### Phase 8 — Continuous compliance (prevent regressions)

**Goal**: compliance stays true as features evolve.

Key work:
- Add CI/PR gates for new telemetry fields, new logs, new data stores: classification + retention + redaction.
- Compliance dashboards: DSAR SLA, purge success, break-glass usage, residency drift = 0.
- Quarterly review cadence: retention schedule, subprocessors list, DSAR metrics, incident learnings.

Deliverables:
- CI “privacy-by-design” checks
- Metrics dashboards and periodic reports

DoD:
- CI fails closed if a new data store/log stream/telemetry field ships without recorded: classification, retention, residency, and redaction requirements (in a registered data inventory entry).

### Phase 9 — Enterprise/on-prem/VPC posture (Design Doc 16.3) and scope control

**Goal**: support enterprise on-prem/VPC deployments in a way that preserves the “software/platform provider” posture and is auditable.

Key work:
- Support enterprise deployment options (on-prem/VPC, or Cloud used only for updates/monitoring by contract).
- Enforce policy options required by the Design Doc:
  - “telemetry stays local” (enterprise)
  - EU-only object store / customer-managed keys (where applicable)
  - RAW telemetry handling (enterprise-only): either “telemetry stays local” OR Cloud `RAW_ORDER_EVENTS` with explicit opt-in + strict controls (as agreed by contract)
- Ensure evidence pack exports remain available in on-prem/air-gapped contexts and are exportable without external connectivity if needed.

Deliverables:
- Enterprise posture note (supported modes, contractual boundaries, and marketing claim guardrails).
- On-prem/VPC deployment checklist including: EU-only data systems, registry mirror, offline verification/signing, evidence export paths, “telemetry stays local” defaults.
- Deployment references to keep in sync with the posture:
  - `deploy/docker/docker-compose.yml`
  - `deploy/helm/ccea-cloud/values-enterprise.yaml`

DoD:
- An on-prem/VPC deployment (EU-only posture) can produce an evidence pack and prove residency/telemetry boundaries in that mode:
  - telemetry stays local by default
  - if Cloud RAW telemetry is enabled (enterprise-only), it is explicitly opted-in, audited, and access-restricted
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L968`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L972`.

## 5) Test strategy (minimum set)

Minimum automated coverage to make the posture durable:

- **Schema/contract tests**: forbid order-like payloads in Cloud→Agent commands; validate telemetry event schema by sensitivity level (including “order-like allowed only in `RAW_ORDER_EVENTS` when gated).
- **Telemetry posture tests**:
  - Cloud telemetry schema contains `AGGREGATED`/`DETAILED_NON_SENSITIVE`/`RAW_ORDER_EVENTS` per Design Doc
  - non-RAW telemetry rejects order/fill-like fields
  - `RAW_ORDER_EVENTS` is rejected unless enterprise + explicit opt-in is proven and audited
- **Redaction tests**: secrets/env vars/account identifiers are always redacted (including nested structures).
- **Residency tests**: EU-only configuration validation (endpoints, buckets, DB regions).
- **Retention tests**: purge jobs remove data past cutoff; legal hold blocks deletion.
- **DSAR integration tests**: access/export/delete workflows and deadline calculations.
- **RBAC/break-glass tests**: unauthorized access denied; break-glass requires reason and expires.
- **Supply-chain tests**: unsigned artifacts rejected; digest pinning enforced; registry allowlist enforced; SBOM required and retrievable.
- **Agent update tests (enterprise)**: signed updates required; staged rollout/rollback produces auditable records; version pinning/change windows enforced.
- **Research isolation tests**: sandbox/egress/quota policies enforced; abuse detection events are generated and exportable.

## 6) Evidence pack (audit-ready, aligned with the Design Doc)

To support customer due diligence and audits, be able to export:

- Artifact inventory: versions, digests, signatures; SBOM; provenance metadata  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L945`.
- Change journal: deploy/upgrade/approval records; config blob digests; who requested/approved; diffs/evidence hashes  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.
- Incident evidence: kill-switch events, halt reasons, incident logs (as applicable)  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L952`.
- Security policies (high level) and operational SOPs relevant to audit (DSAR, retention, breach, break-glass)  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L956`.
- Data lifecycle evidence: retention policies + purge job logs (counts, timestamps) + legal hold actions (if used)
- DSAR evidence: request logs (status, deadlines, identity verification, exports, deletions)
- Access accountability: RBAC policy snapshots + access audit logs + break-glass events (reason, scope, duration)
- Telemetry evidence:
  - telemetry export by sensitivity level:
    - Cloud telemetry: `AGGREGATED` / `DETAILED_NON_SENSITIVE` / `RAW_ORDER_EVENTS` (RAW only when enterprise + explicit opt-in is enabled)
    - Agent-local export option (enterprise, customer-controlled): “telemetry stays local”
  - proof of redaction middleware mandatory (tests + configuration constraints)
  - log export requests with redaction (`REQUEST_EXPORT_LOGS`) as an auditable export type  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1651`.
- EU-only residency evidence:
  - residency policy configuration
  - “EU-only drift check” outputs
  - subprocessors list with EU regions and review timestamps

## 7) References (official and widely-used guidance)

Primary legal text:
- GDPR Regulation (EU) 2016/679 (Articles listed in Section 2).

EU guidance (EDPB):
- EDPB Guidelines on transparency, data subject rights, breach notification, and security measures (use as interpretation guidance when finalizing SOPs/policies).

Operational best practice (for engineering controls):
- ISO/IEC 27001/27002 and NIST CSF as control catalog references for Art. 32 mapping (select only what matches CCEA risk profile).
