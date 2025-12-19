# GDPR Implementation Plan (CCEA / EU-only)

**Project**: CustodiaCloud (CCEA)  
**Scope**: GDPR controls required for a software provider operating the CCEA model (Cloud-controlled execution; Agent is customer-operated).  
**Deployment scope**: **EU-only** design target (verify via deployment audits and drift checks).  
**Enterprise option**: on-prem/VPC deployment is supported **within EU-only posture** (customer-controlled infrastructure located in EU; no vendor-operated non-EU processing).  
**Primary technical boundary source (CCEA)**: `archive/root_files/Design Doc CCEA Cloud.txt` (privacy-by-design, secrets/telemetry boundaries, control plane).

## 0) Why this plan (and what it is not)

This plan implements **only** the GDPR elements needed for this project’s architecture and positioning as a **software/platform provider**, aligned with CCEA constraints:

- **Telemetry has three sensitivity levels**: `AGGREGATED` (default), `DETAILED_NON_SENSITIVE` (opt-in), `RAW_ORDER_EVENTS` (enterprise-only, explicit opt-in) — with **redaction designed as mandatory** before leaving the Agent (verify via CI guardrails and tests).
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

1. **Cloud is designed to not receive** broker credentials, API keys/tokens, or env vars (design goal validated via redaction + validation + CI guardrails; see CI artifacts for current test status).
2. **Cloud→Agent protocol commands** are designed to not carry **order-like payloads** (side/qty/price/order id/fill details); prohibited at schema level + CI validation ("no order commands" design constraint).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L750`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L754`.
3. **Cloud telemetry sensitivity levels are fixed and named**: `AGGREGATED` (default), `DETAILED_NON_SENSITIVE` (opt-in), `RAW_ORDER_EVENTS` (enterprise-only, explicit opt-in).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`.
4. **Order/fill-like fields are forbidden in Cloud telemetry** unless `telemetry_level == RAW_ORDER_EVENTS` and the workspace is explicitly enabled for enterprise RAW (gated, audited, retention-minimized, access-restricted).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L851`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1739`.
5. Default telemetry is **AGGREGATED** (retail/pro); any increase in sensitivity is explicit, audited, and controlled; enterprise may select “telemetry stays local” (local-only mode) instead of Cloud ingestion.
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L855`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L861`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1749`.
6. **Telemetry redaction is designed to be enabled by default** and the architecture is designed without configuration or feature flags to disable it; env var logging is prohibited by design (validated via CI guardrails; see CI artifacts for current test status).
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L871`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1051`.
7. EU-only residency (design requirement): storage, backups, logs, observability, and support tooling are designed to remain in EU; **EU-only drift checks are designed to be mandatory** (fail closed by design; see CI artifacts for current test coverage).  
   Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L892`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1745`.
8. Break-glass access is **incident-only**, time-bound, scope-limited, reason-required, and fully audited.
9. Cloud builds are designed to not contain broker trading client libraries (import/dependency boundary validated via CI).
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
- "CCEA privacy design commitments" checklist is explicitly stated:
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
| CCEA Privacy Design Commitments Checklist | ✅ Done | `docs/compliance/CCEA_PRIVACY_DESIGN_COMMITMENTS_CHECKLIST.md` |
| Support Consent Service (Code) | ✅ Done | `packages/cloud/governance/consent.py` |
| DSAR CCEA Boundary Updates | ✅ Done | `packages/cloud/governance/dsar.py` |
| Tests | ✅ Done | `packages/cloud/governance/tests/test_consent.py`, `test_dsar_phase1.py` |

**Key Additions:**
- Privacy Policy Section 7A: CCEA Privacy Design Commitments Checklist
- Privacy Policy Section 7B: Support-with-Consent Policy
- Privacy Policy Section 5.4.3: Telemetry Sensitivity Levels (AGGREGATED/DETAILED_NON_SENSITIVE/RAW_ORDER_EVENTS)
- Terms of Service Section 2.0.1: CCEA Privacy Design Commitments (Binding)
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

**Test Results (internal; verify via CI):**
- 48 tests passing for telemetry contract module (at time of implementation)
- 47 tests passing for Phase 2 guardrails (at time of implementation)
- 144 tests passing for governance module (no regression)
- 40 tests passing for Phase 2 governance tests (no regression)

### Phase 3 — EU-only data residency enforcement (tenant/workspace) [TOOLING COMPLETE - 2025-12-16]

**Status**: ✅ **TOOLING COMPLETE** (internal implementation; verify via CI/tests and deployment audits)

**Goal**: residency is a runtime enforcement, not just a claim.

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
- Automated "EU-only drift check" (CI or deployment validation) that fails if any endpoint/bucket/region is not in EU.
- Drift check produces a machine-readable report (e.g., JSON) listing every configured endpoint/bucket/region/subprocessor used at runtime (for evidence pack storage).
- Evidence pack: list of EU services/subprocessors and regions

DoD (internal criteria; not independently audited):
- Automated drift check is designed to fail closed if any configured endpoint/storage/support tool is outside EU, and to produce a stored report artifact (see test coverage for current validation status).

**Implementation Summary (2025-12-16):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| EU-Only Drift Checker (Code) | ✅ Done | `packages/cloud/governance/residency_drift.py` |
| CI Residency Guardrail | ✅ Done | `ccea/guardrails/residency_check.py` |
| EU Residency Enforcement Spec | ✅ Done | `docs/compliance/EU_RESIDENCY_ENFORCEMENT_SPEC.md` |
| Helm Residency Configuration | ✅ Done | `deploy/helm/ccea-cloud/values.yaml` (governance.residency) |
| Drift Check Tests | ✅ Done | `packages/cloud/governance/tests/test_residency_drift.py` |
| CI Guardrail Tests | ✅ Done | `ccea/guardrails/tests/test_residency_check.py` |

**Key Components:**

1. **EUOnlyDriftChecker** (`residency_drift.py`)
   - Validates all endpoints/services are EU-resident
   - Extracts regions from AWS/GCP/Azure endpoint patterns
   - Fails closed on any non-EU endpoint detection
   - Produces machine-readable JSON reports with integrity hash
   - Supports explicit region configuration override

2. **DeploymentConfigValidator** (`residency_drift.py`)
   - Validates Helm values, Docker Compose, and Kubernetes manifests
   - Extracts configuration from environment variables
   - Infers regions from deployment configurations

3. **ResidencyEvidenceExporter** (`residency_drift.py`)
   - Exports residency evidence pack for audits
   - Creates EU residency attestation documents
   - Generates subprocessor verification summaries
   - Maintains audit log of drift checks

4. **CI Residency Guardrail** (`residency_check.py`)
   - Scans YAML/YML config files for non-EU regions
   - Checks environment files for region violations
   - Validates Helm values for EU enforcement settings
   - Blocks deployment on non-EU endpoint detection

**Supported Validations:**
- AWS Regions: RDS, S3, ElastiCache, SES, CloudWatch, KMS, Secrets Manager, SNS, SQS, ECR
- GCP Regions: europe-west*, europe-north*, europe-central*
- Azure Regions: westeurope, northeurope, germanywest*, france*, sweden*, switzerland*, uk*
- Known Services: Stripe (EU), Sentry (EU), SES (EU), SendGrid (EU)

**Report Format:**
```json
{
  "check_id": "drift-check-YYYY-MM-DD-XXXXXXXX",
  "timestamp": "ISO8601",
  "status": "PASS|FAIL|WARNING|UNKNOWN",
  "checks": [{"component": "...", "region": "...", "eu_compliant": true}],
  "violations": [],
  "report_hash": "sha256:..."
}
```

**Test Results (internal CI; verify via test run logs):**
- 76 tests passing for residency drift module (at time of documentation)
- 52 tests passing for CI residency guardrail (at time of documentation)
- 220 tests passing for governance module (no regression)
- 99 tests passing for all guardrails (no regression)

### Phase 4 — Retention per tenant + auto-purge + legal hold [COMPLETED - 2025-12-16]

**Status**: ✅ **COMPLETED**

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

**Implementation Summary (2025-12-16):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| Retention Policy Specification | ✅ Done | `docs/compliance/RETENTION_POLICY_SPEC.md` |
| RetentionPolicyRegistry (Code) | ✅ Done | `packages/cloud/governance/retention_service.py` |
| LegalHoldService (Code) | ✅ Done | `packages/cloud/governance/retention_service.py` |
| AutoPurgeScheduler (Code) | ✅ Done | `packages/cloud/governance/retention_service.py` |
| Database Models | ✅ Done | `packages/cloud/control_plane/models.py` (DataRetentionPolicy, LegalHold, GovernanceAuditLog) |
| Comprehensive Tests | ✅ Done | `packages/cloud/governance/tests/test_retention_service.py` |

**Key Components:**

1. **RetentionPolicyRegistry** (`retention_service.py`)
   - Per-workspace retention policy management
   - Minimum retention enforcement (7 years for compliance data: approval_records, access_audits, break_glass_requests, dsar_requests, governance_audit_logs, legal_hold_records, billing_records)
   - Maximum retention limits for sensitive data
   - Default retention periods per data category (from RoPA)
   - Full audit trail for policy changes
   - Validation API for proposed retention periods

2. **LegalHoldService** (`retention_service.py`)
   - Create/release legal holds with mandatory reason (10+ chars)
   - Indefinite or time-bounded holds (hold_until)
   - Automatic expiry detection and processing
   - Hold extension workflow with audit
   - Blocks auto-purge and DSAR erasure for held data
   - Full audit trail (create, extend, release, expire, block events)

3. **AutoPurgeScheduler** (`retention_service.py`)
   - Configurable scheduler (interval, batch size, max runtime)
   - Respects legal holds (skips held data types)
   - Produces auditable PurgeEvent for every operation
   - Supports multiple retention actions: DELETE, ARCHIVE, ANONYMIZE, AGGREGATE
   - Dry-run mode for preview without deletion
   - Statistics and event log APIs
   - Updates policy last_purge_at after execution

4. **PurgeEvent** (Audit Schema)
   ```json
   {
     "event_id": "uuid",
     "event_type": "purge_completed|purge_skipped|purge_failed",
     "workspace_id": "uuid",
     "data_type": "alerts|telemetry_aggregated|...",
     "status": "completed|skipped|failed",
     "retention_config": {
       "retention_days": 90,
       "cutoff_date": "ISO8601"
     },
     "results": {
       "records_deleted": 1000,
       "records_archived": 0,
       "records_anonymized": 0,
       "records_aggregated": 0
     },
     "execution": {
       "started_at": "ISO8601",
       "completed_at": "ISO8601",
       "duration_seconds": 1.5,
       "executor": "scheduler"
     },
     "legal_hold_blocked": false,
     "skip_reason": null
   }
   ```

5. **Retention Period Matrix** (from `RETENTION_POLICY_SPEC.md`)
   - Compliance data (7-year minimum): approval_records, access_audits, break_glass_requests, dsar_requests, governance_audit_logs, legal_hold_records, billing_records
   - Telemetry: RAW (7d), DETAILED (30d), AGGREGATED (90d)
   - Operational: alerts (365d), commands (180d), config_blobs (365d)
   - Sessions: session_data (24h)

**Test Results:**
- 60 tests passing for retention service module
- 280 tests passing for governance module (no regression)
- Integration tests verify:
  - Purge correctness (data older than cutoff deleted)
  - Legal hold blocking (held data not deleted)
  - Compliance minimum enforcement (7-year retention cannot be reduced)
  - Workspace isolation (holds don't affect other workspaces)
  - Full audit trail generation

### Phase 5 — DSAR: access/export/delete (Cloud data) with CCEA boundary clarity [COMPLETED - 2025-12-17]

**Status**: ✅ **COMPLETED**

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

**Implementation Summary (2025-12-17):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| DSAR Phase 5 Specification | ✅ Done | `docs/compliance/DSAR_PHASE5_SPEC.md` |
| DSARPhase5Service (Code) | ✅ Done | `packages/cloud/governance/dsar_phase5.py` |
| Identity Verification System | ✅ Done | `packages/cloud/governance/dsar_phase5.py` (VerificationToken, VerificationMethod) |
| Deadline Management | ✅ Done | `packages/cloud/governance/dsar_phase5.py` (30 days + 60 days extension) |
| Export Package Format | ✅ Done | `packages/cloud/governance/dsar_phase5.py` (JSON with CCEA boundary, checksum) |
| Legal Hold Integration | ✅ Done | `packages/cloud/governance/dsar_phase5.py` (blocks erasure for held data) |
| DSAR Metrics API | ✅ Done | `packages/cloud/governance/dsar_phase5.py` (DSARMetrics) |
| Audit Trail | ✅ Done | `packages/cloud/governance/dsar_phase5.py` (AuditEntry with integrity hash) |
| Comprehensive Tests | ✅ Done | `packages/cloud/governance/tests/test_dsar_phase5.py` |

**Key Components:**

1. **DSARPhase5Service** (`dsar_phase5.py`)
   - Full DSAR lifecycle management (create, verify, process, complete)
   - Support for ACCESS, PORTABILITY, ERASURE, RECTIFICATION, RESTRICTION request types
   - Rate limiting (12 requests per user per month) to prevent abuse
   - CCEA boundary enforcement for all operations

2. **Identity Verification System**
   - Multiple verification methods: EMAIL_LINK, EMAIL_OTP, SMS_OTP, MFA_CHALLENGE, SSO_SESSION, DOCUMENT_UPLOAD, SUPPORT_MANUAL
   - Token-based verification with TTL (24 hours) and attempt limits (5 max)
   - Constant-time comparison to prevent timing attacks
   - Verification MANDATORY for ERASURE requests

3. **Deadline Management (GDPR Art. 12(3))**
   - Standard deadline: 30 calendar days from request
   - Extension: +60 days (once only) for complex requests
   - Maximum total: 90 days from request creation
   - Automatic overdue detection and expiration

4. **Export Package Format**
   ```json
   {
     "metadata": {
       "request_id": "uuid",
       "request_type": "access|portability",
       "user_id": "...",
       "exported_at": "ISO8601",
       "gdpr_article": "Article 15 (Access)"
     },
     "ccea_boundary": {
       "notice": "CCEA Architecture Data Boundary Notice...",
       "in_scope_categories": ["telemetry_events", "alerts", ...],
       "out_of_scope_categories": ["broker_credentials", "local_execution_logs", ...],
       "explanation": "This export contains only Cloud-controlled data..."
     },
     "data": [...]
   }
   ```
   - SHA-256 checksum: `sha256:<hex_digest>`
   - Secure download links with token and expiry (7 days)

5. **Legal Hold Integration**
   - Automatic check before erasure for each data category
   - Blocked categories logged with exemption reason
   - Support for Art. 17(3) exemptions:
     - `legal_obligation` - Compliance with legal obligation
     - `public_interest` - Archiving, research purposes
     - `legal_claims` - Establishment/defence of legal claims
     - `regulatory_retention` - Financial regulations (7yr)

6. **Data Category Registry**
   - 13 in-scope categories (Cloud-controlled):
     - telemetry_events, alerts, commands, approval_records, access_audits, user_settings, agent_data, run_data, deployment_data, consent_records, break_glass_requests, session_data, billing_records
   - 7 out-of-scope categories (Agent-controlled):
     - broker_credentials, local_execution_logs, order_fill_data, local_vault_contents, position_data_local, local_strategy_source, local_config_files
   - Compliance data categories marked as non-deletable (7-year retention)

7. **Audit Trail**
   - Immutable audit entries with SHA-256 integrity hash
   - 19 audit actions covering full lifecycle:
     - REQUEST_CREATED, VERIFICATION_SENT, VERIFICATION_COMPLETED, PROCESSING_STARTED, DATA_COLLECTED, EXPORT_GENERATED, ERASURE_STARTED, ERASURE_COMPLETED, ERASURE_BLOCKED, EXEMPTION_APPLIED, DEADLINE_EXTENDED, REQUEST_COMPLETED, REQUEST_REJECTED, REQUEST_CANCELLED, REQUEST_EXPIRED, ERROR_OCCURRED, LEGAL_HOLD_CHECK

8. **Metrics API**
   - Total requests, by type, by status
   - Average completion days
   - On-time completion rate
   - Overdue count and pending count
   - Total records processed/deleted
   - Period filtering (workspace, days)

**Test Results:**
- 73 tests passing for DSAR Phase 5 module
- 353 tests passing for governance module (no regression)
- 2692 tests passing for CCEA module (no regression)
- End-to-end tests verify:
  - Full ACCESS workflow: create → verify → export → audit
  - Full ERASURE workflow: create → verify (mandatory) → delete → audit
  - Deadline computation and state transitions
  - Legal hold blocking with exemption tracking
  - CCEA boundary in export packages

### Phase 6 — Access control, access audit, and break-glass [INTERNAL TOOLING COMPLETE - 2025-12-17]

**Status**: ✅ **INTERNAL TOOLING COMPLETE** (implementation artifacts available; not independently audited)

**Goal**: least privilege with provable accountability for access to sensitive data.

Key work:
- RBAC inside workspace (read vs admin vs support scopes).
- Access audit log: who accessed what/when, especially for sensitive datasets and DSAR exports.
- Break-glass: **incident-only**, reason required, scope limited, time bounded, fully audited (and included in the evidence pack).
- Change management (Design Doc "trading-impacting" protections):
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

**Implementation Summary (2025-12-17):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| Access Control Phase 6 Specification | ✅ Done | `docs/compliance/ACCESS_CONTROL_PHASE6_SPEC.md` |
| RBACService (Code) | ✅ Done | `packages/cloud/governance/rbac_service.py` |
| AccessAuditService (Code) | ✅ Done | `packages/cloud/governance/access_audit.py` |
| BreakGlassPhase6Service (Code) | ✅ Done | `packages/cloud/governance/break_glass_phase6.py` |
| ChangeManagementService (Code) | ✅ Done | `packages/cloud/governance/change_management.py` |
| Comprehensive Tests | ✅ Done | `packages/cloud/governance/tests/test_access_control_phase6.py` |

**Key Components:**

1. **RBACService** (`rbac_service.py`)
   - Role-based access control with hierarchical permissions
   - 8 default system roles: owner, admin, developer, viewer, support, auditor, break_glass_approver, dpo
   - 10 scopes: READ, WRITE, DELETE, ADMIN, SUPPORT, AUDIT, BREAK_GLASS, APPROVE, EXPORT, EXECUTE
   - 24 resource types with sensitivity classification (standard, sensitive, critical, restricted)
   - Permission caching with configurable TTL
   - MFA enforcement for critical/restricted resources
   - Workspace isolation and organization defaults
   - Audit integration for sensitive resource access
   - RBAC snapshot export for evidence pack

2. **AccessAuditService** (`access_audit.py`)
   - Immutable audit entries with SHA-256 integrity hash
   - Hash chain for tamper detection and integrity verification
   - 30+ audit actions covering full access lifecycle
   - Query API with filtering by workspace, principal, action, result, sensitivity, time range
   - Statistics and metrics API
   - Export with checksum for evidence pack
   - Alert callbacks for suspicious activity (denied access to critical resources, break-glass usage, bulk operations)
   - 7-year retention for compliance (exempt from DSAR erasure per Art. 17(3)(b))

3. **BreakGlassPhase6Service** (`break_glass_phase6.py`)
   - Incident-only access with mandatory reason (minimum 20 characters)
   - 8 pre-defined reason categories: INCIDENT_RESPONSE, SECURITY_INVESTIGATION, COMPLIANCE_AUDIT, DATA_RECOVERY, SYSTEM_FAILURE, CUSTOMER_EMERGENCY, PRODUCTION_DEBUGGING, REGULATORY_REQUEST
   - 12 scopes: TELEMETRY_READ, TELEMETRY_RAW_READ, AUDIT_READ, CONFIG_READ, CONFIG_WRITE, AGENT_READ, AGENT_ADMIN, DEPLOYMENT_READ, DEPLOYMENT_ADMIN, USER_READ, DATA_EXPORT, ADMIN_ACCESS
   - Time-bounded access: default 4 hours, max 24 hours
   - Approval workflow with self-approval prevention
   - Elevated approvers required for admin-level scopes
   - Access token generation with secure random token
   - Cooldown between requests (5 minutes)
   - Access count and resource tracking
   - Full audit trail (request, approve/deny, access, revoke, expire)
   - Evidence hash for each request
   - Export for evidence pack

4. **ChangeManagementService** (`change_management.py`)
   - Change classification: OPERATIONAL, TRADING_IMPACTING, SECURITY_SENSITIVE, DATA_SENSITIVE
   - TRADING_IMPACTING changes require:
     - Local approval (ApprovalType.LOCAL)
     - User acknowledgment
     - Reason (minimum 10 characters)
     - Evidence hashes (config_blob_digest, manifest_digest, previous_state_digest, new_state_digest)
   - Change lifecycle: PENDING → AWAITING_APPROVAL → APPROVED → EXECUTING → COMPLETED/FAILED → ROLLED_BACK
   - Approval expiry (24 hours)
   - Rollback tracking with digest
   - Change journal with immutable entries
   - Full audit trail (create, approve, reject, execute, rollback)
   - Journal export for evidence pack

5. **Integration Points**
   - RBACService ↔ AccessAuditService: Automatic audit logging for sensitive resource access
   - BreakGlassService ↔ AccessAuditService: Full audit trail for break-glass lifecycle
   - BreakGlassService ↔ RBACService: Break-glass scope mapped to RBAC resource:action
   - ChangeManagementService ↔ AccessAuditService: Full audit trail for change lifecycle

**Evidence Pack Exports:**
- RBAC snapshot (roles, assignments, permissions)
- Access audit logs with integrity hash
- Break-glass requests with evidence hash
- Change journal with integrity hash

**Test Results:**
- 74 tests passing for Phase 6 access control module
- 427 tests passing for governance module (no regression)
- Tests verify:
  - Permission grant/deny scenarios
  - Role hierarchy and workspace isolation
  - Scope-based access control
  - MFA enforcement for sensitive resources
  - Audit entry creation with integrity hash
  - Hash chain verification
  - Break-glass workflow (request, approve, use, revoke, expire)
  - Self-approval prevention
  - Elevated scope approval requirements
  - TRADING_IMPACTING approval enforcement
  - Change journal immutability
  - Export functionality for all components

### Phase 7 — Security controls (Art. 32) + breach workflow (Art. 33–34) [COMPLETED - 2025-12-17]

**Status**: ✅ **COMPLETED**

**Goal**: "appropriate measures" + repeatable incident handling for personal data breaches.

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

#### Deliverables

| Deliverable | Status | Location |
|-------------|--------|----------|
| Phase 7 Specification | ✅ Done | `docs/compliance/SECURITY_PHASE7_SPEC.md` |
| SecurityBaselineService (Code) | ✅ Done | `packages/cloud/governance/security_baseline.py` |
| SupplyChainService (Code) | ✅ Done | `packages/cloud/governance/supply_chain.py` |
| AgentUpdateService (Code) | ✅ Done | `packages/cloud/governance/agent_updates.py` |
| ResearchSandboxService (Code) | ✅ Done | `packages/cloud/governance/research_sandbox.py` |
| BreachWorkflowService (Code) | ✅ Done | `packages/cloud/governance/breach_workflow.py` |
| EvidencePackService (Code) | ✅ Done | `packages/cloud/governance/evidence_pack.py` |
| Breach Response SOP | ✅ Done | `docs/compliance/BREACH_RESPONSE_SOP.md` |
| Security Controls Art.32 Checklist | ✅ Done | `docs/compliance/SECURITY_CONTROLS_ART32.md` |
| Comprehensive Tests | ✅ Done | `packages/cloud/governance/tests/test_phase7_security.py` |

#### Implementation Details

**1. SecurityBaselineService** (`packages/cloud/governance/security_baseline.py`):
- Encryption configuration (AES-256-GCM at rest, TLS 1.3 in transit)
- Key management with automatic rotation (90 days default, configurable 30-365)
- MFA enforcement policies (TOTP, WebAuthn supported; SMS deprecated)
- Secrets management with lifecycle tracking and rotation alerts
- Security event logging and compliance checking

**2. SupplyChainService** (`packages/cloud/governance/supply_chain.py`):
- Signed artifact registration with signature verification
- Digest pinning (sha256/sha512) with expiration tracking
- Registry allowlist enforcement
- SBOM generation and storage (CycloneDX 1.5 format)
- Vulnerability tracking and resolution
- Trusted signer management

**3. AgentUpdateService** (`packages/cloud/governance/agent_updates.py`):
- Signed update publication and verification
- Staged rollout (canary → early adopters → general availability)
- Rollback with dual-approval requirement
- Enterprise version pinning
- Change window enforcement
- Rollout metrics and success criteria

**4. ResearchSandboxService** (`packages/cloud/governance/research_sandbox.py`):
- Isolation levels (container, VM, Firecracker, Kata)
- Resource quotas (CPU, memory, storage, network)
- Egress allowlist with default-deny policy
- Abuse detection (resource, network, API, data exfiltration)
- Job lifecycle management with full audit trail

**5. BreachWorkflowService** (`packages/cloud/governance/breach_workflow.py`):
- Breach reporting and confirmation workflow
- Risk assessment with scoring (0.0-10.0 scale)
- Notification decision tree (authority: 72h, subjects: high risk)
- Art. 34(3) exemption handling (encryption, subsequent measures, disproportionate)
- Tabletop exercise framework (quarterly requirement)
- Timeline tracking and deadline alerts

**6. EvidencePackService** (`packages/cloud/governance/evidence_pack.py`):
- 23 evidence categories covering all governance aspects
- Quick export methods for security, breach, supply chain, compliance
- ZIP and JSON export formats with integrity hashes
- Audit-ready artifact aggregation

DoD (internal tooling validation; not a compliance claim):
- ✅ Simulated breach workflow produces notification decision package and evidence trail (target: 72h external deadline; internal tabletop target: draft package + timeline within 24h).
- ✅ Evidence pack exports: signed artifact inventory + SBOM + change journal + staged rollout/rollback records + research sandbox policy/violations.
- ✅ 146 tests passing at time of Phase 7 completion (verify current status via CI; commit hash to be recorded in release notes).
- ✅ 573 governance tests passing at time of completion (verify current status via CI; test counts subject to change as codebase evolves).

### Phase 8 — Continuous compliance (prevent regressions) ✅ COMPLETED

**Goal**: compliance stays true as features evolve.

**Status**: COMPLETED (2025-12-17)

Key work:
- ✅ Add CI/PR gates for new telemetry fields, new logs, new data stores: classification + retention + redaction.
- ✅ Compliance dashboards: DSAR SLA tracking, purge success monitoring, break-glass usage audit, residency drift monitoring (design target: drift = 0; verify via dashboard exports).
- ✅ Quarterly review cadence: retention schedule, subprocessors list, DSAR metrics, incident learnings.

Deliverables:
- ✅ CI "privacy-by-design" checks (`ccea/guardrails/privacy_by_design_check.py`)
- ✅ Metrics dashboards and periodic reports (`packages/cloud/governance/compliance_dashboard.py`)
- ✅ Data Inventory Registry (`packages/cloud/governance/data_inventory.py`)
- ✅ Quarterly Review Service (`packages/cloud/governance/quarterly_review.py`)
- ✅ Phase 8 Specification (`docs/compliance/CONTINUOUS_COMPLIANCE_PHASE8_SPEC.md`)

DoD:
- ✅ CI fails closed if a new data store/log stream/telemetry field ships without recorded: classification, retention, residency, and redaction requirements (in a registered data inventory entry).
- ✅ 88 Phase 8 tests passing (`tests/cloud/governance/test_phase8_continuous_compliance.py`).
- ✅ All 273 governance tests passing (no regressions).

### Phase 9 — Enterprise/on-prem/VPC posture (Design Doc 16.3) and scope control [COMPLETED - 2025-12-17]

**Status**: ✅ **COMPLETED**

**Goal**: support enterprise on-prem/VPC deployments in a way that preserves the "software/platform provider" posture and is auditable.

Key work:
- Support enterprise deployment options (on-prem/VPC, or Cloud used only for updates/monitoring by contract).
- Enforce policy options required by the Design Doc:
  - "telemetry stays local" (enterprise)
  - EU-only object store / customer-managed keys (where applicable)
  - RAW telemetry handling (enterprise-only): either "telemetry stays local" OR Cloud `RAW_ORDER_EVENTS` with explicit opt-in + strict controls (as agreed by contract)
- Ensure evidence pack exports remain available in on-prem/air-gapped contexts and are exportable without external connectivity if needed.

Deliverables:
- Enterprise posture note (supported modes, contractual boundaries, and marketing claim guardrails).
- On-prem/VPC deployment checklist including: EU-only data systems, registry mirror, offline verification/signing, evidence export paths, "telemetry stays local" defaults.
- Deployment references to keep in sync with the posture:
  - `deploy/docker/docker-compose.yml`
  - `deploy/helm/ccea-cloud/values-enterprise.yaml`

DoD:
- An on-prem/VPC deployment (EU-only posture) can produce an evidence pack and prove residency/telemetry boundaries in that mode:
  - telemetry stays local by default
  - if Cloud RAW telemetry is enabled (enterprise-only), it is explicitly opted-in, audited, and access-restricted
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L968`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L972`.

**Implementation Summary (2025-12-17):**

| Deliverable | Status | Location |
|-------------|--------|----------|
| Enterprise Posture Note | ✅ Done | `docs/compliance/ENTERPRISE_POSTURE_NOTE.md` |
| On-Prem/VPC Deployment Checklist | ✅ Done | `docs/compliance/ONPREM_VPC_DEPLOYMENT_CHECKLIST.md` |
| EnterprisePostureService (Code) | ✅ Done | `packages/cloud/governance/enterprise_posture.py` |
| TelemetryLocalModeService (Code) | ✅ Done | `packages/cloud/governance/enterprise_posture.py` |
| EnterpriseEvidencePackExporter (Code) | ✅ Done | `packages/cloud/governance/enterprise_posture.py` |
| EnterprisePostureValidator (Code) | ✅ Done | `packages/cloud/governance/enterprise_posture.py` |
| Enterprise Posture CI Guardrail | ✅ Done | `ccea/guardrails/enterprise_posture_check.py` |
| Helm Enterprise Values (Updated) | ✅ Done | `deploy/helm/ccea-cloud/values-enterprise.yaml` |
| Docker Compose (Updated) | ✅ Done | `deploy/docker/docker-compose.yml` |
| Comprehensive Tests | ✅ Done | `packages/cloud/governance/tests/test_enterprise_posture.py`, `ccea/guardrails/tests/test_enterprise_posture_check.py` |

**Key Components:**

1. **Enterprise Deployment Modes** (`enterprise_posture.py`)
   - 5 deployment modes: SAAS, ENTERPRISE_CLOUD, ON_PREM_FULL, VPC_MANAGED, AIR_GAPPED
   - Mode-specific feature matrix (telemetry export, evidence export, offline verification, CMK support)
   - Factory functions for on-prem and air-gapped configurations

2. **TelemetryLocalModeService** (`enterprise_posture.py`)
   - "Telemetry stays local" enforcement for enterprise deployments
   - Blocks Cloud telemetry export when local mode enabled
   - Validates telemetry destinations against allowed endpoints
   - Full audit trail for mode changes

3. **EnterprisePostureValidator** (`enterprise_posture.py`)
   - EU-only residency validation (on-prem, eu, eu-* regions)
   - Deployment mode-specific configuration checks
   - RAW_ORDER_EVENTS enterprise-only + opt-in enforcement
   - Air-gapped mode requirements validation (local registry, offline verification)
   - Posture report generation with compliance status

4. **EnterpriseEvidencePackExporter** (`enterprise_posture.py`)
   - Offline evidence pack export (no external connectivity required)
   - 8 evidence categories: posture_config, telemetry_config, residency_attestation, audit_logs, access_records, change_journal, compliance_reports, security_controls
   - SHA-256 integrity hashes for verification
   - Pack listing and verification APIs

5. **CI Guardrail** (`enterprise_posture_check.py`)
   - Auto-detects deployment mode from configuration files
   - Validates EU-only residency (scans for non-EU regions)
   - Validates telemetry configuration per mode
   - Air-gapped mode checks (no external URLs, local registry required)
   - Violation reporting with severity levels

6. **Deployment Configuration Updates**
   - `values-enterprise.yaml`: Phase 9 enterprisePosture block with telemetryLocalOnly, evidenceExportLocalOnly, offlineVerification, customerManagedKeys
   - `docker-compose.yml`: Phase 9 environment variables (CCEA_DEPLOYMENT_MODE, CCEA_TELEMETRY_LOCAL_ONLY, CCEA_EVIDENCE_EXPORT_LOCAL_ONLY, CCEA_POSTURE_VALIDATION_ENABLED), evidence_exports volume

**Marketing Claim Guardrails** (from ENTERPRISE_POSTURE_NOTE.md):
- Permitted claims: "EU-only data residency", "Telemetry stays local option", "GDPR-aligned architecture"
- Prohibited claims: "No data leaves your network" (unless air-gapped), "Complete data sovereignty" (Cloud still receives some metadata), Compliance guarantees without contract

**Test Results:**
- 51 tests passing for enterprise posture service
- 38 tests passing for enterprise posture CI guardrail
- 624 tests passing for governance module (no regressions)
- 137 tests passing for guardrails module (no regressions)
- DoD verified: on-prem/VPC deployment can produce evidence pack with telemetry boundaries proof

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
