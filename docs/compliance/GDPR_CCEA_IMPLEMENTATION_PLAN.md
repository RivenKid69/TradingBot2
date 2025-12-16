# GDPR Implementation Plan (CCEA / EU-only)

**Project**: AI-Powered Quantitative Research Platform (CCEA)  
**Scope**: GDPR controls required for a software provider operating the CCEA model (Cloud-controlled execution; Agent is customer-operated).  
**Deployment**: **EU-only** (no non-EU regions).  
**Primary design source**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt` (Sections 13–16; Privacy/GDPR & Data Residency).

## 0) Why this plan (and what it is not)

This plan implements **only** the GDPR elements needed for this project’s architecture and positioning as a **software/platform provider**, aligned with CCEA constraints:

- **Default telemetry is aggregated**, with **mandatory redaction** before leaving the Agent.
- **No secrets** (API keys, tokens), **no env vars**, and **no order-like payloads** are allowed to reach Cloud (enforced by schema/CI).
- **EU data residency by default** (and in this project: EU-only).
- **Retention per tenant + auto-purge + DSAR export/delete**.
- **RBAC + access audit + break-glass** for exceptional access.

This plan is not a substitute for legal advice. It is an engineering/compliance implementation plan that should be reviewed by counsel for the final determination of roles (Controller/Processor) and policy wording.

## 1) Design Doc requirements (CCEA-specific GDPR subset)

The Design Doc explicitly requires:

- **Data minimization**: collect only what’s necessary for monitoring, billing (if needed), support (with consent).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L873` (14.1), `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1736` (6.1).
- **Telemetry sensitivity levels** with **AGGREGATED default** and raw order events as opt-in/enterprise-only.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846` (13.1–13.2).
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
2. Cloud never receives **order-like payloads** (side/qty/price/order id/fill details) unless explicitly enterprise-only and contractually scoped.
3. Default telemetry is **AGGREGATED**; any increase in sensitivity is explicit, audited, and controlled.
4. EU-only residency: all storage, backups, logs, observability, and support tooling remain in EU.
5. Break-glass access is time-bound, scope-limited, reason-required, and fully audited.
6. Cloud builds **must not** contain broker trading client libraries (import/dependency boundary enforced in CI).  
7. Telemetry redaction is **always on** and cannot be disabled by configuration/feature flag.

### 3.1 Canonical stance: `RAW_ORDER_EVENTS`

The Design Doc allows `RAW_ORDER_EVENTS` as an opt-in sensitivity level, but also flags it as dangerous for privacy/IP.  
Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L846`.

**Decision for this EU-only product posture**:
- `RAW_ORDER_EVENTS` is **disabled by default** and **not available** for retail/pro.  
- If supported at all, it is **enterprise-only**, contractually scoped (DPA + explicit lawful basis), requires a privacy/security review (DPIA trigger), and must be technically gated (server-side + agent-side).

**Required alignment work**:
- Ensure the protocol/schema, documentation, and runtime enforcement are consistent: either
  - remove `RAW_ORDER_EVENTS` from the protocol schema and all docs, or
  - keep it in schema but enforce “enterprise-only” with explicit allowlisting, audits, and tests proving non-enterprise cannot enable it.

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
- Each data store/log stream is assigned an owner, retention, lawful basis, and residency location (EU).

### Phase 1 — Transparency + legal artifacts aligned to CCEA

**Goal**: external commitments match the actual CCEA design (no overpromising, no mismatches).

Key work:
- Update Privacy Policy / ToS language to reflect: CCEA zones, what Cloud receives/never receives, telemetry redaction, EU-only, retention and DSAR.
- Finalize DPA templates and “support-with-consent” rules.
- Establish DSAR intake and identity verification approach proportional to risk.

Deliverables:
- Updated: `docs/legal/PRIVACY_POLICY.md`, `docs/legal/TERMS_OF_SERVICE.md`, `docs/legal/DPA_TEMPLATE.md`
- DSAR SOP (intake, verify, process, respond) + templates

DoD:
- Public/legal docs and engineering reality are consistent (no contradictions about credentials, order data, telemetry levels, EU-only residency, DSAR boundaries).
- “CCEA privacy guarantees” checklist is explicitly stated:
  - Cloud never receives secrets/credentials/env vars
  - No order-like payloads exist in protocol/commands
  - Telemetry is redacted and default AGGREGATED
  - EU-only residency for all data systems and subprocessors
  - DSAR scope is Cloud-only; Agent data remains customer-controlled

### Phase 2 — Data minimization enforcement (schema/CI + telemetry contracts)

**Goal**: make violations mechanically hard (or impossible).

Key work:
- Enforce “no order-like payloads” at protocol schema level and CI (explicit prohibited fields like side/qty/price).  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1039`, `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1697`.
- Telemetry contract: AGGREGATED default; sensitivity increases require explicit config and audit.
- Mandatory redaction rules (secrets, identifiers, env vars) validated with tests, including “cannot be disabled by feature flag”.  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1051`.
- Implement CI guardrails as non-bypassable build constraints (see `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md`):
  - No trading libs in cloud (dependency/import boundary)  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1029`.
  - Artifact signature required (pipeline publish gate; agent rejects unsigned)  
    Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1045`.
- Resolve `RAW_ORDER_EVENTS` posture and enforce it:
  - Decide: “removed from schema” vs “enterprise-only gated”
  - Add tests that prove non-enterprise cannot enable/send raw telemetry.
- Add regression tests for redaction + schema guardrails.

Deliverables:
- “Telemetry data dictionary” (allowed fields per telemetry level)
- CI checks/tests proving:
  - order-like payloads rejected
  - secrets/env vars never shipped
  - redaction always enabled

DoD:
- A PR that attempts to introduce order-like payloads or secrets in telemetry fails CI.
- A PR that attempts to disable redaction (even via feature flag/config) fails CI and/or tests.
- `RAW_ORDER_EVENTS` is either removed from protocol schema, or has enterprise-only gating with tests proving enforcement.

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
- Evidence pack: list of EU services/subprocessors and regions

DoD:
- Automated check proves no configured endpoints/storage locations are outside EU.

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
- Data older than retention window is removed/aggregated/anon’d automatically and provably.

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
- End-to-end tests: create DSAR → verify → export/delete → response within SLA.

### Phase 6 — Access control, access audit, and break-glass

**Goal**: least privilege with provable accountability for access to sensitive data.

Key work:
- RBAC inside workspace (read vs admin vs support scopes).
- Access audit log: who accessed what/when, especially for sensitive datasets and DSAR exports.
- Break-glass: reason required, scope limited, time bounded, fully audited.
- Change management (Design Doc “trading-impacting” protections):
  - TRADING_IMPACTING changes always require local approval (no silent updates)
  - Approval records include diff/evidence hashes and are exportable
  - Audit trail includes who requested, who approved, and what changed  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.

Deliverables:
- RBAC policy definitions + enforcement points
- Access-audit event schema + storage + export
- Break-glass workflow and logs

DoD:
- Every sensitive access is attributable to a principal, reason (if break-glass), and time window.

### Phase 7 — Security controls (Art. 32) + breach workflow (Art. 33–34)

**Goal**: “appropriate measures” + repeatable incident handling for personal data breaches.

Key work:
- Security baseline: encryption at rest/in transit, key management, MFA for privileged roles, secrets management, logging/monitoring.
- Personal data breach decision tree and notification workflow (72h to supervisory authority where required).
- Tabletop exercises and evidence retention (runbooks, timelines, outputs).

Deliverables:
- Breach SOP + templates (authority notification, user notification if applicable)
- Tabletop report + evidence artifacts
- Security control checklist mapped to Art. 32

DoD:
- A simulated breach produces a complete notification package and evidence trail within targets.

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
- No new data flows ship without classification, retention, and redaction requirements.

## 5) Test strategy (minimum set)

Minimum automated coverage to make the posture durable:

- **Schema/contract tests**: forbid order-like payloads; validate telemetry event schema by sensitivity level.
- **Redaction tests**: secrets/env vars/account identifiers are always redacted (including nested structures).
- **Residency tests**: EU-only configuration validation (endpoints, buckets, DB regions).
- **Retention tests**: purge jobs remove data past cutoff; legal hold blocks deletion.
- **DSAR integration tests**: access/export/delete workflows and deadline calculations.
- **RBAC/break-glass tests**: unauthorized access denied; break-glass requires reason and expires.

## 6) Evidence pack (audit-ready, aligned with the Design Doc)

To support customer due diligence and audits, be able to export:

- Artifact inventory: versions, digests, signatures; SBOM; provenance metadata  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L945`.
- Change journal: deploy/upgrade/approval records; config blob digests; who requested/approved; diffs/evidence hashes  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L960`.
- Incident evidence: kill-switch events, halt reasons, incident logs (as applicable)  
  Reference: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L952`.
- Data lifecycle evidence: retention policies + purge job logs (counts, timestamps) + legal hold actions (if used)
- DSAR evidence: request logs (status, deadlines, identity verification, exports, deletions)
- Access accountability: RBAC policy snapshots + access audit logs + break-glass events (reason, scope, duration)
- Telemetry evidence:
  - telemetry export by sensitivity level (AGGREGATED/DETAILED; RAW only if enterprise-gated)
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
