# CCEA Privacy Design Commitments Checklist

**Document Type**: Compliance Verification Checklist
**Version**: 1.0.0
**Last Updated**: 2025-12-16
**Owner**: Data Protection Officer
**Purpose**: Explicit verification that CCEA privacy design commitments are enforced by architecture
**Scope**: EU-only CCEA Cloud platform

---

## 1. Overview

This checklist provides explicit verification that the CCEA (Cloud-Controlled Execution Architecture) privacy design commitments are being enforced. These design commitments are:

1. **Architectural invariants** - Cannot be overridden by configuration
2. **Legally binding** - Referenced in Privacy Policy, Terms of Service, and DPA
3. **Verifiable** - Enforced by schema, CI guardrails, and runtime checks
4. **Auditable** - Evidence exportable for customer due diligence

---

## 2. Privacy Design Commitment Categories

### 2.1 Cloud Does Not Receive Secrets (by design)

| # | Design Commitment | Enforcement | Verification Method | Status |
|---|-----------|-------------|---------------------|--------|
| S-01 | No broker API keys in Cloud | Schema validation | Build-time CI test | [ ] |
| S-02 | No broker API secrets in Cloud | Redaction middleware | Runtime + CI test | [ ] |
| S-03 | No OAuth tokens in Cloud | Protocol prohibition | Schema validation | [ ] |
| S-04 | No environment variables in Cloud telemetry | Telemetry schema | CI test | [ ] |
| S-05 | No passwords/passphrases in Cloud | Redaction (mandatory) | Runtime check | [ ] |
| S-06 | Redaction cannot be disabled | No feature flag | CI test + code review | [ ] |

**Evidence Required:**
- [ ] CI test results showing rejection of secrets
- [ ] Redaction middleware configuration (shows no disable option)
- [ ] Schema definition prohibiting secret fields

### 2.2 No Order-like Payloads in Protocol

| # | Design Commitment | Enforcement | Verification Method | Status |
|---|-----------|-------------|---------------------|--------|
| O-01 | No `side` (buy/sell) in commands | JSON Schema | Build-time CI | [ ] |
| O-02 | No `quantity` in commands | JSON Schema | Build-time CI | [ ] |
| O-03 | No `price` in commands | JSON Schema | Build-time CI | [ ] |
| O-04 | No `order_id` in commands | JSON Schema | Build-time CI | [ ] |
| O-05 | No `target_position` in commands | JSON Schema | Build-time CI | [ ] |
| O-06 | No `fill_price`/`fill_qty` in commands | JSON Schema | Build-time CI | [ ] |
| O-07 | No `instrument`/`symbol` in commands | JSON Schema | Build-time CI | [ ] |
| O-08 | Forbidden command types blocked | Schema allowlist | CI + runtime | [ ] |

**Forbidden Command Types:**
- `PLACE_ORDER`
- `SUBMIT_ORDER`
- `EXECUTE_SIGNAL`
- `SET_TARGET_POSITION_NOW`
- Any command with trading payload

**Evidence Required:**
- [ ] JSON Schema showing prohibited fields
- [ ] CI test showing rejection of order-like payloads
- [ ] Allowed command type allowlist

### 2.3 Telemetry Sensitivity Levels

| # | Design Commitment | Enforcement | Verification Method | Status |
|---|-----------|-------------|---------------------|--------|
| T-01 | Default telemetry is AGGREGATED | Config default | Runtime check | [ ] |
| T-02 | DETAILED_NON_SENSITIVE requires opt-in | Explicit configuration | Audit event | [ ] |
| T-03 | RAW_ORDER_EVENTS requires enterprise | Tier check | Runtime rejection | [ ] |
| T-04 | RAW_ORDER_EVENTS requires explicit opt-in | Consent record | Audit trail | [ ] |
| T-05 | Order data forbidden in AGGREGATED | Schema validation | CI + runtime | [ ] |
| T-06 | Order data forbidden in DETAILED_NON_SENSITIVE | Schema validation | CI + runtime | [ ] |
| T-07 | Account identifiers masked in all levels | Redaction | CI test | [ ] |
| T-08 | RAW telemetry minimal retention (7 days) | Retention policy | Auto-purge audit | [ ] |

**Telemetry Level Matrix:**

| Field | AGGREGATED | DETAILED | RAW |
|-------|------------|----------|-----|
| PnL % | Allowed | Allowed | Allowed |
| Drawdown % | Allowed | Allowed | Allowed |
| Error rates | Allowed | Allowed | Allowed |
| Timestamps | Forbidden | Allowed | Allowed |
| Queue depths | Forbidden | Allowed | Allowed |
| Order events | **Forbidden** | **Forbidden** | Allowed (masked) |
| Fill events | **Forbidden** | **Forbidden** | Allowed (masked) |
| Account IDs | **Forbidden** | **Forbidden** | Masked only |

**Evidence Required:**
- [ ] Telemetry schema per level
- [ ] Default configuration showing AGGREGATED
- [ ] CI tests for field rejection
- [ ] Enterprise gate implementation
- [ ] Consent record schema

### 2.4 EU-only Data Residency

| # | Design Commitment | Enforcement | Verification Method | Status |
|---|-----------|-------------|---------------------|--------|
| R-01 | All storage in EU | Region config | Drift check | [ ] |
| R-02 | All backups in EU | Backup region policy | Drift check | [ ] |
| R-03 | All logs in EU | CloudWatch region lock | Drift check | [ ] |
| R-04 | All sub-processors in EU | Contractual + DPA | Quarterly review | [ ] |
| R-05 | Drift check is fail-closed | Deployment gate | CI test | [ ] |
| R-06 | Drift check runs hourly | Scheduled job | Cron config | [ ] |

**Approved EU Regions:**
- `eu-central-1` (Frankfurt, Germany)
- `eu-west-1` (Dublin, Ireland)

**Evidence Required:**
- [ ] Infrastructure configuration
- [ ] Drift check report (last run)
- [ ] Sub-processor register with EU evidence
- [ ] DPA status for each sub-processor

### 2.5 DSAR Scope Boundaries

| # | Design Commitment | Enforcement | Verification Method | Status |
|---|-----------|-------------|---------------------|--------|
| D-01 | DSAR scope is Cloud-only | Architecture | Process documentation | [ ] |
| D-02 | Agent data is customer-controlled | No Cloud access | Cannot export | [ ] |
| D-03 | Response includes boundary explanation | Template enforcement | SOP compliance | [ ] |
| D-04 | Export package excludes Agent data | Export generator | QA review | [ ] |

**IN SCOPE for DSAR (Cloud-controlled):**
- User account data
- Organization membership
- Workspace membership
- Strategy metadata
- Telemetry data (at enabled level)
- Command history
- Approval records
- Access audit logs
- Support records

**OUT OF SCOPE for DSAR (Agent-controlled):**
- Broker credentials
- Local execution logs
- Order/fill data (unless RAW enabled)
- Local vault contents
- Position data (unless transmitted)

**Evidence Required:**
- [ ] DSAR SOP with boundary clarification
- [ ] Export package contents list
- [ ] Response templates with boundary text

### 2.6 Support-with-Consent

| # | Design Commitment | Enforcement | Verification Method | Status |
|---|-----------|-------------|---------------------|--------|
| C-01 | Support access requires consent | Access control | Runtime enforcement | [ ] |
| C-02 | Consent record has required fields | Schema validation | CI test | [ ] |
| C-03 | Consent has expiry | Time-based enforcement | Runtime check | [ ] |
| C-04 | Consent can be revoked | Revocation endpoint | Integration test | [ ] |
| C-05 | Revocation is immediate | Access blocked | Runtime check | [ ] |
| C-06 | All access is logged | Audit trail | Log verification | [ ] |
| C-07 | Data export blocked without consent | Export gate | Integration test | [ ] |

**Consent Record Required Fields:**
- consent_id
- user_id
- workspace_id
- granted_at
- expires_at
- scope
- purpose
- support_ticket_id
- revoked_at (if revoked)

**Evidence Required:**
- [ ] Support consent policy
- [ ] Consent record schema
- [ ] Access control implementation
- [ ] Audit trail sample

---

## 3. Verification Procedures

### 3.1 Automated Verification

Run automated checks:

```bash
# Run CCEA privacy design commitment tests
pytest tests/compliance/test_ccea_privacy_design_commitments.py -v

# Run schema validation tests
pytest tests/compliance/test_protocol_schema.py -v

# Run telemetry level tests
pytest tests/compliance/test_telemetry_levels.py -v

# Run EU residency drift check
python scripts/eu_drift_check.py --report

# Run support consent tests
pytest tests/compliance/test_support_consent.py -v
```

### 3.2 Manual Verification Checklist

**Weekly:**
- [ ] Review drift check reports
- [ ] Verify no failed consent requests due to missing controls
- [ ] Review audit log for anomalies

**Monthly:**
- [ ] Review DSAR response samples for boundary compliance
- [ ] Verify telemetry level configurations
- [ ] Review support consent metrics

**Quarterly:**
- [ ] Full sub-processor audit
- [ ] DPA review status check
- [ ] Security review of schema changes
- [ ] Training compliance verification

---

## 4. Evidence Pack Contents

For customer due diligence and audits, export:

| Evidence Type | Description | Format |
|---------------|-------------|--------|
| Privacy design commitments checklist | This document (completed) | PDF/MD |
| CI test results | Automated verification | JSON/HTML |
| Schema definitions | Protocol schema | JSON Schema |
| Drift check reports | EU residency verification | JSON |
| Sub-processor register | EU-only evidence | PDF/MD |
| Consent policy | Support access rules | PDF/MD |
| DSAR SOP | Rights request process | PDF/MD |
| Audit log samples | Access accountability | JSON |

---

## 5. Non-Compliance Escalation

If any design commitment is found non-compliant:

### 5.1 Severity Levels

| Severity | Definition | Response Time |
|----------|------------|---------------|
| Critical | Secrets exposed, order data leaked | Immediate (< 1 hour) |
| High | EU residency breach, consent bypassed | < 4 hours |
| Medium | Audit gap, documentation mismatch | < 24 hours |
| Low | Process deviation, minor config issue | < 7 days |

### 5.2 Escalation Path

```
Level 1: Engineering Team (fix)
    ↓
Level 2: DPO (assess + communicate)
    ↓
Level 3: Legal (determine notification)
    ↓
Level 4: Executive (external comms)
```

### 5.3 Incident Response

For Critical/High severity:
1. Isolate affected systems
2. Assess scope and impact
3. Notify DPO immediately
4. Document incident timeline
5. Implement fix
6. Conduct post-incident review
7. Update controls if needed

---

## 6. Certification Sign-off

### 6.1 Quarterly Certification

The following roles must sign off quarterly:

| Role | Responsibility | Signature | Date |
|------|----------------|-----------|------|
| Engineering Lead | Technical controls verified | __________ | ________ |
| DPO | Policy compliance verified | __________ | ________ |
| Security Lead | Security controls verified | __________ | ________ |
| Legal | Documentation accuracy | __________ | ________ |

### 6.2 Internal Verification Statement

> We verify that the CCEA privacy design commitments documented in this checklist have been internally reviewed as of the date signed. All automated tests passed at the time of signing (per internal CI), manual verification has been completed, and evidence has been archived. This is an internal verification, not an external certification.

---

## 7. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | Compliance Team | Initial release - GDPR Phase 1 |

---

## 8. References

- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt` - Sections 13-16
- `docs/legal/PRIVACY_POLICY.md` - Section 7A
- `docs/legal/TERMS_OF_SERVICE.md` - Section 2.0.1
- `docs/legal/DPA_TEMPLATE.md`
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`
- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md`
- `docs/compliance/SUBPROCESSORS_REGISTER.md`
- `docs/compliance/SUPPORT_CONSENT_POLICY.md`
- `docs/compliance/DSAR_SOP.md`

---

## Appendix A: Compliance Test Matrix

| Test ID | Description | Category | Automated | Manual |
|---------|-------------|----------|-----------|--------|
| CT-001 | Secrets rejected in telemetry | Secrets | Yes | No |
| CT-002 | Redaction always enabled | Secrets | Yes | No |
| CT-003 | Order fields rejected in commands | Orders | Yes | No |
| CT-004 | Forbidden command types blocked | Orders | Yes | No |
| CT-005 | AGGREGATED is default | Telemetry | Yes | No |
| CT-006 | DETAILED requires opt-in | Telemetry | Yes | No |
| CT-007 | RAW requires enterprise + opt-in | Telemetry | Yes | No |
| CT-008 | Order data rejected in non-RAW | Telemetry | Yes | No |
| CT-009 | All regions are EU | Residency | Yes | No |
| CT-010 | Drift check fails non-EU | Residency | Yes | No |
| CT-011 | Sub-processors are EU-only | Residency | No | Yes |
| CT-012 | DSAR export excludes Agent data | DSAR | Yes | Yes |
| CT-013 | DSAR response has boundary text | DSAR | Yes | Yes |
| CT-014 | Support access requires consent | Consent | Yes | No |
| CT-015 | Consent revocation is immediate | Consent | Yes | No |
| CT-016 | Export blocked without consent | Consent | Yes | No |

---

## Appendix B: Quick Reference Card

**CCEA Privacy Design Commitments - Quick Reference**

```
┌─────────────────────────────────────────────────────────────────────────┐
│           CCEA PRIVACY DESIGN COMMITMENTS                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. SECRETS: Cloud does NOT receive (by design)                         │
│     ❌ API keys  ❌ Secrets  ❌ Tokens  ❌ Env vars  ❌ Passwords        │
│                                                                          │
│  2. ORDERS: Cloud→Agent commands do NOT contain                         │
│     ❌ side  ❌ quantity  ❌ price  ❌ order_id  ❌ target_position       │
│                                                                          │
│  3. TELEMETRY LEVELS:                                                   │
│     AGGREGATED (default) → No order data, no account IDs               │
│     DETAILED (opt-in) → No order data, no account IDs                  │
│     RAW (enterprise+opt-in) → Order data (masked), minimal retention   │
│                                                                          │
│  4. RESIDENCY: EU by default                                            │
│     ✓ eu-central-1 (Frankfurt)  ✓ eu-west-1 (Dublin)                   │
│     ❌ All other regions                                                │
│                                                                          │
│  5. DSAR: Cloud data only                                               │
│     ✓ User accounts  ✓ Telemetry  ✓ Commands  ✓ Audit logs             │
│     ❌ Broker credentials  ❌ Local logs  ❌ Order data (unless RAW)    │
│                                                                          │
│  6. SUPPORT: Consent required                                           │
│     ✓ Explicit consent  ✓ Time-limited  ✓ Revocable  ✓ Audited        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```
