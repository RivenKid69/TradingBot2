# On-Call Capacity Validation
## SLA Notification Tier Assessment

**Version**: 1.0
**Date**: 2025-01-17
**Status**: Validated
**Reference**: DORA Article 30(2)(f), DORA_OPERATIONAL_RESILIENCE_PLAN.md Section 5.4.5

---

## 1. Purpose

This document validates our current on-call capacity and sets achievable notification SLAs based on actual operational capability. Per DORA Article 30(2)(f), we must provide incident assistance to clients, and this requires honest assessment of our response capabilities.

---

## 2. Current On-Call Capacity Assessment

### 2.1 Team Size

| Role | Target Headcount | On-Call Trained | Notes |
|------|------------------|-----------------|-------|
| Platform Engineers | 3 (planned) | TBD | Full stack capability |
| Security Engineer | 1 (planned) | TBD | Security focus |
| DevOps/SRE | 1 (planned) | TBD | Infrastructure focus |
| **Total On-Call Pool** | **5 (target)** | **TBD** | Hiring/training in progress |

### 2.2 Coverage Model

| Option | Description | Current Status | Sustainable |
|--------|-------------|----------------|-------------|
| **A: Business Hours** | 9am-6pm Mon-Fri | ✓ Current | ✓ Yes |
| **B: Extended Hours** | 7am-11pm Mon-Fri | Possible | ✓ With rotation |
| **C: On-Call Rotation** | 24/7 with pager | Possible | ⚠️ 2 engineers minimum |
| **D: Dedicated 24/7** | Full 24/7 team | Not feasible | ✗ Needs 4+ FTE |

### 2.3 Current Coverage

**Business Hours (Current Default)**:
- Monday-Friday: 9:00 - 18:00 CET
- Minimum 2 engineers available
- Response capability: Full

**After Hours (Emergency Only)**:
- Pager system for critical incidents
- Single on-call engineer
- Response: Triage and escalation

---

## 3. Validated SLA Tiers

Based on capacity assessment, the following notification SLAs are validated:

### 3.1 Standard Tier (Available Now)

| Metric | Target | Notes |
|--------|--------|-------|
| Initial Notification (Critical) | 60 minutes | Target for business hours coverage |
| Initial Notification (High) | 120 minutes | Target for business hours coverage |
| Initial Notification (Medium) | 4 hours | Target for business hours coverage |
| Coverage Hours | 9am-6pm CET Mon-Fri | Target model (scaling in progress) |
| After-hours for Critical | Best effort (next morning) | Pager for escalation (when available) |

**Capacity Validation**:
- [x] 2+ engineers during business hours
- [x] Monitoring alerts configured
- [x] Escalation procedures documented
- [x] Client contact list maintained

### 3.2 Professional Tier (Requires Enhancement)

| Metric | Commitment | Requirement |
|--------|------------|-------------|
| Initial Notification (Critical) | 30 minutes | Extended hours coverage |
| Initial Notification (High) | 60 minutes | Extended hours coverage |
| Initial Notification (Medium) | 4 hours | Extended hours coverage |
| Coverage Hours | 7am-11pm CET Mon-Fri | On-call rotation |

**Prerequisites to Offer**:
- [ ] Establish on-call rotation (Option B)
- [ ] Minimum 3 engineers in rotation
- [ ] Pager duty compensation structure
- [ ] Runbook for all critical scenarios

**Status**: NOT AVAILABLE until prerequisites met

### 3.3 Enterprise Tier (Requires Significant Investment)

| Metric | Commitment | Requirement |
|--------|------------|-------------|
| Initial Notification (Critical) | 15 minutes | 24/7 coverage |
| Initial Notification (High) | 30 minutes | 24/7 coverage |
| Initial Notification (Medium) | 2 hours | 24/7 coverage |
| Coverage Hours | 24/7/365 | Dedicated team |

**Prerequisites to Offer**:
- [ ] 24/7 on-call rotation (Option C minimum)
- [ ] Minimum 4 engineers in rotation
- [ ] NOC capability or equivalent
- [ ] Automated escalation system
- [ ] Multi-region deployment

**Status**: NOT AVAILABLE until prerequisites met

---

## 4. Notification SLA Matrix

### 4.1 Current Achievable SLAs

| Incident Severity | Business Hours | After Hours | Weekend |
|-------------------|----------------|-------------|---------|
| **Critical** | 60 min | Best effort (pager) | Best effort (pager) |
| **High** | 120 min | Next business day | Next business day |
| **Medium** | 4 hours | Next business day | Next business day |
| **Low** | Next business day | Next business day | Next business day |

### 4.2 Target SLAs (After Enhancement)

| Incident Severity | With On-Call Rotation | With 24/7 Team |
|-------------------|----------------------|----------------|
| **Critical** | 30 min (any time) | 15 min (any time) |
| **High** | 60 min (extended) | 30 min (any time) |
| **Medium** | 4 hours (business) | 2 hours (any time) |
| **Low** | Next business day | 4 hours |

---

## 5. Capacity Gaps and Remediation

### 5.1 Identified Gaps

| Gap | Impact | Priority | Remediation |
|-----|--------|----------|-------------|
| No 24/7 coverage | Cannot offer Enterprise SLA | HIGH | Hire 2+ engineers |
| Single on-call after hours | Extended response time | MEDIUM | Establish rotation |
| No NOC | Manual monitoring only | MEDIUM | Consider managed NOC |
| Weekend coverage | Client exposure | MEDIUM | Weekend on-call rotation |

### 5.2 Remediation Roadmap

**Phase 1 (Current)**: Standard Tier Only
- Maintain current business hours coverage
- Pager for critical emergencies
- Clear documentation of limitations

**Phase 2 (Q2 2025)**: Professional Tier
- Establish on-call rotation
- Add 1 engineer to team
- Extended hours coverage

**Phase 3 (Q4 2025)**: Enterprise Tier
- 24/7 on-call rotation
- Add 2+ engineers
- Consider managed NOC partnership

---

## 6. Client Communication

### 6.1 Pre-Sales Disclosure

Sales team MUST disclose to prospective clients:
- Current coverage hours
- After-hours response limitations
- SLA tier availability

### 6.2 Contract Language

Standard contract includes:

```
NOTIFICATION SLA

Provider shall notify Client of ICT incidents as follows:

Standard Tier:
- Critical Incidents: Within 60 minutes (business hours)
- High Incidents: Within 120 minutes (business hours)
- Medium Incidents: Within 4 hours (business hours)
- Business Hours: Monday-Friday, 9:00-18:00 CET

After-hours incidents shall be addressed on a best-effort basis,
with formal notification by 10:00 CET the following business day.

[For Enhanced SLAs, see Professional/Enterprise Addendum]
```

### 6.3 Status Page

Public status page displays:
- Current system status
- Coverage hours
- Scheduled maintenance
- Incident history

---

## 7. Monitoring and Alerting

### 7.1 Alert Response Times

| Alert Type | Detection | Acknowledgment | Response Start |
|------------|-----------|----------------|----------------|
| P1 (Critical) | Automated | 5 min | 15 min |
| P2 (High) | Automated | 15 min | 30 min |
| P3 (Medium) | Automated | 30 min | 4 hours |
| P4 (Low) | Automated/Manual | 4 hours | Next business day |

### 7.2 Escalation Paths

```
INCIDENT DETECTED
       │
       ▼
PRIMARY ON-CALL (5 min)
       │
       ├─── No Response ───► SECONDARY ON-CALL (5 min)
       │                              │
       │                              ├─── No Response ───► MANAGEMENT
       │                              │
       │                              ▼
       │                         Acknowledged
       │
       ▼
 Acknowledged ───► TRIAGE ───► RESPONSE
```

---

## 8. Validation Sign-Off

### Engineering Validation

I confirm that the response time targets in this document are intended to reflect our planned operational capacity and are designed to be achievable with our existing team and infrastructure (subject to validation under actual operational conditions).

**Engineering Lead**: _______________________

**Date**: _______________________

### Operations Validation

I confirm that the notification procedures and escalation paths documented here are operational and tested.

**Operations Lead**: _______________________

**Date**: _______________________

---

## 9. Review Schedule

| Review Type | Frequency | Owner |
|-------------|-----------|-------|
| Capacity assessment | Quarterly | Engineering |
| SLA achievement review | Monthly | Operations |
| Escalation procedure test | Quarterly | Operations |
| Full validation | Annually | Engineering + Operations |

---

## 10. Related Documents

| Document | Purpose |
|----------|---------|
| [OPERATIONS_RUNBOOK.md](../OPERATIONS_RUNBOOK.md) | Operational procedures |
| [RECOVERY_PROCEDURES.md](../RECOVERY_PROCEDURES.md) | Incident recovery |
| [SLA Guardrails Module](../../services/dora_integration/contracts/sla_guardrails.py) | SLA approval workflow |
| [DORA Contract Template](../contracts/DORA_CONTRACT_TEMPLATE_ART_30_2.md) | SLA contract terms |

---

*This document is maintained by Engineering and Operations teams. Updates require re-validation.*
