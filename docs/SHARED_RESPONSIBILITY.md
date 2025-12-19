# DORA Shared Responsibility Model

**Version**: 1.0
**Date**: 2025-01-17
**Status**: Active
**Regulation**: DORA (EU) 2022/2554

---

## 1. Overview

This document describes the intended shared responsibility model between our platform (designed to support ICT Third-Party Provider obligations under DORA) and EU financial entity clients (Financial Entities under DORA). It clarifies how obligations may be allocated to support DORA compliance alignment through the contractual relationship.

> **Note**: Regulatory classification (ICT Third-Party Provider vs. other categories) depends on the actual services provided and contractual arrangements. This document describes a design intent; clients should validate regulatory applicability with qualified legal counsel.

### Our Position in DORA Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                     DORA RESPONSIBILITY MODEL                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   FINANCIAL ENTITY (Client)        ICT PROVIDER (Us)            │
│   ┌─────────────────────────┐     ┌─────────────────────────┐   │
│   │ Direct DORA obligations │     │ Contractual obligations │   │
│   │                         │     │                         │   │
│   │ • NCA reporting         │     │ • Art. 28 compliance    │   │
│   │ • TLPT coordination     │     │ • Art. 30 contract terms│   │
│   │ • ROI submission        │     │ • Audit cooperation     │   │
│   │ • Internal governance   │     │ • Incident support      │   │
│   │ • Risk management       │     │ • Exit facilitation     │   │
│   └─────────────────────────┘     └─────────────────────────┘   │
│              ▲                              ▲                    │
│              │        CONTRACT              │                    │
│              └──────────────────────────────┘                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Responsibility Matrix

### 2.1 ICT Risk Management (Articles 5-16)

| Requirement | Client Responsibility | Provider Responsibility |
|-------------|----------------------|------------------------|
| ICT risk management framework | Full ownership and implementation | Support with platform documentation |
| ICT governance | Board-level oversight | Provide governance evidence for audits |
| ICT policies and procedures | Define and maintain | Comply with client policies where applicable |
| ICT asset identification | Maintain asset register | Provide platform asset information |
| Data classification | Define classification scheme | Apply client classification to their data |
| Protection measures | Overall security strategy | Platform security implementation |
| Detection capabilities | SOC/monitoring strategy | Platform monitoring and alerting |
| ICT continuity | BCP/DR strategy | Platform BCP/DR implementation |
| Learning and evolving | Incorporate lessons learned | Share relevant threat intelligence |

### 2.2 ICT Incident Management (Articles 17-23)

| Requirement | Client Responsibility | Provider Responsibility |
|-------------|----------------------|------------------------|
| Incident management process | Define and maintain | Follow agreed incident procedures |
| Incident detection | Detect own environment issues | Detect and report platform incidents |
| Incident classification | Classify per DORA criteria | Provide classification inputs |
| Incident response | Coordinate overall response | Execute platform-side response |
| NCA notification | Decide and submit notifications | Provide incident data for notifications |
| Cyber threat notifications | Submit to NCA if applicable | Share relevant threat information |
| Post-incident analysis | Own root cause analysis | Provide RCA for platform components |

**Incident Notification Timeline:**

```
INCIDENT OCCURS
      │
      ▼ (Detection)
  ┌───────┐
  │PROVIDER│ ──► Detect within monitoring SLA
  └───────┘
      │
      ▼ (Initial Alert: 30 min for Critical)
  ┌───────┐
  │ CLIENT │ ◄── Provider notifies Client
  └───────┘
      │
      ▼ (Classification)
  ┌───────┐
  │ CLIENT │ ──► Client classifies per DORA Art. 18
  └───────┘
      │
      ▼ (NCA Notification: 4h initial / 24h if major)
  ┌───────┐
  │ CLIENT │ ──► Client notifies NCA if required
  └───────┘
```

### 2.3 Digital Operational Resilience Testing (Articles 24-27)

| Requirement | Client Responsibility | Provider Responsibility |
|-------------|----------------------|------------------------|
| Testing programme | Design and maintain | Support and participate |
| Vulnerability assessments | Conduct for own systems | Conduct for platform, share results |
| Scenario-based testing | Conduct and document | Participate in client scenarios |
| Penetration testing | Coordinate and execute | Cooperate with testing |
| TLPT (if designated) | Coordinate with NCA | Full cooperation with testers |
| Test remediation | Prioritize and track | Remediate platform findings |

### 2.4 Third-Party ICT Risk (Articles 28-30)

| Requirement | Client Responsibility | Provider Responsibility |
|-------------|----------------------|------------------------|
| Due diligence | Conduct pre-contract due diligence | Provide required information |
| Contract requirements | Include Art. 30 provisions | Accept and comply with provisions |
| Register of Information | Maintain ROI, submit to NCA | Provide data for Client's ROI |
| Concentration risk | Monitor and manage | Report client mix for CTPP awareness |
| Exit strategy | Maintain exit capability | Provide exit support per contract |
| Audit rights | Exercise audit rights | Enable and support audits |
| Subcontractor oversight | Include in risk assessment | Document and notify of subcontractors |

### 2.5 Information Sharing (Article 45)

| Requirement | Client Responsibility | Provider Responsibility |
|-------------|----------------------|------------------------|
| Sharing arrangements | Participate in sharing communities | Share relevant threat intelligence |
| Threat information | Share with community | Alert clients to relevant threats |
| Information protection | Protect shared information | Sanitize and protect shared data |

---

## 3. Provider Services and Responsibilities

### 3.1 What We Provide (Design Commitments)

**Note**: CustodiaCloud is a pre-seed company. These are design commitments and roadmap targets intended to become contractual terms upon customer contract execution. Actual SLA targets and support coverage will be defined in executed service agreements after operational validation.

| Category | Description | Evidence |
|----------|-------------|----------|
| **Platform Security** | SOC2 Type II readiness roadmap, encryption, access controls | SOC2 report (when available, per roadmap) |
| **Availability** | Design target: 99.9% availability (actual SLA contract-specific) | Monthly SLA reporting (when operational) |
| **Incident Support** | Incident response design (contract tier-dependent when operational) | Incident reporting workflow |
| **Audit Support** | Client and NCA audit cooperation | Audit records |
| **Data Portability** | Standard export formats, no lock-in | Export functionality |
| **Subcontractor Transparency** | Full subcontractor documentation | Subcontractor register |
| **Exit Facilitation** | 90-180 day transition support | Exit plan documentation |

### 3.2 What We Do NOT Provide

| Category | Reason | Client Action |
|----------|--------|---------------|
| NCA regulatory reporting | Not a financial entity | Client submits all NCA reports |
| TLPT coordination | Client obligation | Client coordinates, we cooperate |
| ROI submission | Client obligation | We provide data, Client submits |
| Internal DORA governance | Client internal matter | Client implements own framework |
| Investment advice | Out of scope | Client makes investment decisions |
| Trade execution (unless authorized) | Client authorization required | Explicit authorization needed |

---

## 4. Contractual Flow-Down Requirements

### 4.1 Required Contract Provisions

Standard contract templates for EU financial entity clients are designed to include:

**Article 30(2) - All Contracts (template provisions):**
- [ ] Clear service description
- [ ] Data processing/storage locations
- [ ] Availability, authenticity, integrity, confidentiality provisions
- [ ] Data access, recovery, return procedures
- [ ] Service level descriptions
- [ ] Incident assistance obligations
- [ ] Authority cooperation commitment
- [ ] Termination rights and notice periods
- [ ] Training participation commitment

**Article 30(3) - Critical Functions (Addendum, if applicable):**
- [ ] Full SLA with quantitative targets
- [ ] Extended notice periods
- [ ] Enhanced reporting obligations
- [ ] Business contingency plan requirements
- [ ] Resilience testing participation
- [ ] Unrestricted audit rights
- [ ] Exit strategy requirements
- [ ] Supervisory cooperation
- [ ] Business continuity implementation
- [ ] Security arrangements
- [ ] Subcontracting approval conditions

### 4.2 Subcontractor Requirements

Our subcontractor contracts are designed to require:
- Meeting equivalent security standards
- Allowing audit rights flow-through
- Complying with data location restrictions
- Supporting incident response procedures
- Maintaining required certifications (vendor-reported; verify via vendor trust centers)

---

## 5. Client Onboarding Requirements

### 5.1 Information We Need from Clients

| Information | Purpose | When Required |
|-------------|---------|---------------|
| Legal Entity Identifier (LEI) | ROI data provision | Onboarding |
| NCA identification | Regulatory coordination | Onboarding |
| Critical function designation | Contract tier determination | Onboarding |
| Data classification scheme | Apply to Client data | Onboarding |
| Incident notification contacts | Escalation paths | Onboarding |
| Audit contact | Audit coordination | Onboarding |
| Exit plan contact | Exit coordination | Onboarding |

### 5.2 Information We Provide to Clients

| Information | Purpose | Delivery |
|-------------|---------|----------|
| Provider Information Package | Client's ROI submission | On request / portal |
| Subcontractor Register | Third-party risk assessment | On request / portal |
| SOC2 Report | Due diligence | On request under NDA (if/when obtained; roadmap) |
| SLA Reports | Ongoing monitoring | Periodic (cadence defined per contract) |
| Incident Reports | Regulatory compliance | Per incident |
| Exit Plan Summary | Exit planning | On request |

---

## 6. Operational Interfaces

### 6.1 Incident Communication

```
┌─────────────────────────────────────────────────────────────┐
│                 INCIDENT COMMUNICATION FLOW                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  PROVIDER                              CLIENT                │
│  ┌──────────────┐                     ┌──────────────┐      │
│  │ Detection &  │ ──── Alert ────►    │ Classification│      │
│  │ Triage       │     (30 min)        │ & Assessment │      │
│  └──────────────┘                     └──────────────┘      │
│         │                                    │               │
│         ▼                                    ▼               │
│  ┌──────────────┐                     ┌──────────────┐      │
│  │ Containment  │ ◄── Coordination ── │ NCA Decision │      │
│  │ & Response   │                     │              │      │
│  └──────────────┘                     └──────────────┘      │
│         │                                    │               │
│         ▼                                    ▼               │
│  ┌──────────────┐                     ┌──────────────┐      │
│  │ RCA & Report │ ──── Report ────►   │ NCA Report   │      │
│  │              │     (24 hours)      │ Submission   │      │
│  └──────────────┘                     └──────────────┘      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Audit Coordination

| Audit Type | Lead | Provider Role | Timeline |
|------------|------|---------------|----------|
| Client operational audit | Client | Full cooperation | Per client schedule |
| Client security audit | Client | Evidence provision | Per client schedule |
| NCA inspection | Client/NCA | Cooperation | Per NCA schedule |
| SOC2 audit | Provider | Coordination | Annual |
| Pooled audit | Provider | Coordination | Annual |

### 6.3 Exit Coordination

```
EXIT TIMELINE (Critical Functions)
│
├─ Day 0: Termination notice received
│
├─ Days 1-5: Exit plan activation, data export initiated
│
├─ Days 5-30: Knowledge transfer sessions
│
├─ Days 30-90: Parallel running support
│
├─ Days 90-150: Gradual transition
│
├─ Days 150-180: Final data verification
│
└─ Day 180+: Read-only access period (30 days)
```

---

## 7. Escalation Procedures

### 7.1 Operational Escalation

| Level | Trigger | Provider Contact | Client Contact |
|-------|---------|------------------|----------------|
| L1 | Standard issues | Support team | Operations |
| L2 | SLA breach warning | Support manager | Operations lead |
| L3 | Critical incident | Platform lead | CISO/CTO |
| L4 | Major outage/breach | Executive | Executive |

### 7.2 Regulatory Escalation

| Trigger | Provider Action | Client Action |
|---------|-----------------|---------------|
| NCA contact | Notify Client within 24h | Coordinate response |
| Regulatory investigation | Full cooperation | Lead investigation |
| Enforcement action | Notify Client immediately | Assess impact |
| CTPP designation inquiry | Notify Client immediately | Assess implications |

---

## 8. Document References

| Document | Purpose | Location |
|----------|---------|----------|
| DORA Contract Template (Art. 30(2)) | Base contract provisions | docs/contracts/ |
| Critical Function Addendum (Art. 30(3)) | Critical function provisions | docs/contracts/ |
| Subcontractor Register | Subcontractor disclosure | docs/contracts/ |
| Provider Information Package | ROI data provision | Client portal |
| Operations Runbook | Operational procedures | docs/ |
| Recovery Procedures | DR/BCP documentation | docs/ |
| Service Dependency Map | Architecture documentation | docs/ |

---

## 9. Review and Updates

- **Review Frequency**: Quarterly
- **Owner**: Compliance Team
- **Last Review**: 2025-01-17
- **Next Review**: 2025-04-17

---

## Appendix A: DORA Article Reference

| Article | Title | Relevance |
|---------|-------|-----------|
| Art. 28 | General principles on ICT third-party risk | Core framework |
| Art. 29 | Preliminary assessment of ICT concentration risk | Client obligation |
| Art. 30 | Key contractual provisions | Contract requirements |
| Art. 31-44 | Critical third-party providers | CTPP regime |
| Art. 45 | Information-sharing arrangements | Voluntary sharing |

---

*This document reflects the shared responsibility model as of DORA application date (17 January 2025).*
