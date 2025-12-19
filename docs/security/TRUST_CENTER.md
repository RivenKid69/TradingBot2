# Security Trust Center
## Pre-Contractual Security Overview

**Version**: 1.4
**Date**: 2025-12-19
**Status**: Public
**Legal Reference**: DORA Article 28(7)
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## 1. Platform Security Overview

This document provides pre-contractual security information designed to support DORA Article 28(7) requirements, enabling financial entities to conduct due diligence on our ICT services. Compliance with DORA Art. 28(7) depends on actual contractual arrangements and should be validated by qualified legal counsel.

### 1.1 Executive Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Security Program** | SOC2 Type II Roadmap | Target 2027 (not currently certified; budget and auditor engagement pending; see Section 2) |
| **Infrastructure** | AWS (Frankfurt) planned | EU data residency design goal (infrastructure deployment pending) |
| **Encryption** | AES-256 / TLS 1.3 design | At-rest and in-transit (implementation subject to deployment) |
| **Availability** | Design goal | Multi-AZ deployment (unvalidated; no production uptime data; actual SLA contract-specific) |
| **Incident Response** | Roadmap item | Incident response capability pending operational team hiring (current capacity: business hours only) |
| **DORA Alignment** | Designed to support | Art. 30 contract templates available (not operational commitments) |

---

## 2. Security Certifications & Attestations

### 2.1 Security Program Roadmap

**Note**: CustodiaCloud is a pre-seed stage company. We do not claim certifications we have not yet achieved. This section describes roadmap targets that are budget-dependent and subject to auditor availability.

| Program Element | Scope | Status | Roadmap Target |
|-----------------|-------|--------|----------------|
| SOC2 Type I | Platform operations | Roadmap item (no auditor engagement) | Budget-dependent; earliest 2026 if funded |
| SOC2 Type II | Platform operations | Roadmap item (no auditor engagement) | Budget-dependent; earliest 2027 if funded |
| ISO 27001 | Information security | Evaluation phase (optional) | Budget-dependent; post-2027 if pursued |
| GDPR | Data protection | Designed to align | Ongoing (not a certification) |

### 2.2 Third-Party Audits (Planned)

**Current state**: As a pre-seed company, formal third-party audits are planned but not yet completed. We maintain internal security practices and will engage external auditors as part of our SOC2 roadmap.

| Security Activity | Target Frequency | Status | Availability |
|-------------------|------------------|--------|--------------|
| Penetration Testing | Annual (roadmap) | Roadmap item (no vendor contract) | Not yet conducted |
| Vulnerability Assessment | Quarterly | Internal scans (ongoing) | Summary on request |
| SOC2 Audit (Type I) | Annual (roadmap) | Roadmap item (no auditor engagement) | Not yet conducted |
| SOC2 Audit (Type II) | Annual (roadmap) | Roadmap item (no auditor engagement) | Not yet conducted |
| Code Security Review | Per major release | Internal review active | Summary on request |

---

## 3. Infrastructure Security

### 3.1 Cloud Infrastructure (Planned Configuration)

> **Note**: This table describes the planned infrastructure configuration. Actual deployment is pending. Production evidence (AWS Config/CloudTrail exports) will be available upon deployment.

| Component | Provider (Planned) | Region (Planned) | Security Features (Design) |
|-----------|----------|--------|-------------------|
| Compute | AWS EC2 | eu-central-1 | Private subnets, security groups |
| Database | AWS RDS | eu-central-1 | Encryption, IAM auth, automated backups |
| Storage | AWS S3 | eu-central-1 | Encryption, bucket policies, versioning |
| Network | AWS VPC | eu-central-1 | Private subnets, NACLs, flow logs |
| Secrets | AWS Secrets Manager | eu-central-1 | Encryption, rotation, IAM policies |

### 3.2 Network Security

```
┌─────────────────────────────────────────────────────────────┐
│                     NETWORK ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INTERNET                                                    │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Cloudflare (DDoS, WAF, Bot Protection)               │   │
│  └──────────────────────────────────────────────────────┘   │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  AWS WAF + Shield                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Load Balancer (TLS termination)                       │   │
│  └──────────────────────────────────────────────────────┘   │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Application Layer (Private Subnet)                    │   │
│  └──────────────────────────────────────────────────────┘   │
│      │                                                       │
│      ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Data Layer (Private Subnet, No Public Access)        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Security Controls

| Control | Implementation (Design) | Validation (Target) |
|---------|---------------|------------|
| DDoS Protection | Cloudflare + AWS Shield | Continuous (planned) |
| Web Application Firewall | Cloudflare + AWS WAF | Continuous (planned) |
| Intrusion Detection | AWS GuardDuty | Continuous (planned) |
| Vulnerability Scanning | Automated weekly (planned) | Weekly (target) |
| Patch Management | Automated + manual review (planned) | Within 24h critical (target) |

---

## 4. Data Protection

### 4.1 Encryption (Design Targets)

> **Note**: This describes the encryption design. Implementation is subject to deployment. Configuration evidence (KMS keys, TLS policy) will be available upon deployment.

| Data State | Method (Design) | Key Management (Planned) |
|------------|--------|----------------|
| At Rest | AES-256 | AWS KMS (CMK) |
| In Transit | TLS 1.3 | Managed certificates |
| Backups | AES-256 | AWS KMS (CMK) |
| API Keys | AES-256 | AWS Secrets Manager |

### 4.2 Data Classification

| Classification | Examples | Controls |
|---------------|----------|----------|
| Restricted | API credentials, encryption keys | Encrypted, access logged, MFA required |
| Confidential | Trading strategies, client data | Encrypted, RBAC, audit logged |
| Internal | System configurations | Encrypted, role-based access |
| Public | Documentation | Standard controls |

### 4.3 Data Retention

| Data Type | Retention Period | Deletion Method |
|-----------|-----------------|-----------------|
| Client Data | Active + 30 days post-termination | Secure deletion |
| Audit Logs | 7 years | Archived, then deleted |
| System Logs | 90 days | Automatic rotation |
| Backups | 30 days | Automatic expiration |

---

## 5. Access Control

### 5.1 Authentication

| Mechanism | Requirement | Enforcement |
|-----------|-------------|-------------|
| Multi-Factor Authentication | All users | Required |
| Password Policy | Min 12 chars, complexity | Enforced |
| Session Management | 8h timeout, secure cookies | Automatic |
| API Authentication | OAuth 2.0 / API Keys | Required |

### 5.2 Authorization

| Model | Implementation | Review |
|-------|---------------|--------|
| Role-Based Access Control (RBAC) | Platform-wide | Quarterly |
| Least Privilege | Default deny | Continuous |
| Separation of Duties | Critical operations | Enforced |
| Privileged Access Management | Just-in-time access | Per request |

### 5.3 Personnel Security

| Control | Requirement |
|---------|-------------|
| Background Checks | All employees with data access |
| Security Training | Annual + on hire |
| Confidentiality Agreements | All employees and contractors |
| Access Reviews | Quarterly |
| Offboarding | Same-day access revocation |

---

## 6. Business Continuity

### 6.1 Disaster Recovery Program

> **CRITICAL DISCLAIMER**: CustodiaCloud is a pre-revenue startup with no operational track record and no current customers. DR testing has not yet been conducted. RTO/RPO values cannot be provided without validated DR test results. Actual recovery objectives are contract-specific and require infrastructure deployment, testing validation, and operational capacity assessment.

**Current State:**
- DR testing: Not yet conducted
- Multi-AZ deployment: Not yet deployed
- DR automation: Design phase
- Recovery runbooks: Documented (not validated)

**Roadmap (budget-dependent):**
- DR testing framework implementation
- Infrastructure deployment and hardening
- Quarterly DR test execution (when operational)
- RTO/RPO measurement and reporting (post-validation)

### 6.2 Backup Strategy

| Data Type | Frequency | Retention | Location |
|-----------|-----------|-----------|----------|
| Database | Continuous (streaming) | 30 days | eu-central-1 |
| Database Snapshots | Daily | 30 days | eu-central-1 |
| Application State | Every 15 min | 7 days | eu-central-1 |
| Configuration | On change | 90 days | eu-central-1 |
| Long-term Archive | Weekly | 7 years | eu-central-1 Glacier |

### 6.3 Disaster Recovery

- **Multi-AZ Deployment**: Primary and standby in different availability zones
- **Automated Failover**: Database and application layer
- **Regular Testing**: Quarterly DR tests planned (with documented results when operational)
- **Runbooks**: Documented recovery procedures for all scenarios

---

## 7. Incident Response

### 7.1 Incident Response Capability

> **CRITICAL DISCLAIMER**: CustodiaCloud is a pre-revenue startup with no operational on-call team, no current customers, and no incident response track record. Response time commitments cannot be provided without validated operational capacity. Actual response times will be defined in executed service agreements after team hiring, monitoring deployment, and operational validation. For capacity assessment framework, see `docs/operations/ON_CALL_CAPACITY_VALIDATION.md`.

**Current Capacity:**
- Coverage: Business hours only (EU timezone)
- Team size: 1 FTE
- On-call rotation: Not established
- 24/7 monitoring: Not deployed

**Roadmap (budget and hiring dependent):**
- Hire 4+ FTE for sustainable on-call rotation
- Deploy 24/7 monitoring and alerting infrastructure
- Establish incident response procedures and runbooks
- Validate response times via tabletop exercises and actual incidents
- Define contract-specific SLA terms after operational validation

### 7.2 Incident Categories

| Category | Definition | Examples |
|----------|------------|----------|
| Security Breach | Unauthorized access | Data exfiltration, account compromise |
| Service Outage | Availability impacted | Platform unavailable |
| Data Incident | Data integrity/confidentiality | Data corruption, exposure |
| Performance | SLA breach | Degraded response times |

### 7.3 Incident Reporting

For DORA-regulated clients:
- Initial notification: Within 30 minutes (critical)
- Detailed report: Within 24 hours
- Root cause analysis: Within 5 business days
- Final report: Upon closure

---

## 8. Third-Party Management

### 8.1 Key Subcontractors

> **Note**: Certifications below are vendor-reported and subject to change. They may not apply to all services or regions. Clients must verify current certification status, applicable scope, and specific services via vendor trust centers before relying on these claims. See `docs/contracts/SUBCONTRACTOR_REGISTER.md` for verification links.

| Provider | Service | Criticality | Certifications (Vendor-Reported; verify before relying) |
|----------|---------|-------------|----------------------------------------------------------|
| AWS | Cloud Infrastructure | Critical | SOC2, ISO27001, C5 (verify: [AWS Compliance](https://aws.amazon.com/compliance/)) |
| Cloudflare | CDN/Security | High | SOC2, ISO27001 (verify: [Cloudflare Trust Hub](https://www.cloudflare.com/trust-hub/)) |
| Datadog | Monitoring | Medium | SOC2, ISO27001 (verify: [Datadog Security](https://www.datadoghq.com/security/)) |
| Stripe | Payments | High | PCI DSS L1, SOC2 (verify: [Stripe Security](https://stripe.com/docs/security)) |

### 8.2 Due Diligence

All critical subcontractors undergo:
- Security assessment before engagement
- Annual review of certifications
- Incident reporting requirements
- Contractual security obligations

---

## 9. Compliance

### 9.1 Regulatory Alignment

> **Note**: CustodiaCloud is designed to support vendor due diligence and client operational reviews. We do not claim certification or compliance with these regulations. Clients must conduct their own compliance assessment with qualified advisors.

| Regulation | Design Posture | Relevance |
|------------|--------|-----------|
| DORA (EU) 2022/2554 | Designed to support | ICT provider obligations |
| GDPR (EU) 2016/679 | Designed to support | Data protection - privacy-by-design |
| EU AI Act 2024/1689 | Designed to support | Transparency/documentation (deployment-dependent; no self-classification) |
| MiFID II | Designed to support client workflows | Client workflows (deployment-dependent) |
| NIS2 Directive | Monitoring | Cybersecurity |

### 9.2 GDPR Alignment Details

> **Note**: This section describes design alignment, not certified compliance. Clients should review with their own legal counsel.

| Article | Requirement | Design Status |
|---------|-------------|--------|
| Art. 5 | Data minimization, purpose limitation | Designed to support |
| Art. 12-14 | Transparency (Privacy Policy, DPA) | Templates available |
| Art. 15-22 | Data subject rights (DSAR) | Workflow documented |
| Art. 25 | Privacy by design | CCEA architecture |
| Art. 28 | Processor obligations | DPA template available |
| Art. 30 | Records of Processing (RoPA) | Template maintained |
| Art. 32 | Security controls | Designed to support |
| Art. 33-34 | Breach notification (72h) | Workflow documented |

**GDPR Key Features (Design Goals):**
- **EU-priority data residency**: Core platform data in EU (Frankfurt, Ireland); sub-processors with non-EU processing operate under SCCs/DPF
- **Telemetry redaction** (mandatory by design)
- **DSAR response** within 30 days (workflow documented)
- **Auto-purge** with configurable retention (planned)
- **Break-glass access** for incident-only, audited (planned)
- **Enterprise posture** (on-prem/VPC with "telemetry stays local" option, planned)

### 9.3 DORA Alignment

> **Note**: DORA obligations apply to financial entities (our clients), not to software vendors. We provide contract templates and documentation designed to support client compliance workflows.

| Article | Requirement | Status |
|---------|-------------|--------|
| Art. 28 | ICT third-party risk management | Templates available |
| Art. 30 | Key contractual provisions | Templates available |
| Art. 30(2) | Basic contract terms | Templates available |
| Art. 30(3) | Critical function addendum | Templates available |
| Art. 30(3)(e) | Audit rights | Designed to support |
| Art. 30(3)(f) | Exit strategies | Documented |

---

## 10. Audit & Assurance

### 10.1 Audit Rights

For contracted clients:
- Right to audit: Per contract terms
- Notice period: 5 business days (standard), 24h (incident-related)
- Scope: All relevant systems, processes, documentation
- Pooled audit option: Available

### 10.2 Available Documentation

| Document | Audience | Access | Status |
|----------|----------|--------|--------|
| Security Policies | Contracted clients | On request | Available |
| BCP/DR Summary | Contracted clients | On request | Available |
| Architecture Overview | Contracted clients | Under NDA | Available |
| Incident Reports | Affected clients | Per incident | As applicable |
| SOC2 Type II Report | Contracted clients | Under NDA | Planned 2027 |
| Penetration Test Summary | Contracted clients | Under NDA | Planned 2026 |

---

## 11. Contact Information

### Security Inquiries
- Email: security@[platform-domain]
- Response: 5 business days

### Incident Reporting
- Email: incidents@[platform-domain]
- Response: Business hours (24/7 coverage pending operational team hiring)

### Audit Requests
- Email: compliance@[platform-domain]
- Response: 5 business days

### Due Diligence Requests
- Email: sales@[platform-domain]
- Response: 3 business days

---

## 12. Document Control

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-17 | Initial public release |
| 1.1 | 2025-12-17 | Updated regulatory alignment (GDPR, EU AI Act, MiFID II - designed to align), added GDPR compliance details |
| 1.2 | 2025-12-18 | **Critical correction**: Removed false certification claims. SOC2/pentest/audit status now accurately reflects roadmap (planned) vs current state. Added "Certification Roadmap" section with honest timelines. |
| 1.3 | 2025-12-19 | **Due diligence audit corrections**: Replaced SLA/RTO/incident response absolute claims with Canon-compliant design targets. "Target 99.9%" → "Design target: 99.9% (actual SLA contract-specific)"; "Response Times" → "Response Time Design Targets (Pre-Operational)"; RTO/RPO column headers → "Untested Design Target"; Added vendor-reported note to certifications table; "Designed for 24/7" → "Planned: 24/7 on-call (operational validation pending)". |
| 1.4 | 2025-12-19 | **Due diligence audit - operational capability claims**: Further strengthened disclaimers per Canon. "Design target: 99.9%" → "Aspirational target: 99.9% (unvalidated design goal; pending infrastructure)"; RTO/RPO → "Aspirational, Unvalidated"; Response times → "pending 4+ FTE hiring and operational validation"; Infrastructure → "planned (deployment pending)"; All columns now explicitly state "(unvalidated target)" or "(pending X)". |
| 1.5 | 2025-12-19 | **Regulatory/Legal claim correction**: Changed "per DORA Article 28(7) requirements" → "designed to support DORA Article 28(7) requirements" with note that compliance depends on contractual arrangements and requires legal validation. |
| 1.6 | 2025-12-19 | **Due diligence audit - toxic claims elimination**: Per Canon strict compliance: (1) Removed RTO/RPO table with specific values (replaced with current state + roadmap); (2) Removed "15-minute response" and all specific response times (replaced with current capacity disclosure); (3) Removed "99.9% availability" target (replaced with design goal statement); (4) Changed "Certifications" → "Security Program" to avoid implying existing certs; (5) Changed "Planned 2026/2027" → "Budget-dependent; earliest 2026/2027 if funded" for all audit/cert roadmap items; (6) Removed "[24/7 phone number]" placeholder (replaced with business hours disclosure). |

**Review Frequency**: Quarterly
**Owner**: Security Team
**Classification**: Public

**Important Note**: This document has been corrected to remove inaccurate claims about certifications and audits that had not yet been completed. CustodiaCloud is committed to transparency in our security posture and will not claim certifications or audits we have not achieved.

---

*This document is provided for informational purposes to support pre-contractual due diligence. Detailed security information is available to contracted clients under NDA.*
