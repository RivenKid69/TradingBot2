# Security Trust Center
## Pre-Contractual Security Overview

**Version**: 1.2
**Date**: 2025-12-18
**Status**: Public
**Legal Reference**: DORA Article 28(7)
**Canon Reference**: `docs/DOCUMENTATION_CANON_DESIGN.md`

---

## 1. Platform Security Overview

This document provides pre-contractual security information per DORA Article 28(7) requirements, enabling financial entities to conduct due diligence on our ICT services.

### 1.1 Executive Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Certifications** | SOC2 Type II Roadmap | Target 2027 (see Section 2) |
| **Infrastructure** | AWS (Frankfurt) | EU-only data residency |
| **Encryption** | AES-256 / TLS 1.3 | At-rest and in-transit |
| **Availability** | Target 99.9% | Multi-AZ deployment (design target) |
| **Incident Response** | Designed for 24/7 | Target 15-minute critical response |
| **DORA Alignment** | Designed to support | Art. 30 contract templates available |

---

## 2. Security Certifications & Attestations

### 2.1 Certification Roadmap

**Note**: CustodiaCloud is a pre-seed stage company. We do not claim certifications we have not yet achieved. This section describes our roadmap and current posture.

| Certification | Scope | Status | Target |
|---------------|-------|--------|--------|
| SOC2 Type I | Platform operations | Planned | 2026 |
| SOC2 Type II | Platform operations | Planned | 2027 |
| ISO 27001 | Information security | Planned | 2027+ |
| GDPR | Data protection | Designed to align | Ongoing |

### 2.2 Third-Party Audits (Planned)

**Current state**: As a pre-seed company, formal third-party audits are planned but not yet completed. We maintain internal security practices and will engage external auditors as part of our SOC2 roadmap.

| Audit Type | Planned Frequency | Status | Availability |
|------------|-------------------|--------|--------------|
| Penetration Testing | Annual | Planned 2026 | Summary on request (when available) |
| Vulnerability Assessment | Quarterly | Internal scans active | Summary on request |
| SOC2 Audit (Type I) | Annual | Planned 2026 | Report under NDA (when available) |
| SOC2 Audit (Type II) | Annual | Planned 2027 | Report under NDA (when available) |
| Code Security Review | Per major release | Internal review active | Summary on request |

---

## 3. Infrastructure Security

### 3.1 Cloud Infrastructure

| Component | Provider | Region | Security Features |
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

| Control | Implementation | Validation |
|---------|---------------|------------|
| DDoS Protection | Cloudflare + AWS Shield | Continuous |
| Web Application Firewall | Cloudflare + AWS WAF | Continuous |
| Intrusion Detection | AWS GuardDuty | Continuous |
| Vulnerability Scanning | Automated weekly | Weekly |
| Patch Management | Automated + manual review | Within 24h critical |

---

## 4. Data Protection

### 4.1 Encryption

| Data State | Method | Key Management |
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

### 6.1 Recovery Objectives

| Scenario | RTO | RPO | Last Tested |
|----------|-----|-----|-------------|
| Component Failure | 30 min | 0 (real-time) | Monthly |
| Availability Zone Failure | 1 hour | 15 min | Quarterly |
| Region Failure | 4 hours | 1 hour | Annually |
| Complete Disaster | 24 hours | 4 hours | Annually |

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
- **Regular Testing**: Quarterly DR tests with documented results
- **Runbooks**: Documented recovery procedures for all scenarios

---

## 7. Incident Response

### 7.1 Response Times

| Severity | Detection | Response | Client Notification |
|----------|-----------|----------|-------------------|
| Critical | 5 min | 15 min | 30 min |
| High | 15 min | 30 min | 1 hour |
| Medium | 1 hour | 4 hours | 4 hours |
| Low | 24 hours | Next business day | Weekly summary |

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

| Provider | Service | Criticality | Certifications |
|----------|---------|-------------|----------------|
| AWS | Cloud Infrastructure | Critical | SOC2, ISO27001, C5 |
| Cloudflare | CDN/Security | High | SOC2, ISO27001 |
| Datadog | Monitoring | Medium | SOC2, ISO27001 |
| Stripe | Payments | High | PCI DSS L1, SOC2 |

### 8.2 Due Diligence

All critical subcontractors undergo:
- Security assessment before engagement
- Annual review of certifications
- Incident reporting requirements
- Contractual security obligations

---

## 9. Compliance

### 9.1 Regulatory Alignment

| Regulation | Status | Relevance |
|------------|--------|-----------|
| DORA (EU) 2022/2554 | ✅ Designed to align | ICT provider obligations |
| GDPR (EU) 2016/679 | ✅ Designed to align | Data protection - all 9 phases |
| EU AI Act 2024/1689 | ✅ Designed to support alignment | Transparency/documentation (deployment-dependent; no self-classification) |
| MiFID II | ✅ Designed to support alignment | Client workflows (deployment-dependent) |
| NIS2 Directive | Preparing | Cybersecurity |

### 9.2 GDPR Compliance Details

| Article | Requirement | Status |
|---------|-------------|--------|
| Art. 5 | Data minimization, purpose limitation | ✓ Enforced |
| Art. 12-14 | Transparency (Privacy Policy, DPA) | ✓ Published |
| Art. 15-22 | Data subject rights (DSAR) | ✓ Full workflow |
| Art. 25 | Privacy by design | ✓ CCEA architecture |
| Art. 28 | Processor obligations | ✓ DPA template available |
| Art. 30 | Records of Processing (RoPA) | ✓ Maintained |
| Art. 32 | Security controls | ✓ Implemented |
| Art. 33-34 | Breach notification (72h) | ✓ Workflow ready |

**GDPR Key Features:**
- **EU-only data residency** (Frankfurt, Ireland)
- **Telemetry redaction** (mandatory, cannot be disabled)
- **DSAR response** within 30 days
- **Auto-purge** with configurable retention
- **Break-glass access** for incident-only, audited
- **Enterprise posture** (on-prem/VPC with "telemetry stays local" option)

### 9.3 DORA Readiness

| Article | Requirement | Status |
|---------|-------------|--------|
| Art. 28 | ICT third-party risk management | ✓ Ready |
| Art. 30 | Key contractual provisions | ✓ Templates available |
| Art. 30(2) | Basic contract terms | ✓ Included |
| Art. 30(3) | Critical function addendum | ✓ Available |
| Art. 30(3)(e) | Audit rights | ✓ Supported |
| Art. 30(3)(f) | Exit strategies | ✓ Documented |

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
- Urgent: [24/7 phone number]
- Response: Per severity (see Section 7)

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

**Review Frequency**: Quarterly
**Owner**: Security Team
**Classification**: Public

**Important Note**: This document has been corrected to remove inaccurate claims about certifications and audits that had not yet been completed. CustodiaCloud is committed to transparency in our security posture and will not claim certifications or audits we have not achieved.

---

*This document is provided for informational purposes to support pre-contractual due diligence. Detailed security information is available to contracted clients under NDA.*
