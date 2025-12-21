# DORA Subcontractor Register
## ICT Third-Party Service Provider Disclosure

**Version**: 1.1
**Date**: 2025-12-21
**Status**: Active
**Legal Reference**: DORA Article 30(2)(a), CIR 2024/2956 Template B_99.01

---

## 1. Purpose

This register documents all subcontractors engaged by the Platform for ICT services that may process, store, or have access to Client data. This disclosure supports Client's:
- Third-party risk management obligations (DORA Art. 28)
- Register of Information requirements (DORA Art. 28(3))
- Concentration risk assessment (DORA Art. 29)

> **Note on Vendor Certifications**: All subcontractor certifications listed in this register are vendor-reported and should be independently verified via the vendor's trust center or compliance portal. Links to vendor trust centers are provided where available. Clients should conduct their own due diligence and request current SOC2 reports or certification evidence directly from vendors as needed for their compliance requirements.

---

## 2. Subcontractor Summary

| Subcontractor | Type | Services | Data Access | Criticality | EU Data Only |
|---------------|------|----------|-------------|-------------|--------------|
| AWS (Amazon Web Services) | Cloud Infrastructure | Compute, Storage, Database | Full | CRITICAL | Yes (Frankfurt) |
| Cloudflare | Network Security | CDN, DDoS Protection, WAF | Limited | HIGH | Yes (EU PoPs) |
| Datadog | Monitoring | APM, Logging, Metrics | Metadata | MEDIUM | Configurable |
| Sentry | Error Tracking | Error Monitoring | Error Data | MEDIUM | EU (Frankfurt) |
| SendGrid (Twilio) | Communication | Email Delivery | Email Addresses | LOW | US (with DPA + SCCs; note: non-EU processing—verify transfer impact assessment) |
| Stripe | Payment | Payment Processing | Payment Data | HIGH | EU/US |

---

## 3. Detailed Subcontractor Information

### 3.1 Amazon Web Services (AWS)

**Provider Identification (B_02.01):**
| Field | Value |
|-------|-------|
| Legal Name | Amazon Web Services EMEA SARL |
| LEI | 549300D4RVMJXRXUJN54 |
| Registration Country | Luxembourg (LU) |
| Parent Company | Amazon.com, Inc. (US) |
| Headquarters | 38 Avenue John F. Kennedy, L-1855 Luxembourg |

**Services Provided (B_03.01):**
| Service Code | Service Description | Our Usage |
|--------------|---------------------|-----------|
| CS | Cloud Computing Services | EC2, Lambda, ECS |
| DC | Data Centre Services | Primary infrastructure |
| NI | Network Infrastructure | VPC, Load Balancing, Route 53 |

**Data Processing Details (B_05/B_06):**
| Attribute | Value |
|-----------|-------|
| Data Types Processed | Client trading data, configurations, models, logs |
| Processing Locations | eu-central-1 (Frankfurt, DE) |
| Storage Locations | eu-central-1 (Frankfurt, DE) |
| Encryption | AES-256 at rest, TLS 1.3 in transit |
| Backup Locations | eu-central-1 (Frankfurt, DE) |

**Contractual Provisions (per vendor terms; verify current status):**
| Requirement | Target | Evidence |
|-------------|--------|----------|
| DORA Art. 30 equivalent terms | Included (verify with AWS) | AWS DORA Addendum (if available) |
| Audit rights | Via SOC2/ISO27001 (vendor-reported) | AWS Artifact |
| Data location restrictions | EU-only configured (verify in AWS Config) | AWS Config |
| Subcontracting notification | Via AWS notifications | AWS compliance updates |
| Exit support | S3 data export | AWS documentation |

**Certifications** *(vendor-reported; verify via vendor trust centers)*:
- SOC 1, 2, 3 Type II (vendor-reported; verify via [AWS Artifact](https://aws.amazon.com/artifact/))
- ISO 27001, 27017, 27018 (vendor-reported; verify via AWS Compliance)
- C5 (Germany) (vendor-reported)
- PCI DSS Level 1 (vendor-reported)

**Last Verification**: Certification status should be verified quarterly via AWS Artifact portal and [AWS Compliance](https://aws.amazon.com/compliance/) page.

**Criticality Assessment:**
| Factor | Assessment |
|--------|------------|
| Criticality Level | CRITICAL |
| Substitutability | DIFFICULT (major migration effort) |
| Dependency | All production infrastructure |
| Risk Level | HIGH (mitigated by certifications) |

---

### 3.2 Cloudflare, Inc.

**Provider Identification (B_02.01):**
| Field | Value |
|-------|-------|
| Legal Name | Cloudflare, Inc. |
| LEI | 5493007DY18BGNLDWU14 |
| Registration Country | United States (US) |
| EU Representative | Cloudflare Germany GmbH |
| Headquarters | 101 Townsend Street, San Francisco, CA 94107 |

**Services Provided (B_03.01):**
| Service Code | Service Description | Our Usage |
|--------------|---------------------|-----------|
| IS | ICT Security Services | WAF, DDoS protection |
| NI | Network Infrastructure | CDN, DNS |

**Data Processing Details:**
| Attribute | Value |
|-----------|-------|
| Data Types Processed | HTTP headers, IP addresses (traffic metadata) |
| Processing Locations | EU Points of Presence |
| Storage Locations | Minimal (edge caching only) |
| Data Retention | Real-time processing, logs 72 hours |

**Contractual Provisions (per vendor terms; verify current status):**
| Requirement | Target | Evidence |
|-------------|--------|----------|
| DPA with SCCs | Standard DPA available (verify execution status via contract register) | Cloudflare DPA |
| Audit rights | Via SOC2 (vendor-reported) | Cloudflare Trust Hub |
| EU data processing | Configurable (verify in vendor settings) | Regional services |

**Certifications** *(vendor-reported; verify via vendor trust centers)*:
- SOC 2 Type II (vendor-reported; verify via [Cloudflare Trust Hub](https://www.cloudflare.com/trust-hub/compliance-resources/))
- ISO 27001 (vendor-reported; verify via Cloudflare Trust Hub)
- PCI DSS (vendor-reported)

**Last Verification**: Certification status should be verified quarterly via Cloudflare Trust Hub.

**Criticality Assessment:**
| Factor | Assessment |
|--------|------------|
| Criticality Level | HIGH |
| Substitutability | MEDIUM (alternatives: Akamai, Fastly) |
| Dependency | Security and performance |
| Risk Level | MEDIUM |

---

### 3.3 Datadog, Inc.

**Provider Identification (B_02.01):**
| Field | Value |
|-------|-------|
| Legal Name | Datadog, Inc. |
| LEI | 549300T5TJR1NGFCQ415 |
| Registration Country | United States (US) |
| EU Representative | Datadog UK Limited |
| Headquarters | 620 8th Avenue, New York, NY 10018 |

**Services Provided (B_03.01):**
| Service Code | Service Description | Our Usage |
|--------------|---------------------|-----------|
| DA | Data Analytics Services | APM, metrics analysis |
| IS | ICT Security Services | Log management, SIEM |

**Data Processing Details:**
| Attribute | Value |
|-----------|-------|
| Data Types Processed | System metrics, application logs, traces |
| Processing Locations | EU (Frankfurt) or US (configurable) |
| Storage Locations | EU (Frankfurt) - configured |
| Data Retention | Configurable (default 15 days logs) |
| Sensitive Data | Masked/excluded by policy |

**Contractual Provisions (per vendor terms; verify current status):**
| Requirement | Status | Evidence |
|-------------|--------|----------|
| DPA with SCCs | Standard DPA available (verify execution status via contract register) | Datadog DPA |
| EU data residency | Configured (verify in vendor portal) | EU site configuration |
| Audit rights | Via SOC2 (vendor-reported) | Datadog Trust Center |

**Certifications** *(vendor-reported; verify via vendor trust centers)*:
- SOC 2 Type II (vendor-reported; verify via [Datadog Trust Center](https://www.datadoghq.com/security/))
- ISO 27001 (vendor-reported; verify via Datadog Trust Center)
- HIPAA eligible (vendor-reported)

**Last Verification**: Certification status should be verified quarterly via Datadog Trust Center.

**Criticality Assessment:**
| Factor | Assessment |
|--------|------------|
| Criticality Level | MEDIUM |
| Substitutability | MEDIUM (alternatives: New Relic, Dynatrace) |
| Dependency | Operational visibility |
| Risk Level | LOW |

---

### 3.4 Sentry (Functional Software, Inc.)

**Provider Identification (B_02.01):**
| Field | Value |
|-------|-------|
| Legal Name | Functional Software, Inc. |
| Registration Country | United States (US) |
| Headquarters | 45 Fremont Street, San Francisco, CA 94105 |

**Services Provided:**
| Service Code | Service Description | Our Usage |
|--------------|---------------------|-----------|
| SD | Software Development/Maintenance | Error tracking |

**Data Processing Details:**
| Attribute | Value |
|-----------|-------|
| Data Types Processed | Application errors, stack traces |
| Processing Locations | EU (Frankfurt) |
| Storage Locations | EU (Frankfurt) |
| Data Retention | 90 days |
| Sensitive Data | Scrubbed by configuration |

**Certifications** *(vendor-reported; verify via vendor trust centers)*:
- SOC 2 Type II (vendor-reported; verify via [Sentry Security](https://sentry.io/security/))
- ISO 27001 (vendor-reported; verify via Sentry Security)
- GDPR commitments (vendor-asserted)

**Last Verification**: Certification status should be verified quarterly via Sentry Security page.

**Criticality Assessment:**
| Factor | Assessment |
|--------|------------|
| Criticality Level | MEDIUM |
| Substitutability | EASY (alternatives: Rollbar, Bugsnag) |
| Dependency | Error detection |
| Risk Level | LOW |

---

### 3.5 Twilio SendGrid

**Provider Identification (B_02.01):**
| Field | Value |
|-------|-------|
| Legal Name | Twilio Inc. |
| LEI | 5493004W8TRGD63XZ936 |
| Registration Country | United States (US) |
| Headquarters | 101 Spear Street, San Francisco, CA 94105 |

**Services Provided:**
| Service Code | Service Description | Our Usage |
|--------------|---------------------|-----------|
| OT | Other ICT Services | Transactional email delivery |

**Data Processing Details:**
| Attribute | Value |
|-----------|-------|
| Data Types Processed | Email addresses, notification content |
| Processing Locations | US (with EU SCCs; verify transfer impact assessment has been completed) |
| Storage Locations | US (with EU SCCs; verify transfer impact assessment has been completed) |
| Data Retention | 7 days (logs), real-time delivery |

**Contractual Provisions:**
| Requirement | Status | Evidence |
|-------------|--------|----------|
| DPA with SCCs | Planned (verify actual signature status via contract register) | Twilio DPA |
| Data minimization | ✓ Configured | Limited data sent |

**Certifications** *(vendor-reported; verify via vendor trust centers)*:
- SOC 2 Type II (vendor-reported; verify via [Twilio Trust Center](https://www.twilio.com/en-us/trust-center))
- ISO 27001 (vendor-reported; verify via Twilio Trust Center)

**Last Verification**: Certification status should be verified quarterly via Twilio Trust Center.

**Criticality Assessment:**
| Factor | Assessment |
|--------|------------|
| Criticality Level | LOW |
| Substitutability | EASY (alternatives: AWS SES, Mailgun) |
| Dependency | Email notifications only |
| Risk Level | LOW |

---

### 3.6 Stripe, Inc.

**Provider Identification (B_02.01):**
| Field | Value |
|-------|-------|
| Legal Name | Stripe, Inc. |
| LEI | 549300GSSFC3I4LMIT80 |
| Registration Country | United States (US) |
| EU Entity | Stripe Payments Europe, Ltd. (Ireland) |
| Headquarters | 354 Oyster Point Boulevard, South San Francisco, CA 94080 |

**Services Provided:**
| Service Code | Service Description | Our Usage |
|--------------|---------------------|-----------|
| OT | Other ICT Services | Payment processing |

**Data Processing Details:**
| Attribute | Value |
|-----------|-------|
| Data Types Processed | Payment card data, billing information |
| Processing Locations | EU (Ireland) for EU customers |
| Storage Locations | Stripe-managed (PCI DSS) |
| Data Retention | Per PCI DSS requirements |

**Contractual Provisions (per vendor terms; verify current status):**
| Requirement | Status | Evidence |
|-------------|--------|----------|
| DPA | Standard DPA available (verify execution status via contract register) | Stripe DPA |
| PCI DSS compliance | Vendor-reported Level 1 (verify via vendor trust center) | Attestation of Compliance |
| SCA compliance | Vendor-reported (verify via vendor documentation) | Stripe documentation |

**Certifications** *(vendor-reported; verify via vendor trust centers)*:
- PCI DSS Level 1 (vendor-reported; verify via [Stripe Compliance](https://stripe.com/docs/security/stripe))
- SOC 1, 2 Type II (vendor-reported; verify via Stripe Compliance)
- ISO 27001 (vendor-reported; verify via Stripe Compliance)

**Last Verification**: Certification status should be verified quarterly via Stripe Compliance page and [Stripe Trust Center](https://stripe.com/trust-center).

**Criticality Assessment:**
| Factor | Assessment |
|--------|------------|
| Criticality Level | HIGH |
| Substitutability | MEDIUM (alternatives: Adyen, Checkout.com) |
| Dependency | Payment processing |
| Risk Level | LOW (PCI DSS mitigated) |

---

## 4. Subcontractor Chain (B_99.01)

```
Platform (Us)
    │
    ├── AWS (CRITICAL)
    │   └── AWS Sub-processors (per AWS list)
    │
    ├── Cloudflare (HIGH)
    │   └── Cloudflare network partners
    │
    ├── Datadog (MEDIUM)
    │   └── Datadog infrastructure providers
    │
    ├── Sentry (MEDIUM)
    │   └── Sentry hosting (GCP)
    │
    ├── SendGrid/Twilio (LOW)
    │   └── Twilio sub-processors
    │
    └── Stripe (HIGH)
        └── Stripe payment network partners
```

---

## 5. Change Notification

Clients will be notified of subcontractor changes per contractual terms:

| Change Type | Notice Period | Client Rights |
|-------------|---------------|---------------|
| New critical subcontractor | 60 days | Prior written consent |
| New standard subcontractor | 30 days | Objection within 15 days |
| Service scope change | 30 days | Review and comment |
| Location change | 60 days | Objection within 30 days |
| Subcontractor termination | 14 days | Information only |

---

## 6. Updates

| Date | Change | Description |
|------|--------|-------------|
| 2025-12-21 | Due diligence corrections | Replaced "Signed/Executed" DPA claims with "Standard DPA available (verify execution status)" per Canon §4.2 (avoid unprovable claims). Affected: Cloudflare, Datadog, Stripe sections. |
| 2025-01-17 | Initial | Register created |

---

*This register is updated upon material changes to subcontractor arrangements. Clients are notified per contractual terms.*
