# Subprocessors Register (EU-only)

**Document Type**: Compliance Record
**Version**: 1.0.0
**Last Updated**: 2025-12-16
**Owner**: Data Protection Officer
**Review Frequency**: Quarterly
**Scope**: EU-only CCEA Cloud platform

---

## 1. EU-only Commitment

**The Platform Provider commits to using only EU-based subprocessors for all personal data processing.**

This commitment is:
- A binding term of the Data Processing Agreement (DPA)
- Enforced by automated EU-only drift checks
- Verified quarterly through subprocessor audits
- Communicated to customers in the Privacy Policy

**No personal data is transferred outside the European Union.**

---

## 2. Approved Subprocessors

### 2.1 Infrastructure Providers

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **Amazon Web Services (AWS)** | Amazon Web Services EMEA SARL | Cloud infrastructure | All Cloud-zone data |

**AWS Services Used:**

| AWS Service | Region | Purpose | Data Category |
|-------------|--------|---------|---------------|
| RDS (PostgreSQL) | eu-central-1 (Frankfurt) | Primary database | User accounts, strategies, telemetry, commands, audit logs |
| RDS (PostgreSQL) | eu-west-1 (Ireland) | Disaster recovery replica | Same as primary (encrypted replicas) |
| S3 | eu-central-1 (Frankfurt) | Object storage | Artifacts, models, backtest results, SBOM |
| S3 | eu-west-1 (Ireland) | Backup storage | Encrypted backups |
| ElastiCache (Redis) | eu-central-1 (Frankfurt) | Session management | Session tokens, rate limits |
| CloudWatch | eu-central-1 (Frankfurt) | Logging & monitoring | Application logs (redacted) |
| KMS | eu-central-1 (Frankfurt) | Key management | Encryption keys |
| Secrets Manager | eu-central-1 (Frankfurt) | Secrets storage | Internal service credentials |

**EU Residency Evidence:**
- AWS Region: `eu-central-1` (Frankfurt, Germany)
- AWS Region: `eu-west-1` (Dublin, Ireland)
- AWS GDPR DPA: Signed
- AWS Data Processing Addendum: [AWS DPA](https://aws.amazon.com/compliance/gdpr-center/)
- Data residency controls: Region-lock enforced via IAM policies
- Last Review: 2025-01-15
- Next Review: 2025-04-15

---

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **Supabase** | Supabase Inc. (EU infrastructure) | Database hosting alternative | All Cloud-zone data |

**Supabase Configuration:**

| Service | Region | Purpose |
|---------|--------|---------|
| PostgreSQL | EU (Germany) | Primary database |
| Realtime | EU (Germany) | Real-time subscriptions |
| Auth | EU (Germany) | Authentication (if used) |

**EU Residency Evidence:**
- Data Center: EU (Germany)
- GDPR DPA: Signed
- Data residency commitment: [Supabase Privacy](https://supabase.com/privacy)
- Last Review: 2025-01-15
- Next Review: 2025-04-15

---

### 2.2 Payment Processing

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **Stripe** | Stripe Payments Europe, Limited (Ireland) | Payment processing | Billing data only |

**Stripe Services:**

| Service | Region | Data Category |
|---------|--------|---------------|
| Payments | EU (Ireland) | Payment card data (tokenized) |
| Billing | EU (Ireland) | Invoice data, subscription status |
| Tax | EU (Ireland) | VAT information |

**EU Residency Evidence:**
- Legal Entity: Stripe Payments Europe, Limited (Ireland)
- Data Processing: configured for EU-region processing where supported by the vendor (deployment-dependent; verify in vendor documentation and contract)
- PCI DSS Level 1: vendor-reported certification (verify in current Stripe documentation)
- GDPR DPA: [Stripe DPA](https://stripe.com/legal/dpa)
- Last Review: 2025-01-15
- Next Review: 2025-04-15

**Data Minimization:**
- Payment card numbers never stored by Platform
- Only tokenized references stored
- Billing email stored for invoicing

---

### 2.3 Communication Services

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **AWS SES** | Amazon Web Services EMEA SARL | Transactional email | Email addresses, notification content |

**AWS SES Configuration:**

| Service | Region | Purpose |
|---------|--------|---------|
| Simple Email Service | eu-west-1 (Ireland) | Transactional emails |

**EU Residency Evidence:**
- AWS Region: `eu-west-1` (Dublin, Ireland)
- Covered under AWS DPA
- Last Review: 2025-01-15
- Next Review: 2025-04-15

---

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **SendGrid** (Alternative) | Twilio Ireland Limited | Transactional email | Email addresses, notification content |

**SendGrid Configuration:**

| Service | Region | Purpose |
|---------|--------|---------|
| Email API | EU | Transactional emails |

**EU Residency Evidence:**
- Processing Region: EU
- GDPR DPA: Signed
- Last Review: 2025-01-15
- Next Review: 2025-04-15

---

### 2.4 Monitoring and Error Tracking

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **Sentry** | Functional Software, Inc. (EU data center) | Error monitoring | Error logs (redacted, no PII) |

**Sentry Configuration:**

| Service | Region | Data Category |
|---------|--------|---------------|
| Error Tracking | EU (Germany) | Stack traces, error context |
| Performance | EU (Germany) | Transaction traces |

**EU Residency Evidence:**
- Data Center: EU (Germany)
- GDPR DPA: Signed
- Data scrubbing: Enabled (PII removal)
- Last Review: 2025-01-15
- Next Review: 2025-04-15

**Data Minimization:**
- PII scrubbing enabled by default
- No user identifiers in error reports
- Stack traces stripped of sensitive context

---

## 3. Subprocessor Summary Table

| Subprocessor | Service | EU Region | DPA Status | Last Review | Next Review |
|--------------|---------|-----------|------------|-------------|-------------|
| AWS | Infrastructure | eu-central-1, eu-west-1 | Signed | 2025-01-15 | 2025-04-15 |
| Supabase | Database | EU (Germany) | Signed | 2025-01-15 | 2025-04-15 |
| Stripe | Payments | EU (Ireland) | Signed | 2025-01-15 | 2025-04-15 |
| AWS SES | Email | eu-west-1 | Signed | 2025-01-15 | 2025-04-15 |
| SendGrid | Email | EU | Signed | 2025-01-15 | 2025-04-15 |
| Sentry | Monitoring | EU (Germany) | Signed | 2025-01-15 | 2025-04-15 |

---

## 4. Subprocessor Change Management

### 4.1 Notification Procedure

When engaging a new subprocessor:

1. **30-day advance notice** to all customers (Controllers)
2. **Notification method**: Email to billing contact + in-app notification
3. **Information provided**:
   - Subprocessor name and legal entity
   - Purpose and data categories
   - EU region/location evidence
   - DPA status
   - Effective date

### 4.2 Objection Process

Customers may object to new subprocessors:

1. **Objection window**: 30 days from notification
2. **Objection method**: Written notice to dpo@[company-domain].com
3. **Resolution**: Good faith negotiation within 30 days
4. **Unresolved objection**: Customer may terminate affected Services without penalty

### 4.3 Emergency Changes

In case of emergency (e.g., subprocessor security incident):

1. **Immediate action**: Switch to approved alternative
2. **Notification**: Within 72 hours
3. **Documentation**: Incident report with justification
4. **Review**: Post-incident assessment

---

## 5. EU-only Drift Check

### 5.1 Automated Verification

The platform implements automated EU-only drift checks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EU-ONLY DRIFT CHECK                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CHECK FREQUENCY: Every deployment + hourly runtime check                   │
│                                                                              │
│  VERIFIED ITEMS:                                                            │
│  ├── Database endpoints (RDS) → must be eu-central-1 or eu-west-1          │
│  ├── Object storage (S3) → must be eu-central-1 or eu-west-1               │
│  ├── Cache endpoints (Redis) → must be eu-central-1                         │
│  ├── Log destinations (CloudWatch) → must be eu-central-1                   │
│  ├── Email service (SES) → must be eu-west-1                               │
│  ├── Error tracking (Sentry) → must be EU region                           │
│  └── Payment processor (Stripe) → must be EU entity                         │
│                                                                              │
│  FAILURE MODE: FAIL-CLOSED                                                  │
│  ├── Deployment blocked if any endpoint is non-EU                          │
│  ├── Runtime alert + automatic rollback if drift detected                   │
│  └── Incident created for investigation                                     │
│                                                                              │
│  OUTPUT: Machine-readable report (JSON)                                     │
│  └── Stored in evidence pack for audit                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Drift Check Report Format

```json
{
  "check_id": "drift-check-2025-01-15-001",
  "timestamp": "2025-01-15T10:00:00Z",
  "status": "PASS",
  "checks": [
    {
      "component": "database_primary",
      "endpoint": "xxx.eu-central-1.rds.amazonaws.com",
      "region": "eu-central-1",
      "eu_compliant": true
    },
    {
      "component": "database_replica",
      "endpoint": "xxx.eu-west-1.rds.amazonaws.com",
      "region": "eu-west-1",
      "eu_compliant": true
    },
    {
      "component": "object_storage",
      "bucket": "platform-artifacts-eu",
      "region": "eu-central-1",
      "eu_compliant": true
    },
    {
      "component": "cache",
      "endpoint": "xxx.eu-central-1.cache.amazonaws.com",
      "region": "eu-central-1",
      "eu_compliant": true
    },
    {
      "component": "email_service",
      "service": "ses",
      "region": "eu-west-1",
      "eu_compliant": true
    },
    {
      "component": "error_tracking",
      "service": "sentry",
      "region": "EU",
      "eu_compliant": true
    },
    {
      "component": "payment_processor",
      "service": "stripe",
      "entity": "Stripe Payments Europe, Limited",
      "region": "EU (Ireland)",
      "eu_compliant": true
    }
  ],
  "subprocessors_verified": 6,
  "non_eu_endpoints": 0,
  "next_check": "2025-01-15T11:00:00Z"
}
```

---

## 6. DPA Repository

### 6.1 Signed DPAs

| Subprocessor | DPA Version | Signed Date | Signatory | Storage Location |
|--------------|-------------|-------------|-----------|------------------|
| AWS | AWS DPA v2.0 | 2024-01-15 | DPO | `/legal/dpas/aws_dpa_signed.pdf` |
| Supabase | Supabase DPA v1.0 | 2024-01-15 | DPO | `/legal/dpas/supabase_dpa_signed.pdf` |
| Stripe | Stripe DPA v2.0 | 2024-01-15 | DPO | `/legal/dpas/stripe_dpa_signed.pdf` |
| SendGrid | Twilio DPA v1.0 | 2024-01-15 | DPO | `/legal/dpas/sendgrid_dpa_signed.pdf` |
| Sentry | Sentry DPA v1.0 | 2024-01-15 | DPO | `/legal/dpas/sentry_dpa_signed.pdf` |

### 6.2 DPA Review Schedule

| Subprocessor | Last Review | Next Review | Reviewer |
|--------------|-------------|-------------|----------|
| AWS | 2025-01-15 | 2025-04-15 | DPO |
| Supabase | 2025-01-15 | 2025-04-15 | DPO |
| Stripe | 2025-01-15 | 2025-04-15 | DPO |
| SendGrid | 2025-01-15 | 2025-04-15 | DPO |
| Sentry | 2025-01-15 | 2025-04-15 | DPO |

---

## 7. Audit Evidence

### 7.1 Evidence Pack Contents

For customer due diligence and audits, the following evidence is available:

| Evidence Type | Description | Location |
|---------------|-------------|----------|
| Subprocessor list | This document | `docs/compliance/SUBPROCESSORS_REGISTER.md` |
| Drift check reports | Automated EU verification | Evidence pack export |
| DPA copies | Signed agreements | Available on request |
| Region configuration | Infrastructure settings | Evidence pack export |
| Compliance certifications | SOC 2, ISO 27001 | Available on request |

### 7.2 Customer Access

Customers can request:
- Current subprocessor list (this document)
- EU-only drift check reports (last 90 days)
- DPA summaries (full copies under NDA)
- Change notification history

Request via: dpo@[company-domain].com

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | Compliance Team | Initial release - GDPR Phase 1 |

---

## 9. References

- GDPR Regulation (EU) 2016/679 - Article 28, 44-49
- `docs/legal/DPA_TEMPLATE.md`
- `docs/legal/PRIVACY_POLICY.md`
- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md`
- AWS GDPR Center: https://aws.amazon.com/compliance/gdpr-center/
- Stripe GDPR: https://stripe.com/guides/general-data-protection-regulation
