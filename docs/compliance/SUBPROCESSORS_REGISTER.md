# Subprocessors Register (EU-only)

**Document Type**: Compliance Record
**Version**: 1.0.0
**Last Updated**: 2025-12-16
**Owner**: Data Protection Officer (role to be assigned; see governance plan)
**Review Frequency**: Quarterly (target; pending operational maturity)
**Scope**: CCEA Cloud platform (EU deployment target)

---

## 1. EU-only Commitment

**The Platform is designed to use EU-based subprocessors for personal data processing wherever feasible.**

> **Note on Exceptions**: Certain ancillary services (e.g., transactional email) may involve non-EU processing under SCCs/DPF. See SUBCONTRACTOR_REGISTER.md for details and transfer impact assessments.

This design commitment:
- Is intended to become a binding DPA term upon contract execution
- Is designed to be enforced by automated EU-only drift checks (test coverage available)
- Is planned to be verified quarterly through subprocessor audits (when operational)
- Is communicated to customers in the Privacy Policy

**Design intent**: In standard deployment configurations, the platform is designed to process personal data within the European Union only. *This is a design commitment; operational enforcement depends on infrastructure deployment, configuration validation, and ongoing drift monitoring. Verification artifacts will be available post-deployment.*

> **Note on Vendor Certifications**: All subcontractor certifications listed in this register are vendor-reported. Where specific certifications are referenced, clients should verify current status directly via the vendor's trust center or compliance portal (links provided where available). Certification status should be reviewed quarterly as part of third-party risk management.

---

## 2. Approved Subprocessors

### 2.1 Infrastructure Providers

| Subprocessor | Legal Entity | Purpose | Data Processed |
|--------------|--------------|---------|----------------|
| **Amazon Web Services (AWS)** | Amazon Web Services EMEA SARL | Cloud infrastructure | All Cloud-zone data |

**AWS Configuration (Deployment Pending):**

> **IMPORTANT**: CustodiaCloud infrastructure is not yet deployed. The configuration below describes planned architecture only and should not be treated as current operational state. Actual deployed configuration will be documented upon production deployment.

**Planned AWS Services**: RDS (PostgreSQL), S3, ElastiCache, CloudWatch, KMS, Secrets Manager
**Planned Regions**: eu-central-1 (Frankfurt), eu-west-1 (Dublin)
**Deployment Status**: Infrastructure deployment pending
**Data Categories (when deployed)**: User accounts, strategies, telemetry, artifacts, audit logs

**EU Residency Controls (to be implemented upon deployment):**
- Region lock via IAM policies (not yet configured)
- AWS GDPR DPA: [Standard AWS DPA available](https://aws.amazon.com/compliance/gdpr-center/) (execution upon deployment)
- Evidence exports: AWS Config/CloudTrail (available upon deployment)
- Review schedule: To be established upon deployment

For detailed planned infrastructure architecture, see `docs/ENTERPRISE_DEPLOYMENT_ARCHITECTURE.md`. For current operational status, contact the compliance team.

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

**EU Residency Evidence (planned configuration; verify upon deployment):**
- Data Center: EU (Germany) — planned
- GDPR DPA: Standard DPA available via Supabase
- Data residency commitment: [Supabase Privacy](https://supabase.com/privacy)
- Last Review: To be scheduled upon deployment
- Next Review: To be scheduled upon deployment

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
- PCI DSS Level 1 (vendor-reported; verify directly with Stripe as part of client/vendor due diligence; not a CustodiaCloud certification claim)
- GDPR DPA: [Stripe DPA](https://stripe.com/legal/dpa)
- Last Review: 2025-01-15
- Next Review: 2025-04-15

**Data Minimization (design goal):**
- Payment card numbers are designed not to be stored by the platform (tokenization via Stripe; verify in deployment configuration)
- Only tokenized references designed to be stored
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

**EU Residency Evidence (planned configuration; verify upon deployment):**
- Processing Region: EU
- GDPR DPA: Standard DPA available via Twilio
- Last Review: To be scheduled upon deployment
- Next Review: To be scheduled upon deployment

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

**EU Residency Evidence (planned configuration; verify upon deployment):**
- Data Center: EU (Germany) — planned
- GDPR DPA: Standard DPA available via Sentry
- Data scrubbing: Enabled (PII removal) — design target
- Last Review: To be scheduled upon deployment
- Next Review: To be scheduled upon deployment

**Data Minimization:**
- PII scrubbing enabled by default
- No user identifiers in error reports
- Stack traces stripped of sensitive context

---

## 3. Subprocessor Summary Table

| Subprocessor | Service | EU Region (Planned) | DPA Status | Review Schedule |
|--------------|---------|---------------------|------------|-----------------|
| AWS | Infrastructure | eu-central-1, eu-west-1 | Standard DPA available | Upon deployment |
| Supabase | Database | EU (Germany) | Standard DPA available | Upon deployment |
| Stripe | Payments | EU (Ireland) | Standard DPA available | Upon deployment |
| AWS SES | Email | eu-west-1 | Standard DPA available | Upon deployment |
| SendGrid | Email | EU | Standard DPA available | Upon deployment |
| Sentry | Monitoring | EU (Germany) | Standard DPA available | Upon deployment |

> **Note**: This table describes planned subprocessor configuration. DPA execution and review schedules will commence upon customer contract execution. "Standard DPA available" indicates vendor offers standard DPA terms; actual execution is pending.

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
│  CHECK FREQUENCY: Designed for every deployment + hourly runtime check     │
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
  "subprocessors_listed": 6,
  "eu_only_configuration": true,
  "verification_status": "pending_operational_deployment"
}
```

---

## 6. DPA Repository

### 6.1 DPA Templates

> **Note**: CustodiaCloud is a pre-seed company. DPA templates are available for execution. Actual DPA execution dates will be recorded upon customer contract signature.

| Subprocessor | DPA Template | DPA Status | Execution Plan | Template Location |
|--------------|--------------|------------|----------------|-------------------|
| AWS | AWS DPA v2.0 | Template available | Upon customer contract | `/legal/dpas/aws_dpa_template.pdf` |
| Supabase | Supabase DPA v1.0 | Template available | Upon customer contract | `/legal/dpas/supabase_dpa_template.pdf` |
| Stripe | Stripe DPA v2.0 | Template available | Upon customer contract | `/legal/dpas/stripe_dpa_template.pdf` |
| SendGrid | Twilio DPA v1.0 | Template available | Upon customer contract | `/legal/dpas/sendgrid_dpa_template.pdf` |
| Sentry | Sentry DPA v1.0 | Template available | Upon customer contract | `/legal/dpas/sentry_dpa_template.pdf` |

### 6.2 DPA Review Schedule (When Operational)

> **Note**: Review schedule will commence upon first customer contract execution.

| Subprocessor | Review Frequency | First Review | Reviewer Role |
|--------------|------------------|--------------|---------------|
| AWS | Quarterly | Upon operational commencement | DPO |
| Supabase | Quarterly | Upon operational commencement | DPO |
| Stripe | Quarterly | Upon operational commencement | DPO |
| SendGrid | Quarterly | Upon operational commencement | DPO |
| Sentry | Quarterly | Upon operational commencement | DPO |

---

## 7. Audit Evidence

### 7.1 Evidence Pack Contents

For customer due diligence and audits, the following evidence is available:

| Evidence Type | Description | Location |
|---------------|-------------|----------|
| Subprocessor list | This document | `docs/compliance/SUBPROCESSORS_REGISTER.md` |
| Drift check reports | Automated EU verification (test coverage) | Evidence pack export (when operational) |
| DPA templates | Template agreements for execution | Available on request |
| Region configuration | Infrastructure settings | Evidence pack export |
| Vendor certifications | Vendor-reported; verify via vendor trust centers | Vendor compliance portals |

### 7.2 Customer Access (When Operational)

> **Note**: Customer access process will be established upon first customer contract execution.

Upon operational commencement, customers will be able to request:
- Current subprocessor list (this document)
- EU-only drift check reports (when available)
- DPA summaries (full copies under NDA)
- Change notification history

Request via: dpo@[company-domain].com

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | Compliance Team | Initial release - GDPR Phase 1 |
| 1.0.1 | 2025-12-19 | Internal Review | **Critical correction**: Replaced absolute claims with Canon-compliant language. "Commits" → "designed to use"; "Signed Date" → "Template available"; "Verified quarterly" → "Planned to verify quarterly (when operational)"; "Enforced by" → "Designed to be enforced"; "No personal data transferred" → "No personal data designed to be transferred in standard deployment". |
| 1.0.2 | 2025-12-19 | Internal Review | **Data residency/Privacy correction**: Removed "GDPR DPA: Signed" claims and specific review dates that implied executed contracts. Changed to "Standard DPA available" with "Upon deployment" review schedule. Added clarifying note that DPA execution is pending customer contracts. |

---

## 9. References

- GDPR Regulation (EU) 2016/679 - Article 28, 44-49
- `docs/legal/DPA_TEMPLATE.md`
- `docs/legal/PRIVACY_POLICY.md`
- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md`
- AWS GDPR Center: https://aws.amazon.com/compliance/gdpr-center/
- Stripe GDPR: https://stripe.com/guides/general-data-protection-regulation
