# EU Data Residency Configuration
## DORA Article 30(2)(b) Compliance

**Version**: 1.0
**Date**: 2025-01-17
**Status**: Configuration Specification (illustrative defaults)
**Legal Reference**: DORA Article 30(2)(b), GDPR

> **Note**: This document describes the target EU data residency configuration design. Infrastructure locations and providers shown are illustrative defaults; actual deployment configuration is verified in production infrastructure documentation and provided to contracted clients. This is not a claim that infrastructure is currently deployed.

---

## 1. Overview

This document describes the EU data residency configuration design available to clients requiring data to remain within the European Union/European Economic Area.

### DORA Requirement

Article 30(2)(b) mandates contracts include:
> "the locations (that is to say regions or countries) where the contracted or subcontracted functions are to be provided and where data is to be processed, including the storage location, and the requirement for the ICT third-party service provider to notify the financial entity in advance if it envisages changing such locations"

---

## 2. Default Configuration *(illustrative; actual deployment verified separately)*

### 2.1 Primary Infrastructure (Target Design)

| Component | Location | Provider | EU-Only |
|-----------|----------|----------|---------|
| Application Servers | Frankfurt, DE (eu-central-1) | AWS | ✓ |
| Database (Primary) | Frankfurt, DE (eu-central-1) | AWS RDS | ✓ |
| Database (Replica) | Frankfurt, DE (eu-central-1) | AWS RDS | ✓ |
| Object Storage | Frankfurt, DE (eu-central-1) | AWS S3 | ✓ |
| Cache Layer | Frankfurt, DE (eu-central-1) | AWS ElastiCache | ✓ |
| Message Queue | Frankfurt, DE (eu-central-1) | AWS SQS | ✓ |
| Search/Analytics | Frankfurt, DE (eu-central-1) | AWS OpenSearch | ✓ |

### 2.2 Secondary/DR Infrastructure

| Component | Location | Provider | EU-Only |
|-----------|----------|----------|---------|
| DR Application Servers | Ireland (eu-west-1) | AWS | ✓ |
| DR Database | Ireland (eu-west-1) | AWS RDS | ✓ |
| DR Object Storage | Ireland (eu-west-1) | AWS S3 | ✓ |
| Backup Storage | Frankfurt, DE (eu-central-1) | AWS S3 Glacier | ✓ |

---

## 3. Data Flow Map

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EU DATA RESIDENCY BOUNDARY                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │    Client    │     │   CDN Edge   │     │  Application │            │
│  │   Browser    │────►│ (Cloudflare) │────►│   Servers    │            │
│  │              │     │   EU PoPs    │     │  Frankfurt   │            │
│  └──────────────┘     └──────────────┘     └──────────────┘            │
│                                                   │                      │
│                                                   ▼                      │
│                              ┌──────────────────────────────────┐       │
│                              │         Frankfurt Data Center     │       │
│                              │  ┌────────────┐ ┌────────────┐   │       │
│                              │  │  Database  │ │   Cache    │   │       │
│                              │  │  (RDS)     │ │ (ElastiC)  │   │       │
│                              │  └────────────┘ └────────────┘   │       │
│                              │  ┌────────────┐ ┌────────────┐   │       │
│                              │  │  Storage   │ │   Queue    │   │       │
│                              │  │  (S3)      │ │  (SQS)     │   │       │
│                              │  └────────────┘ └────────────┘   │       │
│                              └──────────────────────────────────┘       │
│                                          │                              │
│                                          │ Replication                  │
│                                          ▼                              │
│                              ┌──────────────────────────────────┐       │
│                              │          Ireland (DR)             │       │
│                              │  ┌────────────┐ ┌────────────┐   │       │
│                              │  │  DR DB     │ │ DR Storage │   │       │
│                              │  └────────────┘ └────────────┘   │       │
│                              └──────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Classification and Location

### 4.1 Client Data (Always EU)

| Data Type | Processing | Storage | Backup | Retention |
|-----------|------------|---------|--------|-----------|
| Trading Strategies | Frankfurt | Frankfurt | Frankfurt | Active + 7 years |
| Backtest Results | Frankfurt | Frankfurt | Frankfurt | Active + 7 years |
| ML/AI Models | Frankfurt | Frankfurt | Frankfurt | Active + 7 years |
| User Configurations | Frankfurt | Frankfurt | Frankfurt | Active + 7 years |
| Audit Logs | Frankfurt | Frankfurt | Frankfurt | 7 years |
| API Credentials | Frankfurt | Frankfurt | Frankfurt | Active |
| Performance Data | Frankfurt | Frankfurt | Frankfurt | 5 years |

### 4.2 Operational Data

| Data Type | Processing | Storage | EU-Only Option |
|-----------|------------|---------|----------------|
| Application Logs | Frankfurt | Frankfurt (Datadog EU) | ✓ Default |
| Error Tracking | Frankfurt | Frankfurt (Sentry EU) | ✓ Default |
| Metrics | Frankfurt | Frankfurt (Datadog EU) | ✓ Default |
| Security Logs | Frankfurt | Frankfurt | ✓ Default |

### 4.3 Third-Party Data

| Data Type | Source | Processing | Notes |
|-----------|--------|------------|-------|
| Market Data | Various exchanges | Frankfurt | Aggregated in EU |
| Payment Data | Stripe | EU (Ireland) | PCI DSS isolated |
| Email Notifications | SendGrid | US | Non-sensitive only |

---

## 5. EU-Only Configuration Option

### 5.1 Availability

EU-Only configuration is available at:
- **Standard Tier**: Included (default)
- **Professional Tier**: Included (default)
- **Enterprise Tier**: Included with dedicated region options

### 5.2 What EU-Only Provides

When EU-Only is enabled:

**Data Processing:**
- All client data processing occurs within EU/EEA
- No client data transfers to non-EU jurisdictions
- All database queries processed in EU regions

**Data Storage:**
- Primary storage: Germany (Frankfurt)
- Backup storage: Germany (Frankfurt) or Ireland
- DR storage: Ireland (EU)
- No storage in non-EU regions

**Data Transit:**
- Encrypted in transit (TLS 1.3)
- Routed through EU network paths where possible
- CDN caches within EU Points of Presence only

### 5.3 What EU-Only Does NOT Cover

| Component | Location | Reason | Data Involved |
|-----------|----------|--------|---------------|
| Email delivery | US (SendGrid) | Service limitation | Email addresses only |
| Some CDN paths | Global | Internet routing | Encrypted traffic only |
| Third-party integrations | Varies | Client configuration | Per integration |

---

## 6. AWS Region Configuration

### 6.1 Approved Regions

| Region | Region Code | Use Case |
|--------|-------------|----------|
| Frankfurt, Germany | eu-central-1 | Primary (all services) |
| Ireland | eu-west-1 | DR/Failover only |
| Paris, France | eu-west-3 | Enterprise option |
| Stockholm, Sweden | eu-north-1 | Enterprise option |

### 6.2 Blocked Regions (Data processing/storage prohibited by design)

The following regions are excluded from the EU-only configuration by design and policy:

- All US regions (us-*)
- All Asia Pacific regions (ap-*)
- All South America regions (sa-*)
- All Middle East regions (me-*)
- All Africa regions (af-*)

### 6.3 AWS Config Rules

```yaml
# AWS Config enforcement
aws_config_rules:
  - name: "s3-bucket-region-restriction"
    description: "Ensure S3 buckets are only in EU regions"
    allowed_regions:
      - "eu-central-1"
      - "eu-west-1"

  - name: "rds-region-restriction"
    description: "Ensure RDS instances are only in EU regions"
    allowed_regions:
      - "eu-central-1"
      - "eu-west-1"

  - name: "ec2-region-restriction"
    description: "Ensure EC2 instances are only in EU regions"
    allowed_regions:
      - "eu-central-1"
      - "eu-west-1"
```

---

## 7. Location Change Notification

### 7.1 Notification Requirements

Per DORA Article 30(2)(b) and contract terms:

| Change Type | Notice Period | Client Rights |
|-------------|---------------|---------------|
| New processing location (EU) | 60 days | Review/comment |
| New processing location (non-EU) | 90 days | Prior written consent |
| New storage location (EU) | 60 days | Review/comment |
| New storage location (non-EU) | 90 days | Prior written consent |
| Subcontractor location change | 60 days | Objection rights |
| DR location change (EU) | 30 days | Information |

### 7.2 Notification Process

1. Provider identifies planned location change
2. Impact assessment conducted
3. Client notification sent with:
   - Description of change
   - Effective date
   - Reason for change
   - Data affected
   - Safeguards in place
4. Client response period (per table above)
5. If objection: good faith resolution process
6. If unresolved: Client may terminate affected services

---

## 8. Client-Specific Configuration

### 8.1 Enterprise Options

Enterprise clients may request:

| Option | Description | Availability |
|--------|-------------|--------------|
| Dedicated Region | Single EU region only | Enterprise |
| Country-Specific | Single EU country only | Enterprise |
| Dedicated VPC | Isolated network | Enterprise |
| Private Link | No public internet | Enterprise |

### 8.2 Configuration Request

To request specific configuration:
1. Submit configuration request via client portal
2. Technical review (5 business days)
3. Implementation proposal with timeline
4. Contract amendment if required
5. Implementation and testing
6. Configuration verification report

---

## 9. Compliance Verification

### 9.1 Automated Checks

| Check | Frequency | Method |
|-------|-----------|--------|
| Resource location audit | Daily | AWS Config |
| Data flow analysis | Weekly | Network monitoring |
| S3 bucket locations | Real-time | S3 policies |
| RDS locations | Real-time | IAM policies |
| Cross-region traffic | Daily | VPC Flow Logs |

### 9.2 Client Verification

Clients may verify EU data residency via:
- API endpoint: `GET /api/v1/compliance/data-locations`
- Monthly compliance report
- On-demand audit request

---

## 10. Legal Basis for Transfers

### 10.1 Within EU/EEA

No additional legal basis required for transfers within EU/EEA.

### 10.2 To Third Countries (if applicable)

If any data processing requires non-EU location:

| Legal Basis | Use Case |
|-------------|----------|
| Standard Contractual Clauses (SCCs) | Subcontractor processing |
| Adequacy Decision | UK (if applicable) |
| Client explicit consent | Special requests |

---

## 11. Related Documents

| Document | Purpose |
|----------|---------|
| DORA Contract Template | Contract provisions |
| Subcontractor Register | Subcontractor locations |
| Data Processing Agreement | GDPR compliance |
| Security Whitepaper | Technical controls |

---

*This document is updated when infrastructure changes affect data locations. Clients are notified per contractual terms.*
