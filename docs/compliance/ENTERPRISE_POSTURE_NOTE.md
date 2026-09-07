# Enterprise Posture Note: CCEA Cloud/On-Prem/VPC Deployments

**Document Version**: 1.0.0
**Effective Date**: 2025-12-17
**Classification**: INTERNAL / COMPLIANCE
**Status**: ACTIVE

## 1. Overview

### 1.1 Purpose

This document defines the official posture for enterprise deployments of the CCEA platform, including on-premises, VPC, and hybrid configurations. It establishes the boundaries for marketing claims, contractual commitments, and technical constraints that preserve the "software/platform provider" posture while supporting enterprise deployment flexibility.

### 1.2 Design Doc Reference

```
Phase 9 — Enterprise/on-prem/VPC posture (Design Doc 16.3) and scope control

Goal: support enterprise on-prem/VPC deployments in a way that preserves the
"software/platform provider" posture and is auditable.

Reference: docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L968-L972
```

### 1.3 Scope

This document applies to:

- Enterprise on-premises deployments
- VPC (Virtual Private Cloud) deployments
- Hybrid configurations (Cloud + local components)
- Air-gapped deployments

---

## 2. Supported Deployment Modes

### 2.1 Deployment Mode Matrix

| Mode | Cloud Control Plane | Telemetry Destination | Update Source | Evidence Export | EU-Only |
|------|--------------------|-----------------------|---------------|-----------------|---------|
| **SaaS (Default)** | Vendor-hosted | Cloud (redacted) | Cloud | Cloud API | Yes |
| **Enterprise Cloud** | Vendor-hosted | Cloud (RAW optional) | Cloud | Cloud API | Yes |
| **On-Prem Full** | Customer-hosted | Local only | Local registry | Local | Yes |
| **VPC Managed** | Customer VPC | Local or Cloud | Cloud or mirror | Local + API | Yes |
| **Air-Gapped** | Customer-hosted | Local only | Offline | Local only | Yes |

### 2.2 Mode Descriptions

#### 2.2.1 SaaS (Default)

- Standard cloud-hosted deployment
- Telemetry: AGGREGATED by default, DETAILED_NON_SENSITIVE opt-in
- All data processing in EU regions
- Evidence pack via Cloud API

#### 2.2.2 Enterprise Cloud

- Cloud-hosted with enterprise features
- Telemetry: RAW_ORDER_EVENTS available with explicit opt-in
- Customer-managed encryption keys (CMK) supported
- Enhanced SLAs and support

#### 2.2.3 On-Prem Full

- Complete on-premises deployment
- Designed so data remains within customer infrastructure (verify via deployment configuration and network egress controls)
- Telemetry stays local by default
- Offline evidence pack generation
- Customer manages all infrastructure

#### 2.2.4 VPC Managed

- Customer controls infrastructure in their VPC
- Choice of telemetry destination (local or Cloud)
- Registry mirror for artifact distribution
- Hybrid evidence export (local + API)

#### 2.2.5 Air-Gapped

- No external network connectivity
- Pre-populated artifact registry
- Offline signature verification
- Local-only evidence export
- Manual update delivery

---

## 3. Contractual Boundaries

### 3.1 Data Processing Roles

| Deployment Mode | CCEA Role | Customer Role | DPA Required |
|-----------------|-----------|---------------|--------------|
| SaaS | Processor | Controller | Yes |
| Enterprise Cloud | Processor | Controller | Yes (Enterprise DPA) |
| On-Prem Full | Licensor only | Controller + Processor | License Agreement |
| VPC Managed | Processor (if Cloud telemetry) | Controller | Conditional |
| Air-Gapped | Licensor only | Controller + Processor | License Agreement |

### 3.2 Data Residency Commitments

**All deployment modes are designed with EU-priority data residency as the default (verify via deployment configuration and audits). Some sub-processors may process limited data outside the EU under Standard Contractual Clauses (SCCs) or Data Privacy Framework (DPF):**

| Component | SaaS/Enterprise Cloud | On-Prem/VPC/Air-Gapped |
|-----------|----------------------|------------------------|
| Control Plane | EU regions | Customer EU infrastructure |
| Databases | EU regions | Customer EU infrastructure |
| Object Storage | EU regions | Customer EU infrastructure |
| Telemetry Storage | EU regions | Local (EU infrastructure) |
| Backups | EU regions | Customer EU infrastructure |
| Logs | EU regions | Customer EU infrastructure |
| Support Data | EU regions (with consent) | Not transmitted by design (verify via support tooling configuration) |

### 3.3 Support and Maintenance Boundaries

| Activity | SaaS | Enterprise Cloud | On-Prem | VPC | Air-Gapped |
|----------|------|------------------|---------|-----|------------|
| Remote Support | With consent | With consent | Never | With consent | Never |
| Log Access | With consent | With consent | Never | With consent | Never |
| Telemetry Access | Redacted only | Contract-defined | Never | Contract-defined | Never |
| Updates | Automatic | Change windows | Manual | Change windows | Manual |
| Monitoring | Vendor | Vendor + Customer | Customer | Customer | Customer |

---

## 4. Telemetry Posture

### 4.1 Telemetry Levels by Mode

| Level | SaaS | Enterprise Cloud | On-Prem | VPC | Air-Gapped |
|-------|------|------------------|---------|-----|------------|
| **AGGREGATED** | Default | Default | Local only | Default | Local only |
| **DETAILED_NON_SENSITIVE** | Opt-in | Opt-in | Local only | Local or Cloud | Local only |
| **RAW_ORDER_EVENTS** | N/A | Enterprise opt-in | Local only | Enterprise opt-in | Local only |

### 4.2 "Telemetry Stays Local" Mode

For enterprise deployments, "Telemetry Stays Local" mode is designed to provide:

1. **No Cloud Transmission**: Telemetry is configured to remain local and not be sent to vendor Cloud infrastructure
2. **Local Storage**: All telemetry stored in customer-controlled storage
3. **Customer Export**: Customer can export telemetry for their own analysis
4. **Audit Trail**: Local audit trail of all telemetry operations
5. **Evidence Pack**: Telemetry can be included in local evidence exports

**Configuration:**

```yaml
# Helm values (on-prem/VPC)
telemetry:
  localMode: true
  cloudExport: false
  storage: local  # or customer-s3, customer-gcs
```

```bash
# Docker Compose
CCEA_TELEMETRY_LOCAL_ONLY=true
CCEA_TELEMETRY_EXTERNAL_EXPORT=false
```

### 4.3 RAW_ORDER_EVENTS (Enterprise Only)

When RAW_ORDER_EVENTS is enabled (enterprise only):

| Requirement | Description |
|-------------|-------------|
| Enterprise License | Active enterprise license required |
| Explicit Opt-In | Per-workspace opt-in with acknowledgment |
| DPA Signature | Enterprise DPA must be signed |
| RAW Addendum | Specific RAW data addendum required |
| Retention Limit | Maximum 30 days (configurable lower) |
| Access Control | Break-glass for support access |
| Audit Trail | Full audit of RAW data access |

**Options for RAW telemetry:**

1. **Cloud RAW (Enterprise Cloud/VPC)**: RAW sent to Cloud with explicit opt-in
2. **Local RAW (On-Prem/Air-Gapped)**: RAW stays local, customer-controlled

---

## 5. Marketing Claim Guardrails

### 5.1 Permitted Claims

The following claims are accurate and can be used in marketing:

| Claim | Accuracy | Notes |
|-------|----------|-------|
| "EU-priority data residency" | TRUE | Core platform data in EU; sub-processors with non-EU processing operate under SCCs/DPF |
| "No credentials leave the Agent" | TRUE | Enforced by redaction + CI |
| "No order commands from Cloud" | TRUE | Protocol constraint |
| "On-premises deployment available" | TRUE | Full on-prem supported |
| "Air-gapped deployment available" | TRUE | Offline mode supported |
| "Customer-managed keys" | TRUE | CMK supported in enterprise |
| "GDPR-aligned design" | TRUE* | *With proper configuration |
| "Telemetry stays local option" | TRUE | Enterprise feature |

### 5.2 Prohibited Claims

The following claims are NOT accurate and must NOT be used:

| Claim | Issue |
|-------|-------|
| "Zero data collection" | FALSE - We collect redacted telemetry in SaaS |
| "Completely offline" | FALSE - SaaS/Enterprise Cloud require connectivity |
| "No personal data processed" | FALSE - Metadata may include personal data |
| "SOC 2 Type II certified" | FALSE*-*Unless certification obtained |
| "ISO 27001 certified" | FALSE*-*Unless certification obtained |

### 5.3 Conditional Claims

These claims require specific conditions to be accurate:

| Claim | Condition | Notes |
|-------|-----------|-------|
| "Data designed to stay in your infrastructure" | On-Prem or Air-Gapped mode | Subject to deployment configuration verification |
| "Telemetry designed to be local-only" | Telemetry Local Mode enabled | Verify via configuration audit |
| "Data sovereignty (design goal)" | On-Prem + CMK + Local telemetry | Requires validated deployment |
| "24/7 support" | Enterprise support contract | Requires signed contract |
| "99.9% SLA" | Enterprise SLA agreement | Requires signed contract |

> **Note**: These claims are conditional and design-oriented. Absolute claims (e.g., "never leaves") should be avoided as they may be difficult to prove definitively and could be undermined by telemetry, updates, or support channels. Use "designed to" language in customer-facing materials.

---

## 6. Evidence Pack Export

### 6.1 Export Capabilities by Mode

| Mode | Export Method | Connectivity | Signature |
|------|---------------|--------------|-----------|
| SaaS | Cloud API | Required | Cloud-signed |
| Enterprise Cloud | Cloud API | Required | Cloud-signed |
| On-Prem | Local CLI/API | None | Offline-signed |
| VPC | Local + Cloud API | Optional | Local or Cloud |
| Air-Gapped | Local CLI/API | None | Offline-signed |

### 6.2 Evidence Categories

All deployment modes can export the following evidence:

| Category | Description | Required |
|----------|-------------|----------|
| Security Baseline | Encryption, keys, MFA status | Yes |
| Supply Chain | Artifacts, SBOMs, signatures | Yes |
| Access Audit | Access logs, RBAC snapshots | Yes |
| Change Journal | Change history, approvals | Yes |
| DSAR Records | Data subject requests | Yes |
| Retention Policies | Retention configuration | Yes |
| Residency Evidence | EU-only drift checks | Yes |
| Telemetry Contracts | Level configuration | Yes |
| Breach Records | Incident documentation | If any |
| Break-Glass Logs | Emergency access records | If any |

### 6.3 Air-Gapped Evidence Export

For air-gapped deployments:

```bash
# Generate evidence pack locally
ccea-cli evidence export \
  --workspace ws-123 \
  --format zip \
  --sign offline \
  --output /secure/evidence-pack-2025-Q1.zip

# Verify pack integrity
ccea-cli evidence verify \
  --pack /secure/evidence-pack-2025-Q1.zip \
  --public-key /keys/evidence-signing.pub
```

---

## 7. Customer-Managed Keys (CMK)

### 7.1 Supported Key Management

| Deployment | Key Management Options |
|------------|----------------------|
| SaaS | Platform-managed only |
| Enterprise Cloud | Platform-managed, AWS KMS, GCP KMS, Azure Key Vault |
| On-Prem | Customer HSM, PKCS#11, Local keystore |
| VPC | AWS KMS, GCP KMS, Azure Key Vault, Customer HSM |
| Air-Gapped | Customer HSM, PKCS#11, Local keystore |

### 7.2 CMK Configuration

```yaml
# Helm values (CMK enabled)
encryption:
  keyManagement: customer_managed
  provider: aws_kms  # or gcp_kms, azure_keyvault, hsm, pkcs11
  aws_kms:
    keyArn: "arn:aws:kms:eu-west-1:123456789012:key/..."
    region: eu-west-1
```

### 7.3 Key Rotation

| Key Type | Platform-Managed | Customer-Managed |
|----------|------------------|------------------|
| Data Encryption Keys | Automatic (90 days) | Customer-controlled |
| Signing Keys | Annual | Customer-controlled |
| Transport Keys | Per-session | Per-session |

---

## 8. Security Controls by Mode

### 8.1 Common Controls (All Modes)

- AES-256-GCM encryption at rest
- TLS 1.3 encryption in transit
- Signed artifacts only
- Digest-pinned dependencies
- Registry allowlist enforcement
- Redaction mandatory (cannot be disabled)

### 8.2 Mode-Specific Controls

| Control | SaaS | Enterprise | On-Prem | VPC | Air-Gapped |
|---------|------|------------|---------|-----|------------|
| mTLS | Optional | Available | Recommended | Recommended | Recommended |
| Network Policies | Platform | Platform | Customer | Customer | Customer |
| Firewall | Platform | Platform | Customer | Customer | Customer |
| WAF | Platform | Platform | Customer | Customer | N/A |
| DDoS Protection | Platform | Platform | Customer | Customer | N/A |
| Intrusion Detection | Platform | Platform | Customer | Customer | Customer |

---

## 9. Audit and Compliance

### 9.1 Audit Artifacts by Mode

| Artifact | SaaS | Enterprise | On-Prem | VPC | Air-Gapped |
|----------|------|------------|---------|-----|------------|
| SOC 2 Report | Roadmap (on request under NDA if obtained) | Roadmap (on request under NDA if obtained) | N/A | N/A | N/A |
| Penetration Test | Planned (annual target) | Planned (annual target) | Customer | Customer | Customer |
| Vulnerability Scan | Planned (continuous target) | Planned (continuous target) | Customer | Customer | Customer |
| Compliance Dashboard | Cloud | Cloud | Local | Local | Local |
| Audit Log Export | API | API | Local | Both | Local |

### 9.2 Compliance Responsibility Matrix

| Compliance Area | SaaS | Enterprise | On-Prem | VPC | Air-Gapped |
|-----------------|------|------------|---------|-----|------------|
| Infrastructure Security | Vendor | Shared | Customer | Customer | Customer |
| Application Security | Vendor | Vendor | Vendor | Vendor | Vendor |
| Data Classification | Customer | Customer | Customer | Customer | Customer |
| Access Management | Shared | Shared | Customer | Customer | Customer |
| Incident Response | Shared | Shared | Customer | Customer | Customer |
| Backup/DR | Vendor | Shared | Customer | Customer | Customer |

---

## 10. Migration Paths

### 10.1 SaaS to On-Prem

1. Export current configuration
2. Deploy on-prem infrastructure
3. Migrate data (customer responsibility)
4. Configure local telemetry mode
5. Verify evidence pack generation
6. Update DPA/contracts

### 10.2 On-Prem to Enterprise Cloud

1. Verify EU residency compliance
2. Configure VPC or direct connectivity
3. Migrate configuration
4. Enable telemetry export (optional)
5. Configure enterprise features
6. Sign Enterprise DPA

---

## 11. Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-17 | CCEA Team | Initial enterprise posture note |

---

## 12. Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| DPO | _________________ | _________ | ________ |
| CISO | _________________ | _________ | ________ |
| Legal | _________________ | _________ | ________ |
| Product | _________________ | _________ | ________ |
