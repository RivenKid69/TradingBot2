# EU-Only Data Residency Enforcement Specification

**Document Type**: Technical Specification
**Version**: 1.0.0
**Last Updated**: 2025-12-16
**Owner**: Compliance Team
**GDPR Phase**: 3 - EU-Only Data Residency Enforcement
**Status**: Specification (verify via tests and deployment evidence)

> **Note**: This document specifies design requirements and implementation patterns for EU data residency enforcement. Actual enforcement status must be verified through test results, CI/CD logs, and deployment audits. This is not a claim of certified compliance.

---

## 1. Overview

This specification defines the EU-only data residency enforcement mechanisms designed for the CCEA Cloud platform. The design goal is that all personal data processing occurs within the European Union, with automated verification and fail-closed enforcement (verify via deployment audits and CI tests).

### 1.1 Design Doc Reference

- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L892` (14.3: EU residency by default)
- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1745` (6.2: EU-only drift checks mandatory)

### 1.2 Key Design Principles

| Principle | Description |
|-----------|-------------|
| **EU-Only by Default** | Data storage and processing designed to default to EU regions |
| **Fail-Closed Design** | Non-EU endpoint detection designed to block deployment/operation |
| **Continuous Verification** | Automated drift checks designed for deployment and runtime |
| **Auditable Evidence** | Machine-readable reports designed for compliance audits |

---

## 2. EU Region Definitions

### 2.1 Approved AWS Regions

```
eu-west-1      # Ireland
eu-west-2      # London
eu-west-3      # Paris
eu-central-1   # Frankfurt
eu-central-2   # Zurich
eu-north-1     # Stockholm
eu-south-1     # Milan
eu-south-2     # Spain
```

### 2.2 Approved GCP Regions

```
europe-west1   # Belgium
europe-west2   # London
europe-west3   # Frankfurt
europe-west4   # Netherlands
europe-west6   # Zurich
europe-west8   # Milan
europe-west9   # Paris
europe-north1  # Finland
europe-central2 # Warsaw
```

### 2.3 Approved Azure Regions

```
westeurope          # Netherlands
northeurope         # Ireland
germanywestcentral  # Frankfurt
francecentral       # Paris
swedencentral       # Gavle
switzerlandnorth    # Zurich
uksouth             # London
```

### 2.4 Explicitly Denied Regions

The following regions are **NOT** permitted (by design and policy):

- **US Regions**: us-east-1, us-east-2, us-west-1, us-west-2, us-gov-*
- **Asia Pacific**: ap-northeast-*, ap-southeast-*, ap-south-*, ap-east-*
- **Other**: sa-east-*, af-south-*, me-south-*, me-central-*, ca-central-*

---

## 3. Components Requiring Verification

Every deployment must verify EU residency for:

| Component | Type | Verification Method |
|-----------|------|---------------------|
| **Database Primary** | RDS/PostgreSQL | Endpoint pattern + explicit config |
| **Database Replica** | RDS/PostgreSQL | Endpoint pattern + explicit config |
| **Object Storage** | S3/GCS/Azure Blob | Region configuration |
| **Backup Storage** | S3/GCS/Azure Blob | Region configuration |
| **Cache** | Redis/ElastiCache | Endpoint pattern |
| **Email Service** | SES/SendGrid | Region configuration |
| **Logging** | CloudWatch/ELK | Endpoint + region |
| **Monitoring** | CloudWatch/Datadog | Endpoint + region |
| **Error Tracking** | Sentry | Data center location |
| **Artifact Registry** | ECR/Docker Registry | Region configuration |
| **Payment Processor** | Stripe/Adyen | Legal entity location |

---

## 4. Drift Check Implementation

### 4.1 Check Frequency

| Trigger | Frequency | Mode |
|---------|-----------|------|
| **Deployment** | Every deployment | Blocking |
| **CI/CD Pipeline** | Every PR/commit | Blocking |
| **Runtime** | Hourly | Alert + Rollback |
| **Manual** | On-demand | Report only |

### 4.2 Drift Check Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EU-ONLY DRIFT CHECK FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐                  │
│  │ Deployment  │────▶│ Load Config  │────▶│ Extract     │                  │
│  │ Triggered   │     │ (Helm/Env)   │     │ Endpoints   │                  │
│  └─────────────┘     └──────────────┘     └──────┬──────┘                  │
│                                                   │                         │
│                                                   ▼                         │
│                      ┌─────────────────────────────────────────┐           │
│                      │       FOR EACH ENDPOINT                  │           │
│                      │  ┌─────────────────────────────────┐    │           │
│                      │  │ 1. Extract region from pattern  │    │           │
│                      │  │ 2. Check explicit configuration │    │           │
│                      │  │ 3. Verify against EU allowlist  │    │           │
│                      │  │ 4. Record result + evidence     │    │           │
│                      │  └─────────────────────────────────┘    │           │
│                      └──────────────────────┬──────────────────┘           │
│                                             │                               │
│                                             ▼                               │
│                      ┌─────────────────────────────────────────┐           │
│                      │         ANY NON-EU ENDPOINTS?           │           │
│                      └───────────┬─────────────────┬───────────┘           │
│                                  │                 │                        │
│                               YES │                 │ NO                    │
│                                  ▼                 ▼                        │
│                      ┌─────────────┐     ┌─────────────┐                   │
│                      │ FAIL CLOSED │     │    PASS     │                   │
│                      │ Block Deploy│     │   Continue  │                   │
│                      └──────┬──────┘     └──────┬──────┘                   │
│                             │                   │                          │
│                             ▼                   ▼                          │
│                      ┌─────────────────────────────────────────┐           │
│                      │      GENERATE EVIDENCE REPORT           │           │
│                      │  - JSON format per Section 5            │           │
│                      │  - Store for audit                      │           │
│                      └─────────────────────────────────────────┘           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Fail-Closed Behavior

When a non-EU endpoint is detected:

1. **Deployment Phase**: Deployment is blocked
2. **Runtime Phase**: Alert triggered + automatic rollback
3. **Incident Created**: For investigation
4. **Evidence Stored**: For audit trail

---

## 5. Report Format

### 5.1 Machine-Readable JSON Report

```json
{
  "check_id": "drift-check-2025-01-15-abc12345",
  "timestamp": "2025-01-15T10:00:00Z",
  "status": "PASS",
  "checks": [
    {
      "component": "database_primary",
      "component_type": "database_primary",
      "endpoint": "xxx.eu-central-1.rds.amazonaws.com",
      "region": "eu-central-1",
      "eu_compliant": true,
      "check_method": "pattern"
    },
    {
      "component": "database_replica",
      "component_type": "database_replica",
      "endpoint": "xxx.eu-west-1.rds.amazonaws.com",
      "region": "eu-west-1",
      "eu_compliant": true,
      "check_method": "pattern"
    },
    {
      "component": "object_storage",
      "component_type": "object_storage",
      "endpoint": "platform-artifacts-eu",
      "region": "eu-central-1",
      "eu_compliant": true,
      "check_method": "explicit"
    },
    {
      "component": "cache",
      "component_type": "cache",
      "endpoint": "xxx.eu-central-1.cache.amazonaws.com",
      "region": "eu-central-1",
      "eu_compliant": true,
      "check_method": "pattern"
    },
    {
      "component": "email_service",
      "component_type": "email_service",
      "endpoint": "ses",
      "region": "eu-west-1",
      "eu_compliant": true,
      "check_method": "explicit"
    },
    {
      "component": "error_tracking",
      "component_type": "error_tracking",
      "endpoint": "sentry",
      "region": "EU (Germany)",
      "eu_compliant": true,
      "check_method": "known_service"
    },
    {
      "component": "payment_processor",
      "component_type": "payment_processor",
      "endpoint": "stripe",
      "region": "EU (Ireland)",
      "eu_compliant": true,
      "check_method": "known_processor"
    }
  ],
  "subprocessors": [
    {
      "name": "AWS",
      "legal_entity": "Amazon Web Services EMEA SARL",
      "service": "Cloud infrastructure",
      "region": "eu-central-1, eu-west-1",
      "eu_compliant": true,
      "dpa_status": "Signed"
    }
  ],
  "violations": [],
  "subprocessors_verified": 1,
  "endpoints_verified": 7,
  "non_eu_endpoints": 0,
  "next_check": "2025-01-15T11:00:00Z",
  "report_hash": "sha256:abc123...",
  "environment": "production",
  "deployment_id": "deploy-12345",
  "triggered_by": "scheduled"
}
```

### 5.2 Violation Format

When violations are detected:

```json
{
  "violations": [
    {
      "component": "database_primary",
      "violation_type": "non_eu_endpoint",
      "severity": "critical",
      "message": "Component 'database_primary' is using non-EU region: us-east-1",
      "region_found": "us-east-1",
      "region_expected": "EU",
      "remediation": "Configure database_primary to use an EU region",
      "blocked": true
    }
  ]
}
```

---

## 6. Configuration

### 6.1 Helm Values Configuration

```yaml
# deploy/helm/ccea-cloud/values.yaml

governance:
  enabled: true

  residency:
    # Enable EU-only enforcement
    enabled: true
    enforceEuOnly: true

    # Drift check configuration
    driftCheck:
      enabled: true
      schedule: "0 * * * *"  # Hourly
      failClosed: true

    # Valid regions
    allowedRegions:
      - "eu-west-1"
      - "eu-west-2"
      - "eu-central-1"
      - "eu-north-1"

    # Evidence pack storage
    evidenceStorage:
      enabled: true
      bucket: "compliance-evidence-eu"
      region: "eu-central-1"
      retentionDays: 365
```

### 6.2 Environment Variables

```bash
# Required for drift check
DATABASE_URL=postgres://xxx.eu-central-1.rds.amazonaws.com:5432/db
DATABASE_REGION=eu-central-1
DATABASE_REPLICA_URL=postgres://xxx.eu-west-1.rds.amazonaws.com:5432/db
DATABASE_REPLICA_REGION=eu-west-1

S3_BUCKET=platform-artifacts-eu
S3_REGION=eu-central-1
AWS_REGION=eu-central-1

REDIS_URL=redis://xxx.eu-central-1.cache.amazonaws.com:6379
REDIS_REGION=eu-central-1

SES_REGION=eu-west-1
EMAIL_REGION=eu-west-1

SENTRY_REGION=EU
ERROR_TRACKING_REGION=EU

# Enforcement flags
CCEA_EU_ONLY_ENFORCEMENT=true
CCEA_RESIDENCY_FAIL_CLOSED=true
```

---

## 7. CI/CD Integration

### 7.1 GitHub Actions Workflow

```yaml
# .github/workflows/residency-check.yml

name: EU Residency Check

on:
  pull_request:
    paths:
      - 'deploy/**'
      - '**/*.env'
      - '**/values*.yaml'
  push:
    branches:
      - main

jobs:
  residency-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install pyyaml

      - name: Run EU Residency Check
        run: |
          python -m ccea.guardrails.residency_check \
            --config-dir . \
            --output residency-report.json

      - name: Upload Evidence
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: residency-evidence
          path: residency-report.json

      - name: Fail on Violations
        if: failure()
        run: |
          echo "::error::EU residency check failed - non-EU endpoints detected"
          exit 1
```

### 7.2 Pre-Commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "Running EU residency check..."
python -m ccea.guardrails.residency_check --config-dir deploy/

if [ $? -ne 0 ]; then
    echo "ERROR: EU residency check failed. Commit blocked."
    exit 1
fi
```

---

## 8. Evidence Pack Integration

### 8.1 Evidence Types

| Evidence Type | Description | Retention |
|--------------|-------------|-----------|
| `drift_check_report` | Full drift check JSON report | 365 days |
| `subprocessors_summary` | Subprocessor verification summary | 365 days |
| `eu_attestation` | EU residency attestation statement | 365 days |
| `audit_log` | Drift check audit trail | 365 days |

### 8.2 Export Structure

```
evidence-pack/
├── drift_check_drift-check-2025-01-15-abc12345.json
├── subprocessors_summary.json
├── eu_residency_attestation.json
└── drift_check_audit.json
```

---

## 9. Incident Response

### 9.1 Non-EU Endpoint Detected at Deployment

1. **Automatic Action**: Deployment blocked
2. **Alert**: Sent to platform-alerts channel
3. **Investigation**: Review configuration change
4. **Remediation**: Fix configuration to use EU endpoint
5. **Verification**: Re-run drift check
6. **Documentation**: Record incident in change journal

### 9.2 Non-EU Endpoint Detected at Runtime

1. **Automatic Action**: Alert triggered
2. **Severity**: CRITICAL
3. **Response Time**: Immediate investigation required
4. **Rollback**: Automatic rollback if possible
5. **Root Cause**: Determine how non-EU endpoint was introduced
6. **Prevention**: Add additional CI checks if needed

---

## 10. Testing

### 10.1 Unit Tests

```python
# Test EU region detection
def test_is_eu_region():
    assert is_eu_region("eu-west-1") == True
    assert is_eu_region("eu-central-1") == True
    assert is_eu_region("us-east-1") == False
    assert is_eu_region("ap-northeast-1") == False

# Test endpoint extraction
def test_extract_region_from_endpoint():
    checker = EUOnlyDriftChecker()

    # AWS RDS
    result = checker.check_endpoint(
        "mydb.eu-central-1.rds.amazonaws.com",
        "database",
        ComponentType.DATABASE_PRIMARY
    )
    assert result.region == "eu-central-1"
    assert result.eu_compliant == True

    # Non-EU endpoint
    result = checker.check_endpoint(
        "mydb.us-east-1.rds.amazonaws.com",
        "database",
        ComponentType.DATABASE_PRIMARY
    )
    assert result.region == "us-east-1"
    assert result.eu_compliant == False
```

### 10.2 Integration Tests

```python
def test_full_drift_check_pass():
    config = ResidencyConfiguration(
        database_primary_endpoint="mydb.eu-central-1.rds.amazonaws.com",
        database_primary_region="eu-central-1",
        object_storage_region="eu-central-1",
        cache_region="eu-central-1",
    )

    checker = EUOnlyDriftChecker()
    report = checker.check(config)

    assert report.passed == True
    assert report.non_eu_endpoints == 0
    assert report.status == DriftCheckStatus.PASS

def test_full_drift_check_fail():
    config = ResidencyConfiguration(
        database_primary_endpoint="mydb.us-east-1.rds.amazonaws.com",
        database_primary_region="us-east-1",
    )

    checker = EUOnlyDriftChecker()
    report = checker.check(config)

    assert report.passed == False
    assert report.non_eu_endpoints > 0
    assert report.status == DriftCheckStatus.FAIL
    assert len(report.blocking_violations) > 0
```

---

## 11. API Reference

### 11.1 EUOnlyDriftChecker

```python
class EUOnlyDriftChecker:
    """EU-Only Data Residency Drift Checker."""

    def __init__(
        self,
        fail_closed: bool = True,
        check_dns: bool = False,
        allowed_regions: Optional[FrozenSet[str]] = None,
    ): ...

    def check(
        self,
        config: ResidencyConfiguration,
        subprocessors: Optional[List[SubprocessorCheck]] = None,
        environment: str = "production",
        deployment_id: str = "",
        triggered_by: str = "scheduled",
    ) -> ResidencyDriftReport: ...

    def check_endpoint(
        self,
        endpoint: str,
        component: str,
        component_type: ComponentType,
        explicit_region: str = "",
    ) -> EndpointCheck: ...
```

### 11.2 Convenience Functions

```python
def check_eu_residency(
    config: Optional[ResidencyConfiguration] = None,
    fail_closed: bool = True,
) -> ResidencyDriftReport:
    """Check EU residency from environment or config."""

def is_eu_region(region: str) -> bool:
    """Check if region is EU-based."""

def validate_endpoint_eu(endpoint: str) -> Tuple[bool, str]:
    """Validate endpoint is EU-based."""
```

---

## 12. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | Compliance Team | Initial release - GDPR Phase 3 |

---

## 13. References

- GDPR Regulation (EU) 2016/679 - Articles 44-49 (International Transfers)
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`
- `docs/compliance/SUBPROCESSORS_REGISTER.md`
- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt`
- `packages/cloud/governance/residency_drift.py`
- `ccea/guardrails/residency_check.py`
