# RAW_ORDER_EVENTS Handling Specification (Enterprise Only)

**Document Version**: 1.0.0
**Effective Date**: 2025-12-16
**Classification**: INTERNAL / COMPLIANCE / ENTERPRISE
**Related Documents**:
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`
- `docs/compliance/TELEMETRY_DATA_DICTIONARY.md`
- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt`

## 1. Overview

This specification defines the handling requirements for `RAW_ORDER_EVENTS` telemetry level in the CCEA architecture. This level is **enterprise-only** and requires explicit opt-in due to the sensitive nature of raw trading data.

**Reference**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L853`, `#L1739`

### 1.1 Purpose

`RAW_ORDER_EVENTS` enables enterprise customers to:

1. **Compliance Audit Trail**: Designed to support client regulatory requirements (MiFID II, SEC, FINRA); clients must conduct their own compliance assessment with qualified advisors
2. **Forensic Analysis**: Investigate trading anomalies with full detail
3. **Latency Analysis**: Measure end-to-end execution timing
4. **Risk Monitoring**: Real-time position and P&L tracking

### 1.2 Risk Classification

| Risk | Level | Mitigation |
|------|-------|------------|
| IP Exposure | HIGH | Strict access control + encryption |
| Data Breach Impact | HIGH | Isolated storage + minimized retention |
| Regulatory | MEDIUM | EU-only residency + audit trail |
| Operational | MEDIUM | Break-glass with reason + audit |

---

## 2. Enterprise Gating Requirements

### 2.1 Prerequisites Checklist

All prerequisites MUST be satisfied before RAW_ORDER_EVENTS can be enabled:

```
[ ] ENTERPRISE LICENSE
    - Active enterprise subscription (license_type = ENTERPRISE)
    - License not expired (expiry > now)
    - RAW_ORDER_EVENTS feature flag enabled in license

[ ] LEGAL AGREEMENT
    - Enterprise DPA signed
    - RAW Data Processing Addendum signed
    - Legal contact designated for breach notification

[ ] EXPLICIT OPT-IN
    - Workspace-level opt-in recorded
    - Opt-in timestamp captured
    - Opt-in principal (who authorized) recorded
    - Opt-in acknowledgment hash stored

[ ] ACCESS CONTROLS
    - RBAC configured for workspace
    - RAW_DATA_ACCESS permission assigned only to authorized users
    - Break-glass procedure documented
    - Access audit enabled

[ ] RETENTION POLICY
    - Custom retention configured (<= 30 days default)
    - Auto-purge enabled
    - Retention policy acknowledged by customer

[ ] ENCRYPTION (Optional but recommended)
    - Customer-managed keys (CMK) configured
    - Key rotation policy in place
    - Key escrow documented
```

### 2.2 Gating Verification Flow

```
┌─────────────────┐
│ Telemetry       │
│ Received        │
│ (RAW level)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Check           │──NO──▶ REJECT (HTTP 422)
│ Enterprise      │        "Enterprise license required"
│ License?        │
└────────┬────────┘
         │YES
         ▼
┌─────────────────┐
│ Check           │──NO──▶ REJECT (HTTP 422)
│ Explicit        │        "RAW opt-in required"
│ Opt-In?         │
└────────┬────────┘
         │YES
         ▼
┌─────────────────┐
│ Check           │──NO──▶ REJECT (HTTP 403)
│ Workspace       │        "RAW not enabled for workspace"
│ Enabled?        │
└────────┬────────┘
         │YES
         ▼
┌─────────────────┐
│ Validate        │──FAIL──▶ REJECT (HTTP 422)
│ Payload         │          "Validation failed"
│ Fields          │
└────────┬────────┘
         │PASS
         ▼
┌─────────────────┐
│ Log Audit       │
│ Event           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Store in        │
│ Enterprise      │
│ Isolated        │
│ Storage         │
└─────────────────┘
```

---

## 3. Cloud Ingestion Gates

### 3.1 Gate Implementation

```python
class RawOrderEventsGate:
    """
    Gate for RAW_ORDER_EVENTS telemetry ingestion.

    ENTERPRISE ONLY - requires explicit opt-in.
    """

    def check_enterprise_license(
        self,
        workspace_id: str,
        organization_id: str,
    ) -> GateResult:
        """
        Verify enterprise license is active and includes RAW feature.

        Returns:
            GateResult with is_allowed and reason
        """
        ...

    def check_explicit_opt_in(
        self,
        workspace_id: str,
    ) -> GateResult:
        """
        Verify explicit opt-in exists for workspace.

        Checks:
            - Opt-in record exists
            - Opt-in not expired
            - Opt-in principal authorized

        Returns:
            GateResult with is_allowed and reason
        """
        ...

    def check_workspace_enabled(
        self,
        workspace_id: str,
    ) -> GateResult:
        """
        Verify workspace has RAW_ORDER_EVENTS enabled.

        Returns:
            GateResult with is_allowed and reason
        """
        ...

    def validate_and_ingest(
        self,
        workspace_id: str,
        payload: Dict[str, Any],
    ) -> IngestResult:
        """
        Full validation and ingestion flow.

        Steps:
            1. Check enterprise license
            2. Check explicit opt-in
            3. Check workspace enabled
            4. Validate payload fields
            5. Log audit event
            6. Store in enterprise storage

        Returns:
            IngestResult with success status and any violations
        """
        ...
```

### 3.2 Audit Events

Every RAW_ORDER_EVENTS operation generates an audit event:

```json
{
  "event_type": "RAW_ORDER_EVENTS_INGESTION",
  "timestamp": "2025-12-16T10:30:00.000Z",
  "workspace_id": "ws_enterprise_123",
  "organization_id": "org_456",
  "agent_id": "agent_abc123def456",
  "telemetry_level": "RAW_ORDER_EVENTS",
  "action": "INGEST",
  "result": "SUCCESS",
  "record_count": 150,
  "fields_present": ["side", "quantity", "price", "order_id"],
  "enterprise_verification": {
    "license_verified": true,
    "opt_in_verified": true,
    "workspace_enabled": true,
    "verification_timestamp": "2025-12-16T10:30:00.000Z"
  },
  "storage": {
    "location": "enterprise_isolated",
    "encrypted": true,
    "retention_days": 14
  }
}
```

---

## 4. Agent-Local Export Path ("Telemetry Stays Local")

Enterprise customers may choose to keep RAW telemetry local instead of Cloud ingestion.

### 4.1 Configuration

```yaml
# Agent configuration: telemetry_stays_local mode
telemetry:
  level: RAW_ORDER_EVENTS
  destination: LOCAL

  local_storage:
    enabled: true
    path: /data/telemetry/raw
    format: parquet  # or json, csv
    encryption:
      enabled: true
      key_source: CUSTOMER_KMS
      key_id: "arn:aws:kms:eu-west-1:123:key/abc"

    rotation:
      max_file_size_mb: 100
      max_age_hours: 24

    retention:
      max_days: 30
      auto_purge: true

  # Cloud receives AGGREGATED only when local mode is enabled
  cloud_fallback:
    level: AGGREGATED
```

### 4.2 Local Storage Requirements

| Requirement | Specification |
|-------------|---------------|
| Format | Parquet (preferred), JSON, CSV |
| Encryption | AES-256-GCM or customer KMS |
| Compression | Snappy or ZSTD |
| Partitioning | By date (YYYY/MM/DD) |
| Indexing | By order_id, timestamp |
| Access Logging | All reads logged |

### 4.3 Export API

```python
class LocalTelemetryExporter:
    """
    Export RAW telemetry to local customer-controlled storage.

    ENTERPRISE ONLY - for "telemetry stays local" mode.
    """

    async def export_batch(
        self,
        events: List[TelemetryEvent],
        destination: LocalStorageConfig,
    ) -> ExportResult:
        """
        Export batch of events to local storage.

        Steps:
            1. Validate destination writable
            2. Encrypt payload (if configured)
            3. Write to destination
            4. Log export audit event
            5. Return success/failure

        Returns:
            ExportResult with file path and record count
        """
        ...

    async def rotate_files(
        self,
        config: LocalStorageConfig,
    ) -> RotationResult:
        """
        Rotate local files based on size/age policy.
        """
        ...

    async def purge_expired(
        self,
        config: LocalStorageConfig,
    ) -> PurgeResult:
        """
        Purge files older than retention period.
        """
        ...
```

---

## 5. Retention Policy

### 5.1 Default Retention

| Telemetry Level | Default Retention | Max Retention |
|-----------------|-------------------|---------------|
| AGGREGATED | 90 days | 365 days |
| DETAILED_NON_SENSITIVE | 30 days | 90 days |
| RAW_ORDER_EVENTS | 14 days | 30 days |

### 5.2 Retention Configuration

```python
@dataclass
class RawOrderRetentionPolicy:
    """
    Retention policy for RAW_ORDER_EVENTS.

    ENTERPRISE ONLY with strict limits.
    """

    workspace_id: str

    # Retention period (must be <= 30 days)
    retention_days: int = field(
        default=14,
        metadata={"max": 30, "min": 1}
    )

    # Auto-purge settings
    auto_purge_enabled: bool = True
    purge_schedule: str = "0 2 * * *"  # 2 AM daily

    # Legal hold support
    legal_hold_enabled: bool = False
    legal_hold_expiry: Optional[datetime] = None
    legal_hold_reason: Optional[str] = None

    # Anonymization option
    anonymize_before_delete: bool = False

    # Audit requirements
    purge_audit_enabled: bool = True
```

### 5.3 Auto-Purge Implementation

```python
class RawOrderPurgeJob:
    """
    Scheduled job for purging expired RAW_ORDER_EVENTS.

    Runs on schedule, respects legal hold.
    """

    async def run_purge(
        self,
        workspace_id: str,
    ) -> PurgeResult:
        """
        Execute purge for workspace.

        Steps:
            1. Check legal hold status
            2. Identify expired records
            3. Optionally anonymize
            4. Delete records
            5. Log purge audit event
            6. Return counts

        Legal hold blocks deletion but still logs attempt.

        Returns:
            PurgeResult with deleted/skipped counts
        """
        ...
```

---

## 6. Access Controls

### 6.1 Required Permissions

| Permission | Description | Default Role |
|------------|-------------|--------------|
| `raw_order:read` | Read RAW telemetry | None (explicit grant) |
| `raw_order:export` | Export RAW data | Compliance Officer |
| `raw_order:admin` | Manage RAW settings | Workspace Admin |
| `raw_order:break_glass` | Emergency access | Security Team |

### 6.2 RBAC Configuration

```python
RAW_ORDER_PERMISSIONS = {
    "raw_order:read": {
        "description": "Read RAW_ORDER_EVENTS telemetry",
        "default_grant": False,
        "requires_justification": True,
        "audit_level": "HIGH",
    },
    "raw_order:export": {
        "description": "Export RAW_ORDER_EVENTS data",
        "default_grant": False,
        "requires_justification": True,
        "requires_approval": True,
        "audit_level": "CRITICAL",
    },
    "raw_order:admin": {
        "description": "Administer RAW_ORDER_EVENTS settings",
        "default_grant": False,
        "restricted_to_roles": ["WORKSPACE_ADMIN", "SECURITY_ADMIN"],
        "audit_level": "CRITICAL",
    },
    "raw_order:break_glass": {
        "description": "Emergency access to RAW_ORDER_EVENTS",
        "default_grant": False,
        "requires_reason": True,
        "time_limited": True,
        "max_duration_hours": 4,
        "requires_two_person": True,
        "audit_level": "CRITICAL",
    },
}
```

### 6.3 Break-Glass Procedure

```
┌─────────────────────────────────────────────────────────────┐
│                    BREAK-GLASS PROCEDURE                     │
│                  RAW_ORDER_EVENTS ACCESS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. INCIDENT REQUIREMENT                                     │
│     - Active security incident OR                            │
│     - Regulatory investigation OR                            │
│     - Customer-authorized forensic analysis                  │
│                                                              │
│  2. REQUEST                                                  │
│     - Submit break-glass request via security portal         │
│     - Provide: incident ID, scope, reason, duration needed   │
│     - Duration must be <= 4 hours                            │
│                                                              │
│  3. APPROVAL                                                 │
│     - Requires TWO approvers:                                │
│       * Security team member                                 │
│       * Engineering manager or above                         │
│     - Approval recorded with timestamp + principal           │
│                                                              │
│  4. ACCESS GRANTED                                           │
│     - Time-limited token issued                              │
│     - Scope limited to specified workspace                   │
│     - All access logged in real-time                         │
│                                                              │
│  5. ACCESS REVOKED                                           │
│     - Automatic at expiry                                    │
│     - Manual revocation available                            │
│     - Post-access review required                            │
│                                                              │
│  6. AUDIT TRAIL                                              │
│     - Immutable log of:                                      │
│       * Request details                                      │
│       * Approvers                                            │
│       * Access start/end                                     │
│       * All queries executed                                 │
│       * Data exported (if any)                               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Encryption Requirements

### 7.1 Encryption at Rest

| Component | Encryption | Key Management |
|-----------|------------|----------------|
| Enterprise Storage | AES-256-GCM | CMK or Platform-managed |
| Backup Storage | AES-256-GCM | CMK or Platform-managed |
| Export Files | AES-256-GCM | Recipient public key |

### 7.2 Customer-Managed Keys (CMK)

```python
@dataclass
class CMKConfiguration:
    """
    Customer-managed key configuration for RAW_ORDER_EVENTS.
    """

    workspace_id: str

    # Key provider
    provider: Literal["AWS_KMS", "AZURE_KEY_VAULT", "GCP_KMS", "HASHICORP_VAULT"]

    # Key reference (provider-specific)
    key_id: str

    # Key usage
    allowed_operations: List[Literal["ENCRYPT", "DECRYPT"]]

    # Rotation
    rotation_enabled: bool = True
    rotation_period_days: int = 90

    # Escrow (optional)
    escrow_enabled: bool = False
    escrow_key_id: Optional[str] = None
```

---

## 8. Validation Rules (RAW Level)

### 8.1 Allowed Fields

See `docs/compliance/TELEMETRY_DATA_DICTIONARY.md` Section 5 for complete list.

### 8.2 Still Forbidden Fields

Even with RAW_ORDER_EVENTS, the following are **designed to be FORBIDDEN** (verify via redaction tests):

```python
ALWAYS_FORBIDDEN_FIELDS = frozenset({
    # Credentials
    "api_key", "api_secret", "secret_key", "private_key",
    "access_token", "refresh_token", "bearer_token",
    "password", "passphrase",

    # Broker-specific credentials
    "binance_key", "binance_secret",
    "alpaca_key", "alpaca_secret",
    "deribit_key", "deribit_secret",

    # Environment variables (pattern-matched)
    # AWS_*, AZURE_*, GCP_*, DATABASE_*, *_PASSWORD, *_TOKEN, *_SECRET
})
```

### 8.3 Validation Process

```python
def validate_raw_order_payload(
    payload: Dict[str, Any],
    workspace_config: WorkspaceConfig,
) -> ValidationResult:
    """
    Validate RAW_ORDER_EVENTS payload.

    Steps:
        1. Verify enterprise + opt-in (done by gate)
        2. Scan for ALWAYS_FORBIDDEN fields
        3. Scan for credential patterns
        4. Validate field types
        5. Return result

    CRITICAL: Credentials are designed to be rejected at all telemetry levels, including RAW (enforced via redaction middleware and schema validation; verify via CI redaction tests).
    """
    ...
```

---

## 9. Evidence Pack Export

RAW_ORDER_EVENTS can be included in evidence pack exports for enterprise audits.

### 9.1 Export Types

| Export Type | Contents | Requires |
|-------------|----------|----------|
| `RAW_TELEMETRY` | RAW telemetry events | `raw_order:export` permission |
| `RAW_AUDIT_TRAIL` | Audit logs of RAW access | `raw_order:admin` permission |
| `RAW_RETENTION_LOG` | Purge audit records | `raw_order:admin` permission |

### 9.2 Export Format

```json
{
  "export_metadata": {
    "export_id": "export_2025121_abc",
    "export_type": "RAW_TELEMETRY",
    "workspace_id": "ws_enterprise_123",
    "requestor": "user@company.com",
    "export_timestamp": "2025-12-16T10:30:00Z",
    "time_range": {
      "start": "2025-12-01T00:00:00Z",
      "end": "2025-12-16T00:00:00Z"
    },
    "record_count": 15000,
    "checksum": "sha256:abc123...",
    "encryption": {
      "algorithm": "AES-256-GCM",
      "key_reference": "export_key_xyz"
    }
  },
  "data": [
    // Encrypted telemetry records
  ]
}
```

---

## 10. Monitoring and Alerting

### 10.1 Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `raw_order_ingestion_rate` | Events/second | > 10000 |
| `raw_order_storage_gb` | Storage used | > 80% quota |
| `raw_order_access_count` | Access attempts | Unusual pattern |
| `raw_order_export_count` | Export operations | Any (alert on each) |
| `raw_order_break_glass_count` | Break-glass uses | Any (alert on each) |

### 10.2 Alert Configuration

```yaml
alerts:
  - name: raw_order_unusual_access
    description: Unusual RAW_ORDER_EVENTS access pattern
    condition: |
      rate(raw_order_access_count[5m]) >
      avg_over_time(raw_order_access_count[7d]) * 3
    severity: HIGH
    notify:
      - security-team@ccea.io
      - on-call-pager

  - name: raw_order_break_glass_used
    description: Break-glass access to RAW_ORDER_EVENTS
    condition: raw_order_break_glass_count > 0
    severity: CRITICAL
    notify:
      - security-team@ccea.io
      - compliance@ccea.io
      - on-call-pager

  - name: raw_order_export_triggered
    description: RAW_ORDER_EVENTS export initiated
    condition: raw_order_export_count > 0
    severity: HIGH
    notify:
      - security-team@ccea.io
      - workspace-admin
```

---

## 11. Implementation Checklist

### 11.1 Cloud Components

- [ ] `RawOrderEventsGate` - Enterprise gating
- [ ] `RawOrderValidator` - Payload validation
- [ ] `RawOrderStorage` - Isolated storage
- [ ] `RawOrderAuditLogger` - Audit trail
- [ ] `RawOrderPurgeJob` - Auto-purge
- [ ] `RawOrderExporter` - Evidence pack export

### 11.2 Agent Components

- [ ] `LocalTelemetryExporter` - Local storage mode
- [ ] `LocalStorageRotator` - File rotation
- [ ] `LocalPurgeJob` - Local purge

### 11.3 Tests

- [ ] Gate enforcement tests
- [ ] Validation tests
- [ ] Storage tests
- [ ] Retention/purge tests
- [ ] Access control tests
- [ ] Break-glass workflow tests
- [ ] Export tests

---

## 12. Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial version for GDPR Phase 2 |
