# Data Retention Policy Specification

**Document Type**: GDPR Compliance Engineering Specification
**Version**: 1.0
**Last Updated**: 2025-12-16
**GDPR Reference**: Article 5(1)(e) - Storage Limitation
**Phase**: GDPR Implementation Phase 4

---

## Executive Summary

This specification defines the data retention policy framework for the CCEA Cloud Platform, implementing GDPR Article 5(1)(e) - Storage Limitation principle. The framework provides:

1. **Retention Schedule** - Per-data-type retention periods with regulatory justification
2. **Auto-Purge Mechanism** - Automated scheduled deletion with auditable events
3. **Legal Hold** - Litigation preservation with strict access control
4. **Tenant Customization** - Per-workspace policy overrides within compliance bounds

---

## 1. Retention Schedule by Data Category

### 1.1 Retention Period Matrix

| Data Category | Default Retention | Min Retention | Max Retention | Justification | GDPR Basis |
|---------------|-------------------|---------------|---------------|---------------|------------|
| **Session Tokens** | 24 hours | 1 hour | 7 days | Active session management | Art. 6(1)(b) Contract |
| **RAW_ORDER_EVENTS Telemetry** | 7 days | 1 day | 30 days | Enterprise debugging (minimized) | Art. 6(1)(a) Consent |
| **DETAILED_NON_SENSITIVE Telemetry** | 30 days | 7 days | 90 days | Technical debugging | Art. 6(1)(b) Contract |
| **AGGREGATED Telemetry** | 90 days | 30 days | 365 days | Operational monitoring | Art. 6(1)(b) Contract |
| **Application Logs** | 30 days | 7 days | 90 days | Incident response | Art. 6(1)(f) Legitimate Interest |
| **Distributed Traces** | 7 days | 1 day | 30 days | Performance analysis | Art. 6(1)(f) Legitimate Interest |
| **Alerts (Resolved)** | 365 days | 30 days | 730 days | Operational history | Art. 6(1)(f) Legitimate Interest |
| **Commands (Terminal State)** | 180 days | 30 days | 730 days | Audit trail | Art. 6(1)(c) Legal Obligation |
| **Config Blobs** | Indefinite | 365 days | Indefinite | Version history | Art. 6(1)(b) Contract |
| **Strategy Versions** | Customer-defined | Indefinite | Indefinite | Customer IP (Processor) | Art. 28 Processor |
| **Backtest Results** | Customer-defined | 30 days | Indefinite | Research data (Processor) | Art. 28 Processor |
| **Approval Records** | 7 years | 7 years | 10 years | Financial compliance | Art. 6(1)(c) Legal Obligation |
| **Access Audits** | 7 years | 7 years | 10 years | Security compliance | Art. 6(1)(c) Legal Obligation |
| **Break-Glass Requests** | 7 years | 7 years | 10 years | Incident accountability | Art. 6(1)(c) Legal Obligation |
| **DSAR Requests** | 7 years | 7 years | 10 years | GDPR compliance proof | Art. 6(1)(c) Legal Obligation |
| **Governance Audit Logs** | 7 years | 7 years | 10 years | Data lifecycle audit | Art. 6(1)(c) Legal Obligation |
| **Legal Hold Records** | 7 years + hold duration | 7 years | Indefinite | Litigation preservation | Art. 6(1)(c) Legal Obligation |
| **User Identity** | Account lifetime + 90 days | 90 days post-deletion | 365 days post-deletion | Recovery window | Art. 6(1)(b) Contract |
| **Billing Records** | 7 years | 7 years | 10 years | Tax/accounting | Art. 6(1)(c) Legal Obligation |

### 1.2 Data Category IDs (System Reference)

```python
class DataCategory:
    """Canonical data category identifiers for retention policies."""

    # Telemetry (by sensitivity level)
    TELEMETRY_RAW = "telemetry_raw_order_events"
    TELEMETRY_DETAILED = "telemetry_detailed_non_sensitive"
    TELEMETRY_AGGREGATED = "telemetry_aggregated"

    # Operational Data
    ALERTS = "alerts"
    COMMANDS = "commands"
    CONFIG_BLOBS = "config_blobs"
    DEPLOYMENT_DATA = "deployment_data"
    RUN_DATA = "run_data"

    # User/Account Data
    USER_IDENTITY = "user_identity"
    USER_SETTINGS = "user_settings"
    SESSION_DATA = "session_data"

    # Audit/Compliance Data (7-year minimum)
    APPROVAL_RECORDS = "approval_records"
    ACCESS_AUDITS = "access_audits"
    BREAK_GLASS_REQUESTS = "break_glass_requests"
    DSAR_REQUESTS = "dsar_requests"
    GOVERNANCE_AUDIT_LOGS = "governance_audit_logs"
    LEGAL_HOLD_RECORDS = "legal_hold_records"
    BILLING_RECORDS = "billing_records"

    # Customer-Controlled Data
    STRATEGY_VERSIONS = "strategy_versions"
    BACKTEST_RESULTS = "backtest_results"
    RESEARCH_ARTIFACTS = "research_artifacts"

    # Infrastructure Data
    APPLICATION_LOGS = "application_logs"
    DISTRIBUTED_TRACES = "distributed_traces"
    METRICS = "metrics"
```

---

## 2. Retention Actions

### 2.1 Action Types

| Action | Description | Use Cases |
|--------|-------------|-----------|
| **DELETE** | Permanent removal from all storage | Default for most data types |
| **ARCHIVE** | Move to cold storage with restricted access | Compliance data after active period |
| **ANONYMIZE** | Remove/mask PII while preserving structure | Analytics data preservation |
| **AGGREGATE** | Convert to statistical summaries | Telemetry after granular period |

### 2.2 Action Mapping by Category

```python
DEFAULT_RETENTION_ACTIONS = {
    "telemetry_raw_order_events": RetentionAction.DELETE,
    "telemetry_detailed_non_sensitive": RetentionAction.DELETE,
    "telemetry_aggregated": RetentionAction.AGGREGATE,
    "alerts": RetentionAction.DELETE,
    "commands": RetentionAction.ARCHIVE,
    "approval_records": RetentionAction.ARCHIVE,
    "access_audits": RetentionAction.ARCHIVE,
    "break_glass_requests": RetentionAction.ARCHIVE,
    "application_logs": RetentionAction.DELETE,
    "session_data": RetentionAction.DELETE,
    "user_identity": RetentionAction.ANONYMIZE,
}
```

---

## 3. Auto-Purge Scheduler

### 3.1 Scheduler Configuration

```yaml
# Default scheduler settings
auto_purge:
  enabled: true
  schedule:
    interval_hours: 24          # Run every 24 hours
    preferred_time_utc: "03:00" # Run at 3 AM UTC (low-traffic)
    jitter_minutes: 30          # Random jitter to spread load

  execution:
    batch_size: 1000            # Records per batch
    max_runtime_minutes: 60     # Per-workspace timeout
    parallel_workspaces: 5      # Concurrent workspace processing

  safety:
    dry_run: false              # Set true for testing
    require_confirmation: false # Manual confirmation (enterprise)
    notify_before_days: 7       # Warning before large purges
    min_records_for_notify: 10000

  retry:
    max_attempts: 3
    backoff_seconds: 300
```

### 3.2 Purge Job Workflow

```
┌──────────────────────────────────────────────────────────────────────┐
│                        AUTO-PURGE WORKFLOW                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐                                                  │
│  │  Scheduler Tick │ (Every 24h at 03:00 UTC ± jitter)               │
│  └────────┬────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  1. Get All Workspaces with Retention Policies                   │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼ (for each workspace)                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  2. Get Retention Policies for Workspace                         │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼ (for each policy)                                          │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  3. Check Prerequisites                                          │  │
│  │     ├── Policy enabled?                                          │  │
│  │     ├── Legal hold active?  ───────────────────────┐             │  │
│  │     └── Auto-purge enabled?                        │             │  │
│  └────────┬───────────────────────────────────────────┼─────────────┘  │
│           │ YES                                       │ NO (skip)     │
│           ▼                                           ▼               │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  4. Calculate Cutoff Date                                        │  │
│  │     cutoff = now() - retention_days                              │  │
│  │     For compliance data: enforce min(cutoff, 7_years_ago)        │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  5. Execute Purge in Batches                                     │  │
│  │     WHILE records_remaining AND runtime < max_runtime:           │  │
│  │       ├── SELECT ids WHERE created_at < cutoff LIMIT batch_size  │  │
│  │       ├── Execute retention_action (DELETE/ARCHIVE/ANONYMIZE)    │  │
│  │       ├── Update counters                                        │  │
│  │       └── Flush/commit batch                                     │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  6. Create Audit Event                                           │  │
│  │     {                                                            │  │
│  │       "event_type": "purge_completed",                           │  │
│  │       "workspace_id": "...",                                     │  │
│  │       "data_type": "...",                                        │  │
│  │       "retention_days": N,                                       │  │
│  │       "cutoff_date": "ISO8601",                                  │  │
│  │       "records_deleted": X,                                      │  │
│  │       "records_archived": Y,                                     │  │
│  │       "records_anonymized": Z,                                   │  │
│  │       "duration_seconds": D,                                     │  │
│  │       "timestamp": "ISO8601",                                    │  │
│  │       "executor": "scheduler"                                    │  │
│  │     }                                                            │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  7. Update Policy Metadata                                       │  │
│  │     last_purge_at = now()                                        │  │
│  │     last_purge_count = records_processed                         │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 Purge Audit Event Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "PurgeAuditEvent",
  "type": "object",
  "required": [
    "event_id",
    "event_type",
    "workspace_id",
    "data_type",
    "timestamp",
    "status"
  ],
  "properties": {
    "event_id": {
      "type": "string",
      "format": "uuid"
    },
    "event_type": {
      "type": "string",
      "enum": ["purge_started", "purge_completed", "purge_failed", "purge_skipped"]
    },
    "workspace_id": {
      "type": "string",
      "format": "uuid"
    },
    "data_type": {
      "type": "string"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "status": {
      "type": "string",
      "enum": ["success", "failed", "skipped", "partial"]
    },
    "retention_config": {
      "type": "object",
      "properties": {
        "retention_days": {"type": "integer"},
        "action": {"type": "string"},
        "cutoff_date": {"type": "string", "format": "date-time"}
      }
    },
    "results": {
      "type": "object",
      "properties": {
        "records_deleted": {"type": "integer", "minimum": 0},
        "records_archived": {"type": "integer", "minimum": 0},
        "records_anonymized": {"type": "integer", "minimum": 0},
        "records_aggregated": {"type": "integer", "minimum": 0},
        "bytes_freed": {"type": "integer", "minimum": 0}
      }
    },
    "execution": {
      "type": "object",
      "properties": {
        "started_at": {"type": "string", "format": "date-time"},
        "completed_at": {"type": "string", "format": "date-time"},
        "duration_seconds": {"type": "number"},
        "batches_processed": {"type": "integer"},
        "executor": {"type": "string"}
      }
    },
    "skip_reason": {
      "type": "string",
      "description": "Reason for skipping (if status=skipped)"
    },
    "error": {
      "type": "object",
      "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "recoverable": {"type": "boolean"}
      }
    }
  }
}
```

---

## 4. Legal Hold

### 4.1 Legal Hold Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                        LEGAL HOLD LIFECYCLE                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐                                                  │
│  │  Request Legal  │ (by authorized user: admin/legal)               │
│  │      Hold       │                                                  │
│  └────────┬────────┘                                                  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Validation                                                      │  │
│  │  ├── User has legal_hold:create permission?                      │  │
│  │  ├── Workspace exists?                                           │  │
│  │  ├── Data type valid?                                            │  │
│  │  ├── Reason provided? (REQUIRED)                                 │  │
│  │  └── Hold duration specified? (optional, indefinite if not)      │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Create Legal Hold Record                                        │  │
│  │  {                                                               │  │
│  │    "id": "uuid",                                                 │  │
│  │    "workspace_id": "uuid",                                       │  │
│  │    "data_type": "telemetry_aggregated",                          │  │
│  │    "reason": "Litigation case #12345",                           │  │
│  │    "hold_until": "2026-12-31T23:59:59Z" | null,                  │  │
│  │    "created_by": "user_id",                                      │  │
│  │    "created_at": "2025-12-16T10:00:00Z",                         │  │
│  │    "is_active": true                                             │  │
│  │  }                                                               │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Create Audit Event                                              │  │
│  │  {                                                               │  │
│  │    "action": "legal_hold_created",                               │  │
│  │    "actor_id": "user_id",                                        │  │
│  │    "workspace_id": "uuid",                                       │  │
│  │    "data_type": "...",                                           │  │
│  │    "reason": "...",                                              │  │
│  │    "hold_until": "..."                                           │  │
│  │  }                                                               │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           │                   LEGAL HOLD ACTIVE                        │
│           │  ┌─────────────────────────────────────────────────────┐  │
│           │  │  Effects:                                            │  │
│           │  │  • Auto-purge SKIPPED for this data_type            │  │
│           │  │  • DSAR erasure BLOCKED for this data_type          │  │
│           │  │  • Manual deletion BLOCKED without release          │  │
│           │  │  • Access audit ENHANCED for held data              │  │
│           │  └─────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Release Legal Hold                                              │  │
│  │  (Triggered by: manual release OR hold_until expiry)             │  │
│  │                                                                  │  │
│  │  Requirements:                                                   │  │
│  │  ├── User has legal_hold:release permission                     │  │
│  │  ├── Release reason provided (REQUIRED)                         │  │
│  │  └── Confirmation if hold was indefinite                        │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Update Legal Hold Record                                        │  │
│  │  {                                                               │  │
│  │    "is_active": false,                                           │  │
│  │    "released_at": "2025-12-16T15:00:00Z",                        │  │
│  │    "released_by": "user_id",                                     │  │
│  │    "release_reason": "Litigation resolved"                       │  │
│  │  }                                                               │  │
│  └────────┬────────────────────────────────────────────────────────┘  │
│           │                                                            │
│           ▼                                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │  Create Audit Event                                              │  │
│  │  {                                                               │  │
│  │    "action": "legal_hold_released",                              │  │
│  │    "actor_id": "user_id",                                        │  │
│  │    "workspace_id": "uuid",                                       │  │
│  │    "data_type": "...",                                           │  │
│  │    "release_reason": "...",                                      │  │
│  │    "hold_duration_days": N                                       │  │
│  │  }                                                               │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                        │
│  POST-RELEASE:                                                         │
│  • Normal retention policies RESUME                                    │
│  • Next scheduled purge will process held data                         │
│  • DSAR erasure becomes available for this data_type                   │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Legal Hold Permissions

| Permission | Description | Required Roles |
|------------|-------------|----------------|
| `legal_hold:create` | Create new legal holds | `admin`, `legal`, `compliance_officer` |
| `legal_hold:view` | View legal hold status | `admin`, `legal`, `compliance_officer`, `workspace_admin` |
| `legal_hold:release` | Release active legal holds | `admin`, `legal` (NOT compliance_officer alone) |
| `legal_hold:audit` | View legal hold audit logs | `admin`, `legal`, `auditor` |

### 4.3 Legal Hold Audit Requirements

Every legal hold operation MUST produce an audit event:

- `legal_hold_created` - New hold established
- `legal_hold_extended` - Hold duration extended
- `legal_hold_released` - Hold removed
- `legal_hold_expired` - Hold expired automatically
- `legal_hold_blocked_purge` - Purge blocked by active hold
- `legal_hold_blocked_dsar` - DSAR erasure blocked by active hold

---

## 5. Tenant Policy Customization

### 5.1 Policy Override Rules

Tenants can customize retention within bounds:

```python
def validate_retention_override(
    data_type: str,
    requested_days: int,
    tenant_tier: str,
) -> tuple[bool, int]:
    """
    Validate and adjust tenant retention override.

    Returns:
        (is_valid, effective_days)
    """
    config = RETENTION_BOUNDS[data_type]

    # Compliance data: CANNOT reduce below minimum
    if config["compliance_required"]:
        effective = max(requested_days, config["min_days"])
    else:
        # Non-compliance data: can reduce to minimum
        effective = max(requested_days, config["min_days"])

    # Enterprise tier: can extend beyond default
    if tenant_tier != "enterprise" and effective > config["default_days"] * 2:
        effective = config["default_days"] * 2

    # Hard maximum
    if config["max_days"] is not None:
        effective = min(effective, config["max_days"])

    return (effective == requested_days, effective)
```

### 5.2 Policy Inheritance

```
Organization Level
      │
      ├── Default Retention Policies (inherited by all workspaces)
      │
      └── Workspace Level
              │
              ├── Workspace-specific overrides (within bounds)
              │
              └── Data Type Level
                      │
                      └── Specific retention for data_type (within bounds)
```

---

## 6. Compliance Evidence

### 6.1 Retention Compliance Report

The system generates monthly retention compliance reports:

```json
{
  "report_id": "uuid",
  "report_type": "retention_compliance",
  "generated_at": "2025-12-16T00:00:00Z",
  "period": {
    "start": "2025-11-01T00:00:00Z",
    "end": "2025-11-30T23:59:59Z"
  },
  "summary": {
    "total_workspaces": 150,
    "compliant_workspaces": 148,
    "non_compliant_workspaces": 2,
    "total_purge_runs": 4380,
    "successful_purge_runs": 4375,
    "failed_purge_runs": 5,
    "total_records_purged": 12500000,
    "total_legal_holds_active": 3,
    "dsar_erasures_completed": 12,
    "dsar_erasures_blocked_by_hold": 1
  },
  "by_data_type": [
    {
      "data_type": "telemetry_aggregated",
      "workspaces_with_policy": 150,
      "purge_runs": 1500,
      "records_purged": 8000000,
      "average_retention_days": 85
    }
  ],
  "legal_holds": [
    {
      "workspace_id": "uuid",
      "data_type": "access_audits",
      "reason": "Litigation case #12345",
      "active_since": "2025-10-01T00:00:00Z",
      "hold_until": "2026-12-31T23:59:59Z"
    }
  ],
  "compliance_issues": [
    {
      "workspace_id": "uuid",
      "issue": "Retention policy below minimum for access_audits",
      "severity": "high",
      "remediation": "Auto-corrected to minimum 7 years"
    }
  ]
}
```

### 6.2 Evidence Pack Integration

Retention evidence is included in the enterprise evidence pack:

- Retention policy snapshots (per workspace)
- Purge job history (counts, timestamps, durations)
- Legal hold history (creations, releases, blockers)
- Compliance violation records (if any)
- DSAR erasure records (linked to DSAR workflow)

---

## 7. API Reference

### 7.1 Policy Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/governance/retention/policies` | List all retention policies for workspace |
| `GET` | `/api/v1/governance/retention/policies/{data_type}` | Get specific policy |
| `PUT` | `/api/v1/governance/retention/policies/{data_type}` | Update policy |
| `POST` | `/api/v1/governance/retention/policies/validate` | Validate proposed policy |

### 7.2 Purge Management Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/governance/retention/purge/status` | Get purge scheduler status |
| `GET` | `/api/v1/governance/retention/purge/history` | Get purge job history |
| `POST` | `/api/v1/governance/retention/purge/trigger` | Manually trigger purge (admin) |
| `GET` | `/api/v1/governance/retention/purge/preview` | Preview purge (dry run) |

### 7.3 Legal Hold Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/governance/legal-holds` | List all legal holds |
| `POST` | `/api/v1/governance/legal-holds` | Create legal hold |
| `GET` | `/api/v1/governance/legal-holds/{id}` | Get legal hold details |
| `POST` | `/api/v1/governance/legal-holds/{id}/release` | Release legal hold |
| `GET` | `/api/v1/governance/legal-holds/{id}/audit` | Get legal hold audit trail |

---

## 8. Testing Requirements

### 8.1 Integration Test Scenarios

1. **Purge Correctness**
   - Seed data older than cutoff → verify deleted
   - Seed data newer than cutoff → verify retained
   - Verify counts match expected

2. **Legal Hold Blocking**
   - Create legal hold → trigger purge → verify skipped
   - Release legal hold → trigger purge → verify deleted

3. **DSAR Integration**
   - Create DSAR erasure → execute → verify deleted
   - Create legal hold → create DSAR erasure → verify blocked

4. **Tenant Customization**
   - Override retention within bounds → verify accepted
   - Override retention below minimum → verify rejected/adjusted
   - Override retention above maximum → verify rejected/adjusted

5. **Audit Trail**
   - Every purge produces audit event
   - Every legal hold operation produces audit event
   - Audit events are immutable

---

## 9. References

### 9.1 GDPR Articles

- **Article 5(1)(e)** - Storage Limitation: "kept in a form which permits identification of data subjects for no longer than is necessary"
- **Article 17** - Right to Erasure (integration with DSAR)
- **Article 30** - Records of Processing Activities (retention periods in RoPA)

### 9.2 Internal Documents

- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md` - RoPA with retention periods
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md` - Phase 4 requirements

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-16 | Platform Compliance Team | Initial release - Phase 4 deliverable |
