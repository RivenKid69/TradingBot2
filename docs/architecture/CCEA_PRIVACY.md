# CCEA Privacy & Data Governance

> **Version**: 2.0.0 | **Last Updated**: 2025-12-17
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 14
>
> **GDPR Status**: Designed to support GDPR requirements (privacy-by-design; not independently audited) - See [GDPR_COMPLIANCE_SUMMARY.md](../compliance/GDPR_COMPLIANCE_SUMMARY.md)

## Overview

This document defines privacy controls, data governance, and GDPR-aligned design measures for the CCEA Platform. The architecture is designed with privacy-by-design principles and intended to support GDPR (EU) 2016/679 requirements.

---

## 1. Data Minimization

### 1.1 Core Principle

**Collect only what is necessary for:**

- Service operation
- Billing (if usage-based)
- Support (with user consent)

### 1.2 Data Categories

| Category | What We Collect | Why | Retention |
|----------|-----------------|-----|-----------|
| **Account Data** | Email, name, org name | Service access | Account lifetime + 30 days |
| **Usage Data** | Feature usage, API calls | Billing, improvement | 90 days |
| **Strategy Metadata** | Names, configs (no code) | Service operation | User-controlled |
| **Aggregated Telemetry** | PnL, drawdown, errors | Monitoring | Configurable (default 90 days) |

### 1.3 Data Cloud Is Designed to Not Collect

| Data Type | Reason |
|-----------|--------|
| Broker API keys | Designed to stay in Agent only |
| Raw order events (by default) | Privacy, IP protection; opt-in for enterprise |
| Exact position sizes (by default) | Privacy; aggregated telemetry only unless opted-in |
| Account balances | Privacy |
| Individual trade details (by default) | Privacy, IP protection; aggregated unless opted-in |
| Strategy source code | IP protection (unless training service used) |

---

## 2. Telemetry Sensitivity Levels

### 2.1 Level Definitions

| Level | Description | Data Included | Default |
|-------|-------------|---------------|---------|
| `AGGREGATED` | Metrics only | PnL, drawdown, error rates, latency | Yes (retail/pro) |
| `DETAILED_NON_SENSITIVE` | Extended metrics | Timestamps, counts, states (no orders) | Opt-in (pro) |
| `RAW_ORDER_EVENTS` | Enterprise-only | Order/fill events (masked), position changes | Enterprise + explicit opt-in |

### 2.2 AGGREGATED Level (Default)

```json
{
  "sensitivity": "AGGREGATED",
  "metrics": {
    "pnl": 12.34,
    "drawdown": -3.2,
    "exposure_usd": 5000.0,
    "orders_per_min": 2,
    "broker_error_rate": 0,
    "latency_p99_ms": 45,
    "run_state": "RUNNING",
    "uptime_minutes": 120
  }
}
```

**Not included:**

- Individual order details
- Symbol-level positions
- Exact entry/exit prices
- Order IDs

### 2.3 DETAILED_NON_SENSITIVE Level (Opt-in)

```json
{
  "sensitivity": "DETAILED_NON_SENSITIVE",
  "metrics": {
    "pnl": 12.34,
    "drawdown": -3.2,
    "strategy_signals_count": 5,
    "orders_submitted": 10,
    "orders_filled": 8,
    "fill_rate": 0.8,
    "avg_latency_ms": 32,
    "max_latency_ms": 120
  },
  "state": {
    "run_state": "RUNNING",
    "last_signal_time": "2025-12-14T10:00:00Z",
    "positions_count": 3
  }
}
```

**Not included:**

- Individual order details
- Symbol names
- Prices
- Quantities

### 2.4 Raw Order Events (ENTERPRISE-ONLY)

This telemetry level is available **only for enterprise** customers with explicit opt-in.

**Requirements for RAW_ORDER_EVENTS:**

- Enterprise tier subscription required
- Explicit per-workspace opt-in (audited)
- Consent record with: who, what, when, scope, expiry
- Minimal retention: 7 days default, 30 days maximum
- Restricted access: workspace admins + break-glass
- Alternative: "telemetry stays local" mode (no Cloud transmission)

**Data included in RAW_ORDER_EVENTS (after mandatory redaction):**

- Order events (masked account IDs)
- Fill events
- Position changes
- Still **NEVER** includes: API keys, secrets, credentials, unmasked account IDs

**Rationale for restrictions:**

- Privacy risk: Order data reveals trading behavior
- IP risk: Reveals strategy logic
- Regulatory risk: Could be construed as advisory data

See [RAW_ORDER_EVENTS_HANDLING_SPEC.md](../compliance/RAW_ORDER_EVENTS_HANDLING_SPEC.md) for full specification.

---

## 3. Data Retention

### 3.1 Default Retention Policies

| Data Type | Retail | Pro | Enterprise |
|-----------|--------|-----|------------|
| Account data | Account + 30 days | Account + 30 days | Contract |
| Usage/billing | 24 months | 24 months | Contract |
| Telemetry | 30 days | 90 days | Configurable |
| Audit logs | 12 months | 24 months | 7 years |
| Strategy artifacts | User-controlled | User-controlled | Contract |
| Backtest results | 30 days | 90 days | Configurable |

### 3.2 Automatic Purge

```python
# Data retention enforcement
async def enforce_retention():
    """
    Automatic data purge based on retention policies.
    Runs daily via scheduled job.
    """
    workspaces = await get_all_workspaces()

    for workspace in workspaces:
        policy = await get_retention_policy(workspace.id)

        # Purge telemetry
        await purge_telemetry(
            workspace_id=workspace.id,
            older_than_days=policy.telemetry_retention_days
        )

        # Purge audit logs (respecting minimum)
        min_audit_days = 365  # Legal minimum
        await purge_audit_logs(
            workspace_id=workspace.id,
            older_than_days=max(policy.audit_retention_days, min_audit_days)
        )

        # Purge backtest results
        await purge_backtest_results(
            workspace_id=workspace.id,
            older_than_days=policy.backtest_retention_days
        )
```

### 3.3 User-Initiated Deletion

Users can request deletion of:

- Account and all associated data
- Specific strategies and artifacts
- Telemetry data
- Backtest results

```
API: DELETE /api/v1/user/data
API: DELETE /api/v1/strategies/{id}
API: DELETE /api/v1/telemetry?before={date}
```

---

## 4. Data Residency

### 4.1 Regions

| Region | Location | Default For |
|--------|----------|-------------|
| `eu-central` | Frankfurt, Germany | EU users |
| `eu-west` | Dublin, Ireland | UK users |
| `us-east` | Virginia, USA | US users |
| `ap-southeast` | Singapore | APAC users |

### 4.2 EU Data Residency (Default for EU)

**For EU users:**

- All data stored in EU region by default
- No data transfer outside EU without explicit consent
- Sub-processors located in EU or with SCCs

**Configuration:**

```yaml
# workspace settings
data_residency:
  region: eu-central
  strict_mode: true  # No cross-region processing
```

### 4.3 Enterprise Options

| Option | Description |
|--------|-------------|
| **Dedicated region** | Single-tenant region deployment |
| **On-premises** | Full stack in customer infrastructure |
| **Customer-managed keys** | BYOK encryption |
| **Air-gapped** | No external connectivity |

---

## 5. Access Controls

### 5.1 RBAC (Role-Based Access Control)

| Role | Permissions |
|------|-------------|
| `owner` | All permissions, billing, delete workspace |
| `admin` | Manage users, agents, deployments |
| `developer` | Create/deploy strategies, view telemetry |
| `viewer` | Read-only access to dashboards |

### 5.2 Sensitive Data Access

Access to sensitive data (detailed telemetry, audit logs) requires:

1. **Role permission** - Must have appropriate role
2. **Audit logging** - All access logged
3. **Time-limited** - Session expires after 30 min
4. **Break-glass** - Emergency access with reason required

### 5.3 Break-Glass Access

For support/incident response:

```python
async def break_glass_access(
    user_id: str,
    resource_type: str,
    resource_id: str,
    reason: str  # REQUIRED
) -> AccessGrant:
    """
    Emergency access to sensitive data.
    Requires explicit reason and creates audit record.
    """
    if not reason or len(reason) < 20:
        raise ValueError("Break-glass requires detailed reason")

    # Create audit record
    await create_audit_record(
        user_id=user_id,
        action="break_glass",
        resource_type=resource_type,
        resource_id=resource_id,
        reason=reason,
        timestamp=datetime.utcnow()
    )

    # Notify workspace owner
    await notify_workspace_owner(
        workspace_id=get_workspace_id(resource_id),
        message=f"Break-glass access by {user_id}: {reason}"
    )

    # Grant time-limited access
    return AccessGrant(
        resource_id=resource_id,
        expires_at=datetime.utcnow() + timedelta(minutes=30)
    )
```

### 5.4 Audit Log

All sensitive data access is logged:

```json
{
  "audit_id": "aud_123",
  "workspace_id": "ws_456",
  "user_id": "user_789",
  "action": "view_telemetry",
  "resource_type": "telemetry",
  "resource_id": "tel_abc",
  "ip_address": "192.168.1.1",
  "user_agent": "Mozilla/5.0...",
  "timestamp": "2025-12-14T10:00:00Z",
  "access_reason": null,  // Required for break_glass
  "session_id": "sess_xyz"
}
```

---

## 6. GDPR Compliance

### 6.1 Legal Basis

| Processing Activity | Legal Basis |
|---------------------|-------------|
| Account management | Contract performance |
| Service provision | Contract performance |
| Billing | Contract performance |
| Security monitoring | Legitimate interest |
| Service improvement | Legitimate interest |
| Marketing | Consent |

### 6.2 Data Subject Rights

| Right | Implementation |
|-------|----------------|
| **Access** | Export all data via dashboard or API |
| **Rectification** | Edit account data in settings |
| **Erasure** | Delete account (with confirmation) |
| **Portability** | Export in JSON/CSV format |
| **Restriction** | Pause processing (support ticket) |
| **Objection** | Opt-out of non-essential processing |

### 6.3 Data Subject Access Request (DSAR)

```
API: GET /api/v1/user/export

Response:
{
  "request_id": "dsar_123",
  "status": "processing",
  "estimated_completion": "2025-12-15T10:00:00Z",
  "download_url": null  // Populated when ready
}

// Export contents:
- account_data.json
- strategies.json
- telemetry.json
- audit_log.json
- billing.json
```

### 6.4 DPA (Data Processing Agreement)

Enterprise customers receive DPA including:

- Data categories processed
- Processing purposes
- Sub-processors list
- Security measures
- Breach notification procedures
- Data return/deletion terms

---

## 7. Redaction Implementation

### 7.1 Mandatory Redaction Patterns

```python
REDACTION_PATTERNS = [
    # API credentials
    (r'api[_-]?key', '[API_KEY_REDACTED]'),
    (r'api[_-]?secret', '[API_SECRET_REDACTED]'),
    (r'secret[_-]?key', '[SECRET_REDACTED]'),

    # Passwords and tokens
    (r'password', '[PASSWORD_REDACTED]'),
    (r'token', '[TOKEN_REDACTED]'),
    (r'auth[_-]?token', '[AUTH_REDACTED]'),

    # Private keys
    (r'private[_-]?key', '[PRIVATE_KEY_REDACTED]'),

    # Account identifiers
    (r'account[_-]?id', '[ACCOUNT_REDACTED]'),
    (r'account[_-]?number', '[ACCOUNT_REDACTED]'),
]
```

### 7.2 Redaction Middleware

```python
class RedactionMiddleware:
    """
    Mandatory middleware for all outbound data.
    Cannot be disabled.
    """

    def __init__(self):
        self._patterns = compile_patterns(REDACTION_PATTERNS)

    def redact(self, data: dict) -> dict:
        """Redact sensitive fields. Always enabled."""
        return self._redact_recursive(data)

    def _redact_recursive(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: self._redact_value(k, v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._redact_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return self._redact_string(obj)
        return obj

    def _redact_value(self, key: str, value: Any) -> Any:
        for pattern, replacement in self._patterns:
            if pattern.match(key):
                return replacement
        return self._redact_recursive(value)

    def _redact_string(self, s: str) -> str:
        # Redact any embedded secrets
        for pattern, replacement in self._patterns:
            s = pattern.sub(replacement, s)
        return s
```

### 7.3 Log Sanitization

```python
class SanitizedLogger:
    """
    Logger that sanitizes all output.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        self._redactor = RedactionMiddleware()

    def info(self, msg: str, **kwargs):
        self._log('info', msg, kwargs)

    def error(self, msg: str, **kwargs):
        self._log('error', msg, kwargs)

    def _log(self, level: str, msg: str, extra: dict):
        sanitized_msg = self._redactor._redact_string(msg)
        sanitized_extra = self._redactor.redact(extra)
        getattr(self._logger, level)(sanitized_msg, extra=sanitized_extra)
```

---

## 8. Privacy Configuration

### 8.1 Workspace Settings

```yaml
# Workspace privacy settings
privacy:
  telemetry_level: AGGREGATED  # AGGREGATED or DETAILED_NON_SENSITIVE
  data_residency: eu-central
  retention:
    telemetry_days: 90
    audit_days: 365
    backtest_days: 90
  sharing:
    share_anonymized_metrics: false  # For platform improvement
    share_crash_reports: true        # For stability
```

### 8.2 Agent Privacy Config

```yaml
# Agent local privacy settings
privacy:
  # What to send to Cloud
  telemetry:
    level: AGGREGATED
    include_latency: true
    include_counts: true
    include_pnl: true  # Aggregated only

  # What stays local
  local_only:
    - order_details
    - position_details
    - symbol_names
    - account_balances

  # Redaction (enabled by default; verify via CI tests)
  redaction:
    enabled: true  # Designed not to be disabled; verify via telemetry tests
    patterns: [default]
```

---

## 9. Compliance Checklist

### 9.1 GDPR Readiness

- [x] Data minimization implemented
- [x] Lawful basis documented
- [x] Privacy policy published
- [x] DPA template available
- [x] DSAR process implemented
- [x] Data retention automated
- [x] EU data residency available (design target; verify via deployment audit)
- [x] Sub-processor list maintained
- [x] Breach notification procedure
- [x] DPO contact published

### 9.2 Ongoing Compliance

| Activity | Frequency |
|----------|-----------|
| Privacy impact assessment | Per new feature |
| Data inventory update | Quarterly |
| Sub-processor review | Annually |
| Retention policy audit | Quarterly |
| Access log review | Monthly |
| Security assessment | Annually |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial privacy doc per Design Doc |
| 2.0.0 | 2025-12-17 | CCEA Team | GDPR controls implementation documented (internal; not independently audited), RAW_ORDER_EVENTS enterprise support documented |

---

**Related Documentation:**

- [CCEA Overview](./CCEA_OVERVIEW.md)
- [Data Model](./CCEA_DATA_MODEL.md)
- [Terms of Service Guidelines](../business/CCEA_TERMS_OF_SERVICE_GUIDELINES.md)
- [GDPR Compliance Summary](../compliance/GDPR_COMPLIANCE_SUMMARY.md)
- [GDPR Implementation Plan](../compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md)
- [Privacy Policy](../legal/PRIVACY_POLICY.md)
- [DPA Template](../legal/DPA_TEMPLATE.md)
- [DSAR SOP](../compliance/DSAR_SOP.md)
- [Breach Response SOP](../compliance/BREACH_RESPONSE_SOP.md)
