# GDPR Phase 6: Access Control, Audit, and Break-Glass Specification

**Version**: 1.0.0
**Date**: 2025-12-17
**Status**: Implemented
**GDPR Reference**: Articles 5(1)(f), 25, 32 (Security of Processing)

## 1. Overview

Phase 6 implements comprehensive access control, audit, and emergency access mechanisms aligned with GDPR security requirements and the CCEA architecture constraints.

### 1.1 Goals

1. **Least Privilege Access**: RBAC inside workspace with granular scopes (read/write/admin/support)
2. **Provable Accountability**: Every sensitive access attributable to principal and request_id
3. **Break-Glass Access**: Incident-only emergency access with full audit trail
4. **Change Management**: TRADING_IMPACTING changes require local approval with evidence

### 1.2 Design Doc References

- Design Doc 14.4: "Access control: RBAC in workspace, audit log of access, break-glass with reason and auditable event"
- Design Doc 6.2, 12.2: "Structured approval evidence with immutable blob references"
- Design Doc 9.4: "Kill switch and trading-impacting protections"

## 2. RBAC Architecture

### 2.1 Permission Model

The permission model follows a hierarchical structure:

```
Organization
    └── Workspace
            └── Role
                    └── Permissions (resource:action)
```

### 2.2 Scopes

| Scope | Description | Typical Users |
|-------|-------------|---------------|
| `READ` | Read-only access to data | All authenticated users |
| `WRITE` | Create/update non-sensitive data | Team members |
| `ADMIN` | Full administrative access | Workspace admins |
| `SUPPORT` | Support access (time-limited) | Support engineers |
| `AUDIT` | Access to audit logs | Compliance officers |
| `BREAK_GLASS` | Emergency elevated access | Incident responders |

### 2.3 Resource Types

| Resource | Description | Sensitive |
|----------|-------------|-----------|
| `strategy` | Strategy definitions and versions | No |
| `deployment` | Deployment configurations | Yes |
| `run` | Run instances | Yes |
| `telemetry` | Telemetry data | Yes (if RAW_ORDER_EVENTS) |
| `audit_log` | Audit records | Yes |
| `config_blob` | Configuration blobs | Yes |
| `agent` | Agent registrations | Yes |
| `approval` | Approval records | Yes |
| `dsar` | DSAR requests | Yes |
| `break_glass` | Break-glass requests | Yes |

### 2.4 Default Roles

| Role | Permissions | Description |
|------|-------------|-------------|
| `owner` | All permissions | Workspace owner |
| `admin` | All except owner transfer | Workspace administrator |
| `developer` | strategy:*, deployment:read/create, run:* | Development team |
| `viewer` | *:read (non-sensitive) | Read-only access |
| `support` | Support scoped access | Time-limited support |
| `auditor` | audit_log:read, approval:read | Compliance audit |

## 3. Access Audit Log

### 3.1 Audit Event Schema

Every access to sensitive data produces an immutable audit record:

```json
{
  "id": "uuid",
  "timestamp": "ISO8601",
  "workspace_id": "uuid",
  "request_id": "uuid",

  "principal": {
    "type": "user|agent|system|service",
    "id": "uuid",
    "email": "user@example.com",
    "roles": ["admin", "developer"],
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0...",
    "session_id": "uuid"
  },

  "action": {
    "type": "read|write|delete|approve|export|break_glass",
    "resource_type": "telemetry|config_blob|...",
    "resource_id": "uuid",
    "operation": "GET /api/v1/telemetry/{id}",
    "data_categories": ["telemetry_events"],
    "sensitivity_level": "standard|sensitive|critical"
  },

  "authorization": {
    "granted": true,
    "method": "rbac|break_glass|api_key",
    "role_used": "admin",
    "permissions_checked": ["telemetry:read"],
    "break_glass_id": null,
    "consent_id": null
  },

  "context": {
    "reason": "Investigating alert #12345",
    "ticket_id": "TICKET-123",
    "related_request_ids": []
  },

  "result": {
    "status": "success|denied|error",
    "records_accessed": 100,
    "error_message": null
  },

  "integrity_hash": "sha256:..."
}
```

### 3.2 Sensitive Data Access Categories

Access to these data categories always produces an audit record:

1. **Telemetry** (especially RAW_ORDER_EVENTS)
2. **DSAR exports and requests**
3. **Audit logs themselves**
4. **Break-glass requests and approvals**
5. **Config blobs**
6. **Agent registrations and trust state changes**
7. **Approval records**
8. **User PII**

### 3.3 Audit Log Retention

- Audit logs: **7 years** (compliance requirement)
- Audit logs are exempt from DSAR erasure (Art. 17(3)(b))
- Integrity hash chain for tamper detection

## 4. Break-Glass Access

### 4.1 Requirements

Break-glass access is **incident-only** with mandatory:

1. **Reason** - Minimum 20 characters, pre-defined categories
2. **Scope** - Limited to specific resources/operations
3. **Time Bound** - Maximum 24 hours, default 4 hours
4. **Approval** - Self-approval not allowed
5. **Audit** - Full audit trail with evidence hash

### 4.2 Break-Glass Reasons

| Reason | Description |
|--------|-------------|
| `INCIDENT_RESPONSE` | Active production incident |
| `SECURITY_INVESTIGATION` | Security breach investigation |
| `COMPLIANCE_AUDIT` | Regulatory audit support |
| `DATA_RECOVERY` | Data recovery operation |
| `SYSTEM_FAILURE` | Critical system failure |
| `CUSTOMER_EMERGENCY` | Customer-reported emergency |

### 4.3 Break-Glass Scopes

| Scope | Access Granted |
|-------|----------------|
| `TELEMETRY_READ` | Read telemetry data |
| `AUDIT_READ` | Read audit logs |
| `CONFIG_READ` | Read configuration |
| `DATA_EXPORT` | Export data |
| `ADMIN_ACCESS` | Full admin (restricted) |

### 4.4 Break-Glass Workflow

```
1. Request Created
   ├── Validate reason (min 20 chars)
   ├── Check cooldown (5 min between requests)
   └── Generate evidence hash

2. Approval Required
   ├── Approver cannot be requester
   ├── Approver must have BREAK_GLASS_APPROVER role
   └── Approval recorded with timestamp

3. Access Granted
   ├── Time-limited token issued
   ├── Scope restrictions enforced
   └── Every access logged

4. Expiry/Revocation
   ├── Automatic expiry at deadline
   ├── Manual revocation supported
   └── All tokens invalidated
```

### 4.5 Evidence Hash

Break-glass evidence is hashed for integrity:

```python
evidence = f"{request_id}:{requester_id}:{reason}:{created_at.isoformat()}"
evidence_hash = sha256(evidence.encode()).hexdigest()
```

## 5. Change Management

### 5.1 Change Classification

| Class | Description | Approval Required |
|-------|-------------|-------------------|
| `OPERATIONAL` | Non-impacting changes | No |
| `TRADING_IMPACTING` | Affects trading behavior | Yes (local) |
| `SECURITY_SENSITIVE` | Security-related changes | Yes + Security review |
| `DATA_SENSITIVE` | Affects personal data | Yes + DPO review |

### 5.2 TRADING_IMPACTING Changes

Per Design Doc 6.2, all TRADING_IMPACTING changes require:

1. **Local approval** - User must explicitly approve
2. **Diff shown** - Changes displayed at approval time
3. **Evidence hashes** - Immutable record of what was approved:
   - `config_blob_digest` - Configuration at approval time
   - `manifest_digest` - Artifact manifest digest
   - `previous_state_digest` - State before change
   - `new_state_digest` - State after change

### 5.3 Approval Record Schema

```json
{
  "id": "uuid",
  "command_id": "uuid",
  "workspace_id": "uuid",
  "approved": true,
  "approved_by": "local|user_id",
  "approved_at": "ISO8601",

  "evidence": {
    "config_blob_digest": "sha256:...",
    "manifest_digest": "sha256:...",
    "previous_state_digest": "sha256:...",
    "new_state_digest": "sha256:..."
  },

  "diff_summary": {
    "strategy_version": {"from": "1.0.0", "to": "1.1.0"},
    "config_changes": ["risk_limit: 100 -> 150"],
    "param_count": 3
  },

  "attestation": {
    "user_acknowledged": true,
    "timestamp": "ISO8601",
    "display_hash": "sha256:..."
  },

  "reason": "Increasing risk limit for Q4"
}
```

### 5.4 Change Journal

All changes are recorded in an exportable journal:

```json
{
  "changes": [
    {
      "change_id": "uuid",
      "timestamp": "ISO8601",
      "workspace_id": "uuid",
      "change_class": "TRADING_IMPACTING",
      "requester_id": "uuid",
      "approver_id": "uuid",
      "approval_record_id": "uuid",
      "artifact_digest": "sha256:...",
      "config_digest": "sha256:...",
      "description": "Deploy strategy v1.1.0"
    }
  ]
}
```

## 6. Export and Evidence Pack

### 6.1 Exportable Artifacts

For customer due diligence and audits:

1. **Access Audit Logs** - Filtered by workspace/time/actor
2. **Break-Glass Records** - All requests, approvals, usage
3. **Change Journal** - Deploy/upgrade/approval records
4. **RBAC Snapshots** - Role/permission definitions at point in time
5. **Approval Evidence** - Diffs, digests, attestations

### 6.2 Export Format

```json
{
  "export_metadata": {
    "export_id": "uuid",
    "exported_at": "ISO8601",
    "exported_by": "uuid",
    "workspace_id": "uuid",
    "export_type": "access_audit|break_glass|change_journal|rbac_snapshot",
    "period_start": "ISO8601",
    "period_end": "ISO8601",
    "record_count": 1000
  },
  "integrity": {
    "checksum": "sha256:...",
    "signature": "...",
    "signing_key_id": "..."
  },
  "data": [...]
}
```

## 7. API Endpoints

### 7.1 RBAC APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/rbac/roles` | List roles |
| POST | `/api/v1/rbac/roles` | Create role |
| GET | `/api/v1/rbac/permissions` | List permissions |
| POST | `/api/v1/rbac/check` | Check permission |
| GET | `/api/v1/rbac/users/{id}/permissions` | Get user permissions |

### 7.2 Audit APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/audit/access` | Query access logs |
| GET | `/api/v1/audit/access/{id}` | Get audit entry |
| POST | `/api/v1/audit/export` | Export audit logs |

### 7.3 Break-Glass APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/break-glass/request` | Create request |
| POST | `/api/v1/break-glass/{id}/approve` | Approve request |
| POST | `/api/v1/break-glass/{id}/revoke` | Revoke access |
| GET | `/api/v1/break-glass/active` | List active requests |
| GET | `/api/v1/break-glass/{id}/audit` | Get audit trail |

### 7.4 Change Management APIs

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/changes/classify` | Classify change |
| POST | `/api/v1/changes/request-approval` | Request approval |
| POST | `/api/v1/changes/{id}/approve` | Record approval |
| GET | `/api/v1/changes/journal` | Get change journal |
| POST | `/api/v1/changes/journal/export` | Export journal |

## 8. Security Considerations

### 8.1 Defense in Depth

1. **Authentication** - MFA required for sensitive operations
2. **Authorization** - RBAC with scope-based permissions
3. **Audit** - All access logged with integrity hashes
4. **Break-Glass** - Time-limited, approved, fully audited

### 8.2 Rate Limiting

| Operation | Limit |
|-----------|-------|
| Permission checks | 1000/minute/user |
| Break-glass requests | 1/5 minutes/user |
| Audit exports | 10/hour/workspace |

### 8.3 Integrity Protection

- All audit entries have SHA-256 integrity hash
- Audit logs are append-only (no updates/deletes)
- Evidence hashes computed at creation time
- Tamper detection via hash chain verification

## 9. Implementation Components

### 9.1 Services

| Service | File | Description |
|---------|------|-------------|
| `RBACService` | `rbac_service.py` | Role-based access control |
| `AccessAuditService` | `access_audit.py` | Audit logging |
| `BreakGlassPhase6Service` | `break_glass_phase6.py` | Enhanced break-glass |
| `ChangeManagementService` | `change_management.py` | Change tracking |

### 9.2 Database Models

Extensions to `models.py`:

- `AccessAuditEntry` - Audit log entry
- `BreakGlassRequestPhase6` - Enhanced break-glass
- `ChangeRecord` - Change journal entry
- `ApprovalEvidence` - Approval evidence blob

## 10. Test Coverage

### 10.1 Required Tests

1. **RBAC Tests**
   - Permission grant/deny scenarios
   - Role hierarchy
   - Scope-based access
   - Default role enforcement

2. **Audit Tests**
   - Audit entry creation
   - Integrity hash verification
   - Query and filtering
   - Export functionality

3. **Break-Glass Tests**
   - Request creation validation
   - Approval workflow
   - Scope enforcement
   - Expiry and revocation
   - Self-approval prevention

4. **Change Management Tests**
   - Classification accuracy
   - Approval evidence capture
   - Journal recording
   - Export format validation

### 10.2 Integration Tests

- End-to-end access with audit
- Break-glass workflow complete cycle
- Change with approval and journal record
- Export and verify integrity

## 11. Compliance Mapping

| Requirement | GDPR Article | Implementation |
|-------------|--------------|----------------|
| Access control | Art. 32(1)(b) | RBAC Service |
| Audit logging | Art. 32(1)(d) | Access Audit Service |
| Accountability | Art. 5(2) | Principal attribution |
| Security measures | Art. 32(1) | Break-glass controls |
| Records of processing | Art. 30 | Change journal |
