# DSAR Phase 5 Specification

**Version**: 1.0.0
**Last Updated**: 2025-12-17
**GDPR Articles**: 12-23 (Data Subject Rights)

## 1. Overview

This specification defines the implementation of GDPR Data Subject Access Requests (DSAR) for Phase 5 of the CCEA GDPR compliance implementation.

### 1.1 Scope

The DSAR service handles requests for:

- **Access** (Art. 15) - Right of access to personal data
- **Portability** (Art. 20) - Right to data portability
- **Erasure** (Art. 17) - Right to erasure ("right to be forgotten")
- **Rectification** (Art. 16) - Right to rectification
- **Restriction** (Art. 18) - Right to restriction of processing

### 1.2 CCEA Boundary

Due to the Cloud-Controlled Execution Architecture (CCEA), DSAR scope is limited to **Cloud-controlled data only**. Agent-zone data remains customer-controlled and outside DSAR scope.

#### In-Scope Data Categories (Cloud)

| Category | Description | Exportable | Deletable |
|----------|-------------|------------|-----------|
| telemetry_events | Agent telemetry | Yes | Yes |
| alerts | Alert notifications | Yes | Yes |
| commands | Commands to agents | Yes | Yes |
| approval_records | Change approvals | Yes | **No** (7yr) |
| access_audits | Access logs | Yes | **No** (7yr) |
| user_settings | User preferences | Yes | Yes |
| agent_data | Agent registration | Yes | Yes |
| run_data | Strategy runs | Yes | Yes |
| deployment_data | Deployments | Yes | Yes |
| consent_records | Consent grants | Yes | **No** (7yr) |
| break_glass_requests | Emergency access | Yes | **No** (7yr) |
| session_data | User sessions | Yes | Yes |
| billing_records | Invoices | Yes | **No** (7yr) |

#### Out-of-Scope Data Categories (Agent)

| Category | Reason |
|----------|--------|
| broker_credentials | Never transmitted to Cloud |
| local_execution_logs | Unless exported to Cloud |
| order_fill_data | Unless RAW_ORDER_EVENTS enabled |
| local_vault_contents | Customer-controlled |
| position_data_local | Unless telemetry enabled |
| local_strategy_source | Customer IP |
| local_config_files | Local storage |

## 2. Request Lifecycle

### 2.1 State Machine

```
PENDING → AWAITING_VERIFICATION → VERIFIED → IN_PROGRESS → COMPLETED
    ↓              ↓                   ↓           ↓
    └──────────────┴───────────────────┴───────────┴→ REJECTED
                                                   └→ CANCELLED
                                                   └→ EXPIRED
                                                   └→ PARTIALLY_COMPLETED
```

### 2.2 Deadline Management

Per GDPR Article 12(3):

- **Standard deadline**: 30 calendar days from request
- **Extension**: +60 days (once only) for complex requests
- **Maximum total**: 90 days from request creation
- **Extension requires**: Written justification + notification to subject

```python
# Deadline calculation
deadline = created_at + timedelta(days=30)
extended_deadline = now() + timedelta(days=60)  # Max: created_at + 90 days
effective_deadline = extended_deadline if extended else deadline
is_overdue = now() > effective_deadline
```

## 3. Identity Verification

### 3.1 Verification Methods

| Method | Use Case | Assurance Level |
|--------|----------|-----------------|
| EMAIL_LINK | Standard requests | Medium |
| EMAIL_OTP | Higher assurance | Medium-High |
| SMS_OTP | Alternative channel | Medium |
| MFA_CHALLENGE | Users with MFA | High |
| SSO_SESSION | Enterprise SSO | High |
| DOCUMENT_UPLOAD | Manual verification | High |
| SUPPORT_MANUAL | Support-assisted | Varies |
| EXISTING_SESSION | Logged-in user | Low-Medium |

### 3.2 Verification Requirements

| Request Type | Verification Required | Notes |
|--------------|----------------------|-------|
| ACCESS | Recommended | Optional for low-risk |
| PORTABILITY | Recommended | Optional for low-risk |
| ERASURE | **Mandatory** | Always required |
| RECTIFICATION | Recommended | Context-dependent |
| RESTRICTION | Recommended | Context-dependent |

### 3.3 Verification Token

```json
{
  "token_hash": "sha256:...",
  "method": "email_link",
  "created_at": "2025-12-17T10:00:00Z",
  "expires_at": "2025-12-18T10:00:00Z",
  "attempts": 0,
  "max_attempts": 5,
  "status": "pending"
}
```

- Token TTL: 24 hours
- Max attempts: 5 (then locked)
- Constant-time comparison to prevent timing attacks

## 4. Request Processing

### 4.1 ACCESS/PORTABILITY Workflow

```
1. Verify identity (if required)
2. Fetch data from all in-scope categories
3. Build export package with CCEA boundary notice
4. Calculate SHA-256 checksum
5. Generate secure download token
6. Return result with download URL
```

### 4.2 ERASURE Workflow

```
1. Verify identity (MANDATORY)
2. Check each category for:
   a. Legal hold → Block, log exemption
   b. Compliance exemption → Block, log exemption
   c. Deletable → Add to deletion set
3. Create pre-deletion audit snapshot
4. Execute deletion for eligible categories
5. Return result with exemption details
```

### 4.3 Exemption Categories

Per GDPR Article 17(3):

| Exemption | Article | Description |
|-----------|---------|-------------|
| legal_obligation | 17(3)(b) | Compliance with legal obligation |
| public_interest | 17(3)(d) | Archiving, research purposes |
| legal_claims | 17(3)(e) | Establishment/defence of legal claims |
| regulatory_retention | - | Financial regulations (7yr) |

### 4.4 Legal Hold Integration

```python
# Before erasure
if legal_hold_service.is_data_held(workspace_id, category):
    exemptions_applied.append(category)
    log_audit(LEGAL_HOLD_CHECK, blocked=True)
    continue  # Skip deletion
```

## 5. Export Package Format

### 5.1 JSON Structure

```json
{
  "metadata": {
    "request_id": "uuid",
    "request_type": "access",
    "user_id": "user-123",
    "workspace_id": "workspace-456",
    "exported_at": "2025-12-17T12:00:00Z",
    "data_categories": ["telemetry_events", "alerts", ...],
    "record_count": 1234,
    "gdpr_article": "Article 15 (Access)"
  },
  "ccea_boundary": {
    "notice": "CCEA Architecture Data Boundary Notice...",
    "in_scope_categories": [...],
    "out_of_scope_categories": [...],
    "explanation": "This export contains only Cloud-controlled data..."
  },
  "data": [
    {"category": "telemetry_events", "records": [...]},
    {"category": "alerts", "records": [...]}
  ]
}
```

### 5.2 Checksum

- Algorithm: SHA-256
- Format: `sha256:<hex_digest>`
- Computed over entire export file

### 5.3 Download Link

- Token: 32-byte URL-safe random
- TTL: 168 hours (7 days)
- URL format: `/dsar/download/{request_id}?token={download_token}`
- Single-use or limited access

## 6. Audit Trail

### 6.1 Audit Actions

| Action | Description |
|--------|-------------|
| REQUEST_CREATED | New DSAR request |
| VERIFICATION_SENT | Verification initiated |
| VERIFICATION_ATTEMPTED | Token validation attempt |
| VERIFICATION_COMPLETED | Identity verified |
| VERIFICATION_FAILED | Verification failed |
| PROCESSING_STARTED | Processing began |
| DATA_COLLECTED | Data fetched |
| EXPORT_GENERATED | Export file created |
| ERASURE_STARTED | Deletion started |
| ERASURE_COMPLETED | Deletion finished |
| ERASURE_BLOCKED | Legal hold blocked |
| EXEMPTION_APPLIED | Art. 17(3) exemption |
| DEADLINE_EXTENDED | Extension granted |
| REQUEST_COMPLETED | Request finished |
| REQUEST_REJECTED | Request denied |
| REQUEST_CANCELLED | User cancelled |
| REQUEST_EXPIRED | Deadline missed |
| ERROR_OCCURRED | Processing error |
| LEGAL_HOLD_CHECK | Hold checked |

### 6.2 Audit Entry Structure

```json
{
  "id": "uuid",
  "action": "request_created",
  "request_id": "uuid",
  "user_id": "user-123",
  "workspace_id": "workspace-456",
  "timestamp": "2025-12-17T10:00:00Z",
  "actor_id": "support-agent-1",
  "actor_type": "user",
  "details": {...},
  "integrity_hash": "sha256:..."
}
```

### 6.3 Integrity

Each audit entry has:

- Unique ID (UUID)
- Timestamp (UTC)
- SHA-256 integrity hash
- Immutable after creation

## 7. Metrics and Monitoring

### 7.1 Key Metrics

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Average completion time | < 20 days | > 25 days |
| On-time completion rate | > 95% | < 90% |
| Overdue requests | 0 | > 0 |
| Verification success rate | > 98% | < 95% |

### 7.2 Dashboard Data

```json
{
  "total_requests": 150,
  "by_type": {
    "access": 100,
    "erasure": 30,
    "portability": 20
  },
  "by_status": {
    "completed": 120,
    "pending": 15,
    "in_progress": 10,
    "expired": 5
  },
  "average_completion_days": 18.5,
  "overdue_count": 2,
  "completed_on_time_rate": 0.96
}
```

## 8. Rate Limiting

To prevent abuse:

- **Maximum**: 12 requests per user per month
- **Cooldown**: 24 hours between same-type requests
- **Manifestly excessive**: May be refused per Art. 12(5)

## 9. API Reference

### 9.1 Create Request

```http
POST /dsar/requests
Content-Type: application/json

{
  "user_id": "user-123",
  "workspace_id": "workspace-456",
  "request_type": "access",
  "data_categories": ["telemetry_events", "alerts"],
  "reason": "Personal data access request"
}
```

### 9.2 Initiate Verification

```http
POST /dsar/requests/{request_id}/verify
Content-Type: application/json

{
  "method": "email_link",
  "send_to": "user@example.com"
}
```

### 9.3 Complete Verification

```http
POST /dsar/requests/{request_id}/verify/complete
Content-Type: application/json

{
  "token": "verification_token"
}
```

### 9.4 Process Request

```http
POST /dsar/requests/{request_id}/process
```

### 9.5 Extend Deadline

```http
POST /dsar/requests/{request_id}/extend
Content-Type: application/json

{
  "reason": "Complex request requiring additional time"
}
```

### 9.6 Get Request Status

```http
GET /dsar/requests/{request_id}
```

### 9.7 Download Export

```http
GET /dsar/download/{request_id}?token={download_token}
```

### 9.8 Get Metrics

```http
GET /dsar/metrics?workspace_id={workspace_id}&period_days=30
```

## 10. Security Considerations

### 10.1 Authentication

- All endpoints require authentication
- Request creator verified via session

### 10.2 Authorization

- Users can only access their own requests
- Support can access workspace requests
- Superuser can access all requests

### 10.3 Data Protection

- Export files encrypted at rest
- Download tokens are single-use
- Audit logs are immutable

### 10.4 Logging

- No PII in application logs
- Full audit trail in audit log
- Error messages sanitized

## 11. Compliance Checklist

- [x] Art. 12 - Transparent information (CCEA boundary notice)
- [x] Art. 12(3) - 30-day deadline with 60-day extension
- [x] Art. 15 - Right of access
- [x] Art. 16 - Right to rectification
- [x] Art. 17 - Right to erasure
- [x] Art. 17(3) - Exemptions documented
- [x] Art. 18 - Right to restriction
- [x] Art. 20 - Right to data portability
- [x] Art. 20(2) - Third-party rights protected (CCEA)
- [x] Identity verification before erasure
- [x] Audit trail for all actions
- [x] Legal hold integration

## 12. Testing Requirements

### 12.1 Unit Tests

- Request creation with all types
- Identity verification flow
- Deadline calculations
- Extension logic
- Exemption handling

### 12.2 Integration Tests

- End-to-end: create → verify → process → complete
- Legal hold blocking
- Audit trail completeness
- Metrics accuracy

### 12.3 Security Tests

- Token validation
- Rate limiting
- Authorization checks
- Download token security

## 13. References

- GDPR Regulation (EU) 2016/679
- EDPB Guidelines on Data Subject Rights
- CCEA Design Doc Section 13-16
- DSAR SOP (`docs/compliance/DSAR_SOP.md`)
