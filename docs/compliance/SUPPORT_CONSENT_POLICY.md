# Support Data Access Consent Policy

**Document Type**: Compliance Operations Policy
**Version**: 1.0.0
**Last Updated**: 2025-12-16
**Owner**: Data Protection Officer
**Scope**: EU-only CCEA Cloud platform

---

## 1. Purpose

This policy defines the consent requirements for support staff access to customer data. It ensures that:

1. Customer data access requires explicit, time-limited consent
2. Consent is designed to be auditable (verify logging implementation via audit trail review)
3. Customers can revoke consent at any time
4. Access is blocked without valid, non-expired consent

---

## 2. Scope

This policy applies to:
- All support staff (Level 1, Level 2, Level 3)
- All customer data in Cloud-zone systems
- All support-related data access including:
  - Viewing customer telemetry
  - Accessing workspace configuration
  - Reviewing strategy metadata
  - Exporting logs for debugging
  - Reviewing audit trails

**Excluded:**
- Aggregated, anonymized metrics (no consent required)
- Security incident investigation (break-glass procedure applies)
- Legal obligation compliance (separate procedure)

---

## 3. Consent Requirements

### 3.1 Consent Record Structure

Every support data access consent must include:

| Field | Description | Required | Example |
|-------|-------------|----------|---------|
| `consent_id` | Unique identifier | Yes | `consent_abc123xyz` |
| `user_id` | Customer user granting consent | Yes | `user_12345` |
| `workspace_id` | Scope of access | Yes | `ws_67890` |
| `organization_id` | Organization context | Yes | `org_11111` |
| `granted_at` | UTC timestamp of grant | Yes | `2025-01-15T10:00:00Z` |
| `expires_at` | UTC timestamp of expiry | Yes | `2025-01-18T10:00:00Z` |
| `scope` | Data types accessible | Yes | `["telemetry", "logs", "config"]` |
| `purpose` | Reason for access | Yes | `Debug performance issue` |
| `support_ticket_id` | Associated ticket | Yes | `TICKET-12345` |
| `granted_by_email` | Email of granting user | Yes | `user@example.com` |
| `support_agent_id` | Support staff requesting | Yes | `support_agent_001` |
| `revoked_at` | Revocation timestamp | If revoked | `2025-01-16T15:00:00Z` |
| `revoked_by` | Who revoked | If revoked | `user_12345` |
| `revocation_reason` | Why revoked | If revoked | `No longer needed` |

### 3.2 Consent Scope Options

| Scope | Description | Data Accessible |
|-------|-------------|-----------------|
| `telemetry` | Telemetry data access | AGGREGATED, DETAILED_NON_SENSITIVE (not RAW unless enterprise) |
| `logs` | Application logs | Workspace-specific logs (redacted) |
| `config` | Configuration access | Workspace settings, preferences |
| `strategy_metadata` | Strategy information | Names, versions, not code |
| `commands` | Command history | Command records, ack status |
| `audit` | Audit log access | Access events where user is subject |
| `full` | All above combined | Full workspace data access |

### 3.3 Expiry Rules

| Consent Type | Default Expiry | Maximum Expiry |
|--------------|----------------|----------------|
| Standard support | 72 hours | 7 days |
| Extended investigation | 7 days | 14 days |
| Complex issue resolution | 14 days | 30 days |

**Note:** Extended expiry requires manager approval and documented justification.

---

## 4. Consent Workflow

### 4.1 Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SUPPORT CONSENT WORKFLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: CONSENT REQUEST                                                    │
│  ───────────────────────                                                    │
│  ┌──────────────────┐    ┌──────────────────┐                              │
│  │  Support Agent   │───▶│  Generate        │                              │
│  │  Opens Ticket    │    │  Consent Request │                              │
│  └──────────────────┘    └────────┬─────────┘                              │
│                                   │                                         │
│                                   ▼                                         │
│  STEP 2: CUSTOMER NOTIFICATION                                              │
│  ─────────────────────────────                                              │
│  ┌──────────────────────────────────────────────┐                          │
│  │  Email sent to authorized workspace admin:   │                          │
│  │  - Support ticket reference                  │                          │
│  │  - Requested scope                           │                          │
│  │  - Expiry duration                           │                          │
│  │  - Approve / Deny buttons                    │                          │
│  └────────────────────┬─────────────────────────┘                          │
│                       │                                                     │
│          ┌────────────┴────────────┐                                       │
│          ▼                         ▼                                        │
│  ┌──────────────┐         ┌──────────────┐                                 │
│  │   APPROVE    │         │    DENY      │                                 │
│  └──────┬───────┘         └──────┬───────┘                                 │
│         │                        │                                          │
│         ▼                        ▼                                          │
│  STEP 3: CONSENT ACTIVATED      │                                          │
│  ─────────────────────────      │                                          │
│  ┌──────────────────┐           │                                          │
│  │ Consent Record   │           │                                          │
│  │ Created + Audit  │           │                                          │
│  │ Event Logged     │           │                                          │
│  └────────┬─────────┘           │                                          │
│           │                     │                                          │
│           ▼                     │                                          │
│  STEP 4: SUPPORT ACCESS         │                                          │
│  ──────────────────────         │                                          │
│  ┌──────────────────┐           │                                          │
│  │ Support Agent    │           │                                          │
│  │ Accesses Data    │           │                                          │
│  │ (All Logged)     │           │                                          │
│  └────────┬─────────┘           │                                          │
│           │                     │                                          │
│           ▼                     │                                          │
│  STEP 5: CONSENT EXPIRES/REVOKED                                           │
│  ───────────────────────────────                                           │
│  ┌──────────────────────────────────────────────┐                          │
│  │ Access automatically blocked when:           │                          │
│  │ - Expiry time reached                        │◀────────────────────────┘
│  │ - Customer revokes consent                   │                          │
│  │ - Ticket closed                              │                          │
│  └──────────────────────────────────────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Consent Request Process

**Step 1: Support Agent Initiates Request**

Support agent in the admin console:
1. Opens support ticket
2. Clicks "Request Data Access"
3. Selects scope (what data needed)
4. Provides justification
5. Selects expiry duration

**Step 2: Customer Notification**

System automatically:
1. Identifies authorized workspace admins
2. Sends consent request email
3. Creates pending consent record
4. Logs request in audit trail

Email template:
```
Subject: Support Data Access Request [TICKET-XXXXX]

Dear [Workspace Admin Name],

Our support team is assisting with your request [TICKET-XXXXX].

To investigate this issue, we need temporary access to your workspace data.

REQUEST DETAILS:
- Ticket: TICKET-XXXXX
- Requested Scope: [telemetry, logs, config]
- Duration: 72 hours
- Purpose: [Debug performance issue]
- Support Agent: [Agent Name]

[APPROVE ACCESS]  [DENY ACCESS]

If you approve:
- Access will be limited to the requested scope
- Access is designed to be logged in your audit trail (verify via audit log review)
- You can revoke access at any time in Settings > Privacy > Support Access
- Access will automatically expire after the specified duration

If you deny:
- No data access will be granted
- Support will continue with information you provide directly

This request will expire in 7 days if no action is taken.

Questions? Contact privacy@[company-domain].com

Regards,
Platform Support Team
```

**Step 3: Customer Approves or Denies**

On approval:
1. Consent record activated
2. Expiry time calculated
3. Audit event created: `SUPPORT_CONSENT_GRANTED`
4. Support agent notified

On denial:
1. Request closed
2. Audit event created: `SUPPORT_CONSENT_DENIED`
3. Support agent notified

**Step 4: Data Access**

While consent is active:
1. Support agent can access approved scope
2. Every data access logged with consent_id reference
3. Customer can view access log in real-time

**Step 5: Consent Ends**

Consent ends when:
- Expiry time reached (automatic)
- Customer revokes (immediate)
- Ticket closed (automatic)

On end:
1. Access immediately blocked
2. Audit event created: `SUPPORT_CONSENT_EXPIRED` or `SUPPORT_CONSENT_REVOKED`
3. Support session terminated

---

## 5. Consent Revocation

### 5.1 Revocation Methods

Customers can revoke consent via:

1. **Self-service UI**: Account Settings > Privacy > Support Access > Revoke
2. **Email**: Reply to consent email with "REVOKE"
3. **Support ticket**: Request revocation in ticket
4. **DPO contact**: Email dpo@[company-domain].com

### 5.2 Revocation Process

1. Customer initiates revocation
2. System immediately:
   - Sets `revoked_at` timestamp
   - Blocks all data access
   - Terminates active support sessions
   - Logs audit event: `SUPPORT_CONSENT_REVOKED`
3. Support agent notified
4. Customer confirmation sent

### 5.3 Revocation Effect

- **Immediate**: Access blocked within seconds
- **Permanent**: Consent cannot be "un-revoked" (new consent required)
- **Logged**: All revocation details recorded

---

## 6. Enforcement

### 6.1 Access Control Enforcement

```python
# Pseudocode for consent enforcement
def check_support_access(support_agent_id, workspace_id, data_scope):
    # Find active consent
    consent = find_active_consent(
        workspace_id=workspace_id,
        support_agent_id=support_agent_id
    )

    if consent is None:
        log_access_denied("NO_ACTIVE_CONSENT")
        raise AccessDenied("No active consent for this workspace")

    if consent.expires_at < now():
        log_access_denied("CONSENT_EXPIRED")
        raise AccessDenied("Consent has expired")

    if consent.revoked_at is not None:
        log_access_denied("CONSENT_REVOKED")
        raise AccessDenied("Consent has been revoked")

    if data_scope not in consent.scope:
        log_access_denied("SCOPE_NOT_AUTHORIZED")
        raise AccessDenied(f"Scope '{data_scope}' not authorized")

    # Access granted - log it
    log_access_granted(consent.consent_id, data_scope)
    return True
```

### 6.2 Data Export Blocking

Support data export is blocked without consent:

```python
def export_workspace_data(support_agent_id, workspace_id, export_type):
    # Verify consent exists and is valid
    if not check_support_access(support_agent_id, workspace_id, export_type):
        raise ExportDenied("Valid consent required for data export")

    # Proceed with export
    export = generate_export(workspace_id, export_type)

    # Log export
    log_support_export(
        consent_id=get_active_consent_id(),
        export_type=export_type,
        records_exported=export.record_count
    )

    return export
```

---

## 7. Audit Trail

### 7.1 Audit Events

All consent-related events are logged:

| Event Type | Description | Data Logged |
|------------|-------------|-------------|
| `SUPPORT_CONSENT_REQUESTED` | Consent request created | ticket_id, scope, agent_id |
| `SUPPORT_CONSENT_GRANTED` | Customer approved | consent_id, granted_by, expiry |
| `SUPPORT_CONSENT_DENIED` | Customer denied | ticket_id, denied_by |
| `SUPPORT_CONSENT_EXPIRED` | Consent auto-expired | consent_id, expired_at |
| `SUPPORT_CONSENT_REVOKED` | Customer revoked | consent_id, revoked_by, reason |
| `SUPPORT_DATA_ACCESS` | Data accessed under consent | consent_id, scope, records |
| `SUPPORT_DATA_EXPORT` | Data exported under consent | consent_id, export_type, count |
| `SUPPORT_ACCESS_DENIED` | Access denied (no consent) | agent_id, workspace_id, reason |

### 7.2 Audit Record Format

```json
{
  "event_id": "evt_abc123",
  "event_type": "SUPPORT_DATA_ACCESS",
  "timestamp": "2025-01-15T14:30:00Z",
  "consent_id": "consent_xyz789",
  "support_agent_id": "agent_001",
  "workspace_id": "ws_67890",
  "scope_accessed": "telemetry",
  "records_accessed": 150,
  "purpose": "Debug performance issue",
  "ticket_id": "TICKET-12345",
  "ip_address": "10.0.0.1",
  "user_agent": "SupportConsole/1.0"
}
```

### 7.3 Audit Retention

| Audit Type | Retention | Reason |
|------------|-----------|--------|
| Consent events | 7 years | Compliance audit trail |
| Access events | 7 years | Compliance audit trail |
| Export events | 7 years | Compliance audit trail |

---

## 8. Customer Visibility

### 8.1 Self-Service Dashboard

Customers can view in Account Settings > Privacy > Support Access:

- **Active Consents**: Currently active support access grants
- **Pending Requests**: Unapproved consent requests
- **History**: Past consents (granted, denied, revoked, expired)
- **Access Log**: What data was accessed under each consent

### 8.2 Real-time Notifications

Customers receive notifications for:
- New consent requests
- Consent activation
- Data access events (optional, configurable)
- Consent expiry approaching
- Consent expired/revoked

---

## 9. RAW_ORDER_EVENTS Special Handling

### 9.1 RAW Telemetry Consent

For enterprise customers with `RAW_ORDER_EVENTS` enabled:

- Standard support consent does **not** include RAW data
- RAW data access requires:
  1. Enterprise tier verified
  2. Explicit `raw_telemetry` scope in consent request
  3. Additional justification required
  4. Manager approval required
  5. Maximum 24-hour expiry

### 9.2 RAW Access Logging

RAW telemetry access generates enhanced audit:

```json
{
  "event_type": "SUPPORT_RAW_TELEMETRY_ACCESS",
  "enhanced_logging": true,
  "fields_accessed": ["order_events", "fill_events"],
  "records_accessed": 50,
  "manager_approval": "manager_001",
  "justification": "Critical production debugging"
}
```

---

## 10. Training Requirements

### 10.1 Support Staff Training

All support staff must complete:
- Consent policy training (initial + annual)
- Data protection awareness (annual)
- CCEA architecture overview (initial)
- Audit compliance procedures (initial)

### 10.2 Training Records

Training completion tracked in HR system with:
- Staff ID
- Training module
- Completion date
- Expiry date
- Assessment score

---

## 11. Compliance Monitoring

### 11.1 Metrics

| Metric | Target | Frequency |
|--------|--------|-----------|
| Consent request → grant time | < 24 hours | Weekly |
| Access without consent attempts | 0 | Daily |
| Consent expiry on-time handling rate | Target: 100% | Weekly |
| Audit log completeness | Target: 100% | Daily |

### 11.2 Reporting

Monthly report to DPO:
- Total consent requests
- Grant/deny ratio
- Average consent duration
- Revocation rate
- Audit log summary

---

## 12. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | Compliance Team | Initial release - GDPR Phase 1 |

---

## 13. References

- GDPR Regulation (EU) 2016/679 - Articles 6, 7, 28
- EDPB Guidelines on Consent (WP259)
- `docs/legal/PRIVACY_POLICY.md` - Section 7B
- `docs/legal/DPA_TEMPLATE.md` - Section 5.9
- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md`
