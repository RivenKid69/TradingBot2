# Data Subject Access Request (DSAR) Standard Operating Procedure

**Document Type**: Compliance Operations Procedure
**Version**: 1.0.0
**Last Updated**: 2025-12-16
**Owner**: Data Protection Officer
**Scope**: EU-only CCEA Cloud platform

---

## 1. Purpose and Scope

### 1.1 Purpose

This Standard Operating Procedure (SOP) defines the process for handling Data Subject Access Requests (DSARs) under GDPR Articles 12-23. It ensures consistent, compliant, and timely responses to data subject rights requests.

### 1.2 Scope

This SOP applies to all DSAR requests received by the platform, including:
- Access requests (Art. 15)
- Rectification requests (Art. 16)
- Erasure requests (Art. 17)
- Restriction requests (Art. 18)
- Data portability requests (Art. 20)
- Objection requests (Art. 21)

### 1.3 CCEA Boundary Clarification

**IMPORTANT:** Due to the CCEA architecture, DSAR scope is limited to **Cloud-controlled data only**.

| Data Zone | DSAR Scope | Responsibility |
|-----------|------------|----------------|
| **Cloud Zone** | IN SCOPE | Platform Provider |
| **Agent Zone** | OUT OF SCOPE | Customer (Controller) |

Agent-zone data (broker credentials, local execution logs, order/fill data unless transmitted via RAW_ORDER_EVENTS) is not accessible to the Platform Provider and cannot be included in DSAR responses.

---

## 2. Roles and Responsibilities

### 2.1 Data Protection Officer (DPO)

- Overall accountability for DSAR compliance
- Final approval of complex or high-risk requests
- Liaison with supervisory authorities
- Policy interpretation and escalation handling

### 2.2 DSAR Processing Team

- Initial request intake and acknowledgment
- Identity verification
- Data collection and compilation
- Response preparation
- Deadline tracking

### 2.3 Engineering Team

- Technical data extraction
- Automated export generation
- Deletion execution
- System audit log maintenance

### 2.4 Legal Team

- Complex request assessment
- Exemption evaluation
- Third-party notification coordination
- Litigation hold checking

---

## 3. DSAR Workflow

### 3.1 Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DSAR WORKFLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: INTAKE (Day 0-1)                                                  │
│  ─────────────────────────                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Request    │───▶│   Log in     │───▶│   Assign     │                  │
│  │   Received   │    │   System     │    │   Handler    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                 │                           │
│                                                 ▼                           │
│  PHASE 2: VERIFICATION (Day 1-5)               │                           │
│  ───────────────────────────────               │                           │
│  ┌──────────────┐    ┌──────────────┐    ┌─────┴────────┐                  │
│  │   Identity   │───▶│   Request    │───▶│   Valid?     │                  │
│  │   Check      │    │   Validation │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                            YES  │   NO                      │
│                                          ┌──────┴───────┐                   │
│                                          ▼              ▼                   │
│                                    ┌──────────┐  ┌──────────────┐          │
│                                    │ Continue │  │ Request More │          │
│                                    │          │  │ Information  │          │
│                                    └────┬─────┘  └──────────────┘          │
│                                         │                                   │
│  PHASE 3: PROCESSING (Day 5-25)         │                                   │
│  ──────────────────────────────         │                                   │
│  ┌──────────────┐    ┌──────────────┐   │   ┌──────────────┐              │
│  │   Collect    │───▶│   Review     │───┼──▶│   Prepare    │              │
│  │   Data       │    │   Exemptions │   │   │   Response   │              │
│  └──────────────┘    └──────────────┘   │   └──────────────┘              │
│                                         │          │                       │
│                                         │          ▼                       │
│  PHASE 4: RESPONSE (Day 25-30)          │   ┌──────────────┐              │
│  ─────────────────────────────          │   │   QA Review  │              │
│  ┌──────────────┐    ┌──────────────┐   │   └──────┬───────┘              │
│  │   DPO        │───▶│   Send       │◀──┴──────────┘                       │
│  │   Approval   │    │   Response   │                                      │
│  └──────────────┘    └──────────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│  PHASE 5: CLOSURE                                                          │
│  ────────────────                                                          │
│  ┌──────────────┐    ┌──────────────┐                                      │
│  │   Archive    │───▶│   Audit      │                                      │
│  │   Request    │    │   Log        │                                      │
│  └──────────────┘    └──────────────┘                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Timeline Requirements

| Phase | Duration | Deadline (from receipt) | Extension Allowed |
|-------|----------|-------------------------|-------------------|
| Intake | 1 day | Day 1 | No |
| Verification | 5 days | Day 5 | No |
| Processing | 20 days | Day 25 | Yes (see 3.3) |
| Response | 5 days | Day 30 | Yes (see 3.3) |
| **Total** | **30 days** | **Day 30** | **+60 days max** |

### 3.3 Extension Rules

Extension to 60 days (total 90 days from receipt) is permitted when:
- Request is complex (multiple data categories, technical extraction required)
- Multiple requests from same data subject
- Large volume of data involved

**Requirements for extension:**
- Data subject must be notified within 30 days
- Notification must include reason for extension
- Extension must be documented in DSAR record

---

## 4. Detailed Procedures

### 4.1 Phase 1: Intake

**Step 1.1: Request Receipt**

Requests may be received via:
- Email: dpo@[company-domain].com
- Web form: Account Settings > Privacy > DSAR Request
- Postal mail: [Company Address]

**Step 1.2: Initial Logging**

Create DSAR record with:

| Field | Description | Required |
|-------|-------------|----------|
| `dsar_id` | Unique identifier (auto-generated) | Yes |
| `received_at` | Receipt timestamp (UTC) | Yes |
| `channel` | How request was received | Yes |
| `request_type` | Access/Rectification/Erasure/etc. | Yes |
| `requestor_email` | Email from request | Yes |
| `requestor_name` | Name from request | If provided |
| `status` | Current status (RECEIVED) | Yes |
| `deadline_30` | 30-day deadline | Yes |
| `deadline_60` | 60-day extension deadline | Calculated |
| `assigned_to` | Handler assignment | Yes |

**Step 1.3: Acknowledgment**

Send acknowledgment within 24 hours:

```
Subject: Your Data Subject Request [DSAR-XXXX] - Acknowledged

Dear [Name],

We have received your data subject request submitted on [Date].

Request Reference: DSAR-XXXX
Request Type: [Access/Erasure/etc.]
Received: [Timestamp]

We will respond within 30 days of receipt. If we need additional
information to verify your identity, we will contact you.

If you have questions, please reference DSAR-XXXX in all communications.

Regards,
Data Protection Team
```

### 4.2 Phase 2: Verification

**Step 2.1: Identity Verification**

Verification level is proportional to request sensitivity:

| Request Type | Verification Level | Methods |
|--------------|-------------------|---------|
| Access (view only) | Standard | Email confirmation + account match |
| Data export | Enhanced | Email + account credentials + security question |
| Erasure | High | Email + account credentials + 2FA + explicit confirmation |
| Rectification | Standard | Email + account match |

**Verification Methods:**

1. **Email Confirmation**
   - Send verification link to registered email
   - Link valid for 72 hours
   - One-time use token

2. **Account Credentials**
   - Request login to platform
   - Verify active session matches requestor

3. **Security Question**
   - Ask for account-specific information (e.g., workspace name, recent activity)

4. **Two-Factor Authentication**
   - Require 2FA code for high-sensitivity requests

**Step 2.2: Request Validation**

Check:
- Is requestor a data subject (natural person)?
- Does request relate to personal data we control?
- Is request within scope (Cloud data)?
- Are there any exemptions applicable?

**Step 2.3: Clarification (if needed)**

If request is unclear:

```
Subject: Your Data Subject Request [DSAR-XXXX] - Clarification Needed

Dear [Name],

Thank you for your data subject request.

To process your request accurately, we need the following clarification:

[Specific questions]

Please respond within 14 days. If we do not receive clarification,
we will process your request based on our best understanding.

Note: The response deadline is paused while awaiting your clarification.

Regards,
Data Protection Team
```

### 4.3 Phase 3: Processing

**Step 3.1: Data Collection**

For each request type, collect data from:

| Data Source | Access | Export | Erasure | Restriction |
|-------------|--------|--------|---------|-------------|
| User account (PostgreSQL) | Y | Y | Y | Y |
| Organization membership | Y | Y | Y | Y |
| Workspace membership | Y | Y | Y | Y |
| Strategy metadata | Y | Y | Customer decision | Y |
| Telemetry (AGGREGATED) | Y | Y | Y | Y |
| Telemetry (DETAILED) | Y | Y | Y | Y |
| Telemetry (RAW) | Y | Y | Y (if enabled) | Y |
| Command history | Y | Y | Y | Y |
| Approval records | Y | Y | Retain (audit) | Y |
| Access audit logs | Y | Y | Retain (legal) | N/A |
| Support records | Y | Y | Y | Y |

**Step 3.2: Exemption Review**

Check for applicable exemptions:

| Exemption | GDPR Article | Example |
|-----------|--------------|---------|
| Legal claims | Art. 17(3)(e) | Active litigation |
| Legal obligation | Art. 17(3)(b) | Regulatory retention |
| Public interest | Art. 17(3)(d) | Fraud investigation |
| Third-party rights | Art. 15(4) | Other data subjects' data |

**Step 3.3: Data Compilation**

Generate export package:

```
DSAR-XXXX-export/
├── README.md                    # Export explanation
├── metadata.json                # Export metadata (timestamp, scope)
├── checksum.sha256              # Integrity verification
├── account/
│   ├── profile.json            # User profile data
│   └── preferences.json        # User preferences
├── organizations/
│   └── memberships.json        # Org membership records
├── workspaces/
│   └── memberships.json        # Workspace membership records
├── strategies/
│   └── metadata.json           # Strategy metadata (not code unless owned)
├── telemetry/
│   ├── aggregated.json         # AGGREGATED telemetry
│   ├── detailed.json           # DETAILED_NON_SENSITIVE (if enabled)
│   └── raw.json                # RAW_ORDER_EVENTS (if enterprise + enabled)
├── commands/
│   └── history.json            # Command history
├── approvals/
│   └── records.json            # Approval records
├── support/
│   └── interactions.json       # Support ticket history
└── audit/
    └── access_log.json         # Access audit where user is subject
```

**Step 3.4: QA Review**

Verify:
- All requested data categories included
- No third-party personal data included
- Exemptions properly applied and documented
- Export format is correct and readable
- Checksum is valid

### 4.4 Phase 4: Response

**Step 4.1: DPO Approval**

For standard requests:
- DSAR team lead approval sufficient

For complex/high-risk requests:
- DPO review and approval required
- Document approval in DSAR record

**Step 4.2: Response Delivery**

**Access Request Response:**

```
Subject: Your Data Subject Request [DSAR-XXXX] - Complete

Dear [Name],

We have completed processing your access request.

Attached/Linked is your personal data export package.

EXPORT DETAILS:
- Reference: DSAR-XXXX
- Generated: [Timestamp]
- Format: JSON files in ZIP archive
- Checksum (SHA-256): [Hash]

The export includes all personal data we hold about you in our
Cloud systems.

IMPORTANT - CCEA BOUNDARY:
Your request has been processed for all personal data held in our
Cloud systems. Data stored in your local Agent environment (including
broker credentials, local logs, and order data) is under your control
and not accessible to us. Please contact your system administrator
for access to Agent-local data.

The export link will expire in 7 days. Please download promptly.

If you have questions about the data provided, please contact us.

Regards,
Data Protection Team
```

**Erasure Request Response:**

```
Subject: Your Data Subject Request [DSAR-XXXX] - Erasure Complete

Dear [Name],

We have completed your erasure request.

DELETION SUMMARY:
- Reference: DSAR-XXXX
- Completed: [Timestamp]
- Data categories deleted: [List]
- Data retained (legal obligation): [List if any]

Deletion confirmation:
- Primary systems: Complete
- Backups: Will complete within 90 days
- Audit logs: Retained per legal requirement (anonymized where possible)

IMPORTANT - CCEA BOUNDARY:
Deletion applies to Cloud-controlled data only. Agent-zone data
(broker credentials, local logs) must be deleted by you from your
local Agent environment.

If you have questions, please contact us.

Regards,
Data Protection Team
```

### 4.5 Phase 5: Closure

**Step 5.1: Archive Request**

Update DSAR record:
- Status: COMPLETED
- Completed timestamp
- Response method
- Any exemptions applied
- Notes

**Step 5.2: Audit Log**

Create immutable audit record:

```json
{
  "event_type": "DSAR_COMPLETED",
  "dsar_id": "DSAR-XXXX",
  "request_type": "ACCESS",
  "received_at": "2025-01-01T10:00:00Z",
  "completed_at": "2025-01-25T14:30:00Z",
  "days_to_complete": 24,
  "extension_used": false,
  "exemptions_applied": [],
  "data_categories_exported": ["account", "telemetry", "commands"],
  "handled_by": "user_id_xxx",
  "approved_by": "dpo_id_xxx"
}
```

**Step 5.3: Retention**

DSAR records retained for 7 years (compliance audit trail).

---

## 5. Request Type Specific Procedures

### 5.1 Access Request (Art. 15)

**Provide:**
- Confirmation of processing
- Copy of personal data
- Processing purposes
- Data categories
- Recipients
- Retention periods
- Rights information
- Source of data (if not from subject)
- Automated decision-making information

**Response format:** JSON export package

### 5.2 Rectification Request (Art. 16)

**Process:**
1. Verify identity
2. Review requested changes
3. Update data if factually inaccurate
4. Notify third parties (if data shared)
5. Confirm completion

**Limitations:**
- Cannot change audit logs (integrity requirement)
- Cannot change data required for legal purposes

### 5.3 Erasure Request (Art. 17)

**Eligible for erasure:**
- Account profile data
- Strategy metadata (if user-owned)
- Telemetry data (within retention period)
- Support records
- Non-audit logs

**NOT eligible for erasure (exemptions):**
- Audit logs (legal obligation, Art. 17(3)(b))
- Approval records (legal obligation)
- Data under legal hold
- Data needed for legal claims (Art. 17(3)(e))

### 5.4 Restriction Request (Art. 18)

**When applicable:**
- Accuracy contested (until verified)
- Processing unlawful but subject prefers restriction
- No longer needed but subject needs for legal claims
- Subject objected (pending verification)

**Implementation:**
- Mark data as restricted in database
- Block processing except storage
- Notify before lifting restriction

### 5.5 Portability Request (Art. 20)

**Scope:**
- Data provided by subject
- Processed by automated means
- Based on consent or contract

**Format:**
- JSON (machine-readable)
- ZIP archive
- Documented schema

### 5.6 Objection Request (Art. 21)

**Types:**
- Objection to legitimate interest processing
- Objection to direct marketing
- Objection to research/statistics

**Response:**
- Cease processing unless compelling legitimate grounds
- Document assessment
- Notify subject of outcome

---

## 6. Edge Cases and Escalation

### 6.1 Unverifiable Identity

If identity cannot be verified after reasonable attempts:
- Document verification attempts
- Inform requestor of failure
- Do not process request
- Log as CLOSED - IDENTITY UNVERIFIED

### 6.2 Excessive or Manifestly Unfounded Requests

If request is:
- Repetitive (same request, same data, short interval)
- Manifestly unfounded

Options:
- Charge reasonable fee, OR
- Refuse to act

Requirements:
- Document reasoning
- DPO approval required
- Inform requestor of decision and appeal rights

### 6.3 Third-Party Requests

Requests from authorized representatives (lawyers, family):
- Require proof of authorization
- Verify subject's consent
- Apply enhanced verification

### 6.4 Deceased Persons

- GDPR does not apply to deceased persons
- Check local law requirements
- Document decision

### 6.5 Escalation Path

```
Level 1: DSAR Processing Team
    ↓ (if complex/unclear)
Level 2: DSAR Team Lead
    ↓ (if high-risk/legal)
Level 3: DPO
    ↓ (if regulatory/litigation)
Level 4: Legal Counsel
```

---

## 7. Metrics and Reporting

### 7.1 Key Metrics

| Metric | Target | Frequency |
|--------|--------|-----------|
| Average response time | < 25 days | Monthly |
| Requests within 30-day deadline | > 95% | Monthly |
| Extension rate | < 10% | Monthly |
| Identity verification success | > 90% | Monthly |
| Requestor satisfaction | > 80% | Quarterly |

### 7.2 Reporting

Monthly DSAR report to DPO:
- Total requests received
- Requests by type
- Average processing time
- Deadline compliance
- Exemptions applied
- Escalations

### 7.3 Annual Review

Annual review of DSAR process:
- Procedure effectiveness
- Common issues
- Process improvements
- Training needs

---

## 8. Training Requirements

### 8.1 DSAR Processing Team

- GDPR fundamentals (annual)
- DSAR procedure training (initial + annual refresh)
- Identity verification procedures
- Data export tools training
- Exemption assessment

### 8.2 Engineering Team

- Technical data extraction procedures
- Deletion procedures
- Audit logging requirements

### 8.3 All Staff

- DSAR awareness (recognize and route requests)
- Privacy basics

---

## 9. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | Compliance Team | Initial release - GDPR Phase 1 |

---

## 10. References

- GDPR Regulation (EU) 2016/679 - Articles 12-23
- EDPB Guidelines on Transparency (WP260)
- EDPB Guidelines on Right of Access
- EDPB Guidelines on Right to Data Portability (WP242)
- ICO Guidance on Individual Rights
- `docs/legal/PRIVACY_POLICY.md`
- `docs/compliance/GDPR_RISK_SCOPE_MEMO.md`

---

## Appendix A: DSAR Request Form Template

```
DATA SUBJECT ACCESS REQUEST FORM

Personal Information:
- Full Name: _______________
- Email Address: _______________
- Account Username (if applicable): _______________

Request Type (select one or more):
[ ] Access - I want a copy of my personal data
[ ] Rectification - I want to correct inaccurate data
[ ] Erasure - I want my data deleted
[ ] Restriction - I want to limit how my data is processed
[ ] Portability - I want my data in machine-readable format
[ ] Objection - I object to certain processing

Details of Request:
[Free text field for specific details]

Identity Verification:
I confirm I am the data subject or authorized representative.

Signature: _______________
Date: _______________

Submit to: dpo@[company-domain].com
```

---

## Appendix B: Response Time Calculator

```python
from datetime import datetime, timedelta

def calculate_dsar_deadline(received_date: datetime,
                           extended: bool = False) -> dict:
    """Calculate DSAR response deadlines."""
    standard_deadline = received_date + timedelta(days=30)
    extended_deadline = received_date + timedelta(days=90)

    return {
        "received": received_date.isoformat(),
        "standard_deadline": standard_deadline.isoformat(),
        "extended_deadline": extended_deadline.isoformat() if extended else None,
        "days_remaining_standard": (standard_deadline - datetime.now()).days,
        "extension_notification_deadline": (received_date + timedelta(days=30)).isoformat()
    }
```

---

## Appendix C: CCEA Boundary Statement

**Standard text for all DSAR responses:**

> **CCEA Architecture Data Boundary Notice**
>
> Your request has been processed for all personal data held in our Cloud systems. Due to our Cloud-Controlled Execution Architecture (CCEA), certain data categories are stored exclusively in your local Agent environment and are not accessible to us:
>
> - Broker API credentials (API keys, secrets, tokens)
> - Local execution logs (unless explicitly exported to Cloud)
> - Order and fill data (unless RAW_ORDER_EVENTS telemetry is enabled)
> - Local vault contents
> - Position data (unless transmitted via enabled telemetry)
>
> For access to Agent-local data, please contact your system administrator or access your Agent's local storage directly. We cannot export or delete data that we do not receive or store.
