# Protocol Change Review Checklist and Journal Format

**Document Version**: 1.0.0
**Effective Date**: 2025-12-16
**Classification**: INTERNAL / SECURITY
**Related Documents**:
- `docs/compliance/GDPR_CCEA_IMPLEMENTATION_PLAN.md`
- `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt`
- `docs/design/CCEA_CLOUD/CI_GUARDRAILS.md`
- `docs/schemas/protocol_messages.schema.json`

## 1. Overview

This document defines the mandatory security review process for all Cloud-Agent protocol changes. Per Design Doc requirements, new protocol command types require security review and auditable approval before merge.

**Reference**: `docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L1043`, `#L960`

### 1.1 Scope

This process applies to:

1. **New Command Types**: Any addition to `CommandType` enum
2. **New Message Types**: Any addition to `MessageType` enum
3. **Field Additions**: New fields in existing message types
4. **Schema Changes**: Modifications to `protocol_messages.schema.json`
5. **Behavioral Changes**: Changes to command semantics or flow

### 1.2 Governance Principle

```
┌─────────────────────────────────────────────────────────────┐
│                 PROTOCOL CHANGE PRINCIPLE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  "No protocol change shall be merged without:                │
│                                                              │
│   1. Documented security review                              │
│   2. Explicit approval from security-designated reviewer     │
│   3. Journal entry recording the change and approval"        │
│                                                              │
│  Reference: Design Doc 10.5/19.2                             │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Security Review Checklist

### 2.1 Pre-Submission Checklist (Author)

Before requesting security review, the author MUST complete:

```markdown
## Protocol Change Pre-Submission Checklist

### Change Identification
- [ ] Change ID assigned: `PROT-YYYY-NNN`
- [ ] PR number: #____
- [ ] Author: @____
- [ ] Date submitted: YYYY-MM-DD

### Change Classification
- [ ] Change type: [ ] New Command / [ ] New Message / [ ] Field Addition / [ ] Schema Change / [ ] Behavioral
- [ ] Change class: [ ] TRADING_IMPACTING / [ ] NON_TRADING_IMPACTING
- [ ] Breaking change: [ ] Yes / [ ] No
- [ ] Backward compatible: [ ] Yes / [ ] No

### Documentation
- [ ] Design document updated or created
- [ ] Schema documentation updated
- [ ] API documentation updated
- [ ] Migration guide (if breaking)

### Security Self-Assessment
- [ ] No order-like payloads introduced in Cloud->Agent direction
- [ ] No credentials/secrets in message fields
- [ ] No PII fields without redaction requirement
- [ ] No new attack surface without mitigation
- [ ] Telemetry impact assessed
- [ ] Rate limiting considered
- [ ] Authentication/authorization requirements defined

### Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Schema validation tests pass
- [ ] Guardrail tests pass
- [ ] No test regressions

### CI Guardrails
- [ ] BT-002 (no-order-payloads-in-schema) passes
- [ ] PM-001 (schema-validation) passes
- [ ] PM-002 (protocol-allowlist) passes
- [ ] All other guardrails green
```

### 2.2 Security Review Checklist (Reviewer)

Security reviewer MUST evaluate:

```markdown
## Protocol Change Security Review

### Review Metadata
- [ ] Review ID: `REV-YYYY-NNN`
- [ ] Change ID: `PROT-YYYY-NNN`
- [ ] Reviewer: @____
- [ ] Review date: YYYY-MM-DD

### 1. CCEA Boundary Compliance
- [ ] **NO order-like payloads in Cloud->Agent commands**
  - Verified: No side/qty/price/intent/signal fields
  - Verified: No execute/place/submit/cancel order commands
- [ ] **Respects Cloud/Agent zone separation**
  - Cloud only sends lifecycle requests
  - Agent controls all trading execution
- [ ] **Telemetry compliance**
  - New telemetry fields classified by level
  - Redaction requirements documented

### 2. Data Security
- [ ] **No credential exposure risk**
  - No API keys, secrets, tokens in protocol
  - No environment variable fields
- [ ] **No PII leakage**
  - PII fields identified and redaction enforced
  - EU residency maintained
- [ ] **Data minimization**
  - Only necessary data transmitted
  - Aggregation preferred over raw data

### 3. Authentication & Authorization
- [ ] **Message signing**
  - New commands require signature
  - Signature verification documented
- [ ] **Authorization model**
  - Required permissions documented
  - RBAC integration verified
- [ ] **Replay protection**
  - Idempotency keys required where applicable
  - Timestamp validation documented

### 4. Protocol Security
- [ ] **Version compatibility**
  - Backward compatibility maintained (or migration path)
  - Version negotiation updated
- [ ] **Input validation**
  - Schema constraints defined
  - Validation implemented
- [ ] **Rate limiting**
  - Rate limits defined for new endpoints
  - Abuse scenarios considered

### 5. Operational Security
- [ ] **Audit logging**
  - New operations logged
  - Audit fields defined
- [ ] **Monitoring**
  - Metrics defined
  - Alerts configured
- [ ] **Rollback plan**
  - Rollback procedure documented
  - Feature flag available if needed

### 6. Risk Assessment
- [ ] **Attack surface analysis**
  - New attack vectors identified
  - Mitigations documented
- [ ] **Impact assessment**
  - Security: [ ] None / [ ] Low / [ ] Medium / [ ] High / [ ] Critical
  - Privacy: [ ] None / [ ] Low / [ ] Medium / [ ] High / [ ] Critical
  - Availability: [ ] None / [ ] Low / [ ] Medium / [ ] High / [ ] Critical

### Review Decision
- [ ] **APPROVED** - No security concerns
- [ ] **APPROVED WITH CONDITIONS** - Requires changes before merge
- [ ] **REJECTED** - Security concerns require redesign

### Conditions (if applicable)
[List any conditions that must be met before merge]

### Reviewer Signature
Reviewer: @____
Date: YYYY-MM-DD
Signature hash: sha256:____
```

---

## 3. Journal Format

All protocol changes MUST be recorded in the change journal.

### 3.1 Journal File Location

```
docs/compliance/protocol_change_journal.json
```

### 3.2 Journal Entry Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Protocol Change Journal Entry",
  "type": "object",
  "required": [
    "change_id",
    "timestamp",
    "change_type",
    "author",
    "reviewer",
    "approval_status",
    "summary"
  ],
  "properties": {
    "change_id": {
      "type": "string",
      "pattern": "^PROT-[0-9]{4}-[0-9]{3}$",
      "description": "Unique change identifier (PROT-YYYY-NNN)"
    },
    "review_id": {
      "type": "string",
      "pattern": "^REV-[0-9]{4}-[0-9]{3}$",
      "description": "Security review identifier (REV-YYYY-NNN)"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of approval"
    },
    "change_type": {
      "type": "string",
      "enum": [
        "NEW_COMMAND",
        "NEW_MESSAGE",
        "FIELD_ADDITION",
        "FIELD_REMOVAL",
        "SCHEMA_CHANGE",
        "BEHAVIORAL_CHANGE"
      ]
    },
    "change_class": {
      "type": "string",
      "enum": ["TRADING_IMPACTING", "NON_TRADING_IMPACTING"]
    },
    "summary": {
      "type": "string",
      "minLength": 10,
      "maxLength": 500,
      "description": "Brief description of the change"
    },
    "author": {
      "type": "object",
      "required": ["name", "email"],
      "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"}
      }
    },
    "reviewer": {
      "type": "object",
      "required": ["name", "email", "role"],
      "properties": {
        "name": {"type": "string"},
        "email": {"type": "string", "format": "email"},
        "role": {"type": "string", "enum": ["SECURITY_ENGINEER", "SECURITY_LEAD", "CISO"]}
      }
    },
    "approval_status": {
      "type": "string",
      "enum": ["APPROVED", "APPROVED_WITH_CONDITIONS", "REJECTED"]
    },
    "conditions": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Conditions that must be met (if APPROVED_WITH_CONDITIONS)"
    },
    "conditions_met": {
      "type": "boolean",
      "description": "Whether all conditions have been satisfied"
    },
    "pr_number": {
      "type": "integer",
      "description": "GitHub PR number"
    },
    "commit_hash": {
      "type": "string",
      "pattern": "^[a-f0-9]{40}$",
      "description": "Git commit hash of the merged change"
    },
    "schema_version_before": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"
    },
    "schema_version_after": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$"
    },
    "breaking_change": {
      "type": "boolean"
    },
    "security_impact": {
      "type": "string",
      "enum": ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    },
    "privacy_impact": {
      "type": "string",
      "enum": ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
    },
    "evidence_hashes": {
      "type": "object",
      "properties": {
        "checklist_hash": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$"
        },
        "review_hash": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$"
        },
        "diff_hash": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$"
        }
      }
    },
    "design_doc_reference": {
      "type": "string",
      "description": "Reference to design document section"
    },
    "notes": {
      "type": "string",
      "description": "Additional notes or context"
    }
  }
}
```

### 3.3 Journal Entry Example

```json
{
  "change_id": "PROT-2025-001",
  "review_id": "REV-2025-001",
  "timestamp": "2025-12-16T10:30:00Z",
  "change_type": "NEW_COMMAND",
  "change_class": "NON_TRADING_IMPACTING",
  "summary": "Added REQUEST_EXPORT_LOGS command for DSAR compliance support",
  "author": {
    "name": "Jane Developer",
    "email": "jane@ccea.io"
  },
  "reviewer": {
    "name": "John Security",
    "email": "john@ccea.io",
    "role": "SECURITY_LEAD"
  },
  "approval_status": "APPROVED",
  "conditions": [],
  "conditions_met": true,
  "pr_number": 1234,
  "commit_hash": "abc123def456789012345678901234567890abcd",
  "schema_version_before": "1.0.0",
  "schema_version_after": "1.1.0",
  "breaking_change": false,
  "security_impact": "LOW",
  "privacy_impact": "LOW",
  "evidence_hashes": {
    "checklist_hash": "sha256:abc123...",
    "review_hash": "sha256:def456...",
    "diff_hash": "sha256:789012..."
  },
  "design_doc_reference": "Design_Doc_CCEA_Cloud.txt#L1651",
  "notes": "Command supports redacted log export for GDPR DSAR compliance"
}
```

---

## 4. CI Enforcement

### 4.1 Protocol Change Detection

CI automatically detects protocol changes by monitoring:

```yaml
protocol_change_detection:
  watch_paths:
    - "ccea/models/protocol.py"
    - "docs/schemas/protocol_messages.schema.json"
    - "ccea/protocol/**"
    - "packages/cloud/control_plane/services/command_service.py"

  watch_patterns:
    - "class.*Command.*BaseCommand"
    - "class.*Message.*BaseMessage"
    - "CommandType\\."
    - "MessageType\\."

  trigger_review: true
```

### 4.2 Review Gate Implementation

```python
# ccea/guardrails/protocol_review_check.py

"""
Protocol Review Gate - CI Enforcement.

Ensures new protocol changes have security review approval.

Reference: Design Doc 10.5/19.2
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Tuple

JOURNAL_PATH = Path("docs/compliance/protocol_change_journal.json")
PROTOCOL_PATHS = [
    "ccea/models/protocol.py",
    "docs/schemas/protocol_messages.schema.json",
]


def load_journal() -> list:
    """Load protocol change journal."""
    if not JOURNAL_PATH.exists():
        return []
    with open(JOURNAL_PATH) as f:
        return json.load(f)


def get_changed_files() -> list:
    """Get files changed in current PR."""
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().split("\n")


def check_protocol_changes() -> Tuple[bool, Optional[str]]:
    """
    Check if protocol changes have security review approval.

    Returns:
        (passed, error_message)
    """
    changed_files = get_changed_files()

    # Check if any protocol files changed
    protocol_changes = [
        f for f in changed_files
        if any(p in f for p in PROTOCOL_PATHS)
    ]

    if not protocol_changes:
        return True, None

    # Protocol changes detected - check for journal entry
    journal = load_journal()

    # Get current PR commit
    import subprocess
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
    )
    current_commit = result.stdout.strip()

    # Look for approved entry
    for entry in journal:
        if entry.get("commit_hash") == current_commit:
            if entry.get("approval_status") in ("APPROVED", "APPROVED_WITH_CONDITIONS"):
                if entry.get("approval_status") == "APPROVED_WITH_CONDITIONS":
                    if not entry.get("conditions_met", False):
                        return False, (
                            f"Protocol change {entry['change_id']} has unmet conditions. "
                            "All conditions must be satisfied before merge."
                        )
                return True, None

    # No approval found
    return False, (
        "Protocol change detected but no security review approval found.\n"
        "Files changed:\n"
        + "\n".join(f"  - {f}" for f in protocol_changes)
        + "\n\n"
        "Required actions:\n"
        "  1. Complete security review checklist\n"
        "  2. Request review from security team\n"
        "  3. Add journal entry with approval\n"
        "  4. Re-run CI\n"
        "\n"
        "Reference: docs/compliance/PROTOCOL_CHANGE_REVIEW.md"
    )


def main() -> int:
    """Main entry point for CI."""
    passed, error = check_protocol_changes()

    if passed:
        print("[PASS] Protocol change review check passed")
        return 0
    else:
        print(f"[FAIL] Protocol change review check failed:\n{error}")
        return 1


if __name__ == "__main__":
    exit(main())
```

### 4.3 CI Workflow Integration

```yaml
# .github/workflows/build-and-test.yml (addition)

  protocol-review-check:
    name: Protocol Change Review Gate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Check Protocol Changes
        run: python ccea/guardrails/protocol_review_check.py
```

---

## 5. Review Workflow

### 5.1 Process Flow

```
┌──────────────┐
│ Author       │
│ Creates PR   │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ CI Detects       │
│ Protocol Change  │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Author Completes │
│ Pre-Submission   │
│ Checklist        │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Author Requests  │
│ Security Review  │
│ (@security-team) │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Security         │──REJECT──▶ Author Redesigns
│ Reviews PR       │
└──────┬───────────┘
       │APPROVE
       ▼
┌──────────────────┐
│ Reviewer Adds    │
│ Journal Entry    │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ CI Re-runs       │
│ Gate Passes      │
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ PR Merged        │
│ with Audit Trail │
└──────────────────┘
```

### 5.2 Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| **Author** | Complete pre-submission checklist, request review, address feedback |
| **Security Reviewer** | Evaluate security impact, complete review checklist, approve/reject |
| **Security Lead** | Escalation point for HIGH/CRITICAL changes |
| **CISO** | Final approval for CRITICAL security impact changes |

### 5.3 SLA

| Security Impact | Review SLA | Reviewer Level |
|-----------------|------------|----------------|
| NONE/LOW | 2 business days | Security Engineer |
| MEDIUM | 3 business days | Security Engineer |
| HIGH | 5 business days | Security Lead |
| CRITICAL | 10 business days | CISO |

---

## 6. Evidence Pack Integration

Protocol change journal entries are included in the evidence pack for audits.

### 6.1 Export Format

```json
{
  "export_type": "PROTOCOL_CHANGE_JOURNAL",
  "export_timestamp": "2025-12-16T10:30:00Z",
  "time_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-12-16T23:59:59Z"
  },
  "entry_count": 15,
  "entries": [
    // Journal entries within time range
  ],
  "summary": {
    "total_changes": 15,
    "by_type": {
      "NEW_COMMAND": 3,
      "FIELD_ADDITION": 8,
      "SCHEMA_CHANGE": 4
    },
    "by_impact": {
      "NONE": 5,
      "LOW": 7,
      "MEDIUM": 2,
      "HIGH": 1,
      "CRITICAL": 0
    },
    "approval_rate": 1.0,
    "avg_review_days": 2.3
  },
  "checksum": "sha256:abc123..."
}
```

---

## 7. Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial version for GDPR Phase 2 |
