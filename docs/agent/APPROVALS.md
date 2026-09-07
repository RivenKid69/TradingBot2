# Local Approval System

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

The Local Approval System is designed to ensure that **trading-impacting changes require explicit local approval** before execution. The architecture is designed so that Cloud cannot bypass this requirement (verify via protocol specification and tests).

## Security Design Commitments

```
Local Approval DESIGN COMMITMENTS (enforced at architecture level):
  - Trading-impacting changes ALWAYS require local approval
  - Cloud CANNOT auto-approve on behalf of user
  - Approval evidence is designed to be cryptographically signed (verify via implementation)
  - Approval audit trail is maintained locally
  - Stop/Pause commands do NOT require approval (safety)
```

---

## Approval Categories

### Trading-Impacting (ALWAYS Require Approval)

| Category | Changes |
|----------|---------|
| **Strategy/Model** | Artifact upgrade, model version change |
| **Universe** | Symbols added/removed, asset class change |
| **Execution** | Execution parameters, slippage settings |
| **Risk** | Risk limits, position limits |
| **Mode** | Paper → Live mode switch |
| **Schedule** | Trading schedule, blackout windows |
| **Account** | Broker account change, adapter config |

### Data-Sensitive (Require Approval)

| Change | Reason |
|--------|--------|
| Log export | May contain sensitive data |
| Telemetry level increase | More data transmitted |
| Diagnostic dump | Contains internal state |

### Safety Commands (NO Approval Needed)

| Command | Reason |
|---------|--------|
| STOP | Safety - must execute immediately |
| PAUSE | Safety - must execute immediately |
| KILL_SWITCH | Emergency - must execute immediately |

---

## Approval Workflow

### Standard Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APPROVAL WORKFLOW                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Cloud                           Agent                        User          │
│    │                               │                            │           │
│    │  1. REQUEST_START_RUN         │                            │           │
│    │  (artifact_digest: sha256:x)  │                            │           │
│    │──────────────────────────────▶│                            │           │
│    │                               │                            │           │
│    │                               │  2. Show approval request  │           │
│    │                               │──────────────────────────▶ │           │
│    │                               │                            │           │
│    │                               │  3. Review changes:        │           │
│    │                               │     - New artifact digest  │           │
│    │                               │     - Risk profile         │           │
│    │                               │     - Universe changes     │           │
│    │                               │                            │           │
│    │                               │  4. APPROVE / REJECT       │           │
│    │                               │◀────────────────────────── │           │
│    │                               │                            │           │
│    │  5. COMMAND_APPROVAL          │                            │           │
│    │  (approved: true,             │                            │           │
│    │   evidence_hash: sha256:y)    │                            │           │
│    │◀──────────────────────────────│                            │           │
│    │                               │                            │           │
│    │                               │  6. Execute command        │           │
│    │                               │  (pull artifact, start)    │           │
│    │                               │                            │           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Approval Request Display

When a trading-impacting command arrives, the agent displays:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     APPROVAL REQUIRED                                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Command: REQUEST_START_RUN                                                ║
║  Request ID: req_abc123                                                    ║
║  Timestamp: 2025-12-14T12:00:00Z                                          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  CHANGES:                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │  Artifact:                                                          │  ║
║  │    Current:  sha256:abc123...                                       │  ║
║  │    New:      sha256:def456...                                       │  ║
║  │                                                                      │  ║
║  │  Risk Profile:                                                       │  ║
║  │    max_position_pct: 5% → 10%  ⚠️ INCREASED                         │  ║
║  │    max_daily_loss_pct: 2% (unchanged)                               │  ║
║  │                                                                      │  ║
║  │  Universe:                                                          │  ║
║  │    + ETHUSDT (added)                                                │  ║
║  │    - DOGEUSDT (removed)                                             │  ║
║  │                                                                      │  ║
║  │  Mode: paper → LIVE  ⚠️ LIVE TRADING                                │  ║
║  └─────────────────────────────────────────────────────────────────────┘  ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  [A]pprove    [R]eject    [D]etails    [?]Help                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Approval Interfaces

### CLI Interface

```bash
# List pending approvals
ccea-agent pending

# Approve specific request
ccea-agent approve --request req_abc123

# Approve with comment
ccea-agent approve --request req_abc123 --comment "Reviewed and approved"

# Reject request
ccea-agent reject --request req_abc123 --reason "Risk too high"

# Show request details
ccea-agent pending --request req_abc123 --details
```

### Interactive CLI

When agent runs in foreground:

```bash
ccea-agent start --foreground --interactive
```

Approval prompts appear inline.

### GUI Interface (Optional)

Desktop notification with approval dialog:

```bash
# Enable GUI notifications
ccea-agent config set approval.gui_notifications true
```

### API Interface

For programmatic approval (e.g., custom approval workflows):

```python
from ccea_agent.approval import ApprovalManager

mgr = ApprovalManager()

# List pending
pending = mgr.list_pending()

# Get details
request = mgr.get_request("req_abc123")

# Approve
mgr.approve(
    request_id="req_abc123",
    comment="Approved after review"
)

# Reject
mgr.reject(
    request_id="req_abc123",
    reason="Exceeds risk tolerance"
)
```

---

## Auto-Approve (Use with Caution)

For specific low-risk changes, auto-approve can be configured:

```yaml
policy:
  auto_approve:
    enabled: true
    whitelist_changes:
      # Only auto-approve non-trading-impacting config changes
      - type: UPDATE_CONFIG
        conditions:
          - field: "logging.level"
          - field: "telemetry.level"
            max_level: DETAILED_NON_SENSITIVE

    # Safety limits
    max_auto_approve_value_change_pct: 1

    # Never auto-approve these
    never_auto_approve:
      - PAPER_TO_LIVE
      - RISK_LIMIT_INCREASE
      - NEW_SYMBOL_ADDITION
```

**WARNING:** Auto-approve should be used sparingly and NEVER for:

- Paper to live mode changes
- Risk limit increases
- Adding new trading symbols
- Broker account changes

---

## Evidence Generation

Every approval generates cryptographic evidence:

### Evidence Hash Components

```python
evidence = {
    "request_id": "req_abc123",
    "command_type": "REQUEST_START_RUN",
    "command_payload_hash": "sha256:...",
    "approval_timestamp": "2025-12-14T12:05:00Z",
    "approver": "local_user",
    "approval_method": "cli",
    "agent_id": "agent_xyz",
    "agent_version": "1.0.0",
    "comment": "Approved after review"
}

evidence_hash = sha256(canonical_json(evidence))
# Result: sha256:abc123...
```

### Evidence Storage

Evidence is stored locally and sent to Cloud:

```
~/.ccea/approvals/
├── 2025-12-14/
│   ├── req_abc123.json      # Approval record
│   ├── req_abc123.sig       # Signature
│   └── req_def456.json
└── index.db                  # SQLite index
```

### Evidence Verification

```bash
# Verify approval evidence
ccea-agent approval verify --request req_abc123

# Export evidence for audit
ccea-agent approval export \
  --from 2025-01-01 \
  --to 2025-12-31 \
  --output approvals-2025.json
```

---

## Approval Timeout

Requests expire if not acted upon:

```yaml
policy:
  approval_timeout_hours: 24  # Default
```

After timeout:

- Request marked as EXPIRED
- Cloud notified
- New request must be sent

---

## Audit Trail

All approval actions are logged:

```bash
# View approval audit log
ccea-agent approval audit-log

# Output:
# TIMESTAMP            REQUEST       ACTION    APPROVER    DETAILS
# 2025-12-14T12:05:00  req_abc123    APPROVED  local_user  "Reviewed"
# 2025-12-14T10:30:00  req_xyz789    REJECTED  local_user  "Risk too high"
# 2025-12-14T09:00:00  req_def456    EXPIRED   -           "Timeout 24h"
```

### Audit Log Schema

```json
{
  "timestamp": "2025-12-14T12:05:00Z",
  "request_id": "req_abc123",
  "action": "APPROVED",
  "approver": "local_user",
  "approval_method": "cli",
  "command_type": "REQUEST_START_RUN",
  "changes_summary": {
    "artifact_changed": true,
    "risk_changed": true,
    "mode_changed": true
  },
  "evidence_hash": "sha256:abc123...",
  "comment": "Reviewed and approved"
}
```

---

## Multi-User Approval (Enterprise)

For enterprise environments with multiple operators:

```yaml
policy:
  multi_approval:
    enabled: true
    required_approvers: 2
    approver_roles:
      - admin
      - risk_manager
    timeout_hours: 4
```

---

## Configuration Reference

```yaml
# approval section in agent.yaml
policy:
  # Approval settings
  approval:
    # Timeout for pending requests
    timeout_hours: 24

    # GUI notifications
    gui_notifications: true

    # Sound alert for new requests
    sound_alert: true

    # Email notifications (requires SMTP config)
    email_notifications: false

  # Auto-approve settings
  auto_approve:
    enabled: false
    whitelist_changes: []
    max_auto_approve_value_change_pct: 1
    never_auto_approve:
      - PAPER_TO_LIVE
      - RISK_LIMIT_INCREASE

  # Multi-approval (enterprise)
  multi_approval:
    enabled: false
    required_approvers: 1
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No approval prompt | Agent not interactive | Use `--foreground --interactive` |
| Approval not reaching Cloud | Network issue | Check connectivity |
| Evidence verification failed | Tampered data | Investigate |
| Auto-approve not working | Conditions not met | Check whitelist config |

### Diagnostics

```bash
# Check pending approvals
ccea-agent pending --verbose

# Verify approval system
ccea-agent doctor --check approval

# View approval configuration
ccea-agent config show approval
```

---

**Related Documentation:**

- [Risk Controls](./RISK_CONTROLS.md)
- [Installation](./INSTALLATION.md)
- [CCEA Overview](../CCEA_OVERVIEW.md)
