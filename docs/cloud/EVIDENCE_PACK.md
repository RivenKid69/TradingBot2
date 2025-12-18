# CCEA Enterprise Evidence Pack

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 16

## Overview

This document describes the Evidence Pack export capability for enterprise customers. The Evidence Pack provides auditable documentation required for regulatory compliance, vendor assessments, and internal governance.

---

## 1. Evidence Pack Contents

### 1.1 Artifact Evidence

| Item | Description | Format |
|------|-------------|--------|
| **Artifact Inventory** | List of all deployed artifacts | JSON/CSV |
| **Artifact Digests** | SHA256 hashes of all artifacts | JSON |
| **Signatures** | Cosign/GPG signatures | Binary |
| **SBOM** | Software Bill of Materials | CycloneDX/SPDX |
| **Provenance** | Build provenance records | JSON |
| **Dependency Locks** | Locked dependency versions | requirements.txt |

### 1.2 Change Management Evidence

| Item | Description | Format |
|------|-------------|--------|
| **Deployment Log** | All deployment events | JSON/CSV |
| **Upgrade History** | Version upgrade records | JSON |
| **Approval Records** | Local approval evidence | JSON |
| **Config Changes** | Configuration change history | JSON |
| **Rollback Events** | Rollback occurrences | JSON |

### 1.3 Incident Evidence

| Item | Description | Format |
|------|-------------|--------|
| **Halt Events** | Kill switch activations | JSON |
| **Halt Reasons** | Detailed halt cause analysis | JSON |
| **Recovery Actions** | Post-halt recovery steps | JSON |
| **Alert History** | Triggered alerts | JSON |
| **Incident Timeline** | Incident chronology | JSON |

### 1.4 Security Evidence

| Item | Description | Format |
|------|-------------|--------|
| **Access Audit Log** | Who accessed what when | JSON/CSV |
| **Authentication Events** | Login/logout records | JSON |
| **Permission Changes** | RBAC modifications | JSON |
| **Agent Registry** | Registered agents and trust state | JSON |
| **Security Policies** | Active security configurations | YAML |

### 1.5 Compliance Documentation

| Item | Description | Format |
|------|-------------|--------|
| **Architecture Diagram** | Cloud/Agent separation | PNG/PDF |
| **Data Flow Diagram** | Data movement visualization | PNG/PDF |
| **Security Controls** | Control descriptions | PDF |
| **Risk Assessment** | Risk analysis summary | PDF |
| **Incident Runbooks** | Response procedures | Markdown |

---

## 2. Export API

### 2.1 Create Export Request (Illustrative Example)

```http
POST /api/v1/enterprise/evidence-pack
Authorization: Bearer <token>
Content-Type: application/json

{
  "workspace_id": "ws_123",
  "date_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-12-31T23:59:59Z"
  },
  "include": [
    "artifacts",
    "deployments",
    "approvals",
    "incidents",
    "audit_logs",
    "security"
  ],
  "format": "zip",
  "encryption": {
    "enabled": true,
    "public_key": "-----BEGIN PUBLIC KEY-----..."
  },
  "destination": {
    "type": "s3",
    "bucket": "customer-evidence-bucket",
    "prefix": "ccea/2025/"
  }
}
```

### 2.2 Export Response (Example - sample data only)

```json
{
  "export_id": "exp_456",
  "status": "processing",
  "estimated_size_mb": 250,
  "estimated_completion": "2025-12-14T11:00:00Z",
  "contents": {
    "artifacts": 150,
    "deployments": 45,
    "approvals": 120,
    "incidents": 8,
    "audit_records": 5000
  }
}
```

### 2.3 Check Export Status

```http
GET /api/v1/enterprise/evidence-pack/exp_456
Authorization: Bearer <token>
```

```json
{
  "export_id": "exp_456",
  "status": "completed",
  "download_url": "https://...",
  "download_expires_at": "2025-12-15T10:00:00Z",
  "checksum": "sha256:abc123...",
  "manifest": {
    "files": [
      {"name": "artifacts.json", "size": 1234567},
      {"name": "deployments.json", "size": 234567},
      {"name": "audit_log.json", "size": 5678901}
    ]
  }
}
```

---

## 3. Evidence Pack Structure

### 3.1 Directory Layout

```
evidence_pack_2025/
├── manifest.json              # Pack metadata and checksums
├── README.md                  # Pack contents description
│
├── artifacts/
│   ├── inventory.json         # All artifacts
│   ├── digests.json           # SHA256 digests
│   ├── signatures/            # Signature files
│   │   ├── artifact_001.sig
│   │   └── artifact_002.sig
│   ├── sbom/                  # SBOM files
│   │   ├── artifact_001.cdx.json
│   │   └── artifact_002.cdx.json
│   └── provenance/            # Build provenance
│       ├── artifact_001.prov.json
│       └── artifact_002.prov.json
│
├── deployments/
│   ├── deployment_log.json    # All deployments
│   ├── state_transitions.json # State machine history
│   └── config_history.json    # Config changes
│
├── approvals/
│   ├── approval_log.json      # All approvals
│   └── evidence/              # Approval evidence hashes
│
├── incidents/
│   ├── halt_events.json       # Kill switch events
│   ├── alerts.json            # Alert history
│   └── recovery_log.json      # Recovery actions
│
├── security/
│   ├── access_audit.json      # Access log
│   ├── auth_events.json       # Authentication log
│   ├── permission_changes.json
│   └── agent_registry.json    # Agent inventory
│
└── documentation/
    ├── architecture.pdf       # Architecture diagrams
    ├── data_flow.pdf          # Data flow diagrams
    ├── security_controls.pdf  # Control descriptions
    └── runbooks/              # Incident runbooks
        ├── kill_switch.md
        ├── recovery.md
        └── escalation.md
```

### 3.2 Manifest File

```json
{
  "pack_version": "1.0.0",
  "generated_at": "2025-12-14T10:00:00Z",
  "workspace_id": "ws_123",
  "organization_id": "org_456",
  "date_range": {
    "start": "2025-01-01T00:00:00Z",
    "end": "2025-12-31T23:59:59Z"
  },
  "statistics": {
    "total_artifacts": 150,
    "total_deployments": 45,
    "total_approvals": 120,
    "total_incidents": 8,
    "total_audit_records": 5000
  },
  "files": [
    {
      "path": "artifacts/inventory.json",
      "size": 1234567,
      "checksum": "sha256:abc..."
    }
  ],
  "integrity": {
    "algorithm": "sha256",
    "pack_checksum": "sha256:xyz..."
  }
}
```

---

## 4. Artifact Evidence Details

### 4.1 Artifact Inventory

```json
{
  "artifacts": [
    {
      "artifact_id": "art_001",
      "strategy_id": "stg_123",
      "version": "2.4.1",
      "digest": "sha256:a1b2c3...",
      "signature_ref": "signatures/artifact_001.sig",
      "sbom_ref": "sbom/artifact_001.cdx.json",
      "provenance_ref": "provenance/artifact_001.prov.json",
      "created_at": "2025-06-15T10:00:00Z",
      "created_by": "user_456",
      "deployed_to_agents": ["ag_001", "ag_002"],
      "deployment_count": 15,
      "last_deployed_at": "2025-12-01T14:30:00Z"
    }
  ]
}
```

### 4.2 SBOM (CycloneDX Format)

```json
{
  "bomFormat": "CycloneDX",
  "specVersion": "1.5",
  "version": 1,
  "metadata": {
    "timestamp": "2025-06-15T10:00:00Z",
    "tools": [{"name": "cyclonedx-python", "version": "3.0.0"}],
    "component": {
      "name": "momentum_strategy",
      "version": "2.4.1"
    }
  },
  "components": [
    {
      "type": "library",
      "name": "numpy",
      "version": "1.24.0",
      "purl": "pkg:pypi/numpy@1.24.0",
      "hashes": [
        {"alg": "SHA-256", "content": "abc123..."}
      ]
    }
  ]
}
```

### 4.3 Provenance Record

```json
{
  "artifact_id": "art_001",
  "build_id": "build_789",
  "provenance": {
    "builder": {
      "id": "ccea-builder-v1",
      "version": "1.2.0"
    },
    "source": {
      "repository": "https://github.com/org/strategies",
      "git_sha": "abc123def456...",
      "branch": "main",
      "tag": "v2.4.1"
    },
    "materials": [
      {
        "uri": "pkg:pypi/torch@2.0.0",
        "digest": {"sha256": "..."}
      }
    ],
    "build": {
      "started_at": "2025-06-15T09:55:00Z",
      "finished_at": "2025-06-15T10:00:00Z",
      "parameters": {
        "python_version": "3.11",
        "cuda_version": "11.8"
      }
    },
    "training_run": {
      "training_run_id": "tr_456",
      "dataset_refs": ["ds_001", "ds_002"],
      "hyperparameters_hash": "sha256:...",
      "final_metrics": {
        "sharpe": 1.8,
        "max_drawdown": -0.15
      }
    }
  }
}
```

---

## 5. Deployment Evidence Details

### 5.1 Deployment Log

```json
{
  "deployments": [
    {
      "deployment_id": "dep_001",
      "artifact_id": "art_001",
      "agent_id": "ag_001",
      "mode": "LIVE",
      "created_at": "2025-06-20T09:00:00Z",
      "created_by": "user_456",
      "state_history": [
        {"state": "CREATED", "at": "2025-06-20T09:00:00Z"},
        {"state": "REQUESTED_START", "at": "2025-06-20T09:00:05Z"},
        {"state": "PENDING_LOCAL_APPROVAL", "at": "2025-06-20T09:00:10Z"},
        {"state": "APPROVED", "at": "2025-06-20T09:05:00Z"},
        {"state": "ACTIVE", "at": "2025-06-20T09:05:30Z"}
      ],
      "approval_id": "apr_001",
      "runs": [
        {"run_id": "run_001", "started_at": "...", "stopped_at": "..."}
      ]
    }
  ]
}
```

### 5.2 Approval Record

```json
{
  "approvals": [
    {
      "approval_id": "apr_001",
      "command_id": "cmd_001",
      "deployment_id": "dep_001",
      "agent_id": "ag_001",
      "decision": "APPROVED",
      "approved_by_local_identity": "local_user:john.doe",
      "approved_at": "2025-06-20T09:05:00Z",
      "evidence_hash": "sha256:evidence_abc...",
      "change_summary": {
        "artifact_version": "2.4.1",
        "mode": "LIVE",
        "risk_limits": {
          "max_position_pct": 10,
          "max_daily_loss_pct": 2
        }
      }
    }
  ]
}
```

---

## 6. Incident Evidence Details

### 6.1 Halt Events

```json
{
  "halt_events": [
    {
      "halt_id": "hlt_001",
      "run_id": "run_001",
      "deployment_id": "dep_001",
      "agent_id": "ag_001",
      "triggered_at": "2025-08-15T14:30:00Z",
      "halt_reason": "MAX_DAILY_LOSS",
      "threshold": -2.0,
      "actual_value": -2.5,
      "actions_taken": [
        {"action": "CANCEL_ORDERS", "count": 3, "success": true},
        {"action": "HALT_RUN", "success": true}
      ],
      "context": {
        "positions_at_halt": 2,
        "open_orders_at_halt": 3,
        "pnl_at_halt": -2.5
      },
      "resolution": {
        "resolved_at": "2025-08-15T15:00:00Z",
        "resolved_by": "user_456",
        "resolution_type": "MANUAL_RESTART"
      }
    }
  ]
}
```

### 6.2 Alert History

```json
{
  "alerts": [
    {
      "alert_id": "alt_001",
      "severity": "WARNING",
      "alert_type": "DRAWDOWN_WARNING",
      "triggered_at": "2025-08-15T14:25:00Z",
      "deployment_id": "dep_001",
      "agent_id": "ag_001",
      "message": "Drawdown approaching limit: -1.8% (limit: -2.0%)",
      "acknowledged_at": "2025-08-15T14:26:00Z",
      "acknowledged_by": "user_456"
    }
  ]
}
```

---

## 7. Security Evidence Details

### 7.1 Access Audit Log

```json
{
  "access_records": [
    {
      "audit_id": "aud_001",
      "timestamp": "2025-12-14T10:00:00Z",
      "user_id": "user_456",
      "action": "view_telemetry",
      "resource_type": "deployment",
      "resource_id": "dep_001",
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "access_level": "normal"
    },
    {
      "audit_id": "aud_002",
      "timestamp": "2025-12-14T11:00:00Z",
      "user_id": "admin_789",
      "action": "break_glass",
      "resource_type": "audit_log",
      "resource_id": "workspace_ws_123",
      "ip_address": "10.0.0.50",
      "access_level": "break_glass",
      "access_reason": "Investigating unauthorized access alert"
    }
  ]
}
```

### 7.2 Agent Registry

```json
{
  "agents": [
    {
      "agent_id": "ag_001",
      "workspace_id": "ws_123",
      "name": "Production Agent 1",
      "status": "ONLINE",
      "trust_state": "ENROLLED",
      "agent_version": "1.2.0",
      "enrolled_at": "2025-01-15T10:00:00Z",
      "last_seen_at": "2025-12-14T10:00:00Z",
      "public_key_fingerprint": "SHA256:abc123...",
      "capabilities": {
        "sandbox": ["docker"],
        "broker_connectors": ["binance", "alpaca"]
      }
    }
  ]
}
```

---

## 8. Automated Export Schedule

### 8.1 Scheduled Exports

```yaml
# Enterprise export schedule
evidence_pack:
  schedule:
    - frequency: monthly
      day_of_month: 1
      include: [artifacts, deployments, approvals]

    - frequency: quarterly
      month: [1, 4, 7, 10]
      day_of_month: 15
      include: [all]
      destination:
        type: s3
        bucket: compliance-archive

    - frequency: on_demand
      trigger: manual
```

### 8.2 Export Notifications

```json
{
  "notifications": {
    "on_export_complete": {
      "email": ["compliance@company.com"],
      "webhook": "https://compliance.company.com/webhook"
    },
    "on_export_failed": {
      "email": ["compliance@company.com", "ops@company.com"],
      "pagerduty": true
    }
  }
}
```

---

## 9. Regulatory Mapping

### 9.1 DORA Requirements

| DORA Article | Evidence Pack Component |
|--------------|------------------------|
| Article 6 (ICT Risk) | Security controls, risk assessment |
| Article 9 (Protection) | Access audit, authentication logs |
| Article 12 (Backup) | Artifact inventory, SBOM |
| Article 15 (ICT Change) | Deployment log, approval records |
| Article 17-19 (Incidents) | Halt events, alert history |

### 9.2 MiFID II RTS 6

| RTS 6 Article | Evidence Pack Component |
|---------------|------------------------|
| Article 5 (Testing) | Provenance, backtest results |
| Article 6 (Deployment) | Deployment log, approvals |
| Article 12 (Records) | Full audit trail |
| Article 17 (Controls) | Security policies, access audit |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial evidence pack per Design Doc |

---

**Related Documentation:**
- [CCEA Overview](../architecture/CCEA_OVERVIEW.md)
- [Enterprise Deployment](./ENTERPRISE.md)
- [Governance](./GOVERNANCE.md)
