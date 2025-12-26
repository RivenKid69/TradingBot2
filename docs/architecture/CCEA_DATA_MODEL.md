# CCEA Data Model Reference

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 6

## Overview

This document defines the core data model entities for the CCEA Platform. All entities follow strict tenant isolation and support the Cloud/Agent separation architecture.

---

## 1. Core Entities

### 1.1 Organization

Top-level tenant entity.

```sql
CREATE TABLE organizations (
    organization_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE NOT NULL,
    tier VARCHAR(50) NOT NULL DEFAULT 'retail',  -- retail, pro, enterprise
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);
```

| Field | Type | Description |
|-------|------|-------------|
| `organization_id` | UUID | Primary key |
| `name` | string | Organization display name |
| `slug` | string | URL-safe identifier |
| `tier` | enum | Subscription tier (retail/pro/enterprise) |
| `metadata` | JSONB | Additional organization settings |

### 1.2 Workspace

Isolated workspace within organization (tenant boundary).

```sql
CREATE TABLE workspaces (
    workspace_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(organization_id),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) NOT NULL,
    settings JSONB DEFAULT '{}',
    data_residency VARCHAR(10) DEFAULT 'eu',  -- eu, us, ap
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(organization_id, slug)
);

-- Row-Level Security for tenant isolation
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
CREATE POLICY workspace_isolation ON workspaces
    USING (workspace_id = current_setting('app.current_workspace_id')::uuid);
```

| Field | Type | Description |
|-------|------|-------------|
| `workspace_id` | UUID | Primary key (tenant boundary) |
| `organization_id` | UUID | Parent organization |
| `data_residency` | enum | Data storage region (eu/us/ap) |
| `settings` | JSONB | Workspace-level settings |

### 1.3 User

Platform user with role-based access.

```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255),  -- NULL for SSO users
    mfa_enabled BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login_at TIMESTAMP WITH TIME ZONE
);

CREATE TABLE workspace_memberships (
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    role VARCHAR(50) NOT NULL,  -- owner, admin, developer, viewer
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (workspace_id, user_id)
);
```

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | UUID | Primary key |
| `email` | string | Unique email address |
| `mfa_enabled` | boolean | Multi-factor authentication status |
| `role` | enum | Role within workspace (owner/admin/developer/viewer) |

### 1.4 Role / Permission

RBAC implementation.

```sql
CREATE TABLE roles (
    role_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    name VARCHAR(100) NOT NULL,
    permissions JSONB NOT NULL DEFAULT '[]',
    is_system BOOLEAN DEFAULT FALSE,  -- System roles cannot be deleted
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Standard Permissions:**
```json
{
  "permissions": [
    "strategy:read",
    "strategy:write",
    "strategy:deploy",
    "agent:manage",
    "agent:view",
    "deployment:start",
    "deployment:stop",
    "telemetry:view",
    "telemetry:export",
    "settings:manage"
  ]
}
```

---

## 2. Strategy Entities

### 2.1 Strategy

Trading strategy definition.

```sql
CREATE TABLE strategies (
    strategy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    repository_url VARCHAR(500),  -- Git repository
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    archived_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(workspace_id, name)
);
```

| Field | Type | Description |
|-------|------|-------------|
| `strategy_id` | UUID | Primary key |
| `workspace_id` | UUID | Owning workspace |
| `name` | string | Strategy name |
| `repository_url` | string | Git repository URL |

### 2.2 StrategyVersion

Versioned strategy code.

```sql
CREATE TABLE strategy_versions (
    version_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id UUID NOT NULL REFERENCES strategies(strategy_id),
    version_number VARCHAR(50) NOT NULL,  -- SemVer: 1.2.3
    git_sha VARCHAR(64),
    changelog TEXT,
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(strategy_id, version_number)
);
```

| Field | Type | Description |
|-------|------|-------------|
| `version_id` | UUID | Primary key |
| `strategy_id` | UUID | Parent strategy |
| `version_number` | string | Semantic version (1.2.3) |
| `git_sha` | string | Git commit hash |

---

## 3. Build & Artifact Entities

### 3.1 Build

Artifact build record with full provenance.

```sql
CREATE TABLE builds (
    build_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_version_id UUID NOT NULL REFERENCES strategy_versions(version_id),
    artifact_digest VARCHAR(100) NOT NULL,  -- sha256:...
    signature_ref VARCHAR(500),  -- Signature storage reference
    sbom_ref VARCHAR(500),  -- SBOM storage reference
    change_class VARCHAR(20) NOT NULL,  -- TRADING_IMPACTING, NON_IMPACTING
    provenance JSONB NOT NULL,  -- Build provenance
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(artifact_digest)
);
```

| Field | Type | Description |
|-------|------|-------------|
| `build_id` | UUID | Primary key |
| `strategy_version_id` | UUID | Source version |
| `artifact_digest` | string | SHA256 content hash (`sha256:abc...`) |
| `signature_ref` | string | Cosign/GPG signature reference |
| `sbom_ref` | string | SBOM (CycloneDX/SPDX) reference |
| `change_class` | enum | TRADING_IMPACTING / NON_IMPACTING |
| `provenance` | JSONB | Full build provenance |

**Provenance Schema:**
```json
{
  "git_sha": "abc123...",
  "dataset_refs": ["dataset_1", "dataset_2"],
  "training_run_id": "tr_789",
  "params_hash": "sha256:...",
  "built_at": "2025-12-12T10:00:00Z",
  "built_by": "user_456",
  "builder_version": "1.2.0"
}
```

### 3.2 Artifact

Registry artifact reference.

```sql
CREATE TABLE artifacts (
    artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID NOT NULL REFERENCES builds(build_id),
    registry_url VARCHAR(500) NOT NULL,  -- OCI registry URL
    digest VARCHAR(100) NOT NULL,  -- sha256:...
    size_bytes BIGINT,
    manifest JSONB NOT NULL,  -- Artifact manifest
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**Manifest Schema (per Design Doc 8.2):**
```json
{
  "schema_version": "1.0",
  "strategy": {
    "strategy_id": "stg_123",
    "version": "2.4.1",
    "git_sha": "abc123...",
    "entrypoint": "my_pkg.strategy:MyStrategy"
  },
  "runtime": {
    "python": "3.11",
    "requires_gpu": false,
    "sandbox": "docker"
  },
  "dependencies": {
    "lock_digest": "sha256:..."
  },
  "artifacts": {
    "model_refs": [
      {"name": "policy.pt", "digest": "sha256:..."}
    ],
    "data_contract": {
      "symbols": ["AAPL", "MSFT"],
      "timeframe": "1m",
      "features": ["rsi", "macd", "vwap"]
    }
  },
  "permissions": {
    "network_egress": "deny_by_default",
    "allowed_hosts": ["api.broker.example"],
    "filesystem": "read_only_except_tmp"
  },
  "risk_profile_suggested": {
    "max_daily_loss": 100.0,
    "max_position_usd": 10000.0,
    "max_order_rate_per_min": 10
  },
  "change_class": "TRADING_IMPACTING",
  "provenance": {
    "built_at": "2025-12-12T10:00:00Z",
    "built_by": "user_456",
    "training_run_id": "tr_789",
    "params_hash": "sha256:..."
  }
}
```

---

## 4. Agent Entities

### 4.1 Agent

Registered agent instance.

```sql
CREATE TABLE agents (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    name VARCHAR(255),
    public_key TEXT NOT NULL,  -- Agent's device public key
    agent_version VARCHAR(50) NOT NULL,
    last_seen_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) NOT NULL DEFAULT 'OFFLINE',  -- ONLINE, OFFLINE
    trust_state VARCHAR(20) NOT NULL DEFAULT 'ENROLLED',  -- ENROLLED, REVOKED
    capabilities JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    revoked_at TIMESTAMP WITH TIME ZONE
);
```

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | UUID | Primary key |
| `workspace_id` | UUID | Owning workspace |
| `public_key` | text | Device public key for authentication |
| `agent_version` | string | Agent software version |
| `last_seen_at` | timestamp | Last heartbeat time |
| `status` | enum | ONLINE / OFFLINE |
| `trust_state` | enum | ENROLLED / REVOKED |
| `capabilities` | JSONB | Agent capabilities |

**Capabilities Schema:**
```json
{
  "cpu": "x86_64",
  "gpu": false,
  "os": "linux",
  "sandbox_types": ["docker", "process"],
  "broker_connectors": ["binance", "alpaca"]
}
```

### 4.2 AgentEnrollmentToken

One-time enrollment token with TTL.

```sql
CREATE TABLE agent_enrollment_tokens (
    token_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    token_hash VARCHAR(64) NOT NULL,  -- SHA256 of token
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used_at TIMESTAMP WITH TIME ZONE,
    used_by_agent_id UUID REFERENCES agents(agent_id),
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Field | Type | Description |
|-------|------|-------------|
| `token_id` | UUID | Primary key |
| `token_hash` | string | SHA256 hash (not raw token) |
| `expires_at` | timestamp | Token expiration (TTL) |
| `used_at` | timestamp | When token was consumed |

---

## 5. Deployment & Run Entities

### 5.1 Deployment

Strategy deployment to agent.

```sql
CREATE TABLE deployments (
    deployment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    agent_id UUID NOT NULL REFERENCES agents(agent_id),
    build_id UUID NOT NULL REFERENCES builds(build_id),
    mode VARCHAR(10) NOT NULL,  -- PAPER, LIVE
    desired_state VARCHAR(30) NOT NULL,  -- See state machine
    current_state VARCHAR(30),  -- Agent-reported state
    config_ref VARCHAR(100),  -- Immutable config blob digest
    trading_impacting BOOLEAN NOT NULL DEFAULT TRUE,
    approval_required BOOLEAN NOT NULL DEFAULT TRUE,
    created_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Field | Type | Description |
|-------|------|-------------|
| `deployment_id` | UUID | Primary key |
| `workspace_id` | UUID | Owning workspace |
| `agent_id` | UUID | Target agent |
| `build_id` | UUID | Strategy build to deploy |
| `mode` | enum | PAPER / LIVE |
| `desired_state` | enum | Cloud's desired state |
| `current_state` | enum | Agent's reported state |
| `config_ref` | string | Immutable config digest |
| `trading_impacting` | boolean | Whether deployment is trading-impacting |
| `approval_required` | boolean | Whether local approval needed |

### 5.2 Run

Execution instance of a deployment.

```sql
CREATE TABLE runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID NOT NULL REFERENCES deployments(deployment_id),
    state VARCHAR(20) NOT NULL,  -- See state machine
    started_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE,
    halt_reason VARCHAR(50),  -- If halted
    halt_details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | UUID | Primary key |
| `deployment_id` | UUID | Parent deployment |
| `state` | enum | Current run state |
| `halt_reason` | string | Reason for halt (if HALTED) |

---

## 6. Command & Approval Entities

### 6.1 Command

Lifecycle command from Cloud to Agent.

```sql
CREATE TABLE commands (
    command_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    deployment_id UUID REFERENCES deployments(deployment_id),
    agent_id UUID NOT NULL REFERENCES agents(agent_id),
    command_type VARCHAR(50) NOT NULL,  -- See allowed commands
    payload_ref VARCHAR(100),  -- Immutable payload blob digest
    change_class VARCHAR(20) NOT NULL,  -- TRADING_IMPACTING, NON_IMPACTING
    requires_approval BOOLEAN NOT NULL,
    issued_by UUID REFERENCES users(user_id),
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    idempotency_key VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE
);
```

| Field | Type | Description |
|-------|------|-------------|
| `command_id` | UUID | Primary key |
| `deployment_id` | UUID | Target deployment (optional) |
| `agent_id` | UUID | Target agent |
| `command_type` | enum | Command type (see allowed commands) |
| `payload_ref` | string | Immutable payload digest |
| `change_class` | enum | TRADING_IMPACTING / NON_IMPACTING |
| `requires_approval` | boolean | Whether local approval needed |
| `status` | enum | PENDING / ACKED / APPLIED / REJECTED / EXPIRED |
| `idempotency_key` | string | Unique key for idempotency |

**Allowed Command Types:**
- `REQUEST_START_RUN`
- `REQUEST_STOP_RUN`
- `REQUEST_PAUSE_RUN`
- `REQUEST_RESUME_RUN`
- `REQUEST_UPGRADE_ARTIFACT`
- `REQUEST_UPDATE_CONFIG`
- `REQUEST_ROTATE_AGENT_SESSION`
- `REQUEST_EXPORT_LOGS`

### 6.2 ApprovalRecord

Local approval evidence from Agent.

```sql
CREATE TABLE approval_records (
    approval_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    command_id UUID NOT NULL REFERENCES commands(command_id),
    agent_id UUID NOT NULL REFERENCES agents(agent_id),
    decision VARCHAR(20) NOT NULL,  -- APPROVED, REJECTED
    approved_by_local_identity VARCHAR(255),  -- Local user identity
    approved_at TIMESTAMP WITH TIME ZONE NOT NULL,
    approval_evidence VARCHAR(100),  -- Evidence hash/attestation
    rejection_reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Field | Type | Description |
|-------|------|-------------|
| `approval_id` | UUID | Primary key |
| `command_id` | UUID | Related command |
| `agent_id` | UUID | Agent that approved |
| `decision` | enum | APPROVED / REJECTED |
| `approved_by_local_identity` | string | Who approved locally |
| `approval_evidence` | string | Cryptographic evidence hash |

---

## 7. Telemetry & Audit Entities

### 7.1 TelemetryEvent

Agent telemetry events (aggregated).

```sql
CREATE TABLE telemetry_events (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID NOT NULL REFERENCES agents(agent_id),
    deployment_id UUID REFERENCES deployments(deployment_id),
    run_id UUID REFERENCES runs(run_id),
    sensitivity VARCHAR(20) NOT NULL,  -- AGGREGATED, DETAILED
    event_type VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metrics JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Partition by time for efficient retention
CREATE TABLE telemetry_events_2025_12 PARTITION OF telemetry_events
    FOR VALUES FROM ('2025-12-01') TO ('2026-01-01');
```

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | UUID | Primary key |
| `sensitivity` | enum | AGGREGATED / DETAILED (RAW requires enterprise opt-in) |
| `event_type` | string | Type of telemetry event |
| `metrics` | JSONB | Metrics payload (redacted) |

**Metrics Schema (AGGREGATED):**
```json
{
  "pnl": 12.34,
  "drawdown": -3.2,
  "exposure_usd": 5000.0,
  "orders_per_min": 2,
  "broker_error_rate": 0,
  "latency_p99_ms": 45
}
```

### 7.2 Alert

System alerts.

```sql
CREATE TABLE alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    agent_id UUID REFERENCES agents(agent_id),
    deployment_id UUID REFERENCES deployments(deployment_id),
    severity VARCHAR(20) NOT NULL,  -- INFO, WARNING, ERROR, CRITICAL
    alert_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID REFERENCES users(user_id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 7.3 AccessAudit

Sensitive data access audit log.

```sql
CREATE TABLE access_audits (
    audit_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    user_id UUID NOT NULL REFERENCES users(user_id),
    resource_type VARCHAR(50) NOT NULL,  -- telemetry, deployment, agent
    resource_id UUID NOT NULL,
    action VARCHAR(50) NOT NULL,  -- view, export, break_glass
    access_reason TEXT,  -- Required for break_glass
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

| Field | Type | Description |
|-------|------|-------------|
| `audit_id` | UUID | Primary key |
| `resource_type` | string | Type of accessed resource |
| `action` | string | Access action (view/export/break_glass) |
| `access_reason` | text | Required for break-glass access |

---

## 8. Governance Entities

### 8.1 DataRetentionPolicy

Per-tenant retention configuration.

```sql
CREATE TABLE data_retention_policies (
    policy_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(workspace_id),
    telemetry_retention_days INTEGER NOT NULL DEFAULT 90,
    audit_retention_days INTEGER NOT NULL DEFAULT 365,
    artifact_retention_days INTEGER,  -- NULL = forever
    auto_purge_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(workspace_id)
);
```

| Field | Type | Description |
|-------|------|-------------|
| `telemetry_retention_days` | int | Days to retain telemetry |
| `audit_retention_days` | int | Days to retain audit logs |
| `auto_purge_enabled` | boolean | Automatic data purging |

---

## 9. Entity Relationships

```
Organization (1) ─────< Workspace (N)
     │                     │
     │                     ├────< User (N:M via membership)
     │                     │
     │                     ├────< Strategy (N)
     │                     │         │
     │                     │         └────< StrategyVersion (N)
     │                     │                     │
     │                     │                     └────< Build (N)
     │                     │                              │
     │                     │                              └────< Artifact (1)
     │                     │
     │                     ├────< Agent (N)
     │                     │         │
     │                     │         └────< AgentEnrollmentToken (N)
     │                     │
     │                     └────< Deployment (N)
     │                               │
     │                               ├────< Run (N)
     │                               │
     │                               └────< Command (N)
     │                                         │
     │                                         └────< ApprovalRecord (1)
     │
     └────< DataRetentionPolicy (1)
```

---

## 10. Indexes and Constraints

### Performance Indexes

```sql
-- Frequently queried fields
CREATE INDEX idx_agents_workspace ON agents(workspace_id);
CREATE INDEX idx_agents_status ON agents(workspace_id, status);
CREATE INDEX idx_deployments_agent ON deployments(agent_id);
CREATE INDEX idx_commands_agent_status ON commands(agent_id, status);
CREATE INDEX idx_telemetry_agent_time ON telemetry_events(agent_id, timestamp);
CREATE INDEX idx_access_audit_workspace ON access_audits(workspace_id, created_at);
```

### Data Integrity Constraints

```sql
-- Prevent deletion of active resources
ALTER TABLE agents ADD CONSTRAINT no_delete_online_agent
    CHECK (status != 'ONLINE' OR revoked_at IS NOT NULL);

-- Ensure command types are valid
ALTER TABLE commands ADD CONSTRAINT valid_command_type
    CHECK (command_type IN (
        'REQUEST_START_RUN',
        'REQUEST_STOP_RUN',
        'REQUEST_PAUSE_RUN',
        'REQUEST_RESUME_RUN',
        'REQUEST_UPGRADE_ARTIFACT',
        'REQUEST_UPDATE_CONFIG',
        'REQUEST_ROTATE_AGENT_SESSION',
        'REQUEST_EXPORT_LOGS'
    ));
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial data model per Design Doc |

---

**Related Documentation:**
- [CCEA Overview](./CCEA_OVERVIEW.md)
- [State Machine](./CCEA_STATE_MACHINE.md)
- [Protocol Schema](../schemas/README.md)
