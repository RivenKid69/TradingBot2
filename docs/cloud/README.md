# Cloud Zone Documentation

> **Version**: 1.1.0 | **Last Updated**: 2025-12-16
>
> **Reference**: This document aligns with `Design Doc CCEA Cloud.txt` (canonical source)

## Overview

The Cloud Zone provides research, backtesting, monitoring, and lifecycle management capabilities. It is designed with strict security boundaries that ensure it **NEVER** has access to trading credentials or order execution capabilities.

### Design Doc Reference (§4.1)

Cloud components per Design Doc:
- **Control Plane** - Deployment/Run lifecycle, Command queue, Telemetry receiver
- **Artifact Registry** - Immutable builds, signed, with SBOM
- **Governance** - RBAC, multi-tenancy, data residency, retention policies
- **Research Environment** - Backtest runner, sandbox isolation

## Security Design Commitments

```
Cloud Zone DESIGN COMMITMENTS (enforced by architecture):
  - NEVER stores broker API keys or trading credentials
  - NEVER generates, transmits, or executes trading orders
  - NEVER has access to exchange trading endpoints
  - NEVER sends order-like payloads (side/qty/price)
  - ALWAYS redacts sensitive data in telemetry
  - ALWAYS requires signature verification for artifacts
```

## Protocol: Allowed Commands (Design Doc §10)

Cloud can ONLY send these commands to Agent:

| Command | Purpose | Requires Local Approval |
|---------|---------|------------------------|
| `REQUEST_START_RUN` | Start strategy execution | YES |
| `REQUEST_STOP_RUN` | Stop execution | NO (safety) |
| `REQUEST_PAUSE_RUN` | Pause execution | NO (safety) |
| `REQUEST_UPGRADE_ARTIFACT` | Deploy new version | YES |
| `REQUEST_UPDATE_CONFIG` | Update configuration | YES (if TRADING_IMPACTING) |
| `REQUEST_ROTATE_AGENT_SESSION` | Rotate session keys | YES |
| `REQUEST_EXPORT_LOGS` | Export logs | YES (data_sensitive) |

**Cloud NEVER sends**: `side`, `qty`, `price`, `order_type`, `target_position` fields.

---

## Components

### Control Plane (`packages/cloud/control_plane/`)

The Control Plane manages the lifecycle of deployments and agents:

| Endpoint | Purpose |
|----------|---------|
| `/api/v1/enrollment/token` | Generate enrollment tokens (TTL) |
| `/api/v1/agents/enroll` | Register new agents |
| `/api/v1/agents/{id}/heartbeat` | Agent health reporting |
| `/api/v1/agents/{id}/commands` | Long-poll for commands |
| `/api/v1/deployments/` | Manage deployments |
| `/api/v1/strategies/` | Strategy management |
| `/api/v1/telemetry/` | Telemetry ingestion |

See: [CONTROL_PLANE_API.md](./CONTROL_PLANE_API.md)

### Artifact Builder (`packages/cloud/builder/`)

The Builder creates immutable, signed artifacts for deployment:

- OCI images (preferred) or ZIP bundles
- Digest-pinned dependencies
- SBOM generation (CycloneDX/SPDX)
- Cosign/GPG signatures
- Manifest with schema versioning

See: [ARTIFACT_BUILDER.md](./ARTIFACT_BUILDER.md)

### Governance (`packages/cloud/governance/`)

Governance provides multi-tenant data management:

- RBAC (Role-Based Access Control)
- Data retention policies
- EU data residency
- Access audit logging
- Break-glass procedures

See: [GOVERNANCE.md](./GOVERNANCE.md)

### Research Job Isolation (`packages/cloud/research/`)

Research jobs (backtests, simulations) run in isolated sandboxes:

- Container/VM isolation
- CPU/RAM/time quotas
- Egress allowlist
- Abuse detection

See: [RESEARCH_JOB_ISOLATION.md](./RESEARCH_JOB_ISOLATION.md)

### Enterprise Features (`packages/cloud/enterprise/`)

Enterprise-specific capabilities:

- On-premises deployment
- Air-gapped support
- Evidence pack export
- Custom SLAs

See: [ENTERPRISE.md](./ENTERPRISE.md)

## Architecture

```
packages/cloud/
├── __init__.py              # Security guarantees documented
├── control_plane/           # API and lifecycle management
│   ├── routers/             # FastAPI routers
│   │   ├── deployments.py
│   │   ├── strategies.py
│   │   ├── agents.py
│   │   ├── commands.py
│   │   └── telemetry.py
│   ├── services/            # Business logic
│   ├── boundary.py          # Protocol boundary enforcement
│   └── models.py            # Database models
├── builder/                 # Artifact building
│   ├── manifest.py          # Manifest generation
│   ├── signing.py           # Artifact signing
│   ├── sbom.py             # SBOM generation
│   └── registry.py          # Registry publishing
├── governance/              # Tenant management
│   ├── rbac.py             # Role-based access
│   ├── retention.py        # Data retention
│   ├── residency.py        # Data residency
│   └── audit.py            # Access audit
├── research/                # Research job execution
│   ├── sandbox.py          # Job isolation
│   ├── quotas.py           # Resource quotas
│   └── abuse_detection.py  # Anti-abuse
└── enterprise/              # Enterprise features
    ├── evidence_pack.py    # Compliance export
    ├── deployment.py       # On-prem deployment
    └── air_gap.py          # Air-gapped support
```

## Data Model

### Core Entities

| Entity | Description |
|--------|-------------|
| `Organization` | Top-level tenant |
| `Workspace` | Isolated workspace within org |
| `User` | Platform user with roles |
| `Strategy` | Trading strategy definition |
| `StrategyVersion` | Versioned strategy code |
| `Build` | Artifact build record |
| `Agent` | Registered agent instance |
| `Deployment` | Strategy deployment to agent |
| `Run` | Execution instance |
| `Command` | Cloud→Agent command |
| `ApprovalRecord` | Local approval evidence |
| `TelemetryEvent` | Agent telemetry |

### Tenant Isolation

- All queries include `workspace_id`
- Postgres Row-Level Security (RLS) enabled
- Cross-tenant access blocked at database level

## CI Guardrails

The Cloud Zone enforces these build-time checks:

| Check | Description |
|-------|-------------|
| `no-trading-libs-in-cloud` | No order_execution modules in Cloud build |
| `no-broker-clients-in-cloud` | No private trading clients |
| `no-order-payloads-in-schema` | Schema prohibits side/qty/price |
| `artifact-signature-required` | All artifacts must be signed |
| `redaction-enabled` | Telemetry redaction mandatory |

## Quick Start

### Running Cloud Stack Locally

```bash
# Docker Compose
docker-compose -f docker/cloud-stack.yml up

# Or with Kubernetes
helm install ccea-cloud ./helm/cloud
```

### Configuration

```yaml
# cloud-config.yaml
cloud:
  region: eu-central-1
  database:
    host: postgres
    port: 5432
    database: ccea_cloud
  redis:
    host: redis
    port: 6379
  security:
    require_signatures: true
    min_schema_version: "1.0.0"
  governance:
    default_retention_days: 90
    eu_residency_default: true
```

## Document Index

| Document | Description |
|----------|-------------|
| [CONTROL_PLANE_API.md](./CONTROL_PLANE_API.md) | REST API reference |
| [ARTIFACT_BUILDER.md](./ARTIFACT_BUILDER.md) | Build and signing guide |
| [GOVERNANCE.md](./GOVERNANCE.md) | RBAC, retention, residency |
| [RESEARCH_JOB_ISOLATION.md](./RESEARCH_JOB_ISOLATION.md) | Sandbox isolation |
| [ENTERPRISE.md](./ENTERPRISE.md) | Enterprise deployment |

---

**Related Documentation:**
- [CCEA Overview](../CCEA_OVERVIEW.md)
- [Agent Documentation](../agent/README.md)
- [Protocol Schemas](../schemas/README.md)
