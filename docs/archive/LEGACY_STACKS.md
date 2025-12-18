# Legacy CCEA Stacks (Deprecated)

> **Status**: DEPRECATED as of v2.0.0
> **Date**: 2025-12-15
> **Work Item**: WI-DEDRIFT-01

## Overview

This document describes the legacy CCEA module stacks that have been **deprecated** in favor of the canonical implementations in `packages/`.

## Deprecated Modules

### 1. ccea.agent (Deprecated)

**Location**: `ccea/agent/`

**Status**: DEPRECATED - Use `packages.agent` instead

**Migration Guide**:

```python
# OLD (deprecated - will emit DeprecationWarning)
from ccea.agent import AgentDaemon, AgentConfig
from ccea.agent import ApprovalManager, ApprovalRequest

# NEW (canonical)
from packages.agent.daemon.agentd import AgentDaemon
from packages.agent.daemon.config import AgentConfig
from packages.agent.approval.manager import ApprovalManager
from packages.agent.approval.models import ApprovalRequest
```

**Reason for Deprecation**:
- `packages.agent` provides complete implementation with:
  - Local vault for credential management
  - Policy firewall with hard caps
  - Full execution engine
  - Position reconciliation
  - Telemetry with mandatory redaction
- The `ccea.agent` module was a skeleton/shim layer

### 2. ccea.control_plane (Deprecated)

**Location**: `ccea/control_plane/`

**Status**: DEPRECATED - Use `packages.cloud.control_plane` instead

**Migration Guide**:

```python
# OLD (deprecated - will emit DeprecationWarning)
from ccea.control_plane import EnrollmentService, CommandService

# NEW (canonical)
from packages.cloud.control_plane.services.enrollment_service import EnrollmentService
from packages.cloud.control_plane.services.command_service import CommandService
```

**Reason for Deprecation**:
- `packages.cloud.control_plane` provides:
  - Full FastAPI application with routers
  - Database models with Alembic migrations
  - RBAC and tenant isolation
  - Complete governance endpoints
  - Comprehensive security layer

## Canonical Implementations

### Production Agent Stack

**Location**: `packages/agent/`

**Entry Point**:
```bash
python -m packages.agent.daemon.agentd --config configs/agent.yaml
```

**Components**:
| Directory | Purpose |
|-----------|---------|
| `packages/agent/daemon/` | Agent daemon, configuration |
| `packages/agent/vault/` | Local credential storage (keychain) |
| `packages/agent/policy/` | Policy firewall, hard caps |
| `packages/agent/execution/` | Live execution engine |
| `packages/agent/approval/` | Local approval workflow |
| `packages/agent/reconciliation/` | Position sync, order journal |
| `packages/agent/telemetry/` | Telemetry with redaction |
| `packages/agent/cloud/` | Cloud client (outbound-only) |

### Production Cloud Control Plane

**Location**: `packages/cloud/control_plane/`

**Entry Point**:
```bash
uvicorn packages.cloud.control_plane.app:app --host 0.0.0.0 --port 8000
```

**Components**:
| Directory | Purpose |
|-----------|---------|
| `packages/cloud/control_plane/routers/` | API endpoints |
| `packages/cloud/control_plane/services/` | Business logic |
| `packages/cloud/control_plane/models.py` | SQLAlchemy models |
| `packages/cloud/control_plane/security/` | Authentication, RBAC |
| `packages/cloud/control_plane/alembic/` | Database migrations |

## Build Artifacts

### Zone-Separated Distributions

Production deployments use zone-separated artifacts:

```bash
# Build Cloud-only artifact (NO trading libs)
make dist-cloud

# Build Agent-only artifact
make dist-agent

# Verify Cloud artifact has no trading code
make artifact-check-cloud
```

**Cloud Artifact** (`ccea_cloud-*.whl`) contains:
- `packages/cloud/`
- `packages/shared/`
- `ccea/artifact/`
- `ccea/crypto/`
- `ccea/models/`
- `ccea/contracts/`
- `ccea/protocol/`
- `ccea/telemetry/`

**Cloud Artifact NEVER contains**:
- `packages/agent/`
- `ccea/agent/`
- `ccea/control_plane/`
- Trading execution modules
- Broker client libraries

## CI Guardrails

The following guardrails prevent usage of deprecated modules in production:

1. **Import Boundary Check**: Detects imports of deprecated modules
2. **Build Artifact Scan**: Verifies Cloud artifact has no agent code
3. **DeprecationWarning**: Runtime warnings on import

## Removal Timeline

| Milestone | Date | Action |
|-----------|------|--------|
| v2.0.0 | 2025-12-15 | Deprecation warnings added |
| v2.1.0 | Q1 2026 | Remove from default imports |
| v3.0.0 | Q2 2026 | Complete removal |

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Current architecture documentation
- [Design Doc CCEA Cloud.txt](../../archive/root_files/Design Doc CCEA Cloud.txt) - Original design document
- [packages/agent/__init__.py](../../packages/agent/__init__.py) - Canonical Agent package
- [packages/cloud/control_plane/](../../packages/cloud/control_plane/) - Canonical Control Plane
