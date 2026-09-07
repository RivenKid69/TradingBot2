# Cloud Control Plane Deployment Guide

> **Version**: 2.0.0
> **Date**: 2025-12-15
> **Architecture**: CCEA (Cloud-Controlled Execution Architecture)

## Overview

This guide describes deployment of the **Cloud Control Plane** using zone-separated artifacts.

**Key Principle**: Cloud zone is designed to not contain trading libraries, broker clients, or execution code (verify via CI guardrails and artifact SBOM).

## Zone Separation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        BUILD PROCESS                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   make dist-cloud           make dist-agent                         │
│         │                         │                                 │
│         ▼                         ▼                                 │
│   ┌─────────────┐           ┌─────────────┐                        │
│   │ ccea_cloud  │           │ ccea_agent  │                        │
│   │   .whl      │           │   .whl      │                        │
│   └─────────────┘           └─────────────┘                        │
│         │                         │                                 │
│         ▼                         │                                 │
│   make artifact-check-cloud       │                                 │
│   (CI Guardrail: no trading code) │                                 │
│                                   │                                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Building Artifacts

### Cloud Distribution

```bash
# Build Cloud-only artifact
make dist-cloud

# Output: dist/ccea_cloud-<version>-py3-none-any.whl
```

**Cloud artifact contains:**

- `packages/cloud/` - Control plane, builder, governance, research
- `packages/shared/` - Shared contracts and models
- `ccea/artifact/` - Artifact utilities
- `ccea/crypto/` - Cryptographic primitives
- `ccea/models/` - Data models
- `ccea/contracts/` - Protocol contracts (enums, state machines)
- `ccea/protocol/` - Protocol validation
- `ccea/telemetry/` - Telemetry schemas (no agent-side code)

**Cloud artifact is designed not to contain (verify via `make artifact-check-cloud`):**

- `packages/agent/` - Agent daemon, execution, vault
- `ccea/agent/` - Legacy agent shim (deprecated)
- `ccea/control_plane/` - Legacy control plane shim (deprecated)
- `execution_providers*.py` - Order execution code
- `service_signal_runner.py` - Live trading service
- Broker client libraries (ccxt, alpaca-trade-api, etc.)

### Artifact Verification

```bash
# Verify Cloud artifact has no trading code
make artifact-check-cloud
```

This runs `ccea.guardrails.build_artifact_check` which scans for:

- Prohibited imports (broker clients, execution modules)
- Prohibited code patterns (order submission, position management)
- Prohibited modules in wheel content

**CI Integration**: This check runs in GitHub Actions on every PR.

## Deployment Options

### Option 1: Docker Compose (Development/Staging)

```bash
# Navigate to deploy directory
cd deploy/docker

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f control-plane
```

**docker-compose.yml** services:

- `control-plane` - FastAPI application
- `postgres` - PostgreSQL database
- `redis` - Cache and pub/sub (optional)

### Option 2: Kubernetes/Helm (Production)

```bash
# Navigate to helm chart
cd deploy/helm/ccea-cloud

# Install/upgrade
helm upgrade --install ccea-cloud . \
  --namespace ccea \
  --create-namespace \
  --values values.yaml \
  --values values-prod.yaml
```

**Helm Chart Components:**

- `Deployment`: Control plane pods
- `Service`: Load-balanced endpoint
- `Ingress`: TLS termination
- `ConfigMap`: Environment configuration
- `Secret`: Database credentials (from external secret manager)
- `PodSecurityPolicy`: Restricted privileges

### Option 3: Manual Deployment

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install Cloud artifact ONLY
pip install dist/ccea_cloud-<version>-py3-none-any.whl

# Set environment variables (CCEA_DATABASE_URL is required for production)
export CCEA_DATABASE_URL="postgresql+asyncpg://user:pass@host:5432/ccea"
export SECRET_KEY="<generate-secure-key>"
export CCEA_ENVIRONMENT="production"

# Run database migrations
alembic -c packages/cloud/control_plane/alembic.ini upgrade head

# Start application
uvicorn packages.cloud.control_plane.app:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `CCEA_DATABASE_URL` | Yes (prod) | PostgreSQL connection string (format: `postgresql+asyncpg://user:pass@host:5432/db`). If not set, defaults to SQLite for dev/test. |
| `SECRET_KEY` | Yes | JWT signing key (min 32 chars) |
| `CCEA_ENVIRONMENT` | No | `development`, `staging`, `production` |
| `CCEA_DB_POOL_SIZE` | No | Connection pool size (default: 10) |
| `CCEA_DB_MAX_OVERFLOW` | No | Max overflow connections (default: 20) |
| `REDIS_URL` | No | Redis connection for caching |
| `SENTRY_DSN` | No | Sentry error tracking |
| `LOG_LEVEL` | No | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CORS_ORIGINS` | No | Allowed CORS origins (comma-separated) |

**Important**: Production deployments MUST set `CCEA_DATABASE_URL` to a PostgreSQL connection. Without this variable, the application falls back to SQLite which is unsuitable for production (no RLS, no concurrent access guarantees).

## Database Migrations

```bash
# Generate new migration
alembic -c packages/cloud/control_plane/alembic.ini revision \
  --autogenerate -m "Description of changes"

# Upgrade to latest
alembic -c packages/cloud/control_plane/alembic.ini upgrade head

# Downgrade one step
alembic -c packages/cloud/control_plane/alembic.ini downgrade -1

# Show current revision
alembic -c packages/cloud/control_plane/alembic.ini current
```

## Health Checks

### Liveness Probe

```
GET /health/live
```

Returns `200 OK` if application is running.

### Readiness Probe

```
GET /health/ready
```

Returns `200 OK` if:

- Database connection is healthy
- Required services are available

### Startup Probe

```
GET /health/startup
```

Returns `200 OK` after initialization completes.

## Security Configuration

### TLS/HTTPS

**Required for production.** Use reverse proxy (nginx, Traefik) or cloud load balancer.

```nginx
server {
    listen 443 ssl http2;
    server_name ccea.example.com;

    ssl_certificate /etc/ssl/certs/ccea.crt;
    ssl_certificate_key /etc/ssl/private/ccea.key;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### RBAC

Cloud Control Plane implements role-based access control:

| Role | Permissions |
|------|-------------|
| `viewer` | Read deployments, view telemetry |
| `operator` | Start/stop runs, approve commands |
| `admin` | Manage deployments, configurations |
| `super_admin` | RBAC management, governance |

### Tenant Isolation

All data is scoped by `workspace_id`. PostgreSQL Row-Level Security (RLS) is enabled in production mode.

## Monitoring

### Prometheus Metrics

```
GET /metrics
```

Key metrics:

- `ccea_commands_total` - Commands issued by type
- `ccea_agents_connected` - Connected agents gauge
- `ccea_telemetry_events` - Telemetry events received
- `ccea_request_duration_seconds` - API latency histogram

### Grafana Dashboards

Import dashboards from `deploy/grafana/`:

- `ccea-overview.json` - System overview
- `ccea-agents.json` - Agent health and status
- `ccea-commands.json` - Command flow and latency

## Troubleshooting

### Common Issues

**Agent cannot connect:**

- Verify agent enrollment token is valid
- Check network connectivity (agent initiates outbound)
- Verify TLS certificates

**Commands stuck in PENDING:**

- Check agent is polling (heartbeat received)
- Verify agent version compatibility
- Check command approval status

**Database connection errors:**

- Verify `CCEA_DATABASE_URL` is set and correct (format: `postgresql+asyncpg://user:pass@host:5432/db`)
- Check PostgreSQL is accessible
- Run migrations: `alembic -c packages/cloud/control_plane/alembic.ini upgrade head`
- Verify migration status: `alembic -c packages/cloud/control_plane/alembic.ini current`

### Logs

```bash
# Docker Compose
docker-compose logs -f control-plane

# Kubernetes
kubectl logs -f deployment/ccea-control-plane -n ccea

# Systemd
journalctl -u ccea-control-plane -f
```

## Prohibited Configurations

**NEVER DO:**

- Install `packages/agent` on Cloud servers
- Configure broker API keys in Cloud environment
- Enable trading-related imports in Cloud build
- Disable artifact content checks in CI
- Use legacy `ccea.agent` or `ccea.control_plane` modules

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System architecture
- [LEGACY_STACKS.md](../archive/LEGACY_STACKS.md) - Deprecated modules
- [Design Doc CCEA Cloud.txt](../../archive/root_files/Design Doc CCEA Cloud.txt) - CCEA technical boundary reference
- [protocol_messages.schema.json](../schemas/protocol_messages.schema.json) - Protocol schema
