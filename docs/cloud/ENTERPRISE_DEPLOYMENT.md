# Enterprise Deployment Guide

**CCEA Cloud - On-Prem/VPC/Air-Gapped Deployment**

**Version:** 1.0.0
**Last Updated:** 2025-12-15
**Design Doc Reference:** Phase 9 - Enterprise/on-prem pack

---

## Overview

This guide covers enterprise deployment scenarios for CCEA Cloud Control Plane:

| Mode | Description | Network Requirements |
|------|-------------|---------------------|
| **On-Premises** | Full stack in customer datacenter | Internal network only |
| **VPC** | Cloud VPC deployment (AWS/GCP/Azure) | VPC + optional egress |
| **Air-Gapped** | Fully isolated, no external network | Zero external connectivity |
| **Hybrid** | Cloud control plane + on-prem agents | Outbound from agents only |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE DEPLOYMENT                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Control Plane  │    │  Artifact       │                    │
│  │  (FastAPI)      │◄───│  Registry       │                    │
│  └────────┬────────┘    └─────────────────┘                    │
│           │                                                     │
│  ┌────────▼────────┐    ┌─────────────────┐                    │
│  │   PostgreSQL    │    │     Redis       │                    │
│  │   (Primary DB)  │    │   (Cache/Queue) │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Governance    │    │   Telemetry     │                    │
│  │   Service       │    │   Ingester      │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Builder       │    │   Monitoring    │                    │
│  │   Service       │    │   (Prometheus)  │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ Outbound-only (Agent → Cloud)
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    CUSTOMER ENVIRONMENT                         │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Agent 1       │    │   Agent N       │                    │
│  │   (Local Exec)  │    │   (Local Exec)  │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended | High Availability |
|-----------|---------|-------------|-------------------|
| **Control Plane** | 2 CPU, 4GB RAM | 4 CPU, 8GB RAM | 8 CPU, 16GB RAM |
| **PostgreSQL** | 2 CPU, 4GB RAM | 4 CPU, 16GB RAM | 8 CPU, 32GB RAM |
| **Redis** | 1 CPU, 1GB RAM | 2 CPU, 2GB RAM | 2 CPU, 4GB RAM |
| **Registry** | 2 CPU, 4GB RAM | 4 CPU, 8GB RAM | 4 CPU, 16GB RAM |
| **Storage** | 100GB SSD | 500GB SSD | 1TB NVMe |

### Software Requirements

- Docker Engine 24.0+ or containerd 1.7+
- Docker Compose 2.20+ (for docker-compose deployment)
- Kubernetes 1.28+ (for Helm deployment)
- PostgreSQL 15+ (if external)
- Redis 7+ (if external)

### Network Requirements

| Port | Service | Direction | Description |
|------|---------|-----------|-------------|
| 8000 | Control Plane API | Inbound | Agent connections |
| 8001 | Telemetry Ingester | Inbound | Telemetry upload |
| 5432 | PostgreSQL | Internal | Database |
| 6379 | Redis | Internal | Cache |
| 5000 | Registry | Internal | Artifact storage |
| 9090 | Prometheus | Internal | Metrics |
| 3000 | Grafana | Internal | Dashboards |

---

## Deployment Methods

### Method 1: Docker Compose (Recommended for Single-Node)

#### Standard Deployment

```bash
# 1. Clone repository
git clone https://github.com/your-org/ccea-cloud.git
cd ccea-cloud/deploy/docker

# 2. Create environment file
cp .env.example .env

# 3. Edit environment variables
nano .env

# 4. Generate secrets
export CCEA_SECRET_KEY=$(openssl rand -base64 32)
export POSTGRES_PASSWORD=$(openssl rand -base64 24)
export REDIS_PASSWORD=$(openssl rand -base64 24)

# 5. Start stack
docker-compose up -d

# 6. Verify health
docker-compose ps
curl http://localhost:8000/health
```

#### Air-Gapped Deployment

```bash
# 1. Pre-pull images on connected machine
docker pull postgres:16-alpine
docker pull redis:7-alpine
docker pull ccea/cloud-control-plane:latest
docker pull ccea/builder:latest
docker pull ccea/telemetry-ingester:latest
docker pull ccea/governance:latest
docker pull registry:2
docker pull prom/prometheus:v2.47.0
docker pull grafana/grafana:10.1.0

# 2. Save images to tarball
docker save -o ccea-images.tar \
  postgres:16-alpine \
  redis:7-alpine \
  ccea/cloud-control-plane:latest \
  ccea/builder:latest \
  ccea/telemetry-ingester:latest \
  ccea/governance:latest \
  registry:2 \
  prom/prometheus:v2.47.0 \
  grafana/grafana:10.1.0

# 3. Transfer to air-gapped environment
# (via secure media)

# 4. Load images on air-gapped machine
docker load -i ccea-images.tar

# 5. Deploy with air-gapped overlay
docker-compose \
  -f docker-compose.yml \
  -f docker-compose.airgapped.yml \
  up -d
```

### Method 2: Helm Chart (Kubernetes)

#### Standard Kubernetes Deployment

```bash
# 1. Add Helm repository (or use local chart)
helm repo add ccea https://charts.ccea.io
helm repo update

# 2. Create namespace
kubectl create namespace ccea-cloud

# 3. Create secrets
kubectl create secret generic ccea-secrets \
  --namespace ccea-cloud \
  --from-literal=secret-key=$(openssl rand -base64 32) \
  --from-literal=postgres-password=$(openssl rand -base64 24) \
  --from-literal=redis-password=$(openssl rand -base64 24)

# 4. Install chart
helm install ccea-cloud ./deploy/helm/ccea-cloud \
  --namespace ccea-cloud \
  --values values-enterprise.yaml

# 5. Verify deployment
kubectl get pods -n ccea-cloud
kubectl get svc -n ccea-cloud
```

#### Air-Gapped Kubernetes Deployment

```bash
# 1. Set up private registry (Harbor, Artifactory, etc.)
# 2. Push images to private registry
# 3. Update values.yaml with private registry

cat > values-airgapped.yaml <<EOF
global:
  imageRegistry: registry.internal.company.com
  airgapped: true

controlPlane:
  config:
    AIR_GAPPED_MODE: "true"
    SKIP_UPDATE_CHECK: "true"
    TELEMETRY_EXTERNAL_EXPORT: "false"

registry:
  enabled: true
  persistence:
    size: 100Gi
EOF

# 4. Install with air-gapped values
helm install ccea-cloud ./deploy/helm/ccea-cloud \
  --namespace ccea-cloud \
  --values values-airgapped.yaml
```

---

## Configuration Reference

### Environment Variables

#### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `CCEA_ENV` | `production` | Environment name |
| `CCEA_SECRET_KEY` | **required** | Application secret key |
| `CCEA_LOG_LEVEL` | `INFO` | Logging level |
| `CCEA_DATA_RESIDENCY` | `on-prem` | Data residency region |

#### Database

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | **required** | PostgreSQL connection string |
| `POSTGRES_USER` | `ccea` | Database user |
| `POSTGRES_PASSWORD` | **required** | Database password |
| `POSTGRES_DB` | `ccea_cloud` | Database name |

#### Security

| Variable | Default | Description |
|----------|---------|-------------|
| `CCEA_MTLS_ENABLED` | `false` | Enable mTLS |
| `CCEA_TLS_CERT_PATH` | `/certs/server.crt` | TLS certificate |
| `CCEA_TLS_KEY_PATH` | `/certs/server.key` | TLS private key |
| `CCEA_CA_CERT_PATH` | `/certs/ca.crt` | CA certificate |

#### Air-Gapped Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `CCEA_AIR_GAPPED_MODE` | `false` | Enable air-gapped mode |
| `CCEA_SKIP_UPDATE_CHECK` | `false` | Skip update checks |
| `CCEA_OFFLINE_VERIFICATION` | `false` | Use offline signature verification |
| `CCEA_TELEMETRY_EXTERNAL_EXPORT` | `true` | Allow external telemetry export |
| `CCEA_TELEMETRY_LOCAL_ONLY` | `false` | Keep all telemetry local |

#### Signing and Verification

| Variable | Default | Description |
|----------|---------|-------------|
| `CCEA_SIGNING_ENABLED` | `true` | Enable artifact signing |
| `CCEA_SIGNING_KEY_PATH` | `/keys/signing.key` | Signing key path |
| `CCEA_SIGNING_OFFLINE_MODE` | `false` | Offline signing mode |
| `CCEA_TRUSTED_ROOT_PATH` | `/certs/trusted-roots` | Trusted root certificates |

---

## Security Configuration

### TLS/mTLS Setup

```bash
# 1. Generate CA (for internal use)
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 \
  -out ca.crt -subj "/CN=CCEA Internal CA"

# 2. Generate server certificate
openssl genrsa -out server.key 2048
openssl req -new -key server.key -out server.csr \
  -subj "/CN=ccea-control-plane"

cat > server.ext <<EOF
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
DNS.2 = ccea-control-plane
DNS.3 = control-plane.ccea-cloud.svc
IP.1 = 127.0.0.1
EOF

openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key \
  -CAcreateserial -out server.crt -days 365 -sha256 \
  -extfile server.ext

# 3. Copy to deploy/docker/certs/
mkdir -p deploy/docker/certs
cp ca.crt server.crt server.key deploy/docker/certs/
```

### Signing Key Setup

```bash
# Generate Ed25519 signing key
mkdir -p deploy/docker/keys

# Using OpenSSL 3.x
openssl genpkey -algorithm Ed25519 -out deploy/docker/keys/signing.key

# Export public key for agents
openssl pkey -in deploy/docker/keys/signing.key -pubout \
  -out deploy/docker/keys/signing.pub

# Set permissions
chmod 600 deploy/docker/keys/signing.key
chmod 644 deploy/docker/keys/signing.pub
```

### Trusted Roots Setup (Air-Gapped)

```bash
# Create trusted roots directory
mkdir -p deploy/docker/trusted-roots

# Copy CA certificates
cp ca.crt deploy/docker/trusted-roots/

# Copy signing public keys
cp deploy/docker/keys/signing.pub deploy/docker/trusted-roots/

# Create trust manifest
cat > deploy/docker/trusted-roots/manifest.json <<EOF
{
  "version": 1,
  "created_at": "$(date -Iseconds)",
  "ca_certificates": ["ca.crt"],
  "signing_keys": ["signing.pub"],
  "policy": {
    "require_signatures": true,
    "minimum_key_bits": 256,
    "allowed_algorithms": ["ed25519", "ecdsa-p256"]
  }
}
EOF
```

---

## Database Setup

### PostgreSQL Initialization

The `init-db.sql` script creates required schemas and extensions:

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enable Row Level Security (RLS) for multi-tenancy
ALTER TABLE workspaces ENABLE ROW LEVEL SECURITY;
ALTER TABLE deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE commands ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemetry_events ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY workspace_isolation ON workspaces
  USING (id = current_setting('app.current_workspace_id')::uuid);
```

### Database Backup (Enterprise)

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/data/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/ccea_cloud_${TIMESTAMP}.sql.gz"

docker exec ccea-postgres pg_dump -U ccea ccea_cloud | gzip > ${BACKUP_FILE}

# Retention: keep 30 days
find ${BACKUP_DIR} -name "ccea_cloud_*.sql.gz" -mtime +30 -delete
```

---

## Monitoring and Observability

### Prometheus Configuration

Create `deploy/docker/monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'control-plane'
    static_configs:
      - targets: ['control-plane:8000']
    metrics_path: '/metrics'

  - job_name: 'telemetry-ingester'
    static_configs:
      - targets: ['telemetry-ingester:8001']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

### Grafana Dashboards

Pre-configured dashboards are available in `deploy/docker/monitoring/grafana/dashboards/`:

- `ccea-overview.json` - System overview
- `agent-health.json` - Agent monitoring
- `trading-metrics.json` - Trading telemetry (redacted)
- `security-audit.json` - Security events

---

## Evidence Pack Export

### Scheduled Export (Enterprise)

```bash
# Add to crontab for daily evidence export
0 2 * * * /opt/ccea/scripts/export-evidence.sh

# export-evidence.sh
#!/bin/bash
set -e

EXPORT_DIR="/data/evidence-exports"
TIMESTAMP=$(date +%Y%m%d)

# Export evidence pack
docker exec ccea-control-plane python -m ccea.cli evidence export \
  --output "${EXPORT_DIR}/evidence-${TIMESTAMP}.zip" \
  --types artifact_digests,approval_records,command_logs,incident_logs \
  --days 7 \
  --sign \
  --compress

# Verify signature
docker exec ccea-control-plane python -m ccea.cli evidence verify \
  --pack "${EXPORT_DIR}/evidence-${TIMESTAMP}.zip"
```

---

## Agent Updates (Enterprise)

### Version Pinning

```bash
# Pin agent version for enterprise
curl -X POST http://localhost:8000/api/v1/enterprise/version-pins \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "scope": "organization",
    "scope_id": "org-123",
    "constraint_type": "exact",
    "version": "1.2.3",
    "change_window": "maintenance_only"
  }'
```

### Change Windows

Configure maintenance windows for updates:

```yaml
# values-enterprise.yaml
enterprise:
  changeWindows:
    enabled: true
    windows:
      - name: "maintenance"
        days: [6]  # Saturday
        startHour: 2
        endHour: 6
        timezone: "UTC"
      - name: "emergency"
        # Always available for critical updates
        enabled: true
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Failed

```bash
# Check PostgreSQL status
docker-compose ps postgres
docker-compose logs postgres

# Test connection
docker exec -it ccea-postgres psql -U ccea -d ccea_cloud -c "SELECT 1"
```

#### 2. Registry Connection Issues

```bash
# Check registry status
curl http://localhost:5000/v2/_catalog

# Check control plane can reach registry
docker exec ccea-control-plane curl http://registry:5000/v2/
```

#### 3. Air-Gapped Verification Failures

```bash
# Verify trusted roots are mounted
docker exec ccea-control-plane ls -la /certs/trusted-roots/

# Check signing key
docker exec ccea-control-plane cat /keys/signing.pub
```

### Health Checks

```bash
# Full health check
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "registry": "healthy"
  },
  "air_gapped": false
}
```

### Log Collection

```bash
# Collect all logs
docker-compose logs > ccea-logs-$(date +%Y%m%d).txt

# Follow specific service
docker-compose logs -f control-plane

# Export for support (redacted)
docker exec ccea-control-plane python -m ccea.cli support-bundle \
  --output support-bundle.tar.gz \
  --redact
```

---

## Upgrade Procedures

### Standard Upgrade

```bash
# 1. Backup database
./scripts/backup-db.sh

# 2. Pull new images
docker-compose pull

# 3. Rolling restart
docker-compose up -d --no-deps --build control-plane
docker-compose up -d --no-deps --build builder
docker-compose up -d --no-deps --build telemetry-ingester
docker-compose up -d --no-deps --build governance

# 4. Verify health
curl http://localhost:8000/health
```

### Air-Gapped Upgrade

```bash
# 1. On connected machine: download new images
# 2. Transfer via secure media
# 3. Load new images
docker load -i ccea-images-v1.2.0.tar

# 4. Update image tags in .env
echo "CCEA_VERSION=1.2.0" >> .env

# 5. Restart with new version
docker-compose up -d
```

---

## Compliance and Audit

### Audit Log Retention

```yaml
# values-enterprise.yaml
governance:
  audit:
    retentionDays: 2555  # 7 years for financial regulations
    immutable: true
    signLogs: true
```

### Evidence Export Schedule

| Frequency | Content | Retention |
|-----------|---------|-----------|
| Daily | Command logs, agent health | 90 days |
| Weekly | Full evidence pack | 1 year |
| Monthly | Compliance snapshot | 7 years |

### Regulatory Compliance

- **SOC 2 Type II**: Audit logs, access controls, encryption
- **GDPR**: Data residency, DSAR support, retention policies
- **MiFID II**: Transaction records, best execution evidence
- **ISO 27001**: Information security controls

---

## Support

### Enterprise Support Channels

- **Email**: enterprise-support@ccea.io
- **Phone**: +1-XXX-XXX-XXXX (24/7 for P1)
- **Portal**: https://support.ccea.io

### Support Bundle

Generate a support bundle for troubleshooting:

```bash
docker exec ccea-control-plane python -m ccea.cli support-bundle \
  --output support-bundle.tar.gz \
  --redact \
  --include-logs \
  --include-config \
  --include-metrics
```

**Note**: Support bundles are automatically redacted to remove:
- Credentials and API keys
- PII (email addresses, user names)
- Broker account details
- Trading positions and orders

---

## Related Documentation

- [CCEA Overview](../CCEA_OVERVIEW.md)
- [Trust Center](../security/TRUST_CENTER.md)
- [Runbooks](../runbooks/README.md)
- [API Reference](../api/README.md)
