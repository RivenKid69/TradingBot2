# On-Prem/VPC Deployment Checklist

**Document Version**: 1.0.0
**Effective Date**: 2025-12-17
**Classification**: INTERNAL / OPERATIONS
**Status**: ACTIVE

## 1. Overview

This checklist is designed to help enterprise on-premises and VPC deployments align with GDPR compliance requirements, maintain the EU-only data residency posture, and produce auditable evidence packs.

### 1.1 Design Doc Reference

```
Phase 9 — Enterprise/on-prem/VPC posture (Design Doc 16.3)

On-prem/VPC deployment checklist including:
- EU-only data systems
- Registry mirror
- Offline verification/signing
- Evidence export paths
- "Telemetry stays local" defaults

Reference: docs/design/CCEA_CLOUD/Design_Doc_CCEA_Cloud.txt#L968-L972
```

---

## 2. Pre-Deployment Checklist

### 2.1 Infrastructure Requirements

#### 2.1.1 Compute Resources

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| Control Plane | 2 vCPU, 2GB RAM | 4 vCPU, 4GB RAM | 2+ replicas for HA |
| Database (PostgreSQL) | 2 vCPU, 4GB RAM | 4 vCPU, 8GB RAM | Persistent storage |
| Cache (Redis) | 1 vCPU, 512MB RAM | 2 vCPU, 1GB RAM | Optional cluster mode |
| Telemetry Ingester | 2 vCPU, 2GB RAM | 4 vCPU, 4GB RAM | Scales with load |
| Governance Service | 1 vCPU, 512MB RAM | 2 vCPU, 1GB RAM | Evidence export |
| Builder Service | 2 vCPU, 2GB RAM | 4 vCPU, 4GB RAM | Signing/SBOM |
| Registry | 1 vCPU, 512MB RAM | 2 vCPU, 1GB RAM | Artifact storage |

#### 2.1.2 Storage Requirements

| Storage Type | Minimum | Recommended | Purpose |
|--------------|---------|-------------|---------|
| Database | 100GB SSD | 500GB SSD | PostgreSQL data |
| Redis | 10GB | 50GB | Cache persistence |
| Registry | 200GB | 1TB | Artifact storage |
| Evidence Export | 100GB | 500GB | Evidence pack storage |
| Logs | 50GB | 200GB | Application logs |
| Backups | 500GB | 2TB | Database backups |

#### 2.1.3 Network Requirements

| Requirement | Description | Mandatory |
|-------------|-------------|-----------|
| Internal Network | Isolated network for inter-service communication | Yes |
| Agent Access | Network path for Agent connections (WebSocket) | Yes |
| DNS | Internal DNS for service discovery | Yes |
| Load Balancer | TLS termination, health checks | Recommended |
| Firewall | Default-deny, explicit allowlist | Yes |

**Pre-Deployment Checklist:**

- [ ] Compute resources provisioned in EU region
- [ ] Storage provisioned with encryption at rest
- [ ] Network isolation configured
- [ ] Load balancer configured with TLS
- [ ] Firewall rules configured (default-deny)
- [ ] DNS entries created for services
- [ ] NTP configured for accurate timestamps

---

### 2.2 EU-Only Data Residency Verification

#### 2.2.1 Infrastructure Location Verification

| Component | EU Region Required | Verified |
|-----------|-------------------|----------|
| Compute instances | Yes | [ ] |
| PostgreSQL database | Yes | [ ] |
| Redis cache | Yes | [ ] |
| Object storage | Yes | [ ] |
| Backup storage | Yes | [ ] |
| Load balancer | Yes | [ ] |
| DNS resolver | Yes | [ ] |

#### 2.2.2 Subprocessor Verification

Verify all subprocessors are suitable for EU data processing (EU residency + GDPR-aligned DPA/SCCs where needed):

| Subprocessor | Purpose | EU Region | DPA Status | Verified |
|--------------|---------|-----------|------------|----------|
| Cloud provider | Infrastructure | [ ] | [ ] | [ ] |
| Monitoring (if external) | Observability | [ ] | [ ] | [ ] |
| Log aggregation (if external) | Logging | [ ] | [ ] | [ ] |
| Email service (if external) | Notifications | [ ] | [ ] | [ ] |

**EU Residency Checklist:**

- [ ] All compute resources in EU regions
- [ ] All storage in EU regions
- [ ] All backups in EU regions
- [ ] All subprocessors verified suitable for EU data processing
- [ ] DPAs signed with all subprocessors
- [ ] No cross-border data transfers configured
- [ ] Residency drift check script tested

---

### 2.3 Security Prerequisites

#### 2.3.1 TLS Certificates

| Certificate | Purpose | Required |
|-------------|---------|----------|
| Control Plane TLS | API encryption | Yes |
| Registry TLS | Artifact registry | Yes |
| Database TLS | DB connection encryption | Yes |
| mTLS CA | Client authentication | Recommended |
| Signing Certificate | Artifact signing | Yes |
| Evidence Signing | Evidence pack signing | Yes |

#### 2.3.2 Secrets Management

| Secret | Description | Storage |
|--------|-------------|---------|
| `CCEA_SECRET_KEY` | Application secret | HSM/Vault |
| `POSTGRES_PASSWORD` | Database password | HSM/Vault |
| `REDIS_PASSWORD` | Cache password | HSM/Vault |
| Signing private key | Artifact signing | HSM/Vault |
| TLS private keys | Certificate keys | HSM/Vault |

**Security Prerequisites Checklist:**

- [ ] TLS certificates generated/obtained
- [ ] mTLS CA configured (if using mTLS)
- [ ] Signing keys generated and secured
- [ ] Secrets stored in secure vault/HSM
- [ ] Key rotation schedule defined
- [ ] Certificate expiry monitoring configured
- [ ] Backup encryption keys secured

---

### 2.4 Registry Mirror Setup (Air-Gapped)

For air-gapped deployments, pre-populate the registry mirror:

#### 2.4.1 Required Images

```bash
# Core images
ccea/cloud-control-plane:${VERSION}
ccea/telemetry-ingester:${VERSION}
ccea/governance:${VERSION}
ccea/builder:${VERSION}

# Infrastructure images
postgres:16-alpine
redis:7-alpine
registry:2

# Monitoring (optional)
prom/prometheus:v2.47.0
grafana/grafana:10.1.0
```

#### 2.4.2 Image Verification

```bash
# Verify image signatures before mirroring
ccea-cli verify image ccea/cloud-control-plane:${VERSION} \
  --public-key /keys/ccea-signing.pub

# Mirror to local registry
skopeo copy \
  --src-tls-verify=true \
  --dest-tls-verify=false \
  docker://registry.ccea.io/ccea/cloud-control-plane:${VERSION} \
  docker://localhost:5000/ccea/cloud-control-plane:${VERSION}
```

**Registry Mirror Checklist:**

- [ ] Local registry deployed
- [ ] All required images identified
- [ ] Image signatures verified
- [ ] Images mirrored to local registry
- [ ] Digest pinning configured
- [ ] Registry TLS configured
- [ ] Registry access controls configured

---

## 3. Deployment Configuration

### 3.1 Helm Deployment (Kubernetes)

#### 3.1.1 Values Configuration

Create a custom values file:

```yaml
# values-enterprise-custom.yaml

global:
  enterpriseMode: true
  airgapped: true  # Set to true for air-gapped
  dataResidency: "on-prem"
  imageRegistry: "registry.internal.company.com"  # Local registry

controlPlane:
  replicaCount: 2
  config:
    CCEA_TELEMETRY_REDACTION_MANDATORY: "true"
    CCEA_TELEMETRY_LOCAL_ONLY: "true"  # Telemetry stays local
    CCEA_AIR_GAPPED_MODE: "true"  # For air-gapped
    CCEA_OFFLINE_VERIFICATION: "true"

postgresql:
  enabled: false  # Use external DB
  external:
    host: "postgres.internal.company.com"
    port: 5432
    database: "ccea_cloud"
    existingSecret: "ccea-db-credentials"
    sslMode: "require"

telemetryIngester:
  config:
    CCEA_TELEMETRY_REDACTION_MANDATORY: "true"
    CCEA_TELEMETRY_LOCAL_ONLY: "true"
    CCEA_TELEMETRY_DEFAULT_LEVEL: "aggregated"

governance:
  config:
    CCEA_EVIDENCE_EXPORT_LOCAL_ONLY: "true"
    CCEA_DSAR_ENABLED: "true"

networkPolicy:
  enabled: true
  defaultDeny: true
  egress:
    allowExternal: false  # No external egress
```

#### 3.1.2 Deployment Commands

```bash
# Add Helm repo (for non-air-gapped)
helm repo add ccea https://charts.ccea.io

# Deploy with enterprise values
helm install ccea-cloud ccea/ccea-cloud \
  -f values-enterprise.yaml \
  -f values-enterprise-custom.yaml \
  --namespace ccea \
  --create-namespace

# Verify deployment
kubectl -n ccea get pods
kubectl -n ccea get svc
```

**Helm Deployment Checklist:**

- [ ] Custom values file created
- [ ] Enterprise mode enabled
- [ ] Air-gapped mode configured (if applicable)
- [ ] Telemetry local mode enabled
- [ ] External database configured
- [ ] Network policies enabled
- [ ] Deployment completed successfully
- [ ] All pods running and healthy
- [ ] Services accessible

---

### 3.2 Docker Compose Deployment

#### 3.2.1 Environment Configuration

Create `.env` file:

```bash
# .env

# Required
CCEA_SECRET_KEY=<generate-secure-key>
POSTGRES_PASSWORD=<secure-password>
REDIS_PASSWORD=<secure-password>

# Deployment mode
CCEA_ENV=production
CCEA_DATA_RESIDENCY=on-prem

# Telemetry (local mode)
CCEA_TELEMETRY_ENABLED=true
CCEA_TELEMETRY_REDACTION_MANDATORY=true
CCEA_TELEMETRY_LOCAL_ONLY=true
CCEA_TELEMETRY_DEFAULT_LEVEL=aggregated

# Air-gapped (if applicable)
CCEA_AIR_GAPPED_MODE=true
CCEA_OFFLINE_VERIFICATION=true
CCEA_SKIP_UPDATE_CHECK=true

# Evidence export
CCEA_EVIDENCE_EXPORT_LOCAL_ONLY=true

# Image versions (use digest for production)
CCEA_VERSION=1.0.0@sha256:abc123...
```

#### 3.2.2 Deployment Commands

```bash
# Standard deployment
docker-compose up -d

# With registry (on-prem)
docker-compose --profile registry up -d

# Full stack with monitoring
docker-compose --profile full up -d

# Air-gapped deployment
docker-compose -f docker-compose.yml -f docker-compose.airgapped.yml up -d
```

**Docker Compose Deployment Checklist:**

- [ ] .env file created with all required variables
- [ ] Secrets securely generated
- [ ] Telemetry local mode configured
- [ ] Air-gapped mode configured (if applicable)
- [ ] Volumes created with proper permissions
- [ ] Deployment completed successfully
- [ ] All containers running and healthy
- [ ] Health checks passing

---

### 3.3 Telemetry Local Mode Configuration

#### 3.3.1 Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `CCEA_TELEMETRY_LOCAL_ONLY` | Disable Cloud telemetry export | `false` |
| `CCEA_TELEMETRY_EXTERNAL_EXPORT` | Allow external export | `true` |
| `CCEA_TELEMETRY_STORAGE_PATH` | Local storage path | `/data/telemetry` |
| `CCEA_TELEMETRY_RETENTION_DAYS` | Local retention period | `90` |
| `CCEA_TELEMETRY_ENCRYPTION` | Encrypt local telemetry | `true` |

#### 3.3.2 Verification

```bash
# Verify telemetry local mode
curl -s http://localhost:8001/config | jq '.telemetry'

# Expected output
{
  "local_only": true,
  "external_export": false,
  "redaction_mandatory": true,
  "default_level": "aggregated"
}
```

**Telemetry Local Mode Checklist:**

- [ ] `CCEA_TELEMETRY_LOCAL_ONLY=true` set
- [ ] `CCEA_TELEMETRY_EXTERNAL_EXPORT=false` set
- [ ] Local storage path configured
- [ ] Local storage encrypted
- [ ] Retention period configured
- [ ] No external telemetry endpoints configured
- [ ] Configuration verified via API

---

## 4. Post-Deployment Verification

### 4.1 Health Checks

```bash
# Control plane health
curl -f http://localhost:8000/health

# Telemetry ingester health
curl -f http://localhost:8001/health

# Database connectivity
psql -h postgres -U ccea -d ccea_cloud -c "SELECT 1"

# Redis connectivity
redis-cli -h redis ping
```

### 4.2 EU Residency Drift Check

```bash
# Run drift check
ccea-cli residency check --config /path/to/config.yaml

# Expected output
{
  "status": "PASS",
  "checks": [
    {"component": "database", "region": "eu-central-1", "eu_compliant": true},
    {"component": "storage", "region": "eu-west-1", "eu_compliant": true},
    {"component": "cache", "region": "eu-central-1", "eu_compliant": true}
  ],
  "violations": []
}
```

### 4.3 Evidence Pack Generation

```bash
# Generate evidence pack
ccea-cli evidence export \
  --workspace ws-default \
  --format zip \
  --categories all \
  --output /evidence/pack-$(date +%Y%m%d).zip

# Verify pack
ccea-cli evidence verify \
  --pack /evidence/pack-$(date +%Y%m%d).zip \
  --public-key /keys/evidence-signing.pub
```

**Post-Deployment Verification Checklist:**

- [ ] All health checks passing
- [ ] Control plane accessible
- [ ] Database connected and healthy
- [ ] Redis connected and healthy
- [ ] Telemetry ingester accepting events
- [ ] EU residency drift check passes
- [ ] Evidence pack generation works
- [ ] Evidence pack verification works

---

## 5. Operational Checklist

### 5.1 Monitoring Setup

| Metric | Alert Threshold | Action |
|--------|-----------------|--------|
| CPU Usage | > 80% | Scale horizontally |
| Memory Usage | > 85% | Investigate leaks |
| Disk Usage | > 75% | Expand storage |
| Error Rate | > 1% | Investigate logs |
| Response Time | > 500ms p95 | Optimize |
| Queue Depth | > 1000 | Scale consumers |

**Monitoring Checklist:**

- [ ] Prometheus scraping all services
- [ ] Grafana dashboards configured
- [ ] Alert rules configured
- [ ] Alert routing configured
- [ ] On-call schedule defined
- [ ] Runbooks documented

---

### 5.2 Backup Configuration

```bash
# PostgreSQL backup
pg_dump -h postgres -U ccea ccea_cloud | gzip > backup-$(date +%Y%m%d).sql.gz

# Verify backup
gunzip -c backup-$(date +%Y%m%d).sql.gz | head -100

# Encrypt backup
gpg --encrypt --recipient backup@company.com backup-$(date +%Y%m%d).sql.gz
```

**Backup Checklist:**

- [ ] Database backup schedule configured (daily minimum)
- [ ] Backup encryption enabled
- [ ] Backups stored in EU region
- [ ] Backup retention policy defined (minimum 7 years for compliance data)
- [ ] Backup restoration tested
- [ ] Backup monitoring configured

---

### 5.3 Update Procedure

For air-gapped deployments:

```bash
# 1. Download update package (on connected system)
ccea-cli update download --version 1.1.0 --output update-1.1.0.tar.gz

# 2. Verify package signature
ccea-cli update verify --package update-1.1.0.tar.gz \
  --public-key /keys/ccea-signing.pub

# 3. Transfer to air-gapped environment
# (use secure transfer method)

# 4. Import to local registry
ccea-cli update import --package update-1.1.0.tar.gz \
  --registry registry.internal.company.com

# 5. Apply update
helm upgrade ccea-cloud ccea/ccea-cloud \
  --set global.imageTag=1.1.0 \
  --reuse-values
```

**Update Checklist:**

- [ ] Update package downloaded
- [ ] Package signature verified
- [ ] Package transferred securely
- [ ] Images imported to local registry
- [ ] Backup created before update
- [ ] Update applied during change window
- [ ] Health checks verified post-update
- [ ] Rollback plan documented

---

## 6. Compliance Verification

### 6.1 Pre-Production Compliance Check

Run the compliance verification script:

```bash
ccea-cli compliance verify --mode on-prem \
  --config /path/to/config.yaml \
  --output compliance-report.json
```

Expected checks:
- [ ] EU residency verified
- [ ] Telemetry local mode enabled
- [ ] Redaction mandatory enabled
- [ ] Evidence pack export works
- [ ] DSAR endpoints accessible
- [ ] Break-glass workflow functional
- [ ] Audit logging enabled
- [ ] Retention policies configured

### 6.2 Evidence Pack Attestation

Generate attestation for deployment:

```bash
ccea-cli evidence attest \
  --workspace ws-default \
  --mode on-prem \
  --output attestation.json

# Contents include:
# - Deployment mode
# - EU residency proof
# - Telemetry configuration
# - Security controls
# - Timestamp and signature
```

---

## 7. Troubleshooting

### 7.1 Common Issues

| Issue | Cause | Resolution |
|-------|-------|------------|
| Drift check fails | Non-EU endpoint | Verify all endpoints in EU |
| Evidence export fails | Missing permissions | Check service account permissions |
| Telemetry rejected | Redaction not applied | Ensure redaction middleware enabled |
| Health check fails | Service not ready | Check logs, increase startup time |
| Database connection fails | SSL/TLS mismatch | Verify certificate chain |

### 7.2 Log Locations

| Component | Log Location |
|-----------|--------------|
| Control Plane | `/var/log/ccea/control-plane.log` |
| Telemetry | `/var/log/ccea/telemetry.log` |
| Governance | `/var/log/ccea/governance.log` |
| Builder | `/var/log/ccea/builder.log` |

---

## 8. Sign-Off

### 8.1 Deployment Sign-Off

| Check | Verified By | Date |
|-------|-------------|------|
| Infrastructure in EU | _____________ | ________ |
| Security controls | _____________ | ________ |
| Telemetry local mode | _____________ | ________ |
| Evidence export | _____________ | ________ |
| Backup/DR | _____________ | ________ |
| Monitoring | _____________ | ________ |

### 8.2 Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Infrastructure Lead | _________________ | _________ | ________ |
| Security Lead | _________________ | _________ | ________ |
| Compliance Lead | _________________ | _________ | ________ |
| Operations Lead | _________________ | _________ | ________ |

---

## 9. Change History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-17 | CCEA Team | Initial checklist |
