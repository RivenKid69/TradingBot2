# Enterprise Deployment Guide

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

Enterprise deployment options provide full control over infrastructure, data, and security. This guide covers on-premises, VPC, and air-gapped deployment scenarios.

---

## 1. Deployment Options

### 1.1 Comparison

| Feature | SaaS | VPC (Hosted) | On-Premises |
|---------|------|--------------|-------------|
| Infrastructure | Shared | Dedicated | Customer-owned |
| Data location | Multi-tenant | Isolated VPC | Customer DC |
| Management | Fully managed | Co-managed | Customer managed |
| Updates | Automatic | Scheduled | Customer controlled |
| Air-gap support | No | No | Yes |
| SLA | Standard | Enhanced | Custom |

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE DEPLOYMENT OPTIONS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Option 1: VPC (Hosted)          Option 2: On-Premises                      │
│  ┌─────────────────────┐         ┌─────────────────────┐                    │
│  │ Customer AWS/GCP/Az │         │   Customer Data     │                    │
│  │  ┌───────────────┐  │         │      Center         │                    │
│  │  │ CCEA Cloud    │  │         │  ┌───────────────┐  │                    │
│  │  │ (dedicated)   │  │         │  │ CCEA Cloud    │  │                    │
│  │  └───────────────┘  │         │  │ (self-hosted) │  │                    │
│  │  ┌───────────────┐  │         │  └───────────────┘  │                    │
│  │  │ Agent Cluster │  │         │  ┌───────────────┐  │                    │
│  │  │ (dedicated)   │  │         │  │ Agent Cluster │  │                    │
│  │  └───────────────┘  │         │  └───────────────┘  │                    │
│  └─────────────────────┘         └─────────────────────┘                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. VPC Deployment (Hosted)

### 2.1 Requirements

| Component | Specification |
|-----------|---------------|
| Cloud provider | AWS, GCP, Azure |
| VPC | Dedicated VPC with private subnets |
| Compute | 4+ nodes (8 vCPU, 32 GB RAM each) |
| Storage | 1 TB+ (encrypted) |
| Database | PostgreSQL 15+ (managed or self-hosted) |
| Redis | Redis 7+ (managed or self-hosted) |

### 2.2 Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CUSTOMER VPC (10.0.0.0/16)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Public Subnet (10.0.1.0/24)                       │    │
│  │  ┌───────────┐  ┌───────────┐                                       │    │
│  │  │    ALB    │  │    NAT    │                                       │    │
│  │  │ (ingress) │  │  Gateway  │                                       │    │
│  │  └───────────┘  └───────────┘                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Private Subnet (10.0.10.0/24)                      │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │    │
│  │  │   Cloud   │  │   Cloud   │  │  Research │  │  Research │        │    │
│  │  │  Control  │  │  Control  │  │    Job    │  │    Job    │        │    │
│  │  │  Plane 1  │  │  Plane 2  │  │  Worker 1 │  │  Worker 2 │        │    │
│  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                  Database Subnet (10.0.20.0/24)                      │    │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                        │    │
│  │  │ PostgreSQL│  │   Redis   │  │    S3     │                        │    │
│  │  │  (RDS)    │  │ (ElastiC) │  │ (bucket)  │                        │    │
│  │  └───────────┘  └───────────┘  └───────────┘                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.3 Terraform Deployment

```hcl
# main.tf
module "ccea_vpc" {
  source = "github.com/ccea/terraform-modules//vpc"

  vpc_cidr             = "10.0.0.0/16"
  availability_zones   = ["eu-central-1a", "eu-central-1b"]
  private_subnet_cidrs = ["10.0.10.0/24", "10.0.11.0/24"]
  public_subnet_cidrs  = ["10.0.1.0/24", "10.0.2.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false  # HA
}

module "ccea_cloud" {
  source = "github.com/ccea/terraform-modules//cloud-stack"

  vpc_id          = module.ccea_vpc.vpc_id
  private_subnets = module.ccea_vpc.private_subnet_ids

  instance_type = "c6i.2xlarge"
  min_nodes     = 2
  max_nodes     = 10

  database_instance_class = "db.r6g.large"
  redis_node_type         = "cache.r6g.large"

  encryption_key_arn = aws_kms_key.ccea.arn
}
```

---

## 3. On-Premises Deployment

### 3.1 Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Servers | 4 nodes | 8+ nodes |
| CPU | 32 cores total | 64+ cores |
| RAM | 128 GB total | 256+ GB |
| Storage | 2 TB SSD | 10+ TB SSD |
| Network | 10 Gbps | 25+ Gbps |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### 3.2 Kubernetes Deployment

```yaml
# values.yaml for Helm chart
global:
  storageClass: local-path
  imageRegistry: registry.internal.company.com

controlPlane:
  replicas: 2
  resources:
    requests:
      cpu: 2
      memory: 4Gi
    limits:
      cpu: 4
      memory: 8Gi

database:
  type: external
  host: postgres.internal.company.com
  port: 5432
  database: ccea
  sslMode: require

redis:
  type: external
  host: redis.internal.company.com
  port: 6379

storage:
  type: s3
  endpoint: minio.internal.company.com
  bucket: ccea-artifacts

security:
  encryption:
    enabled: true
    keyProvider: vault
    vaultAddress: https://vault.internal.company.com
```

### 3.3 Installation Steps

```bash
# 1. Add Helm repository
helm repo add ccea https://charts.ccea.cloud

# 2. Create namespace
kubectl create namespace ccea

# 3. Create secrets
kubectl create secret generic ccea-db-credentials \
  --namespace ccea \
  --from-literal=username=ccea \
  --from-literal=password=$DB_PASSWORD

# 4. Install CCEA Cloud stack
helm install ccea-cloud ccea/cloud-stack \
  --namespace ccea \
  --values values.yaml

# 5. Verify deployment
kubectl get pods -n ccea
kubectl get svc -n ccea
```

---

## 4. Air-Gapped Deployment

### 4.1 Overview

Air-gapped deployment has no external network connectivity:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      AIR-GAPPED ENVIRONMENT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────┐         ┌─────────────────┐                            │
│  │  CCEA Cloud     │◀────────│  Local Registry │                            │
│  │  (offline)      │         │  (mirrored)     │                            │
│  └─────────────────┘         └─────────────────┘                            │
│           │                           ▲                                      │
│           │                           │                                      │
│           ▼                           │                                      │
│  ┌─────────────────┐         ┌─────────────────┐                            │
│  │  Agent Cluster  │         │  Data Transfer  │◀─── Physical media        │
│  │  (isolated)     │         │  Station        │     (USB/DVD)             │
│  └─────────────────┘         └─────────────────┘                            │
│                                                                              │
│  NO EXTERNAL NETWORK CONNECTIVITY                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Bundle Preparation (Online System)

```bash
# On connected system: Create offline bundle
ccea-bundle create \
  --version 1.0.0 \
  --include-images \
  --include-charts \
  --include-sboms \
  --output ccea-bundle-1.0.0.tar.gz

# Verify bundle integrity
ccea-bundle verify ccea-bundle-1.0.0.tar.gz

# Generate checksums for transfer
sha256sum ccea-bundle-1.0.0.tar.gz > ccea-bundle-1.0.0.sha256
```

### 4.3 Bundle Import (Air-Gapped System)

```bash
# Transfer bundle via physical media

# Verify integrity
sha256sum -c ccea-bundle-1.0.0.sha256

# Import to local registry
ccea-bundle import \
  --bundle ccea-bundle-1.0.0.tar.gz \
  --registry registry.internal:5000

# Install from local registry
helm install ccea-cloud ccea/cloud-stack \
  --set global.imageRegistry=registry.internal:5000 \
  --values values-airgap.yaml
```

### 4.4 Data Import (Air-Gapped)

```bash
# Prepare historical data bundle
ccea-data export \
  --symbols "BTCUSDT,ETHUSDT" \
  --from "2024-01-01" \
  --to "2024-12-31" \
  --output market-data.tar.gz.enc \
  --encrypt

# Transfer and import
ccea-data import \
  --bundle market-data.tar.gz.enc \
  --decrypt
```

---

## 5. Customer-Managed Keys (CMK)

### 5.1 KMS Integration

```yaml
# cmk-config.yaml
encryption:
  mode: customer_managed
  provider: aws_kms  # or: vault, azure_keyvault, gcp_kms

  aws_kms:
    key_arn: arn:aws:kms:eu-central-1:123456789:key/xxx
    region: eu-central-1
    role_arn: arn:aws:iam::123456789:role/ccea-kms-role

  data_keys:
    database: aes256
    storage: aes256
    telemetry: aes256
    artifacts: aes256
```

### 5.2 HashiCorp Vault Integration

```yaml
# vault-config.yaml
encryption:
  mode: customer_managed
  provider: vault

  vault:
    address: https://vault.internal.company.com
    namespace: ccea
    auth_method: kubernetes
    transit_mount: ccea-transit

    keys:
      database: ccea-db-key
      storage: ccea-storage-key
```

---

## 6. Evidence Pack Export

### 6.1 Contents

| Category | Contents |
|----------|----------|
| Artifacts | Digests, signatures, SBOMs |
| Deployments | Config changes, approvals |
| Execution | Run logs, telemetry (configurable level) |
| Security | Audit logs, access logs |
| Compliance | Retention records, policy changes |

### 6.2 Export Command

```bash
# Full evidence pack
ccea-admin evidence export \
  --workspace ws_abc123 \
  --from "2024-01-01" \
  --to "2024-12-31" \
  --include artifacts deployments executions security compliance \
  --format json \
  --output evidence-2024.zip \
  --sign

# Compliance-focused export
ccea-admin evidence export \
  --workspace ws_abc123 \
  --compliance-framework dora \
  --output dora-evidence.zip
```

### 6.3 Evidence Schema

```json
{
  "export_version": "1.0.0",
  "workspace_id": "ws_abc123",
  "export_timestamp": "2025-01-15T10:00:00Z",
  "period": {
    "from": "2024-01-01T00:00:00Z",
    "to": "2024-12-31T23:59:59Z"
  },
  "contents": {
    "artifacts": {
      "count": 150,
      "path": "artifacts/"
    },
    "deployments": {
      "count": 500,
      "path": "deployments/"
    },
    "audit_logs": {
      "count": 10000,
      "path": "audit/"
    }
  },
  "signature": "...",
  "checksum": "sha256:xxx"
}
```

---

## 7. Agent Updates

### 7.1 Update Channels

| Channel | Description | Use Case |
|---------|-------------|----------|
| `stable` | Production-tested releases | Production |
| `rc` | Release candidates | Staging |
| `nightly` | Daily builds | Development |
| `pinned` | Specific version | Compliance |

### 7.2 Staged Rollout

```yaml
# update-policy.yaml
agent_updates:
  channel: stable
  auto_update: false  # Require approval

  rollout:
    strategy: canary
    stages:
      - name: canary
        percentage: 5
        duration_hours: 24
      - name: early_adopters
        percentage: 25
        duration_hours: 48
      - name: general
        percentage: 100

  rollback:
    automatic: true
    trigger:
      error_rate_threshold: 0.05
      latency_increase_percent: 50
```

### 7.3 Version Pinning

```yaml
# For compliance environments
agent_updates:
  channel: pinned
  pinned_version: "1.5.2"

  change_windows:
    - day: saturday
      time: "02:00-06:00"
      timezone: UTC

  min_version: "1.5.0"  # Security minimum
  max_version: "1.6.0"  # Compatibility maximum
```

### 7.4 Signed Updates (TUF)

```bash
# Update metadata is designed to be signed using TUF framework (verify via release pipeline)
ccea-agent update \
  --verify-signatures \
  --trust-root /etc/ccea/trust-root.json
```

---

## 8. High Availability

### 8.1 Control Plane HA

```yaml
# ha-config.yaml
control_plane:
  replicas: 3
  anti_affinity:
    type: hard
    topology_key: topology.kubernetes.io/zone

  load_balancer:
    type: internal
    health_check:
      path: /health
      interval: 10s

database:
  type: cluster
  replicas: 3
  replication: synchronous

redis:
  type: cluster
  replicas: 6
  sentinel: true
```

### 8.2 Multi-Region (DR)

```yaml
# dr-config.yaml
disaster_recovery:
  enabled: true
  primary_region: eu-central-1
  secondary_region: eu-west-1

  replication:
    database: async  # RPO: ~1 minute
    storage: sync    # RPO: 0

  failover:
    automatic: false  # Manual approval required
    rpo_target: 5m
    rto_target: 30m
```

---

## 9. Monitoring & Observability

### 9.1 Metrics (Prometheus)

```yaml
# prometheus-config.yaml
scrape_configs:
  - job_name: ccea-control-plane
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: [ccea]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        regex: ccea-control-plane
        action: keep
```

### 9.2 Logging (ELK/Loki)

```yaml
# logging-config.yaml
logging:
  format: json
  level: info

  outputs:
    - type: loki
      endpoint: http://loki.monitoring:3100
      labels:
        app: ccea
        env: production
```

### 9.3 Tracing (Jaeger/OTLP)

```yaml
# tracing-config.yaml
tracing:
  enabled: true
  provider: otlp
  endpoint: http://otel-collector:4317
  sampling_rate: 0.1
```

---

## 10. Support & SLA

### 10.1 Enterprise Support Tiers *(illustrative; actual tiers per executed agreement)*

> **Note**: Support tiers described below are illustrative design targets. Actual support levels, response times, and availability are defined in executed service agreements. CustodiaCloud does not currently operate 24/7 support infrastructure; such tiers require operational validation, staffing, and infrastructure investment before they can be offered.

| Tier | Response Time | Availability | Features |
|------|---------------|--------------|----------|
| Standard | Business hours (illustrative) | Business hours | Email support |
| Premium | Per agreement | Per agreement | Per agreement |
| Critical | Per agreement | Per agreement | Dedicated TAM (if contracted) |

### 10.2 SLA Targets *(illustrative design targets; actual SLA per executed agreement)*

> **Note**: CustodiaCloud has no production uptime history and does not make SLA commitments in documentation. Any SLA metrics are defined exclusively in executed service agreements after operational validation. The table below describes design goals only.

| Metric | Status |
|--------|--------|
| Control plane uptime | Per executed agreement (no public SLA commitment) |
| API latency (p99) | Per executed agreement (pending load testing) |
| Data durability | Per underlying cloud provider SLA |
| Recovery time | Per executed agreement (pending DR validation) |

---

**Related Documentation:**

- [CCEA Overview](../CCEA_OVERVIEW.md)
- [Governance](./GOVERNANCE.md)
- [Security Trust Center](../security/TRUST_CENTER.md)
- [DORA Compliance](../compliance/DORA_INTEGRATION_PLAN.md)
