# Enterprise Technical Implementation Guide

## Overview

This document provides detailed technical specifications for enterprise deployments of the AI-Powered Quantitative Trading Platform. It covers infrastructure requirements, security configurations, integration patterns, and operational procedures.

> **Architecture**: This platform follows the **CCEA (Cloud-Controlled Execution Architecture)** — Cloud handles research and monitoring, Agent handles execution. For enterprise clients, both components can be deployed on-prem or in customer VPC.

---

## CCEA Architecture for Enterprise

### Deployment Modes

| Mode | Cloud | Agent | Use Case |
|------|-------|-------|----------|
| **SaaS + Local Agent** | Our cloud | Customer VPS/machine | Standard enterprise |
| **VPC Deployment** | Customer VPC | Customer VPC | Regulated entities |
| **Full On-Prem** | Customer datacenter | Customer datacenter | Air-gapped environments |

### Security Boundaries (CCEA)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE DEPLOYMENT (CCEA)                         │
│                                                                          │
│  ┌────────────────── CLOUD ZONE ──────────────────┐                     │
│  │  • Research IDE, Backtesting, Simulation       │                     │
│  │  • Artifact Builder (sign + publish)           │                     │
│  │  • Control Plane (lifecycle commands ONLY)     │                     │
│  │  • Telemetry Ingestion (redacted data only)    │                     │
│  │                                                │                     │
│  │  SECRETS: NEVER                                │                     │
│  │  ORDERS: NEVER                                 │                     │
│  └────────────────────────────────────────────────┘                     │
│                              │                                           │
│                              │ REQUEST_START, REQUEST_STOP, etc.         │
│                              │ (Lifecycle commands, NOT order payloads)  │
│                              ▼                                           │
│  ┌────────────────── AGENT ZONE ──────────────────┐                     │
│  │  • Local Vault (HSM/KMS integration)           │                     │
│  │  • Strategy Runner (sandbox)                   │                     │
│  │  • Risk Manager + Kill Switch (enforce)        │                     │
│  │  • Broker Connectors (order creation/sending)  │                     │
│  │  • Local Approval UI (TRADING_IMPACTING)       │                     │
│  │                                                │                     │
│  │  SECRETS: LOCAL ONLY (HSM/KMS)                 │                     │
│  │  ORDERS: CREATED AND SENT HERE                 │                     │
│  └────────────────────────────────────────────────┘                     │
│                              │                                           │
│                              ▼                                           │
│                     [ BROKER / EXCHANGE ]                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Legal Posture for Enterprise Clients

| We Provide | We Do NOT Provide |
|------------|-------------------|
| Software / ICT infrastructure | Investment advice |
| Research and simulation tools | Order execution service |
| Monitoring and lifecycle management | Custody of assets or credentials |
| Strategy development platform | Trading recommendations |

---

## Table of Contents

1. [Infrastructure Requirements](#infrastructure-requirements)
2. [Security Implementation](#security-implementation)
3. [Deployment Configurations](#deployment-configurations)
4. [Integration Patterns](#integration-patterns)
5. [Monitoring & Observability](#monitoring--observability)
6. [Disaster Recovery](#disaster-recovery)
7. [Performance Tuning](#performance-tuning)

---

## Infrastructure Requirements

### Hardware Specifications

#### Minimum Requirements (Development/Testing)

| Component | Specification |
|-----------|---------------|
| CPU | 8 cores (Intel Xeon or AMD EPYC) |
| RAM | 32 GB ECC |
| Storage | 500 GB NVMe SSD |
| Network | 1 Gbps |

#### Recommended Requirements (Production)

| Component | Specification |
|-----------|---------------|
| CPU | 32+ cores (Intel Xeon Gold or AMD EPYC) |
| RAM | 128 GB ECC |
| Storage | 2 TB NVMe SSD (RAID 10) |
| Network | 10 Gbps with redundancy |
| GPU (Optional) | NVIDIA A100/H100 for ML inference |

#### High-Frequency Trading Profile

| Component | Specification |
|-----------|---------------|
| CPU | 64+ cores, high single-thread performance |
| RAM | 256 GB ECC, low latency |
| Storage | 4 TB NVMe (Intel Optane recommended) |
| Network | 25+ Gbps, kernel bypass (DPDK/RDMA) |
| Co-location | Exchange proximity recommended |

### Software Requirements

```yaml
# Base Operating System
os:
  - name: "Ubuntu Server"
    versions: ["22.04 LTS", "24.04 LTS"]
  - name: "Red Hat Enterprise Linux"
    versions: ["8.x", "9.x"]
  - name: "Amazon Linux"
    versions: ["2023"]

# Runtime Environment
runtime:
  python: "3.12+"
  node: "20 LTS"  # For web dashboard

# Container Runtime (Optional)
containers:
  docker: "24.0+"
  kubernetes: "1.28+"

# Database
database:
  primary: "PostgreSQL 15+"
  cache: "Redis 7+"
  timeseries: "TimescaleDB 2.x"  # Optional

# Message Queue
messaging:
  - "Redis Streams"
  - "RabbitMQ 3.12+"  # Alternative
  - "Apache Kafka 3.x"  # High-throughput option
```

### Network Requirements

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        NETWORK ARCHITECTURE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INBOUND CONNECTIONS (from client systems)                                  │
│  ├── HTTPS (443): REST API, Web Dashboard                                   │
│  ├── WSS (443): WebSocket streaming                                         │
│  └── FIX (custom): FIX protocol (if enabled)                                │
│                                                                              │
│  OUTBOUND CONNECTIONS (to exchanges)                                        │
│  ├── HTTPS (443): REST APIs                                                 │
│  │   ├── api.binance.com                                                    │
│  │   ├── api.alpaca.markets                                                 │
│  │   ├── api-fxtrade.oanda.com                                              │
│  │   └── [other exchange endpoints]                                         │
│  │                                                                          │
│  ├── WSS (443): WebSocket streams                                           │
│  │   ├── stream.binance.com                                                 │
│  │   ├── stream.data.alpaca.markets                                         │
│  │   └── [other streaming endpoints]                                        │
│  │                                                                          │
│  └── TWS (7496/7497): Interactive Brokers                                   │
│                                                                              │
│  INTERNAL CONNECTIONS                                                        │
│  ├── PostgreSQL (5432): Database                                            │
│  ├── Redis (6379): Cache/Message Queue                                      │
│  ├── Prometheus (9090): Metrics                                             │
│  └── Grafana (3000): Dashboards                                             │
│                                                                              │
│  FIREWALL RULES                                                              │
│  ├── Deny all by default                                                    │
│  ├── Allow specific exchange IPs (whitelist)                                │
│  ├── Allow internal subnet only for databases                               │
│  └── Rate limiting on all external endpoints                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Implementation

### Secret Management

#### HashiCorp Vault Integration

```python
# services/vault_integration.py
from hvac import Client as VaultClient
from typing import Dict, Any
import os

class SecretManager:
    """Enterprise secret management via HashiCorp Vault."""

    def __init__(self, vault_addr: str, auth_method: str = "kubernetes"):
        self.client = VaultClient(url=vault_addr)
        self._authenticate(auth_method)

    def _authenticate(self, method: str) -> None:
        if method == "kubernetes":
            # Kubernetes service account authentication
            jwt_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
            with open(jwt_path, 'r') as f:
                jwt = f.read()
            self.client.auth.kubernetes.login(
                role="trading-platform",
                jwt=jwt
            )
        elif method == "approle":
            # AppRole authentication for VMs
            self.client.auth.approle.login(
                role_id=os.environ["VAULT_ROLE_ID"],
                secret_id=os.environ["VAULT_SECRET_ID"]
            )

    def get_exchange_credentials(self, exchange: str) -> Dict[str, str]:
        """Retrieve exchange API credentials from Vault."""
        secret = self.client.secrets.kv.v2.read_secret_version(
            path=f"trading/exchanges/{exchange}"
        )
        return secret["data"]["data"]

    def rotate_credentials(self, exchange: str) -> Dict[str, str]:
        """Trigger credential rotation."""
        # Implementation depends on exchange API
        pass
```

#### Environment-Based Configuration (Development)

```yaml
# .env.example (NEVER commit actual values)
# Exchange Credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_secret_here
ALPACA_API_KEY=your_api_key_here
ALPACA_API_SECRET=your_secret_here
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=generate_strong_random_key
ENCRYPTION_KEY=32_byte_key_for_aes256

# Monitoring
PROMETHEUS_PUSHGATEWAY=http://localhost:9091
SENTRY_DSN=https://key@sentry.io/project
```

### Encryption Implementation

#### Data at Rest

```python
# services/encryption.py
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64

class DataEncryption:
    """AES-256-GCM encryption for sensitive data."""

    def __init__(self, master_key: bytes):
        self.aesgcm = AESGCM(master_key)

    @classmethod
    def derive_key(cls, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=600_000,  # OWASP recommendation
        )
        return kdf.derive(password.encode())

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt data with AES-256-GCM."""
        nonce = os.urandom(12)  # 96-bit nonce
        ciphertext = self.aesgcm.encrypt(nonce, plaintext, None)
        return nonce + ciphertext

    def decrypt(self, encrypted: bytes) -> bytes:
        """Decrypt data."""
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        return self.aesgcm.decrypt(nonce, ciphertext, None)


class DatabaseEncryption:
    """Column-level encryption for database fields."""

    def __init__(self, key: bytes):
        self.fernet = Fernet(base64.urlsafe_b64encode(key))

    def encrypt_field(self, value: str) -> str:
        """Encrypt a database field value."""
        return self.fernet.encrypt(value.encode()).decode()

    def decrypt_field(self, encrypted: str) -> str:
        """Decrypt a database field value."""
        return self.fernet.decrypt(encrypted.encode()).decode()
```

#### Data in Transit

```yaml
# TLS Configuration (nginx example)
server {
    listen 443 ssl http2;
    server_name trading.example.com;

    # TLS 1.3 only for maximum security
    ssl_protocols TLSv1.3;
    ssl_ciphers TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256;
    ssl_prefer_server_ciphers off;

    # Certificate configuration
    ssl_certificate /etc/ssl/certs/trading.crt;
    ssl_certificate_key /etc/ssl/private/trading.key;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000" always;

    # Additional security headers
    add_header X-Frame-Options "DENY" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
}
```

### Audit Trail Implementation

```python
# services/audit_logger.py
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class AuditEvent:
    """Immutable audit event record."""
    event_id: str
    timestamp: str
    event_type: str
    actor: str
    action: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    previous_hash: str
    event_hash: str

class AuditLogger:
    """MiFID II-aligned audit logging."""

    def __init__(self, storage_backend: str = "postgresql"):
        self._previous_hash = self._get_last_hash()
        self._storage = self._init_storage(storage_backend)

    def log_event(
        self,
        event_type: str,
        actor: str,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Dict[str, Any],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event with chain integrity.

        MiFID II Requirements:
        - Timestamp precision: microseconds
        - 5-year retention minimum
        - Tamper-evident chain
        - Reconstruction capability
        """
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat(timespec='microseconds')

        # Create event without hash first
        event_data = {
            "event_id": event_id,
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": actor,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "previous_hash": self._previous_hash,
        }

        # Calculate chain hash
        event_hash = self._calculate_hash(event_data)
        event_data["event_hash"] = event_hash

        # Create immutable event
        event = AuditEvent(**event_data)

        # Store event
        self._storage.store(event)

        # Update chain
        self._previous_hash = event_hash

        return event

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for chain integrity."""
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_chain_integrity(self, start_id: str, end_id: str) -> bool:
        """Verify audit chain has not been tampered with."""
        events = self._storage.get_range(start_id, end_id)

        for i, event in enumerate(events):
            # Verify hash calculation
            event_data = asdict(event)
            stored_hash = event_data.pop("event_hash")
            calculated_hash = self._calculate_hash(event_data)

            if stored_hash != calculated_hash:
                return False

            # Verify chain linkage
            if i > 0 and event.previous_hash != events[i-1].event_hash:
                return False

        return True

    # Audit event types for MiFID II
    EVENT_TYPES = {
        "ORDER_SUBMITTED": "Order submission",
        "ORDER_MODIFIED": "Order modification",
        "ORDER_CANCELLED": "Order cancellation",
        "ORDER_FILLED": "Order execution",
        "ORDER_REJECTED": "Order rejection",
        "POSITION_OPENED": "Position opened",
        "POSITION_CLOSED": "Position closed",
        "RISK_LIMIT_BREACH": "Risk limit breach",
        "KILL_SWITCH_ACTIVATED": "Emergency stop",
        "ALGORITHM_STARTED": "Algorithm started",
        "ALGORITHM_STOPPED": "Algorithm stopped",
        "CONFIG_CHANGED": "Configuration change",
        "USER_LOGIN": "User authentication",
        "USER_LOGOUT": "User logout",
        "API_KEY_CREATED": "API key creation",
        "API_KEY_REVOKED": "API key revocation",
    }
```

### Role-Based Access Control (RBAC)

```python
# services/rbac.py
from enum import Enum
from typing import Set, Dict, List
from dataclasses import dataclass

class Permission(Enum):
    """System permissions."""
    # Trading
    TRADING_VIEW = "trading:view"
    TRADING_EXECUTE = "trading:execute"
    TRADING_ADMIN = "trading:admin"

    # Strategy
    STRATEGY_VIEW = "strategy:view"
    STRATEGY_CREATE = "strategy:create"
    STRATEGY_MODIFY = "strategy:modify"
    STRATEGY_DELETE = "strategy:delete"

    # Risk
    RISK_VIEW = "risk:view"
    RISK_MODIFY = "risk:modify"
    RISK_OVERRIDE = "risk:override"
    KILL_SWITCH = "risk:kill_switch"

    # System
    SYSTEM_VIEW = "system:view"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_ADMIN = "system:admin"

    # Audit
    AUDIT_VIEW = "audit:view"
    AUDIT_EXPORT = "audit:export"

class Role(Enum):
    """Predefined roles."""
    VIEWER = "viewer"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    STRATEGY_DEVELOPER = "strategy_developer"
    ADMINISTRATOR = "administrator"
    SUPER_ADMIN = "super_admin"

# Role-permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.TRADING_VIEW,
        Permission.STRATEGY_VIEW,
        Permission.RISK_VIEW,
        Permission.SYSTEM_VIEW,
    },
    Role.TRADER: {
        Permission.TRADING_VIEW,
        Permission.TRADING_EXECUTE,
        Permission.STRATEGY_VIEW,
        Permission.RISK_VIEW,
        Permission.SYSTEM_VIEW,
    },
    Role.RISK_MANAGER: {
        Permission.TRADING_VIEW,
        Permission.RISK_VIEW,
        Permission.RISK_MODIFY,
        Permission.RISK_OVERRIDE,
        Permission.KILL_SWITCH,
        Permission.SYSTEM_VIEW,
        Permission.AUDIT_VIEW,
    },
    Role.STRATEGY_DEVELOPER: {
        Permission.TRADING_VIEW,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_MODIFY,
        Permission.STRATEGY_DELETE,
        Permission.RISK_VIEW,
        Permission.SYSTEM_VIEW,
    },
    Role.ADMINISTRATOR: {
        Permission.TRADING_VIEW,
        Permission.TRADING_EXECUTE,
        Permission.TRADING_ADMIN,
        Permission.STRATEGY_VIEW,
        Permission.STRATEGY_CREATE,
        Permission.STRATEGY_MODIFY,
        Permission.RISK_VIEW,
        Permission.RISK_MODIFY,
        Permission.SYSTEM_VIEW,
        Permission.SYSTEM_CONFIG,
        Permission.AUDIT_VIEW,
        Permission.AUDIT_EXPORT,
    },
    Role.SUPER_ADMIN: {
        p for p in Permission
    },  # All permissions
}

@dataclass
class User:
    """User with RBAC."""
    user_id: str
    username: str
    email: str
    roles: Set[Role]
    custom_permissions: Set[Permission]
    denied_permissions: Set[Permission]

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has specific permission."""
        if permission in self.denied_permissions:
            return False

        if permission in self.custom_permissions:
            return True

        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True

        return False

    def get_all_permissions(self) -> Set[Permission]:
        """Get all effective permissions."""
        permissions = set(self.custom_permissions)

        for role in self.roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))

        return permissions - self.denied_permissions


def require_permission(permission: Permission):
    """Decorator for permission-protected endpoints."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            user = get_current_user()  # From auth context
            if not user.has_permission(permission):
                raise PermissionDeniedError(
                    f"Permission {permission.value} required"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

## Deployment Configurations

### Docker Compose (Development/Small Production)

```yaml
# docker-compose.yml
version: '3.8'

services:
  trading-engine:
    build:
      context: .
      dockerfile: Dockerfile.trading
    environment:
      - DATABASE_URL=postgresql://trading:${DB_PASSWORD}@postgres:5432/trading
      - REDIS_URL=redis://redis:6379/0
      - VAULT_ADDR=http://vault:8200
    volumes:
      - ./configs:/app/configs:ro
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - postgres
      - redis
      - vault
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 32G
        reservations:
          cpus: '4'
          memory: 16G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=trading
      - POSTGRES_PASSWORD=${DB_PASSWORD}
      - POSTGRES_DB=trading
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Kubernetes (Production)

```yaml
# kubernetes/trading-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-engine
  namespace: trading
  labels:
    app: trading-engine
    tier: backend
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: trading-engine
  template:
    metadata:
      labels:
        app: trading-engine
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: trading-engine
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
        - name: trading-engine
          image: trading-platform:latest
          imagePullPolicy: Always
          ports:
            - containerPort: 8080
              name: http
            - containerPort: 9090
              name: metrics
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: trading-secrets
                  key: database-url
            - name: REDIS_URL
              valueFrom:
                secretKeyRef:
                  name: trading-secrets
                  key: redis-url
          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
            limits:
              cpu: "8"
              memory: "32Gi"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 8080
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: config
              mountPath: /app/configs
              readOnly: true
            - name: data
              mountPath: /app/data
      volumes:
        - name: config
          configMap:
            name: trading-config
        - name: data
          persistentVolumeClaim:
            claimName: trading-data
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchExpressions:
                    - key: app
                      operator: In
                      values:
                        - trading-engine
                topologyKey: kubernetes.io/hostname

---
apiVersion: v1
kind: Service
metadata:
  name: trading-engine
  namespace: trading
spec:
  type: ClusterIP
  ports:
    - port: 80
      targetPort: 8080
      name: http
    - port: 9090
      targetPort: 9090
      name: metrics
  selector:
    app: trading-engine

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: trading-engine-hpa
  namespace: trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: trading-engine
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
```

### Terraform (AWS Infrastructure)

```hcl
# terraform/main.tf

# VPC Configuration
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "trading-platform-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["eu-central-1a", "eu-central-1b", "eu-central-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = false
  one_nat_gateway_per_az = true

  enable_dns_hostnames = true
  enable_dns_support   = true

  # VPC Flow Logs for compliance
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true

  tags = {
    Environment = var.environment
    Project     = "trading-platform"
    Compliance  = "MiFID-II"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "trading-platform-${var.environment}"
  cluster_version = "1.28"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = false

  # Encryption at rest
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  eks_managed_node_groups = {
    trading = {
      name           = "trading-nodes"
      instance_types = ["r6i.2xlarge"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            encrypted             = true
            kms_key_id            = aws_kms_key.ebs.arn
            delete_on_termination = true
          }
        }
      }

      labels = {
        role = "trading"
      }

      taints = []
    }
  }

  # Enable IRSA
  enable_irsa = true

  tags = {
    Environment = var.environment
    Project     = "trading-platform"
  }
}

# RDS PostgreSQL
module "rds" {
  source  = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "trading-db-${var.environment}"

  engine               = "postgres"
  engine_version       = "15"
  family               = "postgres15"
  major_engine_version = "15"
  instance_class       = "db.r6i.xlarge"

  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds.arn

  db_name  = "trading"
  username = "trading_admin"
  port     = 5432

  multi_az               = true
  db_subnet_group_name   = module.vpc.database_subnet_group
  vpc_security_group_ids = [module.security_group_rds.security_group_id]

  # Backup configuration
  backup_retention_period = 35  # MiFID II: 5 years recommended
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  # Performance Insights
  performance_insights_enabled          = true
  performance_insights_retention_period = 7

  # Enhanced monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_monitoring.arn

  # Parameter group
  parameters = [
    {
      name  = "log_statement"
      value = "all"
    },
    {
      name  = "log_min_duration_statement"
      value = "1000"  # Log queries > 1s
    }
  ]

  deletion_protection = true

  tags = {
    Environment = var.environment
    Project     = "trading-platform"
    Compliance  = "MiFID-II"
  }
}

# ElastiCache Redis
module "elasticache" {
  source  = "terraform-aws-modules/elasticache/aws"
  version = "~> 1.0"

  cluster_id         = "trading-cache-${var.environment}"
  engine             = "redis"
  engine_version     = "7.0"
  node_type          = "cache.r6g.large"
  num_cache_nodes    = 2

  subnet_group_name  = module.vpc.elasticache_subnet_group_name
  security_group_ids = [module.security_group_redis.security_group_id]

  # Encryption
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                 = var.redis_auth_token

  # Automatic failover
  automatic_failover_enabled = true

  tags = {
    Environment = var.environment
    Project     = "trading-platform"
  }
}

# KMS Keys
resource "aws_kms_key" "eks" {
  description             = "KMS key for EKS cluster encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_key" "rds" {
  description             = "KMS key for RDS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}

resource "aws_kms_key" "ebs" {
  description             = "KMS key for EBS encryption"
  deletion_window_in_days = 30
  enable_key_rotation     = true
}
```

---

## Integration Patterns

### REST API Integration

```python
# Example: Signal Consumer Integration
import requests
from typing import List, Dict, Any
import hmac
import hashlib
import time

class TradingPlatformClient:
    """Client for integrating with trading platform API."""

    def __init__(self, base_url: str, api_key: str, api_secret: str):
        self.base_url = base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = requests.Session()

    def _sign_request(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """Generate HMAC signature for request."""
        message = f"{timestamp}{method}{path}{body}"
        return hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request."""
        timestamp = str(int(time.time() * 1000))
        body = kwargs.get("json", "")
        if body:
            import json
            body = json.dumps(body, sort_keys=True)

        signature = self._sign_request(timestamp, method, path, body)

        headers = {
            "X-API-Key": self.api_key,
            "X-Timestamp": timestamp,
            "X-Signature": signature,
            "Content-Type": "application/json",
        }

        response = self.session.request(
            method,
            f"{self.base_url}{path}",
            headers=headers,
            **kwargs
        )
        response.raise_for_status()
        return response.json()

    def get_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Get trading signals for symbols."""
        return self._request("GET", "/api/v2/signals", params={"symbols": symbols})

    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit order to platform."""
        return self._request("POST", "/api/v2/orders", json=order)

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions."""
        return self._request("GET", "/api/v2/positions")

    def get_risk_limits(self) -> Dict[str, Any]:
        """Get current risk limits."""
        return self._request("GET", "/api/v2/risk/limits")

    def update_risk_limits(self, limits: Dict[str, Any]) -> Dict[str, Any]:
        """Update risk limits."""
        return self._request("PUT", "/api/v2/risk/limits", json=limits)


# Usage example
if __name__ == "__main__":
    client = TradingPlatformClient(
        base_url="https://api.trading.example.com",
        api_key="your_api_key",
        api_secret="your_api_secret"
    )

    # Get signals
    signals = client.get_signals(["BTCUSDT", "ETHUSDT"])

    # Execute based on signal
    for signal in signals:
        if signal["strength"] > 0.7:
            order = {
                "symbol": signal["symbol"],
                "side": signal["direction"],
                "type": "market",
                "quantity": calculate_position_size(signal),
            }
            result = client.submit_order(order)
            print(f"Order submitted: {result}")
```

### WebSocket Integration

```python
# Example: Real-time Signal Consumer
import asyncio
import websockets
import json
from typing import Callable, Awaitable

class SignalWebSocket:
    """WebSocket client for real-time signal consumption."""

    def __init__(
        self,
        ws_url: str,
        api_key: str,
        on_signal: Callable[[dict], Awaitable[None]],
        on_risk_alert: Callable[[dict], Awaitable[None]],
    ):
        self.ws_url = ws_url
        self.api_key = api_key
        self.on_signal = on_signal
        self.on_risk_alert = on_risk_alert
        self._ws = None
        self._running = False

    async def connect(self):
        """Establish WebSocket connection."""
        self._ws = await websockets.connect(
            f"{self.ws_url}?api_key={self.api_key}",
            ping_interval=30,
            ping_timeout=10,
        )
        self._running = True

        # Subscribe to channels
        await self._ws.send(json.dumps({
            "action": "subscribe",
            "channels": ["signals", "risk_alerts", "system_health"]
        }))

    async def listen(self):
        """Listen for messages."""
        while self._running:
            try:
                message = await self._ws.recv()
                data = json.loads(message)

                if data["type"] == "signal":
                    await self.on_signal(data["payload"])
                elif data["type"] == "risk_alert":
                    await self.on_risk_alert(data["payload"])
                elif data["type"] == "heartbeat":
                    pass  # Connection alive

            except websockets.ConnectionClosed:
                if self._running:
                    await self._reconnect()

    async def _reconnect(self):
        """Reconnect with exponential backoff."""
        delays = [1, 2, 5, 10, 30, 60]
        for delay in delays:
            try:
                await asyncio.sleep(delay)
                await self.connect()
                return
            except Exception:
                continue
        raise ConnectionError("Failed to reconnect after multiple attempts")

    async def close(self):
        """Close connection."""
        self._running = False
        if self._ws:
            await self._ws.close()


# Usage example
async def handle_signal(signal: dict):
    """Process incoming trading signal."""
    print(f"Signal received: {signal}")
    # Implement your execution logic here

async def handle_risk_alert(alert: dict):
    """Process risk alert."""
    print(f"Risk alert: {alert}")
    # Implement your risk handling logic here

async def main():
    client = SignalWebSocket(
        ws_url="wss://api.trading.example.com/ws",
        api_key="your_api_key",
        on_signal=handle_signal,
        on_risk_alert=handle_risk_alert,
    )

    await client.connect()
    await client.listen()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Monitoring & Observability

### Prometheus Metrics

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: 'trading-engine'
    static_configs:
      - targets: ['trading-engine:9090']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Alert Rules

```yaml
# monitoring/rules/trading.yml
groups:
  - name: trading_alerts
    rules:
      # High latency alert
      - alert: HighOrderLatency
        expr: histogram_quantile(0.99, trading_order_latency_seconds_bucket) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High order latency detected"
          description: "P99 order latency is {{ $value }}s"

      # Position limit breach
      - alert: PositionLimitBreach
        expr: trading_position_value / trading_position_limit > 0.9
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Position approaching limit"
          description: "Position at {{ $value | humanizePercentage }} of limit"

      # Kill switch activation
      - alert: KillSwitchActivated
        expr: trading_kill_switch_active == 1
        labels:
          severity: critical
        annotations:
          summary: "Kill switch has been activated"
          description: "Trading has been halted by kill switch"

      # Drawdown alert
      - alert: DrawdownExceeded
        expr: trading_drawdown_percent > 5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Drawdown threshold exceeded"
          description: "Current drawdown: {{ $value }}%"

      # System health
      - alert: HighCPUUsage
        expr: process_cpu_percent > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage"
          description: "CPU usage at {{ $value }}%"

      # Database connection
      - alert: DatabaseConnectionFailed
        expr: trading_db_connection_status == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Database connection lost"
          description: "Unable to connect to database"
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Trading Platform Overview",
    "panels": [
      {
        "title": "Order Latency (P99)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, trading_order_latency_seconds_bucket)",
            "legendFormat": "P99 Latency"
          }
        ]
      },
      {
        "title": "Active Positions",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(trading_position_count)",
            "legendFormat": "Positions"
          }
        ]
      },
      {
        "title": "Risk Utilization",
        "type": "gauge",
        "targets": [
          {
            "expr": "trading_risk_utilization_percent",
            "legendFormat": "Risk %"
          }
        ]
      },
      {
        "title": "Orders per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(trading_orders_total[5m])",
            "legendFormat": "Orders/sec"
          }
        ]
      }
    ]
  }
}
```

---

## Disaster Recovery

### Backup Strategy

```yaml
# Backup Configuration
backup:
  database:
    type: "continuous"
    provider: "AWS RDS"
    retention:
      automated: 35  # days
      snapshots: 365  # days (MiFID II: 5 years recommended)
    point_in_time_recovery: true
    cross_region_replica: true
    replica_region: "eu-west-1"  # DR region

  configuration:
    type: "versioned"
    provider: "Git + S3"
    encryption: "AES-256"

  audit_logs:
    type: "continuous"
    provider: "S3 + Glacier"
    retention: 2555  # 7 years (MiFID II compliance)
    encryption: "KMS"
    immutable: true

  state_snapshots:
    frequency: "hourly"
    retention: "30 days"
    provider: "S3"
```

### Recovery Procedures

```python
# services/disaster_recovery.py
from typing import Optional
from datetime import datetime, timedelta
import subprocess

class DisasterRecovery:
    """Disaster recovery procedures."""

    def __init__(self, config: dict):
        self.config = config

    def initiate_failover(self, region: str) -> bool:
        """
        Initiate failover to DR region.

        Steps:
        1. Verify DR region health
        2. Promote RDS read replica to primary
        3. Update DNS to point to DR region
        4. Restart services in DR region
        5. Verify connectivity
        """
        self._verify_dr_region(region)
        self._promote_database_replica(region)
        self._update_dns(region)
        self._restart_services(region)
        return self._verify_failover(region)

    def restore_from_backup(
        self,
        backup_timestamp: datetime,
        target_environment: str
    ) -> bool:
        """
        Restore system from backup.

        Steps:
        1. Create new database from snapshot
        2. Restore configuration from Git
        3. Restore state from S3
        4. Verify data integrity
        5. Start services
        """
        pass

    def point_in_time_recovery(
        self,
        target_time: datetime,
        target_environment: str
    ) -> bool:
        """
        Restore database to specific point in time.

        Used for:
        - Accidental data deletion
        - Corruption recovery
        - Regulatory investigation
        """
        pass

    def test_recovery(self, scenario: str) -> dict:
        """
        Test recovery procedures.

        Required quarterly for DORA compliance.

        Scenarios:
        - database_failure
        - region_outage
        - data_corruption
        - ransomware_simulation
        """
        pass
```

### RTO/RPO Targets

| Scenario | RTO | RPO | Notes |
|----------|-----|-----|-------|
| **Database Failure** | 15 min | 0 (continuous) | Automatic failover |
| **Region Outage** | 1 hour | 5 min | Cross-region replica |
| **Data Corruption** | 2 hours | Point-in-time | Manual recovery |
| **Full DR Test** | 4 hours | N/A | Quarterly testing |

---

## Performance Tuning

### Python Optimization

```python
# Performance configuration
import os

# Use optimized allocator
os.environ["MALLOC_ARENA_MAX"] = "2"

# NumPy threading
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Disable GC during critical paths
import gc
gc.disable()  # Re-enable periodically

# Use memory-efficient data structures
import numpy as np
# Use float32 instead of float64 where precision allows
prices = np.array(prices, dtype=np.float32)
```

### Database Optimization

```sql
-- PostgreSQL performance tuning
-- For 128GB RAM server

-- Memory settings
ALTER SYSTEM SET shared_buffers = '32GB';
ALTER SYSTEM SET effective_cache_size = '96GB';
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';

-- Write performance
ALTER SYSTEM SET wal_buffers = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET max_wal_size = '4GB';

-- Query planning
ALTER SYSTEM SET random_page_cost = 1.1;  -- SSD storage
ALTER SYSTEM SET effective_io_concurrency = 200;

-- Parallel query
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Indexes for common queries
CREATE INDEX CONCURRENTLY idx_orders_symbol_timestamp
    ON orders (symbol, created_at DESC);
CREATE INDEX CONCURRENTLY idx_positions_active
    ON positions (symbol) WHERE closed_at IS NULL;
CREATE INDEX CONCURRENTLY idx_audit_timestamp
    ON audit_events (timestamp DESC);
```

### Redis Optimization

```conf
# redis.conf optimization

# Memory
maxmemory 8gb
maxmemory-policy volatile-lru

# Persistence (trade-off: performance vs durability)
appendonly yes
appendfsync everysec

# Connection handling
timeout 0
tcp-keepalive 300
maxclients 10000

# Cluster mode (for scale)
cluster-enabled yes
cluster-node-timeout 15000
```

---

## Conclusion

This technical implementation guide provides the foundation for enterprise-grade deployments following the CCEA architecture. For specific customization or additional requirements, please contact our enterprise team.

**Key CCEA Guarantees for Enterprise**:
- Cloud **NEVER** stores broker API keys or credentials
- Cloud **NEVER** creates, transmits, or executes orders
- Agent deployed in customer infrastructure is the **ONLY** execution point
- All TRADING_IMPACTING changes require local approval
- Full audit trail for compliance (MiFID II, DORA)

**Support Contacts**:
- Technical Support: support@[company].com
- Enterprise Sales: enterprise@[company].com
- Security Team: security@[company].com

---

*Document Version: 2.0*
*Last Updated: December 2025*
*Classification: Technical - Confidential*
*Aligned with: Design Doc CCEA Cloud v1.0*
