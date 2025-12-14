# Research Job Isolation & Anti-Abuse

> **Version**: 1.0.0 | **Last Updated**: 2025-12-14

## Overview

Cloud research jobs (backtests, simulations, training runs) execute user-provided code in isolated sandboxes. This document describes the isolation mechanisms, resource quotas, and abuse prevention measures.

## Security Requirements

User code execution in Cloud must:
1. **Isolate** - No access to other tenants' data or resources
2. **Limit** - CPU, RAM, time, and network restrictions
3. **Monitor** - Detect and prevent abuse patterns
4. **Audit** - Log all resource usage and anomalies

---

## 1. Sandbox Architecture

### 1.1 Isolation Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLOUD COMPUTE CLUSTER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    VM/Node Isolation (per tenant pool)               │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │              Container Isolation (per job)                    │   │    │
│  │  │  ┌────────────────────────────────────────────────────────┐  │   │    │
│  │  │  │           gVisor/Firecracker (syscall filtering)       │  │   │    │
│  │  │  │  ┌──────────────────────────────────────────────────┐  │  │   │    │
│  │  │  │  │              User Code (sandboxed)               │  │  │   │    │
│  │  │  │  │  - No network (default)                          │  │  │   │    │
│  │  │  │  │  - Read-only filesystem                          │  │  │   │    │
│  │  │  │  │  - No privileged operations                      │  │   │    │
│  │  │  │  └──────────────────────────────────────────────────┘  │  │   │    │
│  │  │  └────────────────────────────────────────────────────────┘  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Node | Dedicated VM pool per tenant tier | Hardware isolation |
| Container | containerd + gVisor | Syscall filtering |
| Network | Cilium + eBPF | Network policy |
| Filesystem | Read-only overlay | Data protection |
| Secrets | None in sandbox | Security |

---

## 2. Resource Quotas

### 2.1 Per-Job Limits

| Resource | Free Tier | Pro Tier | Enterprise |
|----------|-----------|----------|------------|
| CPU cores | 2 | 8 | 32 |
| RAM | 4 GB | 16 GB | 64 GB |
| Max runtime | 30 min | 4 hours | 24 hours |
| Disk (scratch) | 10 GB | 50 GB | 200 GB |
| Network egress | None | Allowlist only | Custom |

### 2.2 Per-Workspace Limits (Concurrent)

| Resource | Free Tier | Pro Tier | Enterprise |
|----------|-----------|----------|------------|
| Concurrent jobs | 2 | 10 | 100 |
| Total CPU | 4 | 80 | 320 |
| Total RAM | 8 GB | 160 GB | 640 GB |
| Daily job hours | 2 | 40 | Unlimited |

### 2.3 Quota Configuration

```yaml
# workspace-quotas.yaml
quotas:
  tier: pro
  per_job:
    cpu_cores: 8
    memory_gb: 16
    max_runtime_minutes: 240
    scratch_disk_gb: 50
    network: allowlist
  concurrent:
    max_jobs: 10
    max_cpu: 80
    max_memory_gb: 160
  daily:
    max_job_hours: 40
    max_egress_gb: 10
```

---

## 3. Network Restrictions

### 3.1 Default: No Network

By default, research jobs have **no network access**:

```python
# Job configuration
job_config = {
    "network": {
        "mode": "none"  # Default
    }
}
```

### 3.2 Allowlist Mode (Pro+)

For jobs requiring external data:

```yaml
network:
  mode: allowlist
  allowed_destinations:
    - "api.binance.com:443"      # Market data
    - "api.polygon.io:443"       # Market data
    - "pypi.org:443"             # Package install (build only)
  denied_destinations:
    - "*"                        # Everything else blocked
  protocols:
    - https
  max_egress_mb_per_hour: 100
```

### 3.3 Network Policy Enforcement

```yaml
# Cilium NetworkPolicy
apiVersion: cilium.io/v2
kind: CiliumNetworkPolicy
metadata:
  name: research-job-egress
spec:
  endpointSelector:
    matchLabels:
      job-type: research
  egress:
    - toFQDNs:
        - matchName: "api.binance.com"
        - matchName: "api.polygon.io"
      toPorts:
        - ports:
            - port: "443"
              protocol: TCP
```

---

## 4. Filesystem Restrictions

### 4.1 Mount Configuration

```yaml
filesystems:
  root: read_only          # Base image is read-only
  scratch: rw              # Temporary workspace
  data: read_only          # Historical data mount
  output: rw               # Results output
  secrets: none            # No secrets in sandbox
```

### 4.2 Directory Structure

```
/
├── app/                   # Read-only: Strategy code
├── data/                  # Read-only: Historical data
├── scratch/               # Read-write: Temp files (quota limited)
├── output/                # Read-write: Results (persisted)
└── home/                  # Read-write: User home (scratch)
```

### 4.3 Prohibited Access

- `/proc/` - Limited (gVisor filter)
- `/sys/` - Limited (gVisor filter)
- `/dev/` - Limited to essential only
- Network namespaces - Isolated
- IPC namespaces - Isolated

---

## 5. Abuse Detection

### 5.1 Abuse Patterns

| Pattern | Detection | Action |
|---------|-----------|--------|
| Crypto mining | CPU pattern analysis | Terminate + ban |
| Port scanning | Network anomaly | Terminate + alert |
| DDoS participation | Egress analysis | Terminate + ban |
| Data exfiltration | Egress volume spike | Throttle + alert |
| Credential stealing | Memory scanning | Terminate + alert |
| Bot activity | Behavioral analysis | Rate limit |

### 5.2 Detection Methods

#### CPU Pattern Analysis

```python
def detect_mining(job_metrics: JobMetrics) -> bool:
    """Detect cryptocurrency mining patterns."""
    return (
        job_metrics.cpu_utilization > 0.95 and  # High sustained CPU
        job_metrics.cpu_variance < 0.05 and      # Constant load
        job_metrics.memory_growth < 0.01 and     # Low memory growth
        job_metrics.io_operations < 100          # Minimal I/O
    )
```

#### Network Anomaly Detection

```python
def detect_scanning(network_metrics: NetworkMetrics) -> bool:
    """Detect port scanning or network abuse."""
    return (
        network_metrics.unique_destinations > 100 or  # Many targets
        network_metrics.connection_failures > 50 or    # Failed connections
        network_metrics.short_connections > 200         # Quick connects
    )
```

### 5.3 Automated Response

```yaml
abuse_response:
  mining_detected:
    action: terminate_and_ban
    ban_duration_days: 30
    notify: [security_team, account_owner]

  scanning_detected:
    action: terminate_and_alert
    rate_limit_hours: 24
    notify: [security_team]

  data_exfiltration:
    action: throttle_egress
    max_egress_kbps: 10
    notify: [security_team, account_owner]
```

---

## 6. Job Lifecycle

### 6.1 States

```
┌───────────┐    ┌───────────┐    ┌───────────┐    ┌───────────┐
│  PENDING  │───▶│  RUNNING  │───▶│ COMPLETED │    │  FAILED   │
└───────────┘    └─────┬─────┘    └───────────┘    └───────────┘
                       │                               ▲
                       │ timeout/abuse                 │
                       └───────────────────────────────┘
```

### 6.2 Job Submission

```python
# Submit research job
job = cloud_client.research.submit_job(
    workspace_id="ws_abc123",
    artifact_digest="sha256:xxx",
    job_type="backtest",
    config={
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "symbols": ["BTCUSDT", "ETHUSDT"]
    },
    resources={
        "cpu_cores": 4,
        "memory_gb": 8,
        "max_runtime_minutes": 60
    }
)

# Monitor status
while not job.is_complete:
    status = cloud_client.research.get_job_status(job.id)
    print(f"Status: {status.state}, Progress: {status.progress_percent}%")
    time.sleep(10)

# Get results
results = cloud_client.research.get_job_results(job.id)
```

### 6.3 Job Timeout

```yaml
timeout:
  soft_timeout_minutes: 55     # Warning issued
  hard_timeout_minutes: 60     # Job terminated
  grace_period_seconds: 30     # Cleanup time
```

---

## 7. Monitoring & Alerting

### 7.1 Job Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `job.cpu_percent` | CPU utilization | >95% sustained |
| `job.memory_percent` | Memory usage | >90% |
| `job.runtime_minutes` | Execution time | Near quota |
| `job.egress_bytes` | Network egress | Near quota |
| `job.io_operations` | Disk I/O | Excessive |

### 7.2 Cluster Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cluster.pending_jobs` | Queue depth | >100 |
| `cluster.utilization` | Overall usage | >80% |
| `cluster.failed_jobs` | Failure rate | >5% |

### 7.3 Alert Configuration

```yaml
alerts:
  - name: high_failure_rate
    condition: cluster.failed_jobs_rate > 0.05
    severity: warning
    notify: [ops_team]

  - name: abuse_detected
    condition: job.abuse_score > 0.8
    severity: critical
    notify: [security_team]
    action: terminate_job

  - name: quota_exceeded
    condition: workspace.usage > workspace.quota * 0.9
    severity: warning
    notify: [account_owner]
```

---

## 8. Enterprise Features

### 8.1 Dedicated Compute

```yaml
# Enterprise dedicated pool
compute:
  mode: dedicated
  node_type: c6i.8xlarge
  min_nodes: 2
  max_nodes: 10
  spot_instances: false
```

### 8.2 Custom Network Policies

```yaml
# Enterprise custom egress
network:
  mode: custom
  allowed_destinations:
    - "*.internal.company.com"
    - "10.0.0.0/8"
  vpc_peering:
    enabled: true
    peer_vpc_id: vpc-xxx
```

### 8.3 Air-Gapped Mode

```yaml
# Enterprise air-gapped
compute:
  mode: air_gapped
  network: none
  data_import:
    method: offline_bundle
    encryption: required
```

---

## 9. Best Practices

### For Users

1. **Estimate resources accurately** - Avoid over-provisioning
2. **Use incremental backtests** - Test on small date ranges first
3. **Cache intermediate results** - Save to scratch disk
4. **Handle timeouts gracefully** - Checkpoint progress

### For Platform Operators

1. **Review abuse patterns weekly** - Tune detection rules
2. **Monitor cluster utilization** - Scale proactively
3. **Audit network egress** - Review allowlists
4. **Update security configs** - Patch gVisor/containerd

---

## 10. Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Job OOMKilled | Memory exceeded | Increase memory or optimize code |
| Job timeout | Long runtime | Optimize or split into smaller jobs |
| Network denied | Destination not in allowlist | Request allowlist update |
| Filesystem full | Scratch disk full | Clean temp files |
| Job stuck pending | Cluster at capacity | Wait or upgrade tier |

### Debugging

```bash
# Get job logs
ccea jobs logs job_abc123

# Get job metrics
ccea jobs metrics job_abc123 --format json

# Get job events
ccea jobs events job_abc123
```

---

**Related Documentation:**
- [CCEA Overview](../CCEA_OVERVIEW.md)
- [Resource Quotas](./GOVERNANCE.md#resource-quotas)
- [Security Trust Center](../security/TRUST_CENTER.md)
