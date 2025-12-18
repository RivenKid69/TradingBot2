# Latency Issues Runbook

> **Severity**: Medium-High | **Last Updated**: 2025-12-15

## Overview

This runbook covers diagnosing and resolving latency issues across the CCEA architecture:
- **Agent zone**: Order execution, broker communication
- **Cloud zone**: Telemetry ingestion, control plane response

---

## Latency Thresholds

| Component | Normal | Warning | Critical | Kill Switch |
|-----------|--------|---------|----------|-------------|
| Broker API | <100ms | 100-500ms | 500-2000ms | >5000ms |
| Order execution | <200ms | 200-1000ms | 1-5s | >10s |
| Telemetry upload | <500ms | 500-2000ms | 2-10s | N/A |
| Cloud API | <200ms | 200-1000ms | 1-5s | N/A |
| Data feed | <50ms | 50-200ms | 200-1000ms | >2000ms |

---

## 1. Quick Diagnosis

```bash
# Overall latency check
ccea-agent doctor --check latency

# Detailed latency metrics
ccea-agent metrics latency

# Expected output:
# Broker API Latency:
#   avg: 45ms, p50: 40ms, p95: 80ms, p99: 150ms
# Order Execution:
#   avg: 120ms, p50: 100ms, p95: 250ms, p99: 500ms
# Telemetry Upload:
#   avg: 200ms, p50: 180ms, p95: 400ms, p99: 800ms
```

---

## 2. Latency Source Identification

### 2.1 Network Latency

```bash
# Check network path to broker
ccea-agent broker ping --count 10

# Check network path to cloud
ccea-agent cloud ping --count 10

# Trace route (if available)
ccea-agent network trace --target broker
ccea-agent network trace --target cloud
```

### 2.2 Processing Latency

```bash
# Check CPU/memory usage
ccea-agent system status

# Check event queue depth
ccea-agent queue status

# Check strategy execution time
ccea-agent metrics strategy-latency
```

### 2.3 Broker-Side Latency

```bash
# Compare with broker-reported latency
ccea-agent broker latency-report

# Check broker status page for known issues
# https://status.broker.com
```

---

## 3. Issue-Specific Procedures

### 3.1 High Broker API Latency

**Symptoms:**
- Order submission taking >500ms
- Heartbeat delays
- Timeout errors

**Procedure:**

```bash
# 1. Check current latency
ccea-agent broker latency

# 2. If consistently high, check broker status
# (external check required)

# 3. If broker OK, check local network
ccea-agent network test

# 4. Consider switching to alternative endpoint
ccea-agent broker config set endpoint wss://stream2.broker.com

# 5. If persistent, reduce trading frequency
ccea-agent config set trading.order_throttle 2/second

# 6. Enable latency alerts
ccea-agent alerts enable broker.latency --warning 500ms --critical 2000ms
```

### 3.2 Order Execution Delays

**Symptoms:**
- Orders taking long to fill
- Strategy decisions delayed
- Position updates slow

**Procedure:**

```bash
# 1. Check execution timing breakdown
ccea-agent metrics execution-breakdown

# Output shows:
# - Signal generation: 10ms
# - Order creation: 5ms
# - Order submission: 200ms  <-- bottleneck
# - Confirmation wait: 50ms

# 2. If submission is slow, check broker latency (3.1)

# 3. If signal generation slow, profile strategy
ccea-agent strategy profile --duration 5m

# 4. Optimize hot path
ccea-agent config set execution.batch_orders true
ccea-agent config set execution.async_fills true
```

### 3.3 Telemetry Upload Delays

**Symptoms:**
- Dashboard data stale
- Alerts delayed
- Cloud monitoring gaps

**Procedure:**

```bash
# 1. Check telemetry queue
ccea-agent telemetry status

# 2. If backlogged, check cloud connectivity
ccea-agent cloud status

# 3. Reduce telemetry frequency temporarily
ccea-agent config set telemetry.batch_interval 60s

# 4. If persistent, check:
# - Network egress
# - TLS handshake time
# - DNS resolution

# 5. Enable local buffering
ccea-agent config set telemetry.local_buffer_size 10000
```

**Note:** Telemetry delays don't affect trading - Agent operates independently.

### 3.4 Data Feed Latency

**Symptoms:**
- Price data stale
- Strategy using outdated quotes
- Spread/slippage increased

**Procedure:**

```bash
# 1. Check data feed latency
ccea-agent data-feed status

# 2. Compare with exchange time
ccea-agent data-feed time-drift

# 3. If drift detected, check NTP
ccea-agent system time-sync

# 4. If feed source slow, switch source
ccea-agent data-feed switch --source websocket

# 5. For critical latency, pause trading
ccea-agent pause --reason "Data feed latency"
```

### 3.5 Strategy Processing Delays

**Symptoms:**
- Signal generation slow
- CPU usage high
- Memory pressure

**Procedure:**

```bash
# 1. Profile strategy
ccea-agent strategy profile --duration 5m

# 2. Check resource usage
ccea-agent system resources

# 3. If CPU bound, optimize:
ccea-agent config set strategy.vectorize true
ccea-agent config set strategy.cache_indicators true

# 4. If memory bound:
ccea-agent config set strategy.rolling_window 1000
ccea-agent config set gc.aggressive true

# 5. Consider strategy simplification
```

---

## 4. Kill Switch Triggers

### Automatic Latency-Based Kill Switch

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Broker latency | >5000ms sustained 30s | Kill switch |
| Order timeout | 5 consecutive | Kill switch |
| Data feed stale | >30s | Kill switch |
| State divergence | Any | Kill switch |

### Manual Kill Switch

```bash
# If latency is causing execution issues
ccea-agent kill-switch trigger --reason "Latency: <specific issue>"
```

---

## 5. Recovery

### 5.1 After Latency Resolution

```bash
# 1. Verify latency returned to normal
ccea-agent metrics latency

# 2. Reconcile any missed updates
ccea-agent reconcile all

# 3. Resume with monitoring
ccea-agent start --reduced-risk

# 4. Watch for recurrence
ccea-agent status --watch --metrics latency
```

### 5.2 Post-Incident Analysis

```bash
# Export latency data for analysis
ccea-agent metrics export --type latency --duration 24h --output latency_analysis.json

# Generate latency report
ccea-agent report latency --format markdown > latency_report.md
```

---

## 6. Prevention

### 6.1 Latency Monitoring

```bash
# Enable latency alerts
ccea-agent alerts enable latency.broker --warning 500ms --critical 2000ms
ccea-agent alerts enable latency.execution --warning 1000ms --critical 5000ms
ccea-agent alerts enable latency.data_feed --warning 200ms --critical 1000ms
```

### 6.2 Configuration Best Practices

```yaml
# agent-config.yaml
latency:
  # Connection settings
  broker_timeout: 10s
  order_timeout: 30s
  data_feed_timeout: 5s

  # Monitoring
  sample_interval: 1s
  alert_window: 30s
  histogram_buckets: [10, 50, 100, 200, 500, 1000, 2000, 5000]

  # Auto-response
  auto_pause_threshold: 5000ms
  auto_pause_duration: 60s
  kill_switch_threshold: 10000ms
```

### 6.3 Infrastructure Recommendations

| Component | Recommendation |
|-----------|---------------|
| **Network** | Dedicated connection, low-latency route to broker |
| **Compute** | Isolated CPU cores for strategy |
| **Time sync** | NTP or PTP with <1ms accuracy |
| **Colocation** | Consider broker-side hosting for HFT |

---

## 7. Escalation

If latency persists:

1. **Broker issues**: Contact broker support
2. **Network issues**: Contact network provider
3. **Cloud issues**: Contact platform support with:
   - Latency metrics export
   - Timeline of issues
   - Network trace data

---

## 8. Related

- [Broker Errors Runbook](./BROKER_ERRORS.md)
- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Recovery Procedures](./RECOVERY.md)
- [Agent Docs](../agent/README.md)
