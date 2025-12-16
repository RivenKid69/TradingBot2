# Degraded Mode Operations Runbook

> **Severity**: Medium | **Last Updated**: 2025-12-16

## Overview

This runbook covers handling various degraded operational modes and their recovery.

---

## Degraded Mode Types

| Mode | Impact | Action |
|------|--------|--------|
| Cloud Unreachable | No commands/telemetry | Continue/Pause/Halt |
| Data Feed Invalid | Bad market data | Halt |
| Broker Errors | API failures | Pause |
| Time Drift | Clock out of sync | Halt |
| Resource Exhaustion | System overload | Pause |
| State Divergence | State mismatch | Halt |

---

## Cloud Unreachable

### Symptoms
- Heartbeat failures in logs
- "Cloud unreachable" warnings
- Telemetry buffer growing

### Diagnosis
```bash
# Check agent status
ccea-agent status

# Check Cloud connectivity
ccea-agent cloud ping

# View telemetry buffer
ccea-agent telemetry buffer-status
```

### Response

**If action=continue (default):**
- Trading continues with last known config
- Telemetry buffers locally
- No new deployments possible
- Monitor closely

**If action=pause:**
- Trading paused
- Positions maintained
- Await recovery

**If action=halt:**
- Trading stopped
- Orders cancelled
- Positions maintained

### Recovery
```bash
# Wait for automatic reconnection
# Or force reconnect
ccea-agent cloud reconnect

# Flush telemetry buffer when connected
ccea-agent telemetry flush
```

### Escalation
- After 1 hour: Alert on-call
- After 4 hours: Escalate to platform team
- After 24 hours: Consider manual halt

---

## Data Feed Invalid

### Symptoms
- "Data feed invalid" alerts
- Kill switch may trigger
- Strategy receiving stale data

### Diagnosis
```bash
# Check data feed status
ccea-agent data-feed status

# View last received data
ccea-agent data-feed last-update

# Check for staleness
ccea-agent doctor --check data-feed
```

### Response
1. Agent automatically halts trading
2. Open orders cancelled
3. Positions maintained

### Recovery
```bash
# Check if data feed recovered
ccea-agent data-feed status

# If auto-recovery after grace period:
# Agent will resume automatically

# If manual recovery needed:
ccea-agent kill-switch acknowledge
ccea-agent start
```

### Escalation
- Immediate: Check broker status page
- After 5 minutes: Contact broker support
- After 30 minutes: Consider alternate data source

---

## Broker Errors

### Symptoms
- Order rejections
- API timeouts
- Rate limiting messages

### Diagnosis
```bash
# Check broker status
ccea-agent broker status

# View error counts
ccea-agent broker errors --last 60m

# Check rate limits
ccea-agent broker rate-limit-status
```

### Response
**If errors < threshold:**
- Retry with backoff
- Continue trading

**If errors >= threshold:**
- Trading paused
- No new orders
- Existing orders maintained

### Recovery
```bash
# Check if errors cleared
ccea-agent broker errors --last 5m

# If rate limited, wait
sleep 60

# Resume
ccea-agent start
```

### Escalation
- Immediate: Check broker status page
- After 10 minutes: Review API permissions
- After 30 minutes: Contact broker support

---

## Time Drift

### Symptoms
- "Time drift" alerts
- Timestamp validation failures
- Kill switch triggered

### Diagnosis
```bash
# Check time sync
ccea-agent doctor --check time

# View drift
ccea-agent time status

# Compare with NTP
ntpdate -q time.google.com
```

### Response
1. Agent halts trading immediately
2. Orders cancelled
3. Positions maintained

### Recovery
```bash
# Sync system time
sudo ntpdate -u time.google.com

# Or restart time service
sudo systemctl restart systemd-timesyncd

# Verify
ccea-agent doctor --check time

# Acknowledge and resume
ccea-agent kill-switch acknowledge
ccea-agent start
```

### Prevention
- Enable automatic NTP sync
- Monitor time drift
- Alert on >500ms drift

---

## Resource Exhaustion

### Symptoms
- High CPU/memory warnings
- Slow response times
- Out of memory errors

### Diagnosis
```bash
# Check resources
ccea-agent resource status

# System resources
free -h
df -h
top
```

### Response
**If above threshold:**
- Trading paused
- Non-essential processes stopped
- GC triggered

### Recovery
```bash
# Free memory
ccea-agent cache clear

# Rotate logs
ccea-agent logs rotate

# Resume
ccea-agent start
```

### Prevention
- Set appropriate resource limits
- Monitor disk space
- Configure log rotation

---

## State Divergence

### Symptoms
- Position mismatch alerts
- Unknown orders detected
- Reconciliation failures

### Diagnosis
```bash
# Compare states
ccea-agent reconcile diff

# View differences
ccea-agent positions list --compare
ccea-agent orders list --compare
```

### Response
1. Agent halts immediately
2. Manual intervention required
3. **NO auto-recovery**

### Recovery
```bash
# Stop agent
ccea-agent stop

# Investigate cause
ccea-agent reconcile diff --verbose

# Resolve
# Option A: Trust broker
ccea-agent reconcile resolve --trust broker

# Option B: Manual resolution
# - Cancel unknown orders at broker
# - Adjust local journal

# Verify
ccea-agent reconcile verify

# Resume
ccea-agent start
```

### Escalation
- Immediate: Review all differences
- Before resolution: Document current state
- After resolution: Root cause analysis

---

## Monitoring Dashboard

### Key Metrics
```
┌─────────────────────────────────────────────────────────────────────┐
│ DEGRADED MODE STATUS                                                │
├─────────────────────────────────────────────────────────────────────┤
│ Cloud Connection:    🟢 OK (last heartbeat: 5s ago)                │
│ Data Feed:           🟢 OK (last update: 100ms ago)                │
│ Broker:              🟢 OK (errors: 0 in last 60m)                 │
│ Time Sync:           🟢 OK (drift: +15ms)                          │
│ Resources:           🟢 OK (CPU: 25%, MEM: 45%)                    │
│ State:               🟢 OK (reconciled)                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Alert Thresholds
| Metric | Warning | Critical |
|--------|---------|----------|
| Cloud disconnect | 5 min | 30 min |
| Data staleness | 10 sec | 30 sec |
| Broker errors | 3 | 5 |
| Time drift | 500 ms | 1000 ms |
| CPU usage | 80% | 95% |
| Memory usage | 80% | 90% |

---

## Related

- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Recovery Procedures](./RECOVERY.md)
- [Degraded Modes Reference](../agent/DEGRADED_MODES.md)
