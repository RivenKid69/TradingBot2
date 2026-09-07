# Degraded Modes & Safe Operations

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

The Agent is designed to operate safely even when components fail. Degraded modes ensure predictable, safe behavior during failures.

## Design Principle

```
Agent MUST:
  - NEVER make unsafe trading decisions during failures
  - ALWAYS prefer safety over profitability
  - Operate independently from Cloud when necessary
  - Provide clear status and recovery options
```

---

## Degraded Mode Types

### 1. Cloud Unreachable

**Trigger:** Cannot connect to Cloud API

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEGRADED MODE: CLOUD_UNREACHABLE                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Symptoms:                                                                  │
│    - Heartbeat failing                                                      │
│    - Command poll timing out                                               │
│    - Telemetry upload failing                                              │
│                                                                              │
│  Behavior:                                                                  │
│    - Trading CONTINUES (if configured)                                     │
│    - Telemetry buffered locally                                            │
│    - No new commands processed                                             │
│    - Approvals cannot complete                                             │
│                                                                              │
│  Auto-recovery:                                                            │
│    - Retry connection with exponential backoff                             │
│    - Resume normal operation when connected                                │
│    - Flush telemetry buffer                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
degraded_mode:
  cloud_unreachable:
    action: continue     # continue, pause, halt
    max_duration_hours: 24
    alert_after_minutes: 5
```

**Actions:**

| Action | Behavior |
|--------|----------|
| `continue` | Keep trading using last known config |
| `pause` | Pause strategy, maintain positions |
| `halt` | Stop strategy, cancel orders |

---

### 2. Data Feed Invalid

**Trigger:** Market data is stale, missing, or invalid

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEGRADED MODE: DATA_FEED_INVALID                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Symptoms:                                                                  │
│    - No market data updates                                                │
│    - Price unchanged for too long                                          │
│    - Invalid price values (0, NaN, negative)                               │
│    - Timestamp drift                                                       │
│                                                                              │
│  Behavior:                                                                  │
│    - Trading HALTED immediately                                            │
│    - Open orders cancelled                                                 │
│    - Alert generated                                                       │
│                                                                              │
│  Auto-recovery:                                                            │
│    - Monitor for valid data                                                │
│    - Resume after grace period with valid data                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
degraded_mode:
  data_feed_invalid:
    action: halt
    detection:
      max_staleness_seconds: 30
      price_validation: true
    grace_period_seconds: 30
    auto_recover: true
```

---

### 3. Broker Errors

**Trigger:** Repeated broker API errors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEGRADED MODE: BROKER_ERRORS                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Symptoms:                                                                  │
│    - Order rejections                                                      │
│    - API rate limiting                                                     │
│    - Authentication failures                                               │
│    - Connection timeouts                                                   │
│                                                                              │
│  Behavior:                                                                  │
│    - Trading PAUSED after threshold                                        │
│    - Existing orders maintained                                            │
│    - Retry with backoff                                                    │
│                                                                              │
│  Auto-recovery:                                                            │
│    - Clear error counter after window                                      │
│    - Resume when API stable                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
degraded_mode:
  broker_errors:
    action: pause
    error_threshold: 5
    window_seconds: 60
    retry:
      max_attempts: 3
      backoff_seconds: 5
    auto_recover: true
```

---

### 4. Time Drift

**Trigger:** System clock out of sync

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEGRADED MODE: TIME_DRIFT                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Symptoms:                                                                  │
│    - NTP sync failing                                                      │
│    - Clock drift > threshold                                               │
│    - Timestamp validation failures                                         │
│                                                                              │
│  Behavior:                                                                  │
│    - Trading HALTED (time-sensitive operations unsafe)                     │
│    - All orders cancelled                                                  │
│    - Alert generated                                                       │
│                                                                              │
│  Auto-recovery:                                                            │
│    - Attempt NTP sync                                                      │
│    - Resume after clock corrected                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
degraded_mode:
  time_drift:
    action: halt
    max_drift_ms: 5000
    check_interval_seconds: 60
    ntp_servers:
      - time.google.com
      - pool.ntp.org
```

---

### 5. Resource Exhaustion

**Trigger:** System resources depleted

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEGRADED MODE: RESOURCE_EXHAUSTION                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Symptoms:                                                                  │
│    - Memory > 90%                                                          │
│    - CPU > 95%                                                             │
│    - Disk full                                                             │
│    - File descriptors exhausted                                            │
│                                                                              │
│  Behavior:                                                                  │
│    - Trading PAUSED                                                        │
│    - Non-essential processes stopped                                       │
│    - Alert generated                                                       │
│                                                                              │
│  Auto-recovery:                                                            │
│    - GC triggered                                                          │
│    - Log rotation                                                          │
│    - Resume when resources available                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
degraded_mode:
  resource_exhaustion:
    action: pause
    thresholds:
      memory_percent: 90
      cpu_percent: 95
      disk_percent: 95
    recovery:
      trigger_gc: true
      rotate_logs: true
```

---

### 6. State Divergence

**Trigger:** Local state differs from broker state

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEGRADED MODE: STATE_DIVERGENCE                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Symptoms:                                                                  │
│    - Position mismatch                                                     │
│    - Unknown orders                                                        │
│    - Missing fills                                                         │
│                                                                              │
│  Behavior:                                                                  │
│    - Trading HALTED immediately                                            │
│    - Manual intervention required                                          │
│    - Full reconciliation logged                                            │
│                                                                              │
│  Auto-recovery:                                                            │
│    - None (manual only)                                                    │
│    - Requires operator acknowledgment                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**

```yaml
degraded_mode:
  state_divergence:
    action: halt
    tolerance:
      position_qty: 0.001  # Allow small rounding
      balance: 0.01
    auto_recover: false    # NEVER auto-recover state issues
```

---

## Degraded Mode States

### State Machine

```
                    ┌───────────┐
                    │  NORMAL   │
                    └─────┬─────┘
                          │ degradation detected
                          ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       DEGRADED MODES                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │   CLOUD     │  │   DATA      │  │   BROKER    │  │   TIME      │      │
│  │ UNREACHABLE │  │   INVALID   │  │   ERRORS    │  │   DRIFT     │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │                │              │
│         └────────────────┴────────────────┴────────────────┘              │
│                                   │                                        │
└───────────────────────────────────┼────────────────────────────────────────┘
                                    │ all cleared
                                    ▼
                              ┌───────────┐
                              │ RECOVERED │
                              └─────┬─────┘
                                    │ verified
                                    ▼
                              ┌───────────┐
                              │  NORMAL   │
                              └───────────┘
```

### Status Display

```bash
ccea-agent status --degraded

# Output:
# DEGRADED MODES:
#   ⚠️  CLOUD_UNREACHABLE (5 min)
#       - Last heartbeat: 5 minutes ago
#       - Telemetry buffer: 150 events
#       - Action: continue
#
#   ✅ DATA_FEED: OK
#   ✅ BROKER: OK
#   ✅ TIME_SYNC: OK
#   ✅ RESOURCES: OK
#
# TRADING STATUS: ACTIVE (degraded)
```

---

## Auto-Recovery

### Recovery Conditions

| Mode | Recovery Condition |
|------|-------------------|
| Cloud Unreachable | Successful heartbeat |
| Data Feed Invalid | Valid data for grace period |
| Broker Errors | No errors for window |
| Time Drift | Clock within tolerance |
| Resource Exhaustion | Resources below threshold |
| State Divergence | Manual acknowledgment only |

### Recovery Verification

Before exiting degraded mode:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RECOVERY VERIFICATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Mode: BROKER_ERRORS                                                        │
│                                                                              │
│  Checks:                                                                    │
│    1. ✅ API connectivity restored                                          │
│    2. ✅ Authentication valid                                               │
│    3. ✅ No errors for 60 seconds                                          │
│    4. ✅ Positions reconciled                                               │
│    5. ✅ Orders reconciled                                                  │
│                                                                              │
│  Result: RECOVERY APPROVED                                                  │
│  Resuming trading...                                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Telemetry Buffer

During Cloud unreachable, telemetry is buffered locally:

```yaml
telemetry:
  buffer:
    path: ~/.ccea/telemetry.db
    max_size_mb: 100
    max_events: 100000
    flush_batch_size: 100
    flush_interval_seconds: 60
```

**Buffer Management:**

```bash
# Check buffer status
ccea-agent telemetry buffer-status

# Output:
# Buffer: ~/.ccea/telemetry.db
# Events: 1,500
# Size: 5.2 MB / 100 MB
# Oldest: 2025-12-14T10:00:00Z
# Newest: 2025-12-14T12:00:00Z

# Manual flush (when connected)
ccea-agent telemetry flush
```

---

## Alerting

### Alert Configuration

```yaml
alerts:
  degraded_mode:
    channels:
      - type: log
        level: warning
      - type: email
        recipients: ["ops@example.com"]
        throttle_minutes: 60
      - type: webhook
        url: "https://alerts.example.com/webhook"

  escalation:
    - after_minutes: 5
      action: notify
    - after_minutes: 30
      action: escalate
    - after_minutes: 60
      action: page_oncall
```

### Alert Messages

```json
{
  "alert_type": "DEGRADED_MODE",
  "mode": "CLOUD_UNREACHABLE",
  "agent_id": "agent_xyz",
  "timestamp": "2025-12-14T12:00:00Z",
  "duration_minutes": 5,
  "action": "continue",
  "details": {
    "last_heartbeat": "2025-12-14T11:55:00Z",
    "retry_count": 10,
    "telemetry_buffer_size": 150
  }
}
```

---

## Manual Intervention

### Force Recovery

```bash
# Force exit degraded mode (USE WITH CAUTION)
ccea-agent degraded-mode clear --mode CLOUD_UNREACHABLE --force

# Acknowledge state divergence
ccea-agent degraded-mode acknowledge --mode STATE_DIVERGENCE

# Restart with clean state
ccea-agent restart --clear-state
```

### Manual Override

```bash
# Temporarily disable degraded mode checks (NOT RECOMMENDED)
ccea-agent start --skip-degraded-checks

# Override specific mode action
ccea-agent config set degraded_mode.cloud_unreachable.action halt
```

---

## Configuration Reference

```yaml
# Full degraded mode configuration
degraded_mode:
  # Cloud unreachable
  cloud_unreachable:
    action: continue
    max_duration_hours: 24
    alert_after_minutes: 5

  # Data feed invalid
  data_feed_invalid:
    action: halt
    detection:
      max_staleness_seconds: 30
      price_validation: true
      timestamp_validation: true
    grace_period_seconds: 30
    auto_recover: true

  # Broker errors
  broker_errors:
    action: pause
    error_threshold: 5
    window_seconds: 60
    retry:
      max_attempts: 3
      backoff_seconds: 5
    auto_recover: true

  # Time drift
  time_drift:
    action: halt
    max_drift_ms: 5000
    check_interval_seconds: 60

  # Resource exhaustion
  resource_exhaustion:
    action: pause
    thresholds:
      memory_percent: 90
      cpu_percent: 95
      disk_percent: 95

  # State divergence
  state_divergence:
    action: halt
    auto_recover: false

# Alerting
alerts:
  degraded_mode:
    enabled: true
    channels: [log, email]
```

---

## Troubleshooting

### Diagnosing Degraded Modes

```bash
# Check all degraded modes
ccea-agent degraded-mode status

# Check specific mode
ccea-agent degraded-mode check CLOUD_UNREACHABLE

# View degraded mode history
ccea-agent degraded-mode history --last 24h
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Stuck in degraded | Auto-recovery disabled | Enable or manual clear |
| False positives | Thresholds too sensitive | Adjust thresholds |
| No alerts | Alerting misconfigured | Check alert config |
| Recovery loops | Underlying issue persists | Fix root cause |

---

**Related Documentation:**

- [Risk Controls](./RISK_CONTROLS.md)
- [Kill Switch Runbook](../runbooks/KILL_SWITCH.md)
- [Recovery Procedures](../runbooks/RECOVERY.md)
