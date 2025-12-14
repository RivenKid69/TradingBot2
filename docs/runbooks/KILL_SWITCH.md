# Kill Switch Runbook

> **Severity**: Critical | **Last Updated**: 2025-12-14

## Overview

The kill switch immediately halts all trading activity when triggered. This runbook covers manual activation, automatic triggers, and recovery procedures.

---

## Trigger Conditions

Use the kill switch when:
- [ ] Unexpected trading behavior observed
- [ ] Excessive losses detected
- [ ] System malfunction suspected
- [ ] Broker API errors burst
- [ ] Data feed appears invalid
- [ ] Security incident in progress

---

## Pre-requisites

- [ ] CLI access to agent (`ccea-agent`)
- [ ] Agent credentials configured
- [ ] Understanding of current positions

---

## Procedure

### 1. Trigger Kill Switch

**Option A: CLI (Recommended)**
```bash
# Trigger kill switch
ccea-agent kill-switch trigger --reason "Manual trigger: <description>"

# With position flatten (if configured)
ccea-agent kill-switch trigger --reason "..." --flatten
```

**Option B: Agent GUI**
1. Open agent dashboard
2. Click "Emergency Stop" button
3. Confirm action

**Option C: API**
```bash
curl -X POST http://localhost:8080/api/v1/kill-switch/trigger \
  -H "Authorization: Bearer $AGENT_TOKEN" \
  -d '{"reason": "Manual trigger"}'
```

### 2. Verify Activation

```bash
# Check status
ccea-agent status

# Expected output:
# Agent State: HALTED
# Kill Switch: ACTIVE
# Reason: Manual trigger: <description>
# Triggered At: 2025-12-14T12:00:00Z
```

### 3. Review Open Orders

```bash
# List any remaining open orders
ccea-agent orders list --status open

# If orders remain, cancel manually
ccea-agent orders cancel-all
```

### 4. Review Positions

```bash
# List current positions
ccea-agent positions list

# Note: Positions are NOT automatically closed
# Manual closure required if needed
```

### 5. Document Incident

```bash
# Export incident data
ccea-agent kill-switch export-incident \
  --output incident_$(date +%Y%m%d_%H%M%S).json
```

---

## Recovery Procedure

### 1. Investigate Root Cause

Before recovering, investigate:
- [ ] What triggered the kill switch?
- [ ] Is the root cause resolved?
- [ ] Are broker APIs functioning?
- [ ] Is data feed valid?

### 2. Acknowledge Kill Switch

```bash
# Acknowledge and prepare for recovery
ccea-agent kill-switch acknowledge

# Provide reason
ccea-agent kill-switch acknowledge \
  --reason "Root cause identified and resolved"
```

### 3. Run Pre-flight Checks

```bash
# Full pre-flight validation
ccea-agent preflight --verbose

# All checks must pass:
# ✅ Time sync OK
# ✅ Broker connectivity OK
# ✅ Credentials valid
# ✅ Data feed OK
# ✅ Policy config OK
```

### 4. Reconcile State

```bash
# Reconcile positions and orders
ccea-agent reconcile all

# Verify clean state
ccea-agent reconcile verify
```

### 5. Resume Trading (Optional)

```bash
# Resume with reduced risk (recommended)
ccea-agent start --reduced-risk

# Or full resume (after thorough verification)
ccea-agent start
```

---

## Automatic Triggers

| Trigger | Threshold | Auto-Recovery |
|---------|-----------|---------------|
| Max daily loss | 2% | No |
| Broker error burst | 10 in 60s | No |
| Latency spike | >5000ms | No |
| Order spam | 100 in 60s | No |
| State divergence | Any | No |
| Data feed invalid | >30s | No |

---

## Verification Checklist

After recovery:
- [ ] Kill switch status cleared
- [ ] Agent state is IDLE or RUNNING
- [ ] No orphan orders
- [ ] Positions reconciled
- [ ] Telemetry flowing
- [ ] Logs captured

---

## Escalation

If kill switch cannot be cleared:
1. Contact platform support
2. Provide incident export
3. Await guidance before manual override

---

## Post-Incident

1. **Document** the incident in incident tracker
2. **Review** what triggered the kill switch
3. **Adjust** thresholds if false positive
4. **Update** runbook if needed
5. **Notify** stakeholders

---

## Related

- [Recovery Procedures](./RECOVERY.md)
- [Degraded Mode Handling](./DEGRADED_MODE.md)
- [Risk Controls](../agent/RISK_CONTROLS.md)
