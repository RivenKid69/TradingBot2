# Broker Errors Runbook

> **Severity**: High | **Last Updated**: 2025-12-15

## Overview

This runbook covers handling of broker connectivity issues, API errors, and execution failures. All broker interactions occur in the **Agent zone only** - Cloud has no broker access.

---

## Common Error Types

| Error Type | Typical Cause | Auto-Recovery | Runbook Section |
|------------|---------------|---------------|-----------------|
| Connection timeout | Network/broker down | Retry with backoff | Section 3.1 |
| Authentication failure | Expired/invalid keys | No | Section 3.2 |
| Rate limiting | Too many requests | Auto-backoff | Section 3.3 |
| Order rejection | Insufficient funds/invalid params | No | Section 3.4 |
| Partial fill | Low liquidity | Handled by strategy | Section 3.5 |
| Market closed | Outside trading hours | No | Section 3.6 |

---

## Prerequisites

- [ ] CLI access to agent (`ccea-agent`)
- [ ] Agent credentials configured
- [ ] Broker status page bookmarked
- [ ] Understanding of current positions

---

## 1. Quick Diagnosis

```bash
# Check overall agent health
ccea-agent doctor

# Check broker connectivity specifically
ccea-agent broker status

# View recent broker errors
ccea-agent logs --source broker --level ERROR --last 30m
```

### Expected Healthy Output

```
Broker Status: CONNECTED
  Connection: Active
  Latency: 45ms (avg)
  Last Heartbeat: 2s ago
  Rate Limit: 85% available
  Market Status: OPEN
```

---

## 2. Error Classification

### 2.1 Classify by Impact

| Impact Level | Criteria | Immediate Action |
|--------------|----------|------------------|
| **Critical** | All orders failing, positions unknown | Kill switch |
| **High** | >50% orders failing | Pause trading, investigate |
| **Medium** | Intermittent failures (<10%) | Monitor, alert |
| **Low** | Single order rejection | Log and continue |

### 2.2 Classify by Recoverability

```bash
# Check if error is transient or persistent
ccea-agent broker diagnose

# Output shows:
# - Error pattern (transient/persistent)
# - Affected endpoints
# - Recommended action
```

---

## 3. Error-Specific Procedures

### 3.1 Connection Timeout

**Symptoms:**

- Orders timing out
- Heartbeat failures
- Telemetry shows broker disconnected

**Procedure:**

```bash
# 1. Verify network connectivity
ccea-agent broker ping

# 2. Check broker status page (external)
# https://status.broker.com (check manually)

# 3. If network OK but broker down:
ccea-agent pause --reason "Broker connectivity issues"

# 4. Monitor for recovery
ccea-agent status --watch

# 5. When recovered, reconcile and resume
ccea-agent reconcile all
ccea-agent start
```

**Automatic Handling:**

- Agent retries with exponential backoff (1s, 2s, 4s, 8s, max 30s)
- After 10 consecutive failures: auto-pause
- After 30s disconnect: kill switch consideration

### 3.2 Authentication Failure

**Symptoms:**

- 401/403 responses from broker
- "Invalid API key" or "Signature mismatch" errors

**Procedure:**

```bash
# 1. Stop trading immediately
ccea-agent stop --immediate

# 2. Check credential validity
ccea-agent broker auth verify

# 3. If expired, rotate credentials
ccea-agent vault credentials rotate --broker binance

# 4. Re-verify
ccea-agent broker auth verify

# 5. If still failing, check at broker:
# - Is IP whitelisted?
# - Is API key active?
# - Are permissions correct (trade-only, no withdraw)?

# 6. Run preflight before resuming
ccea-agent preflight
ccea-agent start
```

**IMPORTANT:**

- Never share API keys
- Cloud does not have access to broker credentials
- Rotation is a local-only operation

### 3.3 Rate Limiting

**Symptoms:**

- 429 (Too Many Requests) responses
- Increasing latency
- Orders queued

**Procedure:**

```bash
# 1. Check current rate limit status
ccea-agent broker rate-limit status

# 2. If near limit, reduce frequency
ccea-agent config set trading.order_rate_limit 10/minute

# 3. For immediate relief
ccea-agent pause --reason "Rate limit approaching"
sleep 60
ccea-agent start

# 4. Long-term: adjust strategy order frequency
```

**Automatic Handling:**

- Agent tracks rate limits
- Auto-queues orders when near limit
- Backpressure applied to strategy

### 3.4 Order Rejection

**Symptoms:**

- Order returns rejected status
- Specific error codes (e.g., insufficient balance, invalid symbol)

**Procedure:**

```bash
# 1. Check rejection reason
ccea-agent orders list --status rejected --last 1h

# 2. For insufficient funds:
ccea-agent positions list
ccea-agent broker balance

# 3. For invalid parameters:
ccea-agent logs --source execution --level ERROR --last 10m
# Review order parameters in logs

# 4. If systematic, pause and investigate
ccea-agent pause --reason "Order rejections"
```

**Common Rejection Reasons:**

| Code | Meaning | Resolution |
|------|---------|------------|
| `INSUFFICIENT_BALANCE` | Not enough funds | Reduce position size |
| `INVALID_SYMBOL` | Symbol not tradable | Update universe |
| `MIN_NOTIONAL` | Order too small | Increase size or skip |
| `MARKET_CLOSED` | Outside trading hours | Check schedule |
| `PRICE_FILTER` | Price outside bounds | Adjust limit price |

### 3.5 Partial Fills

**Symptoms:**

- Order only partially executed
- Remaining quantity open

**Procedure:**

```bash
# 1. Check partial fill status
ccea-agent orders list --status partial

# 2. View fill details
ccea-agent orders show <order_id>

# 3. Decision: wait, cancel remainder, or IOC
# Wait for fill:
ccea-agent orders wait <order_id> --timeout 5m

# Cancel remainder:
ccea-agent orders cancel <order_id>

# 4. Strategy handles partials automatically
# Ensure strategy config is correct:
ccea-agent config get strategy.partial_fill_handling
```

### 3.6 Market Closed

**Symptoms:**

- Orders rejected with "market closed"
- Strategy attempting to trade outside hours

**Procedure:**

```bash
# 1. Verify market status
ccea-agent broker market-status

# 2. Check agent schedule
ccea-agent schedule show

# 3. If schedule incorrect, update:
ccea-agent schedule set trading_hours --market US_EQUITY --hours "09:30-16:00" --tz "America/New_York"

# 4. Agent should auto-pause outside hours
ccea-agent config get agent.respect_market_hours
```

---

## 4. Kill Switch Consideration

### Trigger Kill Switch If

- [ ] Unable to verify position state
- [ ] Broker returning inconsistent data
- [ ] Multiple critical errors in sequence
- [ ] Authentication cannot be restored quickly
- [ ] Suspected security breach

```bash
# Trigger kill switch
ccea-agent kill-switch trigger --reason "Broker errors: <specific reason>"

# Follow KILL_SWITCH.md for recovery
```

---

## 5. Recovery After Broker Issues

### 5.1 Pre-Resume Checklist

- [ ] Broker connectivity verified
- [ ] Credentials valid
- [ ] Rate limits reset
- [ ] Market is open
- [ ] Positions reconciled

### 5.2 Reconciliation

```bash
# Full reconciliation
ccea-agent reconcile all

# Verify no orphan orders
ccea-agent reconcile orders

# Verify position state
ccea-agent reconcile positions

# Verify clean state
ccea-agent reconcile verify
```

### 5.3 Resume Trading

```bash
# Resume with caution
ccea-agent start --reduced-risk

# Monitor closely
ccea-agent status --watch

# After stability confirmed, full resume
ccea-agent start
```

---

## 6. Prevention

### 6.1 Monitoring Setup

```bash
# Enable broker health alerts
ccea-agent alerts enable broker.connectivity --threshold 3
ccea-agent alerts enable broker.latency --threshold 1000ms
ccea-agent alerts enable broker.rate_limit --threshold 80%
```

### 6.2 Configuration Best Practices

```yaml
# agent-config.yaml
broker:
  connection_timeout: 30s
  request_timeout: 10s
  max_retries: 3
  retry_backoff: exponential
  health_check_interval: 10s

rate_limits:
  orders_per_minute: 10
  requests_per_second: 5
  backpressure_threshold: 0.8
```

---

## 7. Escalation

If broker issues persist:

1. **Check broker status page** for known outages
2. **Contact broker support** with:
   - Account ID (not credentials)
   - Error messages
   - Timestamps
3. **Document incident** for post-mortem

---

## 8. Related

- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Recovery Procedures](./RECOVERY.md)
- [Key Rotation](./KEY_ROTATION_RUNBOOK.md)
- [Incident Response](./INCIDENT_RESPONSE.md)
