# Recovery Procedures Runbook

> **Severity**: High | **Last Updated**: 2025-12-16

## Overview

This runbook covers recovery procedures for various failure scenarios.

---

## Scenario 1: Agent Crash Recovery

### Symptoms

- Agent process not running
- No heartbeats to Cloud
- Positions/orders in unknown state

### Procedure

1. **Check Agent Status**

```bash
# Check if running
ccea-agent status
# or
ps aux | grep ccea-agent
```

2. **Review Crash Logs**

```bash
# View recent logs
ccea-agent logs --tail 100 --level ERROR

# Check system journal (Linux)
journalctl -u ccea-agent --since "1 hour ago"
```

3. **Restart Agent**

```bash
# Restart with automatic reconciliation
ccea-agent start --reconcile

# Agent will:
# 1. Load local journal
# 2. Fetch broker state
# 3. Reconcile differences
# 4. Resume if clean
```

4. **Verify Recovery**

```bash
# Check reconciliation
ccea-agent reconcile verify

# Check positions
ccea-agent positions list

# Check orders
ccea-agent orders list
```

---

## Scenario 2: Database Corruption Recovery

### Symptoms

- Agent fails to start
- SQLite errors in logs
- Telemetry buffer errors

### Procedure

1. **Stop Agent**

```bash
ccea-agent stop
```

2. **Backup Corrupted Files**

```bash
cp ~/.ccea/telemetry.db ~/.ccea/telemetry.db.corrupted
cp ~/.ccea/journal/*.db ~/.ccea/journal_backup/
```

3. **Attempt Repair**

```bash
# SQLite repair
sqlite3 ~/.ccea/telemetry.db "PRAGMA integrity_check;"
sqlite3 ~/.ccea/telemetry.db ".recover" | sqlite3 ~/.ccea/telemetry_recovered.db
```

4. **Reset If Needed**

```bash
# Reset telemetry buffer (data loss)
ccea-agent telemetry reset-buffer --confirm

# Reset journal (requires reconciliation)
ccea-agent journal reset --confirm
```

5. **Restart with Full Reconciliation**

```bash
ccea-agent start --full-reconcile
```

---

## Scenario 3: Vault Recovery

### Symptoms

- Cannot unlock vault
- Credential errors
- Broker connection failures

### Procedure

1. **Check Vault Status**

```bash
ccea-agent vault status
```

2. **Verify Encryption Key**

```bash
# Check if key is set
echo $CCEA_VAULT_KEY | wc -c
# Should be 44+ characters (base64)
```

3. **If Key Lost - Re-add Credentials**

```bash
# Reset vault (credentials will be lost)
ccea-agent vault reset --confirm

# Re-add broker credentials
ccea-agent vault add-broker --broker binance
```

4. **If Vault Corrupted**

```bash
# Backup
cp ~/.ccea/vault.enc ~/.ccea/vault.enc.corrupted

# Reset and re-add
ccea-agent vault reset --confirm
ccea-agent vault add-broker --broker binance
```

---

## Scenario 4: Cloud Connection Recovery

### Symptoms

- Heartbeat failures
- Command poll timeouts
- "Cloud unreachable" warnings

### Procedure

1. **Check Network**

```bash
# Test Cloud connectivity
curl -v https://api.ccea.cloud/health

# Check DNS
nslookup api.ccea.cloud
```

2. **Verify Agent Config**

```bash
ccea-agent config show cloud
# Check endpoint is correct
```

3. **Check TLS Certificates**

```bash
openssl s_client -connect api.ccea.cloud:443 -servername api.ccea.cloud
```

4. **Force Reconnect**

```bash
ccea-agent cloud reconnect
```

5. **If Enrollment Lost**

```bash
# Get new enrollment token from Cloud UI
ccea-agent enroll --token <new_token>
```

---

## Scenario 5: State Divergence Recovery

### Symptoms

- Position mismatch alerts
- Unknown orders detected
- Kill switch triggered by divergence

### Procedure

1. **Stop Trading**

```bash
ccea-agent stop
```

2. **Compare States**

```bash
# Get local state
ccea-agent positions list --source local
ccea-agent orders list --source local

# Get broker state
ccea-agent positions list --source broker
ccea-agent orders list --source broker
```

3. **Identify Differences**

```bash
# Detailed comparison
ccea-agent reconcile diff --verbose
```

4. **Resolve Manually**

```bash
# Option A: Trust broker state
ccea-agent reconcile resolve --trust broker

# Option B: Trust local state (rare)
ccea-agent reconcile resolve --trust local

# Option C: Manual resolution
# - Cancel unknown orders at broker
# - Update local journal
```

5. **Verify Resolution**

```bash
ccea-agent reconcile verify
```

6. **Resume**

```bash
ccea-agent start
```

---

## Scenario 6: Time Sync Recovery

### Symptoms

- Time drift alerts
- Timestamp validation failures
- Kill switch triggered by time

### Procedure

1. **Check System Time**

```bash
date
timedatectl status
```

2. **Force NTP Sync**

```bash
# Linux
sudo systemctl restart systemd-timesyncd
# or
sudo ntpdate -u time.google.com

# macOS
sudo sntp -sS time.apple.com
```

3. **Verify Agent Time Sync**

```bash
ccea-agent doctor --check time
```

4. **Resume**

```bash
ccea-agent kill-switch acknowledge
ccea-agent start
```

---

## General Recovery Checklist

After any recovery:

- [ ] Agent status is healthy
- [ ] Positions are reconciled
- [ ] Orders are reconciled
- [ ] Vault is accessible
- [ ] Cloud connection is active
- [ ] Time is synchronized
- [ ] Telemetry is flowing
- [ ] No pending approvals blocked
- [ ] Logs reviewed for errors
- [ ] Root cause documented

---

## Related

- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Agent Revocation](./AGENT_REVOCATION.md)
- [Degraded Modes](../agent/DEGRADED_MODES.md)
