# Data Loss and Recovery Runbook

> **Severity**: Critical | **Last Updated**: 2025-12-15

## Overview

This runbook covers data loss scenarios and recovery procedures for the CCEA architecture. Critical principle: **trading-critical data (positions, credentials) lives in Agent zone only**.

---

## Data Classification

| Data Type | Location | Backup Strategy | Recovery SLA |
|-----------|----------|-----------------|--------------|
| Broker credentials | Agent vault | Local backup | Immediate (if backed up) |
| Position state | Agent + Broker | Broker is source of truth | Minutes |
| Order history | Agent local | Periodic export | Hours |
| Strategy source | Cloud | Git + DB backup | Minutes |
| Telemetry | Cloud | DB backup | Hours-Days |
| Config blobs | Cloud | Immutable + replicated | Minutes |
| Approval records | Cloud | Immutable audit log | N/A (immutable) |

---

## 1. Quick Assessment

```bash
# Check agent state integrity
ccea-agent doctor --check data

# Check cloud connectivity
ccea-agent cloud status

# Verify credential vault
ccea-agent vault verify

# Check position reconciliation
ccea-agent reconcile positions --dry-run
```

---

## 2. Agent-Side Data Loss

### 2.1 Credential Vault Corruption/Loss

**Symptoms:**
- Agent cannot authenticate to broker
- Vault file missing or corrupted
- "Vault integrity check failed" error

**Procedure:**

```bash
# 1. STOP TRADING IMMEDIATELY
ccea-agent stop --immediate

# 2. Check vault status
ccea-agent vault status

# 3. If corrupted, attempt recovery from backup
ccea-agent vault restore --backup /path/to/vault.backup

# 4. If no backup, must re-import credentials
ccea-agent vault init --force
ccea-agent vault import broker --name binance

# 5. Verify credentials
ccea-agent broker auth verify

# 6. Run preflight before resuming
ccea-agent preflight
```

**CRITICAL:**
- Credentials are NEVER stored in Cloud
- If vault is lost and no backup exists, you must obtain new API keys from broker
- Consider rotating keys after any vault compromise

### 2.2 Position State Loss

**Symptoms:**
- Agent shows no positions
- Position mismatch with broker
- "State file corrupted" error

**Procedure:**

```bash
# 1. Pause trading
ccea-agent pause --reason "Position state recovery"

# 2. Query broker for actual positions
ccea-agent broker positions list

# 3. Force reconciliation from broker
ccea-agent reconcile positions --from-broker --force

# 4. Verify reconciliation
ccea-agent positions list
ccea-agent reconcile verify

# 5. If mismatch persists:
# - Manual verification required
# - Compare with broker UI/dashboard

# 6. Resume after verification
ccea-agent start
```

**Broker is Source of Truth:**
- Agent state can always be rebuilt from broker
- Never trust agent state over broker state
- Always reconcile after any state issue

### 2.3 Order History Loss

**Symptoms:**
- Recent orders missing from local history
- "Order not found" for known order IDs
- Incomplete audit trail

**Procedure:**

```bash
# 1. Check local order database
ccea-agent orders list --all --from "24 hours ago"

# 2. Query broker for order history
ccea-agent broker orders history --days 7

# 3. Import missing orders from broker
ccea-agent orders import --from-broker --days 7

# 4. Verify reconciliation
ccea-agent orders reconcile

# 5. Export for future backup
ccea-agent orders export --days 30 --output orders_backup.json
```

### 2.4 Local Configuration Loss

**Symptoms:**
- Agent using default configuration
- Risk limits reset to defaults
- Strategy parameters missing

**Procedure:**

```bash
# 1. Check current configuration
ccea-agent config show

# 2. If Cloud config exists, pull from Cloud
ccea-agent config pull --from-cloud

# 3. If no Cloud config, restore from backup
ccea-agent config restore --backup /path/to/config.backup

# 4. If no backup, reconfigure manually
ccea-agent config set-all --interactive

# 5. Verify critical settings
ccea-agent config verify

# CRITICAL: Always verify risk limits are set
ccea-agent config get risk.max_daily_loss
ccea-agent config get risk.max_position_size
```

---

## 3. Cloud-Side Data Loss

### 3.1 Strategy Source Lost

**Symptoms:**
- Strategy missing from Cloud dashboard
- Build failures (source not found)
- "Strategy not found" errors

**Procedure:**

```bash
# 1. Check local copies
ls -la ~/strategies/

# 2. Check Git history (if using Git)
git log --oneline -- path/to/strategy.py

# 3. Re-upload from local
ccea-cli strategy upload --path ~/strategies/my_strategy.py

# 4. If no local copy, contact support
# Cloud maintains soft-delete for 30 days

# 5. For enterprise, request recovery from backup
ccea-cli support request --type data-recovery --resource strategy
```

### 3.2 Telemetry Data Loss

**Symptoms:**
- Gaps in monitoring dashboard
- Missing historical metrics
- Analytics reports incomplete

**Procedure:**

```bash
# 1. Check telemetry status
ccea-cli telemetry status --workspace <workspace-id>

# 2. Identify gap period
ccea-cli telemetry gaps --days 30

# 3. If Agent has local buffer, re-upload
ccea-agent telemetry upload-buffer

# 4. For Enterprise, request recovery
ccea-cli support request --type data-recovery --resource telemetry

# Note: Some telemetry may be unrecoverable if not buffered
```

### 3.3 Approval Records

**Approval records are IMMUTABLE and should never be lost.**

If missing:
1. Check audit log (separate from approval records)
2. Records are replicated across zones
3. Enterprise has extended retention
4. Contact support for recovery

```bash
# Check audit log for approval events
ccea-cli audit query --type approval --days 30

# Request recovery
ccea-cli support request --type data-recovery --resource approvals
```

---

## 4. Full Disaster Recovery

### 4.1 Agent Complete Reinstall

If Agent system is completely lost:

```bash
# 1. Install Agent on new system
curl -sSL https://get.ccea.io/agent | bash

# 2. Configure with Cloud credentials
ccea-agent init --cloud-token <token>

# 3. Import configuration from Cloud
ccea-agent config pull --from-cloud

# 4. Re-import broker credentials (manually)
ccea-agent vault init
ccea-agent vault import broker --name binance
# (requires new API keys from broker if backup unavailable)

# 5. Reconcile state from broker
ccea-agent reconcile all --from-broker

# 6. Verify everything
ccea-agent preflight
ccea-agent doctor

# 7. Start trading
ccea-agent start
```

### 4.2 Cloud Complete Outage

If Cloud is completely unavailable:

```bash
# 1. Agent continues operating in degraded mode
ccea-agent status
# Shows: Cloud Connection: DISCONNECTED

# 2. Trading continues based on local strategy
# (Agent is autonomous for execution)

# 3. Local telemetry buffered
ccea-agent telemetry buffer status

# 4. When Cloud recovers:
ccea-agent cloud reconnect
ccea-agent telemetry upload-buffer
ccea-agent reconcile all
```

**Key Principle:** Agent can trade without Cloud. Cloud loss is a monitoring/management issue, not a trading issue.

---

## 5. Backup Procedures

### 5.1 Agent Backups

```bash
# Create full agent backup
ccea-agent backup create --output /backups/agent_$(date +%Y%m%d).tar.gz

# Contents:
# - Vault (encrypted)
# - Configuration
# - Local state
# - Order history

# Schedule daily backups (crontab)
0 2 * * * /usr/local/bin/ccea-agent backup create --output /backups/agent_$(date +\%Y\%m\%d).tar.gz
```

### 5.2 Credential Vault Backup

```bash
# Export encrypted vault backup
ccea-agent vault backup --output /secure/vault_backup.enc

# Store securely:
# - Encrypted USB drive
# - Hardware security module
# - Password manager
# - NOT in cloud storage
```

### 5.3 Configuration Export

```bash
# Export configuration (no secrets)
ccea-agent config export --output config_backup.yaml

# This can be stored in Cloud or Git
```

---

## 6. Prevention

### 6.1 Automated Backups

```yaml
# agent-config.yaml
backup:
  enabled: true
  schedule: "0 2 * * *"  # Daily at 2 AM
  retention_days: 30
  destination: /backups/
  encrypt: true

vault_backup:
  enabled: true
  schedule: "0 * * * *"  # Hourly
  retention_count: 24
  destination: /secure/vault/
```

### 6.2 Redundancy

| Component | Redundancy Strategy |
|-----------|-------------------|
| Agent vault | Hourly backup, offline copy |
| Agent config | Cloud sync + local backup |
| Position state | Broker is source of truth |
| Cloud data | Multi-region replication |
| Audit logs | Immutable, replicated |

### 6.3 Testing

```bash
# Test backup restoration monthly
ccea-agent backup test-restore --backup /backups/agent_latest.tar.gz

# Test vault recovery quarterly
ccea-agent vault test-recovery --backup /secure/vault_backup.enc
```

---

## 7. Escalation

### Support Contact

For data recovery assistance:

- **Standard**: [support email]
- **Enterprise**: [enterprise support email] (if applicable)
- **Security incident**: [security contact email]

### Information to Provide

1. Workspace ID
2. Agent ID
3. Timeline of data loss
4. Data types affected
5. Recovery attempts made
6. Backup availability

---

## 8. Related

- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Recovery Procedures](./RECOVERY.md)
- [Agent Docs](../agent/README.md)
- [Degraded Mode](./DEGRADED_MODE.md)
