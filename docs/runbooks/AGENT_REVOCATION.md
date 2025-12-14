# Agent Revocation & Key Rotation Runbook

> **Severity**: High | **Last Updated**: 2025-12-14

## Overview

This runbook covers procedures for revoking agent trust and rotating credentials when security is compromised or for routine rotation.

---

## When to Revoke

- [ ] Agent machine compromised
- [ ] Unauthorized access detected
- [ ] Employee departure (with agent access)
- [ ] Security incident investigation
- [ ] Routine key rotation schedule

---

## Pre-requisites

- [ ] Cloud admin access
- [ ] Agent CLI access (if agent reachable)
- [ ] Broker account access (to revoke API keys)

---

## Emergency Revocation Procedure

### 1. Revoke Agent Trust (Cloud)

**Via Cloud UI:**
1. Navigate to Agents
2. Select agent
3. Click "Revoke Trust"
4. Confirm action

**Via CLI:**
```bash
ccea-admin agents revoke --agent-id agent_xyz --reason "security_incident"
```

**Via API:**
```bash
curl -X POST https://api.ccea.cloud/v1/agents/agent_xyz/revoke \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "security_incident"}'
```

### 2. Revoke Broker API Keys

**CRITICAL: Do this at the broker, not just in the agent**

**Binance:**
1. Log in to Binance
2. API Management → Delete API key
3. Create new API key

**Alpaca:**
1. Log in to Alpaca dashboard
2. API Keys → Regenerate

### 3. Stop Agent (If Accessible)

```bash
# If agent is reachable
ccea-agent stop --immediate

# Force kill if necessary
pkill -9 ccea-agent
```

### 4. Document Revocation

```bash
# Export revocation record
ccea-admin audit export-revocation \
  --agent-id agent_xyz \
  --output revocation_$(date +%Y%m%d).json
```

---

## Key Rotation Procedure (Routine)

### 1. Rotate Agent Session Key

```bash
# On agent
ccea-agent keys rotate-session

# This will:
# 1. Generate new session key
# 2. Register with Cloud
# 3. Invalidate old key
```

### 2. Rotate Vault Encryption Key

```bash
# Generate new key
openssl rand -base64 32 > ~/.ccea/vault_new.key

# Rotate vault encryption
ccea-agent vault rotate-key \
  --new-key-file ~/.ccea/vault_new.key

# Update environment
export CCEA_VAULT_KEY=$(cat ~/.ccea/vault_new.key)

# Secure old key (for recovery period)
mv ~/.ccea/vault.key ~/.ccea/vault_old.key
mv ~/.ccea/vault_new.key ~/.ccea/vault.key
```

### 3. Rotate Broker API Keys

```bash
# Remove old credentials
ccea-agent vault remove --label "main-trading"

# Add new credentials
ccea-agent vault add-broker \
  --broker binance \
  --api-key $NEW_API_KEY \
  --api-secret $NEW_API_SECRET \
  --label "main-trading"
```

### 4. Verify Rotation

```bash
# Test broker connectivity
ccea-agent doctor --check broker

# Test vault access
ccea-agent vault list
```

---

## Re-enrollment Procedure

After revocation, to re-enroll agent:

### 1. Generate New Enrollment Token

**Via Cloud UI:**
1. Navigate to Agents → Add Agent
2. Generate enrollment token
3. Copy token

### 2. Reset Agent State

```bash
# Remove old enrollment
ccea-agent disenroll

# Clear local state if needed
ccea-agent reset --confirm
```

### 3. Enroll with New Token

```bash
ccea-agent enroll --token <new_token>
```

### 4. Reconfigure Agent

```bash
# Add broker credentials
ccea-agent vault add-broker --broker binance

# Set hard caps
ccea-agent policy set-hard-caps --max-position-pct 10

# Run preflight
ccea-agent preflight
```

### 5. Request New Deployment

Via Cloud UI:
1. Navigate to Deployments
2. Create new deployment for new agent
3. Agent will receive approval request

---

## Verification Checklist

After revocation:
- [ ] Agent trust state is REVOKED in Cloud
- [ ] Old broker API keys are deleted at broker
- [ ] Agent cannot connect to Cloud
- [ ] No open orders remain
- [ ] Positions are documented

After re-enrollment:
- [ ] New agent ID assigned
- [ ] New session keys active
- [ ] New broker API keys configured
- [ ] Preflight checks pass
- [ ] Test deployment successful

---

## Audit Trail

All revocations are logged:

```bash
# View revocation history
ccea-admin audit query \
  --event-type "agent.revoked" \
  --from "2025-01-01"
```

---

## Rotation Schedule (Recommended)

| Item | Frequency | Procedure |
|------|-----------|-----------|
| Agent session key | 90 days | Auto-rotate |
| Vault encryption key | 180 days | Manual rotate |
| Broker API keys | 90 days | Manual rotate |
| Agent enrollment | As needed | Re-enroll |

---

## Related

- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Recovery Procedures](./RECOVERY.md)
- [Local Vault](../agent/LOCAL_VAULT.md)
