# Operational Runbooks

> **Version**: 1.1.0 | **Last Updated**: 2025-12-16

## Overview

This directory contains operational runbooks for incident response, recovery procedures, and maintenance tasks.

## Runbook Index

| Runbook | Purpose | Severity |
|---------|---------|----------|
| [KILL_SWITCH.md](./KILL_SWITCH.md) | Kill switch activation and recovery | Critical |
| [RECOVERY.md](./RECOVERY.md) | System recovery procedures | High |
| [AGENT_REVOCATION.md](./AGENT_REVOCATION.md) | Agent revocation and key rotation | High |
| [DEGRADED_MODE.md](./DEGRADED_MODE.md) | Handling degraded operations | Medium |
| [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md) | General incident response | Varies |
| [BROKER_ERRORS.md](./BROKER_ERRORS.md) | Broker connectivity and API errors | High |
| [LATENCY_ISSUES.md](./LATENCY_ISSUES.md) | Latency diagnosis and resolution | Medium-High |
| [DATA_LOSS.md](./DATA_LOSS.md) | Data loss and recovery procedures | Critical |
| [KEY_ROTATION_RUNBOOK.md](./KEY_ROTATION_RUNBOOK.md) | Key rotation procedures | High |

## Quick Reference

### Emergency Actions

| Situation | Action | Runbook |
|-----------|--------|---------|
| Unexpected trading behavior | Trigger kill switch | [KILL_SWITCH.md](./KILL_SWITCH.md) |
| Agent compromise suspected | Revoke agent | [AGENT_REVOCATION.md](./AGENT_REVOCATION.md) |
| Cloud unreachable | Follow degraded mode | [DEGRADED_MODE.md](./DEGRADED_MODE.md) |
| Data breach suspected | Incident response | [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md) |
| System failure | Recovery procedure | [RECOVERY.md](./RECOVERY.md) |
| Broker API errors | Diagnose and resolve | [BROKER_ERRORS.md](./BROKER_ERRORS.md) |
| High latency detected | Investigate source | [LATENCY_ISSUES.md](./LATENCY_ISSUES.md) |
| Data loss suspected | Recovery procedure | [DATA_LOSS.md](./DATA_LOSS.md) |

### Emergency Contacts

| Role | Contact | Escalation |
|------|---------|------------|
| On-call Engineer | ops@example.com | Immediate |
| Security Team | security@example.com | Security incidents |
| Platform Lead | platform@example.com | Major outages |

---

## Runbook Format

Each runbook follows this structure:

```markdown
# Runbook Title

## Trigger Conditions
When to use this runbook

## Pre-requisites
Required access and tools

## Procedure
Step-by-step instructions

## Verification
How to verify success

## Rollback
How to undo if needed

## Post-Incident
Follow-up actions
```

---

## Related Documentation

- [Agent Degraded Modes](../agent/DEGRADED_MODES.md)
- [Risk Controls](../agent/RISK_CONTROLS.md)
- [CCEA Overview](../CCEA_OVERVIEW.md)
