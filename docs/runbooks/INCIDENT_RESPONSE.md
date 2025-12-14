# Incident Response Runbook

> **Severity**: Varies | **Last Updated**: 2025-12-14

## Overview

This runbook provides a general framework for responding to security incidents, system failures, and operational issues.

---

## Incident Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P1 - Critical** | Trading halted, data breach, security compromise | Immediate | Kill switch, credential theft, system down |
| **P2 - High** | Degraded operation, significant loss | 15 min | Data feed issues, broker errors, state divergence |
| **P3 - Medium** | Partial degradation, minor issues | 1 hour | Performance degradation, non-critical errors |
| **P4 - Low** | Informational, no immediate impact | 4 hours | Warnings, threshold alerts |

---

## Incident Response Process

### Phase 1: Detection & Triage (0-5 min)

1. **Identify the incident**
   - Check alerts
   - Review logs
   - Verify symptoms

2. **Classify severity**
   - P1: Engage immediately
   - P2: Begin response
   - P3/P4: Schedule response

3. **Initial assessment**
   ```bash
   # Quick status check
   ccea-agent status
   ccea-agent doctor
   ```

### Phase 2: Containment (5-15 min)

1. **Stop the bleeding**
   - P1: Trigger kill switch
   - P2: Pause trading
   - P3/P4: Monitor

2. **Isolate affected systems**
   ```bash
   # If security incident
   ccea-agent stop --immediate

   # Revoke if compromised
   ccea-admin agents revoke --agent-id agent_xyz
   ```

3. **Preserve evidence**
   ```bash
   # Export logs
   ccea-agent logs export --output incident_logs_$(date +%Y%m%d_%H%M%S).tar.gz

   # Export state
   ccea-agent state export --output incident_state_$(date +%Y%m%d_%H%M%S).json
   ```

### Phase 3: Investigation (15 min - 2 hours)

1. **Gather information**
   ```bash
   # Recent audit logs
   ccea-admin audit query --from "1 hour ago"

   # Error logs
   ccea-agent logs --level ERROR --from "1 hour ago"

   # Telemetry
   ccea-agent telemetry export --from "1 hour ago"
   ```

2. **Identify root cause**
   - Review timeline
   - Check for anomalies
   - Correlate events

3. **Document findings**
   - Timeline of events
   - Affected systems
   - Root cause (if identified)

### Phase 4: Resolution (Varies)

1. **Implement fix**
   - Follow specific runbook for issue type
   - Apply patches/configuration changes

2. **Verify resolution**
   ```bash
   # Run diagnostics
   ccea-agent doctor --verbose

   # Reconcile state
   ccea-agent reconcile all

   # Test connectivity
   ccea-agent preflight
   ```

3. **Restore service**
   ```bash
   # Resume operation
   ccea-agent start

   # Monitor closely
   ccea-agent status --watch
   ```

### Phase 5: Post-Incident (1-7 days)

1. **Post-mortem**
   - What happened?
   - Why did it happen?
   - How was it detected?
   - How was it resolved?
   - What can prevent recurrence?

2. **Action items**
   - Update runbooks
   - Improve monitoring
   - Fix root causes
   - Update documentation

3. **Communication**
   - Internal summary
   - Customer notification (if required)
   - Regulatory notification (if required)

---

## Security Incident Specifics

### Suspected Credential Theft

1. **Immediate actions**
   ```bash
   # Revoke all agents
   ccea-admin agents revoke --all --reason "security_incident"

   # At broker: Delete API keys immediately
   ```

2. **Investigation**
   - How were credentials accessed?
   - What actions were taken?
   - Were orders placed?

3. **Recovery**
   - Generate new API keys at broker
   - Re-enroll agents with new keys
   - Enhanced monitoring

### Unauthorized Access Detected

1. **Containment**
   ```bash
   # Disable affected user
   ccea-admin users disable --user-id user_xyz

   # Revoke sessions
   ccea-admin sessions revoke --user-id user_xyz
   ```

2. **Investigation**
   - Review access logs
   - Check for data exfiltration
   - Identify entry point

3. **Recovery**
   - Reset passwords
   - Review permissions
   - Enable MFA if not enabled

### Data Breach Suspected

1. **Containment**
   - Isolate affected systems
   - Preserve logs
   - Engage security team

2. **Assessment**
   - What data was accessed?
   - Was PII involved?
   - Was data exfiltrated?

3. **Notification (if required)**
   - GDPR: 72 hours to DPA
   - Affected users: Without undue delay
   - Document everything

---

## Communication Templates

### Internal Notification

```
INCIDENT ALERT - [SEVERITY]

Type: [Incident Type]
Status: [Investigating/Contained/Resolved]
Impact: [Description]
Started: [Time]
Current Actions: [What's being done]
Next Update: [Time]

Incident Commander: [Name]
```

### Customer Notification (if required)

```
Service Notification

We experienced [brief description] starting at [time].
Impact: [What customers may have experienced]
Status: [Current status]
Resolution: [Expected or actual]

We apologize for any inconvenience.
```

---

## Escalation Matrix

| Time Since Detection | Severity | Action |
|---------------------|----------|--------|
| 0 min | P1 | Page on-call, start bridge |
| 5 min | P1 | Escalate to engineering lead |
| 15 min | P1/P2 | Notify management |
| 30 min | P1 | Executive notification |
| 1 hour | P2 | Engineering lead involvement |
| 4 hours | P3 | Standard response |

---

## Incident Log Template

```markdown
## Incident: [Title]

**Severity:** P[1-4]
**Status:** [Open/Investigating/Resolved/Closed]
**Commander:** [Name]

### Timeline

| Time | Event |
|------|-------|
| HH:MM | [Event description] |

### Impact

- [List affected systems/users]

### Root Cause

[Description once identified]

### Resolution

[Steps taken to resolve]

### Action Items

- [ ] [Follow-up action]
- [ ] [Preventive measure]

### Lessons Learned

[What we learned]
```

---

## Quick Reference Commands

```bash
# Status check
ccea-agent status
ccea-agent doctor

# Emergency stop
ccea-agent kill-switch trigger --reason "..."

# Export evidence
ccea-agent logs export --output incident.tar.gz
ccea-agent state export --output state.json

# Revoke agent
ccea-admin agents revoke --agent-id X

# Disable user
ccea-admin users disable --user-id X

# Query audit
ccea-admin audit query --from "1 hour ago"
```

---

## Related

- [Kill Switch Runbook](./KILL_SWITCH.md)
- [Recovery Procedures](./RECOVERY.md)
- [Agent Revocation](./AGENT_REVOCATION.md)
- [Security Trust Center](../security/TRUST_CENTER.md)
