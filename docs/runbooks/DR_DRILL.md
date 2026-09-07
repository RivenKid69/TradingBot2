# Disaster Recovery Drill Runbook

> **Severity**: High | **Last Updated**: 2025-12-20 | **Owner**: Operations

## Overview

This runbook documents the procedure for conducting Disaster Recovery (DR) drills
to validate RTO (Recovery Time Objective) and RPO (Recovery Point Objective) targets.

**Tech Debt Closure**: Reliability/Operations - RTO/RPO DR validation
**Control Artifact**: This runbook + drill execution logs

---

## Target Objectives

Per `docs/CYBERSECURITY_FRAMEWORK.md`:

| Metric | Target | Validation Status |
|--------|--------|-------------------|
| RTO | 4 hours | Pending drill validation |
| RPO | 1 hour | Pending drill validation |

> **Note**: These are design targets. Actual validated values will be documented
> after successful DR drills.

---

## Pre-Drill Checklist

- [ ] Schedule drill during low-impact window (non-market hours recommended)
- [ ] Notify all stakeholders (engineering, operations, management)
- [ ] Prepare rollback plan if drill impacts production
- [ ] Ensure backup infrastructure is available
- [ ] Assign drill coordinator and timekeeper
- [ ] Prepare documentation template for results

---

## Drill Types

### Type 1: Agent Recovery Drill

**Objective**: Validate agent can recover from complete failure within RTO.

**Procedure**:

1. **Preparation** (T-30 min)

   ```bash
   # Record current state
   ccea-agent positions list > pre_drill_positions.json
   ccea-agent orders list > pre_drill_orders.json
   date -u > drill_start_time.txt
   ```

2. **Simulate Failure** (T=0)

   ```bash
   # Stop agent abruptly (simulate crash)
   kill -9 $(pgrep ccea-agent)

   # Delete local state (simulate data loss)
   mv ~/.ccea/journal ~/.ccea/journal.backup

   # Record failure time
   date -u >> drill_log.txt
   echo "FAILURE SIMULATED" >> drill_log.txt
   ```

3. **Recovery** (Start clock)

   ```bash
   # Restore from backup
   ccea-agent restore --from-backup latest

   # Or: Re-initialize with reconciliation
   ccea-agent start --full-reconcile

   # Record steps and timing
   ```

4. **Validation**

   ```bash
   # Verify positions match
   ccea-agent positions list > post_drill_positions.json
   diff pre_drill_positions.json post_drill_positions.json

   # Verify orders match
   ccea-agent orders list > post_drill_orders.json
   diff pre_drill_orders.json post_drill_orders.json

   # Record recovery time
   date -u >> drill_log.txt
   echo "RECOVERY COMPLETE" >> drill_log.txt
   ```

5. **Calculate RTO**

   ```bash
   # Compare timestamps
   python3 -c "
   from datetime import datetime
   with open('drill_log.txt') as f:
       lines = f.readlines()
   # Parse and calculate delta
   "
   ```

### Type 2: Database Recovery Drill

**Objective**: Validate RPO for telemetry and journal data.

**Procedure**:

1. **Record baseline**

   ```bash
   # Count records
   sqlite3 ~/.ccea/telemetry.db "SELECT COUNT(*) FROM events;" > baseline_count.txt

   # Record last event timestamp
   sqlite3 ~/.ccea/telemetry.db "SELECT MAX(timestamp) FROM events;" > last_event.txt
   ```

2. **Simulate corruption**

   ```bash
   # Backup first
   cp ~/.ccea/telemetry.db ~/.ccea/telemetry.db.pre_drill

   # Corrupt (truncate)
   truncate -s 50% ~/.ccea/telemetry.db
   ```

3. **Recover from backup**

   ```bash
   # Restore from backup
   cp /backup/telemetry.db ~/.ccea/telemetry.db

   # OR: Recover from journal
   ccea-agent telemetry rebuild --from-journal
   ```

4. **Calculate RPO**

   ```bash
   # Compare event counts
   sqlite3 ~/.ccea/telemetry.db "SELECT COUNT(*) FROM events;" > recovered_count.txt

   # Lost events = baseline - recovered
   # RPO = time span of lost events
   ```

### Type 3: Full Infrastructure Drill

**Objective**: Validate end-to-end recovery including Cloud and Agent.

**Procedure**:

1. **Simulate regional outage** (Cloud unavailable)
2. **Verify Agent enters degraded mode**
3. **Restore Cloud from backup region**
4. **Verify Agent reconnects and syncs**
5. **Validate no trading disruption**

---

## Drill Execution Template

```markdown
# DR Drill Report

**Date**: YYYY-MM-DD
**Type**: [Agent Recovery | Database Recovery | Full Infrastructure]
**Participants**: [Names]
**Coordinator**: [Name]

## Timeline

| Time (UTC) | Event | Notes |
|------------|-------|-------|
| HH:MM | Drill started | |
| HH:MM | Failure simulated | |
| HH:MM | Recovery initiated | |
| HH:MM | Recovery completed | |
| HH:MM | Validation completed | |

## Results

| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| RTO | 4 hours | HH:MM | |
| RPO | 1 hour | HH:MM | |

## Issues Encountered

1. [Issue description]
   - Impact: [Low/Medium/High]
   - Resolution: [How resolved]

## Recommendations

1. [Recommendation]

## Sign-off

- [ ] Operations Lead
- [ ] Engineering Lead
- [ ] Management (if applicable)
```

---

## Post-Drill Actions

1. **Document Results**
   - Archive drill log
   - Update CYBERSECURITY_FRAMEWORK.md with validated values
   - File drill report in docs/operations/drills/

2. **Address Gaps**
   - Create tickets for identified issues
   - Update runbooks with lessons learned

3. **Schedule Next Drill**
   - Recommended: Quarterly
   - Document in ops calendar

---

## Drill Schedule

| Quarter | Drill Type | Status |
|---------|------------|--------|
| Q1 2025 | Agent Recovery | Planned |
| Q2 2025 | Database Recovery | Planned |
| Q3 2025 | Full Infrastructure | Planned |
| Q4 2025 | Annual Comprehensive | Planned |

---

## Related Documents

- [Recovery Procedures](./RECOVERY.md)
- [Cybersecurity Framework](../CYBERSECURITY_FRAMEWORK.md)
- [Incident Response](./INCIDENT_RESPONSE.md)
- [Backup Procedures](./DATA_LOSS.md)

---

*This document follows the Documentation Canon - targets are disclosed as design goals pending validation.*
