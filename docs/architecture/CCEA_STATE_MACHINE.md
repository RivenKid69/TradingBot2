# CCEA State Machine Reference

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 11

## Overview

This document defines the state machines for Deployment and Run entities in the CCEA Platform. State transitions follow strict rules to ensure consistency between Cloud and Agent.

---

## 1. Deployment State Machine

### 1.1 States

| State | Description | Cloud Managed | Agent Managed |
|-------|-------------|---------------|---------------|
| `CREATED` | Deployment created, not yet requested | Yes | - |
| `REQUESTED_START` | Start request sent to Agent | Yes | - |
| `PENDING_LOCAL_APPROVAL` | Waiting for local approval | Yes | Yes |
| `APPROVED` | Local approval received | - | Yes |
| `ACTIVE` | Run is active | - | Yes |
| `PAUSED` | Run is paused | Yes | Yes |
| `REQUESTED_STOP` | Stop request sent | Yes | - |
| `STOPPED` | Run stopped normally | - | Yes |
| `HALTED` | Emergency halt (kill switch) | - | Yes |
| `REVOKED` | Agent/key revoked (security) | Yes | - |
| `ARCHIVED` | Deployment archived | Yes | - |

### 1.2 State Diagram

```
                                    ┌─────────────────────────────────────┐
                                    │           ARCHIVED                   │
                                    └─────────────────────────────────────┘
                                                      ▲
                                                      │ archive
                                                      │
┌─────────────┐                                 ┌─────┴─────┐
│   CREATED   │                                 │  STOPPED  │◄─────────────┐
└──────┬──────┘                                 └───────────┘              │
       │                                              ▲                    │
       │ user: Deploy                                 │ stop_complete      │
       │                                              │                    │
       ▼                                        ┌─────┴─────┐              │
┌──────────────────┐                            │ REQUESTED │              │
│  REQUESTED_START │                            │   _STOP   │              │
└────────┬─────────┘                            └───────────┘              │
         │                                            ▲                    │
         │ agent: received                            │ user/cloud:        │
         │ (if requires_approval)                     │ request_stop       │
         ▼                                            │                    │
┌────────────────────────┐                      ┌─────┴─────┐              │
│ PENDING_LOCAL_APPROVAL │                      │  ACTIVE   │◄────────────┤
└──────────┬─────────────┘                      └─────┬─────┘              │
           │                                          │                    │
           │ ┌───────────────────────────────────────┐│                    │
           │ │                                       ││                    │
           ▼ ▼                                       ▼▼                    │
     ┌──────────┐      start_run              ┌──────────┐                 │
     │ APPROVED │─────────────────────────────►│  PAUSED  │◄────┐          │
     └──────────┘                              └──────────┘     │          │
           │                                        │          │ pause    │
           │ rejected                               │ resume   │          │
           ▼                                        └──────────┘          │
     ┌──────────┐                                                         │
     │ REJECTED │                                                         │
     └──────────┘                                                         │
                                                                          │
                                                         ┌────────────────┘
                                                         │
                                                   ┌─────┴─────┐
              ANY STATE ─────────────────────────► │  HALTED   │
                          kill_switch / error      └───────────┘
                                                         │
                                                         │ acknowledge
                                                         ▼
              ANY STATE ─────────────────────────► ┌───────────┐
                          security_revoke          │  REVOKED  │
                                                   └───────────┘
```

### 1.3 Transitions

| From | To | Trigger | Actor | Notes |
|------|-----|---------|-------|-------|
| `CREATED` | `REQUESTED_START` | Deploy/Start button | User (Cloud) | Creates command |
| `REQUESTED_START` | `PENDING_LOCAL_APPROVAL` | Agent receives command | Agent | If `requires_approval=true` |
| `REQUESTED_START` | `APPROVED` | Auto-approve policy match | Agent | If auto-approve configured |
| `PENDING_LOCAL_APPROVAL` | `APPROVED` | Local approval | User (Agent) | Evidence recorded |
| `PENDING_LOCAL_APPROVAL` | `REJECTED` | Local rejection | User (Agent) | Reason recorded |
| `APPROVED` | `ACTIVE` | Run starts | Agent | Creates Run entity |
| `ACTIVE` | `PAUSED` | Pause request | User/Cloud | Safe pause |
| `PAUSED` | `ACTIVE` | Resume request | User/Cloud | May need approval |
| `ACTIVE` | `REQUESTED_STOP` | Stop request | User/Cloud | |
| `REQUESTED_STOP` | `STOPPED` | Run stops | Agent | Clean shutdown |
| `ANY` | `HALTED` | Kill switch | Agent | Emergency halt |
| `ANY` | `REVOKED` | Security revoke | Cloud | Immediate termination |
| `STOPPED` | `ARCHIVED` | Archive action | User (Cloud) | Read-only state |

### 1.4 State Persistence

**Cloud stores:**

- `desired_state`: What Cloud wants
- `current_state`: Last reported by Agent
- `state_updated_at`: Timestamp of last update

**Agent stores:**

- `local_state`: Actual runtime state
- `pending_commands`: Unacknowledged commands
- `approval_queue`: Pending approvals

---

## 2. Run State Machine

### 2.1 States

| State | Description | Terminal |
|-------|-------------|----------|
| `INIT` | Run initialized, not started | No |
| `STARTING` | Run is starting up | No |
| `RUNNING` | Actively executing strategy | No |
| `DEGRADED` | Running with reduced functionality | No |
| `PAUSED` | Temporarily paused | No |
| `STOPPING` | Graceful shutdown in progress | No |
| `STOPPED` | Normal termination | Yes |
| `HALTED` | Emergency halt | Yes |
| `FAILED` | Unexpected error | Yes |

### 2.2 State Diagram

```
┌────────┐
│  INIT  │
└───┬────┘
    │ initialize_complete
    ▼
┌──────────┐
│ STARTING │
└────┬─────┘
     │ start_complete
     ▼
┌──────────┐                    ┌──────────┐
│ RUNNING  │◄───────────────────│ DEGRADED │
└────┬─────┘    recover         └────┬─────┘
     │                               │
     │ degradation_detected          │
     └──────────────────────────────►│
     │                               │
     │ pause_request                 │ pause_request
     ▼                               ▼
┌──────────┐                    ┌──────────┐
│  PAUSED  │◄───────────────────│  PAUSED  │
└────┬─────┘                    └──────────┘
     │ resume_request
     │
     ▼
┌──────────┐
│ RUNNING  │ (or DEGRADED if issue persists)
└────┬─────┘
     │
     │ stop_request
     ▼
┌──────────┐        stop_complete     ┌──────────┐
│ STOPPING │─────────────────────────►│ STOPPED  │
└──────────┘                          └──────────┘


    ANY STATE ─────────────────────► ┌──────────┐
                  kill_switch        │  HALTED  │
                                     └──────────┘

    ANY STATE ─────────────────────► ┌──────────┐
                  unrecoverable_error│  FAILED  │
                                     └──────────┘
```

### 2.3 Transitions

| From | To | Trigger | Notes |
|------|-----|---------|-------|
| `INIT` | `STARTING` | Pre-flight checks pass | Initializing resources |
| `STARTING` | `RUNNING` | Strategy ready | Begin live loop |
| `RUNNING` | `DEGRADED` | Degradation detected | Cloud down, data stale, etc. |
| `DEGRADED` | `RUNNING` | Recovery complete | Issue resolved |
| `RUNNING` | `PAUSED` | Pause request | Strategy paused |
| `PAUSED` | `RUNNING` | Resume request | May need approval |
| `RUNNING` | `STOPPING` | Stop request | Graceful shutdown |
| `STOPPING` | `STOPPED` | Shutdown complete | Normal termination |
| `ANY` | `HALTED` | Kill switch | Emergency stop |
| `ANY` | `FAILED` | Unrecoverable error | Unexpected failure |

### 2.4 Degraded Mode Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Cloud unreachable | Low | Continue with local limits |
| Data feed stale | Medium | Pause new orders |
| Broker errors > threshold | Medium | Pause execution |
| Position mismatch | High | Halt run |
| Time sync drift > 1s | High | Halt run |

---

## 3. Change Classification

### 3.1 TRADING_IMPACTING Changes

Changes that **require local approval** (by default):

| Change Type | Examples | Approval Required |
|-------------|----------|-------------------|
| **Artifact Version** | New strategy build | Always |
| **Mode Switch** | PAPER → LIVE | Always |
| **Universe Change** | Add/remove symbols | Always |
| **Risk Limits** | Any loosening | Always |
| **Execution Params** | Order types, aggressiveness | Always |
| **Broker/Account** | Change adapter | Always |
| **Schedule** | Live trading hours | Always |
| **Strategy Params** | Signal/entry/exit params | Always |

### 3.2 NON_IMPACTING Changes

Changes that **can apply without approval**:

| Change Type | Examples | Approval Required |
|-------------|----------|-------------------|
| **Logging** | Log level change | Never |
| **Telemetry** | Verbosity (non-sensitive) | Never |
| **UI/UX** | Display preferences | Never |
| **Runner Config** | Buffer sizes | Never |
| **Agent Update** | Retail auto-update | Policy-dependent |

### 3.3 Policy Firewall

Agent maintains local policy that:

1. **Hard Caps** - Cannot be overridden by Cloud

   ```yaml
   hard_caps:
     max_position_pct: 10
     max_daily_loss_pct: 2
     max_order_rate_per_min: 100
     allowed_order_types: [LIMIT, MARKET]
     forbidden_symbols: [SCAM_COIN]
   ```

2. **Auto-Approve Rules** - Skip approval for specific patterns

   ```yaml
   auto_approve:
     enabled: true
     whitelist:
       - workspace: trusted_workspace
         strategies: [momentum_v1, mean_revert_v2]
         instruments: [BTC, ETH]
     forbidden:
       - change_type: PAPER_TO_LIVE
       - change_type: RISK_LOOSENING
   ```

3. **Priority** - Local policy > Cloud suggestions
   - Cloud `risk_profile_suggested` is informational only
   - Agent enforces `hard_caps` regardless of Cloud config

---

## 4. Command Flow Examples

### 4.1 Start Run (with Approval)

```
User (Cloud UI)          Cloud                    Agent              Local User
      │                    │                        │                     │
      │ Click "Start"      │                        │                     │
      │───────────────────►│                        │                     │
      │                    │                        │                     │
      │                    │ REQUEST_START_RUN      │                     │
      │                    │───────────────────────►│                     │
      │                    │                        │                     │
      │                    │                        │ Show approval UI    │
      │                    │                        │────────────────────►│
      │                    │                        │                     │
      │                    │                        │                     │ Approve
      │                    │                        │◄────────────────────│
      │                    │                        │                     │
      │                    │ COMMAND_APPROVAL       │                     │
      │                    │◄───────────────────────│                     │
      │                    │                        │                     │
      │                    │                        │ Start run locally   │
      │                    │                        │                     │
      │                    │ COMMAND_RESULT         │                     │
      │                    │◄───────────────────────│                     │
      │                    │                        │                     │
      │ Show "Active"      │                        │                     │
      │◄───────────────────│                        │                     │
```

### 4.2 Upgrade Artifact (TRADING_IMPACTING)

```
User (Cloud UI)          Cloud                    Agent              Local User
      │                    │                        │                     │
      │ Select new version │                        │                     │
      │───────────────────►│                        │                     │
      │                    │                        │                     │
      │                    │ REQUEST_UPGRADE        │                     │
      │                    │ change_class:          │                     │
      │                    │ TRADING_IMPACTING      │                     │
      │                    │───────────────────────►│                     │
      │                    │                        │                     │
      │                    │                        │ Show diff + approval│
      │                    │                        │────────────────────►│
      │                    │                        │                     │
      │                    │                        │ What changed:       │
      │                    │                        │ - Model weights     │
      │                    │                        │ - Entry threshold   │
      │                    │                        │                     │
      │                    │                        │                     │ Approve
      │                    │                        │◄────────────────────│
      │                    │                        │                     │
      │                    │                        │ 1. Stop old run     │
      │                    │                        │ 2. Pull new artifact│
      │                    │                        │ 3. Verify signature │
      │                    │                        │ 4. Start new run    │
      │                    │                        │                     │
      │                    │ COMMAND_RESULT         │                     │
      │                    │◄───────────────────────│                     │
```

### 4.3 Emergency Halt (Kill Switch)

```
                         Cloud                    Agent
                           │                        │
                           │                        │ Kill switch triggered
                           │                        │ (e.g., max loss exceeded)
                           │                        │
                           │                        │ 1. Cancel open orders
                           │                        │ 2. (Optional) Flatten
                           │                        │ 3. Halt run
                           │                        │ 4. Log halt reason
                           │                        │
                           │ TELEMETRY              │
                           │ type: HALT             │
                           │ reason: MAX_DAILY_LOSS │
                           │◄───────────────────────│
                           │                        │
                           │ Create alert           │
                           │ Notify user            │
                           │                        │
```

---

## 5. State Recovery

### 5.1 Agent Restart

When Agent restarts:

1. **Load persisted state** from local journal
2. **Reconcile with broker** - Fetch open orders, positions
3. **Check for divergence** - Compare local vs broker state
4. **Resume or halt**:
   - If consistent: Resume from last state
   - If uncertain: Halt with `RESTART_DIVERGENCE` reason

```python
async def recover_state() -> RunState:
    """Recover state after restart."""
    # 1. Load from journal
    local_state = await journal.load_latest()

    # 2. Fetch broker state
    broker_positions = await broker.get_positions()
    broker_orders = await broker.get_open_orders()

    # 3. Reconcile
    divergence = reconcile(local_state, broker_positions, broker_orders)

    if divergence.is_critical:
        return RunState.HALTED, HaltReason.RESTART_DIVERGENCE

    if divergence.is_minor:
        await apply_corrections(divergence.corrections)

    return local_state.run_state, None
```

### 5.2 Cloud Reconnection

When Agent reconnects to Cloud after disconnection:

1. **Sync state** - Report current state to Cloud
2. **Fetch pending commands** - Process any missed commands
3. **Resolve conflicts** - Handle state mismatches

---

## 6. Halt Reasons

### 6.1 Standard Halt Reasons

| Code | Reason | Severity | Auto-Restart |
|------|--------|----------|--------------|
| `MAX_DAILY_LOSS` | Daily loss limit exceeded | Critical | No |
| `MAX_POSITION` | Position size exceeded | High | No |
| `BROKER_ERRORS` | Too many broker errors | High | After cooldown |
| `LATENCY_SPIKE` | Execution latency too high | Medium | After check |
| `ORDER_SPAM` | Order rate exceeded | High | No |
| `STATE_DIVERGENCE` | Position mismatch | Critical | No |
| `DATA_FEED_INVALID` | Market data invalid | Medium | After check |
| `TIME_SYNC_DRIFT` | Clock drift > threshold | High | After sync |
| `MANUAL_KILL` | User triggered kill | N/A | Manual |
| `RESTART_DIVERGENCE` | Inconsistent after restart | Critical | No |
| `BROKER_DISCONNECT` | Broker connection lost | High | After reconnect |
| `MARGIN_CALL` | Broker margin warning | Critical | No |
| `RISK_BREACH` | Risk parameter breached | High | No |
| `SANDBOX_VIOLATION` | Strategy escaped sandbox | Critical | No |
| `UPGRADE_FAILURE` | Artifact upgrade failed | Medium | Rollback |
| `APPROVAL_TIMEOUT` | Approval not received | Low | Retry |
| `SYSTEM_ERROR` | Unexpected system error | Critical | No |

### 6.2 Halt Evidence

Each halt records:

```json
{
  "halt_reason": "MAX_DAILY_LOSS",
  "triggered_at": "2025-12-14T14:30:00Z",
  "trigger_value": -2.5,
  "threshold": -2.0,
  "context": {
    "positions_at_halt": [...],
    "open_orders_at_halt": [...],
    "last_pnl_calculation": "2025-12-14T14:29:55Z"
  },
  "actions_taken": [
    {"action": "CANCEL_ORDERS", "count": 3, "success": true},
    {"action": "HALT_RUN", "success": true}
  ]
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial state machine per Design Doc |

---

**Related Documentation:**

- [CCEA Overview](./CCEA_OVERVIEW.md)
- [Data Model](./CCEA_DATA_MODEL.md)
- [Agent Documentation](../agent/README.md)
