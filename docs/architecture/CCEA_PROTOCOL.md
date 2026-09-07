# CCEA Protocol Reference

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16
>
> **Reference**: Design Doc CCEA Cloud.txt (canonical source) - Section 10

## Overview

This document defines the communication protocol between Cloud and Agent in the CCEA Platform. The protocol is designed with security and idempotency as core principles.

---

## 1. Transport Layer

### 1.1 Connection Model

**Principle:** Agent makes outbound connections only - no inbound ports required in customer infrastructure.

| Option | Use Case | Notes |
|--------|----------|-------|
| **HTTPS Long-Poll** | Default | Simple, reliable, firewall-friendly |
| **WebSocket** | Real-time needs | For low-latency telemetry |
| **gRPC over TLS** | Enterprise | High throughput, bidirectional streaming |

**Default: HTTPS Long-Poll**

```
Agent ──────(outbound HTTPS)──────► Cloud

Cloud cannot initiate connection to Agent.
Agent polls Cloud for commands.
```

### 1.2 Endpoints

```
Base URL: https://api.ccea.cloud/v1

Agent → Cloud:
  POST /agents/{agent_id}/heartbeat
  POST /agents/{agent_id}/commands/poll
  POST /agents/{agent_id}/commands/{cmd_id}/ack
  POST /agents/{agent_id}/commands/{cmd_id}/result
  POST /agents/{agent_id}/telemetry
  POST /enrollment/complete
```

### 1.3 Authentication

| Phase | Method |
|-------|--------|
| **Enrollment** | One-time token (short TTL, ~15 min) |
| **Post-Enrollment** | Device key + signed JWT or mTLS |

**Every message includes:**

- `agent_id` - Agent identifier
- `signature` - Message signature (Ed25519 or equivalent)
- `timestamp` - ISO 8601 timestamp (checked for drift)

---

## 2. Schema Versioning

### 2.1 Version Format

All messages include `schema_version` field: `MAJOR.MINOR.PATCH`

```json
{
  "schema_version": "1.0.0",
  "type": "HEARTBEAT",
  ...
}
```

### 2.2 Compatibility

| Agent Version | Cloud Version | Result |
|---------------|---------------|--------|
| 1.0 | 1.0 | Compatible |
| 1.0 | 1.1 | Compatible (Cloud ignores unknown fields) |
| 1.1 | 1.0 | Compatible (Agent uses defaults) |
| 2.0 | 1.x | **Incompatible** - major mismatch |

### 2.3 Negotiation

During enrollment, Agent and Cloud agree on protocol version:

```json
// Agent enrollment request
{
  "min_schema_version": "1.0.0",
  "max_schema_version": "1.2.0"
}

// Cloud response
{
  "negotiated_schema_version": "1.2.0"
}
```

---

## 3. Message Types

### 3.1 Agent → Cloud Messages

| Type | Purpose | Frequency |
|------|---------|-----------|
| `HEARTBEAT` | Health status | Every 30s |
| `POLL_COMMANDS` | Fetch pending commands | Every 5-30s |
| `COMMAND_ACK` | Acknowledge command receipt | Per command |
| `COMMAND_APPROVAL` | Report approval decision | Per command |
| `COMMAND_RESULT` | Report execution result | Per command |
| `TELEMETRY` | Metrics and events | Configurable |

### 3.2 Cloud → Agent Messages

| Type | Purpose | Notes |
|------|---------|-------|
| `COMMAND_BATCH` | Pending commands | Response to poll |

### 3.3 Command Types (Cloud → Agent via poll)

| Command | Description | Change Class | Approval |
|---------|-------------|--------------|----------|
| `REQUEST_START_RUN` | Start strategy run | TRADING_IMPACTING | Yes |
| `REQUEST_STOP_RUN` | Stop running strategy | NON_IMPACTING | No |
| `REQUEST_PAUSE_RUN` | Pause execution | NON_IMPACTING | No |
| `REQUEST_RESUME_RUN` | Resume execution | TRADING_IMPACTING | Yes |
| `REQUEST_UPGRADE_ARTIFACT` | Deploy new version | TRADING_IMPACTING | Yes |
| `REQUEST_UPDATE_CONFIG` | Update configuration | Depends on content | Depends |
| `REQUEST_ROTATE_AGENT_SESSION` | Rotate session keys | NON_IMPACTING | No |
| `REQUEST_EXPORT_LOGS` | Export logs | NON_IMPACTING | Optional |

---

## 4. Message Schemas

### 4.1 Agent Heartbeat

**Direction:** Agent → Cloud

```json
{
  "schema_version": "1.0",
  "type": "HEARTBEAT",
  "agent_id": "ag_123",
  "timestamp": "2025-12-12T10:00:00Z",
  "status": "ONLINE",
  "agent_version": "0.9.3",
  "capabilities": {
    "sandbox": ["docker", "process"],
    "gpu": false,
    "os": "linux",
    "broker_connectors": ["binance", "alpaca"]
  },
  "active_runs": [
    {
      "run_id": "run_555",
      "deployment_id": "dep_777",
      "state": "RUNNING"
    }
  ],
  "metrics": {
    "cpu_percent": 25,
    "memory_mb": 512,
    "disk_free_gb": 50
  }
}
```

### 4.2 Command Poll Request

**Direction:** Agent → Cloud

```json
{
  "schema_version": "1.0",
  "type": "POLL_COMMANDS",
  "agent_id": "ag_123",
  "since": "2025-12-12T09:59:00Z",
  "max": 50,
  "long_poll_timeout_seconds": 25
}
```

### 4.3 Command Batch Response

**Direction:** Cloud → Agent (response to poll)

```json
{
  "schema_version": "1.0",
  "type": "COMMAND_BATCH",
  "agent_id": "ag_123",
  "commands": [
    {
      "command_id": "cmd_001",
      "idempotency_key": "dep_777:REQUEST_START:build_sha256:abc123",
      "command_type": "REQUEST_START_RUN",
      "deployment_id": "dep_777",
      "change_class": "TRADING_IMPACTING",
      "requires_approval": true,
      "payload_ref": "sha256:configblob...",
      "issued_at": "2025-12-12T10:00:00Z",
      "expires_at": "2025-12-12T11:00:00Z"
    }
  ],
  "poll_interval_seconds": 10
}
```

### 4.4 Command Acknowledgment

**Direction:** Agent → Cloud

```json
{
  "schema_version": "1.0",
  "type": "COMMAND_ACK",
  "agent_id": "ag_123",
  "command_id": "cmd_001",
  "status": "ACKED",
  "timestamp": "2025-12-12T10:00:05Z",
  "reason": null
}
```

**Status Values:**

- `ACKED` - Command received, processing started
- `REJECTED` - Command rejected (invalid, unsupported, etc.)

### 4.5 Approval Decision

**Direction:** Agent → Cloud (after local approval)

**Approved:**

```json
{
  "schema_version": "1.0",
  "type": "COMMAND_APPROVAL",
  "agent_id": "ag_123",
  "command_id": "cmd_001",
  "decision": "APPROVED",
  "approved_by": "local_user:boss",
  "approved_at": "2025-12-12T10:02:00Z",
  "evidence_hash": "sha256:approval_evidence..."
}
```

**Rejected:**

```json
{
  "schema_version": "1.0",
  "type": "COMMAND_APPROVAL",
  "agent_id": "ag_123",
  "command_id": "cmd_001",
  "decision": "REJECTED",
  "rejected_by": "local_user:boss",
  "rejected_at": "2025-12-12T10:02:00Z",
  "reason": "Universe changed: includes forbidden symbol SCAM_COIN"
}
```

### 4.6 Command Result

**Direction:** Agent → Cloud

**Success:**

```json
{
  "schema_version": "1.0",
  "type": "COMMAND_RESULT",
  "agent_id": "ag_123",
  "command_id": "cmd_001",
  "status": "APPLIED",
  "timestamp": "2025-12-12T10:03:00Z",
  "details": {
    "run_id": "run_555",
    "started_at": "2025-12-12T10:03:00Z"
  }
}
```

**Failure:**

```json
{
  "schema_version": "1.0",
  "type": "COMMAND_RESULT",
  "agent_id": "ag_123",
  "command_id": "cmd_001",
  "status": "FAILED",
  "timestamp": "2025-12-12T10:03:00Z",
  "error": {
    "code": "PREFLIGHT_FAILED",
    "message": "Broker connectivity check failed",
    "details": {
      "broker": "binance",
      "error": "API key invalid"
    }
  }
}
```

**Result Status Values:**

- `APPLIED` - Command executed successfully
- `FAILED` - Execution failed (with error details)
- `EXPIRED` - Command expired before execution
- `SUPERSEDED` - Replaced by newer command

### 4.7 Telemetry Event

**Direction:** Agent → Cloud

**AGGREGATED level (default):**

```json
{
  "schema_version": "1.0",
  "type": "TELEMETRY",
  "agent_id": "ag_123",
  "deployment_id": "dep_777",
  "run_id": "run_555",
  "sensitivity": "AGGREGATED",
  "timestamp": "2025-12-12T10:05:00Z",
  "event_type": "METRICS",
  "metrics": {
    "pnl": 12.34,
    "drawdown": -3.2,
    "exposure_usd": 5000.0,
    "orders_per_min": 2,
    "broker_error_rate": 0,
    "latency_p99_ms": 45
  }
}
```

**DETAILED level (opt-in, enterprise):**

```json
{
  "schema_version": "1.0",
  "type": "TELEMETRY",
  "agent_id": "ag_123",
  "deployment_id": "dep_777",
  "run_id": "run_555",
  "sensitivity": "DETAILED",
  "timestamp": "2025-12-12T10:05:00Z",
  "event_type": "STATE_CHANGE",
  "state": {
    "run_state": "RUNNING",
    "strategy_signals": {
      "count": 5,
      "avg_confidence": 0.72
    },
    "execution_stats": {
      "orders_submitted": 10,
      "orders_filled": 8,
      "fill_rate": 0.8
    }
  }
}
```

**Halt Event:**

```json
{
  "schema_version": "1.0",
  "type": "TELEMETRY",
  "agent_id": "ag_123",
  "deployment_id": "dep_777",
  "run_id": "run_555",
  "sensitivity": "AGGREGATED",
  "timestamp": "2025-12-12T14:30:00Z",
  "event_type": "HALT",
  "halt": {
    "reason": "MAX_DAILY_LOSS",
    "threshold": -2.0,
    "actual": -2.5,
    "actions_taken": ["CANCEL_ORDERS", "HALT_RUN"]
  }
}
```

---

## 5. Prohibited Protocol Elements

### 5.1 Forbidden Message Types

These message types **MUST NEVER** exist in the protocol:

| Forbidden Type | Reason |
|----------------|--------|
| `PLACE_ORDER` | Cloud must not send orders |
| `SUBMIT_ORDER` | Cloud must not send orders |
| `EXECUTE_SIGNAL` | Cloud must not send signals |
| `SET_TARGET_POSITION_NOW` | Cloud must not control positions |
| `CANCEL_ORDER` | Order management is Agent-only |
| `MODIFY_ORDER` | Order management is Agent-only |
| `FLATTEN_POSITION` | Position control is Agent-only |

### 5.2 Forbidden Payload Fields

Command payloads **MUST NEVER** contain:

| Field | Reason |
|-------|--------|
| `side` | Order field (BUY/SELL) |
| `quantity` / `qty` | Order field |
| `price` | Order field |
| `order_type` | Order field |
| `target_position` | Position control |
| `symbol` (in order context) | Order routing |
| `api_key` / `api_secret` | Credentials |

### 5.3 Schema Enforcement

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "properties": {
    "payload": {
      "type": "object",
      "not": {
        "anyOf": [
          {"required": ["side"]},
          {"required": ["quantity"]},
          {"required": ["qty"]},
          {"required": ["price"]},
          {"required": ["order_type"]},
          {"required": ["target_position"]},
          {"required": ["api_key"]},
          {"required": ["api_secret"]}
        ]
      }
    }
  }
}
```

---

## 6. Idempotency

### 6.1 Idempotency Keys

Every command has a unique `idempotency_key`:

```
Format: {deployment_id}:{command_type}:{content_hash}

Example: dep_777:REQUEST_START:sha256:abc123
```

### 6.2 Duplicate Handling

| Scenario | Agent Behavior |
|----------|----------------|
| Same key, PENDING | Process normally |
| Same key, ACKED | Return cached ACK |
| Same key, APPLIED | Return cached RESULT |
| Same key, different payload | REJECT (hash mismatch) |

### 6.3 Command Expiration

Commands include `expires_at` timestamp:

- Expired commands are not processed
- Agent returns `EXPIRED` status
- Default TTL: 1 hour

---

## 7. Error Handling

### 7.1 Error Response Format

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable message",
    "details": {
      "field": "additional context"
    },
    "retry_after_seconds": 60
  }
}
```

### 7.2 Error Codes

| Code | Description | Retry |
|------|-------------|-------|
| `INVALID_SIGNATURE` | Message signature invalid | No |
| `INVALID_SCHEMA` | Schema validation failed | No |
| `EXPIRED_COMMAND` | Command TTL exceeded | No |
| `UNKNOWN_AGENT` | Agent not registered | No |
| `AGENT_REVOKED` | Agent trust revoked | No |
| `RATE_LIMITED` | Too many requests | Yes |
| `INTERNAL_ERROR` | Server error | Yes |
| `MAINTENANCE` | Planned maintenance | Yes |

### 7.3 Retry Policy

```yaml
retry:
  initial_delay_ms: 1000
  max_delay_ms: 60000
  multiplier: 2.0
  max_retries: 5
  retryable_codes:
    - RATE_LIMITED
    - INTERNAL_ERROR
    - MAINTENANCE
```

---

## 8. Security Considerations

### 8.1 Message Signing

All messages must be signed:

```python
def sign_message(message: dict, private_key: bytes) -> str:
    """Sign message with Ed25519."""
    canonical = canonical_json(message)
    signature = ed25519_sign(canonical, private_key)
    return base64.b64encode(signature).decode()

def verify_message(message: dict, signature: str, public_key: bytes) -> bool:
    """Verify message signature."""
    canonical = canonical_json(message)
    sig_bytes = base64.b64decode(signature)
    return ed25519_verify(canonical, sig_bytes, public_key)
```

### 8.2 Timestamp Validation

```python
MAX_CLOCK_DRIFT_SECONDS = 60

def validate_timestamp(message_timestamp: str) -> bool:
    """Reject messages with excessive clock drift."""
    msg_time = parse_iso8601(message_timestamp)
    now = datetime.utcnow()
    drift = abs((now - msg_time).total_seconds())
    return drift <= MAX_CLOCK_DRIFT_SECONDS
```

### 8.3 Rate Limiting

| Endpoint | Rate Limit |
|----------|------------|
| Heartbeat | 2/min per agent |
| Command Poll | 12/min per agent |
| Telemetry | 60/min per agent |
| Enrollment | 5/hour per IP |

---

## 9. Telemetry Levels

### 9.1 Level Definitions

| Level | Data Included | Default |
|-------|---------------|---------|
| `AGGREGATED` | PnL, drawdown, error rates, health | Yes (retail) |
| `DETAILED` | Timestamps, latency, counts (no orders) | Opt-in (pro) |
| `RAW_ORDER_EVENTS` | **NEVER SENT** | N/A |

### 9.2 Redaction Rules

Before sending telemetry:

```python
REDACT_PATTERNS = [
    r'api[_-]?key',
    r'api[_-]?secret',
    r'password',
    r'token',
    r'credential',
]

def redact_telemetry(data: dict) -> dict:
    """Mandatory redaction before transmission."""
    for key, value in data.items():
        if any(re.match(p, key, re.I) for p in REDACT_PATTERNS):
            data[key] = "[REDACTED]"
        elif isinstance(value, dict):
            data[key] = redact_telemetry(value)
    return data
```

---

## 10. Configuration Reference

### 10.1 Agent Protocol Config

```yaml
protocol:
  cloud_endpoint: https://api.ccea.cloud/v1
  schema_version: "1.0"

  heartbeat:
    interval_seconds: 30
    timeout_seconds: 10

  command_poll:
    interval_seconds: 10
    long_poll_timeout_seconds: 25
    max_commands_per_poll: 50

  telemetry:
    batch_size: 100
    flush_interval_seconds: 60
    level: AGGREGATED

  retry:
    initial_delay_ms: 1000
    max_delay_ms: 60000
    max_retries: 5

  security:
    verify_cloud_signature: true
    max_clock_drift_seconds: 60
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-16 | CCEA Team | Initial protocol per Design Doc |

---

**Related Documentation:**

- [CCEA Overview](./CCEA_OVERVIEW.md)
- [State Machine](./CCEA_STATE_MACHINE.md)
- [JSON Schemas](../schemas/README.md)
