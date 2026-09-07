# Control Plane API Reference

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

The Control Plane API provides lifecycle management for agents, deployments, and strategies. All endpoints enforce strict security boundaries - no order-like payloads are accepted.

## Base URL

```
Production: https://api.ccea.cloud/v1
Staging:    https://api-staging.ccea.cloud/v1
```

## Authentication

### Bearer Token (JWT)

```http
Authorization: Bearer <jwt_token>
```

### API Key (for agents)

```http
X-Agent-Key: <agent_api_key>
X-Agent-Signature: <request_signature>
```

---

## Enrollment

### Create Enrollment Token

Create a time-limited token for agent enrollment.

```http
POST /enrollment/token
```

**Request Body:**

```json
{
  "workspace_id": "ws_abc123",
  "label": "production-agent-1",
  "ttl_minutes": 30
}
```

**Response:**

```json
{
  "token": "enroll_xxxx",
  "expires_at": "2025-12-14T12:30:00Z",
  "agent_id": "agent_pending_xyz"
}
```

### Enroll Agent

Register a new agent using an enrollment token.

```http
POST /agents/enroll
```

**Request Body:**

```json
{
  "token": "enroll_xxxx",
  "public_key": "-----BEGIN PUBLIC KEY-----\n...",
  "agent_version": "1.0.0",
  "capabilities": ["live_trading", "paper_trading"],
  "platform": "linux-x64"
}
```

**Response:**

```json
{
  "agent_id": "agent_xyz",
  "workspace_id": "ws_abc123",
  "trust_state": "ENROLLED",
  "session_key": "sk_xxx",
  "cloud_public_key": "-----BEGIN PUBLIC KEY-----\n..."
}
```

---

## Agents

### List Agents

```http
GET /agents
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `workspace_id` | string | Filter by workspace |
| `trust_state` | string | `ENROLLED`, `REVOKED` |
| `limit` | int | Max results (default: 50) |
| `offset` | int | Pagination offset |

**Response:**

```json
{
  "agents": [
    {
      "agent_id": "agent_xyz",
      "label": "production-agent-1",
      "trust_state": "ENROLLED",
      "last_seen": "2025-12-14T12:00:00Z",
      "agent_version": "1.0.0",
      "capabilities": ["live_trading"]
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

### Get Agent

```http
GET /agents/{agent_id}
```

### Revoke Agent

Revoke an agent's trust status.

```http
POST /agents/{agent_id}/revoke
```

**Request Body:**

```json
{
  "reason": "security_incident",
  "notes": "Suspicious activity detected"
}
```

### Agent Heartbeat

Report agent health and receive pending commands.

```http
POST /agents/{agent_id}/heartbeat
```

**Request Body:**

```json
{
  "timestamp": "2025-12-14T12:00:00Z",
  "state": "RUNNING",
  "active_runs": ["run_abc"],
  "metrics": {
    "cpu_percent": 25.5,
    "memory_mb": 512,
    "uptime_seconds": 3600
  }
}
```

**Response:**

```json
{
  "ack": true,
  "server_time": "2025-12-14T12:00:01Z",
  "pending_commands": 2
}
```

---

## Commands

### Poll Commands

Long-poll for pending commands.

```http
GET /agents/{agent_id}/commands
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `timeout` | int | Long-poll timeout (seconds, max: 30) |
| `last_seen` | string | Last command ID received |

**Response:**

```json
{
  "commands": [
    {
      "command_id": "cmd_abc",
      "type": "REQUEST_START_RUN",
      "idempotency_key": "idem_xyz",
      "payload_ref": "sha256:abc123",
      "requires_approval": true,
      "created_at": "2025-12-14T12:00:00Z",
      "signature": "..."
    }
  ]
}
```

### Acknowledge Command

```http
POST /agents/{agent_id}/commands/{command_id}/ack
```

**Request Body:**

```json
{
  "status": "RECEIVED",
  "timestamp": "2025-12-14T12:00:05Z",
  "signature": "..."
}
```

### Submit Command Result

```http
POST /agents/{agent_id}/commands/{command_id}/result
```

**Request Body:**

```json
{
  "status": "COMPLETED",
  "result": {
    "run_id": "run_abc",
    "started_at": "2025-12-14T12:00:10Z"
  },
  "evidence_hash": "sha256:def456",
  "signature": "..."
}
```

### Submit Approval

```http
POST /agents/{agent_id}/commands/{command_id}/approval
```

**Request Body:**

```json
{
  "approved": true,
  "evidence_hash": "sha256:ghi789",
  "approved_at": "2025-12-14T12:00:08Z",
  "approver": "local_user",
  "signature": "..."
}
```

---

## Strategies

### Create Strategy

```http
POST /strategies
```

**Request Body:**

```json
{
  "workspace_id": "ws_abc123",
  "name": "momentum_strategy",
  "description": "Momentum-based trading strategy",
  "asset_class": "crypto"
}
```

### List Strategy Versions

```http
GET /strategies/{strategy_id}/versions
```

### Create Strategy Version

```http
POST /strategies/{strategy_id}/versions
```

**Request Body:**

```json
{
  "source_ref": "git://repo#sha256:abc",
  "config": {
    "lookback_periods": 20,
    "threshold": 0.02
  }
}
```

---

## Deployments

### Create Deployment

```http
POST /deployments
```

**Request Body:**

```json
{
  "workspace_id": "ws_abc123",
  "strategy_version_id": "sv_abc",
  "agent_id": "agent_xyz",
  "config_digest": "sha256:config123",
  "mode": "paper"
}
```

**Note:** Deployment creation does NOT start the run. The agent must approve and start via local approval.

### Get Deployment

```http
GET /deployments/{deployment_id}
```

**Response:**

```json
{
  "deployment_id": "dep_abc",
  "strategy_version_id": "sv_abc",
  "agent_id": "agent_xyz",
  "state": "ENROLLED",
  "current_run_id": null,
  "created_at": "2025-12-14T11:00:00Z"
}
```

### Request Start Run

Send a request to start a deployment (requires local approval).

```http
POST /deployments/{deployment_id}/request-start
```

**Request Body:**

```json
{
  "artifact_digest": "sha256:artifact123",
  "config_digest": "sha256:config123",
  "reason": "scheduled_start"
}
```

**Note:** This creates a `REQUEST_START_RUN` command that the agent must approve locally.

### Request Stop Run

```http
POST /deployments/{deployment_id}/request-stop
```

**Request Body:**

```json
{
  "reason": "user_requested"
}
```

**Note:** Stop commands do NOT require local approval (safety feature).

### Request Pause Run

```http
POST /deployments/{deployment_id}/request-pause
```

---

## Telemetry

### Ingest Telemetry

```http
POST /telemetry
```

**Request Body:**

```json
{
  "agent_id": "agent_xyz",
  "run_id": "run_abc",
  "events": [
    {
      "timestamp": "2025-12-14T12:00:00Z",
      "type": "EQUITY_UPDATE",
      "level": "AGGREGATED",
      "data": {
        "equity": 10500.00,
        "pnl_percent": 5.0,
        "drawdown_percent": -1.2
      }
    }
  ],
  "signature": "..."
}
```

**Note:** All telemetry is automatically redacted. The `level` field indicates the telemetry detail level.

### Query Telemetry

```http
GET /telemetry
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | Filter by run |
| `type` | string | Event type filter |
| `from` | datetime | Start time |
| `to` | datetime | End time |

---

## Artifacts

### Get Artifact

```http
GET /artifacts/{digest}
```

**Response Headers:**

```
Content-Type: application/octet-stream
X-Artifact-Signature: cosign:xxx
X-Artifact-SBOM-Ref: sbom:sha256:xxx
```

### Get Artifact Manifest

```http
GET /artifacts/{digest}/manifest
```

**Response:**

```json
{
  "schema_version": "1.0.0",
  "artifact_digest": "sha256:xxx",
  "entrypoint": "strategy.main:run",
  "runtime": "python:3.12",
  "deps_lock_digest": "sha256:deps",
  "signature": "...",
  "sbom_ref": "sbom:sha256:xxx",
  "provenance": {
    "git_sha": "abc123",
    "build_timestamp": "2025-12-14T10:00:00Z"
  }
}
```

---

## Error Responses

All errors follow this format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Missing required field: workspace_id",
    "details": {
      "field": "workspace_id"
    }
  },
  "request_id": "req_xxx"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request |
| `UNAUTHORIZED` | 401 | Invalid or missing auth |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `CONFLICT` | 409 | State conflict |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## Rate Limits

| Endpoint Type | Rate Limit |
|---------------|------------|
| Read (GET) | 100 req/min |
| Write (POST/PUT/DELETE) | 30 req/min |
| Heartbeat | 60 req/min |
| Telemetry | 120 req/min |

---

## Webhooks (Optional)

Configure webhooks for real-time notifications:

```http
POST /webhooks
```

**Request Body:**

```json
{
  "url": "https://your-server.com/webhook",
  "events": ["deployment.started", "deployment.stopped", "agent.disconnected"],
  "secret": "your_webhook_secret"
}
```

---

## Protocol Security

### Prohibited Payloads

The API is designed to reject payloads containing order-like fields (schema validation + guardrails; verify via tests in the current deployment):

```json
// REJECTED - returns 400 Bad Request
{
  "side": "BUY",
  "quantity": 100,
  "price": 50000
}
```

### Signature Verification

Commands from Cloud are designed to be signed under the CCEA protocol. Agents MUST verify:

1. Signature is valid for payload
2. Timestamp is within acceptable drift (default: 60 seconds)
3. Idempotency key has not been processed

---

## SDK Examples

### Python

```python
from ccea_cloud_sdk import CloudClient

client = CloudClient(
    api_key="your_api_key",
    base_url="https://api.ccea.cloud/v1"
)

# Create deployment
deployment = client.deployments.create(
    workspace_id="ws_abc123",
    strategy_version_id="sv_abc",
    agent_id="agent_xyz"
)

# Request start (requires agent approval)
client.deployments.request_start(
    deployment_id=deployment.id,
    artifact_digest="sha256:xxx"
)
```

### cURL

```bash
# Get agent status
curl -X GET https://api.ccea.cloud/v1/agents/agent_xyz \
  -H "Authorization: Bearer $JWT_TOKEN"

# Request stop (no approval needed)
curl -X POST https://api.ccea.cloud/v1/deployments/dep_abc/request-stop \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"reason": "maintenance"}'
```
