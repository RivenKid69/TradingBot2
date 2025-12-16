# JSON Schemas: Versioning & Reference Guide

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

This directory contains JSON schemas for CCEA protocol messages and artifact manifests. All schemas follow strict versioning to ensure compatibility between Cloud and Agent.

## Schema Files

| File | Purpose | Current Version |
|------|---------|-----------------|
| `artifact_manifest.schema.json` | Strategy artifact metadata | 1.0.0 |
| `protocol_messages.schema.json` | Cloud↔Agent protocol | 1.0.0 |

---

## Schema Versioning

### Semantic Versioning (SemVer)

Schemas use semantic versioning: `MAJOR.MINOR.PATCH`

| Version Change | Meaning | Compatibility |
|----------------|---------|---------------|
| **MAJOR** (1.0.0 → 2.0.0) | Breaking changes | Not backward compatible |
| **MINOR** (1.0.0 → 1.1.0) | New fields (optional) | Backward compatible |
| **PATCH** (1.0.0 → 1.0.1) | Clarifications only | Fully compatible |

### Version Negotiation

Cloud and Agent negotiate compatible versions:

```
Agent: "I support schema versions 1.0.0 - 1.2.0"
Cloud: "I support schema versions 1.1.0 - 1.3.0"
Result: Use version 1.2.0 (highest mutual)
```

**Negotiation Flow:**
```json
// Agent enrollment
{
  "min_schema_version": "1.0.0",
  "max_schema_version": "1.2.0",
  "preferred_schema_version": "1.2.0"
}

// Cloud response
{
  "negotiated_schema_version": "1.2.0"
}
```

### Compatibility Rules

| Rule | Description |
|------|-------------|
| **Forward Compatible** | Old agent can read new schema (ignores unknown fields) |
| **Backward Compatible** | New agent can read old schema (handles missing optional fields) |
| **Major Version Lock** | Major version must match exactly |

---

## Artifact Manifest Schema

### Purpose

Describes a strategy artifact's metadata, dependencies, and requirements.

### Current Version: 1.0.0

**Location:** `artifact_manifest.schema.json`

### Key Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | Yes | Schema version (e.g., "1.0.0") |
| `artifact_id` | string | Yes | Unique artifact identifier |
| `artifact_digest` | string | Yes | SHA256 content hash |
| `entrypoint` | string | Yes | Python entrypoint (e.g., "strategy.main:run") |
| `runtime` | string | Yes | Runtime requirement (e.g., "python:3.12") |
| `deps_lock_digest` | string | Yes | Dependencies lock file hash |
| `model_refs` | array | No | ML model references |
| `data_contract` | object | No | Input/output specification |
| `permissions` | object | No | Required permissions |
| `risk_profile_suggested` | object | No | Suggested risk settings |
| `live_capabilities` | object | No | Live trading requirements |
| `telemetry_schema_version` | string | No | Telemetry format version |
| `change_class` | string | No | Change classification |
| `provenance` | object | No | Build provenance |
| `signature` | string | No | Artifact signature |
| `sbom_ref` | string | No | SBOM reference |

### Example

```json
{
  "schema_version": "1.0.0",
  "artifact_id": "momentum_strategy_v1",
  "artifact_digest": "sha256:a1b2c3d4e5f6...",
  "entrypoint": "strategy.main:run",
  "runtime": "python:3.12",
  "deps_lock_digest": "sha256:deps123...",
  "model_refs": [
    {
      "name": "policy_network",
      "digest": "sha256:model456..."
    }
  ],
  "data_contract": {
    "required_features": ["close", "volume", "ma_20"],
    "output_type": "OrderIntent"
  },
  "permissions": {
    "filesystem": "read_only",
    "network": "none"
  },
  "risk_profile_suggested": {
    "max_position_pct": 10,
    "max_daily_loss_pct": 2
  },
  "live_capabilities": {
    "requires_broker_access": true,
    "supported_brokers": ["binance", "alpaca"]
  },
  "change_class": "TRADING_IMPACTING",
  "provenance": {
    "git_sha": "abc123...",
    "build_timestamp": "2025-12-14T10:00:00Z"
  }
}
```

---

## Protocol Messages Schema

### Purpose

Defines the structure of all Cloud↔Agent protocol messages.

### Current Version: 1.0.0

**Location:** `protocol_messages.schema.json`

### Message Types

#### Cloud → Agent Commands

| Type | Description | Requires Approval |
|------|-------------|-------------------|
| `REQUEST_START_RUN` | Start strategy execution | Yes |
| `REQUEST_STOP_RUN` | Stop execution | No (safety) |
| `REQUEST_PAUSE_RUN` | Pause execution | No (safety) |
| `REQUEST_UPGRADE_ARTIFACT` | Deploy new version | Yes |
| `REQUEST_UPDATE_CONFIG` | Update configuration | Yes (if trading_impacting) |
| `REQUEST_ROTATE_AGENT_SESSION` | Rotate session keys | Yes |
| `REQUEST_EXPORT_LOGS` | Export logs | Yes (data_sensitive) |

#### Agent → Cloud Messages

| Type | Description |
|------|-------------|
| `HEARTBEAT` | Agent health status |
| `TELEMETRY` | Telemetry events |
| `COMMAND_ACK` | Command acknowledgment |
| `COMMAND_RESULT` | Command execution result |
| `COMMAND_APPROVAL` | Local approval record |

### Prohibited Fields

These fields are **NEVER** allowed in any Cloud→Agent payload:

```json
{
  "prohibited_fields": [
    "side",
    "quantity",
    "price",
    "order_type",
    "target_position",
    "symbol"  // in order context
  ]
}
```

### Message Structure

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "type": "object",
  "required": ["message_type", "message_id", "timestamp"],
  "properties": {
    "message_type": {
      "type": "string",
      "enum": [
        "REQUEST_START_RUN",
        "REQUEST_STOP_RUN",
        "REQUEST_PAUSE_RUN",
        "REQUEST_UPGRADE_ARTIFACT",
        "REQUEST_UPDATE_CONFIG",
        "HEARTBEAT",
        "TELEMETRY",
        "COMMAND_ACK",
        "COMMAND_RESULT"
      ]
    },
    "message_id": {
      "type": "string",
      "format": "uuid"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "idempotency_key": {
      "type": "string"
    },
    "signature": {
      "type": "string"
    },
    "payload": {
      "type": "object"
    }
  }
}
```

### Example Messages

**REQUEST_START_RUN:**
```json
{
  "message_type": "REQUEST_START_RUN",
  "message_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-12-14T12:00:00Z",
  "idempotency_key": "start_run_abc123",
  "payload": {
    "deployment_id": "dep_xyz",
    "artifact_digest": "sha256:abc123...",
    "config_digest": "sha256:cfg456...",
    "requires_approval": true
  },
  "signature": "..."
}
```

**HEARTBEAT:**
```json
{
  "message_type": "HEARTBEAT",
  "message_id": "550e8400-e29b-41d4-a716-446655440001",
  "timestamp": "2025-12-14T12:00:30Z",
  "payload": {
    "agent_state": "RUNNING",
    "active_runs": ["run_abc"],
    "metrics": {
      "cpu_percent": 25,
      "memory_mb": 512
    }
  },
  "signature": "..."
}
```

---

## Schema Validation

### Python Validation

```python
import jsonschema
import json

# Load schema
with open("protocol_messages.schema.json") as f:
    schema = json.load(f)

# Validate message
message = {
    "message_type": "HEARTBEAT",
    "message_id": "...",
    "timestamp": "2025-12-14T12:00:00Z",
    "payload": {}
}

try:
    jsonschema.validate(message, schema)
    print("Valid")
except jsonschema.ValidationError as e:
    print(f"Invalid: {e.message}")
```

### CLI Validation

```bash
# Validate manifest
ccea-cli schema validate \
  --schema artifact_manifest.schema.json \
  --file manifest.json

# Validate message
ccea-cli schema validate \
  --schema protocol_messages.schema.json \
  --file message.json
```

### CI Validation

```yaml
# .github/workflows/schema-check.yml
name: Schema Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Validate schemas
        run: |
          pip install jsonschema
          python scripts/validate_schemas.py

      - name: Check prohibited fields
        run: |
          python scripts/check_prohibited_fields.py
```

---

## Schema Evolution

### Adding New Fields (Minor Version)

When adding a new optional field:

1. Add field with default value
2. Increment minor version
3. Update documentation
4. Both old and new agents work

**Example:**
```json
// v1.0.0
{
  "artifact_id": "...",
  "entrypoint": "..."
}

// v1.1.0 - new optional field
{
  "artifact_id": "...",
  "entrypoint": "...",
  "new_field": "default_value"  // Optional, has default
}
```

### Breaking Changes (Major Version)

When making breaking changes:

1. Increment major version
2. Maintain old version for migration period
3. Provide migration guide
4. Deprecation warning before removal

**Example:**
```json
// v1.x - old field name
{
  "strategy_code": "..."
}

// v2.0 - renamed field
{
  "entrypoint": "..."  // Renamed from strategy_code
}
```

### Migration Support

```python
def migrate_manifest(manifest: dict) -> dict:
    """Migrate manifest to latest schema version."""
    version = manifest.get("schema_version", "1.0.0")

    if version.startswith("1."):
        # v1.x is current, no migration needed
        return manifest

    if version.startswith("0."):
        # v0.x → v1.0 migration
        manifest["schema_version"] = "1.0.0"
        if "strategy_code" in manifest:
            manifest["entrypoint"] = manifest.pop("strategy_code")

    return manifest
```

---

## Security Considerations

### Prohibited Payload Validation

Schemas explicitly prohibit order-like fields:

```json
{
  "not": {
    "anyOf": [
      {"required": ["side"]},
      {"required": ["quantity"]},
      {"required": ["price"]},
      {"required": ["order_type"]},
      {"required": ["target_position"]}
    ]
  }
}
```

### Signature Verification

All messages must include valid signatures:

```python
def verify_message(message: dict, public_key: bytes) -> bool:
    """Verify message signature."""
    payload = canonical_json(message["payload"])
    signature = base64.b64decode(message["signature"])
    return verify_ed25519(payload, signature, public_key)
```

---

## Testing Schemas

### Unit Tests

```python
import pytest
from jsonschema import validate, ValidationError

def test_valid_manifest():
    manifest = {
        "schema_version": "1.0.0",
        "artifact_id": "test",
        "artifact_digest": "sha256:abc",
        "entrypoint": "main:run",
        "runtime": "python:3.12",
        "deps_lock_digest": "sha256:deps"
    }
    validate(manifest, manifest_schema)

def test_rejects_order_payload():
    message = {
        "message_type": "REQUEST_START_RUN",
        "payload": {
            "side": "BUY",  # PROHIBITED
            "quantity": 100
        }
    }
    with pytest.raises(ValidationError):
        validate(message, protocol_schema)
```

### Integration Tests

```bash
# Test schema compatibility
python -m pytest tests/ccea/test_schema_compatibility.py -v
```

---

## Document Index

| File | Description |
|------|-------------|
| `artifact_manifest.schema.json` | Artifact manifest schema |
| `protocol_messages.schema.json` | Protocol messages schema |
| `README.md` | This guide |

---

**Related Documentation:**
- [CCEA Overview](../CCEA_OVERVIEW.md)
- [Artifact Builder](../cloud/ARTIFACT_BUILDER.md)
- [Control Plane API](../cloud/CONTROL_PLANE_API.md)
