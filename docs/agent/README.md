# Agent Zone Documentation

> **Version**: 1.1.0 | **Last Updated**: 2025-12-16
>
> **Reference**: This document aligns with `Design Doc CCEA Cloud.txt` (canonical source)

## Overview

The Agent Zone handles all live trading operations locally on the user's infrastructure. It is designed with complete autonomy and security - secrets never leave the agent, and all orders are created and sent locally.

### Design Doc Reference (§4.2)

Agent components per Design Doc:
- **Local Vault** - Keychain/encrypted file storage for broker API keys
- **Approval UI** - CLI/GUI for local approval of TRADING_IMPACTING changes
- **Policy Firewall** - Layered config + hard caps enforcement
- **Runner** - Live loop: strategy→intent→risk→order
- **Reconciliation** - Position sync with idempotent order IDs
- **Kill Switch** - Local halt mechanism (17 halt reason types)
- **Telemetry Buffer** - SQLite + redaction middleware

## Security Guarantees

```
Agent Zone GUARANTEES:
  - Secrets (API keys) NEVER leave the local environment
  - Orders are created and sent ONLY locally
  - Local hard caps CANNOT be overridden by Cloud
  - Trading-impacting changes require LOCAL approval
  - Agent can operate independently if Cloud is unreachable
  - All telemetry is redacted before transmission
```

## Components

### Agent Daemon (`packages/agent/daemon/`)

The core daemon that manages the agent lifecycle:

| Component | Purpose |
|-----------|---------|
| `agentd.py` | Main daemon with lifecycle management |
| `kill_switch.py` | Emergency halt with 17 halt reason types |
| `preflight.py` | Pre-flight validation (14 check types) |
| `degraded_mode.py` | Safe degradation handling (9 modes) |
| `telemetry_buffer.py` | SQLite-based durable telemetry |
| `time_sync.py` | NTP verification |
| `sandbox.py` | Process/container isolation |
| `keychain.py` | OS keychain integration |

See: [INSTALLATION.md](./INSTALLATION.md)

### Local Vault (`packages/agent/vault/`)

Secure credential storage:

- OS keychain integration (macOS/Linux/Windows)
- Encrypted file fallback
- Multi-account support
- Automatic redaction

See: [LOCAL_VAULT.md](./LOCAL_VAULT.md)

### Approval System (`packages/agent/approval/`)

Local approval for trading-impacting changes:

- CLI approval interface
- GUI approval (optional)
- Evidence hash generation
- Approval audit trail

See: [APPROVALS.md](./APPROVALS.md)

### Policy Firewall (`packages/agent/policy/`)

Risk controls and limits:

- Hard caps (cannot be overridden)
- Position limits
- Order type restrictions
- Symbol restrictions

See: [RISK_CONTROLS.md](./RISK_CONTROLS.md)

### Execution (`packages/agent/execution/`)

Order creation and execution:

- Intent to Order conversion
- Broker connector integration
- Order journaling
- Reconciliation

### Reconciliation (`packages/agent/reconciliation/`)

State consistency:

- Position reconciliation
- Order reconciliation
- Client order ID generation
- Restart recovery

---

## Architecture

```
packages/agent/
├── __init__.py              # Security guarantees documented
├── daemon/                  # Agent daemon
│   ├── agentd.py           # Main daemon
│   ├── kill_switch.py      # Kill switch
│   ├── preflight.py        # Pre-flight checks
│   ├── degraded_mode.py    # Degraded mode handling
│   ├── telemetry_buffer.py # Durable telemetry
│   ├── time_sync.py        # Time sync
│   ├── sandbox.py          # Sandbox isolation
│   └── keychain.py         # OS keychain
├── vault/                   # Credential storage
│   ├── credential_vault.py # Main vault
│   └── keychain.py         # Keychain integration
├── approval/                # Approval system
│   ├── local_approval.py   # Approval UI/CLI
│   └── evidence.py         # Evidence generation
├── policy/                  # Policy firewall
│   ├── firewall.py         # Policy enforcement
│   ├── hard_caps.py        # Hard caps
│   └── risk_checker.py     # Risk validation
├── execution/               # Order execution
│   ├── intent_processor.py # Intent to Order
│   ├── broker_connector.py # Broker integration
│   └── order_journal.py    # Order persistence
├── reconciliation/          # State reconciliation
│   ├── position_reconciler.py
│   └── order_reconciler.py
├── runner/                  # Live runner
│   └── live.py             # Live trading loop
└── telemetry/               # Telemetry
    └── redactor.py         # Mandatory redaction
```

---

## Agent States

```
┌───────────┐
│  CREATED  │
└─────┬─────┘
      │ initialize
      ▼
┌───────────────┐
│ INITIALIZING  │
└───────┬───────┘
        │ enroll
        ▼
┌───────────────┐
│   ENROLLING   │
└───────┬───────┘
        │ enrolled
        ▼
┌───────────────┐
│     IDLE      │◀────────────────┐
└───────┬───────┘                 │
        │ start                   │ stop
        ▼                         │
┌───────────────┐    pause   ┌────┴────┐
│    RUNNING    │◀──────────▶│ PAUSED  │
└───────┬───────┘            └─────────┘
        │
        │ kill_switch / error
        ▼
┌───────────────┐
│    HALTED     │
└───────┬───────┘
        │ acknowledge
        ▼
┌───────────────┐
│   STOPPED     │
└───────────────┘
```

---

## Quick Start

### 1. Install Agent

```bash
# Using pip
pip install ccea-agent

# Using Docker
docker pull ghcr.io/ccea/agent:latest
```

### 2. Configure Credentials

```bash
# Interactive setup
ccea-agent setup

# Or manual configuration
ccea-agent vault add-broker \
  --broker binance \
  --api-key $API_KEY \
  --api-secret $API_SECRET
```

### 3. Enroll with Cloud

```bash
# Get enrollment token from Cloud UI
ccea-agent enroll --token <enrollment_token>
```

### 4. Configure Risk Limits

```bash
# Set hard caps (cannot be overridden)
ccea-agent policy set-hard-caps \
  --max-position-pct 10 \
  --max-daily-loss-pct 2 \
  --allowed-order-types LIMIT,MARKET
```

### 5. Start Agent

```bash
# Start daemon
ccea-agent start

# Or with Docker
docker run -d \
  -v ~/.ccea:/root/.ccea \
  ghcr.io/ccea/agent:latest
```

---

## Configuration Reference

### Agent Config (`~/.ccea/agent.yaml`)

```yaml
agent:
  id: null  # Set after enrollment
  version: "1.0.0"

cloud:
  endpoint: https://api.ccea.cloud
  heartbeat_interval_seconds: 30
  command_poll_timeout_seconds: 25

vault:
  backend: keychain  # or: encrypted_file
  encryption_key_source: env  # CCEA_VAULT_KEY

policy:
  hard_caps:
    max_position_pct: 10
    max_daily_loss_pct: 2
    max_order_value_usd: 10000
    allowed_order_types: [LIMIT, MARKET]
    allowed_symbols: []  # Empty = all allowed
    denied_symbols: []
  auto_approve:
    enabled: false
    whitelist_changes: []

telemetry:
  level: AGGREGATED  # AGGREGATED, DETAILED, RAW
  redaction: mandatory  # Cannot be disabled
  buffer_path: ~/.ccea/telemetry.db

sandbox:
  enabled: true
  backend: process  # or: docker
  resource_limits:
    cpu_percent: 50
    memory_mb: 1024

time_sync:
  enabled: true
  max_drift_ms: 1000
  ntp_servers:
    - time.google.com
    - pool.ntp.org

degraded_mode:
  cloud_unreachable:
    action: continue  # or: pause, halt
    max_duration_hours: 24
  data_feed_invalid:
    action: halt
  broker_errors:
    action: pause
    threshold: 5
```

---

## CLI Reference

```bash
# Agent management
ccea-agent start               # Start daemon
ccea-agent stop                # Stop daemon
ccea-agent status              # Show status
ccea-agent restart             # Restart daemon

# Enrollment
ccea-agent enroll --token <t>  # Enroll with Cloud
ccea-agent disenroll           # Remove enrollment

# Vault management
ccea-agent vault list          # List credentials
ccea-agent vault add-broker    # Add broker credentials
ccea-agent vault remove        # Remove credentials
ccea-agent vault rotate        # Rotate encryption key

# Policy management
ccea-agent policy show         # Show current policy
ccea-agent policy set-hard-caps # Set hard caps
ccea-agent policy test         # Test policy against intent

# Approvals
ccea-agent approve --request <id>  # Approve request
ccea-agent reject --request <id>   # Reject request
ccea-agent pending                 # List pending approvals

# Diagnostics
ccea-agent doctor              # Run health checks
ccea-agent logs                # Show logs
ccea-agent preflight           # Run pre-flight checks
```

---

## Document Index

| Document | Description |
|----------|-------------|
| [INSTALLATION.md](./INSTALLATION.md) | Installation and setup guide |
| [LOCAL_VAULT.md](./LOCAL_VAULT.md) | Credential storage and management |
| [APPROVALS.md](./APPROVALS.md) | Local approval system |
| [RISK_CONTROLS.md](./RISK_CONTROLS.md) | Policy firewall and hard caps |
| [DEGRADED_MODES.md](./DEGRADED_MODES.md) | Safe degradation handling |

---

**Related Documentation:**
- [CCEA Overview](../CCEA_OVERVIEW.md)
- [Cloud Documentation](../cloud/README.md)
- [Runbooks](../runbooks/)
