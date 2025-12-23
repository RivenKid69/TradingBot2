# Agent Installation Guide

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Prerequisites

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04, macOS 12, Windows 10 | Ubuntu 22.04, macOS 14, Windows 11 |
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8+ GB |
| Disk | 10 GB | 50+ GB (for logs/data) |
| Python | 3.10+ | 3.12 |
| Docker | 20.10+ (optional) | 24.0+ |

### Network Requirements

| Destination | Port | Protocol | Purpose |
|-------------|------|----------|---------|
| api.ccea.cloud | 443 | HTTPS | Cloud API |
| registry.ccea.cloud | 443 | HTTPS | Artifact registry |
| Broker API | varies | HTTPS | Trading |
| NTP servers | 123 | UDP | Time sync |

---

## Installation Methods

### Method 1: pip (Recommended)

```bash
# Create virtual environment
python -m venv ~/.ccea/venv
source ~/.ccea/venv/bin/activate  # Linux/macOS
# or: .\.ccea\venv\Scripts\Activate.ps1  # Windows

# Install agent
pip install ccea-agent

# Verify installation
ccea-agent --version
```

### Method 2: Docker

```bash
# Pull image
docker pull ghcr.io/ccea/agent:latest

# Create config directory
mkdir -p ~/.ccea

# Run agent
docker run -d \
  --name ccea-agent \
  --restart unless-stopped \
  -v ~/.ccea:/root/.ccea \
  -e CCEA_VAULT_KEY=$CCEA_VAULT_KEY \
  ghcr.io/ccea/agent:latest
```

### Method 3: From Source

```bash
# Clone repository
git clone https://github.com/ccea/agent.git
cd agent

# Install dependencies
pip install -e ".[dev]"

# Run agent
python -m ccea_agent
```

---

## Initial Setup

### Step 1: Run Setup Wizard

```bash
ccea-agent setup
```

This interactive wizard will:
1. Create configuration directory (`~/.ccea/`)
2. Generate encryption key for vault
3. Configure Cloud endpoint
4. Set up OS keychain (if available)

### Step 2: Add Broker Credentials

```bash
# Interactive
ccea-agent vault add-broker

# Or non-interactive
ccea-agent vault add-broker \
  --broker binance \
  --api-key $BINANCE_API_KEY \
  --api-secret $BINANCE_API_SECRET \
  --label "main-trading"
```

**Supported Brokers:**
- Binance (Spot, Futures)
- Alpaca (US Equities)
- OANDA (Forex)
- Interactive Brokers
- Deribit (Options)

### Step 3: Configure Risk Limits

```bash
# Set hard caps (CANNOT be overridden by Cloud)
ccea-agent policy set-hard-caps \
  --max-position-pct 10 \
  --max-daily-loss-pct 2 \
  --max-order-value-usd 10000 \
  --allowed-order-types LIMIT,MARKET
```

### Step 4: Enroll with Cloud

1. Log in to Cloud UI
2. Navigate to Agents → Add Agent
3. Generate enrollment token
4. Run:

```bash
ccea-agent enroll --token <enrollment_token>
```

### Step 5: Run Pre-flight Checks

```bash
ccea-agent preflight
```

This validates:
- Time synchronization
- Broker connectivity
- Credential validity
- Network connectivity
- Policy configuration

### Step 6: Start Agent

```bash
# Start in foreground (for testing)
ccea-agent start --foreground

# Start as daemon
ccea-agent start

# Check status
ccea-agent status
```

---

## Configuration

### Configuration File Location

| OS | Location |
|----|----------|
| Linux | `~/.ccea/agent.yaml` |
| macOS | `~/.ccea/agent.yaml` |
| Windows | `%USERPROFILE%\.ccea\agent.yaml` |

### Full Configuration Reference

```yaml
# ~/.ccea/agent.yaml

agent:
  # Agent ID (set automatically after enrollment)
  id: null

  # Agent version
  version: "1.0.0"

  # Label for identification
  label: "production-agent-1"

  # Log level
  log_level: INFO

  # Log file location
  log_file: ~/.ccea/logs/agent.log

  # PID file location
  pid_file: ~/.ccea/agent.pid

# Cloud connection settings
cloud:
  # Cloud API endpoint
  endpoint: https://api.ccea.cloud

  # Heartbeat interval
  heartbeat_interval_seconds: 30

  # Command poll timeout (long-poll)
  command_poll_timeout_seconds: 25

  # TLS verification (disable only for development)
  verify_tls: true

  # Retry settings
  retry:
    max_attempts: 3
    backoff_seconds: 5

# Vault (credential storage) settings
vault:
  # Backend: keychain (preferred) or encrypted_file
  backend: keychain

  # Encryption key source for file backend
  # Options: env (CCEA_VAULT_KEY), file, prompt
  encryption_key_source: env

  # File path for encrypted_file backend
  file_path: ~/.ccea/vault.enc

# Policy firewall settings
policy:
  # Hard caps (CANNOT be overridden by Cloud)
  hard_caps:
    # Maximum position size as % of portfolio
    max_position_pct: 10

    # Maximum daily loss as % of portfolio
    max_daily_loss_pct: 2

    # Maximum single order value in USD
    max_order_value_usd: 10000

    # Allowed order types
    allowed_order_types:
      - LIMIT
      - MARKET

    # Allowed symbols (empty = all allowed)
    allowed_symbols: []

    # Denied symbols (blocklist)
    denied_symbols: []

    # Maximum orders per minute
    max_orders_per_minute: 60

    # Maximum leverage
    max_leverage: 1.0

  # Auto-approve settings (USE WITH CAUTION)
  auto_approve:
    enabled: false
    # Whitelist specific change types for auto-approve
    whitelist_changes: []
    # Maximum value change that can be auto-approved
    max_auto_approve_value_change_pct: 1

# Telemetry settings
telemetry:
  # Level: AGGREGATED, DETAILED_NON_SENSITIVE, RAW_ORDER_EVENTS
  level: AGGREGATED

  # Redaction (CANNOT be disabled)
  redaction: mandatory

  # Local buffer settings
  buffer:
    path: ~/.ccea/telemetry.db
    max_size_mb: 100
    flush_interval_seconds: 60

# Sandbox settings
sandbox:
  # Enable sandbox isolation
  enabled: true

  # Backend: process or docker
  backend: process

  # Resource limits
  resource_limits:
    cpu_percent: 50
    memory_mb: 1024
    timeout_seconds: 300

  # Docker-specific settings
  docker:
    image: ghcr.io/ccea/strategy-runtime:latest
    network_mode: none

# Time synchronization
time_sync:
  enabled: true
  max_drift_ms: 1000
  check_interval_seconds: 60
  ntp_servers:
    - time.google.com
    - pool.ntp.org

# Degraded mode handling
degraded_mode:
  cloud_unreachable:
    action: continue  # continue, pause, halt
    max_duration_hours: 24
    alert_after_minutes: 5

  data_feed_invalid:
    action: halt
    grace_period_seconds: 30

  broker_errors:
    action: pause
    error_threshold: 5
    window_seconds: 60

  time_drift:
    action: halt
    max_drift_ms: 5000

# Kill switch triggers
kill_switch:
  triggers:
    max_daily_loss:
      enabled: true
      threshold_pct: 2
    broker_error_burst:
      enabled: true
      threshold: 10
      window_seconds: 60
    latency_spike:
      enabled: true
      threshold_ms: 5000
    order_spam:
      enabled: true
      threshold: 100
      window_seconds: 60
    state_divergence:
      enabled: true
    data_feed_invalid:
      enabled: true

  # Actions on kill switch
  actions:
    cancel_open_orders: true
    flatten_positions: false  # DISABLED by default (local only)
```

---

## Running as a Service

### Linux (systemd)

```ini
# /etc/systemd/system/ccea-agent.service
[Unit]
Description=CCEA Trading Agent
After=network.target

[Service]
Type=simple
User=ccea
Group=ccea
WorkingDirectory=/home/ccea
ExecStart=/home/ccea/.ccea/venv/bin/ccea-agent start --foreground
Restart=always
RestartSec=10
Environment=CCEA_VAULT_KEY_FILE=/home/ccea/.ccea/vault.key

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl enable ccea-agent
sudo systemctl start ccea-agent

# Check status
sudo systemctl status ccea-agent
```

### macOS (launchd)

```xml
<!-- ~/Library/LaunchAgents/com.ccea.agent.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.ccea.agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/Users/username/.ccea/venv/bin/ccea-agent</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

```bash
launchctl load ~/Library/LaunchAgents/com.ccea.agent.plist
```

### Windows (NSSM)

```powershell
# Install NSSM
choco install nssm

# Create service
nssm install CCEAAgent "C:\Users\username\.ccea\venv\Scripts\ccea-agent.exe" "start --foreground"
nssm set CCEAAgent AppDirectory "C:\Users\username\.ccea"
nssm set CCEAAgent DisplayName "CCEA Trading Agent"
nssm set CCEAAgent Start SERVICE_AUTO_START

# Start service
nssm start CCEAAgent
```

---

## Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ccea-agent:
    image: ghcr.io/ccea/agent:latest
    container_name: ccea-agent
    restart: unless-stopped
    volumes:
      - ~/.ccea:/root/.ccea
      - /etc/localtime:/etc/localtime:ro
    environment:
      - CCEA_VAULT_KEY=${CCEA_VAULT_KEY}
      - TZ=UTC
    # No network isolation by default for broker connectivity
    # For extra security, use network_mode: bridge and explicit rules
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Enrollment failed | Invalid/expired token | Generate new token |
| Vault unlock failed | Wrong encryption key | Check CCEA_VAULT_KEY |
| Time sync failed | NTP blocked | Allow UDP 123 outbound |
| Broker connection failed | Invalid credentials | Re-add broker credentials |
| Preflight failed | Various | Run `ccea-agent doctor` |

### Diagnostics

```bash
# Full health check
ccea-agent doctor --verbose

# Check specific component
ccea-agent doctor --check vault
ccea-agent doctor --check broker
ccea-agent doctor --check network
ccea-agent doctor --check time

# View logs
ccea-agent logs --tail 100
ccea-agent logs --level ERROR
```

### Log Locations

| Component | Location |
|-----------|----------|
| Agent log | `~/.ccea/logs/agent.log` |
| Telemetry buffer | `~/.ccea/telemetry.db` |
| Order journal | `~/.ccea/journal/` |
| Audit log | `~/.ccea/logs/audit.log` |

---

## Upgrade

### pip Upgrade

```bash
# Stop agent
ccea-agent stop

# Upgrade
pip install --upgrade ccea-agent

# Run migrations (if needed)
ccea-agent migrate

# Start agent
ccea-agent start
```

### Docker Upgrade

```bash
# Pull new image
docker pull ghcr.io/ccea/agent:latest

# Restart container
docker-compose down
docker-compose up -d
```

---

## Uninstallation

### pip Uninstall

```bash
# Stop agent
ccea-agent stop

# Disenroll from Cloud
ccea-agent disenroll

# Remove package
pip uninstall ccea-agent

# Remove configuration (optional)
rm -rf ~/.ccea
```

### Docker Uninstall

```bash
docker stop ccea-agent
docker rm ccea-agent
docker rmi ghcr.io/ccea/agent:latest
```

---

**Related Documentation:**
- [Local Vault](./LOCAL_VAULT.md)
- [Risk Controls](./RISK_CONTROLS.md)
- [Runbooks](../runbooks/)
