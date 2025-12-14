# Futures Configuration Guide

## Overview

This guide covers all configuration options for futures trading in TradingBot2.

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/config_train_futures.yaml` | Training configuration |
| `configs/config_live_futures.yaml` | Live trading configuration |
| `configs/unified_futures_risk.yaml` | Risk management settings |
| `configs/feature_flags_futures.yaml` | Feature rollout control |

---

## Training Configuration

### Basic Configuration

```yaml
# configs/config_train_futures.yaml
mode: train
asset_class: futures
futures_type: "CRYPTO_PERPETUAL"  # or "CME_EQUITY_INDEX"

data:
  timeframe: "4h"
  paths:
    - "data/futures/*.parquet"
  validation_split: 0.1

env:
  futures:
    enabled: true
    max_leverage: 10
    margin_mode: "isolated"  # isolated, cross
    initial_margin: 10000.0
    liquidation_penalty: 0.005  # 0.5% insurance fund contribution

  reward:
    gamma: 0.99
    use_log_returns: true
    funding_rate_penalty: 1.0  # Scale funding impact on reward

model:
  algo: "ppo"
  optimizer_class: AdaptiveUPGD
  params:
    use_twin_critics: true
    num_atoms: 21
    gamma: 0.99  # Must match env.reward.gamma
```

### Advanced Training Options

```yaml
# Advanced futures training options
env:
  futures:
    # Margin settings
    max_leverage: 20
    margin_mode: "isolated"
    margin_warning_ratio: 1.5
    margin_danger_ratio: 1.2

    # Liquidation simulation
    simulate_liquidation: true
    liquidation_penalty: 0.005
    insurance_fund_enabled: true

    # Funding rate simulation
    funding_rate_enabled: true
    funding_interval_hours: 8
    funding_rate_impact_on_reward: true

    # ADL simulation
    adl_enabled: true
    adl_queue_simulation: true

    # Position limits
    max_position_notional: 1000000
    max_positions_per_symbol: 1
    concentration_limit: 0.5

training:
  # Multi-futures training
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
    - "BNBUSDT"

  # Curriculum learning
  curriculum:
    enabled: true
    stages:
      - {leverage: 5, duration_steps: 100000}
      - {leverage: 10, duration_steps: 200000}
      - {leverage: 20, duration_steps: 300000}
```

---

## Live Trading Configuration

### Basic Live Configuration

```yaml
# configs/config_live_futures.yaml
futures_type: "CRYPTO_PERPETUAL"
exchange: "binance"

symbols:
  - "BTCUSDT"
  - "ETHUSDT"

paper_trading: true

# Timing
main_loop_interval_sec: 1.0
position_sync_interval_sec: 5.0
margin_check_interval_sec: 10.0
funding_check_interval_sec: 60.0

# Feature flags
enable_position_sync: true
enable_margin_monitoring: true
enable_funding_tracking: true
enable_adl_monitoring: true

# Risk settings
strict_mode: true
max_leverage: 10
max_position_value: 100000
max_total_exposure: 500000
```

### Margin Monitoring

```yaml
# Margin monitoring settings
margin:
  warning_ratio: 1.5      # 150%
  danger_ratio: 1.2       # 120%
  critical_ratio: 1.1     # 110%
  alert_cooldown_sec: 300 # 5 minutes between alerts

  # Actions at each level
  actions:
    warning:
      - "log_warning"
      - "send_notification"
    danger:
      - "log_error"
      - "send_notification"
      - "reduce_new_positions"
    critical:
      - "log_critical"
      - "send_notification"
      - "stop_new_positions"
      - "start_deleveraging"
```

### Position Synchronization

```yaml
# Position sync settings
position_sync:
  interval_sec: 5.0
  tolerance: 0.01          # 1% tolerance for qty mismatch
  auto_reconcile: false    # Manual approval for reconciliation
  max_reconcile_qty: 1000  # Max qty to auto-reconcile

  # Events to track
  track_events:
    - "position_opened"
    - "position_closed"
    - "qty_mismatch"
    - "liquidation_detected"
    - "adl_detected"
    - "funding_received"
    - "margin_call"
```

### Funding Tracking

```yaml
# Funding tracking settings
funding:
  data_dir: "data/futures/funding"
  prediction_method: "ewma"  # last, avg, ewma
  cache_ttl_sec: 300

  # Funding rate thresholds
  thresholds:
    normal: 0.0001      # < 0.01% per 8h
    warning: 0.0003     # 0.01-0.03%
    excessive: 0.0005   # > 0.03%

  # Alerts
  alert_on_excessive: true
  alert_cooldown_sec: 3600  # 1 hour
```

---

## Risk Management Configuration

### Unified Risk Configuration

```yaml
# configs/unified_futures_risk.yaml
crypto:
  # Leverage limits
  max_account_leverage: 20.0
  max_symbol_leverage: 125.0

  # Margin thresholds
  margin_warning_threshold: 1.5
  margin_danger_threshold: 1.2
  margin_critical_threshold: 1.05

  # Position limits
  max_single_symbol_pct: 0.5    # 50% max in one symbol
  max_correlated_group_pct: 0.7 # 70% max in correlated assets

  # Funding exposure
  funding_warning_threshold: 0.0001   # 0.01% per 8h
  funding_excessive_threshold: 0.0003 # 0.03% per 8h

  # ADL risk
  adl_warning_percentile: 75.0
  adl_critical_percentile: 90.0

  # Strict mode
  strict_mode: true

cme:
  # SPAN margin thresholds
  margin_warning_ratio: 1.5
  margin_danger_ratio: 1.2
  margin_critical_ratio: 1.05

  # Circuit breaker
  prevent_trades_on_halt: true
  pre_cb_warning_pct: -0.05  # -5% triggers warning

  # Settlement risk
  settlement_warn_minutes: 60
  settlement_critical_minutes: 30
  settlement_block_minutes: 15

  # Rollover risk
  rollover_warn_days: 8
  rollover_critical_days: 3
  rollover_block_days: 1

  # Position limits
  enforce_speculative_limits: true

  # Strict mode
  strict_mode: true

portfolio:
  # Cross-asset risk
  enable_correlation_tracking: true
  correlation_lookback_days: 30
  correlation_spike_threshold: 0.8

  # Aggregation
  aggregate_margin_across_types: true

# Risk profiles
profiles:
  conservative:
    crypto:
      max_account_leverage: 10.0
      margin_warning_threshold: 2.0
    cme:
      margin_warning_ratio: 2.0

  aggressive:
    crypto:
      max_account_leverage: 50.0
      margin_warning_threshold: 1.2
    cme:
      margin_warning_ratio: 1.2
```

### Risk Event Actions

```yaml
# Risk event handling
risk_events:
  # Margin events
  MARGIN_WARNING:
    log_level: "warning"
    notification: true
    action: "monitor"

  MARGIN_DANGER:
    log_level: "error"
    notification: true
    action: "reduce_position_50pct"

  MARGIN_CRITICAL:
    log_level: "critical"
    notification: true
    action: "close_position"

  # Circuit breaker events (CME)
  CIRCUIT_BREAKER_L1:
    log_level: "warning"
    notification: true
    action: "halt_new_orders"

  CIRCUIT_BREAKER_L2:
    log_level: "error"
    notification: true
    action: "reduce_exposure"

  CIRCUIT_BREAKER_L3:
    log_level: "critical"
    notification: true
    action: "emergency_close"

  # ADL events
  ADL_WARNING:
    log_level: "warning"
    notification: true
    action: "monitor"

  ADL_CRITICAL:
    log_level: "error"
    notification: true
    action: "reduce_profitable_positions"
```

---

## Feature Flags Configuration

```yaml
# configs/feature_flags_futures.yaml
features:
  # L3 execution
  futures_l3_execution:
    enabled: true
    stage: "canary"  # disabled, shadow, canary, production
    canary_symbols:
      - "BTCUSDT"
    shadow_percentage: 10

  # Unified risk guard
  unified_risk_guard:
    enabled: true
    stage: "production"

  # Liquidation cascade simulation
  liquidation_cascade:
    enabled: true
    stage: "production"

  # Insurance fund
  insurance_fund:
    enabled: true
    stage: "production"

  # ADL simulation
  adl_simulation:
    enabled: true
    stage: "canary"
    canary_symbols:
      - "BTCUSDT"
      - "ETHUSDT"

  # CME circuit breaker
  cme_circuit_breaker:
    enabled: true
    stage: "production"

# Rollout stages
stages:
  disabled:
    description: "Feature completely disabled"

  shadow:
    description: "Feature runs but results not used"
    log_shadow_results: true

  canary:
    description: "Feature enabled for specific symbols"

  production:
    description: "Feature fully enabled"
```

---

## Environment Variables

```bash
# Binance Futures
BINANCE_FUTURES_API_KEY=...
BINANCE_FUTURES_API_SECRET=...
BINANCE_FUTURES_TESTNET=true  # Use testnet

# Interactive Brokers (CME)
IB_HOST=127.0.0.1
IB_PORT=7497  # 7497=paper, 7496=live
IB_CLIENT_ID=1
IB_ACCOUNT=...

# Notifications
SLACK_WEBHOOK_URL=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...

# Data storage
FUTURES_DATA_DIR=data/futures
FUTURES_FUNDING_CACHE_DIR=data/futures/funding
```

---

## Command-Line Arguments

### Training

```bash
# Basic futures training
python train_model_multi_patch.py \
    --config configs/config_train_futures.yaml

# With custom leverage
python train_model_multi_patch.py \
    --config configs/config_train_futures.yaml \
    --futures-leverage 10

# With specific symbols
python train_model_multi_patch.py \
    --config configs/config_train_futures.yaml \
    --symbols BTCUSDT ETHUSDT

# CME futures training
python train_model_multi_patch.py \
    --config configs/config_train_futures.yaml \
    --futures-type CME_EQUITY_INDEX \
    --symbols ES NQ
```

### Live Trading

**CCEA Architecture Note**: In production, live trading runs via the local Agent daemon.
See `docs/agent/INSTALLATION.md` for setup. The commands below are for development/testing only.

**Production (via Agent daemon):**
```bash
# See docs/agent/INSTALLATION.md for full Agent setup
python -m packages.agent.daemon.agentd --config configs/agent_futures.yaml
```

**Development/Testing Only:**
```bash
# Basic live trading (paper) - DEVELOPMENT ONLY
python script_live.py \
    --config configs/config_live_futures.yaml \
    --paper --dry-run

# With specific risk profile - DEVELOPMENT ONLY
python script_live.py \
    --config configs/config_live_futures.yaml \
    --risk-profile conservative --dry-run
```

### Backtest

```bash
# Futures backtest
python script_backtest.py \
    --config configs/config_backtest_futures.yaml \
    --start 2024-01-01 \
    --end 2024-12-31

# With L3 simulation
python script_backtest.py \
    --config configs/config_backtest_futures.yaml \
    --execution-level L3
```

---

## Default Values

### Crypto Perpetual Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_leverage` | 125 | Maximum leverage allowed |
| `margin_mode` | `isolated` | Default margin mode |
| `funding_interval_hours` | 8 | Funding payment interval |
| `liquidation_penalty` | 0.5% | Insurance fund contribution |
| `maker_fee` | 0.02% | Maker fee rate |
| `taker_fee` | 0.04% | Taker fee rate |

### CME Futures Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `margin_mode` | `span` | Always SPAN for CME |
| `settlement_time` | Product-specific | Daily settlement time |
| `fee_per_contract` | Product-specific | Fixed per-contract fee |
| `circuit_breaker_enabled` | `true` | Rule 80B enabled |
| `velocity_logic_enabled` | `true` | Fat-finger protection |
