# Risk Controls: Policy Firewall & Hard Caps

> **Version**: 1.0.0 | **Last Updated**: 2025-12-16

## Overview

The Agent's Policy Firewall enforces risk controls locally. **Hard caps CANNOT be overridden by Cloud** - they are absolute local limits that protect the user.

## Security Design Commitments

```
Risk Controls DESIGN COMMITMENTS (enforced at architecture level):
  - Hard caps CANNOT be raised by Cloud (ever)
  - Cloud config cannot exceed local limits
  - All orders pass through policy firewall
  - Kill switch triggers halt automatically
  - Reconciliation prevents order duplication
```

---

## Config Layering

Risk configuration follows strict priority:

```
Priority (highest to lowest):
1. Local Hard Caps         ← CANNOT be overridden
2. Local Policy Firewall   ← User-defined limits
3. Artifact risk_profile_suggested  ← Strategy suggestions
4. Cloud Config            ← Remote configuration
5. Defaults                ← System defaults
```

**Example:**
```
Cloud suggests:     max_position_pct: 20%
Artifact suggests:  max_position_pct: 15%
Local policy:       max_position_pct: 10%
Hard cap:           max_position_pct: 10%

Result: max_position_pct = 10% (hard cap enforced)
```

---

## Hard Caps

Hard caps are **absolute limits** that cannot be exceeded under any circumstances.

### Configuration

```yaml
# ~/.ccea/agent.yaml
policy:
  hard_caps:
    # Position limits
    max_position_pct: 10           # Max 10% of portfolio per position
    max_total_exposure_pct: 100    # Max 100% total exposure

    # Loss limits
    max_daily_loss_pct: 2          # Max 2% daily loss
    max_drawdown_pct: 10           # Max 10% drawdown

    # Order limits
    max_order_value_usd: 10000     # Max $10k per order
    max_orders_per_minute: 60      # Max 60 orders/minute

    # Leverage
    max_leverage: 1.0              # No leverage (1x only)

    # Order types
    allowed_order_types:
      - LIMIT
      - MARKET
    denied_order_types:
      - STOP_LOSS              # Disabled if not trusted

    # Symbols
    allowed_symbols: []            # Empty = all allowed
    denied_symbols:
      - LEVERAGED_TOKEN_*      # Glob patterns supported

    # Time restrictions
    trading_hours:
      enabled: false
      timezone: UTC
      start: "09:30"
      end: "16:00"
      days: [1, 2, 3, 4, 5]    # Mon-Fri
```

### Setting Hard Caps via CLI

```bash
# Set position limit
ccea-agent policy set-hard-cap --max-position-pct 10

# Set loss limit
ccea-agent policy set-hard-cap --max-daily-loss-pct 2

# Set order restrictions
ccea-agent policy set-hard-cap \
  --allowed-order-types LIMIT,MARKET \
  --max-order-value-usd 10000

# Set symbol restrictions
ccea-agent policy set-hard-cap --denied-symbols "LEVERAGED_*,*UP,*DOWN"

# View current hard caps
ccea-agent policy show-hard-caps
```

### Hard Cap Enforcement

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HARD CAP ENFORCEMENT                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Strategy generates Intent:                                                  │
│    symbol: BTCUSDT                                                          │
│    side: BUY                                                                 │
│    quantity: 1.0 BTC                                                        │
│    price: $50,000                                                           │
│    order_value: $50,000                                                     │
│                                                                              │
│  Hard Cap Check:                                                            │
│    max_order_value_usd: $10,000                                            │
│    $50,000 > $10,000                                                        │
│                                                                              │
│  Result: ORDER REJECTED                                                     │
│    Reason: "Exceeds hard cap max_order_value_usd ($50,000 > $10,000)"      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Policy Firewall

The Policy Firewall adds additional configurable rules on top of hard caps.

### Pre-Trade Checks

| Check | Description | Action |
|-------|-------------|--------|
| Position limit | Single position size | Reject/Scale |
| Exposure limit | Total exposure | Reject |
| Order value | Single order value | Reject/Scale |
| Order rate | Orders per minute | Throttle |
| Symbol allowed | Symbol in allowlist | Reject |
| Order type | Order type allowed | Reject |
| Leverage | Leverage within limit | Reject |
| Price deviation | Price vs market | Reject |
| Time restrictions | Within trading hours | Reject |

### Configuration

```yaml
policy:
  firewall:
    # Pre-trade validation
    pre_trade:
      enabled: true
      checks:
        - position_limit
        - exposure_limit
        - order_value
        - order_rate
        - symbol_allowed
        - order_type
        - leverage
        - price_deviation
        - time_restrictions

    # Position limits
    position_limits:
      default_max_pct: 5        # Default per position
      per_symbol:               # Symbol-specific overrides
        BTCUSDT: 10
        ETHUSDT: 8

    # Order rate limiting
    rate_limits:
      orders_per_minute: 30
      orders_per_second: 2
      burst_limit: 10

    # Price deviation check
    price_deviation:
      max_deviation_pct: 1      # Max 1% from market
      reference: mid_price       # mid_price, last_trade, bid, ask

    # Scaling behavior (when order exceeds limit)
    scaling:
      enabled: true
      mode: scale_down          # scale_down, reject
```

### Policy Modes

| Mode | Behavior |
|------|----------|
| `enforce` | Reject orders that violate policy |
| `scale` | Scale orders down to fit within limits |
| `warn` | Allow but log warning |
| `disabled` | No enforcement (NOT recommended) |

```yaml
policy:
  firewall:
    mode: enforce  # enforce, scale, warn, disabled
```

---

## Kill Switch

The Kill Switch provides emergency halt functionality.

### Triggers

| Trigger | Description | Default |
|---------|-------------|---------|
| `max_daily_loss` | Daily loss exceeds threshold | 2% |
| `max_drawdown` | Drawdown exceeds threshold | 10% |
| `broker_error_burst` | Multiple broker errors | 10 in 60s |
| `latency_spike` | Execution latency too high | >5000ms |
| `order_spam` | Too many orders | 100 in 60s |
| `state_divergence` | Local vs broker mismatch | Enabled |
| `data_feed_invalid` | Market data issues | Enabled |
| `manual_trigger` | User-initiated halt | N/A |

### Configuration

```yaml
kill_switch:
  enabled: true

  triggers:
    max_daily_loss:
      enabled: true
      threshold_pct: 2
    max_drawdown:
      enabled: true
      threshold_pct: 10
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

  actions:
    cancel_open_orders: true
    flatten_positions: false    # DISABLED by default
    notify_cloud: true
```

### Kill Switch Actions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KILL SWITCH ACTIVATED                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Trigger: max_daily_loss                                                    │
│  Reason: "Daily loss -2.5% exceeds threshold -2%"                          │
│                                                                              │
│  Actions:                                                                   │
│    1. ✅ Halt strategy execution                                            │
│    2. ✅ Cancel all open orders (5 orders cancelled)                        │
│    3. ❌ Flatten positions (DISABLED - manual only)                         │
│    4. ✅ Report to Cloud (telemetry)                                        │
│    5. ✅ Log halt reason locally                                            │
│                                                                              │
│  State: HALTED                                                              │
│  Recovery: Manual acknowledgment required                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Manual Kill Switch

```bash
# Trigger kill switch manually
ccea-agent kill-switch trigger --reason "Manual halt for review"

# With position flatten (if enabled in config)
ccea-agent kill-switch trigger --reason "Emergency" --flatten

# Acknowledge and recover
ccea-agent kill-switch acknowledge
```

---

## Reconciliation

Reconciliation ensures consistency between local state and broker state.

### Position Reconciliation

```bash
# Run position reconciliation
ccea-agent reconcile positions

# Output:
# SYMBOL     LOCAL    BROKER   STATUS
# BTCUSDT    0.5      0.5      ✅ Match
# ETHUSDT    2.0      2.0      ✅ Match
# SOLUSDT    10.0     10.0     ✅ Match
```

### Order Reconciliation

```bash
# Run order reconciliation
ccea-agent reconcile orders

# Output:
# ORDER_ID          LOCAL_STATUS   BROKER_STATUS   ACTION
# ord_abc123        PENDING        FILLED          ✅ Updated
# ord_def456        PENDING        CANCELLED       ✅ Updated
# ord_ghi789        PENDING        PENDING         ✅ Match
```

### Idempotent Order IDs

All orders use deterministic client order IDs:

```python
# Client order ID generation
client_order_id = sha256(
    f"{agent_id}:{strategy_id}:{run_id}:{intent_hash}:{sequence}"
)[:16]

# Example: "ccea_abc123def456"
```

This ensures:
- No duplicate orders on retry
- Order tracking across restarts
- Reconciliation mapping

### Startup Reconciliation

On agent restart:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STARTUP RECONCILIATION                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Load local journal                                                      │
│  2. Fetch open orders from broker                                          │
│  3. Fetch current positions from broker                                    │
│  4. Compare and reconcile:                                                 │
│     - Unknown orders? → Cancel                                             │
│     - Missing fills? → Update journal                                      │
│     - Position mismatch? → HALT (manual intervention)                      │
│  5. Resume if clean, HALT if issues                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Testing Policies

### Policy Simulation

```bash
# Test policy against hypothetical order
ccea-agent policy test \
  --symbol BTCUSDT \
  --side BUY \
  --quantity 1.0 \
  --price 50000

# Output:
# Pre-trade checks:
#   ✅ Symbol allowed
#   ✅ Order type allowed (LIMIT)
#   ❌ Order value ($50,000) exceeds max ($10,000)
#   ✅ Position limit OK
#   ✅ Rate limit OK
#
# Result: REJECTED
# Reason: Order value exceeds max_order_value_usd
```

### Dry Run Mode

```bash
# Run strategy in dry-run mode (orders logged but not sent)
ccea-agent start --dry-run

# All orders are:
# - Validated against policy
# - Logged to journal
# - NOT sent to broker
```

---

## CLI Reference

```bash
# Hard caps
ccea-agent policy show-hard-caps
ccea-agent policy set-hard-cap --KEY VALUE

# Policy firewall
ccea-agent policy show
ccea-agent policy test --symbol X --side Y --quantity Z

# Kill switch
ccea-agent kill-switch status
ccea-agent kill-switch trigger --reason "..."
ccea-agent kill-switch acknowledge

# Reconciliation
ccea-agent reconcile positions
ccea-agent reconcile orders
ccea-agent reconcile all
```

---

## Configuration Reference

```yaml
# Full policy configuration
policy:
  # Hard caps (cannot be exceeded)
  hard_caps:
    max_position_pct: 10
    max_total_exposure_pct: 100
    max_daily_loss_pct: 2
    max_drawdown_pct: 10
    max_order_value_usd: 10000
    max_orders_per_minute: 60
    max_leverage: 1.0
    allowed_order_types: [LIMIT, MARKET]
    denied_order_types: []
    allowed_symbols: []
    denied_symbols: []

  # Policy firewall
  firewall:
    mode: enforce
    pre_trade:
      enabled: true
    position_limits:
      default_max_pct: 5
    rate_limits:
      orders_per_minute: 30
    price_deviation:
      max_deviation_pct: 1
    scaling:
      enabled: true
      mode: scale_down

# Kill switch
kill_switch:
  enabled: true
  triggers:
    max_daily_loss:
      enabled: true
      threshold_pct: 2
    broker_error_burst:
      enabled: true
      threshold: 10
      window_seconds: 60
  actions:
    cancel_open_orders: true
    flatten_positions: false

# Reconciliation
reconciliation:
  on_startup: true
  periodic_interval_minutes: 5
  on_state_divergence: halt
```

---

**Related Documentation:**
- [Approvals](./APPROVALS.md)
- [Degraded Modes](./DEGRADED_MODES.md)
- [Runbooks](../runbooks/)
