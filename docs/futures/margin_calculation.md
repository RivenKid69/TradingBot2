# Margin Calculation Guide

## Overview

Futures trading requires margin as collateral. CustodiaCloud supports two margin systems:

1. **Tiered Margin** - Binance-style leverage brackets for crypto perpetuals
2. **SPAN Margin** - CME's risk-based margin for regulated futures

---

## Tiered Margin System (Crypto)

### Concept

Tiered margin limits leverage based on position size. Larger positions require lower leverage, ensuring adequate collateral for big trades.

### Formula

```
Initial Margin = Position Notional × Initial Margin Rate
Maintenance Margin = Position Notional × Maintenance Margin Rate

where:
  Position Notional = abs(qty) × mark_price
  Initial Margin Rate = 1 / max_leverage_for_tier
  Maintenance Margin Rate = tier_specific_rate
```

### BTC Leverage Brackets

| Tier | Notional (USD) | Max Leverage | IMR | MMR |
|------|----------------|--------------|-----|-----|
| 1 | 0 - 50,000 | 125x | 0.8% | 0.4% |
| 2 | 50K - 250K | 100x | 1.0% | 0.5% |
| 3 | 250K - 1M | 50x | 2.0% | 1.0% |
| 4 | 1M - 5M | 20x | 5.0% | 2.5% |
| 5 | 5M - 20M | 10x | 10.0% | 5.0% |
| 6 | 20M - 50M | 5x | 20.0% | 10.0% |
| 7 | 50M - 100M | 4x | 25.0% | 12.5% |
| 8 | 100M+ | 3x | 33.3% | 16.7% |

### ETH Leverage Brackets

| Tier | Notional (USD) | Max Leverage | IMR | MMR |
|------|----------------|--------------|-----|-----|
| 1 | 0 - 10,000 | 100x | 1.0% | 0.5% |
| 2 | 10K - 100K | 75x | 1.33% | 0.65% |
| 3 | 100K - 500K | 50x | 2.0% | 1.0% |
| 4 | 500K - 2M | 25x | 4.0% | 2.0% |
| 5 | 2M - 5M | 10x | 10.0% | 5.0% |
| 6 | 5M - 10M | 5x | 20.0% | 10.0% |
| 7 | 10M+ | 3x | 33.3% | 16.7% |

### Code Example

```python
from impl_futures_margin import TieredMarginCalculator
from core_futures import FuturesPosition, PositionSide, MarginMode
from decimal import Decimal

calculator = TieredMarginCalculator()

# Create position
position = FuturesPosition(
    symbol="BTCUSDT",
    qty=Decimal("0.5"),
    entry_price=Decimal("50000"),
    side=PositionSide.LONG,
    leverage=10,
    margin_mode=MarginMode.ISOLATED,
)

# Position notional = 0.5 × 50000 = $25,000 (Tier 1)
# Max leverage = 125x, but using 10x

# Calculate initial margin
initial_margin = calculator.calculate_initial_margin(
    position=position,
    current_price=Decimal("50000"),
)
# Initial margin = $25,000 / 10 = $2,500

# Calculate maintenance margin
maint_margin = calculator.calculate_maintenance_margin(
    position=position,
    current_price=Decimal("50000"),
)
# Maintenance margin = $25,000 × 0.4% = $100

print(f"Initial Margin: ${initial_margin}")
print(f"Maintenance Margin: ${maint_margin}")
```

### Liquidation Price Calculation

```
For LONG position:
  Liquidation Price = Entry Price × (1 - (1/Leverage) + MMR)

For SHORT position:
  Liquidation Price = Entry Price × (1 + (1/Leverage) - MMR)
```

**Example:**
- Entry: $50,000
- Leverage: 10x
- MMR: 0.4%

```
Liquidation (LONG) = 50000 × (1 - 0.1 + 0.004) = $45,200
Liquidation (SHORT) = 50000 × (1 + 0.1 - 0.004) = $54,800
```

---

## SPAN Margin System (CME)

### Concept

SPAN (Standard Portfolio Analysis of Risk) is CME's margin methodology. It calculates margin based on potential worst-case portfolio loss across 16 stress scenarios.

### Components

1. **Scanning Risk** - Maximum expected loss under 16 scenarios
2. **Inter-commodity Credit** - Offset for correlated products
3. **Intra-commodity Credit** - Calendar spread credits
4. **Delivery Month Charge** - Extra margin near expiration

### Formula

```
SPAN Margin = max(Scanning Risk - Inter Credit - Intra Credit, 0) + Delivery Charge
```

### Scanning Risk Scenarios

| # | Price Move | Volatility Move |
|---|------------|-----------------|
| 1 | +1/3 range | +1 vol |
| 2 | +1/3 range | -1 vol |
| 3 | -1/3 range | +1 vol |
| 4 | -1/3 range | -1 vol |
| 5 | +2/3 range | +1 vol |
| 6 | +2/3 range | -1 vol |
| 7 | -2/3 range | +1 vol |
| 8 | -2/3 range | -1 vol |
| 9 | +1 range | +1 vol |
| 10 | +1 range | -1 vol |
| 11 | -1 range | +1 vol |
| 12 | -1 range | -1 vol |
| 13 | +extreme | 0 |
| 14 | -extreme | 0 |
| 15 | +1 range | 0 |
| 16 | -1 range | 0 |

### Product-Specific Parameters

| Product | Scanning Range | Volatility Scan | Initial Margin* |
|---------|---------------|-----------------|-----------------|
| ES | 6% | 30% | ~$13,000 |
| NQ | 8% | 35% | ~$17,000 |
| GC | 5% | 25% | ~$10,000 |
| CL | 8% | 35% | ~$8,000 |
| NG | 12% | 50% | ~$4,000 |
| 6E | 4% | 20% | ~$2,500 |
| ZN | 2% | 15% | ~$2,000 |

*Initial margin varies with market conditions

### Inter-Commodity Spread Credits

| Pair | Credit Rate | Rationale |
|------|-------------|-----------|
| ES/NQ | 50% | Correlated equity indices |
| ES/YM | 50% | S&P 500 vs Dow correlation |
| GC/SI | 35% | Precious metals correlation |
| MGC/GC | 85% | Micro/Standard same underlying |
| CL/RB/HO | 40% | Crack spread (refining) |
| ZN/ZB | 60% | Treasury notes/bonds |

### Code Example

```python
from impl_span_margin import SPANMarginCalculator
from core_futures import FuturesPosition, PositionSide, MarginMode
from decimal import Decimal

calculator = SPANMarginCalculator()

# Single position
es_position = FuturesPosition(
    symbol="ES",
    qty=Decimal("5"),
    entry_price=Decimal("4500"),
    side=PositionSide.LONG,
    leverage=1,  # N/A for SPAN
    margin_mode=MarginMode.SPAN,
)

result = calculator.calculate_margin(
    position=es_position,
    current_price=Decimal("4500"),
)

print(f"Scanning Risk: ${result.scanning_risk}")
print(f"Initial Margin: ${result.initial_margin}")
print(f"Maintenance Margin: ${result.maintenance_margin}")

# Portfolio with spread credit
nq_position = FuturesPosition(
    symbol="NQ",
    qty=Decimal("2"),
    entry_price=Decimal("15000"),
    side=PositionSide.LONG,
    leverage=1,
    margin_mode=MarginMode.SPAN,
)

portfolio_result = calculator.calculate_portfolio_margin(
    positions=[es_position, nq_position],
    prices={"ES": Decimal("4500"), "NQ": Decimal("15000")},
)

print(f"Gross Margin: ${portfolio_result.gross_margin}")
print(f"Inter-commodity Credit: ${portfolio_result.inter_commodity_credit}")
print(f"Net Portfolio Margin: ${portfolio_result.net_portfolio_margin}")
```

---

## Margin Modes

### Isolated Margin

- Each position has separate margin
- Liquidation affects only that position
- Maximum loss = position margin

```python
position = FuturesPosition(
    symbol="BTCUSDT",
    qty=Decimal("1.0"),
    entry_price=Decimal("50000"),
    side=PositionSide.LONG,
    leverage=10,
    margin_mode=MarginMode.ISOLATED,  # ← Isolated
)
# Margin for this position only: $5,000
# If liquidated, lose only $5,000
```

### Cross Margin

- All positions share account margin
- Higher capital efficiency
- Risk of total account liquidation

```python
position = FuturesPosition(
    symbol="BTCUSDT",
    qty=Decimal("1.0"),
    entry_price=Decimal("50000"),
    side=PositionSide.LONG,
    leverage=10,
    margin_mode=MarginMode.CROSS,  # ← Cross
)
# Uses available account balance as margin
# If liquidated, entire account at risk
```

### SPAN (CME Only)

- Risk-based portfolio margin
- Spread credits reduce margin
- Used for all CME products

```python
position = FuturesPosition(
    symbol="ES",
    qty=Decimal("5"),
    entry_price=Decimal("4500"),
    side=PositionSide.LONG,
    leverage=1,  # N/A
    margin_mode=MarginMode.SPAN,  # ← SPAN
)
# Margin based on SPAN calculation
# Spread credits with correlated products
```

---

## Margin Ratio

### Formula

```
Margin Ratio = (Account Equity - Unrealized PnL) / Total Maintenance Margin
```

### Status Levels

| Level | Ratio | Action |
|-------|-------|--------|
| HEALTHY | ≥ 1.5 | Normal trading |
| WARNING | 1.2 - 1.5 | Alert, monitor |
| DANGER | 1.05 - 1.2 | Reduce position |
| CRITICAL | 1.0 - 1.05 | Urgent action |
| LIQUIDATION | ≤ 1.0 | Liquidation triggered |

### Code Example

```python
from services.futures_risk_guards import FuturesMarginGuard, MarginStatus

guard = FuturesMarginGuard(
    warning_level=Decimal("1.5"),
    danger_level=Decimal("1.2"),
    critical_level=Decimal("1.05"),
)

result = guard.check_margin_ratio(
    margin_ratio=1.35,  # 135%
    account_equity=10000.0,
    total_margin_used=7407.0,
    symbol="BTCUSDT",
)

print(f"Status: {result.status}")  # MarginStatus.WARNING
print(f"Requires reduction: {result.requires_reduction}")  # False
print(f"Distance to liquidation: {result.distance_to_liquidation:.1%}")
```

---

## Margin Call Handling

### Notification Levels

| Level | Trigger | Action |
|-------|---------|--------|
| INFO | Ratio < 2.0 | Log only |
| WARNING | Ratio < 1.5 | Notification |
| MARGIN_CALL | Ratio < 1.2 | Urgent notification + reduce |
| LIQUIDATION | Ratio < 1.0 | Force close |

### Code Example

```python
from services.futures_risk_guards import MarginCallNotifier, MarginCallEvent

notifier = MarginCallNotifier(
    cooldown_seconds=300,  # 5 minutes between alerts
    callback=send_alert_function,
)

event = notifier.check_and_notify(
    margin_result=margin_result,
    position=position,
    mark_price=Decimal("50000"),
    wallet_balance=Decimal("10000"),
)

if event:
    print(f"Alert Level: {event.level}")
    print(f"Margin Ratio: {event.margin_ratio:.2%}")
    print(f"Shortfall: ${event.shortfall}")
    print(f"Recommended: {event.recommended_action}")
```

---

## Best Practices

### 1. Start Conservative

- Use lower leverage initially (5-10x)
- Increase gradually as you understand the system

### 2. Monitor Margin Ratio

- Keep ratio above 1.5 for safety
- Set alerts at multiple levels

### 3. Understand Liquidation

- Know your liquidation price before entering
- Account for funding rate impact on effective liquidation

### 4. Use Appropriate Mode

- **Isolated** for individual position risk control
- **Cross** for capital efficiency (advanced users)
- **SPAN** automatically applied for CME

### 5. Portfolio Diversification

- Benefit from inter-commodity credits (SPAN)
- Avoid concentration in correlated assets

---

## References

- Binance: "Leverage and Margin of USDⓈ-M Futures"
- CME Group: "SPAN Margin Methodology"
- CME Group: "Understanding Futures Margin"
