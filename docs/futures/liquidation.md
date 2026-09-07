# Liquidation Engine Guide

## Overview

Liquidation occurs when a trader's margin is insufficient to maintain their position. This guide covers liquidation mechanics for both crypto perpetual futures and CME futures.

---

## Liquidation Trigger

### Crypto Perpetuals

Liquidation triggers when:

```
Mark Price reaches Liquidation Price

where:
  Liquidation Price = Entry × (1 ± (1/Leverage - MMR))
  + for SHORT, - for LONG
  MMR = Maintenance Margin Rate
```

### CME Futures

Liquidation triggers when:

```
Account Equity ≤ Maintenance Margin (SPAN)

The broker initiates margin call → forced liquidation if not met
```

---

## Liquidation Price Calculation

### Isolated Margin

```python
from impl_futures_margin import TieredMarginCalculator
from core_futures import FuturesPosition, PositionSide, MarginMode
from decimal import Decimal

calculator = TieredMarginCalculator()

position = FuturesPosition(
    symbol="BTCUSDT",
    qty=Decimal("1.0"),
    entry_price=Decimal("50000"),
    side=PositionSide.LONG,
    leverage=10,
    margin_mode=MarginMode.ISOLATED,
)

liq_price = calculator.calculate_liquidation_price(
    position=position,
    wallet_balance=Decimal("5000"),  # Isolated margin
)

print(f"Liquidation Price: ${liq_price}")
# For 10x LONG: ~$45,200 (entry × (1 - 1/10 + MMR))
```

### Cross Margin

With cross margin, liquidation price depends on total account equity:

```python
liq_price = calculator.calculate_liquidation_price_cross(
    positions=[position1, position2],  # All positions
    total_wallet_balance=Decimal("50000"),
    unrealized_pnl=Decimal("-2000"),
)
```

---

## Liquidation Process

### Crypto Perpetuals (Binance)

1. **Price crosses liquidation level**
2. **Liquidation engine takes over position**
3. **Position closed in market**
4. **Insurance fund handles shortfall**
5. **If insurance depleted → ADL triggered**

### CME Futures

1. **Margin ratio falls below maintenance**
2. **Margin call issued**
3. **Trader has time to add funds**
4. **If not met → forced liquidation**
5. **No insurance fund; broker absorbs loss**

---

## Liquidation Cascade Simulation

### Concept

Large liquidations cause price impact, triggering more liquidations in a cascade effect.

### Kyle Price Impact Model

```
ΔP = λ × sign(x) × |x|

where:
  λ = price impact coefficient
  x = liquidation volume
```

### Cascade Mechanics

1. **Initial liquidation** causes price move
2. **Price move triggers more liquidations**
3. **Each wave is dampened by cascade_decay**
4. **Process continues until no more triggers**

### Code Example

```python
from lob.futures_extensions import (
    LiquidationCascadeSimulator,
    create_cascade_simulator,
    CascadePhase,
)

simulator = create_cascade_simulator(
    price_impact_coef=0.5,  # Kyle λ
    cascade_decay=0.7,      # Wave dampening
    max_waves=5,            # Max iterations
)

result = simulator.simulate_cascade(
    initial_liquidation_volume=1_000_000,  # $1M
    market_price=50000.0,
    adv=500_000_000,  # $500M daily volume
)

print(f"Total waves: {len(result.waves)}")
print(f"Total liquidated: ${result.total_liquidated_volume:,.0f}")
print(f"Price impact: {result.total_price_impact_bps:.2f} bps")
print(f"Final phase: {result.final_phase}")

# Wave details
for i, wave in enumerate(result.waves):
    print(f"  Wave {i+1}: ${wave.volume:,.0f}, impact: {wave.price_impact_bps:.2f} bps")
```

### Configuration

```python
from execution_providers_futures_l3 import FuturesL3Config

config = FuturesL3Config(
    # Cascade parameters
    price_impact_coef=0.5,
    cascade_decay=0.7,
    max_cascade_waves=5,
)
```

---

## Insurance Fund

### Purpose

The insurance fund absorbs losses when liquidation fill price is worse than bankruptcy price.

### Mechanics

```
If fill_price < bankruptcy_price (for long):
    Shortfall = (bankruptcy_price - fill_price) × qty
    Insurance Fund pays shortfall

If fill_price > bankruptcy_price (for long):
    Contribution = (fill_price - bankruptcy_price) × qty
    Goes to Insurance Fund
```

### Code Example

```python
from lob.futures_extensions import (
    InsuranceFundManager,
    create_insurance_fund,
    InsuranceFundState,
)

fund = create_insurance_fund(initial_balance=10_000_000)

# Process liquidation
result = fund.process_liquidation(
    liquidation_info=liq_order,
    fill_price=49500.0,
)

print(f"Fill price: ${result.fill_price}")
print(f"Contribution: ${result.contribution:.2f}")  # If profit
print(f"Payout: ${result.payout:.2f}")  # If loss
print(f"Fund balance: ${fund.get_state().current_balance:,.0f}")
print(f"Utilization: {fund.get_state().utilization_pct:.1%}")
```

### Fund Depletion

When insurance fund is depleted:

1. **ADL (Auto-Deleveraging) activates**
2. **Profitable traders are forcibly deleveraged**
3. **Losses socialized to winning traders**

---

## Auto-Deleveraging (ADL)

### Trigger

ADL triggers when:

- Insurance fund is depleted
- Liquidation cannot be filled at bankruptcy price

### Ranking Formula

```
ADL Score = PnL% × Leverage

Higher score = higher priority for deleveraging
```

### ADL Queue

```python
from lob.futures_extensions import (
    ADLQueueManager,
    create_adl_manager,
    ADLRank,
)

adl_manager = create_adl_manager()

# Build queue from positions
positions = [
    {"address": "user1", "pnl_pct": 0.15, "leverage": 20, "side": "long", "size": 1000},
    {"address": "user2", "pnl_pct": 0.10, "leverage": 10, "side": "long", "size": 2000},
    {"address": "user3", "pnl_pct": 0.05, "leverage": 5, "side": "long", "size": 3000},
]

adl_manager.build_queue(positions, side="long")

# Get candidates for deleveraging
candidates = adl_manager.get_adl_candidates(
    side="long",
    required_amount=500,
)

for candidate in candidates:
    print(f"User: {candidate.address}")
    print(f"  Score: {candidate.adl_score:.2f}")
    print(f"  Rank: {candidate.rank}")
    print(f"  Amount to deleverage: {candidate.deleverage_amount}")
```

### ADL Indicator

| Rank | Light | Risk Level |
|------|-------|------------|
| 1 (bottom 20%) | None | Very Low |
| 2 (20-40%) | One | Low |
| 3 (40-60%) | Two | Medium |
| 4 (60-80%) | Three | High |
| 5 (top 20%) | All Four | Very High |

---

## Risk Guards

### ADL Risk Guard

```python
from services.futures_risk_guards import ADLRiskGuard, ADLRiskLevel

guard = ADLRiskGuard(
    warning_percentile=75.0,
    critical_percentile=90.0,
)

result = guard.check_adl_risk(
    position_pnl_percentile=85.0,  # Top 15%
    position_leverage_percentile=80.0,  # Top 20%
)

print(f"ADL Level: {result.level}")  # ADLRiskLevel.HIGH
print(f"ADL Score: {result.adl_score:.1f}")
print(f"Rank: {result.rank}")
print(f"Action: {result.recommended_action}")
```

### ADL Risk Levels

| Level | Percentile | Action |
|-------|------------|--------|
| **SAFE** | < 50% | Normal trading |
| **WARNING** | 50-75% | Monitor |
| **HIGH** | 75-90% | Consider reducing |
| **CRITICAL** | > 90% | Reduce immediately |

---

## Liquidation in Training

### Environment Configuration

```yaml
env:
  futures:
    # Liquidation settings
    simulate_liquidation: true
    liquidation_penalty: 0.005  # 0.5% to insurance fund

    # Cascade simulation
    cascade_enabled: true
    cascade_price_impact_coef: 0.5
    cascade_decay: 0.7
    max_cascade_waves: 5

    # Insurance fund
    insurance_fund_enabled: true
    insurance_fund_initial: 10000000

    # ADL
    adl_enabled: true
    adl_queue_simulation: true
```

### Reward Impact

```python
# Liquidation reward (very negative)
if is_liquidated:
    reward = -10.0  # Heavy penalty

# Near-liquidation penalty
if margin_ratio < 1.1:
    reward *= (margin_ratio - 1.0) / 0.1  # Scale down reward
```

---

## Liquidation Fee Structure

### Binance USDT-M

| Component | Rate |
|-----------|------|
| Insurance Fund Contribution | 0.5% of notional |
| Taker Fee (liquidation) | 0.04% (normal taker) |

### CME

| Component | Rate |
|-----------|------|
| No insurance fund | Broker absorbs |
| Normal fees apply | Per-contract |

---

## Avoiding Liquidation

### 1. Use Lower Leverage

```
Leverage | Distance to Liquidation (LONG)
---------|-----------------------------
10x      | ~10% price drop
5x       | ~20% price drop
2x       | ~50% price drop
```

### 2. Set Stop-Loss

```python
# Calculate stop-loss above liquidation
liquidation_price = 45000
safety_buffer = 0.05  # 5%
stop_loss = liquidation_price * (1 + safety_buffer)
# stop_loss = $47,250
```

### 3. Monitor Margin Ratio

```python
from services.futures_risk_guards import FuturesMarginGuard

guard = FuturesMarginGuard()

# Set up real-time monitoring
if margin_ratio < guard.warning_level:
    alert("Margin warning - consider reducing position")
```

### 4. Reduce Position Size

```python
# Dynamic position sizing based on volatility
max_position = base_position / (realized_vol / target_vol)
```

### 5. Use Isolated Margin

- Limits loss to position margin
- Prevents account-wide liquidation

---

## Simulation Accuracy

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Liquidation Timing | < 1 bar | Trigger to liquidation |
| Price Impact | < 5% error | Cascade impact accuracy |
| ADL Queue | > 95% accuracy | Ranking correctness |

### Validation

```python
# Run validation tests
pytest tests/test_futures_validation.py::TestLiquidationEngineValidation -v
```

---

## References

- Binance: "Liquidation Protocol"
- Binance: "Insurance Fund"
- Binance: "Auto-Deleveraging (ADL)"
- Kyle (1985): "Continuous Auctions and Insider Trading"
- Academic: Capponi et al. (2021) "Liquidation Cascades in Crypto"
