# Funding Rates Guide

## Overview

Funding rates are periodic payments between long and short traders in perpetual futures contracts. They keep the perpetual price aligned with the spot price.

**Note:** Funding rates apply only to **crypto perpetual futures**. CME futures use daily settlement instead.

---

## How Funding Works

### Payment Direction

| Funding Rate | Longs | Shorts |
|--------------|-------|--------|
| Positive (+) | Pay | Receive |
| Negative (-) | Receive | Pay |

### Formula

```
Funding Payment = Position Notional × Funding Rate

where:
  Position Notional = abs(qty) × mark_price
```

### Timing

- **Binance**: Every 8 hours (00:00, 08:00, 16:00 UTC)
- **Payment window**: ±15 seconds around funding time
- **If closed before**: No payment received/paid

---

## Funding Rate Components

### Formula (Binance)

```
Funding Rate = Average Premium Index + clamp(Interest Rate - Premium, -0.05%, +0.05%)

where:
  Interest Rate = 0.01% (fixed for USDT)
  Premium = (Mark Price - Index Price) / Index Price
```

### Premium Index

```
Premium = (Max(0, Impact Bid - Index) - Max(0, Index - Impact Ask)) / Index
```

### Interest Rate

Default: 0.01% per funding period (0.03% per day)

---

## Funding Rate Regimes

### Normal Market

| Indicator | Range | Description |
|-----------|-------|-------------|
| **Funding Rate** | -0.01% to +0.01% | Balanced market |
| **Annualized** | -10% to +10% | Typical carry |
| **Action** | None | Normal trading |

### Bullish Market

| Indicator | Range | Description |
|-----------|-------|-------------|
| **Funding Rate** | +0.01% to +0.05% | Longs dominant |
| **Annualized** | +10% to +50% | Expensive to long |
| **Action** | Consider shorts | Or reduce long exposure |

### Extreme Bullish

| Indicator | Range | Description |
|-----------|-------|-------------|
| **Funding Rate** | > +0.05% | Overcrowded longs |
| **Annualized** | > +50% | Very expensive |
| **Action** | Short bias | High funding drain |

### Bearish Market

| Indicator | Range | Description |
|-----------|-------|-------------|
| **Funding Rate** | -0.01% to -0.05% | Shorts dominant |
| **Annualized** | -10% to -50% | Expensive to short |
| **Action** | Consider longs | Or reduce short exposure |

---

## Funding in Training

### Reward Impact

Funding affects the reward signal:

```python
# With funding impact enabled
reward = log_return × position - funding_penalty × funding_paid

# Example (simplified)
log_return = 0.001  # 0.1% price move
position = 0.5      # 50% of max position
funding_rate = 0.0001  # 0.01%
position_notional = 10000  # $10,000

# Without funding
reward = 0.001 × 0.5 = 0.0005

# With funding (long position, positive funding)
funding_paid = 10000 × 0.0001 = $1.00
funding_penalty = 1.0  # Scale factor
reward = 0.001 × 0.5 - (1.0 / 10000) = 0.0005 - 0.0001 = 0.0004
```

### Configuration

```yaml
env:
  futures:
    funding_rate_enabled: true
    funding_interval_hours: 8
    funding_rate_impact_on_reward: true
    funding_penalty_scale: 1.0  # Scale factor for funding impact
```

---

## Funding Tracking

### FuturesFundingTracker

```python
from services.futures_funding_tracker import (
    FuturesFundingTracker,
    FundingTrackerConfig,
    FundingRateInfo,
)

config = FundingTrackerConfig(
    data_dir="data/futures/funding",
    prediction_method="ewma",  # last, avg, ewma
    cache_ttl_sec=300,
)

tracker = FuturesFundingTracker(
    funding_provider=funding_provider,
    config=config,
)

# Get current funding info
info = await tracker.get_funding_info("BTCUSDT")

print(f"Current rate: {info.funding_rate:.4%}")
print(f"Next funding: {info.next_funding_time}")
print(f"Predicted rate: {info.predicted_rate:.4%}")
print(f"Countdown: {info.countdown_seconds} sec")
```

### Prediction Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `last` | Use last known rate | Stable markets |
| `avg` | Average of last N rates | General use |
| `ewma` | Exponential weighted moving average | Trending markets |

### Statistics

```python
stats = tracker.get_funding_stats("BTCUSDT", lookback_days=30)

print(f"Average rate: {stats.avg_rate:.4%}")
print(f"Std deviation: {stats.std_rate:.4%}")
print(f"Min rate: {stats.min_rate:.4%}")
print(f"Max rate: {stats.max_rate:.4%}")
print(f"Annualized: {stats.annualized_rate:.2%}")
print(f"Total paid (30d): ${stats.total_paid:.2f}")
```

---

## Funding Exposure Guard

### Risk Levels

| Level | Annualized Rate | Action |
|-------|-----------------|--------|
| **NORMAL** | < 10% | Normal trading |
| **WARNING** | 10% - 25% | Monitor exposure |
| **EXCESSIVE** | 25% - 50% | Consider reducing |
| **EXTREME** | > 50% | Reduce immediately |

### Code Example

```python
from services.futures_risk_guards import (
    FundingExposureGuard,
    FundingExposureLevel,
)

guard = FundingExposureGuard(
    warning_threshold=Decimal("0.0001"),  # 0.01% per 8h
)

result = guard.check_funding_exposure(
    funding_rate=Decimal("0.0005"),  # 0.05% per 8h
    position_side="LONG",
    position_notional=Decimal("100000"),
)

print(f"Level: {result.level}")  # EXTREME
print(f"Annualized: {result.annualized_rate:.1%}")  # ~54.8%
print(f"Daily cost: ${result.daily_cost:.2f}")
print(f"Monthly cost: ${result.monthly_cost:.2f}")
```

### Configuration

```yaml
funding:
  thresholds:
    warning: 0.0001      # 0.01% per 8h = ~10% APR
    excessive: 0.0003    # 0.03% per 8h = ~33% APR
    extreme: 0.0005      # 0.05% per 8h = ~55% APR

  alerts:
    warning: "monitor"
    excessive: "reduce_25pct"
    extreme: "close_position"
```

---

## Funding Strategies

### 1. Funding Arbitrage (Cash-and-Carry)

**Concept:** Long spot + Short perpetual to earn positive funding

**Requirements:**

- Positive funding rate
- Low transaction costs
- Access to both spot and futures

**Risk:** Funding can flip negative

### 2. Funding-Aware Position Sizing

**Concept:** Reduce position when funding is expensive

```python
def adjust_position_for_funding(base_position, funding_rate, threshold=0.0003):
    annualized = funding_rate * 3 * 365  # 3 payments/day × 365 days

    if abs(annualized) > threshold * 3 * 365:
        # Reduce position by funding impact
        reduction = min(0.5, abs(annualized))  # Max 50% reduction
        return base_position * (1 - reduction)

    return base_position
```

### 3. Contrarian Funding

**Concept:** Go against extreme funding

**Signal:**

- Extreme positive funding → Short bias (longs overextended)
- Extreme negative funding → Long bias (shorts overextended)

**Risk:** Funding can stay extreme longer than expected

---

## Funding in Simulation

### L2 Execution

```python
from execution_providers_futures import FuturesFeeProvider

provider = FuturesFeeProvider()

# Calculate funding payment
payment = provider.compute_funding_payment(
    position_notional=100000.0,
    funding_rate=0.0001,
    is_long=True,  # Long pays positive funding
)
# payment = $10.00 (paid by long)
```

### L3 Execution

Funding period dynamics affect queue behavior:

```python
from lob.futures_extensions import FundingPeriodDynamics

dynamics = FundingPeriodDynamics(
    funding_times_utc=[0, 8, 16],
    window_minutes_before=5,
    window_minutes_after=1,
)

state = dynamics.get_state(
    timestamp_ms=current_time_ms,
    funding_rate=0.0001,
)

print(f"In funding window: {state.in_funding_window}")
print(f"Spread multiplier: {state.spread_multiplier:.2f}")  # Wider near funding
print(f"Queue priority: {state.queue_priority_factor:.2f}")  # Lower near funding
```

---

## Funding Data Sources

### Binance API

```python
# Get current funding rate
GET /fapi/v1/premiumIndex
{
    "symbol": "BTCUSDT",
    "markPrice": "50000.00",
    "indexPrice": "49990.00",
    "lastFundingRate": "0.00010000",
    "nextFundingTime": 1640995200000,
}

# Get funding rate history
GET /fapi/v1/fundingRate
```

### Historical Data

```python
from adapters.binance.market_data import BinanceFuturesMarketDataAdapter

adapter = BinanceFuturesMarketDataAdapter()

# Get funding history
history = adapter.get_funding_rate_history(
    symbol="BTCUSDT",
    limit=500,
    start_time=start_ts,
    end_time=end_ts,
)

for rate in history:
    print(f"{rate.timestamp}: {rate.funding_rate:.4%}")
```

---

## Best Practices

### 1. Monitor Funding Regularly

- Check funding before entering positions
- Set alerts for extreme rates

### 2. Account for Funding in PnL

- Include funding in profit calculations
- Understand break-even after funding

### 3. Time Entries/Exits

- Enter shortly after funding to maximize interval
- Exit before funding if rate is against you

### 4. Use Funding as Signal

- Extreme rates indicate overcrowded positions
- Potential reversal signal

### 5. Backtest with Funding

- Include funding in historical simulations
- Don't overfit to funding patterns

---

## References

- Binance: "Funding Rate History and Calculation"
- Binance: "Premium Index and Funding Rate"
- Academic: Capponi et al. (2022) "The Funding Rate Mechanics in Cryptocurrency Perpetual Futures"
