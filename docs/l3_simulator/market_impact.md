# L3 LOB Simulator - Market Impact Models

## Overview

Market impact models estimate how trade execution affects prices. The L3 simulator implements three academically-validated models: Kyle (1985), Almgren-Chriss (2001), and Gatheral (2010).

**Files**:

- `lob/market_impact.py` (~1,150 lines) - Impact models
- `lob/impact_effects.py` (~832 lines) - Effects on LOB
- `lob/impact_calibration.py` (~1,059 lines) - Parameter estimation

## Impact Components

### Temporary Impact

Price displacement during execution that reverses after completion.

- Caused by immediate liquidity consumption
- Decays over time

### Permanent Impact

Lasting price change from information revealed by the trade.

- Persists after execution
- Represents true price discovery

## Kyle Lambda Model (1985)

Linear price impact model from Kyle's seminal paper.

```python
from lob import KyleLambdaModel, ImpactParameters

# Create model
params = ImpactParameters(
    kyle_lambda=1e-6,  # Price impact per unit traded
)
model = KyleLambdaModel(params=params)

# Compute impact
result = model.compute_impact(
    order_qty=10000,
    side=1,  # +1 BUY, -1 SELL
    mid_price=150.0,
)

print(f"Price impact: {result.total_impact_bps:.2f} bps")
print(f"New mid: {result.expected_price:.2f}")
```

**Formula**:

```
ΔP = λ × sign(Q) × |Q|
```

Where:

- `λ` = Kyle lambda (price impact coefficient)
- `Q` = Order quantity

## Almgren-Chriss Model (2001)

Square-root temporary impact + linear permanent impact.

```python
from lob import AlmgrenChrissModel, ImpactParameters

# Create model with default equity parameters
params = ImpactParameters.for_equity()
model = AlmgrenChrissModel(params=params)

# Or custom parameters
params = ImpactParameters(
    eta=0.05,      # Temporary impact coefficient
    gamma=0.03,    # Permanent impact coefficient
    delta=0.5,     # Impact exponent (0.5 = square-root)
)
model = AlmgrenChrissModel(params=params)

# Compute impact
result = model.compute_total_impact(
    order_qty=10000,
    adv=10_000_000,      # Average daily volume
    volatility=0.02,     # Daily volatility
    mid_price=150.0,
)

print(f"Temporary impact: {result.temporary_impact_bps:.2f} bps")
print(f"Permanent impact: {result.permanent_impact_bps:.2f} bps")
print(f"Total impact: {result.total_impact_bps:.2f} bps")
print(f"Impact cost: ${result.impact_cost:.2f}")
```

**Formulas**:

```
Temporary: η × σ × (Q/V)^δ × 10000 bps
Permanent: γ × (Q/V) × 10000 bps
```

Where:

- `η` = Temporary impact coefficient
- `γ` = Permanent impact coefficient
- `δ` = Impact exponent (typically 0.5)
- `σ` = Volatility
- `Q` = Order quantity
- `V` = ADV (average daily volume)

## Gatheral Model (2010)

Transient impact with power-law decay.

```python
from lob import GatheralModel, ImpactParameters, DecayType

params = ImpactParameters(
    eta=0.05,
    gamma=0.03,
    tau_ms=30000,    # Decay time constant (30 sec)
    beta=1.5,        # Power-law exponent
)
model = GatheralModel(params=params, decay_type=DecayType.POWER_LAW)

# Compute impact at different times
result_t0 = model.compute_impact(order_qty=10000, time_since_trade_ms=0, ...)
result_t1 = model.compute_impact(order_qty=10000, time_since_trade_ms=10000, ...)
result_t2 = model.compute_impact(order_qty=10000, time_since_trade_ms=30000, ...)

print(f"Impact at t=0: {result_t0.total_impact_bps:.2f} bps")
print(f"Impact at t=10s: {result_t1.total_impact_bps:.2f} bps")
print(f"Impact at t=30s: {result_t2.total_impact_bps:.2f} bps")
```

**Decay Functions**:

| Type | Formula |
|------|---------|
| Power-law | `G(t) = (1 + t/τ)^(-β)` |
| Exponential | `G(t) = exp(-t/τ)` |
| Linear | `G(t) = max(0, 1 - t/τ)` |

## Factory Functions

```python
from lob import create_impact_model

# Create by type
model = create_impact_model("kyle")
model = create_impact_model("almgren_chriss")
model = create_impact_model("gatheral")

# With custom parameters
model = create_impact_model(
    "almgren_chriss",
    eta=0.05,
    gamma=0.03,
    delta=0.5,
)

# From config
from lob.config import L3ExecutionConfig
config = L3ExecutionConfig.from_yaml("configs/execution_l3.yaml")
model = create_impact_model(
    config.market_impact.model,
    **config.market_impact.dict(),
)
```

## Impact Tracker

Track cumulative impact across multiple trades.

```python
from lob import ImpactTracker, create_impact_tracker

tracker = create_impact_tracker(
    model_type="almgren_chriss",
    decay_type="power_law",
)

# Record trades
tracker.record_trade(
    trade_qty=5000,
    side=1,
    timestamp_ms=1000,
    mid_price=150.0,
    adv=10_000_000,
)

tracker.record_trade(
    trade_qty=3000,
    side=1,
    timestamp_ms=5000,
    mid_price=150.10,
    adv=10_000_000,
)

# Get current cumulative impact
state = tracker.get_impact_state(current_time_ms=10000)
print(f"Cumulative impact: {state.cumulative_impact_bps:.2f} bps")
print(f"Residual temporary: {state.residual_temporary_bps:.2f} bps")
print(f"Permanent: {state.permanent_impact_bps:.2f} bps")
```

## Impact Effects on LOB

Apply impact effects to order book state.

```python
from lob import ImpactEffects, create_impact_effects, LOBImpactSimulator

# Create effects calculator
effects = create_impact_effects(
    quote_shift_enabled=True,
    liquidity_reaction_enabled=True,
    momentum_detection_enabled=True,
)

# Or use full simulator
simulator = LOBImpactSimulator(
    impact_model=model,
    effects=effects,
)

# Simulate trade impact
impact_result, quote_shift, liquidity = simulator.simulate_trade_impact(
    order_book=book,
    order=order,
    fill=fill,
    adv=10_000_000,
    volatility=0.02,
)

print(f"Quote shift: bid {quote_shift.old_bid} -> {quote_shift.new_bid}")
print(f"Quote shift: ask {quote_shift.old_ask} -> {quote_shift.new_ask}")
print(f"Liquidity reaction: {liquidity.liquidity_change:.2%}")
```

### Quote Shifting

```python
from lob import QuoteShiftResult

# Impact causes quotes to move
shift = effects.compute_quote_shift(
    order_book=book,
    impact_bps=5.0,
    side=Side.BUY,
)

print(f"Bid shift: {shift.bid_shift_bps:.2f} bps")
print(f"Ask shift: {shift.ask_shift_bps:.2f} bps")
print(f"New spread: {shift.new_spread:.4f}")
```

### Liquidity Reaction

```python
from lob import LiquidityReactionResult

# Liquidity providers react to large trades
reaction = effects.compute_liquidity_reaction(
    impact_bps=10.0,
    adv=10_000_000,
    volatility=0.02,
)

print(f"Liquidity change: {reaction.liquidity_change:.2%}")
print(f"Spread widening: {reaction.spread_widening_bps:.2f} bps")
```

### Momentum Detection

```python
from lob import MomentumResult

# Detect momentum from trade flow
momentum = effects.detect_momentum(
    recent_trades=[trade1, trade2, trade3],
    lookback_ms=60000,
)

print(f"Momentum signal: {momentum.signal.name}")  # BULLISH, BEARISH, NEUTRAL
print(f"Strength: {momentum.strength:.2f}")
```

## Impact Calibration

Calibrate impact parameters from historical trade data.

```python
from lob import (
    ImpactCalibrationPipeline,
    CalibrationDataset,
    TradeObservation,
    create_impact_calibration_pipeline,
)

# Create pipeline
pipeline = create_impact_calibration_pipeline()

# Build dataset
dataset = CalibrationDataset(avg_adv=10_000_000, avg_volatility=0.02)

for trade in historical_trades:
    obs = TradeObservation(
        timestamp_ms=trade.timestamp,
        price=trade.price,
        qty=trade.qty,
        side=1 if trade.is_buy else -1,
        adv=trade.adv,
        pre_trade_mid=trade.pre_mid,
        post_trade_mid=trade.post_mid,
    )
    dataset.add_observation(obs)

# Calibrate all models
results = pipeline.calibrate_all(dataset)

print(f"Kyle lambda: {results['kyle']['kyle_lambda']:.2e}")
print(f"Almgren-Chriss eta: {results['almgren_chriss']['eta']:.4f}")
print(f"Almgren-Chriss gamma: {results['almgren_chriss']['gamma']:.4f}")

# Get calibrated model
model = pipeline.create_calibrated_model("almgren_chriss")
```

### Calibration Methods

| Model | Method | Parameters |
|-------|--------|------------|
| Kyle | OLS regression | `kyle_lambda` |
| Almgren-Chriss | Grid search + OLS | `eta`, `gamma`, `delta` |
| Gatheral | Non-linear optimization | `tau`, `beta` |

## Parameter Presets

```python
from lob import ImpactParameters

# Equity defaults (large-cap US stocks)
params = ImpactParameters.for_equity()
# eta=0.05, gamma=0.03, delta=0.5

# Crypto defaults (higher impact)
params = ImpactParameters.for_crypto()
# eta=0.10, gamma=0.05, delta=0.5

# Custom
params = ImpactParameters(
    eta=0.08,
    gamma=0.04,
    delta=0.6,
    kyle_lambda=5e-7,
    tau_ms=60000,
    beta=1.2,
)
```

## Performance

| Operation | Complexity | Typical Latency |
|-----------|------------|-----------------|
| Compute impact | O(1) | <5 μs |
| Track cumulative | O(n) | <50 μs (n=recent trades) |
| Quote shift | O(1) | <10 μs |
| Calibration | O(n²) | Seconds (offline) |

## Related Documentation

- [Overview](overview.md) - Architecture overview
- [Calibration](calibration.md) - Full calibration pipeline
- [Configuration](configuration.md) - Config reference
