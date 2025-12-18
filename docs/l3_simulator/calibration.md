# L3 LOB Simulator - Calibration

## Overview

Calibration allows tuning simulator parameters to match real market behavior. The L3 simulator provides calibration pipelines for fill probability, market impact, latency, and queue dynamics.

**Files**:
- `lob/calibration.py` (~1,129 lines) - Fill probability calibration
- `lob/impact_calibration.py` (~1,059 lines) - Market impact calibration
- `lob/calibration_pipeline.py` (~1,182 lines) - Unified L3 calibration

## Calibration Components

| Component | Parameters | Data Required |
|-----------|------------|---------------|
| Fill Probability | Arrival rate, queue decay | Trade history, order history |
| Market Impact | η, γ, λ, β, τ | Trade history with pre/post prices |
| Latency | Distribution params | Timestamp data with latencies |
| Queue Dynamics | Queue sizes, fill rates | Order book snapshots |

## Fill Probability Calibration

### Poisson Rate Calibrator

Estimate volume arrival rate for analytical fill probability.

```python
from lob import (
    CalibrationPipeline,
    TradeRecord,
    create_calibration_pipeline,
    calibrate_from_trades,
)

# Create pipeline
pipeline = create_calibration_pipeline()

# Add trade data
for trade in historical_trades:
    pipeline.add_trade(TradeRecord(
        timestamp_ns=trade.timestamp_ns,
        price=trade.price,
        qty=trade.qty,
        side=trade.side,
    ))

# Calibrate
results = pipeline.run_calibration()

# Get calibrated model
model = pipeline.get_best_model("poisson")
print(f"Arrival rate: {model.arrival_rate:.2f} qty/sec")
```

### Queue-Reactive Calibrator

Calibrate intensity-based model (Huang et al.).

```python
from lob import QueueReactiveCalibrator

calibrator = QueueReactiveCalibrator()

# Add observations with queue state
for obs in observations:
    calibrator.add_observation(
        timestamp_ns=obs.timestamp_ns,
        queue_position=obs.queue_pos,
        queue_size=obs.queue_size,
        spread_bps=obs.spread,
        volatility=obs.vol,
        filled=obs.was_filled,
        time_to_fill_sec=obs.time_to_fill,
    )

# Calibrate using MLE
results = calibrator.calibrate()

print(f"Base rate: {results.base_rate:.2f}")
print(f"Queue decay α: {results.queue_decay_alpha:.4f}")
print(f"Spread sensitivity β: {results.spread_sensitivity_beta:.4f}")
```

### Historical Rate Calibrator

Build historical fill rate lookup table.

```python
from lob import HistoricalRateCalibrator

calibrator = HistoricalRateCalibrator(
    queue_bins=[0, 5, 10, 20, 50, 100],  # Queue position bins
    spread_bins=[0, 5, 10, 20],          # Spread bins (bps)
)

# Add fill observations
for fill in historical_fills:
    calibrator.add_fill_observation(
        queue_position=fill.queue_pos,
        spread_bps=fill.spread,
        time_to_fill_sec=fill.time_to_fill,
    )

# Build lookup table
rates = calibrator.build_rate_table()

# Query rate
rate = rates.get_rate(queue_position=10, spread_bps=8)
print(f"Fill rate: {rate:.2f} per second")
```

## Market Impact Calibration

### Almgren-Chriss Calibrator

Calibrate temporary and permanent impact parameters.

```python
from lob import (
    ImpactCalibrationPipeline,
    CalibrationDataset,
    TradeObservation,
    AlmgrenChrissCalibrator,
)

# Build dataset
dataset = CalibrationDataset(
    avg_adv=10_000_000,
    avg_volatility=0.02,
)

for trade in historical_trades:
    dataset.add_observation(TradeObservation(
        timestamp_ms=trade.timestamp,
        price=trade.price,
        qty=trade.qty,
        side=1 if trade.is_buy else -1,
        adv=trade.adv,
        pre_trade_mid=trade.pre_mid,
        post_trade_mid=trade.post_mid,
        post_5min_mid=trade.mid_5min_later,  # For permanent
    ))

# Calibrate
calibrator = AlmgrenChrissCalibrator()
results = calibrator.calibrate(dataset)

print(f"Temporary η: {results.eta:.4f}")
print(f"Permanent γ: {results.gamma:.4f}")
print(f"Exponent δ: {results.delta:.3f}")
print(f"R²: {results.r_squared:.3f}")
```

### Kyle Lambda Calibrator

Estimate Kyle's lambda from price impact regression.

```python
from lob import KyleLambdaCalibrator

calibrator = KyleLambdaCalibrator()

for trade in trades:
    calibrator.add_observation(
        qty=trade.qty,
        price_change=trade.post_mid - trade.pre_mid,
        side=trade.side,
    )

results = calibrator.calibrate()
print(f"Kyle λ: {results.kyle_lambda:.2e}")
```

### Gatheral Decay Calibrator

Calibrate transient impact decay parameters.

```python
from lob import GatheralDecayCalibrator

calibrator = GatheralDecayCalibrator()

# Add trades with post-trade price series
for trade in trades:
    calibrator.add_trade_with_decay(
        trade_time_ms=trade.timestamp,
        trade_qty=trade.qty,
        trade_side=trade.side,
        post_trade_prices=[
            (trade.timestamp + 1000, trade.mid_1s),
            (trade.timestamp + 5000, trade.mid_5s),
            (trade.timestamp + 30000, trade.mid_30s),
            (trade.timestamp + 60000, trade.mid_60s),
        ],
    )

results = calibrator.calibrate()
print(f"Decay τ: {results.tau_ms:.0f} ms")
print(f"Power β: {results.beta:.3f}")
```

## Unified L3 Calibration Pipeline

Calibrate all L3 components from a single data source.

```python
from lob import (
    L3CalibrationPipeline,
    create_l3_calibration_pipeline,
    calibrate_from_dataframe,
)
import pandas as pd

# Create pipeline
pipeline = create_l3_calibration_pipeline()

# Option 1: Add observations manually
pipeline.add_latency_observation(
    event_time_ns=...,
    receive_time_ns=...,
    latency_type="feed",
)

pipeline.add_trade_observation(
    timestamp_ns=...,
    price=...,
    qty=...,
    side=...,
    pre_mid=...,
    post_mid=...,
)

# Option 2: From DataFrame
df = pd.read_parquet("data/calibration_data.parquet")
results = calibrate_from_dataframe(
    df,
    timestamp_col="timestamp",
    price_col="price",
    qty_col="quantity",
    side_col="side",
)

# Run full calibration
results = pipeline.run_full_calibration()

print("=== Latency Results ===")
print(f"Feed mean: {results.latency.feed_mean_us:.1f} μs")
print(f"Feed std: {results.latency.feed_std_us:.1f} μs")

print("=== Impact Results ===")
print(f"Almgren-Chriss η: {results.impact.eta:.4f}")
print(f"Almgren-Chriss γ: {results.impact.gamma:.4f}")

print("=== Queue Dynamics Results ===")
print(f"Fill rate: {results.queue.fill_rate_per_sec:.2f}")
print(f"Cancel rate: {results.queue.cancel_rate_per_sec:.2f}")
```

## Cross-Validation

Validate calibration results with holdout data.

```python
from lob import CalibrationPipeline

pipeline = create_calibration_pipeline()

# Add all data
for trade in all_trades:
    pipeline.add_trade(trade)

# Cross-validate
cv_results = pipeline.cross_validate(
    n_folds=5,
    models=["poisson", "queue_reactive"],
)

for model_name, scores in cv_results.items():
    print(f"{model_name}:")
    print(f"  Mean log-likelihood: {scores.mean_ll:.4f}")
    print(f"  Std: {scores.std_ll:.4f}")
```

## Rolling Calibration

Update parameters incrementally as new data arrives.

```python
from lob import RollingImpactCalibrator

calibrator = RollingImpactCalibrator(
    lookback_trades=1000,  # Last 1000 trades
    min_trades=100,        # Minimum for estimation
    update_frequency=10,   # Re-calibrate every 10 trades
)

for trade in live_trades:
    calibrator.add_trade(trade)

    if calibrator.should_update():
        params = calibrator.get_current_params()
        print(f"Updated η: {params.eta:.4f}")
```

## Calibration from Files

### LOBSTER Format

```python
from lob import LOBSTERAdapter, create_l3_calibration_pipeline

adapter = LOBSTERAdapter()
orders, trades = adapter.load("data/AAPL_2024-01-15")

pipeline = create_l3_calibration_pipeline()
for trade in trades:
    pipeline.add_trade_observation(...)

results = pipeline.run_full_calibration()
```

### Parquet Format

```python
import pandas as pd
from lob import calibrate_from_dataframe

df = pd.read_parquet("data/trades.parquet")

results = calibrate_from_dataframe(
    df,
    timestamp_col="event_time",
    price_col="price",
    qty_col="quantity",
    side_col="is_buyer_maker",
)
```

## Saving and Loading Calibration

```python
# Save calibration results
results.to_yaml("calibration_results.yaml")
results.to_json("calibration_results.json")

# Load and apply
from lob import L3CalibrationResult

results = L3CalibrationResult.from_yaml("calibration_results.yaml")

# Apply to config
from lob.config import L3ExecutionConfig

config = L3ExecutionConfig.for_equity()
config.apply_calibration(results)
```

## Calibration Quality Metrics

| Metric | Good Range | Description |
|--------|------------|-------------|
| R² (impact) | >0.5 | Explained variance |
| Log-likelihood | Higher better | Model fit |
| RMSE (prediction) | <10% | Prediction error |
| CV std | <20% mean | Stability |

## Recommendations

1. **Minimum Data**: 10,000+ trades for stable estimates
2. **Recalibrate**: Weekly for stable markets, daily for volatile
3. **Validate**: Always cross-validate before production use
4. **Monitor**: Track prediction errors in live trading

## Related Documentation

- [Market Impact](market_impact.md) - Impact models
- [Fill Probability](../../lob/fill_probability.py) - Probability models
- [Configuration](configuration.md) - Config reference
