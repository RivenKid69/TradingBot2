# Observation Vector Structure - Technical Indicators Integration

## Overview

This document describes the complete structure of the observation vector (57 features) used by the trading agent. The observation vector is constructed by `obs_builder.build_observation_vector()` and populated with technical indicators from `prepare_and_run.py` and market microstructure data.

**Total Features**: 57 (with max_num_tokens=1)

## Feature Layout

### Positions 0-2: Bar-Level Features (3 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 0 | `price` | Current market price | Mark price or resolved reward price |
| 1 | `log_volume_norm` | Quote asset volume | `tanh(log1p(quote_volume / 1e6))` |
| 2 | `rel_volume` | Base volume | `tanh(log1p(volume / 100))` |

### Positions 3-4: MA5 (Moving Average 5) (2 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 3 | `ma5` | `df['sma_5']` | 5-period simple moving average |
| 4 | `ma5_valid` | Computed | 1.0 if ma5 is not NaN, 0.0 otherwise |

### Positions 5-6: MA20 (Moving Average 20) (2 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 5 | `ma20` | `df['sma_15']` | 15-period SMA (mapped to ma20 slot) |
| 6 | `ma20_valid` | Computed | 1.0 if ma20 is not NaN, 0.0 otherwise |

### Positions 7-13: Technical Indicators (7 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 7 | `rsi14` | `df['rsi']` or Simulator | Relative Strength Index (14-period) |
| 8 | `macd` | Simulator (`get_macd`) | MACD line |
| 9 | `macd_signal` | Simulator (`get_macd_signal`) | MACD signal line |
| 10 | `momentum` | Simulator (`get_momentum`) | Price momentum |
| 11 | `atr` | Simulator (`get_atr`) | Average True Range (volatility) |
| 12 | `cci` | Simulator (`get_cci`) | Commodity Channel Index |
| 13 | `obv` | Simulator (`get_obv`) | On-Balance Volume |

### Positions 14-15: Derived Price/Volatility Signals (2 features)

| Position | Feature | Formula | Description |
|----------|---------|---------|-------------|
| 14 | `ret_1h` | `tanh((price - prev_price) / (prev_price + 1e-8))` | 1-hour return (normalized) |
| 15 | `vol_proxy` | `tanh(log1p(atr / (price + 1e-8)))` | Volatility proxy based on ATR |

### Positions 16-21: Agent State (6 features)

| Position | Feature | Formula | Description |
|----------|---------|---------|-------------|
| 16 | `cash_ratio` | `cash / (cash + position_value)` | Proportion of cash in portfolio |
| 17 | `position_ratio` | `tanh(position_value / total_worth)` | Position size relative to portfolio |
| 18 | `vol_imbalance` | `state.last_vol_imbalance` | Volume imbalance from last step |
| 19 | `trade_intensity` | `state.last_trade_intensity` | Trade intensity from last step |
| 20 | `realized_spread` | `state.last_realized_spread` | Realized spread from last execution |
| 21 | `agent_fill_ratio` | `state.last_agent_fill_ratio` | Fill ratio of agent orders |

### Positions 22-24: Microstructure Proxies (3 features)

| Position | Feature | Formula | Description |
|----------|---------|---------|-------------|
| 22 | `ofi_proxy` | `mid_ret * vol_intensity` | Order Flow Imbalance proxy |
| 23 | `qimb` | `tanh(last_vol_imbalance)` | Quote imbalance |
| 24 | `micro_dev` | `0.5 * realized_spread * qimb` | Microstructure deviation |

### Positions 25-26: Bollinger Bands (2 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 25 | `bb_position` | Computed from `bb_lower`, `bb_upper` | Price position within Bollinger Bands |
| 26 | `bb_width` | `(bb_upper - bb_lower) / price` | Bollinger Band width (normalized) |

### Positions 27-29: Event Metadata (3 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 27 | `is_high_importance` | `df['is_high_importance']` | Binary flag for high-importance events |
| 28 | `time_since_event` | `tanh(df['time_since_event'] / 24.0)` | Hours since last major event (normalized) |
| 29 | `risk_off_flag` | `fear_greed < 25` | Binary flag for risk-off market regime |

### Positions 30-31: Fear & Greed Index (2 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 30 | `fear_greed_value` | `df['fear_greed_value'] / 100` | Fear & Greed Index (0-100 normalized) |
| 31 | `fear_greed_indicator` | Computed | 1.0 if F&G data available, 0.0 otherwise |

### Positions 32-39: External Normalized Columns (8 features)

**These positions contain advanced technical indicators from `prepare_and_run.py`**:

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 32 | `cvd_24h` | `df['cvd_24h']` | Cumulative Volume Delta (24-hour) |
| 33 | `cvd_168h` | `df['cvd_168h']` | Cumulative Volume Delta (168-hour / 1-week) |
| 34 | `yang_zhang_24h` | `df['yang_zhang_24h']` | Yang-Zhang volatility estimator (24h) |
| 35 | `yang_zhang_168h` | `df['yang_zhang_168h']` | Yang-Zhang volatility estimator (168h) |
| 36 | `garch_12h` | `df['garch_12h']` | GARCH(1,1) conditional volatility (12h) |
| 37 | `garch_24h` | `df['garch_24h']` | GARCH(1,1) conditional volatility (24h) |
| 38 | `ret_15m` | `df['ret_15m']` | 15-minute return |
| 39 | `ret_60m` | `df['ret_60m']` | 60-minute return |

**Note**: All values in positions 32-39 are normalized using `tanh()` to keep them in the range [-1, 1].

### Positions 40-41: Token Metadata (2 features)

| Position | Feature | Formula | Description |
|----------|---------|--------|-------------|
| 40 | `num_tokens_norm` | `num_tokens / max_num_tokens` | Number of tokens normalized |
| 41 | `token_id_norm` | `token_id / max_num_tokens` | Token ID normalized |

### Positions 42+: One-Hot Token Encoding (variable size)

With `max_num_tokens=1` (default), this adds 1 feature:

| Position | Feature | Description |
|----------|---------|-------------|
| 42 | Token 0 | One-hot encoding (1.0 for current token, 0.0 otherwise) |

**Total with max_num_tokens=1**: 43 base features + 1 token = **44 features**

However, the observation space is typically configured as **57 features** to allow for future expansion or different token configurations.

## Data Flow

```
prepare_and_run.py
    ↓
    Creates technical indicators:
    - sma_5, sma_15, sma_60
    - rsi
    - cvd_24h, cvd_168h
    - yang_zhang_24h, yang_zhang_168h
    - garch_12h, garch_24h
    - ret_15m, ret_60m
    - fear_greed_value
    ↓
    Saves to data/processed/*.feather
    ↓
trading_patchnew.py / train_model_multi_patch.py
    ↓
    Loads feather files into TradingEnv.df
    ↓
mediator.py
    ↓
    _build_observation() extracts data from:
    - row (current df row)
    - state (agent state)
    - sim (MarketSimulator for real-time indicators)
    ↓
    Calls obs_builder.build_observation_vector()
    ↓
    Returns observation vector (57 features)
    ↓
RL Agent (DistributionalPPO)
```

## Implementation Details

### Source Files

1. **mediator.py:1027-1174**: Main `_build_observation()` method
   - Extracts all data from row, state, and simulator
   - Calls `obs_builder.build_observation_vector()`
   - Falls back to legacy mode if obs_builder unavailable

2. **obs_builder.pyx:24-307**: Cython implementation
   - `build_observation_vector_c()`: Low-level C implementation
   - `build_observation_vector()`: Python-callable wrapper
   - Constructs the observation vector efficiently without Python overhead

3. **prepare_and_run.py:325-359**: Technical indicator creation
   - Uses `apply_offline_features()` to compute all indicators
   - Saves results to feather files

4. **feature_config.py**: Feature layout configuration
   - Defines `FEATURES_LAYOUT` structure
   - Computes `N_FEATURES` dynamically

### Helper Methods (mediator.py)

- `_get_safe_float(row, col, default)`: Safely extract float from row with fallback
- `_extract_market_data(row, state, mark_price, prev_price)`: Extract price and volume data
- `_extract_technical_indicators(row, sim, row_idx)`: Extract all technical indicators
- `_extract_norm_cols(row)`: Extract cvd, garch, yang_zhang into norm_cols array
- `_coerce_finite(value, default)`: Ensure finite float values

## Testing

Comprehensive tests are available in `tests/test_technical_indicators_in_obs.py`:

1. **test_observation_size_and_non_zero()**: Verifies size and non-zero content
2. **test_technical_indicators_present()**: Checks indicators are in correct positions
3. **test_cvd_garch_yangzhang_in_obs()**: Verifies specific indicators appear
4. **test_observations_in_training_env()**: Tests training scenario
5. **test_observation_works_without_indicators()**: Tests fallback mode

## Fallback Behavior

If technical indicators are missing from the dataframe:

- `_get_safe_float()` returns default values (0.0 or specified default)
- MA5/MA20: Return `NaN` → marked as invalid (valid_flag=0.0)
- RSI: Defaults to 50.0 (neutral)
- CVD/GARCH/Yang-Zhang: Default to 0.0 (neutral)
- Fear & Greed: Defaults to 50.0 (neutral)

The observation vector is always constructed with the correct size (57), but may contain more zeros if indicators are unavailable.

## Usage Examples

### In Training

```python
from trading_patchnew import TradingEnv
from train_model_multi_patch import create_envs

# Environment automatically loads data with technical indicators
env = TradingEnv(df=df_with_indicators, ...)

# Observation is automatically constructed with all 57 features
obs, info = env.reset()
assert obs.shape == (57,)
```

### In Production

```python
from mediator import Mediator

mediator = Mediator(env=trading_env)

# During stepping, observation is built with latest market data
obs = mediator._build_observation(row=current_row, state=state, mark_price=current_price)
```

## Key Design Decisions

1. **Why norm_cols[8]?**: Allows 8 external indicators (cvd, garch, yang_zhang, returns) without hardcoding positions
2. **Why tanh normalization?**: Keeps values in [-1, 1] range, suitable for neural networks
3. **Why fallback to legacy?**: Ensures backward compatibility if obs_builder is not compiled
4. **Why separate ma5_valid/ma20_valid?**: Allows model to distinguish between "zero MA" and "MA not available"

## Future Extensions

To add new indicators:

1. Add column to `prepare_and_run.py` via `apply_offline_features()`
2. Map to `norm_cols` position in `mediator._extract_norm_cols()`
3. Update this documentation
4. Add tests in `test_technical_indicators_in_obs.py`

---

**Last Updated**: 2025-01-09
**Authors**: Technical Indicators Integration Task
**Related Files**: `mediator.py`, `obs_builder.pyx`, `prepare_and_run.py`, `feature_config.py`
