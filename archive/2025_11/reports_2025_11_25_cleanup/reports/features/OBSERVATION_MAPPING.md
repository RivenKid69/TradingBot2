# Observation Vector Structure - Technical Indicators Integration

## Overview

This document describes the complete structure of the observation vector (63 features) used by the trading agent. The observation vector is constructed by `obs_builder.build_observation_vector()` and populated with technical indicators from `prepare_and_run.py` and market microstructure data.

**Total Features**: 63 (with max_num_tokens=1 and EXT_NORM_DIM=21)

**Note**: This document reflects the current implementation with validity flags for all technical indicators. The actual feature count is calculated dynamically in `feature_config.py` based on block sizes.

**Validity Flags (NEW)**: Added 7 explicit validity flags for indicators (rsi, macd, macd_signal, momentum, atr, cci, obv) to eliminate ambiguity between "no data yet" (warmup period) and "meaningful zero value". This increased the observation size from 56 to 62 features (6 flags), then to 63 features (7 flags, added ATR).

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

### Positions 7-20: Technical Indicators (14 features)

**Note**: This section now includes 7 validity flags to distinguish "no data yet" from "meaningful zero value".

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 7 | `rsi14` | `df['rsi']` or Simulator | Relative Strength Index (14-period), fallback: 50.0 |
| 8 | `rsi_valid` | Computed | **NEW**: 1.0 if rsi14 valid, 0.0 if no data yet (warmup < 14 bars) |
| 9 | `macd` | Simulator (`get_macd`) | MACD line, fallback: 0.0 |
| 10 | `macd_valid` | Computed | **NEW**: 1.0 if macd valid, 0.0 if no data yet (warmup < 26 bars) |
| 11 | `macd_signal` | Simulator (`get_macd_signal`) | MACD signal line, fallback: 0.0 |
| 12 | `macd_signal_valid` | Computed | **NEW**: 1.0 if macd_signal valid, 0.0 if no data yet (warmup < 35 bars) |
| 13 | `momentum` | Simulator (`get_momentum`) | Price momentum, fallback: 0.0 |
| 14 | `momentum_valid` | Computed | **NEW**: 1.0 if momentum valid, 0.0 if no data yet (warmup < 10 bars) |
| 15 | `atr` | Simulator (`get_atr`) | Average True Range (volatility), fallback: price*0.01 |
| 16 | `atr_valid` | Computed | **NEW v63**: 1.0 if atr valid, 0.0 if no data yet (warmup < 14 bars). **CRITICAL**: Prevents NaN in vol_proxy |
| 17 | `cci` | Simulator (`get_cci`) | Commodity Channel Index, fallback: 0.0 |
| 18 | `cci_valid` | Computed | 1.0 if cci valid, 0.0 if no data yet (warmup < 20 bars) |
| 19 | `obv` | Simulator (`get_obv`) | On-Balance Volume, fallback: 0.0 |
| 20 | `obv_valid` | Computed | 1.0 if obv valid, 0.0 if no data yet (warmup < 1 bar) |

**Why validity flags?** Indicators like RSI (neutral = 50) and MACD (no divergence = 0) create ambiguity: does value=50 or value=0 mean "no data yet" or "meaningful signal"? Validity flags eliminate this confusion for the neural network.

### Positions 21-22: Derived Price/Volatility Signals (2 features)

| Position | Feature | Formula | Description |
|----------|---------|---------|-------------|
| 21 | `ret_bar` | `tanh((price - prev_price) / (prev_price + 1e-8))` | Bar-to-bar return (normalized) |
| 22 | `vol_proxy` | `tanh(log1p(atr / (price + 1e-8)))` if atr_valid else fallback | **Volatility proxy based on ATR (uses atr_valid flag!)** |

### Positions 23-28: Agent State (6 features)

| Position | Feature | Formula | Description |
|----------|---------|---------|-------------|
| 23 | `cash_ratio` | `cash / (cash + position_value)` | Proportion of cash in portfolio |
| 24 | `position_ratio` | `tanh(position_value / total_worth)` | Position size relative to portfolio |
| 25 | `vol_imbalance` | `state.last_vol_imbalance` | Volume imbalance from last step |
| 26 | `trade_intensity` | `state.last_trade_intensity` | Trade intensity from last step |
| 27 | `realized_spread` | `state.last_realized_spread` | Realized spread from last execution |
| 28 | `agent_fill_ratio` | `state.last_agent_fill_ratio` | Fill ratio of agent orders |

### Positions 29-31: Microstructure Proxies (3 features)

**Note**: These are adapted for 4h timeframe using indicator-based proxies instead of high-frequency order flow data.

| Position | Feature | Formula | Description |
|----------|---------|---------|-------------|
| 29 | `price_momentum` | `tanh(momentum / (price * 0.01))` if momentum_valid else 0.0 | Price momentum strength (uses validity flag) |
| 30 | `bb_squeeze` | `tanh((bb_upper - bb_lower) / price)` if bb_valid else 0.0 | Bollinger Band squeeze (volatility regime) |
| 31 | `trend_strength` | `tanh((macd - macd_signal) / (price * 0.01))` if macd_valid and macd_signal_valid else 0.0 | MACD divergence (uses validity flags) |

### Positions 32-33: Bollinger Bands Context (2 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 32 | `bb_position` | Computed from `bb_lower`, `bb_upper` | Price position within Bollinger Bands (0.5 if unavailable) |
| 33 | `bb_width` | `(bb_upper - bb_lower) / price` | Bollinger Band width (normalized, 0.0 if unavailable) |

### Positions 34-36: Event Metadata (3 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 34 | `is_high_importance` | `df['is_high_importance']` | Binary flag for high-importance events |
| 35 | `time_since_event` | `tanh(df['time_since_event'] / 24.0)` | Hours since last major event (normalized) |
| 36 | `risk_off_flag` | `fear_greed < 25` | Binary flag for risk-off market regime |

### Positions 37-38: Fear & Greed Index (2 features)

| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 37 | `fear_greed_value` | `df['fear_greed_value'] / 100` | Fear & Greed Index (0-100 normalized) |
| 38 | `fear_greed_indicator` | Computed | 1.0 if F&G data available, 0.0 otherwise |

### Positions 39-59: External Normalized Columns (21 features - EXT_NORM_DIM)

**These positions contain advanced technical indicators from `prepare_and_run.py`**:

**Note**: The actual EXT_NORM_DIM is 21 (expanded from 8 to include additional technical features such as taker_buy_ratio derivatives and other indicators). The exact mapping of all 21 features should be documented based on the current implementation in `feature_config.py`.

Confirmed features include:
| Position | Feature | Source | Description |
|----------|---------|--------|-------------|
| 39 | `cvd_24h` | `df['cvd_24h']` | Cumulative Volume Delta (24-hour) |
| 40 | `cvd_7d` | `df['cvd_7d']` | Cumulative Volume Delta (7-day) |
| 41 | `yang_zhang_48h` | `df['yang_zhang_48h']` | Yang-Zhang volatility estimator (48h) |
| 42 | `yang_zhang_7d` | `df['yang_zhang_7d']` | Yang-Zhang volatility estimator (7d) |
| 43 | `garch_200h` | `df['garch_200h']` | GARCH(1,1) conditional volatility (200h) |
| 44 | `garch_14d` | `df['garch_14d']` | GARCH(1,1) conditional volatility (14d) |
| 45-59 | Additional 15 features | `df[...]` | Returns, taker buy ratio derivatives, and other technical indicators (see verify_63_features.py for complete list) |

**Note**: All values in positions 39-59 are normalized using `tanh()` to keep them in the range [-1, 1].

### Positions 60-61: Token Metadata (2 features)

| Position | Feature | Formula | Description |
|----------|---------|--------|-------------|
| 60 | `num_tokens_norm` | `num_tokens / max_num_tokens` | Number of tokens normalized |
| 61 | `token_id_norm` | `token_id / max_num_tokens` | Token ID normalized |

### Position 62: One-Hot Token Encoding (variable size)

With `max_num_tokens=1` (default), this adds 1 feature:

| Position | Feature | Description |
|----------|---------|-------------|
| 62 | Token 0 | One-hot encoding (1.0 for current token, 0.0 otherwise) |

**Total with max_num_tokens=1**:
- Bar (3) + MA features (4) + Indicators with validity flags (14) + Derived (2) + Agent (6) + Microstructure (3) + Bollinger Bands (2) + Event Metadata (3) + Fear & Greed (2) + External/EXT_NORM_DIM (21) + Token metadata (2) + Token one-hot (1) = **63 features**

**Breakdown**: 3 + 4 + 14 + 2 + 6 + 3 + 2 + 3 + 2 + 21 + 2 + 1 = **63** ✅

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
    Returns observation vector (63 features)
    ↓
 RL Agent (DistributionalPPO)
```

## Implementation Details

### Source Files

1. **mediator.py:1027-1174**: Main `_build_observation()` method
   - Extracts all data from row, state, and simulator
   - Calls `obs_builder.build_observation_vector()`
   - Falls back to legacy mode if obs_builder unavailable

2. **obs_builder.pyx:24-547**: Cython implementation
   - `build_observation_vector_c()`: Low-level C implementation with validity flag support
   - `build_observation_vector()`: Python-callable wrapper
   - Constructs the observation vector efficiently without Python overhead

3. **prepare_and_run.py:325-359**: Technical indicator creation
   - Uses `apply_offline_features()` to compute all indicators
   - Saves results to feather files

4. **feature_config.py**: Feature layout configuration
   - Defines `FEATURES_LAYOUT` structure
   - Computes `N_FEATURES` dynamically (currently 62)

### Helper Methods (mediator.py)

- `_get_safe_float(row, col, default)`: Safely extract float from row with fallback
- `_extract_market_data(row, state, mark_price, prev_price)`: Extract price and volume data
- `_extract_technical_indicators(row, sim, row_idx)`: Extract all technical indicators
- `_extract_norm_cols(row)`: Extract cvd, garch, yang_zhang into norm_cols array
- `_coerce_finite(value, default)`: Ensure finite float values

## Critical Size Changes (January 2025)

**Previous Setup (OUTDATED - 56 features):**
- `observation_space.shape = (56,)`
- No validity flags for indicators → ambiguity during warmup period

**Current Setup (CORRECTED - November 2025 - 62 features):**
- `observation_space = (N_FEATURES,)` where N_FEATURES correctly calculated as **62**
- `observation_space.shape = (62,)`
- Added 6 validity flags for indicators (rsi, macd, macd_signal, momentum, cci, obv)
- All 62 positions populated correctly

**Changes Made:**
1. Added validity flags for 6 indicators (rsi, macd, macd_signal, momentum, cci, obv)
2. Updated all index positions after position 7 (+6 shift)
3. Updated `feature_config.py` indicators block size from 13 to 19
4. Updated all tests to expect observation size 62
5. Created FEATURE_MAPPING_62.md and MIGRATION_GUIDE_56_TO_62.md

## Testing

Comprehensive tests should verify the observation vector structure:

1. **test_observation_size_and_non_zero()**: Verifies size=62 and non-zero content
2. **test_technical_indicators_present()**: Checks indicators are in correct positions
3. **test_cvd_garch_yangzhang_in_obs()**: Verifies specific indicators appear
4. **test_observations_in_training_env()**: Tests training scenario with obs_size=62
5. **test_observation_works_without_indicators()**: Tests fallback mode
6. **test_validity_flags()**: Tests that validity flags correctly distinguish warmup from valid data

**Note**: Verify that `tests/test_technical_indicators_in_obs.py` is updated to reflect N_FEATURES=62.

## Fallback Behavior

If technical indicators are missing from the dataframe:

- `_get_safe_float()` returns default values (0.0 or specified default)
- MA5/MA20: Return `NaN` → marked as invalid (valid_flag=0.0)
- RSI: Defaults to 50.0 (neutral), validity flag=0.0
- MACD/MACD_signal: Default to 0.0, validity flag=0.0
- Momentum: Defaults to 0.0, validity flag=0.0
- CCI: Defaults to 0.0, validity flag=0.0
- OBV: Defaults to 0.0, validity flag=0.0
- CVD/GARCH/Yang-Zhang: Default to 0.0 (neutral)
- Fear & Greed: Defaults to 50.0 (neutral)

The observation vector is always constructed with the correct size (62), but may contain more zeros if indicators are unavailable. **The validity flags allow the model to distinguish between "no data" and "meaningful zero".**

## Usage Examples

### In Training

```python
from trading_patchnew import TradingEnv
from train_model_multi_patch import create_envs

# Environment automatically loads data with technical indicators
env = TradingEnv(df=df_with_indicators, ...)

# Observation is automatically constructed with all 62 features
obs, info = env.reset()
assert obs.shape == (62,)
```

### In Production

```python
from mediator import Mediator

mediator = Mediator(env=trading_env)

# During stepping, observation is built with latest market data
obs = mediator._build_observation(row=current_row, state=state, mark_price=current_price)
```

## Key Design Decisions

1. **Why EXT_NORM_DIM=21?**: Allows 21 external indicators (cvd, garch, yang_zhang, returns, taker_buy_ratio derivatives, etc.) without hardcoding positions. This was expanded from 8 to accommodate more technical features.
2. **Why tanh normalization?**: Keeps values in [-1, 1] range, suitable for neural networks
3. **Why fallback to legacy?**: Ensures backward compatibility if obs_builder is not compiled
4. **Why separate validity flags?**: Allows model to distinguish between "zero value" and "value not available yet" during warmup period
5. **Why 62 features instead of 56?**: Added 6 validity flags (rsi, macd, macd_signal, momentum, cci, obv) to eliminate semantic ambiguity

## Future Extensions

To add new indicators:

1. Add column to `prepare_and_run.py` via `apply_offline_features()`
2. Map to `norm_cols` position in `mediator._extract_norm_cols()`
3. Update this documentation
4. Add tests in `test_technical_indicators_in_obs.py`

## Recompilation Required

After updating `lob_state_cython.pyx` or `obs_builder.pyx`, you MUST recompile:

```bash
python setup.py build_ext --inplace
```

This will update `lob_state_cython.N_FEATURES` to return 62.

---

**Last Updated**: 2025-11-16 (Size updated from 56 to 62 with validity flags)
**Authors**: Technical Indicators Integration Task + Validity Flags Migration
**Related Files**: `mediator.py`, `obs_builder.pyx`, `prepare_and_run.py`, `feature_config.py`, `lob_state_cython.pyx`, `trading_patchnew.py`
**Migration Guide**: See `MIGRATION_GUIDE_56_TO_62.md` for detailed migration instructions
