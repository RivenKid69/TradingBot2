# Stock Trading Guide

Complete guide for training, backtesting, and live trading US equities using CustodiaCloud (CCEA).

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Data Setup](#data-setup)
4. [Training Configuration](#training-configuration)
5. [Backtesting](#backtesting)
6. [Live Trading](#live-trading)
7. [Stock vs Crypto Differences](#stock-vs-crypto-differences)
8. [Features & Indicators](#features--indicators)
9. [Risk Management](#risk-management)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The platform supports US equities trading through multiple data providers:

| Provider | Type | Data | Trading | Cost |
|----------|------|------|---------|------|
| **Alpaca** | Broker + Data | REST + WebSocket | Yes | Free (IEX) / Paid (SIP) |
| **Polygon** | Data Only | REST + WebSocket | No | Paid |
| **Yahoo Finance** | Indices | REST | No | Free |

### Supported Symbols

**Tech Stocks**: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA

**Index ETFs**: SPY (S&P 500), QQQ (Nasdaq 100), IWM (Russell 2000)

**Precious Metals ETFs**: GLD, IAU, SGOL (Gold), SLV (Silver)

**Macro Indicators**: ^VIX, DX-Y.NYB (DXY), ^TNX (10Y Treasury)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install alpaca-py  # Alpaca SDK
pip install yfinance   # Yahoo Finance (for VIX, indices)
```

### 2. Set Environment Variables

```bash
# Windows (PowerShell)
$env:ALPACA_API_KEY = "your_api_key"
$env:ALPACA_API_SECRET = "your_api_secret"

# Linux/Mac
export ALPACA_API_KEY="your_api_key"
export ALPACA_API_SECRET="your_api_secret"
```

### 3. Download Data

```bash
# Download popular tech stocks (3 years of history)
python scripts/download_stock_data.py --popular --start 2021-01-01

# Download VIX for market regime indicator
python scripts/download_stock_data.py --vix --start 2021-01-01

# Download all macro indicators
python scripts/download_stock_data.py --macro --start 2021-01-01
```

### 4. Train Model

```bash
python train_model_multi_patch.py --config configs/config_train_stocks.yaml
```

### 5. Backtest

```bash
python script_backtest.py --config configs/config_backtest_stocks.yaml
```

### 6. Live Trading (Paper)

```bash
python script_live.py --config configs/config_live_alpaca.yaml --paper
```

---

## Data Setup

### Downloading Stock Data

The `scripts/download_stock_data.py` script supports multiple providers with auto-detection:

```bash
# Basic usage - download specific symbols
python scripts/download_stock_data.py --symbols AAPL MSFT GOOGL --start 2020-01-01

# Download from file
python scripts/download_stock_data.py --symbols-file data/universe/sp500.txt

# Download with hourly data, resample to 4h
python scripts/download_stock_data.py --symbols AAPL --timeframe 1h --resample 4h

# Include extended hours (pre-market, after-hours)
python scripts/download_stock_data.py --symbols AAPL --include-extended
```

### Convenience Flags

| Flag | Symbols | Provider |
|------|---------|----------|
| `--popular` | AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA | Alpaca |
| `--vix` | ^VIX, ^VXN | Yahoo Finance |
| `--macro` | ^VIX, DX-Y.NYB, ^TNX, ^TYX | Yahoo Finance |

### Data Output Structure

```
data/
├── raw_stocks/           # Downloaded raw data
│   ├── AAPL.parquet
│   ├── MSFT.parquet
│   ├── VIX.parquet       # ^VIX sanitized to VIX
│   └── DX-Y_NYB.parquet  # DX-Y.NYB sanitized
├── stocks/               # Processed data (optional)
└── universe/
    └── alpaca_symbols.json
```

### Data Providers

#### Alpaca (Default for Stocks)

```bash
# Using free IEX feed
python scripts/download_stock_data.py --symbols AAPL --provider alpaca --feed iex

# Using paid SIP feed (real-time)
python scripts/download_stock_data.py --symbols AAPL --provider alpaca --feed sip
```

#### Polygon (Alternative)

```bash
python scripts/download_stock_data.py --symbols AAPL --provider polygon --api-key YOUR_KEY
```

#### Yahoo Finance (Indices)

Auto-selected for symbols starting with `^` or containing special characters:

```bash
python scripts/download_stock_data.py --symbols ^VIX DX-Y.NYB --start 2020-01-01
```

---

## Training Configuration

### Key Configuration File

`configs/config_train_stocks.yaml` - Main training configuration for stocks

### Stock-Specific Settings

```yaml
# Asset class identification
asset_class: equity
data_vendor: alpaca

# Data paths
data:
  paths:
    - "data/raw_stocks/*.parquet"
    - "data/stocks/*.parquet"
  filter_trading_hours: true      # Only use regular hours data
  include_extended_hours: false   # Exclude pre/post market

# US market session
env:
  session:
    calendar: us_equity           # NYSE calendar
    extended_hours: false

# Commission-free trading with regulatory fees
fees:
  structure: flat
  maker_bps: 0.0
  taker_bps: 0.0
  regulatory:
    enabled: true
    sec_fee_per_million: 27.80    # SEC fee on sells
    taf_fee_per_share: 0.000166   # TAF fee per share
    taf_fee_cap: 8.30             # Max TAF per trade
```

### Training Command

```bash
# Standard training
python train_model_multi_patch.py --config configs/config_train_stocks.yaml

# With PBT (Population-Based Training)
python train_model_multi_patch.py --config configs/config_train_stocks.yaml \
    --pbt --population-size 8

# Resume from checkpoint
python train_model_multi_patch.py --config configs/config_train_stocks.yaml \
    --resume models/stocks/checkpoint.zip
```

### Signal-Only Training

For learning market timing without execution simulation:

```bash
python train_model_multi_patch.py --config configs/config_train_signal_only_stocks.yaml
```

---

## Backtesting

### Configuration

`configs/config_backtest_stocks.yaml`:

```yaml
mode: backtest
asset_class: equity

data:
  start_ts: 1735689600    # 2025-01-01
  end_ts: 1759276800      # 2025-10-01

env:
  initial_cash: 100000.0
  max_position: 1.0

output:
  save_trades: true
  save_equity_curve: true
  metrics_path: "artifacts/stocks/backtest/metrics.json"
```

### Running Backtest

```bash
# Standard backtest
python script_backtest.py --config configs/config_backtest_stocks.yaml

# With specific model
python script_backtest.py --config configs/config_backtest_stocks.yaml \
    --model models/stocks/best_model.zip

# Multiple symbols
python script_backtest.py --config configs/config_backtest_stocks.yaml \
    --symbols AAPL MSFT GOOGL
```

### Output Files

```
artifacts/stocks/backtest/
├── metrics.json          # Performance metrics
├── trades.csv            # Trade log
└── equity_curve.csv      # Equity curve
```

---

## Live Trading

### Configuration

`configs/config_live_alpaca.yaml`:

```yaml
mode: live
exchange:
  vendor: "alpaca"
  market_type: "EQUITY"
  alpaca:
    paper: true           # Set to false for real trading
    feed: "iex"           # "iex" (free) or "sip" (paid)
    extended_hours: false

execution_params:
  slippage_bps: 1.0       # Tighter spreads than crypto
  tif: DAY                # Day order (expires at close)

risk:
  max_abs_position_notional: 50000.0
  max_order_notional: 10000.0
  daily_loss_limit: 1000.0
```

### Running Live Trading

```bash
# Paper trading (recommended for testing)
python script_live.py --config configs/config_live_alpaca.yaml --paper

# Real trading (use with caution!)
python script_live.py --config configs/config_live_alpaca.yaml --live

# With extended hours
python script_live.py --config configs/config_live_alpaca.yaml --extended-hours
```

### Asset Class Auto-Detection

The script automatically detects asset class from config:

```python
# Priority order:
# 1. Explicit: --asset-class equity
# 2. Vendor: vendor=alpaca → equity
# 3. Market type: market_type=EQUITY → equity
# 4. Default: crypto (backward compatible)
```

### Position Synchronization

Live trading includes automatic position sync with the broker:

```yaml
# In config_live_alpaca.yaml
position_sync:
  enabled: true
  interval_sec: 30.0
  tolerance: 0.01         # 1% tolerance for position mismatch
  auto_reconcile: true
```

---

## Stock vs Crypto Differences

### Trading Hours

| Aspect | Crypto (Binance) | Stocks (Alpaca) |
|--------|------------------|-----------------|
| **Regular Hours** | 24/7 | 9:30 AM - 4:00 PM ET |
| **Pre-Market** | N/A | 4:00 AM - 9:30 AM ET |
| **After-Hours** | N/A | 4:00 PM - 8:00 PM ET |
| **Holidays** | None | NYSE holidays |

### Fee Structure

| Fee Type | Crypto | Stocks |
|----------|--------|--------|
| **Commission** | 0.02-0.04% | $0 (commission-free) |
| **SEC Fee** | N/A | $27.80 per $1M (sells only) |
| **TAF Fee** | N/A | $0.000166/share (max $8.30) |

### Execution Parameters

| Parameter | Crypto | Stocks |
|-----------|--------|--------|
| Default Spread | 5 bps | 2 bps |
| Impact Coefficient | 0.1 | 0.05 |
| Time-in-Force | GTC | DAY |
| Min Trade Size | LOT_SIZE filter | 1 share |

### Order Types

| Order Type | Crypto | Stocks |
|------------|--------|--------|
| Market | Yes | Regular hours only |
| Limit | Yes | Yes (all hours) |
| Bracket | No (via OCO) | Yes (native) |
| Extended Hours | N/A | Limit only |

---

## Features & Indicators

### Stock-Specific Features

The `stock_features.py` module provides equity-specific indicators:

#### 1. VIX Integration

Analogous to crypto Fear & Greed index:

| VIX Level | Regime | Market Condition |
|-----------|--------|------------------|
| < 12 | Low | Complacency |
| 12-20 | Normal | Normal conditions |
| 20-30 | Elevated | Elevated fear |
| > 30 | Extreme | Crisis/panic |

#### 2. Market Regime Indicator

Bull/Bear/Sideways classification based on:

- SPY SMA crossovers (20/50 day)
- VIX levels
- Price momentum

#### 3. Relative Strength

RS vs benchmark (SPY/QQQ) for momentum strategies:

- 20-day RS (short-term)
- 50-day RS (medium-term)

### Using Stock Features in Training

```yaml
# In config_train_stocks.yaml
features:
  stock_specific:
    enabled: true
    vix_integration: true
    market_regime: true
    relative_strength: true
    benchmark: SPY
```

### Macro Data Integration

Download and integrate macro indicators:

```bash
# Download VIX, DXY, Treasury yields
python scripts/download_stock_data.py --macro --start 2020-01-01
```

Correlation with gold (for precious metals trading):

| Indicator | Correlation with Gold |
|-----------|----------------------|
| DXY (Dollar Index) | Strong inverse |
| Real Yields (TIPS) | Inverse |
| VIX | Positive (fear) |
| Gold/Silver Ratio | Mean-reverts 60-80 |

---

## Risk Management

### PDT Rule

Pattern Day Trader rule for accounts < $25,000:

```yaml
risk:
  pdt_rule:
    enabled: true           # Enable for accounts < $25k
    min_equity: 25000.0
    max_day_trades: 3       # Max 3 day trades per 5 days
```

### Stock-Specific Risk Guards

```yaml
risk:
  max_position: 1.0              # 100% max position
  max_drawdown: 0.20             # 20% max drawdown
  daily_loss_limit: 0.05         # 5% daily loss limit
  max_abs_position_notional: 50000.0
```

### Extended Hours Risk

Extended hours have wider spreads and lower liquidity:

```yaml
env:
  session:
    extended_hours: false         # Disable by default
    extended_hours_spread_mult: 2.0  # 2x spread in extended hours
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No data returned" | Symbol not available | Check symbol format, try Yahoo for indices |
| "API rate limit" | Too many requests | Increase `rate_limit_delay` in config |
| "Market closed" | Trading outside hours | Wait for market hours or enable extended hours |
| "Insufficient buying power" | Paper account limit | Check Alpaca paper account balance |

### Data Download Issues

```bash
# Check if Alpaca credentials are set
echo $ALPACA_API_KEY

# Test with verbose mode
python scripts/download_stock_data.py --symbols AAPL --verbose

# Force re-download (skip existing check)
python scripts/download_stock_data.py --symbols AAPL --no-skip-existing
```

### Live Trading Issues

```bash
# Check position sync
python -c "from services.position_sync import reconcile_alpaca_state; print(reconcile_alpaca_state())"

# Verify API connection
python -c "from adapters.alpaca.market_data import AlpacaMarketDataAdapter; a = AlpacaMarketDataAdapter(); print(a.get_bars('AAPL', '1d', limit=1))"
```

### Testing

```bash
# Run all stock-related tests
pytest tests/test_stock*.py -v

# Test adapters
pytest tests/test_alpaca_adapters.py -v

# Test Phase 9 live trading
pytest tests/test_phase9_live_trading.py -v
```

---

## File Reference

### Configuration Files

| File | Description |
|------|-------------|
| `configs/config_train_stocks.yaml` | Training configuration |
| `configs/config_backtest_stocks.yaml` | Backtest configuration |
| `configs/config_live_alpaca.yaml` | Live trading (Alpaca) |
| `configs/config_train_signal_only_stocks.yaml` | Signal-only training |

### Key Source Files

| File | Description |
|------|-------------|
| `scripts/download_stock_data.py` | Data download script |
| `scripts/fetch_alpaca_universe.py` | Universe management |
| `data_loader_multi_asset.py` | Multi-asset data loader |
| `stock_features.py` | Stock-specific features |
| `services/stock_risk_guards.py` | Stock risk management |
| `adapters/alpaca/` | Alpaca adapters |
| `adapters/yahoo/` | Yahoo Finance adapters |
| `services/position_sync.py` | Position synchronization |
| `services/session_router.py` | Session-aware routing |

### Test Files

| File | Description |
|------|-------------|
| `tests/test_stock_*.py` | Stock functionality tests |
| `tests/test_alpaca_adapters.py` | Alpaca adapter tests |
| `tests/test_phase9_live_trading.py` | Live trading tests |

---

## Next Steps

1. **Set up API credentials** - Get Alpaca API keys from [alpaca.markets](https://alpaca.markets)
2. **Download historical data** - Use `download_stock_data.py` script
3. **Train a model** - Start with `config_train_stocks.yaml`
4. **Backtest** - Validate performance with `config_backtest_stocks.yaml`
5. **Paper trade** - Test live with `--paper` flag
6. **Monitor** - Check TensorBoard logs and position sync

---

**Last Updated**: 2025-11-28
**Version**: 1.0.0
