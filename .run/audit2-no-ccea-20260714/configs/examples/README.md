# Configuration Examples

This directory contains well-documented example configurations for different research/simulation and deployment modes.
Copy and customize these for your own use.

## Available Examples

| File | Mode | Asset Class | Description |
|------|------|-------------|-------------|
| `example_train_stocks.yaml` | train | equity | RL model training for stocks |
| `example_live_stocks.yaml` | live | equity | Live execution (Alpaca) |
| `example_train_crypto.yaml` | train | crypto | RL model training for digital assets (optional) |
| `example_backtest_crypto.yaml` | sim | crypto | Backtesting/simulation (example config; adaptable) |
| `example_live_crypto.yaml` | live | crypto | Live execution (Binance; optional) |

## Quick Start

### 1. Training a Model (Equities-first example)

```bash
# Copy the example
cp configs/examples/example_train_stocks.yaml configs/my_train.yaml

# Edit to customize (data paths, model params, etc.)
# Then run:
python train_model_multi_patch.py --config configs/my_train.yaml
```

### 2. Running a Backtest

```bash
# Copy the example
cp configs/examples/example_backtest_crypto.yaml configs/my_backtest.yaml

# Edit to customize (data path, model checkpoint)
# Then run:
python script_backtest.py --config configs/my_backtest.yaml
```

### 3. Live Execution

**IMPORTANT**: Before running live, always:
1. Run `python scripts/doctor.py` to verify environment
2. Test with `--dry-run` flag first
3. Start with small position limits

```bash
# Copy the example
cp configs/examples/example_live_crypto.yaml configs/my_live.yaml

# Set environment variables
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"

# Test with dry-run first!
python script_live.py --config configs/my_live.yaml --dry-run

# When ready for live (small limits first!)
python script_live.py --config configs/my_live.yaml
```

## Configuration Hierarchy

Configs can be layered using the `_base` field:

```yaml
# my_config.yaml
_base: configs/examples/example_train_crypto.yaml  # Inherit from base

# Override specific values
model:
  params:
    learning_rate: 5.0e-5  # Override just this value
```

## Environment Variables

API keys should NEVER be hardcoded. Use environment variables:

| Variable | Description |
|----------|-------------|
| `BINANCE_API_KEY` | Binance API key |
| `BINANCE_API_SECRET` | Binance API secret |
| `ALPACA_API_KEY` | Alpaca API key |
| `ALPACA_API_SECRET` | Alpaca API secret |
| `POLYGON_API_KEY` | Polygon.io API key |

## Common Configuration Sections

### Asset Class

```yaml
asset_class: crypto  # or "equity"
data_vendor: binance  # or "alpaca", "polygon"
```

This auto-configures:
- FeeProvider (maker/taker for crypto, regulatory fees for equity)
- SlippageProvider (appropriate profiles)
- TradingHoursAdapter (24/7 for crypto, NYSE hours for equity)

### Risk Settings

```yaml
risk:
  enabled: true
  max_abs_position_notional: 10000.0  # Max position size in quote currency
  daily_loss_limit: 500.0  # Max daily loss before pause
  pause_seconds_on_violation: 300  # 5-minute pause on violations
```

### Model Parameters (Training)

```yaml
model:
  params:
    learning_rate: 1.0e-4
    gamma: 0.99  # Discount factor (MUST match reward.gamma!)
    n_steps: 2048  # Rollout length
    batch_size: 64  # Minibatch size
    use_twin_critics: true  # Enable twin critics (recommended)
```

## Troubleshooting

### "Filters file stale" error
Run `python scripts/fetch_binance_filters.py --out data/binance_filters.json`

### "API key not found" error
Ensure environment variables are set (not in config file!)

### Training loss not decreasing
- Check `gamma` matches between `model.params` and `reward`
- Try reducing `learning_rate`
- Ensure data has sufficient variance

## See Also

- [claude.md](../../claude.md) - Full documentation
- [docs/pipeline.md](../../docs/pipeline.md) - Signal pipeline
- [docs/twin_critics.md](../../docs/twin_critics.md) - Twin critics architecture
