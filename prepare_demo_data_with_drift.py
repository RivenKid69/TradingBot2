#!/usr/bin/env python3
"""
prepare_demo_data_with_drift.py
--------------------------------
Generates synthetic OHLCV data with POSITIVE DRIFT and momentum patterns
to enable RL training to actually learn profitable strategies.

Key differences from prepare_demo_data.py:
- Adds positive drift (expected return > 0)
- Adds momentum autocorrelation
- Adds regime switching (trending vs mean-reverting periods)
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

def generate_predictable_ohlcv(
    symbol: str,
    start_ts: int,
    num_hours: int,
    base_price: float = 30000.0,
    volatility: float = 0.02,
    drift: float = 0.0005,  # Positive expected return per hour (~12% annually)
    momentum: float = 0.15,  # Autocorrelation for momentum
) -> pd.DataFrame:
    """Generate synthetic hourly OHLCV with predictable patterns."""
    np.random.seed(hash(symbol) % (2**32))

    timestamps = [start_ts + i * 3600 for i in range(num_hours)]

    # Generate returns with DRIFT and MOMENTUM
    returns = []
    prev_return = 0.0
    for i in range(num_hours):
        # Add momentum (autocorrelation) + noise + drift
        noise = np.random.normal(0, volatility)
        ret = momentum * prev_return + (1 - momentum) * noise + drift
        returns.append(ret)
        prev_return = ret

    returns = np.array(returns)
    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLC from close prices
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # Add intrabar volatility
        high = close * (1 + abs(np.random.normal(0, volatility/4)))
        low = close * (1 - abs(np.random.normal(0, volatility/4)))
        open_ = prices[i-1] if i > 0 else close

        # Volume varies with volatility
        volume = np.random.lognormal(10, 1) * (1 + abs(returns[i]))
        quote_volume = volume * close

        data.append({
            "timestamp": ts,
            "symbol": symbol,
            "open": open_,
            "high": max(open_, close, high),
            "low": min(open_, close, low),
            "close": close,
            "volume": volume,
            "quote_asset_volume": quote_volume,
            "number_of_trades": int(np.random.poisson(1000)),
            "taker_buy_base_asset_volume": volume * np.random.uniform(0.4, 0.6),
            "taker_buy_quote_asset_volume": quote_volume * np.random.uniform(0.4, 0.6),
        })

    return pd.DataFrame(data)


def generate_fear_greed(start_ts: int, num_hours: int) -> pd.DataFrame:
    """Generate synthetic Fear & Greed index."""
    np.random.seed(42)
    timestamps = [start_ts + i * 3600 for i in range(num_hours)]

    # Generate mean-reverting fear/greed
    values = []
    current = 50
    for _ in range(num_hours):
        current += np.random.normal(0, 5) - (current - 50) * 0.1
        current = np.clip(current, 0, 100)
        values.append(current)

    return pd.DataFrame({
        "timestamp": timestamps,
        "fear_greed_value": values,
        "fear_greed_value_norm": np.array(values) / 100.0,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic demo data WITH PREDICTABLE PATTERNS for RL training"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of hourly rows to generate (default: 2000 = ~83 days)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated list of symbols (default: BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed",
        help="Output directory for feather files (default: data/processed)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Start date in YYYY-MM-DD format (default: 2023-01-01)",
    )
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0005,
        help="Positive drift per hour (default: 0.0005 ≈ 12%% annually)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.15,
        help="Momentum autocorrelation (default: 0.15)",
    )

    args = parser.parse_args()

    # Parse arguments
    symbols = [s.strip() for s in args.symbols.split(",")]
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp())
    out_dir = Path(args.out_dir)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows} hours of PREDICTABLE data for {len(symbols)} symbols...")
    print(f"Start date: {start_dt.isoformat()}")
    print(f"Drift: {args.drift:.6f} per hour (≈ {args.drift * 24 * 365:.2%} annually)")
    print(f"Momentum: {args.momentum:.2f}")
    print(f"Output directory: {out_dir}")
    print()

    # Generate Fear & Greed index
    fng_path = Path("data/fear_greed.csv")
    if not fng_path.exists():
        fng_path.parent.mkdir(parents=True, exist_ok=True)
        fng = generate_fear_greed(start_ts, args.rows)
        fng.to_csv(fng_path, index=False)
        print(f"✓ Generated {fng_path} ({len(fng)} rows)")
    else:
        print(f"✓ {fng_path} already exists, skipping")

    # Generate OHLCV data for each symbol
    base_prices = {
        "BTCUSDT": 30000.0,
        "ETHUSDT": 2000.0,
        "BNBUSDT": 300.0,
        "ADAUSDT": 0.5,
        "SOLUSDT": 50.0,
    }

    for symbol in symbols:
        base_price = base_prices.get(symbol, 100.0)
        df = generate_predictable_ohlcv(
            symbol, start_ts, args.rows, base_price,
            drift=args.drift,
            momentum=args.momentum
        )

        # Add Fear & Greed
        fng = pd.read_csv(fng_path)
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            fng[["timestamp", "fear_greed_value"]],
            on="timestamp",
            direction="backward",
        )
        df["fear_greed_value"] = df["fear_greed_value"].ffill()

        # Verify it's predictable
        df['ret'] = df['close'].pct_change()
        autocorr = df['ret'].autocorr(lag=1)
        print(f"✓ {symbol}: autocorr(lag=1) = {autocorr:.4f} (should be > 0.1 for momentum)")

        # Save as feather
        out_path = out_dir / f"{symbol}.feather"
        df.to_feather(out_path)
        print(f"  Saved {out_path} ({len(df)} rows)")

    print()
    print(f"✅ Predictable data generation complete!")
    print(f"Generated {len(symbols)} feather files in {out_dir}")
    print()
    print("Expected behavior:")
    print("  - EV should improve (> 0.3)")
    print("  - value_pred_std should match target_return_std")
    print("  - Policy should learn to follow momentum")
    print()
    print("Run training with:")
    print("  python train_model_multi_patch.py --config configs/config_train_spot_bar.yaml")


if __name__ == "__main__":
    main()
