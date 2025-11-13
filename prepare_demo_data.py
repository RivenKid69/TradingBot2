#!/usr/bin/env python3
"""
prepare_demo_data.py
--------------------
Creates minimal demo data for testing the training pipeline when real market data
is not available. This generates synthetic 4h OHLCV data for testing purposes only.
(Changed from 1h to 4h timeframe)

IMPORTANT: Run this script in your project's Python environment (not system Python)
to ensure numpy/pandas are available. For example:
    source venv/bin/activate  # or conda activate your_env
    python prepare_demo_data.py --rows 2000 --symbols BTCUSDT,ETHUSDT

Usage:
    python prepare_demo_data.py [--rows NUM_ROWS] [--symbols SYMBOL1,SYMBOL2]

Example:
    # Generate 2000 4h bars (~333 days) of synthetic data for 2 symbols
    python prepare_demo_data.py --rows 2000 --symbols BTCUSDT,ETHUSDT

    # Generate 5000 4h bars with custom start date
    python prepare_demo_data.py --rows 5000 --symbols BTCUSDT,ETHUSDT,BNBUSDT --start-date 2023-01-01
"""
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_synthetic_ohlcv(
    symbol: str,
    start_ts: int,
    num_bars: int,
    base_price: float = 30000.0,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """Generate synthetic 4h OHLCV data for testing (changed from 1h to 4h timeframe)."""
    np.random.seed(hash(symbol) % (2**32))

    timestamps = [start_ts + i * 14400 for i in range(num_bars)]  # Changed from 3600 (1h) to 14400 (4h)

    # Generate price walk
    returns = np.random.normal(0, volatility, num_bars)
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


def generate_fear_greed(start_ts: int, num_bars: int) -> pd.DataFrame:
    """Generate synthetic Fear & Greed index (4h bars, changed from 1h)."""
    np.random.seed(42)
    timestamps = [start_ts + i * 14400 for i in range(num_bars)]  # Changed from 3600 (1h) to 14400 (4h)

    # Generate mean-reverting fear/greed
    values = []
    current = 50
    for _ in range(num_bars):
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
        description="Generate synthetic demo data for training pipeline testing"
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=2000,
        help="Number of 4h bars to generate (default: 2000 = ~333 days, changed from hourly)",
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

    args = parser.parse_args()

    # Parse arguments
    symbols = [s.strip() for s in args.symbols.split(",")]
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp())
    out_dir = Path(args.out_dir)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.rows} 4h bars of synthetic data for {len(symbols)} symbols...")
    print(f"Start date: {start_dt.isoformat()}")
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
        df = generate_synthetic_ohlcv(symbol, start_ts, args.rows, base_price)

        # Add Fear & Greed
        fng = pd.read_csv(fng_path)
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            fng[["timestamp", "fear_greed_value"]],
            on="timestamp",
            direction="backward",
        )
        df["fear_greed_value"] = df["fear_greed_value"].ffill()

        # Save as feather
        out_path = out_dir / f"{symbol}.feather"
        df.to_feather(out_path)
        print(f"✓ Generated {out_path} ({len(df)} rows)")

    print()
    print(f"Demo data generation complete!")
    print(f"Generated {len(symbols)} feather files in {out_dir}")
    print()
    print("You can now run training with:")
    print("  python train_model_multi_patch.py --config configs/config_train.yaml")


if __name__ == "__main__":
    main()
