#!/usr/bin/env python3
"""
Convert cached historical data to training format with technical indicators

This script:
1. Loads data from cache/http/*.parquet
2. Adds technical indicators (to create predictable patterns)
3. Tests if result is learnable
4. Saves to data/train/*.feather (or csv)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to create predictable patterns"""
    df = df.copy()

    # Moving averages (momentum signals)
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # Price position relative to MA (mean reversion)
    df['close_over_sma20'] = df['close'] / df['sma_20'] - 1
    df['close_over_sma50'] = df['close'] / df['sma_50'] - 1

    # MA crossover (momentum)
    df['ma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI (overbought/oversold)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

    # Bollinger Bands (mean reversion)
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # ATR (volatility)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)

    # Price momentum
    df['momentum_1'] = df['close'].pct_change(1)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)

    # Trend strength
    df['adx_pos'] = ((df['high'] - df['high'].shift(1)).clip(lower=0)).rolling(14).mean()
    df['adx_neg'] = ((df['low'].shift(1) - df['low']).clip(lower=0)).rolling(14).mean()

    return df


def analyze_predictability(df: pd.DataFrame, symbol: str):
    """Analyze if data (with indicators) is learnable"""
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['return_next'] = df['return'].shift(-1)

    print(f"\n{'='*70}")
    print(f"PREDICTABILITY ANALYSIS: {symbol}")
    print(f"{'='*70}\n")

    # Raw autocorrelation
    autocorr_1 = df['return'].autocorr(lag=1)
    print(f"Raw autocorrelation: {autocorr_1:+.4f}  ", end='')
    if abs(autocorr_1) > 0.10:
        print("✅ Strong")
    elif abs(autocorr_1) > 0.05:
        print("⚠️  Weak")
    else:
        print("❌ Random walk")
    print()

    # Test indicator signals
    tests = [
        ("RSI < 30 (oversold)", df[df['rsi'] < 30]['return_next']),
        ("RSI > 70 (overbought)", df[df['rsi'] > 70]['return_next']),
        ("Price < BB lower", df[df['bb_position'] < 0.2]['return_next']),
        ("Price > BB upper", df[df['bb_position'] > 0.8]['return_next']),
        ("MA bullish cross", df[(df['ma_cross'] == 1) & (df['ma_cross'].shift(1) == 0)]['return_next']),
        ("MACD > Signal", df[df['macd'] > df['macd_signal']]['return_next']),
        ("Close > SMA20", df[df['close'] > df['sma_20']]['return_next']),
    ]

    found_signals = 0
    total_edge = 0

    for name, returns in tests:
        if len(returns) > 30:
            mean_ret = returns.mean()
            pct_positive = (returns > 0).sum() / len(returns) * 100
            edge = pct_positive - 50

            print(f"{name:25s}: ", end='')
            print(f"{pct_positive:5.1f}% ↑  edge={edge:+5.1f}%  ", end='')

            if abs(edge) > 5:
                print("✅ STRONG")
                found_signals += 2
                total_edge += abs(edge)
            elif abs(edge) > 2:
                print("✓ Good")
                found_signals += 1
                total_edge += abs(edge)
            else:
                print("✗ Weak")

    print()
    print(f"Signals found: {found_signals}")
    print(f"Total edge: {total_edge:.1f}%")
    print()

    # Verdict
    print("="*70)
    if found_signals >= 5:
        print("✅✅ EXCELLENT - Strong predictable patterns!")
        print("   Technical indicators ADD significant structure")
        print("   → Expected EV > 0.5")
        print("   → Training will work well")
        verdict = "excellent"
    elif found_signals >= 3:
        print("✅ GOOD - Moderate predictable patterns")
        print("   Technical indicators help create structure")
        print("   → Expected EV 0.3-0.5")
        print("   → Training should work")
        verdict = "good"
    elif found_signals >= 1:
        print("⚠️  MARGINAL - Weak patterns")
        print("   Some signals detected but weak")
        print("   → Expected EV 0.1-0.3")
        print("   → Training will be difficult")
        verdict = "marginal"
    else:
        print("❌ POOR - No predictable patterns")
        print("   Even with indicators, no clear signals")
        print("   → Expected EV ≈ 0")
        print("   → Training likely won't work")
        verdict = "poor"

    print("="*70 + "\n")

    return verdict


def convert_cached_data(
    input_dir: Path,
    output_dir: Path,
    add_indicators: bool = True,
    output_format: str = "feather"
):
    """Convert cached parquet files to training format"""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_dir.glob("*.parquet"))

    if not parquet_files:
        print(f"❌ No parquet files found in {input_dir}")
        return

    print(f"Found {len(parquet_files)} cached data files")
    print(f"Output: {output_dir}")
    print(f"Format: {output_format}")
    print(f"Add indicators: {add_indicators}")
    print()

    all_good = True

    for parquet_file in parquet_files:
        symbol = parquet_file.stem.split('_')[0]  # BTCUSDT_1h → BTCUSDT
        print(f"\nProcessing {symbol}...")

        # Load
        df = pd.read_parquet(parquet_file)
        print(f"  Loaded {len(df)} rows")

        # Rename/convert columns
        if 'ts_ms' in df.columns:
            df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms')
        elif 'timestamp' not in df.columns:
            print(f"  ❌ No timestamp column!")
            continue

        # Ensure required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"  ❌ Missing columns: {missing}")
            continue

        # Add symbol column if missing
        if 'symbol' not in df.columns:
            df['symbol'] = symbol

        # Add indicators
        if add_indicators:
            print(f"  Adding technical indicators...")
            df = add_technical_indicators(df)
            print(f"  → {len(df.columns)} total columns")

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Drop NaN from indicators
        initial_rows = len(df)
        df = df.dropna()
        dropped = initial_rows - len(df)
        if dropped > 0:
            print(f"  Dropped {dropped} rows with NaN (from indicator warmup)")

        # Save
        output_file = output_dir / f"{symbol}.{output_format}"
        if output_format == "feather":
            df.to_feather(output_file)
        elif output_format == "csv":
            df.to_csv(output_file, index=False)
        elif output_format == "parquet":
            df.to_parquet(output_file, index=False)
        else:
            print(f"  ❌ Unknown format: {output_format}")
            continue

        print(f"  ✓ Saved {len(df)} rows to {output_file}")

        # Analyze predictability
        verdict = analyze_predictability(df, symbol)
        if verdict in ["poor", "marginal"]:
            all_good = False

    print("\n" + "="*70)
    if all_good:
        print("✅ All data processed successfully!")
        print("\nYou can now train with:")
        print(f"  python train_model_multi_patch.py \\")
        print(f"    --config configs/config_train_spot_bar.yaml \\")
        print(f"    --n-envs 4")
        print("\nExpected results:")
        print("  - EV > 0.3 (value function learns)")
        print("  - value_pred_std matches target_std")
        print("  - Agent learns from indicator signals")
    else:
        print("⚠️  Some data has weak predictability")
        print("\nOptions:")
        print("  1. Try longer timeframe (4h instead of 1h)")
        print("  2. Add more/different indicators")
        print("  3. Use different time period (2024 instead of 2021?)")
        print("  4. Try different assets")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert cached data to training format"
    )
    parser.add_argument(
        "--input", type=str, default="cache/http",
        help="Input directory with parquet files"
    )
    parser.add_argument(
        "--output", type=str, default="data/train",
        help="Output directory"
    )
    parser.add_argument(
        "--no-indicators", action="store_true",
        help="Don't add technical indicators (not recommended!)"
    )
    parser.add_argument(
        "--format", type=str, default="feather",
        choices=["feather", "csv", "parquet"],
        help="Output format"
    )

    args = parser.parse_args()

    convert_cached_data(
        input_dir=args.input,
        output_dir=args.output,
        add_indicators=not args.no_indicators,
        output_format=args.format
    )


if __name__ == "__main__":
    main()
