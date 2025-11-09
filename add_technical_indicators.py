#!/usr/bin/env python3
"""
Add technical indicators to create predictable patterns
Even if raw prices are random walk, indicators can create structure!
"""
import pandas as pd
import numpy as np
from pathlib import Path

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators that create predictable patterns"""
    df = df.copy()

    # Moving averages
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # Price relative to MA (mean reversion signal)
    df['close_over_sma20'] = df['close'] / df['sma_20'] - 1  # % above/below MA
    df['close_over_sma50'] = df['close'] / df['sma_50'] - 1

    # MA crossover (momentum signal)
    df['ma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # RSI (overbought/oversold)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)  # Buy signal
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)  # Sell signal

    # Bollinger Bands (mean reversion)
    df['bb_middle'] = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # Volatility
    df['atr'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'],
                     abs(x['high'] - x['close']),
                     abs(x['low'] - x['close'])),
        axis=1
    ).rolling(14).mean()

    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Price momentum
    df['momentum_1'] = df['close'].pct_change(1)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)

    return df

def test_indicator_predictability(df: pd.DataFrame):
    """Test if indicators create predictable patterns"""
    df = df.copy()
    df['return_next'] = df['close'].pct_change().shift(-1)

    print("\n" + "="*60)
    print("TECHNICAL INDICATOR PREDICTABILITY TEST")
    print("="*60)

    tests = [
        ("RSI Oversold → UP?", df[df['rsi_oversold'] == 1]['return_next']),
        ("RSI Overbought → DOWN?", df[df['rsi_overbought'] == 1]['return_next']),
        ("Price below BB lower → UP?", df[df['bb_position'] < 0.2]['return_next']),
        ("Price above BB upper → DOWN?", df[df['bb_position'] > 0.8]['return_next']),
        ("MA cross bullish → UP?", df[(df['ma_cross'] == 1) & (df['ma_cross'].shift(1) == 0)]['return_next']),
    ]

    found_signal = False

    for name, returns in tests:
        if len(returns) > 10:
            mean_ret = returns.mean()
            pct_positive = (returns > 0).sum() / len(returns) * 100
            print(f"\n{name}")
            print(f"  Mean return: {mean_ret:+.4%}")
            print(f"  % Positive: {pct_positive:.1f}%")

            if abs(pct_positive - 50) > 5:  # More than 5% edge
                print(f"  ✓ SIGNAL FOUND! Edge: {pct_positive - 50:+.1f}%")
                found_signal = True
            else:
                print(f"  ✗ No edge")

    print("\n" + "="*60)
    if found_signal:
        print("✅ Technical indicators CREATE predictable patterns!")
        print("   Even if raw prices are random, indicators add structure.")
    else:
        print("⚠️  Indicators don't add much predictability")
        print("   Consider using longer timeframes or different assets")
    print("="*60 + "\n")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/train",
                       help="Input directory with CSV files")
    parser.add_argument("--output", type=str, default="data/train_with_indicators",
                       help="Output directory")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in input_dir.glob("*.csv"):
        print(f"Processing {csv_file.name}...")

        df = pd.read_csv(csv_file)
        df_with_indicators = add_technical_indicators(df)

        # Test predictability
        test_indicator_predictability(df_with_indicators)

        # Save
        output_path = output_dir / csv_file.name
        df_with_indicators.to_csv(output_path, index=False)
        print(f"✓ Saved to {output_path}")

    print("\n✅ Done! Update your config to use:")
    print(f'   paths: ["{args.output}/*.csv"]')

if __name__ == "__main__":
    main()
