#!/usr/bin/env python3
"""
Download historical OHLCV data from Binance
Usage: python download_historical_data.py --symbols BTCUSDT,ETHUSDT --timeframe 1h --days 365
"""
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import requests
import time

def download_binance_klines(symbol: str, interval: str, start_time: int, end_time: int):
    """Download klines from Binance API"""
    url = "https://api.binance.com/api/v3/klines"

    all_data = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000  # Max per request
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_data.extend(data)

            # Update start time for next batch
            current_start = data[-1][0] + 1

            print(f"  Downloaded {len(data)} candles, total: {len(all_data)}")
            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"  Error: {e}")
            break

    return all_data

def klines_to_dataframe(klines):
    """Convert Binance klines to DataFrame"""
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
        df[col] = df[col].astype(float)
    df['number_of_trades'] = df['number_of_trades'].astype(int)

    # Keep only needed columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume',
             'quote_asset_volume', 'number_of_trades',
             'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']]

    return df

def analyze_predictability(df: pd.DataFrame, symbol: str):
    """Analyze if data has predictable patterns"""
    df = df.copy()
    df['return'] = df['close'].pct_change()

    print(f"\n{'='*60}")
    print(f"PREDICTABILITY ANALYSIS: {symbol}")
    print(f"{'='*60}")

    # Autocorrelation
    autocorr_1 = df['return'].autocorr(lag=1)
    autocorr_5 = df['return'].autocorr(lag=5)
    autocorr_20 = df['return'].autocorr(lag=20)

    print(f"\nAutocorrelation:")
    print(f"  Lag 1:  {autocorr_1:+.4f}  {'✓ Good' if abs(autocorr_1) > 0.10 else '✗ Weak' if abs(autocorr_1) > 0.05 else '✗✗ Random walk'}")
    print(f"  Lag 5:  {autocorr_5:+.4f}  {'✓ Good' if abs(autocorr_5) > 0.08 else '✗ Weak' if abs(autocorr_5) > 0.03 else '✗✗ Random walk'}")
    print(f"  Lag 20: {autocorr_20:+.4f} {'✓ Good' if abs(autocorr_20) > 0.05 else '✗ Weak'}")

    # Mean and std
    mean_ret = df['return'].mean()
    std_ret = df['return'].std()
    sharpe_annual = (mean_ret / std_ret) * (365*24)**0.5 if std_ret > 0 else 0

    print(f"\nReturn Statistics:")
    print(f"  Mean: {mean_ret:.6f} ({mean_ret*365*24:.2%} annualized)")
    print(f"  Std:  {std_ret:.6f}")
    print(f"  Sharpe (annualized): {sharpe_annual:.2f}")

    # Momentum test
    df['prev_positive'] = df['return'] > 0
    pos_after_pos = df[df['prev_positive'] == True]['return'].shift(-1)
    pos_after_neg = df[df['prev_positive'] == False]['return'].shift(-1)

    pct_up_after_up = (pos_after_pos > 0).sum() / len(pos_after_pos) * 100
    pct_up_after_down = (pos_after_neg > 0).sum() / len(pos_after_neg) * 100

    print(f"\nMomentum Test:")
    print(f"  After UP move:   {pct_up_after_up:.1f}% chance of UP")
    print(f"  After DOWN move: {pct_up_after_down:.1f}% chance of UP")
    print(f"  Difference: {pct_up_after_up - pct_up_after_down:+.1f}%  {'✓ Momentum!' if abs(pct_up_after_up - pct_up_after_down) > 3 else '✗ Random'}")

    # Overall verdict
    print(f"\n{'='*60}")
    score = 0
    if abs(autocorr_1) > 0.10: score += 2
    elif abs(autocorr_1) > 0.05: score += 1

    if abs(pct_up_after_up - pct_up_after_down) > 3: score += 2
    elif abs(pct_up_after_up - pct_up_after_down) > 1: score += 1

    if mean_ret > 0: score += 1

    if score >= 4:
        verdict = "✅ LEARNABLE - Good predictive structure!"
    elif score >= 2:
        verdict = "⚠️  MARGINAL - Weak patterns, training may be difficult"
    else:
        verdict = "❌ RANDOM WALK - No predictable patterns, training will fail"

    print(f"VERDICT: {verdict}")
    print(f"Score: {score}/5")
    print(f"{'='*60}\n")

    return score >= 2

def main():
    parser = argparse.ArgumentParser(description="Download historical data from Binance")
    parser.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT",
                       help="Comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="1h",
                       choices=["1m", "5m", "15m", "1h", "4h", "1d"],
                       help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=365,
                       help="Number of days to download")
    parser.add_argument("--output-dir", type=str, default="data/train",
                       help="Output directory")

    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=args.days)).timestamp() * 1000)

    print(f"Downloading {args.timeframe} data for {len(symbols)} symbols")
    print(f"Period: {args.days} days")
    print(f"Output: {output_dir}\n")

    all_learnable = True

    for symbol in symbols:
        print(f"Downloading {symbol}...")

        klines = download_binance_klines(symbol, args.timeframe, start_time, end_time)

        if not klines:
            print(f"  ❌ No data downloaded for {symbol}")
            continue

        df = klines_to_dataframe(klines)

        # Save as CSV
        output_path = output_dir / f"{symbol}.csv"
        df.to_csv(output_path, index=False)
        print(f"  ✓ Saved {len(df)} rows to {output_path}")

        # Analyze predictability
        is_learnable = analyze_predictability(df, symbol)
        if not is_learnable:
            all_learnable = False

    print("\n" + "="*60)
    if all_learnable:
        print("✅ All symbols have learnable patterns!")
        print("You can proceed with training.")
    else:
        print("⚠️  WARNING: Some symbols appear to be random walk!")
        print("\nRecommendations:")
        print("1. Use longer timeframes (1h or 4h instead of 1m)")
        print("2. Add technical indicators as features")
        print("3. Consider different markets or assets")
        print("4. Test with prepare_demo_data_with_drift.py first")
    print("="*60)

if __name__ == "__main__":
    main()
