# -*- coding: utf-8 -*-
"""
prepare_training_data.py
Download Binance klines and compute all required features for training.
No API key needed - uses public endpoints only.
"""
from __future__ import annotations

import argparse
import sys
import os
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binance_public import BinancePublicClient


def download_klines(
    client: BinancePublicClient,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    market: str = "spot",
) -> pd.DataFrame:
    """
    Download klines from Binance with pagination.

    Args:
        client: BinancePublicClient instance
        symbol: Trading pair (e.g. "BTCUSDT")
        interval: Timeframe (e.g. "4h", "1h")
        start_date: Start date ISO format (e.g. "2023-01-01")
        end_date: End date ISO format (e.g. "2025-11-01")
        market: "spot" or "futures"

    Returns:
        DataFrame with OHLCV data
    """
    # Parse dates to milliseconds
    start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # Interval to milliseconds mapping
    interval_ms_map = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "2h": 2 * 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "6h": 6 * 60 * 60_000,
        "8h": 8 * 60 * 60_000,
        "12h": 12 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }

    interval_ms = interval_ms_map.get(interval)
    if interval_ms is None:
        raise ValueError(f"Unknown interval: {interval}")

    all_klines: List[List[Any]] = []
    current_start = start_ms
    limit = 1000  # Binance max per request

    print(f"Downloading {symbol} {interval} from {start_date} to {end_date}...")

    request_count = 0
    while current_start < end_ms:
        try:
            klines = client.get_klines(
                market=market,
                symbol=symbol,
                interval=interval,
                start_ms=current_start,
                end_ms=end_ms,
                limit=limit,
            )

            if not klines:
                break

            all_klines.extend(klines)
            request_count += 1

            # Move to next batch
            last_open_time = int(klines[-1][0])
            current_start = last_open_time + interval_ms

            # Progress
            progress_pct = min(100, (current_start - start_ms) / (end_ms - start_ms) * 100)
            print(f"  Progress: {progress_pct:.1f}% ({len(all_klines)} bars)", end="\r")

            # Rate limiting - be nice to Binance
            if request_count % 10 == 0:
                time.sleep(0.5)

        except Exception as e:
            print(f"\nError fetching klines: {e}")
            time.sleep(2)
            continue

    print(f"\nDownloaded {len(all_klines)} bars in {request_count} requests")

    # Convert to DataFrame
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df = pd.DataFrame(all_klines, columns=columns)

    # Convert types
    df["open_time"] = pd.to_numeric(df["open_time"])
    df["close_time"] = pd.to_numeric(df["close_time"])
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["number_of_trades"] = (
        pd.to_numeric(df["number_of_trades"], errors="coerce").fillna(0).astype(int)
    )

    # Add derived columns
    df["symbol"] = symbol
    df["ts_ms"] = df["open_time"]
    df["timestamp"] = df["open_time"]
    df["price"] = df["close"]

    # Drop duplicates and sort
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

    return df


def compute_technical_indicators(df: pd.DataFrame, interval_minutes: int = 240) -> pd.DataFrame:
    """
    Compute all technical indicators required for training.

    Args:
        df: DataFrame with OHLCV data
        interval_minutes: Bar interval in minutes (240 for 4h)

    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()

    # ================= BASIC INDICATORS =================

    # SMA (Simple Moving Average)
    df["sma_1200"] = df["close"].rolling(window=5, min_periods=1).mean()  # 5 bars = 1200 min (20h)
    df["sma_5040"] = (
        df["close"].rolling(window=21, min_periods=1).mean()
    )  # 21 bars = 5040 min (84h)
    df["sma_12000"] = df["close"].rolling(window=50, min_periods=1).mean()  # 50 bars = 200h

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilder's smoothed RSI
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    df["rsi"] = df["rsi"].fillna(50.0)
    df["rsi_valid"] = (df.index >= 13).astype(float)

    # MACD (Moving Average Convergence Divergence)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    df["macd_valid"] = (df.index >= 25).astype(float)

    # ATR (Average True Range)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=14, min_periods=1).mean()
    df["atr_valid"] = (df.index >= 13).astype(float)

    # CCI (Commodity Channel Index)
    tp = (df["high"] + df["low"] + df["close"]) / 3  # Typical Price
    sma_tp = tp.rolling(window=20, min_periods=1).mean()
    mad = tp.rolling(window=20, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    df["cci"] = (tp - sma_tp) / (0.015 * mad + 1e-10)
    df["cci"] = df["cci"].fillna(0.0)
    df["cci_valid"] = (df.index >= 19).astype(float)

    # OBV (On-Balance Volume)
    obv = [0.0]
    for i in range(1, len(df)):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            obv.append(obv[-1] + df["volume"].iloc[i])
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            obv.append(obv[-1] - df["volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["obv"] = obv
    df["obv_valid"] = 1.0

    # Bollinger Bands
    bb_period = 20
    df["bb_middle"] = df["close"].rolling(window=bb_period, min_periods=1).mean()
    bb_std = df["close"].rolling(window=bb_period, min_periods=1).std()
    df["bb_upper"] = df["bb_middle"] + 2 * bb_std
    df["bb_lower"] = df["bb_middle"] - 2 * bb_std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_middle"] + 1e-10)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_valid"] = (df.index >= bb_period - 1).astype(float)

    # Momentum
    df["momentum"] = df["close"].diff(periods=10) / (df["close"].shift(10) + 1e-10)
    df["momentum"] = df["momentum"].fillna(0.0)
    df["momentum_valid"] = (df.index >= 10).astype(float)

    # ================= RETURNS =================

    df["ret_4h"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)  # 1 bar
    df["ret_12h"] = np.log(df["close"] / df["close"].shift(3)).fillna(0.0)  # 3 bars
    df["ret_24h"] = np.log(df["close"] / df["close"].shift(6)).fillna(0.0)  # 6 bars
    df["ret_7d"] = np.log(df["close"] / df["close"].shift(42)).fillna(0.0)  # 42 bars = 7 days

    # ================= VOLATILITY ESTIMATORS =================

    # Yang-Zhang Volatility
    def yang_zhang_vol(ohlc_df: pd.DataFrame, window: int) -> pd.Series:
        """Yang-Zhang volatility estimator."""
        o = np.log(ohlc_df["open"])
        h = np.log(ohlc_df["high"])
        l = np.log(ohlc_df["low"])
        c = np.log(ohlc_df["close"])
        c_prev = c.shift(1)

        # Overnight variance
        overnight = (o - c_prev) ** 2
        vo = overnight.rolling(window=window, min_periods=1).mean()

        # Open-to-close variance
        oc = (c - o) ** 2
        vc = oc.rolling(window=window, min_periods=1).mean()

        # Rogers-Satchell variance
        rs = (h - c) * (h - o) + (l - c) * (l - o)
        vrs = rs.rolling(window=window, min_periods=1).mean()

        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        yz = np.sqrt(vo + k * vc + (1 - k) * vrs)
        return yz

    df["yang_zhang_48h"] = yang_zhang_vol(df, 12)  # 12 bars = 48h
    df["yang_zhang_7d"] = yang_zhang_vol(df, 42)  # 42 bars = 7d
    df["yang_zhang_30d"] = yang_zhang_vol(df, 180)  # 180 bars = 30d

    # Parkinson Volatility
    def parkinson_vol(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """Parkinson volatility estimator."""
        hl_ratio = np.log(high / low) ** 2
        factor = 1 / (4 * np.log(2))
        return np.sqrt(factor * hl_ratio.rolling(window=window, min_periods=1).mean())

    df["parkinson_48h"] = parkinson_vol(df["high"], df["low"], 12)
    df["parkinson_7d"] = parkinson_vol(df["high"], df["low"], 42)

    # Simplified GARCH proxy (using rolling std of returns)
    df["garch_200h"] = df["ret_4h"].rolling(window=50, min_periods=10).std()  # 50 bars = 200h
    df["garch_14d"] = df["ret_4h"].rolling(window=84, min_periods=20).std()  # 84 bars = 14d
    df["garch_30d"] = df["ret_4h"].rolling(window=180, min_periods=30).std()  # 180 bars = 30d

    # ================= VOLUME FEATURES =================

    # Taker buy ratio
    df["taker_buy_ratio"] = df["taker_buy_base_asset_volume"] / (df["volume"] + 1e-10)
    df["taker_buy_ratio_sma_8h"] = df["taker_buy_ratio"].rolling(window=2, min_periods=1).mean()
    df["taker_buy_ratio_sma_16h"] = df["taker_buy_ratio"].rolling(window=4, min_periods=1).mean()
    df["taker_buy_ratio_sma_24h"] = df["taker_buy_ratio"].rolling(window=6, min_periods=1).mean()

    # Taker buy ratio momentum
    df["taker_buy_ratio_momentum_4h"] = df["taker_buy_ratio"].diff(1).fillna(0.0)
    df["taker_buy_ratio_momentum_8h"] = df["taker_buy_ratio"].diff(2).fillna(0.0)
    df["taker_buy_ratio_momentum_12h"] = df["taker_buy_ratio"].diff(3).fillna(0.0)
    df["taker_buy_ratio_momentum_24h"] = df["taker_buy_ratio"].diff(6).fillna(0.0)

    # CVD (Cumulative Volume Delta) proxy
    # Buy volume - Sell volume (estimated)
    buy_volume = df["taker_buy_base_asset_volume"]
    sell_volume = df["volume"] - buy_volume
    delta = buy_volume - sell_volume

    df["cvd_24h"] = delta.rolling(window=6, min_periods=1).sum()  # 6 bars = 24h
    df["cvd_7d"] = delta.rolling(window=42, min_periods=1).sum()  # 42 bars = 7d

    # ================= DATA QUALITY FLAGS =================

    # Add validity flags for warmup periods
    df["data_valid"] = 1.0

    # Fill NaN with sensible defaults
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            if "valid" in col:
                df[col] = df[col].fillna(0.0)
            elif col in ["rsi"]:
                df[col] = df[col].fillna(50.0)
            elif col in ["taker_buy_ratio"]:
                df[col] = df[col].fillna(0.5)
            else:
                df[col] = df[col].fillna(0.0)

    return df


def add_train_val_test_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> pd.DataFrame:
    """Add wf_role column for train/val/test split."""
    df = df.copy()
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    df["wf_role"] = "test"
    df.loc[:train_end, "wf_role"] = "train"
    df.loc[train_end:val_end, "wf_role"] = "val"

    return df


def main():
    parser = argparse.ArgumentParser(description="Download and prepare training data from Binance")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--interval", type=str, default="4h", help="Timeframe")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-11-26", help="End date (YYYY-MM-DD)")
    parser.add_argument("--market", type=str, default="spot", choices=["spot", "futures"])
    parser.add_argument("--output", type=str, default="data/processed/BTCUSDT.feather")
    parser.add_argument("--output-csv", type=str, help="Also save as CSV")

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize client (no API key needed)
    print("Initializing Binance public client...")
    client = BinancePublicClient()

    try:
        # Download klines
        df = download_klines(
            client=client,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            market=args.market,
        )

        print(f"\nRaw data: {len(df)} bars")
        print(f"Date range: {df['open_time'].min()} to {df['open_time'].max()}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        # Compute indicators
        print("\nComputing technical indicators...")
        interval_minutes = {"1h": 60, "4h": 240, "1d": 1440}.get(args.interval, 240)
        df = compute_technical_indicators(df, interval_minutes=interval_minutes)

        # Add train/val/test split
        print("Adding train/val/test split...")
        df = add_train_val_test_split(df)

        # Summary
        print(f"\nFinal dataset: {len(df)} bars, {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"\nSplit:")
        print(df["wf_role"].value_counts())

        # Save
        print(f"\nSaving to {args.output}...")
        df.to_feather(args.output)

        if args.output_csv:
            print(f"Saving CSV to {args.output_csv}...")
            df.to_csv(args.output_csv, index=False)

        # Verify
        df_check = pd.read_feather(args.output)
        print(f"\nVerification: {len(df_check)} rows, {len(df_check.columns)} columns")
        print("Done!")

    finally:
        client.close()


if __name__ == "__main__":
    main()
