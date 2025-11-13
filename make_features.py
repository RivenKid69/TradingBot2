# scripts/make_features.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import pandas as pd

from transformers import FeatureSpec, apply_offline_features


def _read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main():
    p = argparse.ArgumentParser(description="Build offline features using the same transformers as realtime.")
    p.add_argument("--in", dest="in_path", required=True, help="Вход: parquet/csv с колонками ts_ms,symbol,price (обычно prices.parquet)")
    p.add_argument("--out", dest="out_path", default="data/features.parquet", help="Куда сохранить parquet фич")
    p.add_argument("--price-col", default="price", help="Имя колонки с ценой (по умолчанию 'price')")
    p.add_argument("--lookbacks", default="240,720,1200,1440,5040,12000", help="Окна SMA/ret в минутах для 4h интервала (по умолчанию 240,720,1200,1440,5040,12000 = 4h,12h,20h,24h,3.5d,200h)")
    p.add_argument("--rsi-period", type=int, default=14, help="Период RSI (Wilder)")
    p.add_argument("--yang-zhang-windows", default="2880,10080,43200", help="Окна Yang-Zhang в минутах для 4h (по умолчанию 2880,10080,43200 = 48h,7d,30d)")
    p.add_argument("--open-col", default=None, help="Имя колонки open для Yang-Zhang (опционально)")
    p.add_argument("--high-col", default=None, help="Имя колонки high для Yang-Zhang (опционально)")
    p.add_argument("--low-col", default=None, help="Имя колонки low для Yang-Zhang (опционально)")
    p.add_argument("--taker-buy-ratio-windows", default="480,960,1440", help="Окна Taker Buy Ratio SMA в минутах для 4h (по умолчанию 480,960,1440 = 8h,16h,24h)")
    p.add_argument("--taker-buy-ratio-momentum", default="240,480,720", help="Окна Taker Buy Ratio momentum в минутах для 4h (по умолчанию 240,480,720 = 4h,8h,12h)")
    p.add_argument("--volume-col", default=None, help="Имя колонки volume для Taker Buy Ratio (опционально)")
    p.add_argument("--taker-buy-base-col", default=None, help="Имя колонки taker_buy_base для Taker Buy Ratio (опционально)")
    p.add_argument("--cvd-windows", default="1440,10080", help="Окна Cumulative Volume Delta в минутах для 4h (по умолчанию 1440,10080 = 24h,7d)")
    p.add_argument("--parkinson-windows", default="2880,10080", help="Окна Parkinson волатильности в минутах для 4h (по умолчанию 2880,10080 = 48h,7d)")
    p.add_argument("--garch-windows", default="12000,20160,43200", help="Окна GARCH(1,1) волатильности в минутах для 4h (по умолчанию 12000,20160,43200 = 200h/8.3d, 14d, 30d)")
    p.add_argument("--bar-duration-minutes", type=int, default=240, help="Длительность одного бара в минутах (по умолчанию 240 для 4h интервала)")
    args = p.parse_args()

    df = _read_any(args.in_path)
    if df is None or df.empty:
        raise SystemExit("входной файл пуст")

    lookbacks = [int(s.strip()) for s in str(args.lookbacks).split(",") if s.strip()]
    yang_zhang_wins = [int(s.strip()) for s in str(args.yang_zhang_windows).split(",") if s.strip()]
    taker_buy_ratio_wins = [int(s.strip()) for s in str(args.taker_buy_ratio_windows).split(",") if s.strip()]
    taker_buy_ratio_mom = [int(s.strip()) for s in str(args.taker_buy_ratio_momentum).split(",") if s.strip()]
    cvd_wins = [int(s.strip()) for s in str(args.cvd_windows).split(",") if s.strip()]
    parkinson_wins = [int(s.strip()) for s in str(args.parkinson_windows).split(",") if s.strip()]
    garch_wins = [int(s.strip()) for s in str(args.garch_windows).split(",") if s.strip()]

    spec = FeatureSpec(
        lookbacks_prices=lookbacks,
        rsi_period=int(args.rsi_period),
        yang_zhang_windows=yang_zhang_wins,
        parkinson_windows=parkinson_wins,
        garch_windows=garch_wins,
        taker_buy_ratio_windows=taker_buy_ratio_wins,
        taker_buy_ratio_momentum=taker_buy_ratio_mom,
        cvd_windows=cvd_wins,
        bar_duration_minutes=int(args.bar_duration_minutes),
    )

    # Определяем, есть ли OHLC колонки в данных
    open_col = args.open_col if args.open_col and args.open_col in df.columns else None
    high_col = args.high_col if args.high_col and args.high_col in df.columns else None
    low_col = args.low_col if args.low_col and args.low_col in df.columns else None

    # Определяем, есть ли volume и taker_buy_base колонки в данных
    volume_col = args.volume_col if args.volume_col and args.volume_col in df.columns else None
    taker_buy_base_col = args.taker_buy_base_col if args.taker_buy_base_col and args.taker_buy_base_col in df.columns else None

    feats = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col=args.price_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
        volume_col=volume_col,
        taker_buy_base_col=taker_buy_base_col,
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    if args.out_path.lower().endswith(".parquet"):
        feats.to_parquet(args.out_path, index=False)
    else:
        feats.to_csv(args.out_path, index=False)
    print(f"Wrote {len(feats)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
