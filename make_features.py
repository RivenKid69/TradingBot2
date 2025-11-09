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
    p.add_argument("--lookbacks", default="5,15,60", help="Окна SMA/ret через запятую, например '5,15,60'")
    p.add_argument("--rsi-period", type=int, default=14, help="Период RSI (Wilder)")
    p.add_argument("--yang-zhang-windows", default="1440,10080,43200", help="Окна Yang-Zhang в минутах (по умолчанию 24ч,168ч,720ч)")
    p.add_argument("--open-col", default=None, help="Имя колонки open для Yang-Zhang (опционально)")
    p.add_argument("--high-col", default=None, help="Имя колонки high для Yang-Zhang (опционально)")
    p.add_argument("--low-col", default=None, help="Имя колонки low для Yang-Zhang (опционально)")
    args = p.parse_args()

    df = _read_any(args.in_path)
    if df is None or df.empty:
        raise SystemExit("входной файл пуст")

    lookbacks = [int(s.strip()) for s in str(args.lookbacks).split(",") if s.strip()]
    yang_zhang_wins = [int(s.strip()) for s in str(args.yang_zhang_windows).split(",") if s.strip()]
    spec = FeatureSpec(
        lookbacks_prices=lookbacks,
        rsi_period=int(args.rsi_period),
        yang_zhang_windows=yang_zhang_wins
    )

    # Определяем, есть ли OHLC колонки в данных
    open_col = args.open_col if args.open_col and args.open_col in df.columns else None
    high_col = args.high_col if args.high_col and args.high_col in df.columns else None
    low_col = args.low_col if args.low_col and args.low_col in df.columns else None

    feats = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col=args.price_col,
        open_col=open_col,
        high_col=high_col,
        low_col=low_col,
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    if args.out_path.lower().endswith(".parquet"):
        feats.to_parquet(args.out_path, index=False)
    else:
        feats.to_csv(args.out_path, index=False)
    print(f"Wrote {len(feats)} rows to {args.out_path}")


if __name__ == "__main__":
    main()
