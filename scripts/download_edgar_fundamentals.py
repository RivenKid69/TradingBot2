# -*- coding: utf-8 -*-
"""
scripts/download_edgar_fundamentals.py
======================================

Скачать НАСТОЯЩИЙ point-in-time фундаментал из SEC EDGAR (бесплатно, без ключей)
в parquet для слота ``fundamentals_path`` cross-sectional equity-пайплайна.

    PYTHONPATH=.venv/Lib/site-packages python scripts/download_edgar_fundamentals.py \
        --symbols AAPL MSFT GOOGL AMZN NVDA META JPM XOM JNJ PG \
        --out data/fundamentals_edgar/edgar_pit.parquet

publish_ts каждой записи = дата подачи отчёта (filed) -> честный PIT (анти-look-ahead).
SEC требует User-Agent: задайте env ``SEC_EDGAR_USER_AGENT="Your Name your@email"``.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.edgar_fundamentals import write_pit_parquet, build_pit_fundamentals_frame


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Download SEC EDGAR PIT fundamentals")
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--out", default="data/fundamentals_edgar/edgar_pit.parquet")
    a = p.parse_args(argv)

    df = build_pit_fundamentals_frame(a.symbols)
    if not len(df):
        print("No fundamentals fetched (check tickers / SEC_EDGAR_USER_AGENT / network).",
              file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"Wrote {len(df)} rows for {df['symbol'].nunique()} symbols -> {a.out}")
    print("Per-symbol filings:")
    print(df.groupby("symbol")["publish_ts"].count().to_string())
    print("\nDate range:",
          str(__import__('pandas').to_datetime(df['publish_ts'].min(), unit='ms').date()),
          "->",
          str(__import__('pandas').to_datetime(df['publish_ts'].max(), unit='ms').date()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
