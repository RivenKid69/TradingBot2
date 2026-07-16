# -*- coding: utf-8 -*-
"""CLI для интрадей-фидов: минутные бары и тиковый бэкфилл (P2-M).

Примеры:
    # Минутки crypto (бесплатно, Binance klines)
    python scripts/download_premium_data.py bars --vendor binance --symbols BTCUSDT ETHUSDT \
        --start 2026-07-01 --end 2026-07-10 --timeframe 1m

    # Минутки US equities (Polygon, нужен POLYGON_API_KEY платного плана)
    python scripts/download_premium_data.py bars --vendor polygon --symbols AAPL SPY \
        --start 2026-07-01 --end 2026-07-10

    # Тиковый бэкфилл crypto (Binance aggTrades, настоящие сделки)
    python scripts/download_premium_data.py ticks --symbols BTCUSDT \
        --start 2026-07-10 --end 2026-07-11

    # Матрица вендоров (кто готов: адаптер + ключи)
    python scripts/download_premium_data.py vendors
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# запуск как `python scripts/download_premium_data.py` из корня репо
sys.path.insert(0, str(Path(__file__).parent.parent))


def _to_ms(date_str: str) -> int:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Интрадей-данные: минутные бары / тики")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_bars = sub.add_parser("bars", help="минутные бары → data/minute/")
    p_bars.add_argument("--vendor", required=True, choices=["binance", "dukascopy", "polygon", "alpaca", "oanda"])
    p_bars.add_argument("--symbols", nargs="+", required=True)
    p_bars.add_argument("--timeframe", default="1m")
    p_bars.add_argument("--start", required=True, help="ISO-дата, напр. 2026-07-01")
    p_bars.add_argument("--end", required=True)
    p_bars.add_argument("--out", default=None, help="каталог вывода (default data/minute)")

    p_ticks = sub.add_parser("ticks", help="тиковый бэкфилл (Binance aggTrades) → data/ticks/")
    p_ticks.add_argument("--symbols", nargs="+", required=True)
    p_ticks.add_argument("--start", required=True)
    p_ticks.add_argument("--end", required=True)
    p_ticks.add_argument("--out", default=None)

    sub.add_parser("vendors", help="entitlement-матрица вендоров")

    args = parser.parse_args(argv)

    from services.premium_data import (
        DEFAULT_OUT_DIR, DEFAULT_TICKS_DIR,
        download_binance_agg_trades, download_minute_bars, vendor_status,
    )

    if args.cmd == "vendors":
        print(json.dumps(vendor_status(), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "bars":
        results = download_minute_bars(
            args.vendor, args.symbols, timeframe=args.timeframe,
            start_ts_ms=_to_ms(args.start), end_ts_ms=_to_ms(args.end),
            out_dir=args.out or DEFAULT_OUT_DIR,
        )
    else:  # ticks
        results = download_binance_agg_trades(
            args.symbols, start_ts_ms=_to_ms(args.start), end_ts_ms=_to_ms(args.end),
            out_dir=args.out or DEFAULT_TICKS_DIR,
        )

    ok = True
    for r in results:
        line = r.to_dict()
        print(json.dumps(line, ensure_ascii=False))
        ok = ok and r.ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
