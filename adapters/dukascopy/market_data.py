# -*- coding: utf-8 -*-
"""
adapters/dukascopy/market_data.py
Dukascopy forex market-data adapter — free public historical tick feed.

Closes the Dukascopy stub gap: the UI offered Dukascopy for forex ("Public
ticks, без авторизации") but there was no backend — only a 43-line Phase-0
placeholder. This implements the real, widely-used **public bi5 tick feed**
(the basis of duka / dukascopy-node / dukascopy-python), which needs NO
credentials — matching exactly what the UI advertises. The credentialed JForex
API is a separate live-trading product and is intentionally NOT used here.

Data source
-----------
Dukascopy publishes tick history as hourly LZMA-compressed ``.bi5`` files:

    https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5

**Gotcha (famous):** the month segment is **0-indexed** (January = ``00``).
Each decompressed record is 20 bytes, big-endian ``>IIIff``:

    uint32  ms offset from the hour start
    uint32  ask price in points (integer)
    uint32  bid price in points (integer)
    float32 ask volume (millions)
    float32 bid volume (millions)

Prices are integers in "points" scaled by 10^decimals per instrument
(most FX = 1e5, JPY pairs & metals = 1e3). Weekends/holidays have no file
(HTTP 404) and are skipped gracefully — this is a data-only provider, so only
MarketDataAdapter is implemented (no order execution via the public feed).

References
----------
- Historical data feed format (community-documented, e.g. duka, dukascopy-node).
- https://www.dukascopy.com/swiss/english/marketwatch/historical/
"""

from __future__ import annotations

import logging
import lzma
import struct
import time as _time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from core_models import Bar, Tick
from adapters.base import MarketDataAdapter
from adapters.models import ExchangeVendor

logger = logging.getLogger(__name__)

# 20-byte big-endian tick record: ms-offset, ask, bid, ask-vol, bid-vol
_TICK_STRUCT = struct.Struct(">IIIff")
_TICK_SIZE = _TICK_STRUCT.size  # 20

# Timeframe → seconds (bars are resampled from ticks).
_TIMEFRAME_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    # tolerate a few common aliases
    "m1": 60,
    "m5": 300,
    "m15": 900,
    "m30": 1800,
    "h1": 3600,
    "h4": 14400,
    "d1": 86400,
}


class DukascopyMarketDataAdapter(MarketDataAdapter):
    """
    Dukascopy public tick-feed market-data adapter (forex, metals, CFDs).

    Configuration:
        base_url: feed host (default https://datafeed.dukascopy.com)
        point_values: {INSTRUMENT: divisor} overrides for price scaling
        timeout: HTTP timeout seconds (default 30)
        max_hours: safety cap on hourly files fetched per get_bars (default 720)

    Example:
        >>> a = DukascopyMarketDataAdapter()
        >>> a.connect()
        >>> bars = a.get_bars("EURUSD", "1h", limit=24)
    """

    DEFAULT_BASE_URL = "https://datafeed.dukascopy.com"

    def __init__(
        self,
        vendor: ExchangeVendor = ExchangeVendor.DUKASCOPY,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(vendor, config)
        self._base_url = str(self._config.get("base_url", self.DEFAULT_BASE_URL)).rstrip("/")
        self._timeout = int(self._config.get("timeout", 60))
        self._max_hours = int(self._config.get("max_hours", 720))
        self._point_overrides = {
            str(k).upper(): float(v)
            for k, v in (self._config.get("point_values", {}) or {}).items()
        }
        self._session = None

    # ------------------------------------------------------------------ transport

    def _get_session(self):
        # bi5 is a BINARY download. RestBudgetSession is JSON-oriented and
        # decodes bodies to text (corrupting bytes), so use plain requests and
        # read resp.content (raw bytes) directly.
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({"User-Agent": "riven-dukascopy/1.0"})
        return self._session

    def _do_connect(self) -> None:
        self._get_session()

    def _do_disconnect(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Dukascopy uses uppercase, no separator: EUR/USD → EURUSD."""
        return symbol.upper().replace("/", "").replace("_", "").replace("-", "")

    def _point_value(self, instrument: str) -> float:
        """Price divisor for an instrument (points → price)."""
        inst = instrument.upper()
        if inst in self._point_overrides:
            return self._point_overrides[inst]
        if inst.endswith("JPY"):
            return 1000.0
        if inst.startswith(("XAU", "XAG")) or inst in ("XAUUSD", "XAGUSD"):
            return 1000.0
        return 100000.0

    def _bi5_url(self, instrument: str, dt_hour: datetime) -> str:
        # Month is 0-indexed in the Dukascopy path (January = 00).
        return (
            f"{self._base_url}/datafeed/{instrument}/"
            f"{dt_hour.year:04d}/{dt_hour.month - 1:02d}/{dt_hour.day:02d}/"
            f"{dt_hour.hour:02d}h_ticks.bi5"
        )

    def _download_bi5(self, instrument: str, dt_hour: datetime) -> Optional[bytes]:
        """Fetch one hourly bi5 file. Returns raw bytes, or None on 404/empty
        (weekend/holiday/no-data hours are normal)."""
        url = self._bi5_url(instrument, dt_hour)
        session = self._get_session()
        try:
            resp = session.get(url, timeout=self._timeout)
        except Exception as exc:
            logger.debug("dukascopy: fetch failed %s (%s)", url, exc)
            return None
        status = getattr(resp, "status_code", 200)
        if status == 404:
            return None
        if status >= 400:
            logger.debug("dukascopy: HTTP %s for %s", status, url)
            return None
        # resp.content is raw bytes for requests; support test doubles that
        # expose bytes directly.
        content = getattr(resp, "content", resp)
        if not content:
            return None
        return content if isinstance(content, (bytes, bytearray)) else bytes(content)

    @staticmethod
    def _decompress(raw: bytes) -> bytes:
        """Decompress a bi5 payload (Dukascopy uses LZMA; be format-tolerant)."""
        for fmt in (lzma.FORMAT_AUTO, lzma.FORMAT_ALONE):
            try:
                return lzma.decompress(raw, format=fmt)
            except Exception:
                continue
        # some hosts double-wrap; last resort: streaming decompressor
        try:
            return lzma.LZMADecompressor().decompress(raw)
        except Exception:
            return b""

    def _parse_ticks(
        self, raw: bytes, hour_start_ms: int, point: float
    ) -> List[Tuple[int, float, float]]:
        """Decode a bi5 payload into ``[(ts_ms, bid, ask), ...]``."""
        data = self._decompress(raw)
        ticks: List[Tuple[int, float, float]] = []
        for off in range(0, len(data) - len(data) % _TICK_SIZE, _TICK_SIZE):
            ms, ask_i, bid_i, _av, _bv = _TICK_STRUCT.unpack_from(data, off)
            ticks.append((hour_start_ms + int(ms), bid_i / point, ask_i / point))
        return ticks

    def _collect_ticks(
        self, instrument: str, start_ms: int, end_ms: int
    ) -> List[Tuple[int, float, float]]:
        """Download + decode every hourly file spanning [start_ms, end_ms]."""
        point = self._point_value(instrument)
        start_hour = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        end_dt = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc)

        ticks: List[Tuple[int, float, float]] = []
        cur = start_hour
        hours = 0
        while cur <= end_dt and hours < self._max_hours:
            raw = self._download_bi5(instrument, cur)
            if raw:
                hour_ms = int(cur.timestamp() * 1000)
                for t in self._parse_ticks(raw, hour_ms, point):
                    if start_ms <= t[0] <= end_ms:
                        ticks.append(t)
            cur += timedelta(hours=1)
            hours += 1
        if hours >= self._max_hours:
            logger.warning(
                "dukascopy: hit max_hours=%s for %s — range truncated (raise config max_hours)",
                self._max_hours,
                instrument,
            )
        ticks.sort(key=lambda x: x[0])
        return ticks

    # ------------------------------------------------------------------ interface

    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        *,
        limit: int = 500,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
    ) -> List[Bar]:
        """Aggregate the public tick feed into OHLCV bars (mid price)."""
        instrument = self._normalize_symbol(symbol)
        tf_sec = _TIMEFRAME_SECONDS.get(timeframe.lower())
        if tf_sec is None:
            raise ValueError(
                f"Unsupported timeframe {timeframe!r}; "
                f"supported: {sorted(set(_TIMEFRAME_SECONDS))}"
            )

        now_ms = int(_time.time() * 1000)
        if end_ts is None:
            end_ts = now_ms
        if start_ts is None:
            # last `limit` bars, capped by max_hours worth of ticks
            start_ts = end_ts - int(limit) * tf_sec * 1000

        ticks = self._collect_ticks(instrument, int(start_ts), int(end_ts))
        if not ticks:
            return []

        try:
            import pandas as pd
        except Exception:
            logger.warning("dukascopy: pandas unavailable — cannot aggregate bars")
            return []

        df = pd.DataFrame(ticks, columns=["ts_ms", "bid", "ask"])
        df["mid"] = (df["bid"] + df["ask"]) / 2.0
        df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.set_index("dt")

        rule = f"{tf_sec}s"
        agg = (
            df.resample(rule, label="left", closed="left")
            .agg(
                open=("mid", "first"),
                high=("mid", "max"),
                low=("mid", "min"),
                close=("mid", "last"),
                bid_open=("bid", "first"),
                bid_high=("bid", "max"),
                bid_low=("bid", "min"),
                bid_close=("bid", "last"),
                ask_open=("ask", "first"),
                ask_high=("ask", "max"),
                ask_low=("ask", "min"),
                ask_close=("ask", "last"),
                ticks=("mid", "count"),
            )
            .dropna(subset=["open"])
        )

        bars: List[Bar] = []
        for idx, row in agg.iterrows():
            ts_ms = int(idx.timestamp() * 1000)
            spread = float(row["ask_close"]) - float(row["bid_close"])
            bars.append(
                Bar(
                    ts=ts_ms,
                    symbol=instrument,
                    open=Decimal(str(row["open"])),
                    high=Decimal(str(row["high"])),
                    low=Decimal(str(row["low"])),
                    close=Decimal(str(row["close"])),
                    volume_base=Decimal(str(int(row["ticks"]))),  # tick count (honest proxy)
                    volume_quote=Decimal(
                        str(round(spread * self._point_value(instrument)))
                    ),  # spread in points
                    bid_open=Decimal(str(row["bid_open"])),
                    bid_high=Decimal(str(row["bid_high"])),
                    bid_low=Decimal(str(row["bid_low"])),
                    bid_close=Decimal(str(row["bid_close"])),
                    ask_open=Decimal(str(row["ask_open"])),
                    ask_high=Decimal(str(row["ask_high"])),
                    ask_low=Decimal(str(row["ask_low"])),
                    ask_close=Decimal(str(row["ask_close"])),
                )
            )
        return bars[-int(limit) :] if limit else bars

    def get_latest_bar(self, symbol: str, timeframe: str) -> Optional[Bar]:
        tf_sec = _TIMEFRAME_SECONDS.get(timeframe.lower(), 3600)
        now_ms = int(_time.time() * 1000)
        # look back a few bars to be robust to the feed's ~ hourly finalization lag
        bars = self.get_bars(
            symbol,
            timeframe,
            start_ts=now_ms - max(3, 1) * tf_sec * 1000 - 2 * 3600 * 1000,
            end_ts=now_ms,
        )
        return bars[-1] if bars else None

    def get_tick(self, symbol: str) -> Optional[Tick]:
        """Most recent available tick (public feed lags by up to ~an hour)."""
        instrument = self._normalize_symbol(symbol)
        now = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        point = self._point_value(instrument)
        # scan back a few hours for the latest hour that has data
        for back in range(0, 6):
            hour = now - timedelta(hours=back)
            raw = self._download_bi5(instrument, hour)
            if not raw:
                continue
            parsed = self._parse_ticks(raw, int(hour.timestamp() * 1000), point)
            if parsed:
                ts_ms, bid, ask = parsed[-1]
                return Tick(
                    ts=ts_ms,
                    symbol=instrument,
                    price=Decimal(str((bid + ask) / 2.0)),
                    bid=Decimal(str(bid)),
                    ask=Decimal(str(ask)),
                )
        return None

    def stream_bars(self, symbols: Sequence[str], interval_ms: int) -> Iterator[Bar]:
        """Near-real-time bar stream by polling the feed (NOT sub-second — the
        public bi5 feed finalizes hourly). Honest: for true low-latency use a
        live broker adapter (OANDA/IB)."""
        tf = "1m" if interval_ms <= 60_000 else "1h"
        seen: Dict[str, int] = {}
        poll = max(5.0, interval_ms / 1000.0)
        while True:
            for sym in symbols:
                bar = self.get_latest_bar(sym, tf)
                if bar and seen.get(sym) != bar.ts:
                    seen[sym] = bar.ts
                    yield bar
            _time.sleep(poll)

    def stream_ticks(self, symbols: Sequence[str]) -> Iterator[Tick]:
        """Near-real-time tick stream by polling (see stream_bars caveat)."""
        seen: Dict[str, int] = {}
        while True:
            for sym in symbols:
                tick = self.get_tick(sym)
                if tick and seen.get(sym) != tick.ts:
                    seen[sym] = tick.ts
                    yield tick
            _time.sleep(5.0)
