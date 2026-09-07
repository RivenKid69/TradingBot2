# -*- coding: utf-8 -*-
"""
impl_data_sources.py
====================

Единый, источник-агностичный слой данных для cross-sectional контура (Stage A2).

Два семейства источников, **одинаковый интерфейс для free и BYO**:

* **PriceSource** — котировки/бары:
    - ``AdapterPriceSource`` (free): поверх существующих адаптеров
      (``adapters/yahoo``, ``adapters/binance`` … через registry). Цены наблюдаемы →
      ``pit_quality='true'``.
    - ``ParquetPriceSource`` (BYO): parquet/CSV пользователя.
* **FundamentalsSource** — фундаментал (для equity-сигналов value/quality):
    - ``ParquetFundamentals`` (BYO): с честным ``publish_ts`` → ``pit_quality='true'``.
    - ``FreeFundamentals`` (yfinance): СНИМОК на «сейчас», историзации нет →
      ``pit_quality='none'`` (для бэктеста непригоден; явно помечается и логируется).

Каждый источник несёт ``DataSourceMeta`` с флагом ``pit_quality`` (``true|approx|none``) —
этот флаг показывается в UI/логах, чтобы пользователь понимал ограничения данных.

Total-return (реинвестирование дивидендов + сплиты) реализован детерминированно
(``total_return_index`` / ``add_total_return``) и не требует внешних данных; опционально
делегирует в ``services/corporate_actions.py`` если данные доступны.

Слой ``impl_`` (зависит от ``core_portfolio``, ``impl_panel``). Никаких сетевых вызовов на
импорте — все внешние зависимости импортируются лениво внутри методов.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from glob import glob
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd

from core_portfolio import Panel
from impl_panel import PanelBuilder, normalize_ts_ms

logger = logging.getLogger(__name__)

# Допустимые значения PIT-качества (показываются в UI/логах).
PIT_TRUE = "true"  # настоящий point-in-time (можно бэктестить честно)
PIT_APPROX = "approx"  # приблизительный PIT (есть лаг/допущения)
PIT_NONE = "none"  # снимок «сейчас», историзации нет (НЕ для бэктеста)
VALID_PIT_QUALITY = (PIT_TRUE, PIT_APPROX, PIT_NONE)

# Канонические колонки прайс-кадра.
PRICE_COLUMNS = ("timestamp", "symbol", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class DataSourceMeta:
    """Метаданные источника — для UI/логов и honest-предупреждений."""

    name: str
    vendor: str
    kind: str  # 'price' | 'fundamentals'
    pit_quality: str = PIT_TRUE  # true | approx | none
    survivorship_biased: Optional[bool] = None
    free: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        if self.pit_quality not in VALID_PIT_QUALITY:
            raise ValueError(
                f"pit_quality must be one of {VALID_PIT_QUALITY}, got {self.pit_quality!r}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "kind": self.kind,
            "pit_quality": self.pit_quality,
            "survivorship_biased": self.survivorship_biased,
            "free": self.free,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class PriceSource(Protocol):
    meta: DataSourceMeta

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """``{symbol -> DataFrame}`` с колонками PRICE_COLUMNS (timestamp в мс)."""
        ...


@runtime_checkable
class FundamentalsSource(Protocol):
    meta: DataSourceMeta

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        fields: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Длинный DataFrame: колонки [publish_ts(ms), symbol, <fields...>]."""
        ...


# ---------------------------------------------------------------------------
# Bar -> frame helpers
# ---------------------------------------------------------------------------
def _bar_value(bar: Any, *names: str, default: float = float("nan")) -> float:
    for n in names:
        v = getattr(bar, n, None)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return default


def bars_to_frame(bars: Sequence[Any], symbol: str) -> pd.DataFrame:
    """Список Bar-объектов (core_models.Bar или duck-typed) → канонический прайс-кадр."""
    rows = []
    for b in bars or []:
        ts = getattr(b, "ts", None)
        if ts is None:
            ts = getattr(b, "timestamp", None)
        if ts is None:
            continue
        rows.append(
            {
                "timestamp": int(ts),
                "open": _bar_value(b, "open"),
                "high": _bar_value(b, "high"),
                "low": _bar_value(b, "low"),
                "close": _bar_value(b, "close"),
                "volume": _bar_value(b, "volume_base", "volume", default=0.0),
            }
        )
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["symbol"] = symbol
    return df


# ---------------------------------------------------------------------------
# Price sources
# ---------------------------------------------------------------------------
class AdapterPriceSource:
    """Free прайс-источник поверх market-data адаптера (yahoo/binance/…).

    Адаптер можно передать явно (DI, удобно для тестов) или создать лениво по
    ``vendor`` через ``adapters.registry``. Сетевые сбои/отсутствие адаптера
    обрабатываются мягко: символ пропускается, ошибка логируется.
    """

    def __init__(
        self,
        vendor: str = "yahoo",
        *,
        adapter: Any = None,
        config: Optional[Mapping[str, Any]] = None,
        pit_quality: str = PIT_TRUE,
        survivorship_biased: Optional[bool] = None,
        name: Optional[str] = None,
    ) -> None:
        self.vendor = str(vendor)
        self._adapter = adapter
        self._config = dict(config or {})
        self.meta = DataSourceMeta(
            name=name or f"free:{self.vendor}",
            vendor=self.vendor,
            kind="price",
            pit_quality=pit_quality,
            survivorship_biased=survivorship_biased,
            free=True,
            notes="Live/observed prices via market-data adapter.",
        )
        logger.info("PriceSource ready: %s (pit_quality=%s)", self.meta.name, self.meta.pit_quality)

    def _ensure_adapter(self) -> Any:
        if self._adapter is not None:
            return self._adapter
        try:
            from adapters.registry import create_market_data_adapter

            self._adapter = create_market_data_adapter(self.vendor, dict(self._config))
        except Exception as exc:  # pragma: no cover - зависит от окружения
            logger.warning("AdapterPriceSource: cannot create adapter '%s': %s", self.vendor, exc)
            self._adapter = None
        return self._adapter

    def available(self) -> bool:
        return self._ensure_adapter() is not None

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: str,
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        adapter = self._ensure_adapter()
        out: Dict[str, pd.DataFrame] = {}
        if adapter is None:
            logger.warning("AdapterPriceSource '%s' unavailable; returning empty.", self.vendor)
            return out
        for sym in symbols:
            try:
                bars = adapter.get_bars(
                    sym, timeframe, limit=limit, start_ts=start_ms, end_ts=end_ms
                )
                frame = bars_to_frame(bars, sym)
                if len(frame):
                    out[sym] = frame
            except Exception as exc:  # pragma: no cover - сетевые/вендорные сбои
                logger.warning("get_bars failed for %s@%s: %s", sym, self.vendor, exc)
        return out


class ParquetPriceSource:
    """BYO прайс-источник: parquet/CSV на диске.

    ``path_map``: ``{symbol -> filepath}`` ИЛИ ``root`` каталог с файлами
    ``<symbol>.parquet``/``.csv``. Цены пользователя считаются настоящими (PIT true),
    но survivorship зависит от того, какие файлы предоставлены (по умолчанию unknown).
    """

    def __init__(
        self,
        *,
        root: Optional[Union[str, os.PathLike]] = None,
        path_map: Optional[Mapping[str, str]] = None,
        ext: str = "parquet",
        pit_quality: str = PIT_TRUE,
        survivorship_biased: Optional[bool] = None,
        name: str = "byo:parquet",
    ) -> None:
        if root is None and not path_map:
            raise ValueError("ParquetPriceSource requires 'root' or 'path_map'")
        self.root = str(root) if root is not None else None
        self.path_map = dict(path_map or {})
        self.ext = ext.lstrip(".")
        self.meta = DataSourceMeta(
            name=name,
            vendor="byo",
            kind="price",
            pit_quality=pit_quality,
            survivorship_biased=survivorship_biased,
            free=False,
            notes="User-provided price files (BYO).",
        )
        logger.info("PriceSource ready: %s (pit_quality=%s)", self.meta.name, self.meta.pit_quality)

    def _resolve(self, symbol: str) -> Optional[str]:
        if symbol in self.path_map:
            return self.path_map[symbol]
        if self.root is None:
            return None
        for cand in (
            os.path.join(self.root, f"{symbol}.{self.ext}"),
            os.path.join(self.root, f"{symbol}.parquet"),
            os.path.join(self.root, f"{symbol}.csv"),
        ):
            if os.path.exists(cand):
                return cand
        return None

    @staticmethod
    def _read(path: str) -> pd.DataFrame:
        if path.lower().endswith(".csv"):
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def get_bars(
        self,
        symbols: Sequence[str],
        timeframe: str = "",
        *,
        start_ms: Optional[int] = None,
        end_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            path = self._resolve(sym)
            if path is None:
                logger.warning("ParquetPriceSource: no file for symbol %s", sym)
                continue
            try:
                df = self._read(path).copy()
            except Exception as exc:  # pragma: no cover
                logger.warning("ParquetPriceSource: failed to read %s: %s", path, exc)
                continue
            if "symbol" not in df.columns:
                df["symbol"] = sym
            # фильтр по диапазону, если есть распознаваемый таймстемп
            ts_col = PanelBuilder._detect_ts_col(df)
            if ts_col is not None and (start_ms is not None or end_ms is not None):
                ts_ms = normalize_ts_ms(df[ts_col])
                mask = np.ones(len(df), dtype=bool)
                if start_ms is not None:
                    mask &= ts_ms >= int(start_ms)
                if end_ms is not None:
                    mask &= ts_ms < int(end_ms)
                df = df.loc[mask]
            out[sym] = df
        return out


# ---------------------------------------------------------------------------
# Fundamentals sources
# ---------------------------------------------------------------------------
class ParquetFundamentals:
    """BYO фундаментал с честным ``publish_ts`` → PIT-true.

    Файл (parquet/CSV) с колонками: publish-таймстемп + ``symbol`` + поля метрик.
    Используется через ``PanelBuilder.asof_join(..., publish_lag_ms=...)`` — это
    гарантирует отсутствие look-ahead.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        *,
        publish_ts_col: Optional[str] = None,
        symbol_col: str = "symbol",
        pit_quality: str = PIT_TRUE,
        name: str = "byo:fundamentals",
    ) -> None:
        self.path = str(path)
        self.publish_ts_col = publish_ts_col
        self.symbol_col = symbol_col
        self.meta = DataSourceMeta(
            name=name,
            vendor="byo",
            kind="fundamentals",
            pit_quality=pit_quality,
            free=False,
            notes="User-provided point-in-time fundamentals (BYO).",
        )
        logger.info(
            "FundamentalsSource ready: %s (pit_quality=%s)", self.meta.name, self.meta.pit_quality
        )

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        fields: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        if self.path.lower().endswith(".csv"):
            df = pd.read_csv(self.path)
        else:
            df = pd.read_parquet(self.path)
        ts_col = self.publish_ts_col
        if ts_col is None:
            ts_col = "publish_ts" if "publish_ts" in df.columns else PanelBuilder._detect_ts_col(df)
        if ts_col is None:
            raise ValueError("ParquetFundamentals: cannot detect publish timestamp column")
        out = df.copy()
        out["publish_ts"] = normalize_ts_ms(out[ts_col])
        if self.symbol_col != "symbol" and self.symbol_col in out.columns:
            out = out.rename(columns={self.symbol_col: "symbol"})
        out = out[out["symbol"].isin(list(symbols))]
        keep = ["publish_ts", "symbol"]
        if fields is not None:
            keep += [f for f in fields if f in out.columns]
        else:
            keep += [c for c in out.columns if c not in (ts_col, "publish_ts", "symbol")]
        return out[keep].reset_index(drop=True)


class FreeFundamentals:
    """Free фундаментал (yfinance) — СНИМОК на «сейчас», PIT-none.

    Историзации нет: yfinance отдаёт текущие значения. Поэтому ``pit_quality='none'``
    и громкое предупреждение — такой источник нельзя честно бэктестить, он годится
    лишь для live-скрининга. Для бэктеста используйте ``ParquetFundamentals`` (BYO).

    ``fetcher`` можно внедрить (DI/тесты): ``fetcher(symbol) -> dict`` метрик.
    """

    def __init__(
        self,
        *,
        asof_ms: Optional[int] = None,
        fetcher: Optional[Callable[[str], Mapping[str, Any]]] = None,
        name: str = "free:yfinance",
    ) -> None:
        self.asof_ms = asof_ms
        self._fetcher = fetcher
        self.meta = DataSourceMeta(
            name=name,
            vendor="yfinance",
            kind="fundamentals",
            pit_quality=PIT_NONE,
            free=True,
            notes=(
                "Snapshot fundamentals (no point-in-time history). NOT backtest-safe; "
                "use only for live screening. Bring BYO PIT fundamentals for backtests."
            ),
        )
        logger.warning(
            "FreeFundamentals '%s' is pit_quality=NONE: snapshot only, not backtest-safe.",
            name,
        )

    def _default_fetcher(self, symbol: str) -> Mapping[str, Any]:  # pragma: no cover - сеть
        import yfinance as yf

        info = yf.Ticker(symbol).info or {}
        return {
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": info.get("dividendYield"),
        }

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        fields: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        fetch = self._fetcher or self._default_fetcher
        asof = self.asof_ms if self.asof_ms is not None else 0
        rows = []
        for sym in symbols:
            try:
                metrics = dict(fetch(sym))
            except Exception as exc:  # pragma: no cover
                logger.warning("FreeFundamentals fetch failed for %s: %s", sym, exc)
                continue
            row = {"publish_ts": int(asof), "symbol": sym}
            if fields is not None:
                row.update({f: metrics.get(f) for f in fields})
            else:
                row.update(metrics)
            rows.append(row)
        cols = ["publish_ts", "symbol"] + (list(fields) if fields else [])
        return pd.DataFrame(rows, columns=cols if rows and fields else None)


# ---------------------------------------------------------------------------
# Total return (детерминированно; без внешних данных)
# ---------------------------------------------------------------------------
def total_return_index(
    close: pd.Series,
    *,
    dividends: Optional[Mapping[int, float]] = None,
    splits: Optional[Mapping[int, float]] = None,
) -> pd.Series:
    """Тотал-ретёрн индекс (реинвестирование дивидендов + корректировка сплитов).

    ``close`` индексируется по ts (по возрастанию). ``dividends``/``splits`` —
    ``{ts -> значение}`` по тем же ts. Сплит ``r`` (например 2.0 = 2-за-1) означает,
    что сырая цена на этом баре уже пост-сплит (вдвое меньше) — компенсируем ×r, чтобы
    не считать это «обвалом». Дивиденд на баре t (ex-date) добавляется к цене.

    Семантика: ``ret[t] = (close[t]*split_r[t] + div[t]) / close[t-1] - 1``;
    индекс = cumprod(1+ret), нормирован на ``close.iloc[0]``.
    """
    div = dict(dividends or {})
    spl = dict(splits or {})
    idx = list(close.index)
    vals = close.astype("float64").to_numpy()
    out = np.empty(len(vals), dtype="float64")
    if len(vals) == 0:
        return pd.Series(out, index=close.index, name="tr_close")
    out[0] = vals[0]
    for i in range(1, len(vals)):
        prev = vals[i - 1]
        if not np.isfinite(prev) or prev == 0:
            out[i] = out[i - 1]
            continue
        r = float(spl.get(idx[i], 1.0))
        d = float(div.get(idx[i], 0.0))
        ret = (vals[i] * r + d) / prev - 1.0
        out[i] = out[i - 1] * (1.0 + ret)
    return pd.Series(out, index=close.index, name="tr_close")


def add_total_return(
    df: pd.DataFrame,
    *,
    close_col: str = "close",
    ts_col: Optional[str] = None,
    dividends: Optional[Mapping[int, float]] = None,
    splits: Optional[Mapping[int, float]] = None,
    out_col: str = "tr_close",
) -> pd.DataFrame:
    """Добавить колонку тотал-ретёрн цены к одному прайс-кадру (per-symbol)."""
    if close_col not in df.columns:
        raise ValueError(f"add_total_return: missing close column '{close_col}'")
    out = df.copy()
    tcol = ts_col or PanelBuilder._detect_ts_col(out)
    if tcol is not None:
        ts_ms = normalize_ts_ms(out[tcol])
        close = pd.Series(out[close_col].astype("float64").to_numpy(), index=ts_ms)
    else:
        close = pd.Series(out[close_col].astype("float64").to_numpy(), index=np.arange(len(out)))
    tr = total_return_index(close, dividends=dividends, splits=splits)
    out[out_col] = tr.to_numpy()
    return out


def apply_corporate_actions_if_available(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Опционально делегировать в services.corporate_actions (если данные есть).

    Мягкая обёртка: при отсутствии сервиса/данных возвращает df без изменений.
    """
    try:
        from services.corporate_actions import adjust_prices_with_dividends

        return adjust_prices_with_dividends(df, symbol)
    except Exception as exc:  # pragma: no cover - зависит от наличия данных
        logger.debug("corporate_actions unavailable for %s: %s", symbol, exc)
        return df


# ---------------------------------------------------------------------------
# Convenience: source -> Panel
# ---------------------------------------------------------------------------
def build_price_panel(
    source: PriceSource,
    symbols: Sequence[str],
    timeframe: str,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
    columns: Optional[Sequence[str]] = None,
    align: str = "union",
    fill: str = "ffill",
) -> Panel:
    """Загрузить бары из источника и собрать Panel (через PanelBuilder)."""
    frames = source.get_bars(symbols, timeframe, start_ms=start_ms, end_ms=end_ms, limit=limit)
    return PanelBuilder.from_frames(frames, columns=columns, align=align, fill=fill)


def free_price_source(vendor: str = "yahoo", **kwargs: Any) -> AdapterPriceSource:
    """Фабрика free прайс-источника по вендору (yahoo/binance/alpaca/polygon/…)."""
    return AdapterPriceSource(vendor=vendor, **kwargs)


__all__ = [
    "PIT_TRUE",
    "PIT_APPROX",
    "PIT_NONE",
    "VALID_PIT_QUALITY",
    "PRICE_COLUMNS",
    "DataSourceMeta",
    "PriceSource",
    "FundamentalsSource",
    "AdapterPriceSource",
    "ParquetPriceSource",
    "ParquetFundamentals",
    "FreeFundamentals",
    "bars_to_frame",
    "total_return_index",
    "add_total_return",
    "apply_corporate_actions_if_available",
    "build_price_panel",
    "free_price_source",
]
