# -*- coding: utf-8 -*-
"""
impl_panel.py
=============

``PanelBuilder`` — сборка cross-sectional **Panel** (MultiIndex ``(ts_ms, symbol)``)
из произвольных per-symbol источников: бесплатных адаптеров, BYO-parquet/CSV или
``data_loader_multi_asset``. Слой ``impl_`` (зависит от ``core_portfolio``).

Принципы (см. CROSS_SECTIONAL_BUILD_ROADMAP.md, Stage A1):

* **Источник-агностично.** Принимает любой ``Dict[symbol -> DataFrame]`` (free или BYO).
* **Каноническое время.** Любой таймстемп (секунды / мс / микро / нано / datetime)
  нормализуется в ``ts_ms`` (int64 миллисекунды).
* **PIT-безопасность.** ``asof_join`` присоединяет внешние данные строго как-of с
  опциональным лагом публикации — без look-ahead.
* **Без побочных эффектов на существующий MVP.**
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from core_portfolio import (
    PANEL_INDEX_NAMES,
    SYMBOL_LEVEL,
    TS_LEVEL,
    Panel,
    validate_panel,
)

# Магнитудные пороги для определения единицы измерения времени.
_SEC_MAX = 1e11      # < 1e11  → секунды
_MS_MAX = 1e14       # < 1e14  → миллисекунды
_US_MAX = 1e17       # < 1e17  → микросекунды; иначе наносекунды


def normalize_ts_ms(values: Any) -> np.ndarray:
    """Нормализовать произвольные таймстемпы в int64 миллисекунды UTC.

    Поддерживает: datetime64/Timestamp, pandas.DatetimeIndex, числовые серии в
    секундах/мс/мкс/нс. Возвращает ``np.ndarray`` dtype int64.
    """
    s = pd.Series(values)

    # datetime-подобные → ms (устойчиво к разрешению ns/us и tz; pandas 3.0 без Series.view)
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.to_numpy().astype("datetime64[ms]").astype("int64")
    if pd.api.types.is_object_dtype(s):
        # попытка распарсить строки/даты
        try:
            dt = pd.to_datetime(s, utc=True, errors="raise")
            return dt.to_numpy().astype("datetime64[ms]").astype("int64")
        except Exception:
            s = pd.to_numeric(s, errors="coerce")

    s = pd.to_numeric(s, errors="coerce")
    arr = s.to_numpy(dtype="float64")
    finite = arr[np.isfinite(arr)]
    scale = 1000  # по умолчанию трактуем как секунды
    if finite.size:
        ref = float(np.nanmax(np.abs(finite)))
        if ref < _SEC_MAX:
            scale = 1000          # seconds → ms
        elif ref < _MS_MAX:
            scale = 1             # already ms
        elif ref < _US_MAX:
            scale = -1000         # microseconds → ms (divide)
        else:
            scale = -1_000_000    # nanoseconds → ms (divide)
    if scale >= 1:
        out = arr * scale
    else:
        out = arr / float(-scale)
    return np.rint(out).astype("int64")


class PanelBuilder:
    """Конструктор Panel из per-symbol источников."""

    DEFAULT_TS_CANDIDATES = (
        "ts_ms",
        "timestamp",
        "ts",
        "time",
        "datetime",
        "date",
        "close_time",
    )

    # ------------------------------------------------------------------
    # Основные конструкторы
    # ------------------------------------------------------------------
    @classmethod
    def from_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        *,
        columns: Optional[Sequence[str]] = None,
        ts_col: Optional[str] = None,
        symbol_col: str = "symbol",
        align: str = "union",            # 'union' | 'intersection' | 'none'
        fill: str = "ffill",             # 'ffill' | 'none'
        dropna_all_rows: bool = True,
    ) -> Panel:
        """Собрать Panel из словаря ``{symbol -> DataFrame}``.

        Каждый кадр может быть time-indexed или иметь столбец таймстемпа.
        Выравнивание ``union`` строит общую временную сетку по всем символам;
        ``fill='ffill'`` заполняет внутренние дыры последним известным значением
        (as-of корректно, без look-ahead).
        """
        if not isinstance(frames, Mapping):
            raise TypeError("frames must be a mapping {symbol: DataFrame}")

        long_parts = []
        for symbol, df in frames.items():
            if df is None or len(df) == 0:
                continue
            part = cls._to_long_single(df, str(symbol), ts_col, symbol_col, columns)
            if part is not None and len(part):
                long_parts.append(part)

        if not long_parts:
            from core_portfolio import empty_panel
            return empty_panel(columns)

        long = pd.concat(long_parts, axis=0)
        # снять дубликаты (ts_ms, symbol), оставив последний
        long = long[~long.index.duplicated(keep="last")]
        long = long.sort_index()

        panel = cls._align(long, align=align, fill=fill)

        if dropna_all_rows:
            panel = panel.dropna(how="all")

        panel.index.names = PANEL_INDEX_NAMES
        validate_panel(panel)
        return panel

    @classmethod
    def from_long(
        cls,
        df: pd.DataFrame,
        *,
        ts_col: Optional[str] = None,
        symbol_col: str = "symbol",
        columns: Optional[Sequence[str]] = None,
        align: str = "none",
        fill: str = "none",
    ) -> Panel:
        """Собрать Panel из уже «длинного» DataFrame (строка = (ts, symbol))."""
        if symbol_col not in df.columns:
            raise ValueError(f"long frame must contain '{symbol_col}' column")
        frames = {sym: g for sym, g in df.groupby(symbol_col, sort=False)}
        return cls.from_frames(
            frames,
            columns=columns,
            ts_col=ts_col,
            symbol_col=symbol_col,
            align=align,
            fill=fill,
        )

    @classmethod
    def from_data_loader(
        cls,
        paths: Sequence[Union[str, "os.PathLike[str]"]],
        *,
        asset_class: str = "crypto",
        timeframe: str = "4h",
        columns: Optional[Sequence[str]] = None,
        align: str = "union",
        fill: str = "ffill",
        **loader_kwargs: Any,
    ) -> Panel:
        """Удобный конструктор: ленивый импорт ``data_loader_multi_asset``.

        Не делает жёсткой зависимости — импорт внутри метода. BYO-пользователи
        могут обойтись ``from_frames`` без загрузчика.
        """
        from data_loader_multi_asset import load_multi_asset_data, AssetClass

        ac = asset_class
        try:
            ac = AssetClass(str(asset_class).lower())
        except Exception:
            ac = asset_class  # пусть загрузчик сам разберётся / упадёт явно

        frames, _obs = load_multi_asset_data(
            paths=list(paths),
            asset_class=ac,
            timeframe=timeframe,
            **loader_kwargs,
        )
        return cls.from_frames(frames, columns=columns, align=align, fill=fill)

    # ------------------------------------------------------------------
    # PIT as-of join
    # ------------------------------------------------------------------
    @classmethod
    def asof_join(
        cls,
        panel: Panel,
        other: pd.DataFrame,
        *,
        value_cols: Optional[Sequence[str]] = None,
        ts_col: Optional[str] = None,
        symbol_col: str = "symbol",
        publish_lag_ms: int = 0,
        suffix: str = "",
    ) -> Panel:
        """Присоединить внешние данные (например, фундаментал) строго as-of.

        Для каждой ``(ts_ms, symbol)`` берётся последняя запись ``other`` такая, что
        ``publish_ts + publish_lag_ms <= ts_ms`` — это исключает look-ahead.
        """
        validate_panel(panel)
        if other is None or len(other) == 0:
            return panel.copy()

        oth = other.copy()
        # определить столбец публикации
        pub_col = ts_col or cls._detect_ts_col(oth)
        if pub_col is None:
            raise ValueError("asof_join: cannot detect publish timestamp column in 'other'")
        if symbol_col not in oth.columns:
            raise ValueError(f"asof_join: 'other' must contain '{symbol_col}' column")

        oth["__pub_ms"] = normalize_ts_ms(oth[pub_col]) + int(publish_lag_ms)
        if value_cols is None:
            value_cols = [
                c for c in oth.columns
                if c not in (pub_col, symbol_col, "__pub_ms")
            ]
        value_cols = list(value_cols)
        if suffix:
            rename = {c: f"{c}{suffix}" for c in value_cols}
        else:
            rename = {}

        # left = панель в длинном виде с ts_ms колонкой
        left = panel.reset_index()  # columns: ts_ms, symbol, <features>
        out_chunks = []
        for sym, lg in left.groupby(SYMBOL_LEVEL, sort=False):
            rg = oth[oth[symbol_col] == sym]
            lg = lg.sort_values(TS_LEVEL)
            if len(rg) == 0:
                for c in value_cols:
                    lg[rename.get(c, c)] = np.nan
                out_chunks.append(lg)
                continue
            rg = rg.sort_values("__pub_ms")
            merged = pd.merge_asof(
                lg,
                rg[["__pub_ms"] + value_cols].rename(columns=rename),
                left_on=TS_LEVEL,
                right_on="__pub_ms",
                direction="backward",
            )
            merged = merged.drop(columns=["__pub_ms"], errors="ignore")
            out_chunks.append(merged)

        res = pd.concat(out_chunks, axis=0)
        res = res.set_index(list(PANEL_INDEX_NAMES)).sort_index()
        res.index.names = PANEL_INDEX_NAMES
        validate_panel(res)
        return res

    # ------------------------------------------------------------------
    # Forward returns (target helper для бэктеста/обучения alpha)
    # ------------------------------------------------------------------
    @staticmethod
    def add_forward_returns(
        panel: Panel,
        *,
        price_col: str = "close",
        horizon: int = 1,
        out_col: str = "fwd_return",
    ) -> Panel:
        """Добавить forward-return по каждому символу: r[t] = price[t+h]/price[t] - 1.

        Это ЯВНО будущая величина — использовать только как target (обучение alpha,
        оценка IC), НИКОГДА как фичу. Последние ``horizon`` точек по символу = NaN.
        """
        validate_panel(panel)
        if price_col not in panel.columns:
            raise ValueError(f"add_forward_returns: missing price column '{price_col}'")
        out = panel.copy()
        prices = out[price_col].astype("float64")
        fwd = prices.groupby(level=SYMBOL_LEVEL).transform(
            lambda s: s.shift(-int(horizon)) / s - 1.0
        )
        out[out_col] = fwd
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @classmethod
    def _to_long_single(
        cls,
        df: pd.DataFrame,
        symbol: str,
        ts_col: Optional[str],
        symbol_col: str,
        columns: Optional[Sequence[str]],
    ) -> Optional[pd.DataFrame]:
        d = df.copy()

        # 1) определить таймстемп
        col = ts_col or cls._detect_ts_col(d)
        if col is not None and col in d.columns:
            ts_ms = normalize_ts_ms(d[col])
        elif isinstance(d.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(d.index):
            ts_ms = normalize_ts_ms(pd.Series(d.index))
        elif pd.api.types.is_integer_dtype(getattr(d.index, "dtype", None)):
            ts_ms = normalize_ts_ms(pd.Series(d.index))
        else:
            raise ValueError(
                f"PanelBuilder: cannot detect timestamp for symbol '{symbol}' "
                f"(no ts column among {cls.DEFAULT_TS_CANDIDATES}, no datetime/int index)"
            )

        # 2) выбрать колонки фич
        drop = {symbol_col}
        if col is not None:
            drop.add(col)
        if columns is not None:
            feat_cols = [c for c in columns if c in d.columns]
        else:
            feat_cols = [c for c in d.columns if c not in drop]

        body = d[feat_cols].copy()
        body.index = pd.MultiIndex.from_arrays(
            [np.asarray(ts_ms, dtype="int64"), np.full(len(d), symbol, dtype=object)],
            names=PANEL_INDEX_NAMES,
        )
        body = body[~body.index.duplicated(keep="last")]
        return body

    @classmethod
    def _detect_ts_col(cls, df: pd.DataFrame) -> Optional[str]:
        lower = {c.lower(): c for c in df.columns}
        for cand in cls.DEFAULT_TS_CANDIDATES:
            if cand in lower:
                return lower[cand]
        return None

    @staticmethod
    def _align(long: pd.DataFrame, *, align: str, fill: str) -> pd.DataFrame:
        if align == "none":
            return long

        all_ts = np.array(
            sorted(set(int(t) for t in long.index.get_level_values(TS_LEVEL))),
            dtype="int64",
        )
        symbols = sorted(set(long.index.get_level_values(SYMBOL_LEVEL)))

        if align == "intersection":
            # пересечение таймстемпов, где присутствуют ВСЕ символы
            counts = long.groupby(level=TS_LEVEL).size()
            keep = counts[counts == len(symbols)].index
            grid_ts = np.array(sorted(int(t) for t in keep), dtype="int64")
        elif align == "union":
            grid_ts = all_ts
        else:
            raise ValueError(f"unknown align mode: {align!r}")

        full_index = pd.MultiIndex.from_product(
            [grid_ts, symbols], names=PANEL_INDEX_NAMES
        )
        aligned = long.reindex(full_index)

        if fill == "ffill":
            # ffill ТОЛЬКО внутри символа (as-of), не поперёк символов
            aligned = aligned.groupby(level=SYMBOL_LEVEL, group_keys=False).apply(
                lambda g: g.ffill()
            )
        elif fill != "none":
            raise ValueError(f"unknown fill mode: {fill!r}")

        return aligned.sort_index()


__all__ = ["PanelBuilder", "normalize_ts_ms"]
