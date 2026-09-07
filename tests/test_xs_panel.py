# -*- coding: utf-8 -*-
"""
Stage A1 tests — core_portfolio contracts + impl_panel.PanelBuilder.

Покрытие:
  * импорт контрактов, константы режимов, runtime_checkable Protocols
  * normalize_ts_ms: секунды / мс / мкс / нс / datetime
  * PanelBuilder.from_frames: MultiIndex (ts_ms, symbol), сортировка, dedup
  * union-выравнивание + ffill (нет внутренних NaN-дыр)
  * intersection-выравнивание
  * validate_panel: good / bad
  * cross_section helper
  * asof_join: PIT (publish lag блокирует look-ahead)
  * add_forward_returns: корректность
  * core_config.CommonRunConfig.mode (аддитивно, дефолт = current)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import core_portfolio as cp
from impl_panel import PanelBuilder, normalize_ts_ms


# Реалистичные таймстемпы (иначе магнитудная эвристика спутает мелкие числа).
T0_SEC = 1_700_000_000  # ~2023-11-14, секунды
STEP_SEC = 3_600  # 1 час
T0_MS = T0_SEC * 1000


def _sym_frame(ts_sec, closes, symbol):
    return pd.DataFrame(
        {
            "timestamp": np.asarray(ts_sec, dtype="int64"),
            "symbol": symbol,
            "close": np.asarray(closes, dtype="float64"),
        }
    )


# ---------------------------------------------------------------------------
# Контракты / константы
# ---------------------------------------------------------------------------
def test_contracts_and_constants_import():
    assert cp.PANEL_INDEX_NAMES == ("ts_ms", "symbol")
    assert cp.MODE_SINGLE_INSTRUMENT == "single_instrument"
    assert cp.MODE_CROSS_SECTIONAL == "cross_sectional"
    assert set(cp.VALID_RUN_MODES) == {"single_instrument", "cross_sectional"}
    for proto in (
        cp.UniverseProvider,
        cp.Signal,
        cp.AlphaModel,
        cp.RiskModel,
        cp.PortfolioConstructor,
        cp.CrossSectionalStrategy,
    ):
        assert hasattr(proto, "_is_runtime_protocol")


def test_rebalance_event_helpers():
    w = pd.Series({"A": 0.6, "B": -0.4})
    ev = cp.RebalanceEvent(ts_ms=T0_MS, target_weights=w)
    assert ev.ts_ms == T0_MS
    assert set(ev.symbols()) == {"A", "B"}
    assert ev.gross() == pytest.approx(1.0)
    assert ev.net() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# normalize_ts_ms
# ---------------------------------------------------------------------------
def test_normalize_ts_ms_units():
    # dtype=int64 обязателен: на Windows np.array(python ints) даёт int32 и *1000 переполняется
    sec = np.array([T0_SEC, T0_SEC + STEP_SEC], dtype="int64")
    ms = sec * 1000
    us = ms * 1000
    ns = ms * 1_000_000

    assert list(normalize_ts_ms(sec)) == list(ms)
    assert list(normalize_ts_ms(ms)) == list(ms)
    assert list(normalize_ts_ms(us)) == list(ms)
    assert list(normalize_ts_ms(ns)) == list(ms)


def test_normalize_ts_ms_datetime():
    dt = pd.to_datetime(["2023-11-14T00:00:00Z", "2023-11-14T01:00:00Z"])
    out = normalize_ts_ms(dt)
    assert out.dtype == np.dtype("int64")
    # шаг между точками = 1 час = 3_600_000 мс
    assert int(out[1] - out[0]) == 3_600_000


# ---------------------------------------------------------------------------
# from_frames: структура
# ---------------------------------------------------------------------------
def test_from_frames_multiindex_and_sorted():
    ts = [T0_SEC, T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC]
    frames = {
        "AAA": _sym_frame(ts, [10, 11, 12], "AAA"),
        "BBB": _sym_frame(ts, [20, 21, 22], "BBB"),
    }
    panel = PanelBuilder.from_frames(frames)
    cp.validate_panel(panel, allow_empty=False)
    assert tuple(panel.index.names) == ("ts_ms", "symbol")
    assert panel.index.is_monotonic_increasing
    assert cp.panel_symbols(panel) == ["AAA", "BBB"]
    assert cp.panel_timestamps(panel) == [t * 1000 for t in ts]
    # ts_ms — целочисленный уровень
    assert pd.api.types.is_integer_dtype(panel.index.get_level_values("ts_ms").dtype)


def test_from_frames_dedup_keeps_last():
    ts = [T0_SEC, T0_SEC, T0_SEC + STEP_SEC]  # дубликат первой точки
    frames = {"AAA": _sym_frame(ts, [10, 99, 11], "AAA")}
    panel = PanelBuilder.from_frames(frames, align="none")
    assert not panel.index.has_duplicates
    val = panel.xs(T0_MS, level="ts_ms")["close"].iloc[0]
    assert val == pytest.approx(99.0)  # keep='last'


# ---------------------------------------------------------------------------
# Выравнивание
# ---------------------------------------------------------------------------
def test_union_alignment_ffill_no_internal_gaps():
    full_ts = [T0_SEC, T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC]
    # BBB пропускает среднюю точку
    frames = {
        "AAA": _sym_frame(full_ts, [10, 11, 12], "AAA"),
        "BBB": _sym_frame([full_ts[0], full_ts[2]], [20, 22], "BBB"),
    }
    panel = PanelBuilder.from_frames(frames, align="union", fill="ffill")
    # общая сетка = 3 точки × 2 символа
    assert len(cp.panel_timestamps(panel)) == 3
    # внутренняя дыра BBB@t1 заполнена последним значением (20), не NaN
    mid_ms = (T0_SEC + STEP_SEC) * 1000
    bbb_mid = panel.loc[(mid_ms, "BBB"), "close"]
    assert bbb_mid == pytest.approx(20.0)
    assert panel["close"].isna().sum() == 0


def test_intersection_alignment():
    frames = {
        "AAA": _sym_frame([T0_SEC, T0_SEC + STEP_SEC], [10, 11], "AAA"),
        "BBB": _sym_frame([T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC], [20, 21], "BBB"),
    }
    panel = PanelBuilder.from_frames(frames, align="intersection", fill="none")
    # общий таймстемп только один — T0+STEP
    assert cp.panel_timestamps(panel) == [(T0_SEC + STEP_SEC) * 1000]
    assert set(cp.panel_symbols(panel)) == {"AAA", "BBB"}


# ---------------------------------------------------------------------------
# validate_panel / helpers
# ---------------------------------------------------------------------------
def test_validate_panel_rejects_bad():
    with pytest.raises(TypeError):
        cp.validate_panel([1, 2, 3])
    df = pd.DataFrame({"close": [1.0, 2.0]})  # single index
    with pytest.raises(ValueError):
        cp.validate_panel(df)
    # неправильные имена уровней
    idx = pd.MultiIndex.from_tuples([(1, "A")], names=["t", "sym"])
    bad = pd.DataFrame({"close": [1.0]}, index=idx)
    with pytest.raises(ValueError):
        cp.validate_panel(bad)


def test_empty_panel_is_valid():
    p = cp.empty_panel(["close", "rsi"])
    cp.validate_panel(p, allow_empty=True)
    assert list(p.columns) == ["close", "rsi"]
    assert len(p) == 0


def test_cross_section_helper():
    ts = [T0_SEC, T0_SEC + STEP_SEC]
    frames = {
        "AAA": _sym_frame(ts, [10, 11], "AAA"),
        "BBB": _sym_frame(ts, [20, 21], "BBB"),
    }
    panel = PanelBuilder.from_frames(frames)
    xs = cp.cross_section(panel, T0_MS)
    assert set(xs.index) == {"AAA", "BBB"}
    assert xs.loc["BBB", "close"] == pytest.approx(20.0)
    # несуществующая дата → пустой срез с теми же колонками
    empty = cp.cross_section(panel, 1)
    assert list(empty.columns) == list(panel.columns)
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# PIT as-of join
# ---------------------------------------------------------------------------
def test_asof_join_is_point_in_time():
    ts = [T0_SEC, T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC]
    panel = PanelBuilder.from_frames({"AAA": _sym_frame(ts, [10, 11, 12], "AAA")})

    bar0, bar1, bar2 = (t * 1000 for t in ts)
    # фундаментал опубликован МЕЖДУ барами (в мс, согласовано с панелью)
    pub0 = bar0 + 1_800_000  # между bar0 и bar1
    pub1 = bar1 + 1_400_000  # между bar1 и bar2
    other = pd.DataFrame({"timestamp": [pub0, pub1], "symbol": ["AAA", "AAA"], "ep": [0.05, 0.06]})

    joined = PanelBuilder.asof_join(panel, other, value_cols=["ep"])
    # bar0: ничего ещё не опубликовано → NaN (нет look-ahead)
    assert np.isnan(joined.loc[(bar0, "AAA"), "ep"])
    # bar1: виден pub0
    assert joined.loc[(bar1, "AAA"), "ep"] == pytest.approx(0.05)
    # bar2: виден pub1
    assert joined.loc[(bar2, "AAA"), "ep"] == pytest.approx(0.06)


def test_asof_join_publish_lag_blocks_lookahead():
    ts = [T0_SEC, T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC]
    panel = PanelBuilder.from_frames({"AAA": _sym_frame(ts, [10, 11, 12], "AAA")})
    bar0, bar1, bar2 = (t * 1000 for t in ts)
    pub0 = bar0 + 1_800_000
    other = pd.DataFrame({"timestamp": [pub0], "symbol": ["AAA"], "ep": [0.05]})

    # лаг публикации 1 час сдвигает доступность за bar1 → виден только к bar2
    joined = PanelBuilder.asof_join(panel, other, value_cols=["ep"], publish_lag_ms=3_600_000)
    assert np.isnan(joined.loc[(bar0, "AAA"), "ep"])
    assert np.isnan(joined.loc[(bar1, "AAA"), "ep"])
    assert joined.loc[(bar2, "AAA"), "ep"] == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# forward returns
# ---------------------------------------------------------------------------
def test_add_forward_returns():
    ts = [T0_SEC, T0_SEC + STEP_SEC, T0_SEC + 2 * STEP_SEC]
    panel = PanelBuilder.from_frames({"AAA": _sym_frame(ts, [100, 110, 121], "AAA")})
    out = PanelBuilder.add_forward_returns(panel, price_col="close", horizon=1)
    bar0, bar1, bar2 = (t * 1000 for t in ts)
    assert out.loc[(bar0, "AAA"), "fwd_return"] == pytest.approx(0.10)  # 110/100-1
    assert out.loc[(bar1, "AAA"), "fwd_return"] == pytest.approx(0.10)  # 121/110-1
    assert np.isnan(out.loc[(bar2, "AAA"), "fwd_return"])  # последний → NaN


# ---------------------------------------------------------------------------
# config integration (аддитивно)
# ---------------------------------------------------------------------------
def test_config_mode_field_is_additive():
    # CommonRunConfig имеет обязательные под-секции (components/risk_guards), поэтому
    # проверяем добавленное поле через model_fields — это и есть контракт Stage A1:
    # поле есть, дефолт = текущее поведение, тип строковый, extra по-прежнему allow.
    from core_config import CommonRunConfig

    fields = CommonRunConfig.model_fields
    assert "mode" in fields
    assert fields["mode"].default == "single_instrument"
    assert fields["mode"].default in cp.VALID_RUN_MODES
    # не сломали разрешение extra-полей (обратная совместимость конфигов)
    assert CommonRunConfig.model_config.get("extra") == "allow"
    # существующие поля на месте
    assert "asset_class" in fields and fields["asset_class"].default == "crypto"
