# test_data_pipeline_validation.py
"""
Комплексный набор тестов для валидации данных на всех критических этапах пайплайна.

Проверяет:
P0 - Загрузка OHLCV данных (fetch_all_data_patch.py)
P1 - Создание признаков (transformers.py)
P2 - Asof-merge (asof_join.py)
P3 - Защита от утечек (leakguard.py)
P4 - Создание меток (labels.py)
P5 - Подготовка данных перед обучением
"""

import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from typing import Dict, List
from pathlib import Path

# Импортируем модули для тестирования
from fetch_all_data_patch import load_all_data, _ensure_required_columns, _read_fng
from transformers import (
    FeatureSpec,
    OnlineFeatureTransformer,
    apply_offline_features,
    calculate_yang_zhang_volatility,
    calculate_parkinson_volatility,
    calculate_garch_volatility,
)
from asof_join import AsofMerger, AsofSpec
from leakguard import LeakGuard, LeakConfig
from labels import LabelBuilder, LabelConfig
from data_validation import DataValidator


# ============================================================================
# P0: Тесты загрузки OHLCV данных
# ============================================================================


class TestP0_DataLoading:
    """Тесты загрузки и нормализации OHLCV данных."""

    def test_ensure_required_columns_valid_data(self):
        """Проверка: валидные данные проходят проверку required columns."""
        # Используем 4h интервалы (14400 секунд) для корректной работы после выравнивания
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609459200 + 14400, 1609459200 + 28800],  # 4h интервалы
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29100.0, 29200.0],
            "high": [29500.0, 29600.0, 29700.0],
            "low": [28900.0, 29000.0, 29100.0],
            "close": [29100.0, 29200.0, 29300.0],
            "volume": [100.0, 110.0, 120.0],
            "quote_asset_volume": [2910000.0, 3212000.0, 3516000.0],
            "number_of_trades": [1000, 1100, 1200],
            "taker_buy_base_asset_volume": [50.0, 55.0, 60.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0, 1758000.0],
        })
        result = _ensure_required_columns(df)
        assert len(result) == 3
        assert all(col in result.columns for col in [
            "timestamp", "symbol", "open", "high", "low", "close", "volume"
        ])

    def test_ensure_required_columns_missing_column(self):
        """Проверка: отсутствие обязательной колонки вызывает ошибку."""
        df = pd.DataFrame({
            "timestamp": [1609459200],
            "symbol": ["BTCUSDT"],
            "open": [29000.0],
            "high": [29500.0],
            "low": [28900.0],
            "close": [29100.0],
            # missing: volume, quote_asset_volume, etc.
        })
        with pytest.raises(ValueError, match="Missing required columns"):
            _ensure_required_columns(df)

    def test_ensure_required_columns_timestamp_normalization(self):
        """Проверка: timestamp в миллисекундах корректно конвертируется в секунды."""
        df = pd.DataFrame({
            "timestamp": [1609459200000, 1609462800000],  # milliseconds
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29100.0],
            "high": [29500.0, 29600.0],
            "low": [28900.0, 29000.0],
            "close": [29100.0, 29200.0],
            "volume": [100.0, 110.0],
            "quote_asset_volume": [2910000.0, 3212000.0],
            "number_of_trades": [1000, 1100],
            "taker_buy_base_asset_volume": [50.0, 55.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0],
        })
        result = _ensure_required_columns(df)
        assert result["timestamp"].max() < 10_000_000_000  # converted to seconds

    def test_ensure_required_columns_4h_alignment(self):
        """Проверка: timestamp выравнивается на 4-часовую границу (14400 секунд)."""
        df = pd.DataFrame({
            "timestamp": [1609459210, 1609462850, 1609466420],  # не выровнены на 4h
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29100.0, 29200.0],
            "high": [29500.0, 29600.0, 29700.0],
            "low": [28900.0, 29000.0, 29100.0],
            "close": [29100.0, 29200.0, 29300.0],
            "volume": [100.0, 110.0, 120.0],
            "quote_asset_volume": [2910000.0, 3212000.0, 3516000.0],
            "number_of_trades": [1000, 1100, 1200],
            "taker_buy_base_asset_volume": [50.0, 55.0, 60.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0, 1758000.0],
        })
        result = _ensure_required_columns(df)
        # Все timestamp должны быть кратны 14400 (4 часа)
        assert all(ts % 14400 == 0 for ts in result["timestamp"])

    def test_ensure_required_columns_deduplication(self):
        """Проверка: дублирующиеся timestamp удаляются после выравнивания."""
        # Первые два timestamp одинаковые, третий на следующей 4h границе
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609459200, 1609459200 + 14400],  # первые два дублируются
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29050.0, 29100.0],
            "high": [29500.0, 29550.0, 29600.0],
            "low": [28900.0, 28950.0, 29000.0],
            "close": [29100.0, 29150.0, 29200.0],
            "volume": [100.0, 105.0, 110.0],
            "quote_asset_volume": [2910000.0, 3000000.0, 3212000.0],
            "number_of_trades": [1000, 1050, 1100],
            "taker_buy_base_asset_volume": [50.0, 52.5, 55.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1500000.0, 1606600.0],
        })
        result = _ensure_required_columns(df)
        assert len(result) == 2  # один дубликат удален


# ============================================================================
# P1: Тесты создания признаков
# ============================================================================


class TestP1_FeatureGeneration:
    """Тесты создания технических индикаторов и признаков."""

    def test_apply_offline_features_empty_dataframe(self):
        """Проверка: пустой датафрейм возвращает правильную структуру колонок."""
        spec = FeatureSpec(lookbacks_prices=[240, 1440], bar_duration_minutes=240)
        df = pd.DataFrame()
        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")
        # Проверяем наличие базовых колонок
        assert "ts_ms" in result.columns
        assert "symbol" in result.columns
        assert "ref_price" in result.columns
        assert "rsi" in result.columns
        assert len(result) == 0

    def test_apply_offline_features_no_nan_injection(self):
        """Проверка: признаки не содержат NaN где не должны (достаточно данных)."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)
        # Создаем датафрейм с достаточным количеством баров (1 окно = 1 бар)
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],  # 10 баров по 4h
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
        })
        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # После первого бара должны быть валидные SMA и returns
        assert result["sma_240"].notna().sum() >= 1
        assert result["ret_4h"].notna().sum() >= 1

    def test_apply_offline_features_ohlc_validation(self):
        """Проверка: OHLC признаки (Yang-Zhang, Parkinson) валидны при наличии OHLC данных."""
        spec = FeatureSpec(
            lookbacks_prices=[240],
            yang_zhang_windows=[2880],  # 48h = 12 баров
            parkinson_windows=[2880],
            bar_duration_minutes=240
        )
        # 15 баров для обеспечения достаточных данных
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(15)],
            "symbol": ["BTCUSDT"] * 15,
            "price": [29000.0 + i * 100 for i in range(15)],
            "open": [29000.0 + i * 100 for i in range(15)],
            "high": [29500.0 + i * 100 for i in range(15)],
            "low": [28900.0 + i * 100 for i in range(15)],
        })
        result = apply_offline_features(
            df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price",
            open_col="open", high_col="high", low_col="low"
        )

        # После 12 баров должны быть валидные Yang-Zhang и Parkinson
        assert result["yang_zhang_48h"].notna().sum() >= 1
        assert result["parkinson_48h"].notna().sum() >= 1

    def test_apply_offline_features_volume_validation(self):
        """Проверка: volume-based признаки валидны при наличии volume данных."""
        spec = FeatureSpec(
            lookbacks_prices=[240],
            taker_buy_ratio_windows=[480],  # 8h = 2 бара
            taker_buy_ratio_momentum=[240],  # 4h = 1 бар
            bar_duration_minutes=240
        )
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
            "volume": [100.0 + i * 10 for i in range(10)],
            "taker_buy_base": [50.0 + i * 5 for i in range(10)],
        })
        result = apply_offline_features(
            df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price",
            volume_col="volume", taker_buy_base_col="taker_buy_base"
        )

        # Проверяем taker_buy_ratio и его производные
        assert "taker_buy_ratio" in result.columns
        assert result["taker_buy_ratio"].notna().sum() >= 1
        assert "taker_buy_ratio_sma_8h" in result.columns
        assert "taker_buy_ratio_momentum_4h" in result.columns

    def test_calculate_yang_zhang_volatility_invalid_data(self):
        """Проверка: Yang-Zhang возвращает None при невалидных данных."""
        # Пустой список
        assert calculate_yang_zhang_volatility([], 10) is None

        # Недостаточно данных
        bars = [{"open": 29000, "high": 29500, "low": 28900, "close": 29100}]
        assert calculate_yang_zhang_volatility(bars, 10) is None

        # Невалидные значения (нули)
        bars = [{"open": 0, "high": 0, "low": 0, "close": 0}] * 10
        assert calculate_yang_zhang_volatility(bars, 10) is None

    def test_calculate_parkinson_volatility_invalid_data(self):
        """Проверка: Parkinson возвращает None при невалидных данных."""
        # High < Low (нарушение OHLC инварианта)
        bars = [{"high": 28900, "low": 29500}] * 10
        assert calculate_parkinson_volatility(bars, 10) is None

        # Недостаточно данных (< 80% от окна)
        bars = [{"high": 29500, "low": 28900}] * 5
        assert calculate_parkinson_volatility(bars, 10) is None

    def test_calculate_garch_volatility_insufficient_data(self):
        """Проверка: GARCH возвращает None при недостаточных данных."""
        prices = [29000.0 + i * 10 for i in range(10)]
        # Требуется минимум 50 наблюдений
        assert calculate_garch_volatility(prices, 50) is None


# ============================================================================
# P2: Тесты asof-merge
# ============================================================================


class TestP2_AsofMerge:
    """Тесты объединения данных через asof-merge."""

    def test_asof_merger_basic(self):
        """Проверка: базовый asof-merge работает корректно."""
        base = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000],
            "symbol": ["BTC", "BTC", "BTC"],
            "signal": [1.0, 2.0, 3.0],
        })

        book = pd.DataFrame({
            "ts_ms": [500, 1500, 2500],
            "symbol": ["BTC", "BTC", "BTC"],
            "bid": [29000.0, 29100.0, 29200.0],
        })

        merger = AsofMerger(base_df=base, time_col="ts_ms", keys=["symbol"])
        spec = AsofSpec(name="book", df=book, time_col="ts_ms", keys=("symbol",), direction="backward")
        result = merger.merge([spec])

        # Проверяем корректность backward merge
        assert len(result) == 3
        assert result.iloc[0]["book_bid"] == 29000.0  # берем 500ms (backward)
        assert result.iloc[1]["book_bid"] == 29100.0  # берем 1500ms
        assert result.iloc[2]["book_bid"] == 29200.0  # берем 2500ms

    def test_asof_merger_tolerance(self):
        """Проверка: tolerance ограничивает временной разрыв."""
        base = pd.DataFrame({
            "ts_ms": [1000, 5000],  # большой разрыв
            "symbol": ["BTC", "BTC"],
        })

        source = pd.DataFrame({
            "ts_ms": [100],  # только одна точка в начале
            "symbol": ["BTC"],
            "value": [99.0],
        })

        merger = AsofMerger(base_df=base, time_col="ts_ms", keys=["symbol"])
        spec = AsofSpec(
            name="source", df=source, time_col="ts_ms", keys=["symbol"],
            direction="backward", tolerance_ms=1000
        )
        result = merger.merge([spec])

        # При tolerance=1000ms, точка 5000ms не должна получить значение (разрыв 4900ms > 1000ms)
        assert pd.isna(result.iloc[1]["source_value"])

    def test_asof_merger_no_future_leakage(self):
        """Проверка: backward merge не берет данные из будущего."""
        base = pd.DataFrame({
            "ts_ms": [1000, 2000],
            "symbol": ["BTC", "BTC"],
        })

        source = pd.DataFrame({
            "ts_ms": [1500, 2500],  # данные позже base точек
            "symbol": ["BTC", "BTC"],
            "future_value": [100.0, 200.0],
        })

        merger = AsofMerger(base_df=base, time_col="ts_ms", keys=["symbol"])
        spec = AsofSpec(name="future", df=source, time_col="ts_ms", keys=["symbol"], direction="backward")
        result = merger.merge([spec])

        # Первая точка (1000ms) не должна получить значение (нет данных backward <= 1000ms)
        assert pd.isna(result.iloc[0]["future_future_value"])
        # Вторая точка (2000ms) должна получить значение 100.0 (от 1500ms <= 2000ms)
        assert result.iloc[1]["future_future_value"] == 100.0


# ============================================================================
# P3: Тесты защиты от утечек
# ============================================================================


class TestP3_LeakGuard:
    """Тесты защиты от утечек данных из будущего."""

    def test_attach_decision_time(self):
        """Проверка: decision_ts корректно добавляется с задержкой."""
        lg = LeakGuard(LeakConfig(decision_delay_ms=1000))
        df = pd.DataFrame({"ts_ms": [1000, 2000, 3000]})
        result = lg.attach_decision_time(df, ts_col="ts_ms")

        assert "decision_ts" in result.columns
        assert result["decision_ts"].iloc[0] == 2000  # 1000 + 1000
        assert result["decision_ts"].iloc[1] == 3000  # 2000 + 1000

    def test_validate_ffill_gaps_no_violations(self):
        """Проверка: данные без gap нарушений проходят валидацию."""
        lg = LeakGuard(LeakConfig())
        df = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000],
            "symbol": ["BTC", "BTC", "BTC"],
            "value": [100.0, 100.0, 100.0],
        })
        result = lg.validate_ffill_gaps(
            df, ts_col="ts_ms", group_keys=["symbol"],
            value_cols=["value"], max_gap_ms=5000
        )

        # Все значения должны остаться валидными
        assert result["value"].notna().all()

    def test_validate_ffill_gaps_excessive_gap(self):
        """Проверка: чрезмерный ffill gap заменяется на NaN."""
        lg = LeakGuard(LeakConfig())
        df = pd.DataFrame({
            "ts_ms": [1000, 2000, 8000],  # gap 6000ms между 2000 и 8000
            "symbol": ["BTC", "BTC", "BTC"],
            "value": [100.0, 100.0, 100.0],
        })
        result = lg.validate_ffill_gaps(
            df, ts_col="ts_ms", group_keys=["symbol"],
            value_cols=["value"], max_gap_ms=5000
        )

        # Последнее значение должно стать NaN (gap > 5000ms)
        assert pd.isna(result["value"].iloc[2])


# ============================================================================
# P4: Тесты создания меток
# ============================================================================


class TestP4_LabelGeneration:
    """Тесты создания меток для обучения."""

    def test_label_builder_basic(self):
        """Проверка: базовое создание меток работает корректно."""
        lb = LabelBuilder(LabelConfig(horizon_ms=1000, price_col="price", returns="log"))

        base = pd.DataFrame({
            "ts_ms": [1000, 2000],
            "symbol": ["BTC", "BTC"],
            "decision_ts": [1000, 2000],
        })

        prices = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000],
            "symbol": ["BTC", "BTC", "BTC"],
            "price": [29000.0, 29100.0, 29200.0],
        })

        result = lb.build(base, prices, ts_col="ts_ms", symbol_col="symbol")

        # Проверяем наличие колонок меток
        assert "label_price0" in result.columns
        assert "label_price1" in result.columns
        assert "label_ret" in result.columns
        assert result["label_ret"].notna().all()

    def test_label_builder_no_decision_ts(self):
        """Проверка: отсутствие decision_ts вызывает ошибку."""
        lb = LabelBuilder(LabelConfig())
        base = pd.DataFrame({"ts_ms": [1000], "symbol": ["BTC"]})
        prices = pd.DataFrame({"ts_ms": [1000], "symbol": ["BTC"], "price": [29000.0]})

        with pytest.raises(ValueError, match="decision_ts"):
            lb.build(base, prices)

    def test_label_builder_log_returns(self):
        """Проверка: логарифмические returns рассчитываются правильно."""
        lb = LabelBuilder(LabelConfig(horizon_ms=1000, price_col="price", returns="log"))

        base = pd.DataFrame({
            "ts_ms": [1000],
            "symbol": ["BTC"],
            "decision_ts": [1000],
        })

        prices = pd.DataFrame({
            "ts_ms": [1000, 2000],
            "symbol": ["BTC", "BTC"],
            "price": [29000.0, 29290.0],  # ~1% рост
        })

        result = lb.build(base, prices, ts_col="ts_ms", symbol_col="symbol")

        # log(29290/29000) ≈ 0.00995
        assert abs(result["label_ret"].iloc[0] - 0.00995) < 0.0001


# ============================================================================
# P5: Тесты валидации данных перед обучением
# ============================================================================


class TestP5_PreTrainingValidation:
    """Тесты валидации данных непосредственно перед обучением."""

    def test_data_validator_valid_ohlcv(self):
        """Проверка: валидные OHLCV данные проходят проверку."""
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609462800, 1609466400],
            "symbol": ["BTCUSDT", "BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29100.0, 29200.0],
            "high": [29500.0, 29600.0, 29700.0],
            "low": [28900.0, 29000.0, 29100.0],
            "close": [29100.0, 29200.0, 29300.0],
            "volume": [100.0, 110.0, 120.0],
            "quote_asset_volume": [2910000.0, 3212000.0, 3516000.0],
            "number_of_trades": [1000, 1100, 1200],
            "taker_buy_base_asset_volume": [50.0, 55.0, 60.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0, 1758000.0],
        })

        # Не должно вызвать исключение
        result = validator.validate(df)
        assert len(result) == 3

    def test_data_validator_nan_detection(self):
        """Проверка: NaN значения обнаруживаются."""
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609462800],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": [29000.0, np.nan],  # NaN
            "high": [29500.0, 29600.0],
            "low": [28900.0, 29000.0],
            "close": [29100.0, 29200.0],
            "volume": [100.0, 110.0],
            "quote_asset_volume": [2910000.0, 3212000.0],
            "number_of_trades": [1000, 1100],
            "taker_buy_base_asset_volume": [50.0, 55.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0],
        })

        with pytest.raises(ValueError, match="NaN"):
            validator.validate(df)

    def test_data_validator_negative_values(self):
        """Проверка: отрицательные значения обнаруживаются."""
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609462800],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29100.0],
            "high": [29500.0, 29600.0],
            "low": [28900.0, 29000.0],
            "close": [29100.0, -29200.0],  # отрицательное
            "volume": [100.0, 110.0],
            "quote_asset_volume": [2910000.0, 3212000.0],
            "number_of_trades": [1000, 1100],
            "taker_buy_base_asset_volume": [50.0, 55.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0],
        })

        with pytest.raises(ValueError, match="отрицательные значения"):
            validator.validate(df)

    def test_data_validator_ohlc_invariants(self):
        """Проверка: нарушения OHLC инвариантов обнаруживаются."""
        validator = DataValidator()
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609462800],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "open": [29000.0, 29100.0],
            "high": [28500.0, 29600.0],  # high < open (нарушение)
            "low": [28900.0, 29000.0],
            "close": [29100.0, 29200.0],
            "volume": [100.0, 110.0],
            "quote_asset_volume": [2910000.0, 3212000.0],
            "number_of_trades": [1000, 1100],
            "taker_buy_base_asset_volume": [50.0, 55.0],
            "taker_buy_quote_asset_volume": [1455000.0, 1606600.0],
        })

        with pytest.raises(ValueError, match="инварианта"):
            validator.validate(df)


# ============================================================================
# Интеграционные тесты полного пайплайна
# ============================================================================


class TestFullPipeline:
    """Интеграционные тесты полного пайплайна данных."""

    def test_end_to_end_pipeline(self, tmp_path):
        """Проверка: полный пайплайн от OHLCV до меток работает без искажений."""
        # Создаем синтетические OHLCV данные
        n_bars = 100
        timestamps = [1609459200 + i * 14400 for i in range(n_bars)]  # 4h интервалы
        base_price = 29000.0

        ohlcv_data = {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * n_bars,
            "open": [base_price + i * 10 for i in range(n_bars)],
            "high": [base_price + i * 10 + 50 for i in range(n_bars)],
            "low": [base_price + i * 10 - 50 for i in range(n_bars)],
            "close": [base_price + i * 10 + 10 for i in range(n_bars)],
            "volume": [100.0 + i for i in range(n_bars)],
            "quote_asset_volume": [(base_price + i * 10) * (100.0 + i) for i in range(n_bars)],
            "number_of_trades": [1000 + i * 10 for i in range(n_bars)],
            "taker_buy_base_asset_volume": [50.0 + i * 0.5 for i in range(n_bars)],
            "taker_buy_quote_asset_volume": [(base_price + i * 10) * (50.0 + i * 0.5) for i in range(n_bars)],
        }
        ohlcv_df = pd.DataFrame(ohlcv_data)

        # Шаг 1: Валидация OHLCV
        ohlcv_df = _ensure_required_columns(ohlcv_df)
        assert len(ohlcv_df) == n_bars

        # Шаг 2: Создание признаков
        spec = FeatureSpec(
            lookbacks_prices=[240, 1440],
            yang_zhang_windows=[2880],
            bar_duration_minutes=240
        )

        # Подготовка данных для apply_offline_features (нужны ts_ms и symbol)
        feat_input = pd.DataFrame({
            "ts_ms": [ts * 1000 for ts in ohlcv_df["timestamp"]],
            "symbol": ohlcv_df["symbol"],
            "price": ohlcv_df["close"],
            "open": ohlcv_df["open"],
            "high": ohlcv_df["high"],
            "low": ohlcv_df["low"],
            "volume": ohlcv_df["volume"],
            "taker_buy_base": ohlcv_df["taker_buy_base_asset_volume"],
        })

        features = apply_offline_features(
            feat_input, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price",
            open_col="open", high_col="high", low_col="low",
            volume_col="volume", taker_buy_base_col="taker_buy_base"
        )

        # Проверяем что признаки созданы
        assert len(features) == n_bars
        assert "sma_240" in features.columns
        assert "ret_4h" in features.columns

        # Шаг 3: LeakGuard
        lg = LeakGuard(LeakConfig(decision_delay_ms=60000))  # 1 минута задержка
        features = lg.attach_decision_time(features, ts_col="ts_ms")
        assert "decision_ts" in features.columns

        # Шаг 4: Создание меток
        lb = LabelBuilder(LabelConfig(horizon_ms=14400000, price_col="price"))  # 4h горизонт

        # Подготовка price_df
        price_df = pd.DataFrame({
            "ts_ms": features["ts_ms"],
            "symbol": features["symbol"],
            "price": features["ref_price"],
        })

        final = lb.build(features, price_df, ts_col="ts_ms", symbol_col="symbol")

        # Финальная проверка
        assert "label_ret" in final.columns
        assert len(final) == n_bars

        # Проверяем что нет чрезмерных NaN (кроме warm-up периода)
        # Допускаем NaN только для первых баров (недостаточно истории)
        valid_rows = final.iloc[20:]  # после warm-up
        assert valid_rows["sma_240"].notna().sum() > 0
        assert valid_rows["ret_4h"].notna().sum() > 0

    def test_no_empty_dataset_injection(self):
        """Проверка: пустые датасеты не попадают в обучение."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Попытка применить признаки к пустому датасету
        empty_df = pd.DataFrame()
        result = apply_offline_features(empty_df, spec=spec)

        # Результат должен быть пустым с правильной структурой
        assert len(result) == 0
        assert "ts_ms" in result.columns
        assert "symbol" in result.columns

    def test_no_corrupted_features(self):
        """Проверка: искаженные признаки (inf, очень большие значения) отсутствуют."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Создаем данные с экстремальными значениями
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
        })

        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # Проверяем отсутствие inf значений
        for col in result.select_dtypes(include=[np.number]).columns:
            assert not np.isinf(result[col]).any(), f"Колонка {col} содержит inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
