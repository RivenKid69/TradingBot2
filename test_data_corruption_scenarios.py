# test_data_corruption_scenarios.py
"""
Тесты для специфических сценариев искажения данных.

Проверяет edge cases и распространенные паттерны ошибок:
- Look-ahead bias (использование данных из будущего)
- Survivorship bias (отсутствие делистингов)
- Data snooping (утечка информации через preprocessing)
- Feature leakage (признаки содержат информацию о таргете)
- Time alignment issues (несовпадение временных меток)
- Missing data handling (неправильная обработка пропусков)
- Outliers and anomalies (выбросы и аномалии)
"""

import pytest
import pandas as pd
import numpy as np
import math
from typing import List, Dict

from fetch_all_data_patch import load_all_data, _ensure_required_columns
from transformers import FeatureSpec, apply_offline_features, OnlineFeatureTransformer
from asof_join import AsofMerger, AsofSpec
from leakguard import LeakGuard, LeakConfig
from labels import LabelBuilder, LabelConfig


class TestLookAheadBias:
    """Тесты на look-ahead bias - использование данных из будущего."""

    def test_no_future_data_in_features(self):
        """Проверка: признаки на момент t не используют данные после t."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Создаем данные с явным трендом
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 1000 for i in range(10)],  # растущий тренд
        })

        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # Проверяем что SMA на момент i использует только данные до i включительно
        # SMA_240 (1 бар) на позиции 5 должна быть равна price[5]
        idx = 5
        expected_sma = df.iloc[idx]["price"]
        actual_sma = result.iloc[idx]["sma_240"]

        assert abs(actual_sma - expected_sma) < 0.01, \
            f"SMA использует будущие данные: expected={expected_sma}, actual={actual_sma}"

    def test_asof_merge_no_forward_leakage(self):
        """Проверка: asof backward merge не берет данные из будущего."""
        base = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000, 4000],
            "symbol": ["BTC"] * 4,
            "feature": [1.0, 2.0, 3.0, 4.0],
        })

        # Источник данных с временными метками ПОСЛЕ base
        future_source = pd.DataFrame({
            "ts_ms": [1500, 2500, 3500, 4500],  # все позже
            "symbol": ["BTC"] * 4,
            "future_data": [100.0, 200.0, 300.0, 400.0],
        })

        merger = AsofMerger(base_df=base, time_col="ts_ms", keys=["symbol"])
        spec = AsofSpec(
            name="future", df=future_source, time_col="ts_ms",
            keys=["symbol"], direction="backward"
        )
        result = merger.merge([spec])

        # Первая точка не должна получить значение (нет данных backward)
        assert pd.isna(result.iloc[0]["future_future_data"])

        # Вторая точка должна получить значение 100.0 (от 1500ms)
        assert result.iloc[1]["future_future_data"] == 100.0

    def test_labels_use_only_future_prices(self):
        """Проверка: метки используют только цены ПОСЛЕ decision_ts."""
        lb = LabelBuilder(LabelConfig(horizon_ms=1000, price_col="price", returns="log"))

        base = pd.DataFrame({
            "ts_ms": [2000],
            "symbol": ["BTC"],
            "decision_ts": [2000],
        })

        prices = pd.DataFrame({
            "ts_ms": [1000, 2000, 3000],  # есть цены до и после decision_ts
            "symbol": ["BTC"] * 3,
            "price": [28000.0, 29000.0, 30000.0],
        })

        result = lb.build(base, prices, ts_col="ts_ms", symbol_col="symbol")

        # label_price0 должна быть >= decision_ts (2000ms)
        assert result["label_t0_ts"].iloc[0] >= 2000

        # label_price1 должна быть >= decision_ts + horizon (3000ms)
        assert result["label_t1_ts"].iloc[0] >= 3000


class TestTimeAlignment:
    """Тесты корректности временного выравнивания."""

    def test_timestamp_4h_alignment(self):
        """Проверка: все timestamp выравниваются на 4-часовую границу."""
        df = pd.DataFrame({
            "timestamp": [
                1609459200,  # точно на границе 4h
                1609459210,  # +10 секунд
                1609459200 + 3600,  # +1 час
                1609459200 + 7200,  # +2 часа
                1609459200 + 14400,  # +4 часа (следующая граница)
            ],
            "symbol": ["BTCUSDT"] * 5,
            "open": [29000.0] * 5,
            "high": [29500.0] * 5,
            "low": [28900.0] * 5,
            "close": [29100.0] * 5,
            "volume": [100.0] * 5,
            "quote_asset_volume": [2910000.0] * 5,
            "number_of_trades": [1000] * 5,
            "taker_buy_base_asset_volume": [50.0] * 5,
            "taker_buy_quote_asset_volume": [1455000.0] * 5,
        })

        result = _ensure_required_columns(df)

        # Проверяем что все timestamp кратны 14400 (4 часа = 14400 секунд)
        for ts in result["timestamp"]:
            assert ts % 14400 == 0, f"Timestamp {ts} не выровнен на 4h границу"

        # После выравнивания и дедупликации должно остаться только 2 уникальных timestamp
        # (1609459200 и 1609459200+14400)
        assert len(result) == 2

    def test_features_bar_duration_consistency(self):
        """Проверка: окна признаков корректно конвертируются с учетом bar_duration."""
        # Для 4h баров: 240 минут = 1 бар
        spec = FeatureSpec(
            lookbacks_prices=[240, 480, 1440],  # 4h, 8h, 24h в минутах
            bar_duration_minutes=240
        )

        # После __post_init__ окна должны быть в барах
        assert spec.lookbacks_prices == [1, 2, 6]  # 1 бар, 2 бара, 6 баров

        # Исходные минуты сохранены для именования
        assert spec._lookbacks_prices_minutes == [240, 480, 1440]


class TestMissingDataHandling:
    """Тесты обработки пропущенных данных."""

    def test_nan_propagation_in_features(self):
        """Проверка: NaN в исходных данных корректно обрабатывается в признаках."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 if i % 3 != 0 else np.nan for i in range(10)],  # каждый 3-й NaN
        })

        # dropna() в apply_offline_features должен удалить строки с NaN
        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # Строки с NaN price должны быть удалены
        assert len(result) < len(df)
        assert result["ref_price"].notna().all()

    def test_ffill_gap_validation(self):
        """Проверка: чрезмерный forward fill обнаруживается и блокируется."""
        lg = LeakGuard(LeakConfig())

        df = pd.DataFrame({
            "ts_ms": [1000, 2000, 10000],  # большой gap между 2000 и 10000
            "symbol": ["BTC", "BTC", "BTC"],
            "value": [100.0, 100.0, 100.0],  # одинаковое значение (симуляция ffill)
        })

        # С max_gap=5000ms, последнее значение должно стать NaN
        result = lg.validate_ffill_gaps(
            df, ts_col="ts_ms", group_keys=["symbol"],
            value_cols=["value"], max_gap_ms=5000
        )

        assert pd.isna(result["value"].iloc[2]), "Чрезмерный ffill не был заблокирован"

    def test_insufficient_warmup_handling(self):
        """Проверка: недостаточные данные для warm-up корректно помечаются NaN."""
        spec = FeatureSpec(lookbacks_prices=[14400], bar_duration_minutes=240)  # 60 баров

        # Только 10 баров - недостаточно для окна 60 баров
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
        })

        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # SMA_14400 (60 баров) должна быть NaN для всех строк
        assert result["sma_14400"].isna().all(), "Недостаточные данные не помечены как NaN"


class TestOutliersAndAnomalies:
    """Тесты обработки выбросов и аномалий."""

    def test_extreme_price_changes_validation(self):
        """Проверка: экстремальные изменения цены не приводят к inf в признаках."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0, 29100.0, 100000.0, 29200.0, 29300.0, 29400.0, 29500.0, 29600.0, 29700.0, 29800.0],
            # скачок на позиции 2
        })

        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # Проверяем отсутствие inf
        assert not np.isinf(result["ret_4h"]).any(), "Экстремальные изменения привели к inf"
        assert not np.isinf(result["sma_240"]).any(), "Экстремальные изменения привели к inf в SMA"

    def test_zero_volume_handling(self):
        """Проверка: нулевые объемы не приводят к division by zero."""
        spec = FeatureSpec(
            lookbacks_prices=[240],
            taker_buy_ratio_windows=[480],
            bar_duration_minutes=240
        )

        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
            "volume": [0.0] * 10,  # нулевой объем
            "taker_buy_base": [0.0] * 10,
        })

        # Не должно вызвать исключение
        result = apply_offline_features(
            df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price",
            volume_col="volume", taker_buy_base_col="taker_buy_base"
        )

        # taker_buy_ratio должен быть либо валидным, либо NaN (но не inf, не ошибка)
        assert not np.isinf(result["taker_buy_ratio"]).any()

    def test_negative_values_rejection(self):
        """Проверка: отрицательные значения в OHLCV отклоняются."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [-29000.0] + [29000.0 + i * 100 for i in range(1, 10)],  # отрицательная первая цена
        })

        # dropna() удалит невалидные строки после валидации
        result = apply_offline_features(df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # Все цены должны быть положительными
        assert (result["ref_price"] > 0).all()


class TestFeatureLeakage:
    """Тесты на feature leakage - утечку информации о таргете в признаки."""

    def test_features_independent_of_future_labels(self):
        """Проверка: признаки на момент t не зависят от меток в будущем."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Создаем два датасета с разными future ценами
        df1 = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
        })

        df2 = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 if i < 5 else 50000.0 + i * 100 for i in range(10)],
            # разная цена после момента 5
        })

        result1 = apply_offline_features(df1, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")
        result2 = apply_offline_features(df2, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price")

        # Признаки до момента 5 должны быть идентичны (будущее не влияет)
        for i in range(5):
            assert abs(result1.iloc[i]["sma_240"] - result2.iloc[i]["sma_240"]) < 0.01, \
                f"Признаки на позиции {i} зависят от будущих данных"

    def test_online_offline_feature_consistency(self):
        """Проверка: онлайн и оффлайн признаки идентичны (нет утечки через batch processing)."""
        spec = FeatureSpec(lookbacks_prices=[240], bar_duration_minutes=240)

        # Создаем тестовые данные
        df = pd.DataFrame({
            "ts_ms": [i * 240 * 60 * 1000 for i in range(10)],
            "symbol": ["BTCUSDT"] * 10,
            "price": [29000.0 + i * 100 for i in range(10)],
        })

        # Оффлайн расчет
        offline_result = apply_offline_features(
            df, spec=spec, ts_col="ts_ms", symbol_col="symbol", price_col="price"
        )

        # Онлайн расчет (последовательное обновление)
        transformer = OnlineFeatureTransformer(spec)
        online_results = []
        for _, row in df.iterrows():
            features = transformer.update(
                symbol="BTCUSDT",
                ts_ms=int(row["ts_ms"]),
                close=float(row["price"])
            )
            online_results.append(features)

        online_result = pd.DataFrame(online_results)

        # Сравниваем результаты (на последней позиции, где достаточно данных)
        idx = 5
        offline_sma = offline_result.iloc[idx]["sma_240"]
        online_sma = online_result.iloc[idx]["sma_240"]

        assert abs(offline_sma - online_sma) < 0.01, \
            f"Онлайн и оффлайн признаки не совпадают: offline={offline_sma}, online={online_sma}"


class TestDataIntegrity:
    """Тесты целостности данных."""

    def test_no_duplicate_timestamps_after_processing(self):
        """Проверка: после обработки нет дублирующихся timestamp."""
        df = pd.DataFrame({
            "timestamp": [1609459200, 1609459200, 1609462800, 1609462800, 1609466400],
            "symbol": ["BTCUSDT"] * 5,
            "open": [29000.0, 29050.0, 29100.0, 29150.0, 29200.0],
            "high": [29500.0] * 5,
            "low": [28900.0] * 5,
            "close": [29100.0] * 5,
            "volume": [100.0] * 5,
            "quote_asset_volume": [2910000.0] * 5,
            "number_of_trades": [1000] * 5,
            "taker_buy_base_asset_volume": [50.0] * 5,
            "taker_buy_quote_asset_volume": [1455000.0] * 5,
        })

        result = _ensure_required_columns(df)

        # Не должно быть дублирующихся timestamp
        assert result["timestamp"].duplicated().sum() == 0

    def test_monotonic_increasing_timestamps(self):
        """Проверка: timestamp монотонно возрастают после сортировки."""
        df = pd.DataFrame({
            "timestamp": [1609466400, 1609459200, 1609462800],  # не отсортированы
            "symbol": ["BTCUSDT"] * 3,
            "open": [29000.0, 29100.0, 29200.0],
            "high": [29500.0] * 3,
            "low": [28900.0] * 3,
            "close": [29100.0] * 3,
            "volume": [100.0] * 3,
            "quote_asset_volume": [2910000.0] * 3,
            "number_of_trades": [1000] * 3,
            "taker_buy_base_asset_volume": [50.0] * 3,
            "taker_buy_quote_asset_volume": [1455000.0] * 3,
        })

        result = _ensure_required_columns(df)

        # Проверяем монотонность
        assert result["timestamp"].is_monotonic_increasing

    def test_consistent_column_order(self):
        """Проверка: порядок колонок стабилен после обработки."""
        df = pd.DataFrame({
            "volume": [100.0] * 3,
            "timestamp": [1609459200, 1609462800, 1609466400],
            "close": [29100.0] * 3,
            "symbol": ["BTCUSDT"] * 3,
            "open": [29000.0, 29100.0, 29200.0],
            "high": [29500.0] * 3,
            "low": [28900.0] * 3,
            "quote_asset_volume": [2910000.0] * 3,
            "number_of_trades": [1000] * 3,
            "taker_buy_base_asset_volume": [50.0] * 3,
            "taker_buy_quote_asset_volume": [1455000.0] * 3,
        })

        result = _ensure_required_columns(df)

        # Проверяем ожидаемый порядок колонок (базовый префикс)
        expected_prefix = [
            "timestamp", "symbol", "open", "high", "low", "close", "volume",
            "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
        ]

        actual_prefix = list(result.columns[:len(expected_prefix)])
        assert actual_prefix == expected_prefix, \
            f"Нарушен порядок колонок: expected={expected_prefix}, actual={actual_prefix}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
