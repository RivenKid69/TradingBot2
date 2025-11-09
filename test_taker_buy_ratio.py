#!/usr/bin/env python3
"""Тест для проверки интеграции Taker Buy Ratio признака."""

import pandas as pd
import numpy as np
from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features


def test_taker_buy_ratio_online():
    """Тест онлайн вычисления taker_buy_ratio."""
    print("Тест 1: Онлайн вычисление taker_buy_ratio")

    spec = FeatureSpec(
        lookbacks_prices=[5, 10],
        rsi_period=14,
        yang_zhang_windows=[1440],
        taker_buy_ratio_windows=[360, 720],  # 6ч, 12ч
        taker_buy_ratio_momentum=[60, 360],  # 1ч, 6ч
    )

    transformer = OnlineFeatureTransformer(spec)

    # Создаем тестовые данные
    base_ts = 1700000000000
    symbol = "BTCUSDT"

    # Первые 800 баров (чуть больше 12 часов) для накопления истории
    for i in range(800):
        ts_ms = base_ts + i * 60000  # каждая минута
        close = 50000.0 + i * 10  # растущая цена
        open_price = close - 5
        high = close + 10
        low = close - 10
        volume = 100.0 + i * 0.1
        taker_buy_base = volume * 0.6  # 60% покупки

        feats = transformer.update(
            symbol=symbol,
            ts_ms=ts_ms,
            close=close,
            open_price=open_price,
            high=high,
            low=low,
            volume=volume,
            taker_buy_base=taker_buy_base,
        )

    # Проверяем последний результат
    assert "taker_buy_ratio" in feats, "taker_buy_ratio не найден в признаках"
    assert "taker_buy_ratio_sma_6h" in feats, "taker_buy_ratio_sma_6h не найден"
    assert "taker_buy_ratio_sma_12h" in feats, "taker_buy_ratio_sma_12h не найден"
    assert "taker_buy_ratio_momentum_1h" in feats, "taker_buy_ratio_momentum_1h не найден"
    assert "taker_buy_ratio_momentum_6h" in feats, "taker_buy_ratio_momentum_6h не найден"

    # Проверяем, что значения находятся в разумных пределах
    assert 0 <= feats["taker_buy_ratio"] <= 1, f"taker_buy_ratio должен быть в диапазоне [0, 1], получено: {feats['taker_buy_ratio']}"

    # Проверяем, что taker_buy_ratio примерно равен 0.6 (60% покупки)
    assert abs(feats["taker_buy_ratio"] - 0.6) < 0.01, f"taker_buy_ratio должен быть ~0.6, получено: {feats['taker_buy_ratio']}"

    print(f"  ✓ taker_buy_ratio: {feats['taker_buy_ratio']:.4f}")
    print(f"  ✓ taker_buy_ratio_sma_6h: {feats['taker_buy_ratio_sma_6h']:.4f}")
    print(f"  ✓ taker_buy_ratio_sma_12h: {feats['taker_buy_ratio_sma_12h']:.4f}")
    print(f"  ✓ taker_buy_ratio_momentum_1h: {feats['taker_buy_ratio_momentum_1h']:.6f}")
    print(f"  ✓ taker_buy_ratio_momentum_6h: {feats['taker_buy_ratio_momentum_6h']:.6f}")
    print("  ✓ Тест пройден!\n")


def test_taker_buy_ratio_offline():
    """Тест оффлайн вычисления taker_buy_ratio."""
    print("Тест 2: Оффлайн вычисление taker_buy_ratio")

    # Создаем тестовый DataFrame
    n_rows = 1500
    base_ts = 1700000000000

    data = {
        "ts_ms": [base_ts + i * 60000 for i in range(n_rows)],
        "symbol": ["BTCUSDT"] * n_rows,
        "price": [50000.0 + i * 10 for i in range(n_rows)],
        "open": [50000.0 + i * 10 - 5 for i in range(n_rows)],
        "high": [50000.0 + i * 10 + 10 for i in range(n_rows)],
        "low": [50000.0 + i * 10 - 10 for i in range(n_rows)],
        "volume": [100.0 + i * 0.1 for i in range(n_rows)],
        "taker_buy_base": [(100.0 + i * 0.1) * 0.55 for i in range(n_rows)],  # 55% покупки
    }

    df = pd.DataFrame(data)

    spec = FeatureSpec(
        lookbacks_prices=[5, 10],
        rsi_period=14,
        yang_zhang_windows=[1440],
        taker_buy_ratio_windows=[360, 720, 1440],  # 6ч, 12ч, 24ч
        taker_buy_ratio_momentum=[60, 360, 720],  # 1ч, 6ч, 12ч
    )

    result = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        open_col="open",
        high_col="high",
        low_col="low",
        volume_col="volume",
        taker_buy_base_col="taker_buy_base",
    )

    # Проверяем, что все колонки созданы
    assert "taker_buy_ratio" in result.columns, "taker_buy_ratio не найден"
    assert "taker_buy_ratio_sma_6h" in result.columns, "taker_buy_ratio_sma_6h не найден"
    assert "taker_buy_ratio_sma_12h" in result.columns, "taker_buy_ratio_sma_12h не найден"
    assert "taker_buy_ratio_sma_24h" in result.columns, "taker_buy_ratio_sma_24h не найден"
    assert "taker_buy_ratio_momentum_1h" in result.columns, "taker_buy_ratio_momentum_1h не найден"
    assert "taker_buy_ratio_momentum_6h" in result.columns, "taker_buy_ratio_momentum_6h не найден"
    assert "taker_buy_ratio_momentum_12h" in result.columns, "taker_buy_ratio_momentum_12h не найден"

    # Проверяем последние значения (где уже накоплена история)
    last_row = result.iloc[-1]

    # Проверяем диапазон значений
    assert 0 <= last_row["taker_buy_ratio"] <= 1, f"taker_buy_ratio вне диапазона: {last_row['taker_buy_ratio']}"

    # Проверяем, что ratio примерно равен 0.55
    assert abs(last_row["taker_buy_ratio"] - 0.55) < 0.01, f"taker_buy_ratio должен быть ~0.55, получено: {last_row['taker_buy_ratio']}"

    # Проверяем, что скользящие средние не NaN там, где должны быть значения
    row_1440 = result.iloc[1440]  # Строка, где должны быть все значения
    assert not pd.isna(row_1440["taker_buy_ratio_sma_6h"]), "taker_buy_ratio_sma_6h не должен быть NaN"
    assert not pd.isna(row_1440["taker_buy_ratio_sma_12h"]), "taker_buy_ratio_sma_12h не должен быть NaN"
    assert not pd.isna(row_1440["taker_buy_ratio_sma_24h"]), "taker_buy_ratio_sma_24h не должен быть NaN"

    print(f"  ✓ Обработано {len(result)} строк")
    print(f"  ✓ taker_buy_ratio (последняя): {last_row['taker_buy_ratio']:.4f}")
    print(f"  ✓ taker_buy_ratio_sma_24h (строка 1440): {row_1440['taker_buy_ratio_sma_24h']:.4f}")
    print(f"  ✓ Не-NaN значений в taker_buy_ratio: {result['taker_buy_ratio'].notna().sum()}")
    print("  ✓ Тест пройден!\n")


def test_taker_buy_ratio_edge_cases():
    """Тест граничных случаев."""
    print("Тест 3: Граничные случаи")

    spec = FeatureSpec(
        lookbacks_prices=[5],
        rsi_period=14,
        taker_buy_ratio_windows=[360],
        taker_buy_ratio_momentum=[60],
    )

    transformer = OnlineFeatureTransformer(spec)

    # Случай 1: volume = 0 (должен быть пропущен)
    feats1 = transformer.update(
        symbol="BTCUSDT",
        ts_ms=1700000000000,
        close=50000.0,
        volume=0.0,
        taker_buy_base=10.0,
    )
    # taker_buy_ratio не должен быть добавлен, так как volume = 0

    # Случай 2: taker_buy_base = 0 (корректный случай, ratio = 0)
    feats2 = transformer.update(
        symbol="BTCUSDT",
        ts_ms=1700000060000,
        close=50000.0,
        volume=100.0,
        taker_buy_base=0.0,
    )
    assert "taker_buy_ratio" in feats2
    assert feats2["taker_buy_ratio"] == 0.0, "taker_buy_ratio должен быть 0 когда taker_buy_base = 0"

    # Случай 3: taker_buy_base = volume (100% покупки)
    feats3 = transformer.update(
        symbol="BTCUSDT",
        ts_ms=1700000120000,
        close=50000.0,
        volume=100.0,
        taker_buy_base=100.0,
    )
    assert feats3["taker_buy_ratio"] == 1.0, "taker_buy_ratio должен быть 1.0 когда taker_buy_base = volume"

    print("  ✓ Граничные случаи обработаны корректно")
    print("  ✓ Тест пройден!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Тестирование Taker Buy Ratio Feature")
    print("=" * 60 + "\n")

    try:
        test_taker_buy_ratio_online()
        test_taker_buy_ratio_offline()
        test_taker_buy_ratio_edge_cases()

        print("=" * 60)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ ОШИБКА ТЕСТА: {e}")
        raise
    except Exception as e:
        print(f"\n❌ НЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
        raise
