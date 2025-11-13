#!/usr/bin/env python3
"""Тест для Cumulative Volume Delta (CVD) feature."""

from __future__ import annotations

import pandas as pd
from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features


def test_cvd_online():
    """Тест онлайн вычисления CVD."""
    print("Тест 1: Онлайн вычисление CVD")

    spec = FeatureSpec(
        lookbacks_prices=[240, 720],  # 4h, 12h для 4h таймфрейма
        rsi_period=14,
        cvd_windows=[5, 10],  # Малые окна для теста (в минутах)
        bar_duration_minutes=1,  # Используем минутные данные для теста
    )

    transformer = OnlineFeatureTransformer(spec)

    # Создаем тестовые данные
    # CVD = Σ(buy_volume - sell_volume)
    # где buy_volume = taker_buy_base, sell_volume = volume - taker_buy_base
    test_data = [
        # ts, close, volume, taker_buy_base, expected_delta
        (1000, 100.0, 1000.0, 600.0),  # delta = 600 - 400 = 200
        (2000, 101.0, 1000.0, 700.0),  # delta = 700 - 300 = 400
        (3000, 102.0, 1000.0, 500.0),  # delta = 500 - 500 = 0
        (4000, 103.0, 1000.0, 800.0),  # delta = 800 - 200 = 600
        (5000, 104.0, 1000.0, 550.0),  # delta = 550 - 450 = 100
        (6000, 105.0, 1000.0, 900.0),  # delta = 900 - 100 = 800
        (7000, 106.0, 1000.0, 400.0),  # delta = 400 - 600 = -200
        (8000, 107.0, 1000.0, 300.0),  # delta = 300 - 700 = -400
        (9000, 108.0, 1000.0, 650.0),  # delta = 650 - 350 = 300
        (10000, 109.0, 1000.0, 750.0),  # delta = 750 - 250 = 500
    ]

    # Ожидаемые CVD для окна 5:
    # После 5 баров: sum([200, 400, 0, 600, 100]) = 1300
    # После 6 баров: sum([400, 0, 600, 100, 800]) = 1900
    # После 10 баров: sum([100, 800, -200, -400, 300, 500]) = 1100

    results = []
    for ts, close, volume, taker_buy_base in test_data:
        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=ts,
            close=close,
            volume=volume,
            taker_buy_base=taker_buy_base,
        )
        results.append(feats)

    # Проверяем последний результат (должно быть 10 баров)
    last_feats = results[-1]

    print(f"CVD 5 периодов (окно 5m): {last_feats.get('cvd_5m', 'N/A')}")
    print(f"CVD 10 периодов (окно 10m): {last_feats.get('cvd_10m', 'N/A')}")

    # Проверяем наличие CVD признаков
    assert 'cvd_5m' in last_feats, f"CVD признак cvd_5m не найден. Доступные: {list(last_feats.keys())}"

    # Вычисляем ожидаемое значение для окна 5 (последние 5 баров)
    # Бары 6-10: deltas = [800, -200, -400, 300, 500]
    expected_cvd_5 = 800 + (-200) + (-400) + 300 + 500
    actual_cvd_5 = last_feats['cvd_5m']

    print(f"Ожидаемый CVD (5 баров): {expected_cvd_5}")
    print(f"Фактический CVD (5 баров): {actual_cvd_5}")

    assert abs(actual_cvd_5 - expected_cvd_5) < 0.01, f"CVD не совпадает: ожидалось {expected_cvd_5}, получено {actual_cvd_5}"

    print("✓ Тест 1 пройден\n")


def test_cvd_offline():
    """Тест оффлайн вычисления CVD."""
    print("Тест 2: Оффлайн вычисление CVD")

    # Создаем тестовый DataFrame
    data = {
        "ts_ms": [i * 60000 for i in range(1440 * 2)],  # 2 дня минутных данных
        "symbol": ["BTCUSDT"] * (1440 * 2),
        "price": [100.0 + i * 0.01 for i in range(1440 * 2)],
        "volume": [1000.0] * (1440 * 2),
        "taker_buy_base_asset_volume": [600.0 if i % 2 == 0 else 400.0 for i in range(1440 * 2)],
    }
    df = pd.DataFrame(data)

    spec = FeatureSpec(
        lookbacks_prices=[240, 720],  # 4h, 12h для 4h таймфрейма
        rsi_period=14,
        cvd_windows=[1440, 10080],  # 24ч, 7д (в минутах)
        bar_duration_minutes=1,  # Минутные данные в тесте
    )

    result = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        volume_col="volume",
        taker_buy_base_col="taker_buy_base_asset_volume",
    )

    print(f"Результат содержит {len(result)} строк")
    print(f"Колонки: {result.columns.tolist()}")

    # Проверяем наличие CVD признаков
    assert "cvd_24h" in result.columns, "cvd_24h не найден"
    assert "cvd_7d" in result.columns, "cvd_7d не найден"  # было cvd_168h, но _format_window_name(10080) = "7d"

    # Проверяем последнюю строку (должна иметь данные для 24ч окна)
    last_row = result.iloc[-1]
    cvd_24h = last_row["cvd_24h"]

    print(f"CVD 24h в последней строке: {cvd_24h}")

    # CVD должен быть не NaN после 1440 баров
    assert not pd.isna(cvd_24h), "CVD 24h не должен быть NaN в конце"

    # Проверяем что CVD 7d недоступен (только 2 дня данных)
    cvd_7d = last_row["cvd_7d"]  # было cvd_168h
    print(f"CVD 7d в последней строке: {cvd_7d}")
    assert pd.isna(cvd_7d), "CVD 7d должен быть NaN (недостаточно данных)"

    print("✓ Тест 2 пройден\n")


def test_cvd_edge_cases():
    """Тест граничных случаев для CVD."""
    print("Тест 3: Граничные случаи CVD")

    spec = FeatureSpec(
        lookbacks_prices=[240],  # 4h для 4h таймфрейма
        rsi_period=14,
        cvd_windows=[5],  # Малое окно для теста (5 минут)
        bar_duration_minutes=1,  # Используем минутные данные для теста
    )

    transformer = OnlineFeatureTransformer(spec)

    # Случай 1: Чистая покупка (100% taker buy)
    feats1 = transformer.update(
        symbol="BTCUSDT",
        ts_ms=1000,
        close=100.0,
        volume=1000.0,
        taker_buy_base=1000.0,
    )
    # delta = 1000 - 0 = 1000

    # Случай 2: Чистая продажа (0% taker buy)
    feats2 = transformer.update(
        symbol="BTCUSDT",
        ts_ms=2000,
        close=101.0,
        volume=1000.0,
        taker_buy_base=0.0,
    )
    # delta = 0 - 1000 = -1000

    # Случай 3: Сбалансированный (50/50)
    feats3 = transformer.update(
        symbol="BTCUSDT",
        ts_ms=3000,
        close=102.0,
        volume=1000.0,
        taker_buy_base=500.0,
    )
    # delta = 500 - 500 = 0

    # После 3 баров CVD должен быть недоступен (окно 5)
    assert pd.isna(feats3.get('cvd_5m', float('nan'))), "CVD должен быть NaN (недостаточно данных)"

    # Добавляем еще 2 бара
    for i in range(2):
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=4000 + i * 1000,
            close=103.0 + i,
            volume=1000.0,
            taker_buy_base=500.0,
        )

    # После 5 баров CVD должен быть доступен
    feats_final = transformer.update(
        symbol="BTCUSDT",
        ts_ms=6000,
        close=105.0,
        volume=1000.0,
        taker_buy_base=700.0,
    )

    cvd = feats_final.get('cvd_5m', float('nan'))
    assert not pd.isna(cvd), "CVD не должен быть NaN после 5 баров"

    # Последние 5 дельт: [-1000, 0, 0, 0, 400]
    expected = -1000 + 0 + 0 + 0 + 400
    print(f"Ожидаемый CVD: {expected}, Фактический: {cvd}")
    assert abs(cvd - expected) < 0.01, f"CVD не совпадает: ожидалось {expected}, получено {cvd}"

    print("✓ Тест 3 пройден\n")


def main():
    """Запуск всех тестов."""
    print("=" * 60)
    print("Тестирование Cumulative Volume Delta (CVD)")
    print("=" * 60 + "\n")

    test_cvd_online()
    test_cvd_offline()
    test_cvd_edge_cases()

    print("=" * 60)
    print("Все тесты пройдены успешно!")
    print("=" * 60)


if __name__ == "__main__":
    main()
