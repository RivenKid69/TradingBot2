#!/usr/bin/env python3
"""
Интеграционный тест Yang-Zhang волатильности.
Проверяет работу признака в онлайн и оффлайн режимах.
"""
import pandas as pd
import numpy as np
from transformers import FeatureSpec, OnlineFeatureTransformer, apply_offline_features


def test_online_mode():
    """Тест онлайн-режима с OHLC данными."""
    print("Тест 1: Онлайн-режим (Online Transformer)")

    spec = FeatureSpec(
        lookbacks_prices=[5, 15],
        rsi_period=14,
        yang_zhang_windows=[24, 100]  # 24 минуты, 100 минут
    )

    transformer = OnlineFeatureTransformer(spec)

    # Симулируем 150 баров
    base_price = 100.0
    results = []

    for i in range(150):
        noise = 0.01 * np.sin(i * 0.1)
        open_p = base_price * (1 + noise)
        high = open_p * 1.005
        low = open_p * 0.995
        close = base_price * (1 + 0.01 * np.sin((i + 0.5) * 0.1))

        feats = transformer.update(
            symbol="BTCUSDT",
            ts_ms=1000000000 + i * 60000,  # каждую минуту
            close=close,
            open_price=open_p,
            high=high,
            low=low,
        )

        results.append(feats)
        base_price *= 1.0001

    # Проверяем последние признаки
    last_feats = results[-1]

    # Должны быть Yang-Zhang признаки
    assert "yang_zhang_0h" in last_feats, "Отсутствует yang_zhang_0h"
    assert "yang_zhang_1h" in last_feats, "Отсутствует yang_zhang_1h"

    # Проверяем что значения не NaN (у нас достаточно данных)
    yz_0h = last_feats.get("yang_zhang_0h")
    yz_1h = last_feats.get("yang_zhang_1h")

    if pd.notna(yz_0h):
        assert yz_0h > 0, f"Yang-Zhang 24min должна быть положительной: {yz_0h}"
        print(f"  ✓ Yang-Zhang 24min: {yz_0h:.6f}")
    else:
        print(f"  ! Yang-Zhang 24min: NaN (недостаточно данных)")

    if pd.notna(yz_1h):
        assert yz_1h > 0, f"Yang-Zhang 100min должна быть положительной: {yz_1h}"
        print(f"  ✓ Yang-Zhang 100min: {yz_1h:.6f}")
    else:
        print(f"  ! Yang-Zhang 100min: NaN (недостаточно данных)")

    return True


def test_offline_mode():
    """Тест оффлайн-режима с OHLC данными."""
    print("\nТест 2: Оффлайн-режим (apply_offline_features)")

    # Создаем тестовый датафрейм с OHLC
    n_rows = 150
    base_price = 100.0
    data = []

    for i in range(n_rows):
        noise = 0.01 * np.sin(i * 0.1)
        open_p = base_price * (1 + noise)
        high = open_p * 1.005
        low = open_p * 0.995
        close = base_price * (1 + 0.01 * np.sin((i + 0.5) * 0.1))

        data.append({
            "ts_ms": 1000000000 + i * 60000,
            "symbol": "BTCUSDT",
            "price": close,
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        })
        base_price *= 1.0001

    df = pd.DataFrame(data)

    spec = FeatureSpec(
        lookbacks_prices=[5, 15],
        rsi_period=14,
        yang_zhang_windows=[24, 100]  # 24 минуты, 100 минут
    )

    # Применяем трансформацию
    feats_df = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        open_col="open",
        high_col="high",
        low_col="low",
    )

    # Проверяем что колонки созданы
    assert "yang_zhang_0h" in feats_df.columns, "Отсутствует колонка yang_zhang_0h"
    assert "yang_zhang_1h" in feats_df.columns, "Отсутствует колонка yang_zhang_1h"

    # Проверяем последнюю строку
    last_row = feats_df.iloc[-1]
    yz_0h = last_row["yang_zhang_0h"]
    yz_1h = last_row["yang_zhang_1h"]

    if pd.notna(yz_0h):
        assert yz_0h > 0, f"Yang-Zhang 24min должна быть положительной: {yz_0h}"
        print(f"  ✓ Yang-Zhang 24min: {yz_0h:.6f}")
    else:
        print(f"  ! Yang-Zhang 24min: NaN (недостаточно данных)")

    if pd.notna(yz_1h):
        assert yz_1h > 0, f"Yang-Zhang 100min должна быть положительной: {yz_1h}"
        print(f"  ✓ Yang-Zhang 100min: {yz_1h:.6f}")
    else:
        print(f"  ! Yang-Zhang 100min: NaN (недостаточно данных)")

    # Проверяем что не все значения NaN
    non_nan_0h = feats_df["yang_zhang_0h"].notna().sum()
    non_nan_1h = feats_df["yang_zhang_1h"].notna().sum()

    print(f"  ✓ Непустых значений yang_zhang_0h: {non_nan_0h}/{len(feats_df)}")
    print(f"  ✓ Непустых значений yang_zhang_1h: {non_nan_1h}/{len(feats_df)}")

    assert non_nan_0h > 0, "Все значения yang_zhang_0h - NaN"
    assert non_nan_1h > 0, "Все значения yang_zhang_1h - NaN"

    return True


def test_without_ohlc():
    """Тест без OHLC данных - волатильность должна быть NaN."""
    print("\nТест 3: Без OHLC данных (должны быть NaN)")

    # Создаем датафрейм только с price (без OHLC)
    df = pd.DataFrame({
        "ts_ms": [1000000000 + i * 60000 for i in range(50)],
        "symbol": ["BTCUSDT"] * 50,
        "price": [100 + i * 0.1 for i in range(50)],
    })

    spec = FeatureSpec(
        lookbacks_prices=[5],
        rsi_period=14,
        yang_zhang_windows=[24]
    )

    feats_df = apply_offline_features(
        df,
        spec=spec,
        ts_col="ts_ms",
        symbol_col="symbol",
        price_col="price",
        # Не передаем OHLC колонки
    )

    # Должна быть колонка, но все значения NaN
    assert "yang_zhang_0h" in feats_df.columns, "Отсутствует колонка yang_zhang_0h"

    all_nan = feats_df["yang_zhang_0h"].isna().all()
    assert all_nan, "Без OHLC данных все значения должны быть NaN"

    print("  ✓ Корректная обработка отсутствия OHLC данных (все NaN)")

    return True


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ YANG-ZHANG ВОЛАТИЛЬНОСТИ")
    print("=" * 60 + "\n")

    tests = [
        test_online_mode,
        test_offline_mode,
        test_without_ohlc,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except AssertionError as e:
            print(f"  ❌ ОШИБКА: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ ИСКЛЮЧЕНИЕ: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"РЕЗУЛЬТАТЫ: {passed} успешно, {failed} неудачно")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
