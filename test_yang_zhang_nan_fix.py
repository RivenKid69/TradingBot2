"""
Тест для проверки исправления проблемы NaN в Yang-Zhang волатильности.

ПРОБЛЕМА:
Yang-Zhang возвращал NaN в 5-10% случаев когда OHLC данные отсутствовали.

РЕШЕНИЕ:
Hybrid подход: Yang-Zhang если OHLC доступны, иначе fallback к close-to-close volatility.
"""

import math
from transformers import (
    FeatureSpec,
    OnlineFeatureTransformer,
    calculate_yang_zhang_volatility,
    calculate_close_to_close_volatility,
)


def test_close_to_close_volatility():
    """Тест базовой функции close-to-close волатильности."""
    print("\n=== Тест 1: Close-to-Close Volatility ===")

    # Генерируем простые цены с известной волатильностью
    prices = [100.0, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0, 105.0]

    vol = calculate_close_to_close_volatility(prices, len(prices))

    assert vol is not None, "Close-to-close volatility не должна быть None"
    assert vol > 0, "Волатильность должна быть положительной"
    print(f"✓ Close-to-close volatility: {vol:.6f}")


def test_yang_zhang_with_full_ohlc():
    """Тест Yang-Zhang с полными OHLC данными."""
    print("\n=== Тест 2: Yang-Zhang с полными OHLC ===")

    # Создаем полные OHLC бары
    ohlc_bars = []
    for i in range(24):
        base = 100 + i * 0.5
        ohlc_bars.append({
            "open": base,
            "high": base + 1.0,
            "low": base - 0.5,
            "close": base + 0.3,
        })

    vol = calculate_yang_zhang_volatility(ohlc_bars, 24)

    assert vol is not None, "Yang-Zhang с полными OHLC не должен возвращать None"
    assert vol > 0, "Волатильность должна быть положительной"
    print(f"✓ Yang-Zhang volatility (полные OHLC): {vol:.6f}")


def test_yang_zhang_fallback_to_close():
    """Тест fallback к close-to-close когда OHLC недоступны."""
    print("\n=== Тест 3: Yang-Zhang Fallback к Close-to-Close ===")

    # Пустые OHLC бары, но есть close цены
    ohlc_bars = []
    close_prices = [100.0, 102.0, 101.0, 103.0, 102.5, 104.0, 103.0, 105.0] * 3

    vol = calculate_yang_zhang_volatility(ohlc_bars, 24, close_prices=close_prices)

    assert vol is not None, "Fallback к close-to-close не должен возвращать None"
    assert vol > 0, "Волатильность должна быть положительной"
    print(f"✓ Yang-Zhang volatility (fallback): {vol:.6f}")


def test_transformer_with_ohlc():
    """Тест трансформера с полными OHLC данными."""
    print("\n=== Тест 4: Transformer с OHLC данными ===")

    spec = FeatureSpec(
        lookbacks_prices=[240],  # 4h для 4h интервала
        rsi_period=14,
        yang_zhang_windows=[2880],  # 48h
        sma_periods=[],
        garch_windows=[],
        parkinson_windows=[],
        taker_buy_ratio_windows=[],
        cvd_windows=[],
    )

    transformer = OnlineFeatureTransformer(spec)

    # Добавляем 20 баров с полными OHLC
    for i in range(20):
        base = 50000 + i * 100
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=i * 240 * 60 * 1000,  # 4h интервал
            close=base + 50,
            open_price=base,
            high=base + 100,
            low=base - 50,
        )

    # Проверяем последний бар
    feats = transformer.update(
        symbol="BTCUSDT",
        ts_ms=20 * 240 * 60 * 1000,
        close=52100,
        open_price=52000,
        high=52200,
        low=51900,
    )

    assert "yang_zhang_48h" in feats, "Должен быть признак yang_zhang_48h"
    yz = feats["yang_zhang_48h"]

    # С полными OHLC должна быть валидная волатильность
    assert not math.isnan(yz), "Yang-Zhang НЕ должна быть NaN с полными OHLC"
    assert yz > 0, "Волатильность должна быть положительной"
    print(f"✓ yang_zhang_48h с OHLC: {yz:.6f}")


def test_transformer_without_ohlc():
    """
    КРИТИЧЕСКИЙ ТЕСТ: Трансформер БЕЗ OHLC данных.
    Это главный тестовый случай для проверки исправления.
    """
    print("\n=== Тест 5: Transformer БЕЗ OHLC (КРИТИЧЕСКИЙ) ===")

    spec = FeatureSpec(
        lookbacks_prices=[240],  # 4h
        rsi_period=14,
        yang_zhang_windows=[2880],  # 48h
        sma_periods=[],
        garch_windows=[],
        parkinson_windows=[],
        taker_buy_ratio_windows=[],
        cvd_windows=[],
    )

    transformer = OnlineFeatureTransformer(spec)

    # Добавляем 20 баров БЕЗ OHLC (только close)
    for i in range(20):
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=i * 240 * 60 * 1000,
            close=50000 + i * 100,
            # НЕ передаем open_price, high, low
        )

    # Проверяем последний бар
    feats = transformer.update(
        symbol="BTCUSDT",
        ts_ms=20 * 240 * 60 * 1000,
        close=52100,
        # НЕ передаем OHLC
    )

    assert "yang_zhang_48h" in feats, "Должен быть признак yang_zhang_48h"
    yz = feats["yang_zhang_48h"]

    # КРИТИЧЕСКАЯ ПРОВЕРКА: НЕ должна быть NaN благодаря fallback
    assert not math.isnan(yz), "КРИТИЧНО: Yang-Zhang НЕ должна быть NaN благодаря fallback!"
    assert yz > 0, "Волатильность должна быть положительной"
    print(f"✓ yang_zhang_48h БЕЗ OHLC (fallback): {yz:.6f}")
    print("✓ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ РАБОТАЕТ!")


def test_transformer_mixed_ohlc():
    """Тест с частичными OHLC данными (смешанный сценарий)."""
    print("\n=== Тест 6: Transformer со смешанными OHLC ===")

    spec = FeatureSpec(
        lookbacks_prices=[240],
        rsi_period=14,
        yang_zhang_windows=[2880],  # 48h = 12 баров
        sma_periods=[],
        garch_windows=[],
        parkinson_windows=[],
        taker_buy_ratio_windows=[],
        cvd_windows=[],
    )

    transformer = OnlineFeatureTransformer(spec)

    # Сначала 5 баров с OHLC
    for i in range(5):
        base = 50000 + i * 100
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=i * 240 * 60 * 1000,
            close=base + 50,
            open_price=base,
            high=base + 100,
            low=base - 50,
        )

    # Потом 10 баров БЕЗ OHLC
    for i in range(5, 15):
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=i * 240 * 60 * 1000,
            close=50000 + i * 100,
        )

    # Еще 10 баров с OHLC
    for i in range(15, 25):
        base = 50000 + i * 100
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=i * 240 * 60 * 1000,
            close=base + 50,
            open_price=base,
            high=base + 100,
            low=base - 50,
        )

    # Проверяем последний бар
    feats = transformer.update(
        symbol="BTCUSDT",
        ts_ms=25 * 240 * 60 * 1000,
        close=52600,
        open_price=52500,
        high=52700,
        low=52400,
    )

    yz = feats["yang_zhang_48h"]

    # Должна работать благодаря hybrid подходу
    assert not math.isnan(yz), "Yang-Zhang должна работать со смешанными данными"
    assert yz > 0, "Волатильность должна быть положительной"
    print(f"✓ yang_zhang_48h со смешанными OHLC: {yz:.6f}")


def test_multiple_windows():
    """Тест с несколькими окнами волатильности."""
    print("\n=== Тест 7: Несколько окон Yang-Zhang ===")

    spec = FeatureSpec(
        lookbacks_prices=[240],
        rsi_period=14,
        yang_zhang_windows=[2880, 10080, 43200],  # 48h, 7d, 30d
        sma_periods=[],
        garch_windows=[],
        parkinson_windows=[],
        taker_buy_ratio_windows=[],
        cvd_windows=[],
    )

    transformer = OnlineFeatureTransformer(spec)

    # Добавляем много баров БЕЗ OHLC
    for i in range(200):  # 200 баров = ~33 дня для 4h интервала
        transformer.update(
            symbol="BTCUSDT",
            ts_ms=i * 240 * 60 * 1000,
            close=50000 + i * 10 + (i % 5) * 50,  # добавляем волатильность
        )

    feats = transformer.update(
        symbol="BTCUSDT",
        ts_ms=200 * 240 * 60 * 1000,
        close=52000,
    )

    # Проверяем все окна
    for window_name in ["yang_zhang_48h", "yang_zhang_7d", "yang_zhang_30d"]:
        assert window_name in feats, f"Должен быть признак {window_name}"
        vol = feats[window_name]

        # НЕ должно быть NaN благодаря fallback
        assert not math.isnan(vol), f"{window_name} НЕ должна быть NaN!"
        assert vol > 0, f"{window_name} должна быть положительной"
        print(f"✓ {window_name}: {vol:.6f}")


if __name__ == "__main__":
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ ИСПРАВЛЕНИЯ YANG-ZHANG NaN ПРОБЛЕМЫ")
    print("=" * 70)

    try:
        test_close_to_close_volatility()
        test_yang_zhang_with_full_ohlc()
        test_yang_zhang_fallback_to_close()
        test_transformer_with_ohlc()
        test_transformer_without_ohlc()
        test_transformer_mixed_ohlc()
        test_multiple_windows()

        print("\n" + "=" * 70)
        print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
        print("=" * 70)
        print("\n📊 ИТОГИ:")
        print("  • Close-to-close volatility работает корректно")
        print("  • Yang-Zhang работает с полными OHLC")
        print("  • Fallback к close-to-close работает без OHLC")
        print("  • Hybrid подход работает со смешанными данными")
        print("  • Все окна волатильности возвращают валидные значения")
        print("\n🎯 ПРОБЛЕМА NaN В YANG-ZHANG РЕШЕНА!")

    except AssertionError as e:
        print(f"\n❌ ТЕСТ ПРОВАЛЕН: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        raise
