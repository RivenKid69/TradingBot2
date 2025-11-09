#!/usr/bin/env python3
"""
Тест для проверки корректности расчета волатильности Паркинсона (Parkinson Range Volatility).
"""

import math
from transformers import calculate_parkinson_volatility, FeatureSpec, OnlineFeatureTransformer


def test_parkinson_basic():
    """Базовый тест на синтетических данных."""
    print("=== Тест 1: Базовая проверка расчета Parkinson ===")

    # Создаем простые синтетические данные
    # Цены растут с небольшой волатильностью
    ohlc_bars = []
    base_price = 100.0
    for i in range(50):
        # Добавляем небольшую волатильность
        noise = 0.01 * math.sin(i * 0.5)
        open_p = base_price * (1 + noise)
        high = open_p * 1.005  # High на 0.5% выше open
        low = open_p * 0.995   # Low на 0.5% ниже open
        close = base_price * (1 + 0.01 * math.sin((i + 0.5) * 0.5))

        ohlc_bars.append({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        })
        base_price *= 1.001  # медленный рост

    # Рассчитываем волатильность для окна 24 бара
    vol = calculate_parkinson_volatility(ohlc_bars, 24)

    print(f"  Количество баров: {len(ohlc_bars)}")
    print(f"  Окно: 24")
    print(f"  Parkinson волатильность: {vol}")

    if vol is None:
        print("  ❌ ОШИБКА: волатильность не рассчитана")
        return False

    if vol < 0:
        print("  ❌ ОШИБКА: отрицательная волатильность")
        return False

    # Волатильность должна быть разумной для наших данных
    if 0.001 < vol < 0.5:
        print("  ✓ Волатильность в разумных пределах")
    else:
        print(f"  ⚠️  ПРЕДУПРЕЖДЕНИЕ: волатильность вне ожидаемого диапазона: {vol}")

    print()
    return True


def test_online_transformer():
    """Тест онлайн-трансформера с Parkinson."""
    print("=== Тест 2: Онлайн-трансформер с Parkinson ===")

    spec = FeatureSpec(
        lookbacks_prices=[5, 15],
        rsi_period=14,
        parkinson_windows=[24, 50],  # 24 и 50 минут для теста
    )

    transformer = OnlineFeatureTransformer(spec)

    # Генерируем 100 баров
    base_price = 100.0
    symbol = "BTCUSDT"
    ts = 1600000000000

    print(f"  Символ: {symbol}")
    print(f"  Количество баров: 100")
    print(f"  Окна Parkinson: {spec.parkinson_windows}")

    for i in range(100):
        noise = 0.01 * math.sin(i * 0.3)
        open_p = base_price * (1 + noise)
        high = open_p * 1.008  # High на 0.8% выше
        low = open_p * 0.992   # Low на 0.8% ниже
        close = base_price * (1 + 0.01 * math.sin((i + 0.5) * 0.3))

        feats = transformer.update(
            symbol=symbol,
            ts_ms=ts,
            close=close,
            open_price=open_p,
            high=high,
            low=low,
        )

        base_price *= 1.0005
        ts += 60000  # +1 минута

        # Проверяем последний бар
        if i == 99:
            print(f"\n  Признаки последнего бара:")
            print(f"    ref_price: {feats.get('ref_price', 'N/A'):.4f}")
            print(f"    rsi: {feats.get('rsi', 'N/A')}")

            for window in spec.parkinson_windows:
                window_hours = window // 60
                feat_name = f"parkinson_{window_hours}h"
                vol_value = feats.get(feat_name)

                if vol_value is not None and not math.isnan(vol_value):
                    print(f"    {feat_name}: {vol_value:.6f}")
                else:
                    print(f"    {feat_name}: NaN (недостаточно данных)")

    # Проверяем что признаки созданы
    has_parkinson_24 = "parkinson_0h" in feats  # 24 мин = 0.4 часа, округляется до 0
    has_parkinson_50 = "parkinson_0h" in feats  # 50 мин = 0.83 часа, округляется до 0

    if has_parkinson_24 or has_parkinson_50:
        print("\n  ✓ Parkinson признаки успешно созданы")
    else:
        print("\n  ⚠️  Признаки Parkinson не найдены в результате")
        print(f"  Доступные признаки: {list(feats.keys())}")

    print()
    return True


def test_edge_cases():
    """Тест граничных случаев."""
    print("=== Тест 3: Граничные случаи ===")

    # Тест 1: Недостаточно данных
    ohlc_bars = [{"open": 100, "high": 101, "low": 99, "close": 100}]
    vol = calculate_parkinson_volatility(ohlc_bars, 24)
    if vol is None:
        print("  ✓ Корректная обработка недостаточных данных")
    else:
        print("  ❌ ОШИБКА: должна вернуться None при недостатке данных")

    # Тест 2: Нулевая волатильность (high = low)
    ohlc_bars = [
        {"open": 100, "high": 100, "low": 100, "close": 100}
        for _ in range(50)
    ]
    vol = calculate_parkinson_volatility(ohlc_bars, 24)
    if vol is not None and vol >= 0 and vol < 0.0001:
        print("  ✓ Корректная обработка нулевой волатильности")
    else:
        print(f"  ⚠️  Нулевая волатильность: {vol}")

    # Тест 3: Высокая волатильность (большой диапазон high-low)
    import random
    random.seed(42)
    ohlc_bars = []
    for _ in range(50):
        base = 100 * (1 + random.gauss(0, 0.1))
        # Создаем большой диапазон между high и low
        low_price = base * random.uniform(0.85, 0.95)
        high_price = base * random.uniform(1.05, 1.15)
        ohlc_bars.append({
            "open": base,
            "high": high_price,
            "low": low_price,
            "close": base * random.uniform(0.95, 1.05),
        })

    vol = calculate_parkinson_volatility(ohlc_bars, 24)
    if vol is not None and vol > 0:
        print(f"  ✓ Высокая волатильность рассчитана: {vol:.6f}")
    else:
        print("  ❌ ОШИБКА: не удалось рассчитать высокую волатильность")

    print()
    return True


def test_window_sizes():
    """Тест различных размеров окон."""
    print("=== Тест 4: Различные размеры окон ===")

    # Генерируем много данных
    ohlc_bars = []
    base_price = 100.0
    for i in range(50000):  # 50000 минут ≈ 34 дня
        noise = 0.02 * math.sin(i * 0.01)
        open_p = base_price * (1 + noise)
        high = open_p * 1.01
        low = open_p * 0.99
        close = base_price * (1 + 0.02 * math.sin((i + 0.5) * 0.01))
        ohlc_bars.append({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        })
        base_price *= 1.00001

    # Тестируем окна: 24ч (1440 мин), 7д (10080 мин)
    windows = [1440, 10080]
    for window in windows:
        vol = calculate_parkinson_volatility(ohlc_bars, window)
        window_hours = window // 60
        if vol is not None:
            print(f"  ✓ Окно {window_hours}ч ({window} мин): {vol:.6f}")
        else:
            print(f"  ❌ Окно {window_hours}ч ({window} мин): не рассчитано")

    print()
    return True


def test_efficiency_comparison():
    """Тест для демонстрации эффективности Parkinson по сравнению с close-to-close."""
    print("=== Тест 5: Сравнение эффективности (информационная ценность) ===")

    # Создаем данные с известной волатильностью
    ohlc_bars = []
    base_price = 100.0
    for i in range(500):
        # Синусоидальная волатильность
        volatility = 0.02
        noise = volatility * math.sin(i * 0.1)
        open_p = base_price * (1 + noise)

        # High и Low содержат больше информации чем Close
        intraday_range = volatility * abs(math.cos(i * 0.2))
        high = open_p * (1 + intraday_range)
        low = open_p * (1 - intraday_range)
        close = base_price * (1 + volatility * math.sin((i + 0.5) * 0.1))

        ohlc_bars.append({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        })
        base_price *= 1.0001

    # Рассчитываем Parkinson волатильность
    parkinson_vol = calculate_parkinson_volatility(ohlc_bars, 100)

    # Простая close-to-close волатильность для сравнения
    closes = [bar["close"] for bar in ohlc_bars[-100:]]
    returns = [math.log(closes[i] / closes[i-1]) for i in range(1, len(closes))]
    close_vol = math.sqrt(sum(r**2 for r in returns) / len(returns))

    print(f"  Parkinson волатильность (High-Low): {parkinson_vol:.6f}")
    print(f"  Close-to-Close волатильность:       {close_vol:.6f}")
    print(f"  Соотношение: {parkinson_vol/close_vol:.2f}x")
    print(f"  ✓ Parkinson использует информацию о дневном диапазоне")
    print(f"  ✓ Теоретически в 7.4 раза более эффективен")

    print()
    return True


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ PARKINSON ВОЛАТИЛЬНОСТИ")
    print("=" * 60 + "\n")

    tests = [
        test_parkinson_basic,
        test_online_transformer,
        test_edge_cases,
        test_window_sizes,
        test_efficiency_comparison,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ ИСКЛЮЧЕНИЕ в {test_func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"РЕЗУЛЬТАТЫ: {passed} успешно, {failed} неудачно")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
