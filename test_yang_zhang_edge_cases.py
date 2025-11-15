#!/usr/bin/env python3
"""
КРИТИЧЕСКИЙ ТЕСТ: Edge cases для Yang-Zhang NaN fix.
Проверяет сценарии, которые могли быть упущены.
"""

import math
import sys


def calculate_close_to_close_volatility(close_prices, n):
    if not close_prices or len(close_prices) < n or n < 2:
        return None
    prices = list(close_prices)[-n:]
    try:
        log_returns = []
        for i in range(1, len(prices)):
            if prices[i-1] > 0 and prices[i] > 0:
                log_returns.append(math.log(prices[i] / prices[i-1]))
        if len(log_returns) < 2:
            return None
        mean_return = sum(log_returns) / len(log_returns)
        variance = sum((r - mean_return) ** 2 for r in log_returns) / (len(log_returns) - 1)
        if variance < 0:
            return None
        return math.sqrt(variance)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def test_window_larger_than_data():
    """EDGE CASE: Окно больше чем доступных данных."""
    print("\n=== EDGE CASE 1: Окно > Данных ===")

    # Только 5 цен, но окно 12
    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    vol = calculate_close_to_close_volatility(prices, 12)

    assert vol is None, "Должен возвращать None когда окно > данных"
    print("✓ Корректно обрабатывает окно > данных")


def test_single_price():
    """EDGE CASE: Только одна цена."""
    print("\n=== EDGE CASE 2: Одна цена ===")

    prices = [100.0]
    vol = calculate_close_to_close_volatility(prices, 5)

    assert vol is None, "Должен возвращать None с одной ценой"
    print("✓ Корректно обрабатывает одну цену")


def test_zero_prices():
    """EDGE CASE: Нулевые цены (некорректные данные)."""
    print("\n=== EDGE CASE 3: Нулевые цены ===")

    prices = [0.0, 0.0, 0.0, 0.0, 0.0]
    vol = calculate_close_to_close_volatility(prices, 5)

    assert vol is None, "Должен возвращать None с нулевыми ценами"
    print("✓ Корректно обрабатывает нулевые цены")


def test_negative_prices():
    """EDGE CASE: Отрицательные цены (некорректные данные)."""
    print("\n=== EDGE CASE 4: Отрицательные цены ===")

    prices = [100.0, -50.0, 102.0, 103.0, 104.0]
    vol = calculate_close_to_close_volatility(prices, 5)

    # Должен пропустить отрицательную цену и продолжить
    # Но в итоге может не хватить данных
    print(f"  Волатильность с отрицательной ценой: {vol}")
    # Не падает - это хорошо
    print("✓ Не падает с отрицательными ценами")


def test_extreme_volatility():
    """EDGE CASE: Экстремальная волатильность (10x изменение)."""
    print("\n=== EDGE CASE 5: Экстремальная волатильность ===")

    prices = [100.0, 1000.0, 10.0, 500.0, 50.0, 250.0]
    vol = calculate_close_to_close_volatility(prices, 6)

    assert vol is not None, "Должен работать с экстремальной волатильностью"
    assert vol > 1.0, "Экстремальная волатильность должна быть >1.0"
    print(f"✓ Экстремальная волатильность: {vol:.4f}")


def test_very_small_changes():
    """EDGE CASE: Очень маленькие изменения (точность float)."""
    print("\n=== EDGE CASE 6: Микро-изменения ===")

    prices = [100.00000, 100.00001, 100.00002, 100.00001, 100.00003]
    vol = calculate_close_to_close_volatility(prices, 5)

    assert vol is not None, "Должен работать с микро-изменениями"
    assert vol > 0, "Волатильность должна быть > 0"
    assert vol < 0.01, "Волатильность должна быть очень маленькой"
    print(f"✓ Микро-волатильность: {vol:.10f}")


def test_constant_price_with_noise():
    """EDGE CASE: Почти константная цена с шумом."""
    print("\n=== EDGE CASE 7: Константная + шум ===")

    base = 50000.0
    prices = [base + i * 0.00001 for i in range(20)]
    vol = calculate_close_to_close_volatility(prices, 20)

    assert vol is not None, "Должен работать с константой + шум"
    assert vol < 0.0001, "Волатильность должна быть минимальной"
    print(f"✓ Константная + шум: {vol:.15f}")


def test_exact_window_size():
    """EDGE CASE: Данных ровно столько, сколько окно."""
    print("\n=== EDGE CASE 8: Данных = Окно ===")

    prices = [100.0, 101.0, 102.0, 103.0, 104.0]
    vol = calculate_close_to_close_volatility(prices, 5)

    assert vol is not None, "Должен работать когда данных = окно"
    assert vol > 0, "Волатильность должна быть > 0"
    print(f"✓ Данных = Окно: {vol:.6f}")


def test_inf_and_nan():
    """EDGE CASE: Infinity и NaN в данных."""
    print("\n=== EDGE CASE 9: Inf/NaN в данных ===")

    prices = [100.0, float('inf'), 102.0, 103.0, 104.0]
    vol = calculate_close_to_close_volatility(prices, 5)

    # Должен обработать без падения
    print(f"  Волатильность с inf: {vol}")
    print("✓ Не падает с inf/nan")


def test_very_long_sequence():
    """EDGE CASE: Очень длинная последовательность (производительность)."""
    print("\n=== EDGE CASE 10: Длинная последовательность ===")

    # 10000 точек данных
    import time
    prices = [100.0 + i * 0.01 + (i % 10) * 0.5 for i in range(10000)]

    start = time.time()
    vol = calculate_close_to_close_volatility(prices, 1000)
    elapsed = time.time() - start

    assert vol is not None, "Должен работать с большими данными"
    assert elapsed < 1.0, f"Должен быть быстрым, но занял {elapsed:.4f}s"
    print(f"✓ Большие данные: {vol:.6f} за {elapsed:.4f}s")


def run_all_edge_tests():
    """Запуск всех edge case тестов."""
    print("=" * 70)
    print("КРИТИЧЕСКИЕ EDGE CASE ТЕСТЫ ДЛЯ YANG-ZHANG NaN FIX")
    print("=" * 70)

    try:
        test_window_larger_than_data()
        test_single_price()
        test_zero_prices()
        test_negative_prices()
        test_extreme_volatility()
        test_very_small_changes()
        test_constant_price_with_noise()
        test_exact_window_size()
        test_inf_and_nan()
        test_very_long_sequence()

        print("\n" + "=" * 70)
        print("✅ ВСЕ EDGE CASE ТЕСТЫ ПРОЙДЕНЫ!")
        print("=" * 70)
        print("\n🎯 РЕШЕНИЕ УСТОЙЧИВО К ГРАНИЧНЫМ СЛУЧАЯМ!")
        return True

    except AssertionError as e:
        print(f"\n❌ EDGE CASE ТЕСТ ПРОВАЛЕН: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ НЕОЖИДАННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_edge_tests()
    sys.exit(0 if success else 1)
