#!/usr/bin/env python3
"""
Упрощенный тест Yang-Zhang волатильности без внешних зависимостей.
"""

import math


def calculate_yang_zhang_volatility(ohlc_bars, n):
    """
    Локальная копия функции для тестирования.
    """
    if not ohlc_bars or len(ohlc_bars) < n or n < 2:
        return None

    bars = list(ohlc_bars)[-n:]

    try:
        k = 0.34

        # Ночная волатильность
        overnight_returns = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].get("close", 0.0)
            curr_open = bars[i].get("open", 0.0)
            if prev_close > 0 and curr_open > 0:
                overnight_returns.append(math.log(curr_open / prev_close))

        if len(overnight_returns) < 2:
            return None

        mean_overnight = sum(overnight_returns) / len(overnight_returns)
        sigma_o_sq = sum((r - mean_overnight) ** 2 for r in overnight_returns) / (len(overnight_returns) - 1)

        # Open-close волатильность
        oc_returns = []
        for bar in bars:
            open_price = bar.get("open", 0.0)
            close_price = bar.get("close", 0.0)
            if open_price > 0 and close_price > 0:
                oc_returns.append(math.log(close_price / open_price))

        if len(oc_returns) < 2:
            return None

        mean_oc = sum(oc_returns) / len(oc_returns)
        sigma_c_sq = sum((r - mean_oc) ** 2 for r in oc_returns) / (len(oc_returns) - 1)

        # Rogers-Satchell
        rs_sum = 0.0
        rs_count = 0
        for bar in bars:
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)
            open_price = bar.get("open", 0.0)
            close_price = bar.get("close", 0.0)

            if high > 0 and low > 0 and open_price > 0 and close_price > 0:
                term1 = math.log(high / close_price) * math.log(high / open_price)
                term2 = math.log(low / close_price) * math.log(low / open_price)
                rs_sum += term1 + term2
                rs_count += 1

        if rs_count == 0:
            return None

        sigma_rs_sq = rs_sum / rs_count
        sigma_yz_sq = sigma_o_sq + k * sigma_c_sq + (1 - k) * sigma_rs_sq

        if sigma_yz_sq < 0:
            return None

        return math.sqrt(sigma_yz_sq)

    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def test_basic():
    """Базовый тест."""
    print("Тест 1: Базовая проверка")

    ohlc_bars = []
    base_price = 100.0
    for i in range(50):
        noise = 0.01 * math.sin(i * 0.5)
        open_p = base_price * (1 + noise)
        high = open_p * 1.005
        low = open_p * 0.995
        close = base_price * (1 + 0.01 * math.sin((i + 0.5) * 0.5))

        ohlc_bars.append({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        })
        base_price *= 1.001

    vol = calculate_yang_zhang_volatility(ohlc_bars, 24)

    assert vol is not None, "Волатильность должна быть рассчитана"
    assert vol > 0, "Волатильность должна быть положительной"
    assert 0.001 < vol < 0.5, f"Волатильность вне ожидаемого диапазона: {vol}"

    print(f"  ✓ Yang-Zhang волатильность: {vol:.6f}")
    return True


def test_insufficient_data():
    """Тест недостаточных данных."""
    print("Тест 2: Недостаточно данных")

    ohlc_bars = [{"open": 100, "high": 101, "low": 99, "close": 100}]
    vol = calculate_yang_zhang_volatility(ohlc_bars, 24)

    assert vol is None, "Должна вернуться None при недостатке данных"
    print("  ✓ Корректная обработка недостаточных данных")
    return True


def test_zero_volatility():
    """Тест нулевой волатильности."""
    print("Тест 3: Нулевая волатильность")

    ohlc_bars = [
        {"open": 100, "high": 100, "low": 100, "close": 100}
        for _ in range(50)
    ]
    vol = calculate_yang_zhang_volatility(ohlc_bars, 24)

    assert vol is not None, "Волатильность должна быть рассчитана"
    assert vol >= 0, "Волатильность не должна быть отрицательной"
    assert vol < 0.0001, f"Ожидалась близкая к нулю волатильность, получено: {vol}"

    print(f"  ✓ Нулевая волатильность: {vol:.6f}")
    return True


def test_formula_components():
    """Тест компонентов формулы."""
    print("Тест 4: Проверка компонентов формулы")

    # Создаем данные где известны компоненты
    ohlc_bars = []
    for i in range(30):
        # Чередуем рост и падение для создания волатильности
        direction = 1 if i % 2 == 0 else -1
        base = 100 + i * 0.1
        open_p = base
        close = base + direction * 0.5
        high = max(open_p, close) + 0.2
        low = min(open_p, close) - 0.2

        ohlc_bars.append({
            "open": open_p,
            "high": high,
            "low": low,
            "close": close,
        })

    vol = calculate_yang_zhang_volatility(ohlc_bars, 24)

    assert vol is not None, "Волатильность должна быть рассчитана"
    assert vol > 0, "Волатильность должна быть положительной"

    print(f"  ✓ Волатильность с чередующимися движениями: {vol:.6f}")
    return True


def test_window_sizes():
    """Тест разных размеров окон."""
    print("Тест 5: Разные размеры окон")

    ohlc_bars = []
    base_price = 100.0
    for i in range(1000):
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

    windows = [24, 100, 500]
    for window in windows:
        vol = calculate_yang_zhang_volatility(ohlc_bars, window)
        assert vol is not None, f"Волатильность для окна {window} должна быть рассчитана"
        assert vol > 0, f"Волатильность для окна {window} должна быть положительной"
        print(f"  ✓ Окно {window}: {vol:.6f}")

    return True


def main():
    """Запуск всех тестов."""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ YANG-ZHANG ВОЛАТИЛЬНОСТИ (упрощенная версия)")
    print("=" * 60 + "\n")

    tests = [
        test_basic,
        test_insufficient_data,
        test_zero_volatility,
        test_formula_components,
        test_window_sizes,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            print()
        except AssertionError as e:
            print(f"  ❌ ОШИБКА: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ❌ ИСКЛЮЧЕНИЕ: {e}\n")
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
