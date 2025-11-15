#!/usr/bin/env python3
"""
Standalone тесты функции calculate_parkinson_volatility.

Этот файл тестирует только функцию Parkinson, не требуя полного импорта transformers.
"""

import math
import sys
from typing import List, Dict, Optional


def calculate_parkinson_volatility(ohlc_bars: List[Dict[str, float]], n: int) -> Optional[float]:
    """
    Копия функции из transformers.py для standalone тестирования.
    """
    if not ohlc_bars or len(ohlc_bars) < n or n < 2:
        return None

    bars = list(ohlc_bars)[-n:]

    try:
        sum_sq = 0.0
        valid_bars = 0

        for bar in bars:
            high = bar.get("high", 0.0)
            low = bar.get("low", 0.0)

            if high > 0 and low > 0 and high >= low:
                log_hl = math.log(high / low)
                sum_sq += log_hl ** 2
                valid_bars += 1

        # Требуем минимум 2 валидных бара и минимум 80% от запрошенного окна
        min_required = max(2, int(0.8 * n))
        if valid_bars < min_required:
            return None

        # Используем valid_bars (количество реально использованных данных)
        parkinson_var = sum_sq / (4 * valid_bars * math.log(2))
        return math.sqrt(parkinson_var)

    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None


def test_formula_uses_valid_bars():
    """КРИТИЧЕСКИЙ ТЕСТ: Формула использует valid_bars, а не n."""
    print("=" * 70)
    print("ТЕСТ 1: Формула использует valid_bars (НЕ n)")
    print("=" * 70)

    # Используем данные где valid_bars >= 80% (4 из 5 = 80%)
    ohlc_bars = [
        {"high": 110.0, "low": 100.0},
        {"high": 120.0, "low": 110.0},
        {"high": 130.0, "low": 120.0},
        {"high": 140.0, "low": 130.0},
        {"high": 0.0, "low": 0.0},  # 1 невалидный
    ]

    n = 5
    valid_bars_count = 4

    result = calculate_parkinson_volatility(ohlc_bars, n)

    assert result is not None, "С 80% валидных баров должно работать"

    # Правильная формула (valid_bars)
    sum_sq = (
        math.log(110.0 / 100.0) ** 2 +
        math.log(120.0 / 110.0) ** 2 +
        math.log(130.0 / 120.0) ** 2 +
        math.log(140.0 / 130.0) ** 2
    )
    expected_correct = math.sqrt(sum_sq / (4 * valid_bars_count * math.log(2)))

    # Неправильная формула (n)
    expected_wrong = math.sqrt(sum_sq / (4 * n * math.log(2)))

    print(f"\nДанные: n={n}, valid_bars={valid_bars_count}")
    print(f"Результат:                    {result:.10f}")
    print(f"✓ Правильная (valid_bars={valid_bars_count}): {expected_correct:.10f}")
    print(f"✗ Неправильная (n={n}):        {expected_wrong:.10f}")

    diff_pct = ((expected_correct - expected_wrong) / expected_wrong) * 100
    print(f"Δ Разница: {diff_pct:.2f}% (правильная выше при пропусках)")

    assert abs(result - expected_correct) < 1e-10, "Должна использовать valid_bars!"
    assert abs(result - expected_wrong) > 1e-6, "НЕ должна использовать n!"

    print("\n✅ ТЕСТ ПРОЙДЕН: Формула корректна (valid_bars)")
    return True


def test_threshold_80_percent():
    """ТЕСТ: Порог 80% валидных баров."""
    print("\n" + "=" * 70)
    print("ТЕСТ 2: Порог 80% валидных баров")
    print("=" * 70)

    # 80% валидных - должно работать
    valid_80 = [{"high": 101.0, "low": 100.0}] * 8 + [{"high": 0.0, "low": 0.0}] * 2
    result_80 = calculate_parkinson_volatility(valid_80, 10)
    assert result_80 is not None, "При 80% должно работать"
    print(f"\n80% валидных (8/10): {result_80:.6f} ✓")

    # 70% валидных - не должно работать
    valid_70 = [{"high": 101.0, "low": 100.0}] * 7 + [{"high": 0.0, "low": 0.0}] * 3
    result_70 = calculate_parkinson_volatility(valid_70, 10)
    assert result_70 is None, "При 70% не должно работать"
    print(f"70% валидных (7/10): None ✓ (< 80%)")

    print("\n✅ ТЕСТ ПРОЙДЕН: Порог 80% работает корректно")
    return True


def test_edge_cases():
    """ТЕСТ: Граничные случаи."""
    print("\n" + "=" * 70)
    print("ТЕСТ 3: Граничные случаи")
    print("=" * 70)

    # Пустой список
    assert calculate_parkinson_volatility([], 10) is None
    print("\n✓ Пустой список: None")

    # None
    assert calculate_parkinson_volatility(None, 10) is None
    print("✓ None: None")

    # Недостаточно баров
    assert calculate_parkinson_volatility([{"high": 101, "low": 100}] * 5, 10) is None
    print("✓ Баров (5) < n (10): None")

    # n < 2
    assert calculate_parkinson_volatility([{"high": 101, "low": 100}], 1) is None
    print("✓ n=1: None")

    # Нулевая волатильность
    zero_vol = calculate_parkinson_volatility([{"high": 100, "low": 100}] * 10, 10)
    assert zero_vol is not None and zero_vol < 1e-10
    print(f"✓ H=L: σ={zero_vol:.10f} ≈ 0")

    # high < low
    assert calculate_parkinson_volatility([{"high": 95, "low": 100}] * 2, 2) is None
    print("✓ H<L: None")

    print("\n✅ ТЕСТ ПРОЙДЕН: Граничные случаи обработаны")
    return True


def test_statistical_properties():
    """ТЕСТ: Статистические свойства."""
    print("\n" + "=" * 70)
    print("ТЕСТ 4: Статистические свойства")
    print("=" * 70)

    # Монотонность: больше диапазон → больше волатильность
    ranges = [1.01, 1.02, 1.05, 1.10, 1.20]
    vols = []

    print("\nМонотонность (диапазон ↑ → волатильность ↑):")
    for r in ranges:
        ohlc = [{"high": 100.0 * r, "low": 100.0}] * 10
        vol = calculate_parkinson_volatility(ohlc, 10)
        vols.append(vol)
        print(f"  Диапазон {(r-1)*100:4.0f}%: σ = {vol:.6f}")

    for i in range(len(vols) - 1):
        assert vols[i] < vols[i+1], "Волатильность должна расти"

    # Масштабная инвариантность
    print("\nМасштабная инвариантность (5% диапазон):")
    price_levels = [1.0, 10.0, 100.0, 1000.0]
    vols_scale = []

    for price in price_levels:
        ohlc = [{"high": price * 1.05, "low": price}] * 10
        vol = calculate_parkinson_volatility(ohlc, 10)
        vols_scale.append(vol)
        print(f"  Цена={price:7.1f}: σ = {vol:.10f}")

    reference = vols_scale[0]
    for vol in vols_scale[1:]:
        assert abs(vol - reference) < 1e-8, "Должна быть масштабно инвариантной"

    print("\n✅ ТЕСТ ПРОЙДЕН: Статистические свойства корректны")
    return True


def test_numerical_accuracy():
    """ТЕСТ: Численная точность."""
    print("\n" + "=" * 70)
    print("ТЕСТ 5: Численная точность")
    print("=" * 70)

    ohlc_bars = [
        {"high": 105.0, "low": 100.0},
        {"high": 110.0, "low": 105.0},
        {"high": 115.0, "low": 110.0},
    ]

    result = calculate_parkinson_volatility(ohlc_bars, 3)

    # Вручную
    sum_sq = sum(math.log(bar["high"] / bar["low"]) ** 2 for bar in ohlc_bars)
    expected = math.sqrt(sum_sq / (4 * 3 * math.log(2)))

    print(f"\nРезультат: {result:.15f}")
    print(f"Ожидалось: {expected:.15f}")
    print(f"Разница:   {abs(result - expected):.2e}")

    assert abs(result - expected) < 1e-12, "Численная точность < 1e-12"

    print("\n✅ ТЕСТ ПРОЙДЕН: Численная точность высокая")
    return True


def test_parkinson_1980_formula():
    """ТЕСТ: Соответствие оригинальной формуле Parkinson (1980)."""
    print("\n" + "=" * 70)
    print("ТЕСТ 6: Соответствие формуле Parkinson (1980)")
    print("=" * 70)

    observations = [
        {"high": 105, "low": 98},
        {"high": 107, "low": 100},
        {"high": 110, "low": 103},
        {"high": 112, "low": 106},
        {"high": 115, "low": 108},
    ]

    n_obs = len(observations)

    # Формула из статьи: σ² = [1/(4n·ln(2))] · Σ(ln(H/L))²
    sum_sq = sum(math.log(obs["high"] / obs["low"]) ** 2 for obs in observations)
    expected_paper = math.sqrt(sum_sq / (4 * n_obs * math.log(2)))

    result = calculate_parkinson_volatility(observations, n_obs)

    print(f"\nФормула из статьи: σ = {expected_paper:.10f}")
    print(f"Наша реализация:   σ = {result:.10f}")
    print(f"Разница:           {abs(result - expected_paper):.2e}")

    assert abs(result - expected_paper) < 1e-12

    print("\n✅ ТЕСТ ПРОЙДЕН: Соответствует Parkinson (1980)")
    return True


def test_real_world_scenarios():
    """ТЕСТ: Реальные сценарии."""
    print("\n" + "=" * 70)
    print("ТЕСТ 7: Реальные сценарии")
    print("=" * 70)

    # Weekend gaps (71% валидных)
    valid = [{"high": 101.0, "low": 100.0}] * 10
    weekend = [{"high": 0.0, "low": 0.0}] * 4
    weekend_bars = valid[:5] + weekend[:2] + valid[5:] + weekend[2:]

    result_weekend = calculate_parkinson_volatility(weekend_bars, 14)
    assert result_weekend is None, "71% < 80%"
    print("\n✓ Weekend gaps (10/14 = 71%): None (< 80%)")

    # Crypto 24/7 (100% валидных)
    crypto_bars = [{"high": 101.0 + i*0.1, "low": 100.0 + i*0.1} for i in range(168)]
    result_crypto = calculate_parkinson_volatility(crypto_bars, 168)
    assert result_crypto is not None
    print(f"✓ Crypto 24/7 (168h): σ = {result_crypto:.6f}")

    # Maintenance (95.8% валидных)
    before = [{"high": 101.0, "low": 100.0}] * 46
    maintenance = [{"high": 0.0, "low": 0.0}] * 4
    after = [{"high": 101.0, "low": 100.0}] * 46
    maint_bars = before + maintenance + after

    result_maint = calculate_parkinson_volatility(maint_bars, 96)
    assert result_maint is not None, "95.8% > 80%"
    print(f"✓ Maintenance (92/96 = 95.8%): σ = {result_maint:.6f}")

    print("\n✅ ТЕСТ ПРОЙДЕН: Реальные сценарии обработаны")
    return True


def run_all_standalone_tests():
    """Запуск всех standalone тестов."""
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "КОМПЛЕКСНОЕ ТЕСТИРОВАНИЕ PARKINSON" + " " * 19 + "║")
    print("║" + " " * 21 + "(Standalone версия)" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")

    tests = [
        ("Формула (valid_bars vs n)", test_formula_uses_valid_bars),
        ("Порог 80%", test_threshold_80_percent),
        ("Граничные случаи", test_edge_cases),
        ("Статистические свойства", test_statistical_properties),
        ("Численная точность", test_numerical_accuracy),
        ("Формула Parkinson (1980)", test_parkinson_1980_formula),
        ("Реальные сценарии", test_real_world_scenarios),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                errors.append(f"{name}: FAILED")
        except AssertionError as e:
            failed += 1
            errors.append(f"{name}: {str(e)}")
        except Exception as e:
            failed += 1
            errors.append(f"{name}: ERROR - {str(e)}")

    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print(f"Всего тестов: {len(tests)}")
    print(f"Успешно:      {passed}")
    print(f"Провалено:    {failed}")

    if failed == 0:
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\nПОДТВЕРЖДЕНО:")
        print("  ✓ Формула использует valid_bars (корректно)")
        print("  ✓ Порог 80% работает правильно")
        print("  ✓ Граничные случаи обработаны")
        print("  ✓ Статистические свойства корректны")
        print("  ✓ Численная точность высокая")
        print("  ✓ Соответствует оригиналу Parkinson (1980)")
        print("  ✓ Реальные сценарии работают")
        print("=" * 70)
        return True
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ:")
        for error in errors:
            print(f"  - {error}")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_standalone_tests()
    sys.exit(0 if success else 1)
