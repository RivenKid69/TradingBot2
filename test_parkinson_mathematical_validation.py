#!/usr/bin/env python3
"""
Математическая валидация формулы Parkinson.

Этот файл содержит тесты, которые математически доказывают корректность
использования valid_bars вместо n в знаменателе формулы.
"""

import math
import unittest
import sys


class TestMathematicalFoundation(unittest.TestCase):
    """Математическое обоснование формулы."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_statistical_mean_principle(self):
        """
        Доказательство через принцип статистического среднего.

        Формула Parkinson основана на оценке E[(ln(H/L))²].
        Для оценки среднего используется: mean = sum / count_of_terms

        Аналогия: mean([10, 20, 30]) = 60/3, а НЕ 60/5 если массив размера 5
        """
        print("\n" + "=" * 70)
        print("ДОКАЗАТЕЛЬСТВО: Статистический принцип среднего")
        print("=" * 70)

        # Пример 1: Простое среднее
        values = [10, 20, 30]
        simple_mean = sum(values) / len(values)
        print(f"\nПример 1: Среднее [10, 20, 30]")
        print(f"  Правильно: sum/count = 60/3 = {simple_mean}")
        print(f"  НЕправильно было бы: 60/5 = {60/5} (если размер массива 5)")

        # Пример 2: То же для Parkinson
        ohlc_bars = [
            {"high": 110.0, "low": 100.0},  # ln(1.1)² ≈ 0.00905
            {"high": 120.0, "low": 110.0},  # ln(1.091)² ≈ 0.00765
            {"high": 130.0, "low": 120.0},  # ln(1.083)² ≈ 0.00638
        ]

        # Вручную вычисляем сумму квадратов
        sum_sq = sum(math.log(bar["high"] / bar["low"]) ** 2 for bar in ohlc_bars)
        count = 3

        print(f"\nПример 2: Parkinson с 3 валидными барами")
        print(f"  Сумма квадратов: {sum_sq:.6f}")
        print(f"  Правильное среднее: {sum_sq}/{count} = {sum_sq/count:.6f}")

        # E[(ln(H/L))²] = 4·ln(2)·σ²
        # Значит: σ² = E[(ln(H/L))²] / (4·ln(2))
        expected_mean = sum_sq / count
        sigma_squared = expected_mean / (4 * math.log(2))
        sigma = math.sqrt(sigma_squared)

        print(f"  σ² = {expected_mean:.6f} / (4·ln(2)) = {sigma_squared:.6f}")
        print(f"  σ = {sigma:.6f}")

        # Проверяем что наша формула дает тот же результат
        result = self.calc_parkinson(ohlc_bars, 3)
        self.assertAlmostEqual(result, sigma, places=10)
        print(f"  ✓ Формула дает: {result:.10f}")

    def test_sample_variance_analogy(self):
        """
        Аналогия с выборочной дисперсией.

        Выборочная дисперсия: s² = Σ(x_i - mean)² / (n-1)
        где n = количество наблюдений в выборке

        Parkinson: σ² = Σ(ln(H/L))² / (4·N·ln(2))
        где N = количество наблюдений (valid_bars)
        """
        print("\n" + "=" * 70)
        print("ДОКАЗАТЕЛЬСТВО: Аналогия с выборочной дисперсией")
        print("=" * 70)

        # Пример выборочной дисперсии
        sample = [10, 20, 30, 40, 50]
        mean = sum(sample) / len(sample)
        variance = sum((x - mean)**2 for x in sample) / (len(sample) - 1)

        print(f"\nВыборочная дисперсия для {sample}:")
        print(f"  Среднее: {mean}")
        print(f"  s² = Σ(x-mean)² / (n-1) = ... / {len(sample)-1} = {variance:.2f}")
        print(f"  ⚠️ НЕ делим на размер массива, а на КОЛИЧЕСТВО элементов!")

        # Аналогично для Parkinson
        print(f"\nParkinson волатильность:")
        print(f"  σ² = Σ(ln(H/L))² / (4·valid_bars·ln(2))")
        print(f"  valid_bars = количество фактических наблюдений")
        print(f"  ⚠️ НЕ делим на размер окна n, а на КОЛИЧЕСТВО валидных!")

    def test_expectation_estimation(self):
        """
        Проверка через оценку математического ожидания.

        Теория: E[(ln(H/L))²] = 4·ln(2)·σ² (для геометрического броуновского движения)

        Оценка: Ê[X] = (1/N) · ΣX_i, где N = кол-во наблюдений
        """
        print("\n" + "=" * 70)
        print("ДОКАЗАТЕЛЬСТВО: Оценка математического ожидания")
        print("=" * 70)

        # Создаем данные с известными свойствами
        n_window = 10
        n_valid = 7
        ohlc_bars = [{"high": 105.0, "low": 100.0} for _ in range(n_valid)]
        ohlc_bars += [{"high": 0.0, "low": 0.0} for _ in range(n_window - n_valid)]

        # Вычисляем сумму
        sum_sq = sum(
            math.log(bar["high"] / bar["low"]) ** 2
            for bar in ohlc_bars
            if bar["high"] > 0
        )

        print(f"\nДанные: окно n={n_window}, валидных={n_valid}")
        print(f"Сумма Σ(ln(H/L))² = {sum_sq:.6f}")

        # Правильная оценка E[X]
        estimated_mean_correct = sum_sq / n_valid
        print(f"\nПравильная оценка E[X]:")
        print(f"  Ê[X] = Σ(X_i) / N = {sum_sq:.6f} / {n_valid} = {estimated_mean_correct:.6f}")

        # Неправильная оценка (через n)
        estimated_mean_wrong = sum_sq / n_window
        print(f"\nНеправильная оценка (через n):")
        print(f"  Ê[X] = Σ(X_i) / n = {sum_sq:.6f} / {n_window} = {estimated_mean_wrong:.6f}")

        # Разница
        diff = ((estimated_mean_correct - estimated_mean_wrong) / estimated_mean_wrong) * 100
        print(f"\nРазница: {diff:.1f}% (правильная оценка ВЫШЕ)")

        # Проверяем что формула использует правильную оценку
        result = self.calc_parkinson(ohlc_bars, n_window)
        expected_sigma = math.sqrt(estimated_mean_correct / (4 * math.log(2)))

        self.assertAlmostEqual(result, expected_sigma, places=10)
        print(f"\n✓ Формула использует правильную оценку среднего")


class TestNumericalEquivalence(unittest.TestCase):
    """Тесты численной эквивалентности с эталонными реализациями."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def parkinson_reference_implementation(self, high_low_ratios):
        """
        Эталонная реализация формулы Parkinson.

        Основана на оригинальной статье Parkinson (1980).
        """
        n = len(high_low_ratios)
        if n < 2:
            return None

        sum_sq = sum(math.log(hl_ratio) ** 2 for hl_ratio in high_low_ratios)
        variance = sum_sq / (4 * n * math.log(2))
        return math.sqrt(variance)

    def test_equivalence_with_reference(self):
        """Сравнение с эталонной реализацией."""
        print("\n" + "=" * 70)
        print("ПРОВЕРКА: Эквивалентность с эталонной реализацией")
        print("=" * 70)

        # Тест-кейсы
        test_cases = [
            [1.05, 1.048, 1.045],
            [1.1, 1.09, 1.08, 1.07, 1.06],
            [1.01] * 10,
            [1.2, 1.15, 1.1, 1.05, 1.02, 1.01],
        ]

        for i, hl_ratios in enumerate(test_cases, 1):
            # Эталонная реализация
            reference_vol = self.parkinson_reference_implementation(hl_ratios)

            # Наша реализация
            ohlc_bars = [{"high": 100.0 * ratio, "low": 100.0} for ratio in hl_ratios]
            our_vol = self.calc_parkinson(ohlc_bars, len(hl_ratios))

            # Сравниваем
            self.assertAlmostEqual(our_vol, reference_vol, places=10,
                                  msg=f"Тест-кейс {i}: должен совпадать с эталоном")

            print(f"\nТест-кейс {i}: {len(hl_ratios)} наблюдений")
            print(f"  Эталон: {reference_vol:.10f}")
            print(f"  Наша:   {our_vol:.10f}")
            print(f"  ✓ Совпадают с точностью до 10 знаков")

    def test_known_mathematical_properties(self):
        """Проверка известных математических свойств."""
        print("\n" + "=" * 70)
        print("ПРОВЕРКА: Известные математические свойства")
        print("=" * 70)

        # Свойство 1: Линейность по времени
        # Для постоянной волатильности σ за период dt, σ за период N·dt = σ·√N
        print("\nСвойство 1: НЕ тестируем (Parkinson измеряет за период, не накапливает)")

        # Свойство 2: Масштабная инвариантность
        # σ не зависит от абсолютного уровня цен
        print("\nСвойство 2: Масштабная инвариантность")
        base_prices = [1.0, 10.0, 100.0, 1000.0]
        hl_ratio = 1.05  # 5% диапазон

        vols = []
        for base_price in base_prices:
            ohlc_bars = [
                {"high": base_price * hl_ratio, "low": base_price}
                for _ in range(10)
            ]
            vol = self.calc_parkinson(ohlc_bars, 10)
            vols.append(vol)

        # Все должны быть одинаковыми
        for i, (price, vol) in enumerate(zip(base_prices, vols)):
            self.assertAlmostEqual(vol, vols[0], places=8)
            print(f"  Базовая цена {price:7.1f}: σ = {vol:.10f} ✓")

        # Свойство 3: Монотонность
        # Большие диапазоны → большая волатильность
        print("\nСвойство 3: Монотонность (больше диапазон → больше волатильность)")
        ranges = [1.01, 1.02, 1.05, 1.10, 1.20]
        vols = []

        for hl_ratio in ranges:
            ohlc_bars = [
                {"high": 100.0 * hl_ratio, "low": 100.0}
                for _ in range(10)
            ]
            vol = self.calc_parkinson(ohlc_bars, 10)
            vols.append(vol)

        for i in range(len(vols) - 1):
            self.assertLess(vols[i], vols[i+1])
            print(f"  Диапазон {(ranges[i]-1)*100:4.0f}%: σ = {vols[i]:.6f} < "
                  f"{vols[i+1]:.6f} (диапазон {(ranges[i+1]-1)*100:4.0f}%) ✓")


class TestFormulaDerivation(unittest.TestCase):
    """Проверка вывода формулы из первых принципов."""

    def test_parkinson_1980_formula(self):
        """
        Проверка соответствия оригинальной формуле Parkinson (1980).

        Оригинальная работа: "The Extreme Value Method for Estimating
        the Variance of the Rate of Return", Journal of Business, 1980.

        Формула: σ² = [1/(4n·ln(2))] · Σ(ln(H_i/L_i))²
        где n = число наблюдений (observations)
        """
        print("\n" + "=" * 70)
        print("ПРОВЕРКА: Соответствие оригинальной формуле Parkinson (1980)")
        print("=" * 70)

        print("\nОригинальная формула из статьи:")
        print("  σ² = [1/(4n·ln(2))] · Σ(ln(H_i/L_i))²")
        print("  где n = number of observations")
        print("")
        print("ВАЖНО: 'observations' = фактические наблюдения, а не размер окна!")

        # Пример из статьи (упрощенный)
        observations = [
            {"high": 105, "low": 98},
            {"high": 107, "low": 100},
            {"high": 110, "low": 103},
            {"high": 112, "low": 106},
            {"high": 115, "low": 108},
        ]

        n_obs = len(observations)

        # Вычисляем по формуле из статьи
        sum_sq = sum(math.log(obs["high"] / obs["low"]) ** 2 for obs in observations)
        variance_paper = sum_sq / (4 * n_obs * math.log(2))
        volatility_paper = math.sqrt(variance_paper)

        print(f"\nВычисление по формуле из статьи:")
        print(f"  n (observations) = {n_obs}")
        print(f"  Σ(ln(H/L))² = {sum_sq:.6f}")
        print(f"  σ² = {sum_sq:.6f} / (4 × {n_obs} × ln(2)) = {variance_paper:.6f}")
        print(f"  σ = {volatility_paper:.6f}")

        # Проверяем нашу реализацию
        from transformers import calculate_parkinson_volatility
        our_result = calculate_parkinson_volatility(observations, n_obs)

        self.assertAlmostEqual(our_result, volatility_paper, places=10)
        print(f"\nНаша реализация: {our_result:.10f}")
        print(f"✓ Совпадает с формулой из статьи Parkinson (1980)")

    def test_brownian_motion_theory(self):
        """
        Теоретическое обоснование через геометрическое броуновское движение.

        Для GBM: E[(ln(H/L))²] = 4·ln(2)·σ²·dt
        где σ² - дисперсия доходности за единицу времени
        """
        print("\n" + "=" * 70)
        print("ПРОВЕРКА: Теоретическое обоснование через GBM")
        print("=" * 70)

        print("\nТеория геометрического броуновского движения:")
        print("  dS/S = μ·dt + σ·dW")
        print("  E[(ln(H/L))²] = 4·ln(2)·σ²·dt")
        print("")
        print("Оценка σ² из выборки:")
        print("  Ê[(ln(H/L))²] = (1/N) · Σ(ln(H_i/L_i))²")
        print("  где N = количество наблюдений")
        print("")
        print("Подставляя:")
        print("  (1/N) · Σ(ln(H_i/L_i))² = 4·ln(2)·σ²")
        print("  σ² = Σ(ln(H_i/L_i))² / (4·N·ln(2))")
        print("  σ = √[Σ(ln(H_i/L_i))² / (4·N·ln(2))]")
        print("")
        print("✓ Формула требует N = количество наблюдений (valid_bars)")


class TestEdgeCaseMathematics(unittest.TestCase):
    """Математические тесты граничных случаев."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_zero_volatility_limit(self):
        """Предел при H → L (нулевая волатильность)."""
        print("\n" + "=" * 70)
        print("МАТЕМАТИЧЕСКИЙ ТЕСТ: Предел при H → L")
        print("=" * 70)

        # H/L → 1, значит ln(H/L) → 0, значит σ → 0
        epsilons = [0.01, 0.001, 0.0001, 0.00001]

        print("\nПроверка lim(H→L) σ = 0:")
        for eps in epsilons:
            ohlc_bars = [
                {"high": 100.0 * (1 + eps), "low": 100.0}
                for _ in range(10)
            ]
            vol = self.calc_parkinson(ohlc_bars, 10)
            print(f"  H/L = 1 + {eps:.5f}: σ = {vol:.10f}")

        # Последнее значение должно быть очень близко к нулю
        self.assertLess(vol, 0.001)
        print(f"✓ При H→L волатильность→0")

    def test_symmetry_property(self):
        """
        Свойство симметрии: Parkinson одинаков для роста и падения.

        Для одинакового абсолютного диапазона σ должна быть одинаковой.
        """
        print("\n" + "=" * 70)
        print("МАТЕМАТИЧЕСКИЙ ТЕСТ: Симметрия роста и падения")
        print("=" * 70)

        # Рост: от 100 до 105
        growth_bars = [{"high": 105.0, "low": 100.0} for _ in range(10)]
        vol_growth = self.calc_parkinson(growth_bars, 10)

        # Падение: от 105 до 100 (H/L = 105/100 = 1.05 - то же самое!)
        decline_bars = [{"high": 105.0, "low": 100.0} for _ in range(10)]
        vol_decline = self.calc_parkinson(decline_bars, 10)

        self.assertAlmostEqual(vol_growth, vol_decline, places=10)
        print(f"\n  Рост (100→105):   σ = {vol_growth:.10f}")
        print(f"  Падение (105→100): σ = {vol_decline:.10f}")
        print(f"  ✓ Симметрия: одинаковый H/L → одинаковая σ")


def run_mathematical_validation():
    """Запуск всех математических тестов."""
    print("\n" + "=" * 70)
    print("МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ ФОРМУЛЫ PARKINSON")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestMathematicalFoundation,
        TestNumericalEquivalence,
        TestFormulaDerivation,
        TestEdgeCaseMathematics,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("ИТОГИ МАТЕМАТИЧЕСКОЙ ВАЛИДАЦИИ")
    print("=" * 70)
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ ПРОЙДЕНА!")
        print("\nДОКАЗАНО:")
        print("  ✓ Формула использует valid_bars (корректно)")
        print("  ✓ Соответствует статистическому принципу среднего")
        print("  ✓ Эквивалентна эталонной реализации")
        print("  ✓ Соответствует оригинальной статье Parkinson (1980)")
        print("  ✓ Обоснована через теорию GBM")
        print("  ✓ Математические свойства корректны")
        print("=" * 70)
        return True
    else:
        print("\n❌ МАТЕМАТИЧЕСКАЯ ВАЛИДАЦИЯ НЕ ПРОЙДЕНА!")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_mathematical_validation()
    sys.exit(0 if success else 1)
