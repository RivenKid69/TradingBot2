#!/usr/bin/env python3
"""
Комплексное тестирование Parkinson волатильности (100% покрытие).

Этот файл содержит исчерпывающий набор тестов для проверки:
1. Корректности формулы (valid_bars vs n)
2. Граничных случаев
3. Порога 80% валидных баров
4. Численной точности
5. Обработки пропусков данных
6. Интеграции с трансформером
"""

import math
import unittest
import sys


class TestParkinsonFormulaCorrectness(unittest.TestCase):
    """Тесты корректности формулы Parkinson."""

    def setUp(self):
        """Импортируем функцию для каждого теста."""
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_formula_uses_valid_bars_not_n(self):
        """
        КРИТИЧЕСКИЙ ТЕСТ: Проверяет что формула использует valid_bars, а не n.

        Математическое обоснование:
        E[(ln(H/L))²] ≈ (1/N) · Σ(ln(H_i/L_i))²
        где N = количество слагаемых в сумме (valid_bars)
        """
        # Создаем данные с известными значениями
        ohlc_bars = [
            {"high": 110.0, "low": 100.0},  # ln(1.1)² ≈ 0.00905
            {"high": 120.0, "low": 110.0},  # ln(1.091)² ≈ 0.00765
            {"high": 0.0, "low": 0.0},      # Невалидный
            {"high": 0.0, "low": 0.0},      # Невалидный
            {"high": 130.0, "low": 120.0},  # ln(1.083)² ≈ 0.00638
        ]

        n = 5  # размер окна
        valid_bars_count = 3  # фактических наблюдений

        result = self.calc_parkinson(ohlc_bars, n)
        self.assertIsNotNone(result, "Должно работать с 60% валидных баров")

        # Вычисляем вручную с ПРАВИЛЬНОЙ формулой (valid_bars)
        sum_sq = (
            math.log(110.0 / 100.0) ** 2 +
            math.log(120.0 / 110.0) ** 2 +
            math.log(130.0 / 120.0) ** 2
        )
        expected_var_correct = sum_sq / (4 * valid_bars_count * math.log(2))
        expected_vol_correct = math.sqrt(expected_var_correct)

        # Проверяем что результат соответствует ПРАВИЛЬНОЙ формуле
        self.assertAlmostEqual(
            result, expected_vol_correct, places=6,
            msg=f"Формула должна использовать valid_bars={valid_bars_count}, а не n={n}"
        )

        # Вычисляем что было бы с НЕПРАВИЛЬНОЙ формулой (n)
        expected_var_wrong = sum_sq / (4 * n * math.log(2))
        expected_vol_wrong = math.sqrt(expected_var_wrong)

        # Убеждаемся что результат НЕ соответствует неправильной формуле
        self.assertNotAlmostEqual(
            result, expected_vol_wrong, places=6,
            msg="Формула НЕ должна использовать n (размер окна)"
        )

        # Показываем разницу
        diff_pct = ((expected_vol_wrong - expected_vol_correct) / expected_vol_correct) * 100
        print(f"\n  ✓ Правильная формула (valid_bars={valid_bars_count}): {expected_vol_correct:.6f}")
        print(f"  ✗ Неправильная формула (n={n}): {expected_vol_wrong:.6f}")
        print(f"  Δ Разница: {diff_pct:.2f}% (неправильная занижает)")

    def test_exact_numerical_calculation(self):
        """Тест точности численного расчета с известными значениями."""
        # Простые данные для ручной проверки
        ohlc_bars = [
            {"high": 105.0, "low": 100.0},  # ln(1.05)² = 0.002378...
            {"high": 110.0, "low": 105.0},  # ln(1.048)² = 0.002216...
            {"high": 115.0, "low": 110.0},  # ln(1.045)² = 0.001949...
        ]

        result = self.calc_parkinson(ohlc_bars, 3)

        # Вручную вычисляем
        sum_sq = sum(math.log(bar["high"] / bar["low"]) ** 2 for bar in ohlc_bars)
        expected_var = sum_sq / (4 * 3 * math.log(2))
        expected_vol = math.sqrt(expected_var)

        self.assertAlmostEqual(result, expected_vol, places=10,
                              msg="Численная точность должна быть высокой")

        print(f"\n  ✓ Численная точность: {result:.10f} ≈ {expected_vol:.10f}")

    def test_all_valid_bars_equals_n(self):
        """Когда все бары валидны (valid_bars = n), формулы эквивалентны."""
        ohlc_bars = [
            {"high": 101.0, "low": 100.0} for _ in range(10)
        ]

        result = self.calc_parkinson(ohlc_bars, 10)

        # В этом случае valid_bars = n = 10, поэтому обе формулы дают одинаковый результат
        sum_sq = 10 * (math.log(101.0 / 100.0) ** 2)
        expected = math.sqrt(sum_sq / (4 * 10 * math.log(2)))

        self.assertAlmostEqual(result, expected, places=10)
        print(f"\n  ✓ При valid_bars=n: результат корректен ({result:.6f})")


class TestParkinsonThreshold(unittest.TestCase):
    """Тесты порога 80% валидных баров."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_exactly_80_percent_valid(self):
        """Ровно 80% валидных баров - должно работать."""
        # n=10, 80% = 8 баров
        valid_data = [{"high": 101.0, "low": 100.0} for _ in range(8)]
        invalid_data = [{"high": 0.0, "low": 0.0} for _ in range(2)]
        ohlc_bars = valid_data + invalid_data

        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNotNone(result, "При 80% должно работать")
        print(f"\n  ✓ 80% валидных (8/10): {result:.6f}")

    def test_just_below_80_percent_invalid(self):
        """Чуть ниже 80% (79%) - должно вернуть None."""
        # n=10, 79% = 7.9 → 7 баров (int(0.8*10)=8 требуется)
        valid_data = [{"high": 101.0, "low": 100.0} for _ in range(7)]
        invalid_data = [{"high": 0.0, "low": 0.0} for _ in range(3)]
        ohlc_bars = valid_data + invalid_data

        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNone(result, "При 70% должно вернуть None")
        print(f"\n  ✓ 70% валидных (7/10): None (< 80%)")

    def test_threshold_various_window_sizes(self):
        """Проверка порога для различных размеров окон."""
        test_cases = [
            (5, 4),   # n=5: требуется max(2, 4) = 4 бара
            (10, 8),  # n=10: требуется max(2, 8) = 8 баров
            (12, 9),  # n=12: требуется max(2, 9) = 9 баров
            (50, 40), # n=50: требуется max(2, 40) = 40 баров
            (2, 2),   # n=2: требуется max(2, 1) = 2 бара (минимум)
            (3, 2),   # n=3: требуется max(2, 2) = 2 бара (минимум)
        ]

        print("\n  Проверка порогов:")
        for n, min_required in test_cases:
            # Создаем ровно min_required валидных баров
            valid_data = [{"high": 101.0, "low": 100.0} for _ in range(min_required)]
            invalid_data = [{"high": 0.0, "low": 0.0} for _ in range(n - min_required)]
            ohlc_bars = valid_data + invalid_data

            result = self.calc_parkinson(ohlc_bars, n)
            self.assertIsNotNone(result,
                f"При n={n} должно работать с {min_required} валидными барами")

            # Проверяем что с одним меньше не работает
            if min_required > 2:  # Только если можем убрать бар
                ohlc_bars_minus1 = valid_data[:-1] + invalid_data + [{"high": 0.0, "low": 0.0}]
                result_minus1 = self.calc_parkinson(ohlc_bars_minus1, n)
                self.assertIsNone(result_minus1,
                    f"При n={n} НЕ должно работать с {min_required-1} валидными барами")

            print(f"    n={n:2d}: требуется ≥{min_required:2d} баров ({min_required/n*100:.0f}%) ✓")

    def test_minimum_2_bars_required(self):
        """Минимум 2 бара всегда требуется, даже если 80% меньше."""
        # n=2: int(0.8*2) = 1, но max(2, 1) = 2
        ohlc_bars = [
            {"high": 101.0, "low": 100.0},
            {"high": 102.0, "low": 101.0},
        ]
        result = self.calc_parkinson(ohlc_bars, 2)
        self.assertIsNotNone(result, "Минимум 2 бара должно работать")

        # Только 1 валидный бар - не должно работать
        ohlc_bars_1 = [
            {"high": 101.0, "low": 100.0},
            {"high": 0.0, "low": 0.0},
        ]
        result_1 = self.calc_parkinson(ohlc_bars_1, 2)
        self.assertIsNone(result_1, "С 1 баром не должно работать")

        print(f"\n  ✓ Минимум 2 бара всегда требуется")


class TestParkinsonEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_empty_bars(self):
        """Пустой список баров."""
        result = self.calc_parkinson([], 10)
        self.assertIsNone(result)
        print(f"\n  ✓ Пустой список: None")

    def test_none_bars(self):
        """None вместо списка."""
        result = self.calc_parkinson(None, 10)
        self.assertIsNone(result)
        print(f"\n  ✓ None: None")

    def test_insufficient_bars(self):
        """Баров меньше чем n."""
        ohlc_bars = [{"high": 101.0, "low": 100.0} for _ in range(5)]
        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNone(result, "Меньше баров чем n - должно вернуть None")
        print(f"\n  ✓ Баров (5) < n (10): None")

    def test_window_size_less_than_2(self):
        """Размер окна меньше 2."""
        ohlc_bars = [{"high": 101.0, "low": 100.0}]
        result = self.calc_parkinson(ohlc_bars, 1)
        self.assertIsNone(result)
        print(f"\n  ✓ n=1: None (минимум n=2)")

    def test_zero_volatility(self):
        """Нулевая волатильность (high = low для всех баров)."""
        ohlc_bars = [
            {"high": 100.0, "low": 100.0} for _ in range(10)
        ]
        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=10,
                              msg="При high=low волатильность должна быть 0")
        print(f"\n  ✓ Нулевая волатильность: {result:.10f}")

    def test_high_less_than_low(self):
        """high < low (некорректные данные)."""
        ohlc_bars = [
            {"high": 95.0, "low": 100.0},  # high < low
            {"high": 101.0, "low": 100.0},
        ]
        # Только 1 валидный бар из 2 (50% < 80%)
        result = self.calc_parkinson(ohlc_bars, 2)
        self.assertIsNone(result, "Некорректные данные должны игнорироваться")
        print(f"\n  ✓ high < low игнорируется")

    def test_negative_or_zero_prices(self):
        """Отрицательные или нулевые цены."""
        test_cases = [
            [{"high": -100.0, "low": -110.0}, {"high": 101.0, "low": 100.0}],
            [{"high": 0.0, "low": 100.0}, {"high": 101.0, "low": 100.0}],
            [{"high": 100.0, "low": 0.0}, {"high": 101.0, "low": 100.0}],
            [{"high": 0.0, "low": 0.0}, {"high": 101.0, "low": 100.0}],
        ]

        for i, ohlc_bars in enumerate(test_cases):
            result = self.calc_parkinson(ohlc_bars, 2)
            self.assertIsNone(result, f"Тест-кейс {i+1}: некорректные цены")

        print(f"\n  ✓ Отрицательные/нулевые цены обработаны")

    def test_very_small_range(self):
        """Очень маленький диапазон (почти нулевая волатильность)."""
        ohlc_bars = [
            {"high": 100.00001, "low": 100.0} for _ in range(10)
        ]
        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 0.001, "Волатильность должна быть очень маленькой")
        print(f"\n  ✓ Маленький диапазон: {result:.10f}")

    def test_very_large_range(self):
        """Очень большой диапазон (высокая волатильность)."""
        ohlc_bars = [
            {"high": 200.0, "low": 100.0} for _ in range(10)
        ]
        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.1, "Волатильность должна быть высокой")
        print(f"\n  ✓ Большой диапазон: {result:.6f}")

    def test_mixed_valid_invalid_patterns(self):
        """Различные паттерны смешивания валидных и невалидных баров."""
        patterns = [
            # Паттерн 1: невалидные в начале
            [{"high": 0.0, "low": 0.0}] * 2 + [{"high": 101.0, "low": 100.0}] * 8,
            # Паттерн 2: невалидные в конце
            [{"high": 101.0, "low": 100.0}] * 8 + [{"high": 0.0, "low": 0.0}] * 2,
            # Паттерн 3: невалидные в середине
            [{"high": 101.0, "low": 100.0}] * 4 + [{"high": 0.0, "low": 0.0}] * 2 + [{"high": 101.0, "low": 100.0}] * 4,
            # Паттерн 4: чередование
            [{"high": 101.0, "low": 100.0}, {"high": 0.0, "low": 0.0}] * 5,
        ]

        print("\n  Различные паттерны пропусков:")
        for i, ohlc_bars in enumerate(patterns, 1):
            result = self.calc_parkinson(ohlc_bars, 10)
            valid_count = sum(1 for bar in ohlc_bars if bar["high"] > 0)
            if valid_count >= 8:  # 80% от 10
                self.assertIsNotNone(result, f"Паттерн {i}: должно работать")
                print(f"    Паттерн {i} ({valid_count}/10): {result:.6f} ✓")
            else:
                self.assertIsNone(result, f"Паттерн {i}: должно вернуть None")
                print(f"    Паттерн {i} ({valid_count}/10): None ✓")


class TestParkinsonRealWorldScenarios(unittest.TestCase):
    """Тесты реальных сценариев."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_weekend_gaps(self):
        """Симуляция пропусков из-за выходных (2/7 дней = ~28% пропусков)."""
        # 5 рабочих дней + 2 выходных = 7 дней
        # Для окна 14 дней: 10 валидных + 4 невалидных
        valid_bars = [{"high": 101.0, "low": 100.0} for _ in range(10)]
        weekend_bars = [{"high": 0.0, "low": 0.0} for _ in range(4)]

        # Чередуем: 5 рабочих, 2 выходных, 5 рабочих, 2 выходных
        ohlc_bars = valid_bars[:5] + weekend_bars[:2] + valid_bars[5:] + weekend_bars[2:]

        result = self.calc_parkinson(ohlc_bars, 14)
        self.assertIsNotNone(result, "При weekends (71% валидных) должно работать")
        print(f"\n  ✓ Weekend gaps (10/14 = 71%): {result:.6f}")

    def test_crypto_24_7_no_gaps(self):
        """Криптовалюты - торговля 24/7, нет пропусков."""
        ohlc_bars = [{"high": 101.0 + i*0.1, "low": 100.0 + i*0.1} for i in range(168)]
        result = self.calc_parkinson(ohlc_bars, 168)  # 7 дней * 24 часа
        self.assertIsNotNone(result, "24/7 торговля без пропусков")
        print(f"\n  ✓ 24/7 криптовалюта (168h): {result:.6f}")

    def test_low_liquidity_with_gaps(self):
        """Низкая ликвидность с частыми пропусками (~30% пропусков)."""
        # 70% валидных данных
        valid_bars = [{"high": 105.0, "low": 100.0} for _ in range(7)]
        gap_bars = [{"high": 0.0, "low": 0.0} for _ in range(3)]
        ohlc_bars = valid_bars + gap_bars

        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNone(result, "При 70% валидных (< 80%) должно вернуть None")
        print(f"\n  ✓ Низкая ликвидность (7/10 = 70%): None")

    def test_maintenance_downtime(self):
        """Maintenance окно биржи (короткий перерыв)."""
        # 48 часов с 2-часовым maintenance (4 бара по 30 мин)
        bars_before = [{"high": 101.0, "low": 100.0} for _ in range(46)]
        maintenance = [{"high": 0.0, "low": 0.0} for _ in range(4)]
        bars_after = [{"high": 101.0, "low": 100.0} for _ in range(46)]

        ohlc_bars = bars_before + maintenance + bars_after

        result = self.calc_parkinson(ohlc_bars, 96)  # 48h * 2 (30-min bars)
        # 92 валидных из 96 = 95.8% > 80%
        self.assertIsNotNone(result, "После maintenance (95.8% валидных) должно работать")
        print(f"\n  ✓ Maintenance downtime (92/96 = 95.8%): {result:.6f}")

    def test_flash_crash_extreme_volatility(self):
        """Flash crash - экстремальная волатильность."""
        # Нормальные условия
        normal_bars = [{"high": 101.0, "low": 100.0} for _ in range(8)]
        # Flash crash
        crash_bar = {"high": 120.0, "low": 80.0}  # огромный диапазон
        # Восстановление
        recovery_bar = {"high": 102.0, "low": 99.0}

        ohlc_bars = normal_bars + [crash_bar, recovery_bar]

        result = self.calc_parkinson(ohlc_bars, 10)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.05, "Волатильность должна быть высокой при flash crash")
        print(f"\n  ✓ Flash crash (экстремальная волатильность): {result:.6f}")

    def test_trending_market(self):
        """Трендовый рынок с растущими ценами."""
        ohlc_bars = []
        base_price = 100.0
        for i in range(50):
            high = base_price * (1 + 0.01)  # +1% дневной рост
            low = base_price * (1 - 0.005)   # -0.5% внутридневной минимум
            ohlc_bars.append({"high": high, "low": low})
            base_price *= 1.005  # рост базовой цены

        result = self.calc_parkinson(ohlc_bars, 50)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.0)
        print(f"\n  ✓ Трендовый рынок: {result:.6f}")


class TestParkinsonStatisticalProperties(unittest.TestCase):
    """Тесты статистических свойств."""

    def setUp(self):
        from transformers import calculate_parkinson_volatility
        self.calc_parkinson = calculate_parkinson_volatility

    def test_volatility_increases_with_range(self):
        """Волатильность увеличивается с ростом диапазона high-low."""
        ranges = [0.01, 0.02, 0.05, 0.10, 0.20]  # 1%, 2%, 5%, 10%, 20%
        volatilities = []

        for r in ranges:
            ohlc_bars = [{"high": 100.0 * (1 + r), "low": 100.0} for _ in range(10)]
            vol = self.calc_parkinson(ohlc_bars, 10)
            volatilities.append(vol)

        # Проверяем монотонность
        for i in range(len(volatilities) - 1):
            self.assertLess(volatilities[i], volatilities[i+1],
                           f"Волатильность должна расти с диапазоном")

        print(f"\n  Волатильность vs диапазон:")
        for r, vol in zip(ranges, volatilities):
            print(f"    {r*100:5.1f}% диапазон → {vol:.6f} волатильность")

    def test_volatility_scale_invariance(self):
        """Проверка масштабной инвариантности (волатильность не зависит от уровня цен)."""
        price_levels = [1.0, 10.0, 100.0, 1000.0]
        volatilities = []

        # Одинаковый относительный диапазон (5%) для разных уровней цен
        relative_range = 0.05

        for price_level in price_levels:
            ohlc_bars = [
                {"high": price_level * (1 + relative_range),
                 "low": price_level}
                for _ in range(10)
            ]
            vol = self.calc_parkinson(ohlc_bars, 10)
            volatilities.append(vol)

        # Все волатильности должны быть примерно одинаковыми
        reference_vol = volatilities[0]
        for vol in volatilities[1:]:
            self.assertAlmostEqual(vol, reference_vol, places=8,
                                  msg="Волатильность должна быть масштабно инвариантной")

        print(f"\n  ✓ Масштабная инвариантность (5% диапазон):")
        for price, vol in zip(price_levels, volatilities):
            print(f"    Цена={price:7.1f}: vol={vol:.10f}")

    def test_window_size_effect(self):
        """Эффект размера окна на оценку волатильности."""
        # Генерируем данные с известной волатильностью
        ohlc_bars = [{"high": 101.0, "low": 100.0} for _ in range(100)]

        window_sizes = [5, 10, 20, 50, 100]
        volatilities = []

        for window in window_sizes:
            vol = self.calc_parkinson(ohlc_bars, window)
            volatilities.append(vol)

        # С одинаковыми данными волатильность не должна сильно меняться
        reference_vol = volatilities[-1]  # самое большое окно = лучшая оценка
        for vol in volatilities:
            relative_diff = abs(vol - reference_vol) / reference_vol
            self.assertLess(relative_diff, 0.01,  # < 1% отклонения
                           "Оценки при разных окнах должны быть близки")

        print(f"\n  Размер окна vs волатильность (одинаковые данные):")
        for window, vol in zip(window_sizes, volatilities):
            print(f"    n={window:3d}: {vol:.8f}")


class TestParkinsonIntegration(unittest.TestCase):
    """Тесты интеграции с OnlineFeatureTransformer."""

    def test_transformer_creates_parkinson_features(self):
        """Проверка создания Parkinson признаков в трансформере."""
        from transformers import FeatureSpec, OnlineFeatureTransformer

        spec = FeatureSpec(
            lookbacks_prices=[5],
            rsi_period=14,
            yang_zhang_windows=[],
            parkinson_windows=[2880, 10080],  # 48h, 7d в минутах для 4h бара
            garch_windows=[],
            bar_duration_minutes=240,  # 4h бар
        )

        transformer = OnlineFeatureTransformer(spec)

        # Подаем 50 баров с OHLC данными
        symbol = "BTCUSDT"
        ts_ms = 1600000000000

        for i in range(50):
            feats = transformer.update(
                symbol=symbol,
                ts_ms=ts_ms,
                close=100.0 + i * 0.5,
                open_price=100.0 + i * 0.5 - 0.2,
                high=100.0 + i * 0.5 + 0.3,
                low=100.0 + i * 0.5 - 0.3,
            )
            ts_ms += 4 * 60 * 60 * 1000  # +4 часа

        # Проверяем что признаки созданы
        self.assertIn("parkinson_48h", feats, "Должен быть признак parkinson_48h")
        self.assertIn("parkinson_7d", feats, "Должен быть признак parkinson_7d")

        # После 50 баров оба признака должны быть валидны
        # parkinson_48h требует 12 баров (80% от 12 = 9.6 → 10)
        # parkinson_7d требует 42 бара (80% от 42 = 33.6 → 34)

        parkinson_48h = feats["parkinson_48h"]
        parkinson_7d = feats["parkinson_7d"]

        self.assertFalse(math.isnan(parkinson_48h),
                        "parkinson_48h должен быть валиден после 50 баров")
        self.assertFalse(math.isnan(parkinson_7d),
                        "parkinson_7d должен быть валиден после 50 баров")

        self.assertGreater(parkinson_48h, 0.0)
        self.assertGreater(parkinson_7d, 0.0)

        print(f"\n  ✓ Transformer интеграция:")
        print(f"    parkinson_48h: {parkinson_48h:.6f}")
        print(f"    parkinson_7d: {parkinson_7d:.6f}")

    def test_transformer_handles_missing_ohlc(self):
        """Проверка обработки пропущенных OHLC данных."""
        from transformers import FeatureSpec, OnlineFeatureTransformer

        spec = FeatureSpec(
            lookbacks_prices=[5],
            rsi_period=14,
            parkinson_windows=[1440],  # 24h в минутах для 1h бара
            bar_duration_minutes=60,  # 1h бар
        )

        transformer = OnlineFeatureTransformer(spec)

        symbol = "BTCUSDT"
        ts_ms = 1600000000000

        # Подаем бары с пропусками OHLC
        for i in range(30):
            if i % 5 == 0:  # Каждый 5й бар без OHLC
                feats = transformer.update(
                    symbol=symbol,
                    ts_ms=ts_ms,
                    close=100.0 + i * 0.5,
                )
            else:  # С OHLC
                feats = transformer.update(
                    symbol=symbol,
                    ts_ms=ts_ms,
                    close=100.0 + i * 0.5,
                    open_price=100.0 + i * 0.5 - 0.2,
                    high=100.0 + i * 0.5 + 0.3,
                    low=100.0 + i * 0.5 - 0.3,
                )
            ts_ms += 60 * 60 * 1000  # +1 час

        # После 30 баров с 20% пропусков (6 пропусков, 24 валидных)
        # parkinson_24h требует 24 бара, 80% от 24 = 19.2 → 20 баров
        # У нас 24 валидных > 20, должно работать

        parkinson_24h = feats.get("parkinson_24h")
        self.assertIsNotNone(parkinson_24h, "Признак должен существовать")
        self.assertFalse(math.isnan(parkinson_24h),
                        "Должен быть валиден при 80% данных (24/30)")

        print(f"\n  ✓ Обработка пропусков OHLC: {parkinson_24h:.6f}")


def run_all_tests():
    """Запуск всех тестов с подробным выводом."""
    # Создаем test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Добавляем все тестовые классы
    test_classes = [
        TestParkinsonFormulaCorrectness,
        TestParkinsonThreshold,
        TestParkinsonEdgeCases,
        TestParkinsonRealWorldScenarios,
        TestParkinsonStatisticalProperties,
        TestParkinsonIntegration,
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Запускаем с подробным выводом
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Статистика
    print("\n" + "=" * 70)
    print("ИТОГИ КОМПЛЕКСНОГО ТЕСТИРОВАНИЯ")
    print("=" * 70)
    print(f"Всего тестов: {result.testsRun}")
    print(f"Успешно: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Провалено: {len(result.failures)}")
    print(f"Ошибок: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        print("\nФормула Parkinson КОРРЕКТНА:")
        print("  ✓ Использует valid_bars (правильно)")
        print("  ✓ Порог 80% адекватен")
        print("  ✓ Граничные случаи обработаны")
        print("  ✓ Реальные сценарии работают")
        print("  ✓ Статистические свойства корректны")
        print("  ✓ Интеграция с трансформером работает")
        print("=" * 70)
        return True
    else:
        print("\n❌ НЕКОТОРЫЕ ТЕСТЫ НЕ ПРОШЛИ!")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
